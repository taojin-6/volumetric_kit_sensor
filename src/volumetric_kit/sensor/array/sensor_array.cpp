// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "trigger_grouper.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::recon::sensor {

namespace {

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("SensorArray: " + what);
}

core::Status unsupported(const std::string& what) {
  return core::Status::unsupported("SensorArray: " + what);
}

std::uint64_t now_us() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// What the sensors' notifications wake: a count each bumps, under the
// mutex, and the condition variable wait_set blocks on. Held by every
// sensor's notify as well as the array, so a call in flight as the array
// goes touches nothing freed.
struct Wake {
  std::mutex mutex;
  std::condition_variable changed;
  std::uint64_t count = 0;  // guarded by mutex

  void notify() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++count;
    }
    changed.notify_one();
  }
};

}  // namespace

struct SensorArray::Impl {
  std::vector<std::unique_ptr<IRgbdSensor>> sensors;
  std::size_t queue_depth = 0;
  // Each sensor's pose from the calibration; empty without one.
  std::vector<std::optional<camera::Mat4d>> poses;
  std::vector<std::size_t> start_order;  // secondaries, then the primary
  std::size_t primary = 0;
  bool running = false;

  // The frames drained and not yet handed out, by the id the grouper knows
  // them by.
  std::optional<TriggerGrouper> grouper;
  std::unordered_map<std::uint64_t, RgbdFrame> held;
  std::uint64_t next_id = 0;
  std::vector<std::uint64_t> released;

  std::vector<GpuFramePrep> preps;  // one per sensor, with a device
  std::vector<RgbdFrame> drained;   // reused per sensor per poll
  std::uint64_t sets = 0;
  std::uint64_t incomplete = 0;
  std::uint64_t unmatched = 0;
  // The sensors' notifications, and whether every sensor gives them.
  std::shared_ptr<Wake> wake = std::make_shared<Wake>();
  bool notifies = false;

  void release() {
    for (const std::uint64_t id : released) {
      if (held.erase(id) > 0) ++unmatched;
    }
    released.clear();
  }

  void forget() {
    grouper->clear(&released);
    release();
    held.clear();
  }

  // The primary first, so the secondaries see no trigger after their own
  // stop.
  void stop() noexcept {
    for (auto it = start_order.rbegin(); it != start_order.rend(); ++it) {
      sensors[*it]->stop();
    }
    forget();
    running = false;
  }

  // Also when a move assignment replaces the array. A sensor started outside
  // it (sensor()) is stopped too, so none delivers into an array that is
  // gone; a notification in flight touches only its own Wake.
  ~Impl() {
    if (running) {
      stop();
      return;
    }
    for (auto it = start_order.rbegin(); it != start_order.rend(); ++it) {
      if (*it < sensors.size()) sensors[*it]->stop();
    }
  }

  FrameSet finish(FrameSet set) {
    for (std::size_t c = 0; c < set.frames.size(); ++c) {
      if (!set.frames[c]) continue;
      if (poses[c]) set.frames[c]->color_to_world = *poses[c];
      set.arrival_ns = std::max(set.arrival_ns, set.frames[c]->arrival_ns);
    }
    ++sets;
    if (!set.complete()) ++incomplete;
    return set;
  }

  core::Result<std::optional<FrameSet>> take_trigger();
  // Whether `set` can be prepared here, for `op`'s message.
  core::Status check(const FrameSet& set, const char* op) const;
};

SensorArray::SensorArray(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
SensorArray::SensorArray(SensorArray&& other) noexcept = default;
SensorArray& SensorArray::operator=(SensorArray&& other) noexcept = default;
SensorArray::~SensorArray() = default;

core::Result<SensorArray> SensorArray::open(
    std::vector<std::unique_ptr<IRgbdSensor>>&& sensors,
    const Options& options) {
  if (sensors.empty()) return bad("an array needs at least one sensor");
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    if (sensors[i] == nullptr) {
      return bad("sensor " + std::to_string(i) + " is null");
    }
    const std::string& id = sensors[i]->info().id;
    if (id.empty()) return bad("sensor " + std::to_string(i) + " has no id");
    for (std::size_t j = 0; j < i; ++j) {
      if (sensors[j]->info().id == id) {
        return bad("sensor " + id + " is listed twice");
      }
    }
    if (sensors[i]->info().pose == PoseSource::Tracked) {
      // TODO: tracked members, registered to the array's world.
      return unsupported("sensor " + id +
                         " tracks its own pose, which an array does not take "
                         "yet");
    }
  }
  if (options.queue_depth == 0) {
    return bad("a queue holds at least one frame");
  }
  if ((options.device == nullptr) != (options.allocator == nullptr)) {
    return bad(
        "a device and the allocator its passes' buffers come from are "
        "given together");
  }

  auto impl = std::make_unique<Impl>();
  impl->queue_depth = options.queue_depth;
  impl->poses.resize(sensors.size());
  if (!options.calibration.sensors.empty()) {
    const core::Status valid =
        camera::validate_array_calibration(options.calibration);
    if (!valid.ok()) return bad(valid.message());
    for (std::size_t i = 0; i < sensors.size(); ++i) {
      // A sensor the file lists without a pose (its lenses alone) is as
      // unposed as one it does not list.
      const camera::SensorCalibration* s =
          camera::find_sensor(options.calibration, sensors[i]->info().id);
      if (s == nullptr || !s->color_to_world) {
        return bad("the calibration does not pose sensor " +
                   sensors[i]->info().id);
      }
      impl->poses[i] = *s->color_to_world;
    }
  }

  // The fastest and slowest rates the sensors report, for the tolerance's
  // bound and the grouping's wait.
  std::uint32_t fastest = 0;
  std::uint32_t slowest = std::numeric_limits<std::uint32_t>::max();
  for (const auto& s : sensors) {
    const std::uint32_t fps = s->info().fps;
    if (fps == 0) continue;
    fastest = std::max(fastest, fps);
    slowest = std::min(slowest, fps);
  }

  // Secondaries first, then free-running sensors, the primary last: the
  // primary's first trigger is what the secondaries wait for.
  std::optional<std::size_t> primary;
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    const SyncRole role = sensors[i]->info().role;
    if (role == SyncRole::Secondary) impl->start_order.push_back(i);
  }
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    const SyncRole role = sensors[i]->info().role;
    if (role == SyncRole::FreeRun) impl->start_order.push_back(i);
    if (role == SyncRole::Primary) {
      if (primary) {
        return unsupported("sensors " + sensors[*primary]->info().id + " and " +
                           sensors[i]->info().id + " are both sync primaries");
      }
      primary = i;
    }
  }
  if (primary) impl->start_order.push_back(*primary);

  if (sensors.size() > 1) {
    if (!primary) {
      return unsupported("no sync primary, so nothing triggers the others");
    }
    for (const auto& s : sensors) {
      const SensorInfo& info = s->info();
      if (info.role == SyncRole::FreeRun) {
        // TODO: free-running members, joined by their nearest frame.
        return unsupported("sensor " + info.id +
                           " runs on its own clock rather than the "
                           "primary's trigger");
      }
      if (info.clock != ClockDomain::Host) {
        return unsupported("sensor " + info.id +
                           " stamps its frames on its own clock; grouping "
                           "by trigger needs every sensor on the host's");
      }
    }
  }
  impl->primary = primary.value_or(0);
  const std::uint64_t half_period_us =
      fastest > 0 ? 500000u / fastest
                  : std::numeric_limits<std::uint64_t>::max();
  // By default 0.4 of a period, as Orbbec matches its rigs' frames within half
  // of one: a trigger's frames sit well inside it, the next trigger's well
  // outside.
  const std::uint64_t tolerance_us = options.tolerance_us != 0
                                         ? options.tolerance_us
                                     : fastest > 0 ? 400000u / fastest
                                                   : 5000u;
  if (tolerance_us == 0 || tolerance_us >= half_period_us) {
    return bad("tolerance_us is " + std::to_string(tolerance_us) +
               "; it must be non-zero and under half a frame period (" +
               std::to_string(half_period_us) + " us)");
  }
  TriggerGrouper::Config grouping;
  grouping.cameras = sensors.size();
  grouping.anchor = impl->primary;
  grouping.tolerance_us = tolerance_us;
  // A frame and a half of the slowest sensor: long enough for a late frame
  // to arrive, short enough that a silent sensor costs one set.
  grouping.max_wait_us = fastest > 0 ? 1500000u / slowest : 50000u;
  grouping.queue_depth = options.queue_depth;
  impl->grouper.emplace(grouping);
  if (options.device != nullptr) {
    for (std::size_t i = 0; i < sensors.size(); ++i) {
      VKC_ASSIGN(GpuFramePrep prep,
                 GpuFramePrep::create(*options.device, *options.allocator,
                                      options.prep));
      impl->preps.push_back(std::move(prep));
    }
  }
  // Last, once nothing else can refuse: each sensor wakes wait_set, unless
  // one notifies nothing, when wait_set is refused and the caller polls.
  impl->notifies = true;
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    const core::Status asked =
        sensors[i]->set_frame_notify([wake = impl->wake] { wake->notify(); });
    if (asked.ok()) continue;
    // A sensor that cannot notify is polled; one that refuses otherwise (a
    // started one) refuses the array, and the others notify nothing.
    if (asked.domain() == core::Status::Code::Unsupported) {
      impl->notifies = false;
      continue;
    }
    for (std::size_t j = 0; j < i; ++j) {
      static_cast<void>(sensors[j]->set_frame_notify({}));
    }
    return asked;
  }
  impl->sensors = std::move(sensors);
  return SensorArray(std::move(impl));
}

std::size_t SensorArray::size() const noexcept {
  return impl_ != nullptr ? impl_->sensors.size() : 0;
}

std::optional<std::int64_t> FrameSet::skew_ns(std::size_t i) const {
  VKC_CHECK(i < frames.size(), "FrameSet::skew_ns: index past the set");
  if (!frames[i]) return std::nullopt;
  const std::uint64_t t = frames[i]->timestamp_ns;
  // The gap each way in unsigned, then signed, saturated where it does not
  // fit: -2^63 does, so a frame that far before the trigger is exact.
  constexpr auto kMax =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (t >= timestamp_ns) {
    return static_cast<std::int64_t>(std::min(t - timestamp_ns, kMax));
  }
  const std::uint64_t before = timestamp_ns - t;
  return before > kMax ? std::numeric_limits<std::int64_t>::min()
                       : -static_cast<std::int64_t>(before);
}

std::uint64_t FrameSet::spread_ns() const noexcept {
  std::uint64_t earliest = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t latest = 0;
  for (const std::optional<RgbdFrame>& frame : frames) {
    if (!frame) continue;
    earliest = std::min(earliest, frame->timestamp_ns);
    latest = std::max(latest, frame->timestamp_ns);
  }
  return latest >= earliest ? latest - earliest : 0;
}

IRgbdSensor& SensorArray::sensor(std::size_t i) {
  VKC_CHECK(i < size(), "SensorArray::sensor: index past the array");
  return *impl_->sensors[i];
}

std::size_t SensorArray::primary() const noexcept {
  return impl_ != nullptr ? impl_->primary : 0;
}

SensorArrayStats SensorArray::stats() const {
  SensorArrayStats s;
  if (impl_ == nullptr) return s;
  s.sets = impl_->sets;
  s.incomplete = impl_->incomplete;
  s.unmatched = impl_->unmatched;
  for (const auto& sensor : impl_->sensors)
    s.sensors.push_back(sensor->stats());
  return s;
}

core::Status SensorArray::start() {
  if (impl_ == nullptr) return bad("start on a moved-from array");
  Impl& a = *impl_;
  if (!a.running) {
    for (const auto& s : a.sensors) {
      VKC_TRY(s->set_queue_depth(a.queue_depth));
    }
  }
  // Running sensors too: their start is OK, and a failed one's -- a camera
  // gone away -- says why.
  for (const std::size_t i : a.start_order) {
    const core::Status started = a.sensors[i]->start();
    if (!started.ok()) {
      a.stop();
      return started;
    }
  }
  if (!a.running) {
    a.sets = 0;
    a.incomplete = 0;
    a.unmatched = 0;
    a.running = true;
  }
  return {};
}

void SensorArray::stop() noexcept {
  if (impl_ != nullptr) impl_->stop();
}

core::Result<std::optional<FrameSet>> SensorArray::poll_set() {
  if (impl_ == nullptr) return bad("poll_set on a moved-from array");
  if (!impl_->running) return std::optional<FrameSet>{};
  return impl_->take_trigger();
}

core::Result<std::optional<FrameSet>> SensorArray::Impl::take_trigger() {
  const std::uint64_t now = now_us();
  for (std::size_t c = 0; c < sensors.size(); ++c) {
    // The frames before a failure were counted delivered: held all the same.
    const core::Status status = sensors[c]->drain(&drained);
    for (RgbdFrame& frame : drained) {
      if (frame.timestamp_ns == 0) {  // no clock to group it by
        ++unmatched;
        continue;
      }
      const std::uint64_t id = next_id++;
      const std::uint64_t ts = frame.timestamp_ns / 1000;
      held.emplace(id, std::move(frame));
      grouper->add(c, ts, id, now, &released);
    }
    drained.clear();  // and with it the frames not held
    // add() can evict queued frames even when a later drain reports failure.
    // Release those owners before returning, retaining only matchable frames.
    release();
    VKC_TRY(status);
  }
  const std::optional<TriggerGrouper::Group> group =
      grouper->take(now, &released);
  release();
  if (!group) return std::optional<FrameSet>{};

  FrameSet set;
  set.frames.resize(sensors.size());
  for (std::size_t c = 0; c < sensors.size(); ++c) {
    if (!group->ids[c]) continue;
    const auto it = held.find(*group->ids[c]);
    set.frames[c] = std::move(it->second);
    held.erase(it);
  }
  const RgbdFrame& anchor = *set.frames[primary];
  set.timestamp_ns = anchor.timestamp_ns;
  set.sequence = anchor.sequence;
  return std::optional<FrameSet>{finish(std::move(set))};
}

core::Result<std::optional<FrameSet>> SensorArray::wait_set(
    std::chrono::milliseconds timeout, core::StageMetrics* metrics) {
  if (impl_ == nullptr) return bad("wait_set on a moved-from array");
  Impl& a = *impl_;
  if (!a.running) return std::optional<FrameSet>{};
  if (!a.notifies) {
    return unsupported(
        "wait_set: a sensor notifies nothing of its frames; poll_set it");
  }
  using Clock = std::chrono::steady_clock;
  const Clock::time_point give_up = Clock::now() + timeout;
  for (;;) {
    // Read before the poll, so a frame posted while it runs wakes the wait.
    std::uint64_t seen = 0;
    {
      std::lock_guard<std::mutex> lock(a.wake->mutex);
      seen = a.wake->count;
    }
    std::optional<FrameSet> set;
    {
      const core::StageScope poll(metrics, "poll");
      VKC_ASSIGN(set, a.take_trigger());
    }
    if (set) return set;
    // Until a sensor posts, a waiting trigger's late sensors run out of
    // time, or the caller's timeout passes.
    Clock::time_point until = give_up;
    if (const std::optional<std::uint64_t> due =
            a.grouper->next_deadline_us()) {
      const std::uint64_t now = now_us();
      const auto left = std::chrono::microseconds(due > now ? *due - now : 0);
      until = std::min(until, Clock::now() + left);
    }
    if (Clock::now() >= give_up) return std::optional<FrameSet>{};
    const core::WaitScope wait(metrics, "wait: cameras");
    std::unique_lock<std::mutex> lock(a.wake->mutex);
    a.wake->changed.wait_until(lock, until,
                               [&] { return a.wake->count != seen; });
  }
}

core::Status SensorArray::Impl::check(const FrameSet& set,
                                      const char* op) const {
  if (preps.empty()) {
    return bad(std::string(op) + " needs an array opened with a device");
  }
  if (set.frames.size() != sensors.size()) {
    return bad("a set of " + std::to_string(set.frames.size()) +
               " frames for an array of " + std::to_string(sensors.size()));
  }
  return {};
}

core::Result<DeviceFrameSet> SensorArray::record_process(
    core::CommandBatch& batch, const FrameSet& set,
    core::StageMetrics* metrics) {
  if (impl_ == nullptr) return bad("record_process on a moved-from array");
  VKC_TRY(impl_->check(set, "record_process"));
  DeviceFrameSet out;
  out.timestamp_ns = set.timestamp_ns;
  out.sequence = set.sequence;
  VKC_ASSIGN(out.frames, GpuFramePrep::record_set(batch, impl_->preps,
                                                  set.frames, metrics));
  return out;
}

core::Result<DeviceFrameSet> SensorArray::process(const FrameSet& set,
                                                  core::StageMetrics* metrics) {
  if (impl_ == nullptr) return bad("process on a moved-from array");
  VKC_TRY(impl_->check(set, "process"));
  DeviceFrameSet out;
  out.timestamp_ns = set.timestamp_ns;
  out.sequence = set.sequence;
  VKC_ASSIGN(out.frames,
             GpuFramePrep::prepare_batch(impl_->preps, set.frames, metrics));
  return out;
}

bool SensorArray::exhausted() const noexcept {
  if (impl_ == nullptr) return true;
  return std::any_of(impl_->sensors.begin(), impl_->sensors.end(),
                     [](const auto& s) { return s->exhausted(); });
}

}  // namespace volumetric_kit::recon::sensor
