// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/recon/sensor/trigger_grouper.hpp"

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

}  // namespace

const char* to_string(SyncMode mode) noexcept {
  return mode == SyncMode::Sequence ? "sequence" : "trigger";
}

struct SensorArray::Impl {
  std::vector<std::unique_ptr<IRgbdSensor>> sensors;
  SyncMode sync = SyncMode::Trigger;
  std::size_t queue_depth = 0;
  // Each sensor's pose from the calibration; empty without one.
  std::vector<std::optional<camera::Mat4d>> poses;
  std::vector<std::size_t> start_order;  // secondaries, then the primary
  std::size_t primary = 0;
  bool running = false;

  // SyncMode::Trigger: the frames drained and not yet handed out, by the id
  // the grouper knows them by.
  std::optional<TriggerGrouper> grouper;
  std::unordered_map<std::uint64_t, RgbdFrame> held;
  std::uint64_t next_id = 0;
  std::vector<std::uint64_t> released;
  // SyncMode::Sequence: each sensor's frames not yet in a set, in order, and
  // the last set handed out, which a later frame must be past.
  std::vector<std::deque<RgbdFrame>> queues;
  std::optional<std::uint64_t> last_sequence;

  std::vector<GpuFramePrep> preps;  // one per sensor, with a device
  std::vector<RgbdFrame> drained;   // reused per sensor per poll
  std::uint64_t sets = 0;
  std::uint64_t incomplete = 0;
  std::uint64_t unmatched = 0;

  void release() {
    for (const std::uint64_t id : released) {
      if (held.erase(id) > 0) ++unmatched;
    }
    released.clear();
  }

  void forget() {
    if (grouper) grouper->clear(&released);
    release();
    held.clear();
    for (auto& q : queues) {
      unmatched += q.size();
      q.clear();
    }
    last_sequence.reset();  // a restart may number its frames from 0
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

  // Also when a move assignment replaces the array.
  ~Impl() {
    if (running) stop();
  }

  FrameSet finish(FrameSet set) {
    for (std::size_t c = 0; c < set.frames.size(); ++c) {
      if (set.frames[c] && poses[c]) set.frames[c]->color_to_world = *poses[c];
    }
    ++sets;
    if (!set.complete()) ++incomplete;
    return set;
  }

  core::Result<std::optional<FrameSet>> take_trigger();
  core::Result<std::optional<FrameSet>> take_sequence();
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
  impl->sync = options.sync;
  impl->queue_depth = options.queue_depth;
  impl->poses.resize(sensors.size());
  if (!options.calibration.sensors.empty()) {
    const core::Status valid =
        camera::validate_array_calibration(options.calibration);
    if (!valid.ok()) return bad(valid.message());
    for (std::size_t i = 0; i < sensors.size(); ++i) {
      const camera::SensorCalibration* s =
          camera::find_sensor(options.calibration, sensors[i]->info().id);
      if (s == nullptr) {
        return bad("the calibration does not pose sensor " +
                   sensors[i]->info().id);
      }
      impl->poses[i] = s->color_to_world;
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
      if (primary && options.sync == SyncMode::Trigger) {
        return unsupported("sensors " + sensors[*primary]->info().id + " and " +
                           sensors[i]->info().id + " are both sync primaries");
      }
      if (!primary) primary = i;
    }
  }
  if (primary) impl->start_order.push_back(*primary);
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    if (sensors[i]->info().role == SyncRole::Primary && i != *primary) {
      impl->start_order.push_back(i);  // only under SyncMode::Sequence
    }
  }

  if (options.sync == SyncMode::Trigger) {
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
    if (options.tolerance_us == 0 || options.tolerance_us >= half_period_us) {
      return bad("tolerance_us is " + std::to_string(options.tolerance_us) +
                 "; it must be non-zero and under half a frame period (" +
                 std::to_string(half_period_us) + " us)");
    }
    TriggerGrouper::Config grouping;
    grouping.cameras = sensors.size();
    grouping.anchor = impl->primary;
    grouping.tolerance_us = options.tolerance_us;
    // A frame and a half of the slowest sensor: long enough for a late frame
    // to arrive, short enough that a silent sensor costs one set.
    grouping.max_wait_us = fastest > 0 ? 1500000u / slowest : 50000u;
    grouping.queue_depth = options.queue_depth;
    impl->grouper.emplace(grouping);
  } else {
    impl->queues.resize(sensors.size());
  }
  if (options.device != nullptr) {
    for (std::size_t i = 0; i < sensors.size(); ++i) {
      VKC_ASSIGN(GpuFramePrep prep,
                 GpuFramePrep::create(*options.device, *options.allocator,
                                      options.prep));
      impl->preps.push_back(std::move(prep));
    }
  }
  impl->sensors = std::move(sensors);
  return SensorArray(std::move(impl));
}

std::size_t SensorArray::size() const noexcept {
  return impl_ != nullptr ? impl_->sensors.size() : 0;
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
  return impl_->sync == SyncMode::Trigger ? impl_->take_trigger()
                                          : impl_->take_sequence();
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
    VKC_TRY(status);
  }
  release();
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

core::Result<std::optional<FrameSet>> SensorArray::Impl::take_sequence() {
  for (std::size_t c = 0; c < sensors.size(); ++c) {
    // The frames before a failure were counted delivered: queued all the same.
    const core::Status status = sensors[c]->drain(&drained);
    std::deque<RgbdFrame>& q = queues[c];
    for (RgbdFrame& frame : drained) {
      if (last_sequence && frame.sequence <= *last_sequence) {
        ++unmatched;  // its set has gone out
        continue;
      }
      // In sequence order: a source may hand one out of turn.
      auto at = q.end();
      while (at != q.begin() && std::prev(at)->sequence > frame.sequence) --at;
      q.insert(at, std::move(frame));
    }
    drained.clear();  // and with it the frames not queued
    // A sensor ahead of one that has fallen behind costs its oldest frames,
    // not memory.
    while (q.size() > queue_depth) {
      q.pop_front();
      ++unmatched;
    }
    VKC_TRY(status);
  }
  // The next set is the lowest sequence number held; it is ready once every
  // sensor has sent a frame at or past it, or will send none.
  std::optional<std::uint64_t> next;
  for (const auto& q : queues) {
    if (!q.empty())
      next = std::min(next.value_or(q.front().sequence), q.front().sequence);
  }
  if (!next) return std::optional<FrameSet>{};
  for (std::size_t c = 0; c < sensors.size(); ++c) {
    if (queues[c].empty() && !sensors[c]->exhausted()) {
      return std::optional<FrameSet>{};
    }
  }
  FrameSet set;
  set.sequence = *next;
  set.frames.resize(sensors.size());
  for (std::size_t c = 0; c < sensors.size(); ++c) {
    std::deque<RgbdFrame>& q = queues[c];
    if (q.empty() || q.front().sequence != *next) continue;
    // The first present sensor's.
    if (set.count() == 0) set.timestamp_ns = q.front().timestamp_ns;
    set.frames[c] = std::move(q.front());
    q.pop_front();
  }
  last_sequence = *next;
  return std::optional<FrameSet>{finish(std::move(set))};
}

core::Result<DeviceFrameSet> SensorArray::process(const FrameSet& set,
                                                  core::StageMetrics* metrics) {
  if (impl_ == nullptr) return bad("process on a moved-from array");
  if (impl_->preps.empty()) {
    return bad("process needs an array opened with a device");
  }
  if (set.frames.size() != impl_->sensors.size()) {
    return bad("a set of " + std::to_string(set.frames.size()) +
               " frames for an array of " +
               std::to_string(impl_->sensors.size()));
  }
  DeviceFrameSet out;
  out.timestamp_ns = set.timestamp_ns;
  out.sequence = set.sequence;
  VKC_ASSIGN(out.frames,
             GpuFramePrep::prepare_batch(impl_->preps, set.frames, metrics));
  return out;
}

bool SensorArray::exhausted() const noexcept {
  if (impl_ == nullptr) return true;
  const Impl& a = *impl_;
  if (a.sync == SyncMode::Trigger) {
    return std::any_of(a.sensors.begin(), a.sensors.end(),
                       [](const auto& s) { return s->exhausted(); });
  }
  for (std::size_t c = 0; c < a.sensors.size(); ++c) {
    if (!a.sensors[c]->exhausted() || !a.queues[c].empty()) return false;
  }
  return true;
}

}  // namespace volumetric_kit::recon::sensor
