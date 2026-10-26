// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A run's sets (sensor/array/rig_sets.hpp) over arrays of pushed fake
// sensors. The run: made without throwing, each skew empty before a set and
// read as a copy; each set waited for on their notifications, counted and
// cut, the run done at exactly N frames, a set missing a secondary counted as
// it came, and each secondary's skew to the primary over every frame it sent,
// the primary's none. The cut: the set that reaches N keeps its first frames
// in the array's order, past a missing one, and lets the rest go, their
// pictures with them; a set past N keeps none. The silence: the limit counted
// from the last set, then the run's error naming a lone sync secondary's
// cause. And the array's error, from a sensor that notifies nothing. Host
// only; the silence waits out the limit, about 10 s.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_check.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/sensor/array/rig_sets.hpp"
#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;

namespace {

// A sensor on the host's clock at 1 kHz whose frames the test pushes, each
// holding a picture the test may watch; the array's next drain hands them
// over. With `notifies` false it keeps the interface's refusal to notify.
class Pushed final : public sensor::IRgbdSensor {
 public:
  Pushed(std::string id, sensor::SyncRole role, bool notifies = true)
      : notifies_(notifies) {
    info_.id = std::move(id);
    info_.role = role;
    info_.clock = sensor::ClockDomain::Host;
    info_.fps = 1000;
  }

  // Push a frame stamped `timestamp_ns`; returns its picture, to watch.
  std::weak_ptr<const void> push(std::uint64_t timestamp_ns) {
    sensor::RgbdFrame frame;
    frame.timestamp_ns = timestamp_ns;
    frame.sequence = timestamp_ns;
    auto picture = std::make_shared<int>(0);
    std::weak_ptr<const void> watched = picture;
    frame.pixels = std::move(picture);
    frames_.push_back(std::move(frame));
    if (notify_) notify_();
    return watched;
  }

  const sensor::SensorInfo& info() const noexcept override { return info_; }
  vkc::Status set_queue_depth(std::size_t) override { return {}; }
  vkc::Status start() override { return {}; }
  void stop() noexcept override { frames_.clear(); }
  vkc::Result<std::optional<sensor::RgbdFrame>> poll() override {
    if (frames_.empty()) return no_frame();
    sensor::RgbdFrame frame = std::move(frames_.back());
    frames_.clear();
    return some_frame(std::move(frame));
  }
  vkc::Status drain(std::vector<sensor::RgbdFrame>* out) override {
    for (sensor::RgbdFrame& frame : frames_) out->push_back(std::move(frame));
    frames_.clear();
    return {};
  }
  sensor::SensorStats stats() const noexcept override { return {}; }
  vkc::Status set_frame_notify(std::function<void()> notify) override {
    if (!notifies_) return IRgbdSensor::set_frame_notify(std::move(notify));
    notify_ = std::move(notify);
    return {};
  }

 private:
  bool notifies_;
  sensor::SensorInfo info_;
  std::vector<sensor::RgbdFrame> frames_;
  std::function<void()> notify_;
};

// A started array of `sensors`, in that order; null if it does not open.
std::optional<sensor::SensorArray> started(
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors) {
  sensor::SensorArray::Options options;
  options.tolerance_us = 200;  // under half of 1 kHz's period
  auto opened = sensor::SensorArray::open(std::move(sensors), options);
  if (!opened.ok()) return std::nullopt;
  sensor::SensorArray array = std::move(opened).value();
  if (!array.start().ok()) return std::nullopt;
  return array;
}

// Trigger `t`'s time, at 1 kHz.
std::uint64_t at(std::uint64_t t) { return 5'000'000'000 + t * 1'000'000; }

// A run of 4 frames from a primary "P" and its secondary "S", in that order:
// a complete set (2 frames), one missing S (1), and a complete one cut to the
// last frame the run takes, P's.
int run() {
  auto primary = std::make_unique<Pushed>("P", sensor::SyncRole::Primary);
  auto secondary = std::make_unique<Pushed>("S", sensor::SyncRole::Secondary);
  Pushed* p = primary.get();
  Pushed* s = secondary.get();
  std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
  sensors.push_back(std::move(primary));
  sensors.push_back(std::move(secondary));
  std::optional<sensor::SensorArray> array = started(std::move(sensors));
  CHECK(array && array->primary() == 0);

  // A run is made with no allocation, so nothing it could throw; each
  // sensor's skew is empty until a set, and read as a copy, so none read
  // before a set is left behind by it.
  static_assert(
      std::is_nothrow_constructible_v<sensor::RigSets, sensor::SensorArray&,
                                      std::size_t>);
  static_assert(
      std::is_same_v<decltype(std::declval<const sensor::RigSets&>().skew(0)),
                     sensor::RigSets::Skew>);
  sensor::RigSets sets(*array, 4);
  vkc::StageMetrics stages;
  CHECK(!sets.done() && sets.sets() == 0 && sets.frames() == 0);
  CHECK(sets.skew(0).frames == 0 && sets.skew(1).frames == 0);
  // Trigger `t`: P's frame, and S's `lag_ns` after it unless S is silent.
  const auto trigger = [&](std::uint64_t t,
                           std::optional<std::int64_t> lag_ns) {
    p->push(at(t));
    if (lag_ns) s->push(at(t) + static_cast<std::uint64_t>(*lag_ns));
  };

  trigger(0, 80'000);
  vkc::Result<sensor::FrameSet> set = sets.next(&stages);
  CHECK(set.ok() && set.value().complete());
  CHECK(!sets.done() && sets.sets() == 1 && sets.frames() == 2);

  // S silent: the set comes once the array stops waiting for it, and counts
  // the one frame it holds.
  trigger(1, std::nullopt);
  set = sets.next(&stages);
  CHECK(set.ok() && set.value().count() == 1 && set.value().frames[0]);
  CHECK(!sets.done() && sets.sets() == 2 && sets.frames() == 3);

  // One frame left to take: the set is cut to P's, the first in the array's
  // order, and the run is done at exactly 4.
  trigger(2, -120'000);
  set = sets.next(&stages);
  CHECK(set.ok() && set.value().count() == 1 && set.value().frames[0]);
  CHECK(sets.done() && sets.sets() == 3 && sets.frames() == 4);

  // S's skew over the two frames it sent, the one the cut let go included;
  // the primary has none.
  const sensor::RigSets::Skew lag = sets.skew(1);
  CHECK(lag.frames == 2);
  CHECK(std::fabs(lag.sum_ms - (0.08 - 0.12)) < 1e-9);
  CHECK(std::fabs(lag.worst_ms - 0.12) < 1e-9);
  CHECK(sets.skew(0).frames == 0 && sets.skew(0).sum_ms == 0.0);

  // The polls, and the wait for the silent secondary, blocked on the
  // sensors' notifications until the array stopped waiting for it.
  CHECK(stages.rows().size() == 2);
  CHECK(std::string(stages.rows()[0].name) == "poll");
  CHECK(std::string(stages.rows()[1].name) == "wait: cameras");
  CHECK(stages.rows()[1].wait_ms > 0.0);
  return 0;
}

// A run of 2 frames over "P", "A", "B" and "C", A silent: the set holds P's,
// B's and C's frames and keeps the first two in the array's order, past the
// missing A; C's goes, and its picture with it. The set keeps an entry per
// sensor. A set past the run's frames keeps none.
int cut() {
  std::vector<Pushed*> fakes;
  std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
  for (const char* id : {"P", "A", "B", "C"}) {
    const sensor::SyncRole role =
        fakes.empty() ? sensor::SyncRole::Primary : sensor::SyncRole::Secondary;
    auto fake = std::make_unique<Pushed>(id, role);
    fakes.push_back(fake.get());
    sensors.push_back(std::move(fake));
  }
  std::optional<sensor::SensorArray> array = started(std::move(sensors));
  CHECK(array);

  sensor::RigSets sets(*array, 2);
  const std::weak_ptr<const void> p = fakes[0]->push(at(0));
  const std::weak_ptr<const void> b = fakes[2]->push(at(0) + 50'000);
  const std::weak_ptr<const void> c = fakes[3]->push(at(0) - 50'000);
  vkc::Result<sensor::FrameSet> set = sets.next();
  CHECK(set.ok() && set.value().frames.size() == 4);
  const sensor::FrameSet& kept = set.value();
  CHECK(kept.count() == 2 && kept.frames[0] && !kept.frames[1]);
  CHECK(kept.frames[2] && kept.frames[2]->timestamp_ns == at(0) + 50'000);
  CHECK(!kept.frames[3]);
  CHECK(!p.expired() && !b.expired() && c.expired());
  CHECK(sets.done() && sets.sets() == 1 && sets.frames() == 2);
  // C's skew counts, though the cut let its frame go; A sent none.
  CHECK(sets.skew(3).frames == 1 && sets.skew(1).frames == 0);

  // Past the run's frames: an entry per sensor, with no frame, every picture
  // let go, and the frames counted unchanged.
  const std::weak_ptr<const void> p_past = fakes[0]->push(at(1));
  const std::weak_ptr<const void> b_past = fakes[2]->push(at(1) + 50'000);
  set = sets.next();
  CHECK(set.ok() && set.value().frames.size() == 4 && set.value().count() == 0);
  CHECK(p_past.expired() && b_past.expired());
  CHECK(sets.done() && sets.sets() == 2 && sets.frames() == 2);
  return 0;
}

// A lone sync secondary, which captures only on another camera's signal: one
// set half a second into the run, then none. The limit counts from that set,
// not from the run's start, and the run then fails naming the camera and why
// it is silent, counting nothing more.
int silence() {
  using Clock = std::chrono::steady_clock;
  auto lone = std::make_unique<Pushed>("S", sensor::SyncRole::Secondary);
  Pushed* s = lone.get();
  std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
  sensors.push_back(std::move(lone));
  std::optional<sensor::SensorArray> array = started(std::move(sensors));
  CHECK(array);

  sensor::RigSets sets(*array, 2);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  s->push(at(0));
  const Clock::time_point before_set = Clock::now();
  CHECK(sets.next().ok() && sets.sets() == 1 && sets.frames() == 1);
  const vkc::Result<sensor::FrameSet> none = sets.next();
  const Clock::duration waited = Clock::now() - before_set;
  CHECK(!none.ok() && none.status().domain() == vkc::Status::Code::IoError);
  const std::string why = none.status().message();
  CHECK(why.find("camera S ") != std::string::npos);
  CHECK(why.find("sync secondary") != std::string::npos);
  // The whole limit from the set, less the wait's rounding to milliseconds; a
  // limit counted from the run's start ends half a second early.
  CHECK(waited >=
        sensor::RigSets::kSilenceLimit - std::chrono::milliseconds(50));
  CHECK(!sets.done() && sets.sets() == 1 && sets.frames() == 1);
  return 0;
}

// A sensor that notifies nothing: the array refuses to wait on it, and the
// run hands that refusal on, counting no set.
int refused() {
  std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
  sensors.push_back(std::make_unique<Pushed>("P", sensor::SyncRole::Primary,
                                             /*notifies=*/false));
  std::optional<sensor::SensorArray> array = started(std::move(sensors));
  CHECK(array);
  sensor::RigSets sets(*array, 1);
  const vkc::Result<sensor::FrameSet> set = sets.next();
  CHECK(!set.ok() && set.status().domain() == vkc::Status::Code::Unsupported);
  CHECK(!sets.done() && sets.sets() == 0 && sets.frames() == 0);
  return 0;
}

}  // namespace

int main() {
  CHECK(run() == 0);
  CHECK(cut() == 0);
  CHECK(refused() == 0);
  CHECK(silence() == 0);
  std::puts("sensor_array_rig_sets: OK");
  return 0;
}
