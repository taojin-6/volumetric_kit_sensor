// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sensor array over scripted fake sensors: the arrays it refuses, the
// order it starts sensors in and its stop on a failed start, trigger grouping
// (complete sets, a silent secondary, the newest set winning, frames with no
// clock), sequence grouping (in order, a missing frame, waiting on a live
// sensor, the end of a recording), poses from the calibration, and a
// moved-from array. Host-only.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"
#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;
namespace sensor = volumetric_kit::recon::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

using Code = vkc::Status::Code;

// What the fakes did, in order: "start A", "stop B", ...
std::vector<std::string> g_log;

// A sensor whose frames the test pushes, which drain hands over.
class Scripted final : public sensor::IRgbdSensor {
 public:
  Scripted(std::string id, sensor::SyncRole role,
           sensor::ClockDomain clock = sensor::ClockDomain::Host,
           std::uint32_t fps = 1000) {
    info_.id = std::move(id);
    info_.role = role;
    info_.clock = clock;
    info_.fps = fps;
    info_.depth_to_color[3] = glm::dvec4(-0.032, 0.0, 0.0, 1.0);
  }

  void push(std::uint64_t timestamp_ns, std::uint64_t sequence) {
    sensor::RgbdFrame f;
    f.timestamp_ns = timestamp_ns;
    f.sequence = sequence;
    f.depth_to_color = info_.depth_to_color;
    f.color_to_world[3] = glm::dvec4(9.0, 9.0, 9.0, 1.0);  // the driver's
    frames_.push_back(std::move(f));
  }
  sensor::SensorInfo& mutable_info() { return info_; }
  bool fail_start = false;
  bool done = false;  // a recording's end
  std::size_t queue_depth = 0;

  const sensor::SensorInfo& info() const noexcept override { return info_; }
  vkc::Status set_queue_depth(std::size_t frames) override {
    queue_depth = frames;
    return {};
  }
  vkc::Status start() override {
    g_log.push_back("start " + info_.id);
    if (fail_start) return vkc::Status::io_error(info_.id + " will not start");
    return {};
  }
  void stop() noexcept override { g_log.push_back("stop " + info_.id); }
  vkc::Result<std::optional<sensor::RgbdFrame>> poll() override {
    if (frames_.empty()) return no_frame();
    sensor::RgbdFrame f = std::move(frames_.back());
    frames_.clear();
    return some_frame(std::move(f));
  }
  vkc::Status drain(std::vector<sensor::RgbdFrame>* out) override {
    for (auto& f : frames_) out->push_back(std::move(f));
    frames_.clear();
    return {};
  }
  bool exhausted() const noexcept override { return done && frames_.empty(); }
  sensor::SensorStats stats() const noexcept override { return {}; }

 private:
  sensor::SensorInfo info_;
  std::vector<sensor::RgbdFrame> frames_;
};

struct Rig {
  std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
  std::vector<Scripted*> fakes;  // the same sensors, to script
  Scripted& add(std::unique_ptr<Scripted> s) {
    fakes.push_back(s.get());
    sensors.push_back(std::move(s));
    return *fakes.back();
  }
};

// A primary "P" between two secondaries "S1" and "S2", all on the host clock.
Rig triggered_rig() {
  Rig r;
  r.add(std::make_unique<Scripted>("S1", sensor::SyncRole::Secondary));
  r.add(std::make_unique<Scripted>("P", sensor::SyncRole::Primary));
  r.add(std::make_unique<Scripted>("S2", sensor::SyncRole::Secondary));
  return r;
}

sensor::SensorArray::Options trigger_options() {
  sensor::SensorArray::Options o;
  o.sync = sensor::SyncMode::Trigger;
  o.tolerance_us = 200;  // under half of 1 kHz's period
  return o;
}

// Each sensor posed one metre further along x.
camera::ArrayCalibration calibration(const std::vector<std::string>& ids) {
  camera::ArrayCalibration c;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    camera::SensorCalibration s;
    s.id = ids[i];
    s.color_to_world[3] = glm::dvec4(static_cast<double>(i), 0.0, 0.0, 1.0);
    c.sensors.push_back(s);
  }
  return c;
}

Code refusal(Rig r, const sensor::SensorArray::Options& o) {
  const auto a = sensor::SensorArray::open(std::move(r.sensors), o);
  if (a.ok()) return Code::Ok;
  std::printf("  refused as expected: %s\n", a.status().message().c_str());
  return a.status().domain();
}

int test_refusals() {
  CHECK(refusal(Rig{}, trigger_options()) == Code::InvalidArgument);
  {
    Rig r = triggered_rig();
    r.sensors.push_back(nullptr);
    CHECK(refusal(std::move(r), trigger_options()) == Code::InvalidArgument);
  }
  {
    Rig r = triggered_rig();
    r.add(std::make_unique<Scripted>("P", sensor::SyncRole::Secondary));
    CHECK(refusal(std::move(r), trigger_options()) == Code::InvalidArgument);
  }
  {
    Rig r = triggered_rig();
    r.fakes[0]->mutable_info().pose = sensor::PoseSource::Tracked;
    CHECK(refusal(std::move(r), trigger_options()) == Code::Unsupported);
  }
  auto o = trigger_options();
  o.queue_depth = 0;
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  o = trigger_options();
  o.tolerance_us = 0;
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  o.tolerance_us = 500;  // half of 1 kHz's period
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  {  // no primary
    Rig r = triggered_rig();
    r.fakes[1]->mutable_info().role = sensor::SyncRole::Secondary;
    CHECK(refusal(std::move(r), trigger_options()) == Code::Unsupported);
  }
  {  // two primaries
    Rig r = triggered_rig();
    r.fakes[0]->mutable_info().role = sensor::SyncRole::Primary;
    CHECK(refusal(std::move(r), trigger_options()) == Code::Unsupported);
  }
  {  // a free-running member
    Rig r = triggered_rig();
    r.fakes[2]->mutable_info().role = sensor::SyncRole::FreeRun;
    CHECK(refusal(std::move(r), trigger_options()) == Code::Unsupported);
  }
  {  // a member on its own clock
    Rig r = triggered_rig();
    r.fakes[2]->mutable_info().clock = sensor::ClockDomain::Device;
    CHECK(refusal(std::move(r), trigger_options()) == Code::Unsupported);
  }
  // A calibration must pose every sensor.
  o = trigger_options();
  o.calibration = calibration({"S1", "P"});
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  // A sequence array takes any roles and clocks.
  {
    Rig r = triggered_rig();
    r.fakes[0]->mutable_info().role = sensor::SyncRole::FreeRun;
    r.fakes[1]->mutable_info().clock = sensor::ClockDomain::Device;
    sensor::SensorArray::Options s;
    s.sync = sensor::SyncMode::Sequence;
    CHECK(refusal(std::move(r), s) == Code::Ok);
  }
  return 0;
}

int test_start_order() {
  Rig r = triggered_rig();
  Scripted* p = r.fakes[1];
  auto o = trigger_options();
  o.queue_depth = 6;
  auto opened = sensor::SensorArray::open(std::move(r.sensors), o);
  CHECK(opened.ok());
  sensor::SensorArray array = std::move(opened).value();
  CHECK(array.size() == 3 && array.primary() == 1);
  g_log.clear();
  CHECK(array.start().ok());
  CHECK((g_log == std::vector<std::string>{"start S1", "start S2", "start P"}));
  CHECK(p->queue_depth == 6);
  g_log.clear();
  array.stop();
  CHECK((g_log == std::vector<std::string>{"stop P", "stop S2", "stop S1"}));

  // A sensor that will not start stops them all.
  p->fail_start = true;
  g_log.clear();
  const vkc::Status failed = array.start();
  CHECK(!failed.ok() && failed.domain() == Code::IoError);
  CHECK(g_log.size() == 6);  // three starts, then a stop each
  auto none = array.poll_set();
  CHECK(none.ok() && !none.value());  // not running
  return 0;
}

int test_trigger() {
  Rig r = triggered_rig();
  std::vector<Scripted*> f = r.fakes;  // S1, P, S2
  auto o = trigger_options();
  o.calibration = calibration({"S1", "P", "S2"});
  auto opened = sensor::SensorArray::open(std::move(r.sensors), o);
  CHECK(opened.ok());
  sensor::SensorArray array = std::move(opened).value();
  CHECK(array.start().ok());

  const std::uint64_t base = 5'000'000'000;
  const auto trigger = [&](std::uint64_t t, bool s2) {
    const std::uint64_t ts = base + t * 1'000'000;  // 1 kHz
    f[1]->push(ts, 100 + t);
    f[0]->push(ts + 80'000, 200 + t);  // 80 us after the primary
    if (s2) f[2]->push(ts + 150'000, 300 + t);
  };

  // Every sensor's frame: a complete set, around the primary's.
  trigger(0, true);
  auto set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->complete());
  CHECK(set.value()->timestamp_ns == base && set.value()->sequence == 100);
  CHECK(set.value()->frames[0]->sequence == 200);
  CHECK(set.value()->frames[2]->sequence == 300);
  // Posed by the calibration; the sensor's own extrinsic kept.
  for (std::size_t c = 0; c < 3; ++c) {
    const auto& frame = *set.value()->frames[c];
    CHECK(frame.color_to_world[3].x == static_cast<double>(c));
    CHECK(frame.depth_to_color[3].x == -0.032);
  }

  // A silent secondary: the set waits for it, then goes without it.
  trigger(1, false);
  set = array.poll_set();
  CHECK(set.ok() && !set.value());
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->count() == 2);
  CHECK(set.value()->frames[1]->sequence == 101 && !set.value()->frames[2]);

  // A slow poll gets the newest set; the older ones' frames go unmatched.
  for (std::uint64_t t = 2; t < 5; ++t) trigger(t, true);
  set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->sequence == 104);
  // A frame without a timestamp has no trigger.
  f[0]->push(0, 999);
  set = array.poll_set();
  CHECK(set.ok() && !set.value());

  const sensor::SensorArrayStats st = array.stats();
  CHECK(st.sets == 3 && st.incomplete == 1);
  CHECK(st.unmatched == 2 * 3 + 1);
  CHECK(st.sensors.size() == 3);
  CHECK(!array.exhausted());
  return 0;
}

int test_sequence() {
  Rig r;
  Scripted& a = r.add(std::make_unique<Scripted>("A", sensor::SyncRole::FreeRun,
                                                 sensor::ClockDomain::Device));
  Scripted& b = r.add(std::make_unique<Scripted>("B", sensor::SyncRole::FreeRun,
                                                 sensor::ClockDomain::Device));
  sensor::SensorArray::Options o;
  o.sync = sensor::SyncMode::Sequence;
  auto opened = sensor::SensorArray::open(std::move(r.sensors), o);
  CHECK(opened.ok());
  sensor::SensorArray array = std::move(opened).value();
  CHECK(array.start().ok());

  // A has 0, 2, 1 (out of turn); B has only 0 so far and is still live.
  a.push(1000, 0);
  a.push(3000, 2);
  a.push(2000, 1);
  b.push(1100, 0);
  auto set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->complete());
  CHECK(set.value()->sequence == 0 && set.value()->timestamp_ns == 1000);
  // B has sent nothing past 0 and has not ended: set 1 waits for it.
  set = array.poll_set();
  CHECK(set.ok() && !set.value());
  // B skips 1 and ends after 2.
  b.push(3100, 2);
  b.done = true;
  set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->sequence == 1);
  CHECK(set.value()->frames[0] && !set.value()->frames[1]);
  set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->sequence == 2 &&
        set.value()->complete());
  CHECK(!array.exhausted());  // A has not ended
  a.done = true;
  set = array.poll_set();
  CHECK(set.ok() && !set.value());
  CHECK(array.exhausted());
  // No calibration: frames keep their driver's pose.
  return 0;
}

int test_moved_from() {
  Rig r = triggered_rig();
  auto opened =
      sensor::SensorArray::open(std::move(r.sensors), trigger_options());
  CHECK(opened.ok());
  sensor::SensorArray a = std::move(opened).value();
  sensor::SensorArray b = std::move(a);
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is checked
  CHECK(a.size() == 0 && a.exhausted());
  CHECK(a.poll_set().status().domain() == Code::InvalidArgument);
  CHECK(a.start().domain() == Code::InvalidArgument);
  CHECK(b.size() == 3);
  return 0;
}

}  // namespace

int main() {
  if (test_refusals() != 0) return 1;
  if (test_start_order() != 0) return 1;
  if (test_trigger() != 0) return 1;
  if (test_sequence() != 0) return 1;
  if (test_moved_from() != 0) return 1;
  std::printf("sensor array tests passed\n");
  return 0;
}
