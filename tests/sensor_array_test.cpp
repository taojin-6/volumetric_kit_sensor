// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sensor array over scripted fake sensors: the arrays it refuses, the
// order it starts and stops sensors in, a failed start, a start of a running
// array and a move over one, trigger grouping (complete sets, a silent
// secondary, the newest set winning, frames with no clock), a set's skew and
// spread, the default tolerance, an exhausted sensor, a drain that fails
// partway, poses from the calibration (an unposed sensor refused) whatever
// its world, the colour exposure a calibration names (a sensor opened at
// another refused), and a moved-from array. Host-only.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/array/sensor_array.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::sensor::camera;
namespace sensor = volumetric_kit::sensor;

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
    auto pixels = std::make_shared<int>(0);
    last_pixels = pixels;
    f.pixels = std::move(pixels);
    frames_.push_back(std::move(f));
  }
  sensor::SensorInfo& mutable_info() { return info_; }
  bool fail_start = false;
  bool done = false;  // no more frames will come
  std::size_t queue_depth = 0;
  // Hand out this many frames, then fail, as a driver may.
  std::optional<std::size_t> fail_drain_after;
  std::weak_ptr<const void> last_pixels;  // the last frame pushed's

  const sensor::SensorInfo& info() const noexcept override { return info_; }
  vkc::Status set_queue_depth(std::size_t frames) override {
    if (started_) return vkc::Status::invalid_argument("already started");
    queue_depth = frames;
    return {};
  }
  vkc::Status start() override {
    g_log.push_back("start " + info_.id);
    if (fail_start) return vkc::Status::io_error(info_.id + " will not start");
    started_ = true;
    return {};
  }
  void stop() noexcept override {
    g_log.push_back("stop " + info_.id);
    started_ = false;
  }
  vkc::Result<std::optional<sensor::RgbdFrame>> poll() override {
    if (frames_.empty()) return no_frame();
    sensor::RgbdFrame f = std::move(frames_.back());
    frames_.clear();
    return some_frame(std::move(f));
  }
  vkc::Status drain(std::vector<sensor::RgbdFrame>* out) override {
    const std::size_t n =
        std::min(frames_.size(), fail_drain_after.value_or(frames_.size()));
    for (std::size_t i = 0; i < n; ++i) out->push_back(std::move(frames_[i]));
    const bool fail = n < frames_.size();
    frames_.clear();
    if (fail) return vkc::Status::io_error(info_.id + " failed mid-drain");
    return {};
  }
  bool exhausted() const noexcept override { return done && frames_.empty(); }
  sensor::SensorStats stats() const noexcept override { return {}; }

 private:
  sensor::SensorInfo info_;
  std::vector<sensor::RgbdFrame> frames_;
  bool started_ = false;
};

// A sensor that notifies: frames posted, or a failure, from any thread, as a
// driver's own thread delivers them.
class Notifying final : public sensor::IRgbdSensor {
 public:
  Notifying(std::string id, sensor::SyncRole role, std::uint32_t fps) {
    info_.id = std::move(id);
    info_.role = role;
    info_.clock = sensor::ClockDomain::Host;
    info_.fps = fps;
  }
  // A frame, stamped as arriving at `arrival_ns`, then the wake-up.
  void post(std::uint64_t timestamp_ns, std::uint64_t arrival_ns) {
    std::function<void()> wake;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      sensor::RgbdFrame f;
      f.timestamp_ns = timestamp_ns;
      f.arrival_ns = arrival_ns;
      frames_.push_back(std::move(f));
      wake = notify_;
    }
    if (wake) wake();
  }
  // A failure the next drain reports, then the wake-up.
  void fail(const std::string& why) {
    std::function<void()> wake;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fault_ = why;
      wake = notify_;
    }
    if (wake) wake();
  }
  std::function<void()> notify() {
    std::lock_guard<std::mutex> lock(mutex_);
    return notify_;
  }

  const sensor::SensorInfo& info() const noexcept override { return info_; }
  vkc::Status set_queue_depth(std::size_t) override { return {}; }
  vkc::Status start() override {
    started_ = true;
    return {};
  }
  void stop() noexcept override { started_ = false; }
  vkc::Result<std::optional<sensor::RgbdFrame>> poll() override {
    return no_frame();
  }
  vkc::Status drain(std::vector<sensor::RgbdFrame>* out) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fault_.empty()) return vkc::Status::io_error(fault_);
    for (sensor::RgbdFrame& f : frames_) out->push_back(std::move(f));
    frames_.clear();
    return {};
  }
  sensor::SensorStats stats() const noexcept override { return {}; }
  vkc::Status set_frame_notify(std::function<void()> notify) override {
    if (started_) return vkc::Status::invalid_argument("already started");
    std::lock_guard<std::mutex> lock(mutex_);
    notify_ = std::move(notify);
    return {};
  }

 private:
  sensor::SensorInfo info_;
  std::mutex mutex_;
  std::vector<sensor::RgbdFrame> frames_;  // guarded by mutex_
  std::function<void()> notify_;           // guarded by mutex_
  std::string fault_;                      // guarded by mutex_
  bool started_ = false;
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
  o.tolerance_us = 200;  // under half of 1 kHz's period
  return o;
}

// Each sensor posed one metre further along x.
camera::ArrayCalibration calibration(const std::vector<std::string>& ids) {
  camera::ArrayCalibration c;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    camera::SensorCalibration s;
    s.id = ids[i];
    camera::Mat4d pose(1.0);
    pose[3] = glm::dvec4(static_cast<double>(i), 0.0, 0.0, 1.0);
    s.color_to_world = pose;
    c.sensors.push_back(s);
  }
  return c;
}

sensor::SensorArray open_array(Rig* r, const sensor::SensorArray::Options& o) {
  auto opened = sensor::SensorArray::open(std::move(r->sensors), o);
  return std::move(opened).value();  // aborts if refused
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
  o.tolerance_us = 500;  // half of 1 kHz's period
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  o.tolerance_us = 499;
  CHECK(refusal(triggered_rig(), o) == Code::Ok);
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
  // A calibration must pose every sensor: one it lists for its lenses alone
  // is as unposed as one it does not list.
  o = trigger_options();
  o.calibration = calibration({"S1", "P"});
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  o.calibration = calibration({"S1", "S2", "P"});
  CHECK(refusal(triggered_rig(), o) == Code::Ok);
  o.calibration.sensors[1].color_to_world.reset();
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  {  // A refusal leaves the sensors with the caller.
    Rig r = triggered_rig();
    auto bad = trigger_options();
    bad.queue_depth = 0;
    CHECK(!sensor::SensorArray::open(std::move(r.sensors), bad).ok());
    // NOLINTNEXTLINE(bugprone-use-after-move): open takes them only on success
    CHECK(r.sensors.size() == 3 && r.sensors[1] != nullptr);
  }
  {  // One sensor needs no primary, nor the host's clock.
    Rig r;
    r.add(std::make_unique<Scripted>("A", sensor::SyncRole::FreeRun,
                                     sensor::ClockDomain::Device));
    CHECK(refusal(std::move(r), trigger_options()) == Code::Ok);
  }
  return 0;
}

int test_start_order() {
  Rig r = triggered_rig();
  Scripted* p = r.fakes[1];
  // A live sensor inherits "not exhausted" unless its driver overrides it.
  CHECK(!p->IRgbdSensor::exhausted());
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

  // A sensor that will not start stops them all, the primary first.
  p->fail_start = true;
  g_log.clear();
  const vkc::Status failed = array.start();
  CHECK(!failed.ok() && failed.domain() == Code::IoError);
  CHECK((g_log == std::vector<std::string>{"start S1", "start S2", "start P",
                                           "stop P", "stop S2", "stop S1"}));
  auto none = array.poll_set();
  CHECK(none.ok() && !none.value());  // not running

  // Starting a running array starts each sensor again, without setting its
  // queue, and stops them all once one has failed since.
  p->fail_start = false;
  CHECK(array.start().ok());
  CHECK(array.start().ok());
  p->fail_start = true;
  g_log.clear();
  CHECK(array.start().domain() == Code::IoError);
  CHECK((g_log == std::vector<std::string>{"start S1", "start S2", "start P",
                                           "stop P", "stop S2", "stop S1"}));
  none = array.poll_set();
  CHECK(none.ok() && !none.value());  // stopped

  // A move over a running array stops its sensors in order.
  p->fail_start = false;
  CHECK(array.start().ok());
  Rig other = triggered_rig();
  g_log.clear();
  array = open_array(&other, trigger_options());
  CHECK((g_log == std::vector<std::string>{"stop P", "stop S2", "stop S1"}));
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
  CHECK(set.value()->skew_ns(0) == 80'000 && set.value()->skew_ns(1) == 0);
  CHECK(set.value()->skew_ns(2) == 150'000);
  CHECK(set.value()->spread_ns() == 150'000);
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
  CHECK(!set.value()->skew_ns(2) && set.value()->spread_ns() == 80'000);

  // A slow poll gets the newest set; the older ones' frames go unmatched.
  for (std::uint64_t t = 2; t < 5; ++t) trigger(t, true);
  set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->sequence == 104);
  // A frame without a timestamp has no trigger, and is let go at once.
  f[0]->push(0, 999);
  set = array.poll_set();
  CHECK(set.ok() && !set.value());
  CHECK(f[0]->last_pixels.expired());

  const sensor::SensorArrayStats st = array.stats();
  CHECK(st.sets == 3 && st.incomplete == 1);
  CHECK(st.unmatched == 2 * 3 + 1);
  CHECK(st.sensors.size() == 3);

  // An exhausted sensor (a disconnected camera) exhausts the array.
  CHECK(!array.exhausted());
  f[2]->done = true;
  CHECK(array.exhausted());
  return 0;
}

// A calibration's world: a tag's file poses each sensor as it says, none at
// the origin; a sensor world must still be a sensor at the origin.
int test_world() {
  camera::ArrayCalibration tagged = calibration({"S1", "P", "S2"});
  for (camera::SensorCalibration& s : tagged.sensors) {
    (*s.color_to_world)[3].z = 2.5;  // over the floor tag
  }
  tagged.world.apriltag = camera::AprilTagWorld{"tag36h11", 0, 0.15};
  const auto text = camera::format_array_calibration(tagged);
  CHECK(text.ok());
  auto o = trigger_options();
  auto read = camera::parse_array_calibration(text.value());
  CHECK(read.ok() && read.value().world.apriltag);
  o.calibration = std::move(read).value();
  Rig r = triggered_rig();
  std::vector<Scripted*> f = r.fakes;  // S1, P, S2
  sensor::SensorArray array = open_array(&r, o);
  CHECK(array.start().ok());
  f[1]->push(1'000'000'000, 1);
  f[0]->push(1'000'000'000, 2);
  f[2]->push(1'000'000'000, 3);
  const auto set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->complete());
  for (std::size_t c = 0; c < 3; ++c) {
    const camera::Vec3d centre(set.value()->frames[c]->color_to_world[3]);
    CHECK(glm::length(centre -
                      camera::Vec3d(static_cast<double>(c), 0.0, 2.5)) < 1e-15);
  }

  // A sensor world: S1, at the origin, may be it; P, a metre off, may not.
  o = trigger_options();
  o.calibration = calibration({"S1", "P", "S2"});
  o.calibration.world.sensor = "S1";
  CHECK(refusal(triggered_rig(), o) == Code::Ok);
  o.calibration.world.sensor = "P";
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  // Never both.
  o.calibration.world.sensor = "S1";
  o.calibration.world.apriltag = camera::AprilTagWorld{"tag36h11", 0, 0.15};
  CHECK(refusal(triggered_rig(), o) == Code::InvalidArgument);
  return 0;
}

// A calibration that names the colour cameras' exposure takes only sensors
// opened at it, naming one that is not, whether it lists sensors or not; one
// that names none checks nothing.
int test_color_controls() {
  const camera::ColorControls lock{167, 60};
  auto o = trigger_options();
  o.calibration = calibration({"S1", "P", "S2"});
  {  // None in the calibration: whatever the sensors run at.
    Rig r = triggered_rig();
    r.fakes[0]->mutable_info().color_controls = lock;
    CHECK(refusal(std::move(r), o) == Code::Ok);
  }
  o.calibration.color_controls = lock;
  {  // Every sensor at them.
    Rig r = triggered_rig();
    for (Scripted* f : r.fakes) f->mutable_info().color_controls = lock;
    CHECK(refusal(std::move(r), o) == Code::Ok);
  }
  // One at auto exposure, or at another exposure.
  for (const auto& [other, says] :
       {std::pair{std::optional<camera::ColorControls>(), "auto exposure"},
        std::pair{std::optional(camera::ColorControls{167, 61}),
                  "exposure 167, gain 61"}}) {
    Rig r = triggered_rig();
    for (Scripted* f : r.fakes) f->mutable_info().color_controls = lock;
    r.fakes[2]->mutable_info().color_controls = other;
    const auto a = sensor::SensorArray::open(std::move(r.sensors), o);
    CHECK(!a.ok() && a.status().domain() == Code::InvalidArgument);
    const std::string& m = a.status().message();
    std::printf("  refused as expected: %s\n", m.c_str());
    CHECK(m.find(std::string("sensor S2 runs at ") + says) !=
          std::string::npos);
    CHECK(m.find("color_controls (exposure 167, gain 60)") !=
          std::string::npos);
  }
  // A calibration that names the controls but lists no sensor checks them
  // all the same.
  auto bare = trigger_options();
  bare.calibration.color_controls = lock;
  CHECK(refusal(triggered_rig(), bare) == Code::InvalidArgument);
  {
    Rig r = triggered_rig();
    for (Scripted* f : r.fakes) f->mutable_info().color_controls = lock;
    CHECK(refusal(std::move(r), bare) == Code::Ok);
  }
  return 0;
}

// The default tolerance: 0.4 of the fastest sensor's frame period, 400 us at
// 1 kHz, or 5000 us when no sensor reports a rate. A secondary just inside it
// joins the primary's set; one just outside joins none.
int test_default_tolerance() {
  for (const std::uint32_t fps : {1000u, 0u}) {
    const std::uint64_t tolerance_ns = fps > 0 ? 400'000 : 5'000'000;
    Rig r;
    r.add(std::make_unique<Scripted>("S1", sensor::SyncRole::Secondary,
                                     sensor::ClockDomain::Host, fps));
    r.add(std::make_unique<Scripted>("P", sensor::SyncRole::Primary,
                                     sensor::ClockDomain::Host, fps));
    r.add(std::make_unique<Scripted>("S2", sensor::SyncRole::Secondary,
                                     sensor::ClockDomain::Host, fps));
    std::vector<Scripted*> f = r.fakes;
    sensor::SensorArray array = open_array(&r, {});
    CHECK(array.start().ok());
    const std::uint64_t ts = 5'000'000'000;
    f[1]->push(ts, 100);
    f[0]->push(ts + tolerance_ns - 50'000, 200);
    f[2]->push(ts + tolerance_ns + 50'000, 300);
    const auto set = array.poll_set();
    CHECK(set.ok() && set.value() && set.value()->count() == 2);
    CHECK(set.value()->frames[0]->sequence == 200 && !set.value()->frames[2]);
  }
  return 0;
}

int test_failed_drain() {
  Rig r = triggered_rig();
  std::vector<Scripted*> f = r.fakes;  // S1, P, S2
  sensor::SensorArray array = open_array(&r, trigger_options());
  CHECK(array.start().ok());

  // S1 hands out its first frame before it fails: the failure is returned
  // and the frame kept for its trigger.
  const std::uint64_t ts = 5'000'000'000;
  f[0]->push(ts + 80'000, 200);
  f[0]->push(ts + 1'080'000, 201);
  f[0]->fail_drain_after = 1;
  f[1]->push(ts, 100);
  f[2]->push(ts + 150'000, 300);
  CHECK(array.poll_set().status().domain() == Code::IoError);
  f[0]->fail_drain_after.reset();
  auto set = array.poll_set();
  CHECK(set.ok() && set.value() && set.value()->sequence == 100);
  CHECK(set.value()->complete() && set.value()->frames[0]->sequence == 200);
  return 0;
}

int test_failed_drain_eviction() {
  Rig r = triggered_rig();
  std::vector<Scripted*> f = r.fakes;  // S1, P, S2
  auto options = trigger_options();
  options.queue_depth = 4;
  sensor::SensorArray array = open_array(&r, options);
  CHECK(array.start().ok());

  // The last sensor repeatedly fails while the earlier two keep delivering.
  // Their grouping queues must still release evicted pixel owners on every
  // failed poll, and retain only the frames a later successful poll can use.
  f[2]->fail_drain_after = 0;
  std::vector<std::weak_ptr<const void>> pixels[2];
  const std::uint64_t ts = 5'000'000'000;
  constexpr std::size_t count = 20;
  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t c = 0; c < f.size(); ++c) {
      f[c]->push(ts + i * 1'000'000, i);
      if (c < 2) pixels[c].push_back(f[c]->last_pixels);
    }
    CHECK(array.poll_set().status().domain() == Code::IoError);
    const std::size_t evicted =
        i + 1 > options.queue_depth ? i + 1 - options.queue_depth : 0;
    for (const auto& camera : pixels) {
      for (std::size_t j = 0; j < camera.size(); ++j) {
        CHECK(camera[j].expired() == (j < evicted));
      }
    }
    CHECK(array.stats().unmatched == 2 * evicted);
  }

  // Recovery can still complete the newest trigger from the frames retained
  // through those errors, without redelivering either healthy camera's frame.
  f[2]->fail_drain_after.reset();
  f[2]->push(ts + (count - 1) * 1'000'000, count - 1);
  {
    auto set = array.poll_set();
    CHECK(set.ok() && set.value() && set.value()->complete());
    CHECK(set.value()->sequence == count - 1);
    CHECK(array.stats().unmatched == 2 * (count - 1));
    for (const auto& camera : pixels) {
      for (std::size_t j = 0; j < camera.size(); ++j) {
        CHECK(camera[j].expired() == (j + 1 < count));
      }
    }
  }
  for (const auto& camera : pixels) {
    CHECK(camera.back().expired());
  }
  return 0;
}

// A set's skew, each frame against the trigger, signed and saturated, and its
// spread, over the frames it holds, whichever are missing.
int test_skew_and_spread() {
  const std::uint64_t trigger = 7'000'000'000;
  const auto frame = [](std::uint64_t timestamp_ns) {
    sensor::RgbdFrame f;
    f.timestamp_ns = timestamp_ns;
    return std::optional<sensor::RgbdFrame>(f);
  };
  sensor::FrameSet set;
  set.timestamp_ns = trigger;
  set.frames = {frame(trigger + 1'200'000), std::nullopt, frame(trigger),
                frame(trigger - 2'500'000)};
  CHECK(set.skew_ns(0) == 1'200'000);
  CHECK(!set.skew_ns(1).has_value());
  CHECK(set.skew_ns(2) == 0);
  CHECK(set.skew_ns(3) == -2'500'000);  // before the trigger: negative
  CHECK(set.spread_ns() == 3'700'000);

  // The primary's frame gone: the trigger stays the set's, and the spread is
  // over the frames left.
  set.frames[2].reset();
  CHECK(set.skew_ns(0) == 1'200'000 && set.spread_ns() == 3'700'000);
  set.frames[3].reset();
  CHECK(set.spread_ns() == 0);  // one frame
  set.frames[0].reset();
  CHECK(set.spread_ns() == 0 && !set.skew_ns(0));  // none
  CHECK(sensor::FrameSet{}.spread_ns() == 0);

  // Gaps past int64_t's range saturate, either way; 2^63 before the
  // trigger is exactly its least.
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  constexpr std::uint64_t kHalf = std::uint64_t{1} << 63;
  set.timestamp_ns = 0;
  set.frames = {frame(kHalf - 1), frame(kHalf), frame(~std::uint64_t{0})};
  CHECK(set.skew_ns(0) == kMax && set.skew_ns(1) == kMax &&
        set.skew_ns(2) == kMax);
  set.timestamp_ns = kHalf;
  set.frames = {frame(0), frame(1)};
  CHECK(set.skew_ns(0) == kMin && set.skew_ns(1) == kMin + 1);
  set.timestamp_ns = ~std::uint64_t{0};
  CHECK(set.skew_ns(0) == kMin && set.skew_ns(1) == kMin);
  return 0;
}

// Milliseconds since `start`.
double ms_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

// wait_set: blocked on the sensors' notifications, it returns a frame posted
// from another thread, and a failure, at once; with none, at its timeout; a
// trigger missing a secondary, at the grouping's deadline with no new frame;
// a set's arrival is its latest frame's; a sensor that notifies nothing makes
// it refuse; and a notification after the array is gone touches nothing
// freed (ASan).
int test_wait_set() {
  using Clock = std::chrono::steady_clock;
  {
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> one;
    auto owned =
        std::make_unique<Notifying>("N", sensor::SyncRole::FreeRun, 30);
    Notifying* n = owned.get();
    one.push_back(std::move(owned));
    auto opened = sensor::SensorArray::open(std::move(one), {});
    CHECK(opened.ok());
    sensor::SensorArray array = std::move(opened).value();
    // Not started: nothing to wait for.
    auto idle = array.wait_set(std::chrono::milliseconds(1));
    CHECK(idle.ok() && !idle.value());
    CHECK(array.start().ok());
    Clock::time_point t = Clock::now();
    auto none = array.wait_set(std::chrono::milliseconds(30));
    CHECK(none.ok() && !none.value() && ms_since(t) >= 29.0);

    std::thread poster([n] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      n->post(7'000'000, 123);
    });
    t = Clock::now();
    vkc::StageMetrics metrics;
    auto got = array.wait_set(std::chrono::seconds(5), &metrics);
    const double waited = ms_since(t);
    poster.join();
    CHECK(got.ok() && got.value() && got.value()->arrival_ns == 123);
    std::printf("  a posted frame woke the wait after %.1f ms\n", waited);
    CHECK(waited < 1000.0);
    CHECK(metrics.rows().size() == 2);  // "poll", then "wait: cameras"
    CHECK(std::string(metrics.rows()[1].name) == "wait: cameras");
    CHECK(metrics.rows()[1].wait_ms == metrics.rows()[1].cpu_ms);

    std::thread failer([n] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      n->fail("camera gone");
    });
    t = Clock::now();
    auto failed = array.wait_set(std::chrono::seconds(5));
    const double told = ms_since(t);
    failer.join();
    CHECK(!failed.ok() && failed.status().domain() == Code::IoError);
    CHECK(told < 1000.0);

    // A notify kept past the array: harmless.
    const std::function<void()> late = n->notify();
    CHECK(late != nullptr);
    {
      sensor::SensorArray gone = std::move(array);
    }
    late();
  }
  {
    // A trigger missing its secondary, out at 1.5 frame periods (50 ms).
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> rig;
    auto p = std::make_unique<Notifying>("P", sensor::SyncRole::Primary, 30);
    auto s = std::make_unique<Notifying>("S", sensor::SyncRole::Secondary, 30);
    Notifying* primary = p.get();
    Notifying* secondary = s.get();
    rig.push_back(std::move(p));
    rig.push_back(std::move(s));
    auto opened = sensor::SensorArray::open(std::move(rig), {});
    CHECK(opened.ok());
    sensor::SensorArray array = std::move(opened).value();
    CHECK(array.start().ok());
    primary->post(9'000'000'000, 100);
    secondary->post(9'000'100'000, 200);
    auto both = array.wait_set(std::chrono::seconds(5));
    CHECK(both.ok() && both.value() && both.value()->complete());
    CHECK(both.value()->arrival_ns == 200);  // the latest frame's
    primary->post(9'033'333'000, 300);
    const Clock::time_point t = Clock::now();
    auto alone = array.wait_set(std::chrono::seconds(5));
    const double waited = ms_since(t);
    CHECK(alone.ok() && alone.value() && !alone.value()->complete());
    std::printf("  a trigger missing its secondary went out after %.1f ms\n",
                waited);
    CHECK(waited >= 40.0 && waited < 1000.0);
  }
  {
    // A sensor that notifies nothing: wait_set refuses; poll_set it.
    Rig r = triggered_rig();
    auto opened =
        sensor::SensorArray::open(std::move(r.sensors), trigger_options());
    CHECK(opened.ok());
    CHECK(opened->start().ok());
    CHECK(opened->wait_set(std::chrono::milliseconds(1)).status().domain() ==
          Code::Unsupported);
  }
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
  if (test_world() != 0) return 1;
  if (test_color_controls() != 0) return 1;
  if (test_skew_and_spread() != 0) return 1;
  if (test_default_tolerance() != 0) return 1;
  if (test_failed_drain() != 0) return 1;
  if (test_failed_drain_eviction() != 0) return 1;
  if (test_wait_set() != 0) return 1;
  if (test_moved_from() != 0) return 1;
  std::printf("sensor array tests passed\n");
  return 0;
}
