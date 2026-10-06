// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The RGB-D sensor contract, implemented by a fake and driven through a
// base-class reference, as a consumer holds one: the no_frame/some_frame
// returns compile and mean what they say, poll hands out the newest frame and
// drain every held one oldest first, a frame outlives the sensor's buffer
// through `pixels`, and the enum names are stable. Host-only.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

// A sensor that "captures" a 2x2 depth frame each time the test calls
// capture(), holding at most `depth` of them as a driver's mailbox does.
class FakeSensor final : public sensor::IRgbdSensor {
 public:
  FakeSensor() {
    info_.id = "fake";
    info_.depth = volumetric_kit::recon::camera::CameraModel{
        {2, 2}, {1.0, 1.0, 0.5, 0.5}, {}};
  }

  void capture() {
    ++stats_.received;
    // Each frame's pixels live in a buffer the frame holds, not the sensor.
    auto pixels = std::make_shared<std::vector<std::uint16_t>>(
        4, static_cast<std::uint16_t>(1000 + next_));
    sensor::RgbdFrame f;
    f.depth = pixels->data();
    f.pixels = pixels;
    f.depth_camera = *info_.depth;
    f.sequence = next_++;
    held_.push_back(std::move(f));
    while (held_.size() > depth_) {
      held_.erase(held_.begin());
      ++stats_.dropped;
    }
  }

  const sensor::SensorInfo& info() const noexcept override { return info_; }
  vkc::Status set_queue_depth(std::size_t frames) override {
    if (frames == 0 || running_) {
      return vkc::Status::invalid_argument("FakeSensor: bad queue depth");
    }
    depth_ = frames;
    return {};
  }
  vkc::Status start() override {
    running_ = true;
    return {};
  }
  void stop() noexcept override {
    running_ = false;
    held_.clear();
  }
  vkc::Result<std::optional<sensor::RgbdFrame>> poll() override {
    if (held_.empty()) return no_frame();
    sensor::RgbdFrame newest = std::move(held_.back());
    stats_.dropped += held_.size() - 1;
    held_.clear();
    ++stats_.delivered;
    return some_frame(std::move(newest));
  }
  vkc::Status drain(std::vector<sensor::RgbdFrame>* out) override {
    for (sensor::RgbdFrame& f : held_) out->push_back(std::move(f));
    stats_.delivered += held_.size();
    held_.clear();
    return {};
  }
  sensor::SensorStats stats() const noexcept override { return stats_; }

 private:
  sensor::SensorInfo info_;
  std::vector<sensor::RgbdFrame> held_;
  std::size_t depth_ = 1;
  bool running_ = false;
  std::uint64_t next_ = 0;
  sensor::SensorStats stats_;
};

int test_poll_and_drain(FakeSensor& fake) {
  sensor::IRgbdSensor& s = fake;  // as a consumer holds one
  CHECK(s.info().id == "fake" && s.info().depth && !s.info().color);
  CHECK(s.info().role == sensor::SyncRole::FreeRun);
  CHECK(s.info().clock == sensor::ClockDomain::Device);
  CHECK(s.info().pose == sensor::PoseSource::Fixed);
  CHECK(!s.set_queue_depth(0).ok());
  CHECK(s.set_queue_depth(3).ok());
  CHECK(s.start().ok());
  CHECK(!s.set_queue_depth(2).ok());  // set before start

  auto none = s.poll();
  CHECK(none.ok() && !none.value());  // nothing this tick: not an error
  CHECK(!s.exhausted());

  for (int i = 0; i < 5; ++i) fake.capture();  // 0..4; 0 and 1 pushed out
  std::vector<sensor::RgbdFrame> frames;
  CHECK(s.drain(&frames).ok());
  CHECK(frames.size() == 3);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    CHECK(frames[i].sequence == 2 + i);  // oldest first
  }
  frames.clear();
  CHECK(s.drain(&frames).ok() && frames.empty());

  fake.capture();
  fake.capture();
  auto newest = s.poll();
  CHECK(newest.ok() && newest.value() && newest.value()->sequence == 6);
  const sensor::SensorStats st = s.stats();
  CHECK(st.received == 7 && st.delivered == 4 && st.dropped == 3);
  CHECK(st.delivered + st.dropped + st.failed <= st.received);
  return 0;
}

int test_frames_hold_pixels(FakeSensor& fake) {
  fake.capture();
  auto polled = fake.poll();
  CHECK(polled.ok() && polled.value());
  sensor::RgbdFrame kept = *polled.value();
  // The sensor lets everything go; the frame still reads its pixels.
  fake.stop();
  for (int i = 0; i < 3; ++i) fake.capture();
  CHECK(kept.pixels != nullptr && kept.depth[3] == 1000 + kept.sequence);
  return 0;
}

int test_names() {
  CHECK(std::strcmp(sensor::to_string(sensor::SyncRole::Primary), "primary") ==
        0);
  CHECK(std::strcmp(sensor::to_string(sensor::SyncRole::Secondary),
                    "secondary") == 0);
  CHECK(std::strcmp(sensor::to_string(sensor::SyncRole::FreeRun), "free-run") ==
        0);
  CHECK(std::strcmp(sensor::to_string(sensor::ClockDomain::Host), "host") == 0);
  CHECK(std::strcmp(sensor::to_string(sensor::PoseSource::Tracked),
                    "tracked") == 0);
  return 0;
}

}  // namespace

int main() {
  FakeSensor fake;
  if (test_poll_and_drain(fake) != 0) return 1;
  if (test_frames_hold_pixels(fake) != 0) return 1;
  if (test_names() != 0) return 1;
  std::printf("rgbd sensor contract tests passed\n");
  return 0;
}
