// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The frame contract's checker (testing/rgbd_contract.hpp) on fake frames,
// host-only: a clean script passes, a gap in the sequences among them, and
// so does one whose counters sum to exactly what was received; each rule
// broken once is the one named, with its frame's place: a repeated sequence
// and a time that goes back, a frame without depth or with a depth camera of
// another size, with a colour camera a frame without colour, with a colour
// camera or a picture of another size, without one a frame with colour,
// delivered other than taken and the counters past received. Only the first
// breach is kept, and rgbd_breach over a vector says what the class says.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/testing/rgbd_contract.hpp"

namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace testing = volumetric_kit::sensor::testing;

namespace {

const std::uint16_t kDepth[1] = {1000};
const std::uint8_t kPlanes[1] = {0};
const std::uint32_t kPacked[1] = {0};

camera::CameraModel model(std::uint32_t width, std::uint32_t height) {
  camera::CameraModel m;
  m.size = {width, height};
  m.intrinsics = {0.75 * width, 0.75 * width, width / 2.0, height / 2.0};
  return m;
}

// A tracked phone's: 256 x 192 depth beside a 1920 x 1440 picture.
sensor::SensorInfo info_with_color() {
  sensor::SensorInfo info;
  info.id = "fake";
  info.depth = model(256, 192);
  info.color = model(1920, 1440);
  return info;
}

// Frame `i` of a clean script: host depth and a host picture at the
// cameras' sizes, the sequence with a gap after frame 1, times 33 ms apart.
sensor::RgbdFrame frame(const sensor::SensorInfo& info, std::uint64_t i) {
  sensor::RgbdFrame f;
  f.depth = kDepth;
  f.depth_camera = *info.depth;
  if (info.color) {
    f.color.layout = sensor::YuvLayout::Nv12;
    f.color.host = kPlanes;
    f.color.width = info.color->size.width;
    f.color.height = info.color->size.height;
    f.color_camera = *info.color;
  }
  f.sequence = 1 + i + (i >= 2 ? 1 : 0);  // 1, 2, 4, 5: frame 3 was lost
  f.timestamp_ns = 1'000'000'000 + 33'333'333 * i;
  return f;
}

std::vector<sensor::RgbdFrame> script(const sensor::SensorInfo& info) {
  std::vector<sensor::RgbdFrame> frames;
  for (std::uint64_t i = 0; i < 4; ++i) frames.push_back(frame(info, i));
  return frames;
}

// Six received: four delivered, one dropped, one failed, so the counters sum
// to exactly what was received.
sensor::SensorStats stats_of(std::size_t delivered) {
  sensor::SensorStats s;
  s.received = 6;
  s.delivered = delivered;
  s.dropped = 1;
  s.failed = 1;
  return s;
}

// The class's answer for `frames`, taken one at a time.
std::string taken(const sensor::SensorInfo& info,
                  const std::vector<sensor::RgbdFrame>& frames,
                  const sensor::SensorStats& stats) {
  testing::RgbdContract contract(info);
  for (const sensor::RgbdFrame& f : frames) contract.take(f);
  return contract.breach(stats);
}

// Whether the class and rgbd_breach both name `why` for `frames`, word for
// word the same.
bool names(const sensor::SensorInfo& info,
           const std::vector<sensor::RgbdFrame>& frames,
           const sensor::SensorStats& stats, const std::string& why) {
  const std::string got = taken(info, frames, stats);
  const std::string over = testing::rgbd_breach(info, frames, stats);
  if (got != why || over != got) {
    std::fprintf(stderr,
                 "expected \"%s\": the class \"%s\", the vector \"%s\"\n",
                 why.c_str(), got.c_str(), over.c_str());
    return false;
  }
  return true;
}

int test_rules() {
  const sensor::SensorInfo info = info_with_color();
  const std::vector<sensor::RgbdFrame> clean = script(info);
  CHECK(names(info, clean, stats_of(4), ""));
  CHECK(names(info, {}, sensor::SensorStats{}, ""));

  // The order: a repeated sequence, and a time that goes back.
  std::vector<sensor::RgbdFrame> bad = clean;
  bad[2].sequence = bad[1].sequence;
  CHECK(names(info, bad, stats_of(4), "frame 2: sequence 2 after 2"));
  bad = clean;
  bad[1].timestamp_ns = bad[0].timestamp_ns - 1;
  CHECK(names(info, bad, stats_of(4),
              "frame 1: time 999999999 ns after 1000000000"));

  // Depth: none, on the host or the device, and a camera of another size.
  bad = clean;
  bad[0].depth = nullptr;
  CHECK(names(info, bad, stats_of(4), "frame 0: no depth"));
  bad = clean;
  bad[3].depth_camera = model(128, 96);
  CHECK(names(info, bad, stats_of(4),
              "frame 3: depth camera 128 x 96, not the sensor's 256 x 192"));

  // Colour, from a sensor with a colour camera: none, a camera of another
  // size, a picture of another size than its camera's.
  bad = clean;
  bad[1].color = sensor::YuvImage{};
  CHECK(names(info, bad, stats_of(4),
              "frame 1: no colour from a sensor with a colour camera"));
  bad = clean;
  bad[2].color_camera = model(1440, 1920);
  CHECK(names(info, bad, stats_of(4),
              "frame 2: colour camera 1440 x 1920, not the sensor's 1920 x "
              "1440"));
  bad = clean;
  bad[1].color.width = 960;
  bad[1].color.height = 720;
  CHECK(names(info, bad, stats_of(4),
              "frame 1: picture 960 x 720, not its camera's 1920 x 1440"));
  // Packed host words are their camera's size: no picture size to break.
  std::vector<sensor::RgbdFrame> packed = clean;
  for (sensor::RgbdFrame& f : packed) {
    f.color = sensor::YuvImage{};
    f.color_packed = kPacked;
  }
  CHECK(names(info, packed, stats_of(4), ""));

  // A sensor without a colour camera: frames without colour keep it, one
  // with colour breaks it.
  sensor::SensorInfo depth_only = info;
  depth_only.color.reset();
  const std::vector<sensor::RgbdFrame> plain = script(depth_only);
  CHECK(names(depth_only, plain, stats_of(4), ""));
  bad = plain;
  bad[2].color_packed = kPacked;
  CHECK(names(depth_only, bad, stats_of(4),
              "frame 2: colour from a sensor with no colour camera"));
  // Without colour the order still counts.
  bad = plain;
  bad[3].sequence = bad[2].sequence;
  CHECK(names(depth_only, bad, stats_of(4), "frame 3: sequence 4 after 4"));
  // A sensor that names no depth camera holds no size to keep.
  sensor::SensorInfo unnamed = depth_only;
  unnamed.depth.reset();
  CHECK(names(unnamed, plain, stats_of(4), ""));

  // The counters: delivered not the frames taken, and past received.
  CHECK(names(info, clean, stats_of(3), "delivered 3, taken 4"));
  sensor::SensorStats over = stats_of(4);
  over.failed = 2;
  CHECK(
      names(info, clean, over, "delivered + dropped + failed past received 6"));
  std::puts("each rule broken once is the one named");

  // The first breach is kept: frame 1's, not frame 3's or the counters'.
  bad = clean;
  bad[1].depth = nullptr;
  bad[3].sequence = 1;
  CHECK(names(info, bad, stats_of(9), "frame 1: no depth"));
  std::puts("only the first breach is kept");
  return 0;
}

// The counter and order rules on their own, as the stereo contract calls
// them.
int test_rules_alone() {
  CHECK(testing::order_breach(1, 10, 2, 11).empty());
  CHECK(testing::order_breach(5, 10, 5, 11) == "sequence 5 after 5");
  CHECK(testing::order_breach(5, 10, 6, 10) == "time 10 ns after 10");
  sensor::SensorStats s;
  s.received = 3;
  s.delivered = 2;
  s.failed = 1;
  CHECK(testing::stats_breach(s, 2).empty());
  CHECK(testing::stats_breach(s, 1) == "delivered 2, taken 1");
  s.dropped = 1;
  CHECK(testing::stats_breach(s, 2) ==
        "delivered + dropped + failed past received 3");
  std::puts("order_breach and stats_breach");
  return 0;
}

}  // namespace

int main() {
  if (test_rules() != 0 || test_rules_alone() != 0) return 1;
  std::puts("sensor_rgbd_contract: OK");
  return 0;
}
