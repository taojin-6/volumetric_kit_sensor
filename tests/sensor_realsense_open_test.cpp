// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// RealSenseSensor::open and RealSenseStereo::open with no camera needed:
// options each refuses before it looks for one, each named; and a serial no
// camera has, whatever is attached, refused as NotFound naming what is.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"
#include "volumetric_kit/sensor/realsense/realsense_stereo.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

template <typename Driver>
bool refused(const typename Driver::Options& options, vkc::Status::Code code,
             const char* why) {
  const vkc::Status s = Driver::open(options).status();
  if (s.domain() != code || s.message().find(why) == std::string::npos) {
    std::fprintf(stderr, "unexpected: %s\n", s.message().c_str());
    return false;
  }
  return true;
}

int test_sensor() {
  using Driver = sensor::RealSenseSensor;
  using Options = Driver::Options;
  const auto invalid = vkc::Status::Code::InvalidArgument;
  Options o;
  o.depth_width = 0;
  CHECK(refused<Driver>(o, invalid, "width or height is 0"));
  o = Options{};
  o.color_height = 0;
  CHECK(refused<Driver>(o, invalid, "width or height is 0"));
  o = Options{};
  o.fps = 0;
  CHECK(refused<Driver>(o, invalid, "frame rate is 0"));
  o = Options{};
  o.min_depth = 0.0f;
  CHECK(refused<Driver>(o, invalid, "0 < min_depth < max_depth"));
  o = Options{};
  o.max_depth = std::numeric_limits<float>::infinity();
  CHECK(refused<Driver>(o, invalid, "must be finite"));
  o = Options{};
  o.min_depth = 7.0f;
  CHECK(refused<Driver>(o, invalid, "0 < min_depth < max_depth"));
  o = Options{};
  o.color_controls = camera::ColorControls{166, 0};
  CHECK(refused<Driver>(o, invalid, "exposure and a gain above 0"));
  o = Options{};
  o.color_to_world[0][0] = 2.0;  // a scale, not a rotation
  CHECK(refused<Driver>(o, invalid, "color_to_world"));

  o = Options{};
  o.serial = "000000000000";
  CHECK(refused<Driver>(o, vkc::Status::Code::NotFound,
                        "RealSenseSensor: camera 000000000000 is not "
                        "attached; attached: "));
  return 0;
}

int test_stereo() {
  using Driver = sensor::RealSenseStereo;
  using Options = Driver::Options;
  const auto invalid = vkc::Status::Code::InvalidArgument;
  Options o;
  o.infrared_width = 0;
  CHECK(refused<Driver>(o, invalid, "RealSenseStereo: a stream's width"));
  o = Options{};
  o.color_width = 0;
  CHECK(refused<Driver>(o, invalid, "width or height is 0"));
  o = Options{};
  o.fps = 0;
  CHECK(refused<Driver>(o, invalid, "frame rate is 0"));
  o = Options{};
  o.color_controls = camera::ColorControls{0, 64};
  CHECK(refused<Driver>(o, invalid, "exposure and a gain above 0"));
  o = Options{};
  o.color_to_world[2][2] = -1.0;  // a mirror, not a rotation
  CHECK(refused<Driver>(o, invalid, "RealSenseStereo: color_to_world"));

  o = Options{};
  o.serial = "000000000000";
  CHECK(refused<Driver>(o, vkc::Status::Code::NotFound,
                        "RealSenseStereo: camera 000000000000 is not "
                        "attached; attached: "));
  return 0;
}

}  // namespace

int main() {
  if (test_sensor() != 0) return 1;
  if (test_stereo() != 0) return 1;
  std::puts("sensor_realsense_open: OK");
  return 0;
}
