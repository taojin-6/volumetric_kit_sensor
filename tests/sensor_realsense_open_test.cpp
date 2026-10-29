// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// RealSenseSensor::open with no camera needed: options it refuses before it
// looks for one, each named; and a serial no camera has, whatever is attached,
// refused as NotFound naming what is.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

bool refused(const sensor::RealSenseSensor::Options& options,
             vkc::Status::Code code, const char* why) {
  const vkc::Status s = sensor::RealSenseSensor::open(options).status();
  if (s.domain() != code || s.message().find(why) == std::string::npos) {
    std::fprintf(stderr, "unexpected: %s\n", s.message().c_str());
    return false;
  }
  return true;
}

}  // namespace

int main() {
  using Options = sensor::RealSenseSensor::Options;
  const auto invalid = vkc::Status::Code::InvalidArgument;
  Options o;
  o.depth_width = 0;
  CHECK(refused(o, invalid, "width or height is 0"));
  o = Options{};
  o.color_height = 0;
  CHECK(refused(o, invalid, "width or height is 0"));
  o = Options{};
  o.fps = 0;
  CHECK(refused(o, invalid, "frame rate is 0"));
  o = Options{};
  o.min_depth = 0.0f;
  CHECK(refused(o, invalid, "0 < min_depth < max_depth"));
  o = Options{};
  o.max_depth = std::numeric_limits<float>::infinity();
  CHECK(refused(o, invalid, "must be finite"));
  o = Options{};
  o.min_depth = 7.0f;
  CHECK(refused(o, invalid, "0 < min_depth < max_depth"));
  o = Options{};
  o.color_controls = camera::ColorControls{166, 0};
  CHECK(refused(o, invalid, "exposure and a gain above 0"));
  o = Options{};
  o.color_to_world[0][0] = 2.0;  // a scale, not a rotation
  CHECK(refused(o, invalid, "color_to_world"));

  o = Options{};
  o.serial = "000000000000";
  CHECK(refused(o, vkc::Status::Code::NotFound,
                "camera 000000000000 is not attached; attached: "));
  std::puts("sensor_realsense_open: OK");
  return 0;
}
