// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sensor array's calibration file: poses read as the family's
// device_calibration layout writes them, and the refusals. Host-only.

#include <cmath>
#include <cstdio>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

// The family's layout (config_nunga.json), trimmed: an unrelated section, a
// non-object entry, lens fields this reader ignores, and two posed cameras:
// A at the origin, B turned 90 degrees about +Y with tvec (0, 0, 2).
const char* kCalibration = R"({
  "kinect_config": {"device_count": 2, "master_serial": "A"},
  "device_calibration": {
    "pose_calib_file_path": "/somewhere/all_extrinsics.npy",
    "B": {"intrinsics": {"fx": 746.5, "fy": 746.3, "cx": 630.4, "cy": 345.6},
          "pose": {"rvec": [0.0, 1.5707963267948966, 0.0],
                   "tvec": [0.0, 0.0, 2.0]}},
    "A": {"pose": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}}
  }
})";

bool refused(const std::string& json) {
  const auto r = camera::parse_array_calibration(json);
  if (r.ok()) return false;
  std::printf("  refused as expected: %s\n", r.status().message().c_str());
  return r.status().domain() == vkc::Status::Code::InvalidArgument;
}

int test_read() {
  const auto r = camera::parse_array_calibration(kCalibration);
  CHECK(r.ok());
  const camera::ArrayCalibration& a = r.value();
  CHECK(a.sensors.size() == 2);
  CHECK(a.sensors[0].id == "A" && a.sensors[1].id == "B");  // in id order
  CHECK(a.sensors[0].color_to_world == camera::Mat4d(1.0));
  // The camera turned 90 degrees about +Y, tvec (0, 0, 2), sits at (2, 0, 0).
  const camera::SensorCalibration* b = camera::find_sensor(a, "B");
  CHECK(b != nullptr);
  CHECK(glm::length(camera::Vec3d(b->color_to_world[3]) -
                    camera::Vec3d(2.0, 0.0, 0.0)) < 1e-15);
  CHECK(camera::find_sensor(a, "missing") == nullptr);
  return 0;
}

int test_refusals() {
  CHECK(refused("{not json"));
  CHECK(refused("[1, 2]"));
  CHECK(refused(R"({"kinect_config": {}})"));
  CHECK(refused(R"({"device_calibration": {}})"));  // no sensors
  CHECK(refused(R"({"device_calibration": {"A": {"intrinsics":
      {"fx": 1, "fy": 1, "cx": 0, "cy": 0}}}})"));  // no pose
  CHECK(refused(R"({"device_calibration": {"A": {"pose":
      {"rvec": [0, 0], "tvec": [0, 0, 0]}}}})"));
  CHECK(refused(R"({"device_calibration": {"A": {"pose":
      {"rvec": [0, 0, 0], "tvec": [0, "0", 0]}}}})"));

  camera::SensorCalibration a;
  a.id = "A";
  camera::ArrayCalibration array;
  CHECK(!camera::validate_array_calibration(array).ok());  // no sensors
  array.sensors = {a, a};
  CHECK(!camera::validate_array_calibration(array).ok());  // twice
  a.color_to_world = camera::Mat4d(2.0);
  array.sensors = {a};
  CHECK(!camera::validate_array_calibration(array).ok());  // not rigid
  a = {};
  array.sensors = {a};
  CHECK(!camera::validate_array_calibration(array).ok());  // no id
  return 0;
}

int test_file() {
  const std::string path =
      std::string(VR_TEST_SCRATCH_DIR) + "/array_calibration.json";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  CHECK(f != nullptr);
  CHECK(std::fputs(kCalibration, f) >= 0 && std::fclose(f) == 0);
  const auto read = camera::read_array_calibration(path);
  CHECK(read.ok() && read.value().sensors.size() == 2);

  const auto missing = camera::read_array_calibration(path + ".missing");
  CHECK(!missing.ok() &&
        missing.status().domain() == vkc::Status::Code::IoError);
  // A refused document's message names the file.
  f = std::fopen(path.c_str(), "wb");
  CHECK(f != nullptr);
  CHECK(std::fputs("{\"device_calibration\": {}}", f) >= 0 &&
        std::fclose(f) == 0);
  const auto refused_file = camera::read_array_calibration(path);
  CHECK(!refused_file.ok() &&
        refused_file.status().message().rfind(path, 0) == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_read() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_file() != 0) return 1;
  std::printf("array calibration tests passed\n");
  return 0;
}
