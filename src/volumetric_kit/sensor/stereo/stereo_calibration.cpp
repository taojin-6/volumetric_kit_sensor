// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/stereo/stereo_calibration.hpp"

#include <new>
#include <string>

#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace volumetric_kit::sensor {

namespace {

constexpr const char* kWho = "stereo_calibration";

core::Status refuse(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

}  // namespace

core::Result<StereoCalibration> stereo_calibration(
    const camera::ArrayCalibration& file, std::string_view left_id,
    std::string_view right_id) try {
  VKC_TRY(camera::validate_array_calibration(file).with_context(kWho));
  if (left_id == right_id) {
    return refuse("the left and right cameras are both \"" +
                  std::string(left_id) + "\"");
  }
  const std::string_view ids[2] = {left_id, right_id};
  const camera::SensorCalibration* cameras[2] = {};
  for (int k = 0; k < 2; ++k) {
    const std::string camera = std::string(k == 0 ? "the left" : "the right") +
                               " camera \"" + std::string(ids[k]) + "\"";
    cameras[k] = camera::find_sensor(file, ids[k]);
    if (cameras[k] == nullptr) return refuse(camera + " is not in the file");
    if (!cameras[k]->color_to_world) return refuse(camera + " is unposed");
    if (!cameras[k]->color) return refuse(camera + " has no colour model");
  }
  const camera::CameraModel& left = cameras[0]->color->model;
  const camera::CameraModel& right = cameras[1]->color->model;
  if (left.size.width != right.size.width ||
      left.size.height != right.size.height) {
    return refuse(
        "the two lenses' sizes differ: " + std::to_string(left.size.width) +
        "x" + std::to_string(left.size.height) + " and " +
        std::to_string(right.size.width) + "x" +
        std::to_string(right.size.height));
  }
  const camera::Mat4d& left_to_world = *cameras[0]->color_to_world;
  const camera::Mat4d& right_to_world = *cameras[1]->color_to_world;
  if (camera::Vec3d(left_to_world[3]) == camera::Vec3d(right_to_world[3])) {
    return refuse("the two cameras are at one place");
  }
  StereoCalibration out;
  out.left = left;
  out.right = right;
  out.left_to_right = camera::rigid_inverse(right_to_world) * left_to_world;
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory(std::string(kWho) +
                                     ": out of host memory");
}

}  // namespace volumetric_kit::sensor
