// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "frame_conversion.hpp"

#include <cmath>
#include <string>

namespace volumetric_kit::recon::sensor::orbbec {

Result<ColorCameraParams> color_camera_from(const OBCameraIntrinsic& intrinsic,
                                            const Mat4f& cam_to_world) {
  if (intrinsic.width <= 0 || intrinsic.height <= 0) {
    return Status::invalid_argument(
        "Orbbec colour intrinsics report a " + std::to_string(intrinsic.width) +
        "x" + std::to_string(intrinsic.height) + " image");
  }
  for (const float f : {intrinsic.fx, intrinsic.fy}) {
    if (!std::isfinite(f) || !(f > 0.0f)) {
      return Status::invalid_argument(
          "Orbbec colour intrinsics report a focal length that is not finite "
          "and positive");
    }
  }
  if (!std::isfinite(intrinsic.cx) || !std::isfinite(intrinsic.cy)) {
    return Status::invalid_argument(
        "Orbbec colour intrinsics report a non-finite principal point");
  }
  ColorCameraParams cam{};
  cam.fx = intrinsic.fx;
  cam.fy = intrinsic.fy;
  cam.cx = intrinsic.cx;
  cam.cy = intrinsic.cy;
  cam.width = static_cast<std::uint32_t>(intrinsic.width);
  cam.height = static_cast<std::uint32_t>(intrinsic.height);
  cam.cam_to_world = cam_to_world;
  return cam;
}

bool same_pinhole(const OBCameraIntrinsic& intrinsic,
                  const ColorCameraParams& cam, float tol) noexcept {
  const auto near = [tol](float a, float b) { return std::fabs(a - b) <= tol; };
  return near(intrinsic.fx, cam.fx) && near(intrinsic.fy, cam.fy) &&
         near(intrinsic.cx, cam.cx) && near(intrinsic.cy, cam.cy);
}

void depth_to_metres(const std::uint16_t* src, std::size_t count,
                     float value_scale_mm, float* dst) {
  const float metres_per_unit = value_scale_mm * 0.001f;
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = static_cast<float>(src[i]) * metres_per_unit;
  }
}

void pack_rgb(const std::uint8_t* rgb, std::size_t count, std::uint32_t* dst) {
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint8_t* p = rgb + 3 * i;
    dst[i] = static_cast<std::uint32_t>(p[0]) |
             (static_cast<std::uint32_t>(p[1]) << 8) |
             (static_cast<std::uint32_t>(p[2]) << 16);
  }
}

OrbbecSyncMode sync_mode_from(OBMultiDeviceSyncMode mode) noexcept {
  switch (mode) {
    case OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN:
      return OrbbecSyncMode::FreeRun;
    case OB_MULTI_DEVICE_SYNC_MODE_STANDALONE:
      return OrbbecSyncMode::Standalone;
    case OB_MULTI_DEVICE_SYNC_MODE_PRIMARY:
      return OrbbecSyncMode::Primary;
    case OB_MULTI_DEVICE_SYNC_MODE_SECONDARY:
      return OrbbecSyncMode::Secondary;
    case OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED:
      return OrbbecSyncMode::SecondarySynced;
    case OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING:
      return OrbbecSyncMode::SoftwareTriggering;
    case OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING:
      return OrbbecSyncMode::HardwareTriggering;
    default:
      return OrbbecSyncMode::Other;
  }
}

Status validate(const OrbbecCapture::Options& options) {
  if (options.depth_width == 0 || options.depth_height == 0 ||
      options.color_width == 0 || options.color_height == 0) {
    return Status::invalid_argument(
        "OrbbecCapture: depth and colour sizes must be non-zero");
  }
  if (options.fps == 0) {
    return Status::invalid_argument("OrbbecCapture: fps must be non-zero");
  }
  // NaN fails every comparison, so test for the good range rather than the
  // bad one; a NaN gate would otherwise reject every sample in silence.
  if (!std::isfinite(options.min_depth) || !std::isfinite(options.max_depth) ||
      !(options.min_depth >= 0.0f) ||
      !(options.min_depth < options.max_depth)) {
    return Status::invalid_argument(
        "OrbbecCapture: depth range [" + std::to_string(options.min_depth) +
        ", " + std::to_string(options.max_depth) +
        "] m must be finite, non-negative and non-empty");
  }
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (!std::isfinite(options.cam_to_world[c][r])) {
        return Status::invalid_argument(
            "OrbbecCapture: cam_to_world must be finite");
      }
    }
  }
  return {};
}

}  // namespace volumetric_kit::recon::sensor::orbbec
