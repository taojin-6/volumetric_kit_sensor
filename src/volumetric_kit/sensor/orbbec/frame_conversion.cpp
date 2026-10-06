// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "frame_conversion.hpp"

#include <cmath>
#include <string>

namespace volumetric_kit::recon::sensor::orbbec {

namespace {

// A stream's pinhole camera, checked; `what` names the stream in the errors.
core::Result<ColorCameraParams> pinhole_from(const OBCameraIntrinsic& intrinsic,
                                             const Mat4f& cam_to_world,
                                             const std::string& what) {
  if (intrinsic.width <= 0 || intrinsic.height <= 0) {
    return core::Status::invalid_argument(
        "Orbbec " + what + " intrinsics report a " +
        std::to_string(intrinsic.width) + "x" +
        std::to_string(intrinsic.height) + " image");
  }
  for (const float f : {intrinsic.fx, intrinsic.fy}) {
    if (!std::isfinite(f) || !(f > 0.0f)) {
      return core::Status::invalid_argument(
          "Orbbec " + what +
          " intrinsics report a focal length that is not finite and positive");
    }
  }
  if (!std::isfinite(intrinsic.cx) || !std::isfinite(intrinsic.cy)) {
    return core::Status::invalid_argument(
        "Orbbec " + what + " intrinsics report a non-finite principal point");
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

}  // namespace

core::Result<ColorCameraParams> color_camera_from(
    const OBCameraIntrinsic& intrinsic, const Mat4f& cam_to_world) {
  return pinhole_from(intrinsic, cam_to_world, "colour");
}

core::Result<camera::CameraModel> camera_model_from(
    const OBCameraIntrinsic& intrinsic, const OBCameraDistortion& distortion,
    const std::string& what) {
  VKC_ASSIGN(const ColorCameraParams pinhole,
             pinhole_from(intrinsic, Mat4f(1.0f), what));
  switch (distortion.model) {
    case OB_DISTORTION_NONE:
    case OB_DISTORTION_BROWN_CONRADY:
    case OB_DISTORTION_BROWN_CONRADY_K6:
      break;
    default:
      return core::Status::unsupported(
          "Orbbec " + what + " stream reports lens model " +
          std::to_string(static_cast<int>(distortion.model)) +
          ", which the GPU pass cannot undistort (it takes Brown-Conrady)");
  }
  camera::CameraModel cam;
  cam.size = {pinhole.width, pinhole.height};
  cam.intrinsics = {pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy};
  camera::RationalDistortion& lens = cam.distortion;
  if (distortion.model != OB_DISTORTION_NONE) {
    lens.k1 = distortion.k1;
    lens.k2 = distortion.k2;
    lens.p1 = distortion.p1;
    lens.p2 = distortion.p2;
    lens.k3 = distortion.k3;
  }
  // Only the K6 model has the rational denominator: the plain one is the
  // polynomial k1..k3, so whatever the SDK leaves in k4..k6 there is not a
  // term of it.
  if (distortion.model == OB_DISTORTION_BROWN_CONRADY_K6) {
    lens.k4 = distortion.k4;
    lens.k5 = distortion.k5;
    lens.k6 = distortion.k6;
  }
  if (!camera::check_camera_model(cam).ok()) {
    return core::Status::invalid_argument("Orbbec " + what +
                                          " stream reports a non-finite lens");
  }
  return cam;
}

camera::Mat4d transform_from(const OBExtrinsic& extrinsic) noexcept {
  camera::Mat3d rotation(1.0);  // column-major: rotation[column][row]
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) rotation[c][r] = extrinsic.rot[3 * r + c];
  }
  // A Femto Mega's factory depth-to-colour rotation has its first row 0.6%
  // short of unit length (measured 2026-10-06); its nearest rotation is the
  // tilt the rest of the matrix describes.
  camera::Mat4d m(camera::nearest_rotation(rotation));
  for (int r = 0; r < 3; ++r) m[3][r] = extrinsic.trans[r] / 1000.0;
  return m;
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

namespace {

core::Status validate_streams(const OrbbecStreamOptions& streams,
                              const std::string& who) {
  if (streams.depth_width == 0 || streams.depth_height == 0 ||
      streams.color_width == 0 || streams.color_height == 0) {
    return core::Status::invalid_argument(
        who + ": depth and colour sizes must be non-zero");
  }
  if (streams.fps == 0) {
    return core::Status::invalid_argument(who + ": fps must be non-zero");
  }
  // NaN fails every comparison, so test for the good range rather than the
  // bad one; a NaN gate would otherwise reject every sample in silence.
  if (!std::isfinite(streams.min_depth) || !std::isfinite(streams.max_depth) ||
      !(streams.min_depth >= 0.0f) ||
      !(streams.min_depth < streams.max_depth)) {
    return core::Status::invalid_argument(
        who + ": depth range [" + std::to_string(streams.min_depth) + ", " +
        std::to_string(streams.max_depth) +
        "] m must be finite, non-negative and non-empty");
  }
  return {};
}

core::Status validate_pose(const Mat4f& cam_to_world, const std::string& who) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (!std::isfinite(cam_to_world[c][r])) {
        return core::Status::invalid_argument(who +
                                              ": cam_to_world must be finite");
      }
    }
  }
  return {};
}

}  // namespace

OBMultiDeviceSyncMode sdk_sync_mode(OrbbecSyncMode mode) noexcept {
  switch (mode) {
    case OrbbecSyncMode::FreeRun:
      return OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN;
    case OrbbecSyncMode::Primary:
      return OB_MULTI_DEVICE_SYNC_MODE_PRIMARY;
    case OrbbecSyncMode::Secondary:
      return OB_MULTI_DEVICE_SYNC_MODE_SECONDARY;
    case OrbbecSyncMode::SecondarySynced:
      return OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED;
    case OrbbecSyncMode::SoftwareTriggering:
      return OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING;
    case OrbbecSyncMode::HardwareTriggering:
      return OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING;
    case OrbbecSyncMode::Standalone:
    case OrbbecSyncMode::Other:
      break;
  }
  return OB_MULTI_DEVICE_SYNC_MODE_STANDALONE;
}

OrbbecSyncSettings sync_settings_from(const OBMultiDeviceSyncConfig& sdk) {
  OrbbecSyncSettings s;
  s.mode = sync_mode_from(sdk.syncMode);
  s.depth_delay_us = sdk.depthDelayUs;
  s.color_delay_us = sdk.colorDelayUs;
  s.trigger_to_image_delay_us = sdk.trigger2ImageDelayUs;
  s.trigger_out_enable = sdk.triggerOutEnable;
  s.trigger_out_delay_us = sdk.triggerOutDelayUs;
  s.frames_per_trigger = sdk.framesPerTrigger;
  return s;
}

OBMultiDeviceSyncConfig sdk_sync_config(const OrbbecSyncSettings& s) {
  OBMultiDeviceSyncConfig sdk{};
  sdk.syncMode = sdk_sync_mode(s.mode);
  sdk.depthDelayUs = s.depth_delay_us;
  sdk.colorDelayUs = s.color_delay_us;
  sdk.trigger2ImageDelayUs = s.trigger_to_image_delay_us;
  sdk.triggerOutEnable = s.trigger_out_enable;
  sdk.triggerOutDelayUs = s.trigger_out_delay_us;
  sdk.framesPerTrigger = s.frames_per_trigger;
  return sdk;
}

std::vector<std::string> sync_differences(const OrbbecSyncSettings& wanted,
                                          const OrbbecSyncSettings& actual) {
  const auto role = [](OrbbecSyncMode m) -> std::string {
    return m == OrbbecSyncMode::SecondarySynced
               ? to_string(OrbbecSyncMode::Secondary)
               : to_string(m);
  };
  std::vector<std::string> out;
  const auto differ = [&out](const char* field, const std::string& have,
                             const std::string& want) {
    if (have != want) {
      out.push_back(std::string(field) + " is " + have + ", configured " +
                    want);
    }
  };
  differ("syncMode", role(actual.mode), role(wanted.mode));
  differ("depthDelayUs", std::to_string(actual.depth_delay_us),
         std::to_string(wanted.depth_delay_us));
  differ("colorDelayUs", std::to_string(actual.color_delay_us),
         std::to_string(wanted.color_delay_us));
  differ("triggerOutEnable", actual.trigger_out_enable ? "true" : "false",
         wanted.trigger_out_enable ? "true" : "false");
  differ("triggerOutDelayUs", std::to_string(actual.trigger_out_delay_us),
         std::to_string(wanted.trigger_out_delay_us));
  if (wanted.mode == OrbbecSyncMode::SoftwareTriggering ||
      wanted.mode == OrbbecSyncMode::HardwareTriggering) {
    differ("framesPerTrigger", std::to_string(actual.frames_per_trigger),
           std::to_string(wanted.frames_per_trigger));
  }
  return out;
}

core::Status validate(const OrbbecCapture::Options& options) {
  VKC_TRY(validate_streams(options, "OrbbecCapture"));
  return validate_pose(options.cam_to_world, "OrbbecCapture");
}

core::Status validate(const OrbbecRig::Options& options) {
  VKC_TRY(validate_streams(options, "OrbbecRig"));
  if (options.sync.devices.size() < 2) {
    return core::Status::invalid_argument(
        "OrbbecRig: a rig needs at least two cameras; OrbbecCapture opens "
        "one");
  }
  for (std::size_t i = 0; i < options.sync.devices.size(); ++i) {
    const std::string& serial = options.sync.devices[i].serial;
    if (serial.empty()) {
      return core::Status::invalid_argument(
          "OrbbecRig: camera " + std::to_string(i) + " has no serial");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (options.sync.devices[j].serial == serial) {
        return core::Status::invalid_argument("OrbbecRig: camera " + serial +
                                              " is listed twice");
      }
    }
  }
  if (!options.calibration.sensors.empty()) {
    const core::Status calibration =
        camera::validate_array_calibration(options.calibration);
    if (!calibration.ok()) {
      return core::Status::invalid_argument("OrbbecRig: " +
                                            calibration.message());
    }
    for (const OrbbecSyncDevice& device : options.sync.devices) {
      const camera::SensorCalibration* sensor =
          camera::find_sensor(options.calibration, device.serial);
      if (sensor == nullptr) {
        return core::Status::invalid_argument(
            "OrbbecRig: the calibration does not pose camera " + device.serial);
      }
    }
  }
  // At half a frame period or more, a secondary's frame can sit within the
  // tolerance of two neighbouring triggers' primary frames.
  const std::uint64_t half_period_us = 500000u / options.fps;
  if (options.sync_tolerance_us == 0 ||
      options.sync_tolerance_us >= half_period_us) {
    return core::Status::invalid_argument(
        "OrbbecRig: sync_tolerance_us is " +
        std::to_string(options.sync_tolerance_us) +
        "; it must be non-zero and under half a frame period (" +
        std::to_string(half_period_us) + " us at " +
        std::to_string(options.fps) + " fps)");
  }
  return {};
}

}  // namespace volumetric_kit::recon::sensor::orbbec
