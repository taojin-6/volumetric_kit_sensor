// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the arithmetic between librealsense's types and
// this package's -- the camera models, the extrinsics, the depth's noise, a
// frame and a stereo frame -- and the option checks, kept out of the drivers'
// .cpp so host tests pin them, with librealsense's own rsutil.h as the
// reference, and no camera attached.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <librealsense2/h/rs_sensor.h>
#include <librealsense2/h/rs_types.h>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/frame/depth_noise.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"
#include "volumetric_kit/sensor/realsense/realsense_stereo.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor::realsense {

// What `RealSenseSensor::open` refuses before it looks for a camera, as
// InvalidArgument: a zero size or rate, a depth range that is not finite with
// 0 < min_depth < max_depth, colour controls with an exposure or a gain of 0,
// and a colour_to_world that is not rigid.
core::Status check_options(const RealSenseSensor::Options& options);

// What `RealSenseStereo::open` refuses before it looks for a camera, as
// InvalidArgument: a zero size or rate, colour controls with an exposure or a
// gain of 0, and a colour_to_world that is not rigid.
core::Status check_options(const RealSenseStereo::Options& options);

// A stream's camera as it captures, for the GPU pass: its size, intrinsics
// and lens. librealsense's Brown-Conrady and inverse Brown-Conrady
// coefficients are OpenCV's k1, k2, p1, p2, k3 (k4..k6 zero), which project as
// rsutil.h does within a few hundredths of a pixel at the D455's; any model
// with all coefficients zero is a pinhole. With `lens` false (what the camera
// rectifies itself: its depth, its infrared cameras) any coefficient is
// refused. Refused as Unsupported, naming `who` and the `stream` ("depth",
// "colour", "left infrared"): another model with coefficients (named), and a
// size, focal length or principal point that is not finite and positive.
core::Result<camera::CameraModel> camera_model_from(const rs2_intrinsics& in,
                                                    const char* stream,
                                                    bool lens,
                                                    const std::string& who);

// One stream's camera frame to another's from librealsense's extrinsic (its
// rotation column-major, its translation in metres), as
// rs2_transform_point_to_point applies it; refused as Unsupported unless
// rigid, naming `what` ("depth-to-colour").
core::Result<camera::Mat4d> transform_from(const rs2_extrinsics& extrinsics,
                                           const char* what,
                                           const std::string& who);

// The two infrared cameras' calibration as the stereo tier takes it: each a
// pinhole (the camera rectifies them: a lens on either is refused), the left
// one's point in the right camera and in the colour camera, and the colour
// camera with its lens. Refused as camera_model_from and transform_from
// refuse.
core::Result<StereoInfo> stereo_info(const rs2_intrinsics& left,
                                     const rs2_intrinsics& right,
                                     const rs2_extrinsics& left_to_right,
                                     const rs2_intrinsics& color,
                                     const rs2_extrinsics& left_to_color,
                                     const std::string& who);

// The disparity's frame-to-frame sigma of the camera's own depth, in pixels:
// 0.13, measured 2026-10-29 on a D455 at 1280x720, its projector on and off,
// pooled over 0.33-3 m (0.10-0.17 px a bin), on the pixels with a return in
// every frame. So it is low where it counts most: past 3 m the bins read
// 0.18-0.27 px, and pixels that come and go are noisier. TODO(sensor): its
// other modes, and the other D400s the driver opens, are not measured and
// state the same in their pixels.
inline constexpr double kChipDisparitySigma = 0.13;

// The noise of the camera's own depth, which every frame states: rectified
// stereo's (DepthNoise::stereo), kChipDisparitySigma over the depth camera's fx
// and the baseline, the length of `depth_to_right`'s translation (the depth
// camera is the left infrared camera, and `depth_to_right` its extrinsic to
// the right one). Unknown (all zero) where either is not finite and above 0.
DepthNoise chip_depth_noise(const camera::CameraModel& depth_camera,
                            const rs2_extrinsics& depth_to_right) noexcept;

// Whether two frames stamped `a_ms` and `b_ms` are of one moment: no more
// than half a frame apart at `fps`. As it starts, the D455 restarts its
// depth for about 1.2 s, and librealsense's pipeline matches its last depth
// frame with newer colour meanwhile.
bool same_moment(double a_ms, double b_ms, std::uint32_t fps) noexcept;

// Whether a frameset of `count` frames stamped `stamps_ms` is one usable
// moment of a camera opened at `opened` modes: a frame of each, and each
// within half a frame of the first. A frameset that is not is counted
// failed.
bool together(const double* stamps_ms, std::size_t count, std::size_t opened,
              std::uint32_t fps) noexcept;

// One picture of a frameset, as plain values.
struct PictureView {
  const void* data = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t stride = 0;  // bytes a row
};

// One frameset's depth (Z16) and colour (YUYV) and its stamps.
struct FrameView {
  PictureView depth;
  PictureView color;
  double timestamp_ms = 0.0;  // librealsense's global time
  std::uint64_t sequence = 0;
  std::uint64_t arrival_ns = 0;
};

// What every frame of a sensor carries, from open.
struct FrameSetup {
  camera::CameraModel depth_camera;
  camera::CameraModel color_camera;
  camera::Mat4d depth_to_color = camera::Mat4d(1.0);
  camera::Mat4d color_to_world = camera::Mat4d(1.0);
  float metres_per_unit = 0.001f;
  float min_depth = 0.0f;
  float max_depth = 0.0f;
  DepthNoise depth_noise{};  // chip_depth_noise's
};

// The frame, pointing into `view`'s pixels, which `pixels` holds: the depth
// as it is, with the setup's range and noise, the colour as YUYV on the host
// (BT.601, limited range, chroma sited left: what librealsense converts by),
// the time in nanoseconds. A view whose sizes differ from the cameras', or
// whose depth rows are not packed, contradicts its stream: IoError, naming
// `who`.
core::Result<RgbdFrame> rgbd_frame(const FrameView& view,
                                   const FrameSetup& setup,
                                   std::shared_ptr<const void> pixels,
                                   const std::string& who);

// One frameset's two infrared pictures (Y8), colour (YUYV) and stamps, the
// left infrared camera's.
struct StereoView {
  PictureView left;
  PictureView right;
  PictureView color;
  double timestamp_ms = 0.0;  // librealsense's global time
  std::uint64_t sequence = 0;
  std::uint64_t arrival_ns = 0;
};

// The stereo frame, pointing into `view`'s pixels, which `pixels` holds: each
// infrared picture as luma on the host (full range: the camera's 8-bit
// infrared as it is), the colour as rgbd_frame's, the time in nanoseconds,
// placed by `color_to_world`. A view whose sizes differ from `info`'s
// cameras', or whose rows are shorter than its width, contradicts its
// streams: IoError, naming `who`.
core::Result<StereoFrame> stereo_frame(const StereoView& view,
                                       const StereoInfo& info,
                                       const camera::Mat4d& color_to_world,
                                       std::shared_ptr<const void> pixels,
                                       const std::string& who);

}  // namespace volumetric_kit::sensor::realsense
