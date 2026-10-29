// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the arithmetic between librealsense's types and
// this package's -- the camera models, the extrinsic, a frame -- and the
// option checks, kept out of the driver's .cpp so host tests pin them, with
// librealsense's own rsutil.h as the reference, and no camera attached.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <librealsense2/h/rs_sensor.h>
#include <librealsense2/h/rs_types.h>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"

namespace volumetric_kit::sensor::realsense {

// What `RealSenseSensor::open` refuses before it looks for a camera, as
// InvalidArgument: a zero size or rate, a depth range that is not finite with
// 0 < min_depth < max_depth, colour controls with an exposure or a gain of 0,
// and a colour_to_world that is not rigid.
core::Status check_options(const RealSenseSensor::Options& options);

// A stream's camera as it captures, for the GPU pass: its size, intrinsics
// and lens. librealsense's Brown-Conrady and inverse Brown-Conrady
// coefficients are OpenCV's k1, k2, p1, p2, k3 (k4..k6 zero), which project as
// rsutil.h does within a few hundredths of a pixel at the D455's; any model
// with all coefficients zero is a pinhole. With `lens` false (the depth
// stream, rectified on the camera) any coefficient is refused. Refused as
// Unsupported, naming `who` and the stream: another model with coefficients
// (named), and a size, focal length or principal point that is not finite and
// positive.
core::Result<camera::CameraModel> camera_model_from(const rs2_intrinsics& in,
                                                    bool lens,
                                                    const std::string& who);

// The depth camera's frame to the colour camera's from librealsense's
// extrinsic (its rotation column-major, its translation in metres), as
// rs2_transform_point_to_point applies it; refused as Unsupported unless
// rigid.
core::Result<camera::Mat4d> transform_from(const rs2_extrinsics& extrinsics,
                                           const std::string& who);

// Whether a depth and a colour frame stamped `depth_ms` and `color_ms` are of
// one moment: no more than half a frame apart at `fps`. librealsense's syncer
// matches depth with a stale colour frame while the colour camera starts.
bool same_moment(double depth_ms, double color_ms, std::uint32_t fps) noexcept;

// One frameset's pixels and stamps, as plain values.
struct FrameView {
  const void* depth = nullptr;  // Z16
  std::uint32_t depth_width = 0;
  std::uint32_t depth_height = 0;
  std::size_t depth_stride = 0;  // bytes a row
  const void* color = nullptr;   // YUYV
  std::uint32_t color_width = 0;
  std::uint32_t color_height = 0;
  std::size_t color_stride = 0;
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
};

// The frame, pointing into `view`'s pixels, which `pixels` holds: the depth
// as it is, the colour as YUYV on the host (BT.601, limited range, chroma
// sited left: what librealsense converts by), the time in nanoseconds. A view
// whose sizes differ from the cameras', or whose depth rows are not packed,
// contradicts its stream: IoError, naming `who`.
core::Result<RgbdFrame> rgbd_frame(const FrameView& view,
                                   const FrameSetup& setup,
                                   std::shared_ptr<const void> pixels,
                                   const std::string& who);

}  // namespace volumetric_kit::sensor::realsense
