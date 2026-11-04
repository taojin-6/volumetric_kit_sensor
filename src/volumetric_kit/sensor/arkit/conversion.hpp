// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): ArkitSensor's host arithmetic on one ARFrame's
// parts, with no session and no device -- the depth copy, the picture's
// checks and colour, the cameras and the pose, the clock, the missed
// periods -- so the conversion test pins each on the host.

#include <CoreVideo/CoreVideo.h>

#include <array>
#include <cstdint>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/chroma_location.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"

namespace volumetric_kit::sensor::arkit {

// The depth map out of ARKit's buffers in one pass: `depth` (DepthFloat32,
// metres, its rows padded as Core Video pads them) to `depth_out`'s packed
// millimetres by arkit_depth_mm, and `confidence` (OneComponent8) to
// `confidence_out`'s packed levels as they are, both `width` x `height`;
// `confidence_out` null copies no levels. InvalidArgument naming the
// refusal: depth not DepthFloat32 or not `width` x `height`, no confidence
// map, one not OneComponent8 or of another size, or a buffer that does not
// lock.
core::Status copy_depth(CVPixelBufferRef depth, CVPixelBufferRef confidence,
                        std::uint32_t width, std::uint32_t height,
                        std::uint16_t* depth_out, std::uint8_t* confidence_out);

// A picture the sensor takes: 8-bit NV12 ('420f' or '420v') on an IOSurface,
// `width` x `height`. InvalidArgument naming what it is not.
core::Status check_picture(CVPixelBufferRef picture, std::uint32_t width,
                           std::uint32_t height);

// How a picture's samples are coded, from its own tags (color_codes_of, then
// video/frame_color.hpp): an untagged matrix is BT.601, ARKit's.
struct PictureColor {
  float kr = 0.0f;
  float kb = 0.0f;
  bool full_range = true;
  ChromaLocation chroma_location = ChromaLocation::Left;
  ColorEncoding encoding{};
};

// `picture`'s colour; InvalidArgument for a tag Core Video gives no code for,
// or a transfer and primaries ColorEncoding cannot name (HLG among them).
core::Result<PictureColor> picture_color(CVPixelBufferRef picture);

// The picture's pinhole from ARCamera.intrinsics (column-major: fx at [0],
// fy at [4], cx at [6], cy at [7]) at its size, with no lens.
// InvalidArgument where camera::check_camera_model refuses it.
core::Result<camera::CameraModel> color_camera_of(
    const std::array<float, 9>& intrinsics, std::uint32_t width,
    std::uint32_t height);

// The depth camera registered to `color` at `width` x `height`, through
// depth_from_registered_color.
core::Result<camera::CameraModel> depth_camera_of(
    const camera::CameraModel& color, std::uint32_t width, std::uint32_t height,
    float min_depth, float max_depth);

// ARCamera.transform (column-major, ARKit's axes) as the colour camera's
// pose in this package's: cv_from_gl_camera, in double, made exactly rigid
// (camera::exact_rigid: the bottom row [0, 0, 0, 1], the rotation the
// nearest one). InvalidArgument where exact_rigid refuses it.
core::Result<camera::Mat4d> color_to_world_of(
    const std::array<float, 16>& transform);

// ARKit's capture times (the uptime clock, seconds) on the host's
// system_clock (ns): plus an offset measured at each start, and again when
// an interruption ends, never smaller than before, so mapped times increase
// as the uptime clock does.
class ClockMap {
 public:
  // At a start: the offset as measured now, whatever it was.
  void start(std::int64_t system_ns, std::int64_t uptime_ns) noexcept;
  // When an interruption ends: the offset measured again, kept where it was
  // if smaller (the system clock set back), taken where larger (the device
  // slept, which the uptime clock does not count).
  void remeasure(std::int64_t system_ns, std::int64_t uptime_ns) noexcept;
  // `uptime_s` on the system clock; 0 below its epoch.
  std::uint64_t to_system_ns(double uptime_s) const noexcept;
  std::int64_t offset_ns() const noexcept { return offset_ns_; }

 private:
  std::int64_t offset_ns_ = 0;
};

// The capture periods at `fps` skipped between captures at `before_s` and
// `now_s`: the gap in periods, rounded, less one; 0 for a gap of one period
// or less.
std::uint64_t missed_periods(double before_s, double now_s,
                             std::uint32_t fps) noexcept;

// The clocks now: the system clock's and the uptime clock's (ARFrame's), ns.
std::int64_t system_now_ns() noexcept;
std::int64_t uptime_now_ns() noexcept;

// The device's model id: "iPad16,3" on iOS, "Mac16,5" on a Mac.
std::string device_model();

}  // namespace volumetric_kit::sensor::arkit
