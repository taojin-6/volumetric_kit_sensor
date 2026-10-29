// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame/device_frame.hpp
/// @brief The frame contract: one posed, undistorted RGB-D frame on the
///        device, the shape every consumer reads.
///
/// A camera stack is anything that produces these: this package's frame
/// preparation from a raw RGB-D sensor's frames, or a pipeline that makes
/// depth on the device itself. Every producer keeps the same contract:
///
/// - the buffers are device-local storage buffers, allocated on the
///   consumer's `VkDevice`;
/// - their writes are complete when the frame is handed over (host-ordered);
/// - they are owned by the queue family the producer was created for, or
///   shared CONCURRENT with the families it was given;
/// - a frame holds its buffers: the producer reuses one only once no frame
///   holds it, and the `Allocator` outlives every frame.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/frame/camera_params.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"

namespace volumetric_kit::sensor {

/// @brief One frame on the device, pinhole and R'G'B'.
///
/// Depth and colour keep their own cameras, and neither is registered to the
/// other: @ref depth is read through @ref depth_camera, and @ref color through
/// @ref color_camera, its high byte the pixel's coverage.
///
/// The frame holds its buffers, which are `device_storage_buffer`s, so a
/// `CommandBatch` can copy or read them back. One kept past the producer's
/// next frame keeps its contents, and the producer writes that frame to new
/// buffers instead; drop a frame once it is fused and the producer reuses
/// them. The `Allocator` must outlive the frame, as it must the producer.
struct DeviceFrame {
  /// Row-major depth in metres, `depth_camera.width * height` floats, less
  /// any depth offset the producer corrects for; 0 where the sensor had no
  /// return, the offset leaves none, or the lens maps outside its image.
  std::shared_ptr<const core::Buffer> depth;
  /// Row-major colour, `color_camera.width * height` words: R, G and B in the
  /// low three bytes, and the pixel's coverage in the high one -- 0xFF where
  /// the lens maps inside the captured picture, 0 (and black) where it maps
  /// outside, which a consumer skips rather than reads as black. Null when
  /// the frame has no colour.
  std::shared_ptr<const core::Buffer> color;
  DepthCameraParams depth_camera{};  ///< The undistorted depth camera.
  ColorCameraParams color_camera{};  ///< The undistorted colour camera.
  ColorEncoding color_encoding{};    ///< What @ref color is encoded as.
  std::uint64_t timestamp_ns = 0;    ///< The raw frame's.

  /// @return `true` if this frame carries colour.
  bool has_color() const noexcept { return color != nullptr; }
};

/// @brief A set of frames from one trigger, each on the device, in the
///        producer's sensor order.
struct DeviceFrameSet {
  std::uint64_t timestamp_ns = 0;  ///< The trigger's time (ns).
  std::uint64_t sequence = 0;      ///< The trigger's sequence number.
  /// One per sensor, in the producer's order; empty where the set had no
  /// frame.
  std::vector<std::optional<DeviceFrame>> frames;
};

}  // namespace volumetric_kit::sensor
