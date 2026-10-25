// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/rgbd_frame.hpp
/// @brief A frame as an RGB-D sensor captured it, before undistortion and
///        colour conversion: what every driver hands out, and what
///        `sensor/utils`'s GPU pass prepares.
///
/// Each camera keeps its own model, lens included (`camera::CameraModel`),
/// the depth in the sensor's units, and the colour as the decoder left it.
/// Nothing registers depth to colour: the depth camera sits at
/// @ref RgbdFrame::depth_to_color from the colour camera, and `tsdf` fuses the
/// two as they are.
///
/// It reaches no Vulkan, so a driver produces it without compiling against a
/// GPU API: a picture a decoder left on the device is named by `core`'s
/// `Buffer` or `Image`, declared here (`sensor/yuv_image.hpp`).

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief The colour as the camera coded it, before any decoding: bytes a
///        consumer can keep as they came.
///
/// Only a frame that decodes on its own carries one, an MJPEG camera's JPEG;
/// an H.265 frame needs the frames before it, so none is handed out for it.
struct ColorBitstream {
  /// @brief What @ref data holds.
  enum class Codec : std::uint8_t {
    None,  ///< Nothing: the frame carries no bitstream.
    Jpeg,  ///< One JPEG, as the camera sent it.
  };
  Codec codec = Codec::None;  ///< What @ref data holds.
  /// The bytes, held by `RgbdFrame::pixels`; null with `Codec::None`.
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;  ///< The count of @ref data; 0 with `Codec::None`.
};

/// @brief One RGB-D frame as the sensor captured it.
///
/// It holds what it points to, so a consumer may keep several -- a sensor
/// array groups each sensor's frames by trigger before it prepares any: the
/// depth and any host colour through @ref pixels, the colour picture through
/// `YuvImage::device` or `YuvImage::image`. A driver's buffers go back to it
/// once every copy of the frame is gone, so a consumer that keeps frames
/// keeps the driver's buffers from it.
///
/// @code
/// RgbdFrame frame;  // as a driver fills one
/// frame.depth = depth_pixels;               // the sensor's u16 units
/// frame.metres_per_unit = 0.001f;
/// frame.depth_camera = factory_depth_model;  // lens included
/// frame.min_depth = 0.25f;
/// frame.max_depth = 5.0f;
/// frame.color = decoded_picture.yuv;         // Y'CbCr, on the device
/// frame.color_camera = factory_color_model;
/// frame.depth_to_color = factory_extrinsic;  // depth camera -> colour camera
/// frame.color_to_world = pose;
/// frame.sequence = sdk_frame_index;
/// frame.pixels = sdk_frameset;  // keeps depth_pixels alive
/// @endcode
struct RgbdFrame {
  /// Row-major depth in the sensor's units, `depth_camera.size.width *
  /// height` samples; 0 is "no return". Held by @ref pixels.
  const std::uint16_t* depth = nullptr;
  float metres_per_unit = 0.001f;    ///< What one depth unit is, in metres.
  camera::CameraModel depth_camera;  ///< The depth camera, lens included.
  /// Nearer samples are dropped (metres). Set it: the pass needs
  /// `0 < min_depth < max_depth`, since 0 is "no return", and refuses these
  /// zeros rather than fuse nothing.
  float min_depth = 0.0f;
  float max_depth = 0.0f;  ///< Farther samples are dropped (metres).

  /// The colour picture; with none of `color.device`, `color.image` or
  /// @ref color_packed set, the frame has none. Its size is
  /// @ref color_camera's.
  YuvImage color{};
  /// Or the colour on the host, as a dataset decodes it:
  /// `color_camera.size.width * height` row-major words, R | G << 8 | B << 16,
  /// the high byte ignored (`io::load_color_packed`'s layout). Set this or
  /// @ref color, not both. Held by @ref pixels.
  const std::uint32_t* color_packed = nullptr;
  camera::CameraModel color_camera;  ///< The colour camera, lens included.
  /// What the R'G'B' -- the matrix's, or @ref color_packed's -- is encoded
  /// as; the pass converts only from an encoding @ref is_canonical accepts.
  ColorEncoding color_encoding{};
  /// The colour as the camera coded it, beside the decoded @ref color: an
  /// MJPEG camera's JPEG, byte for byte. Held by @ref pixels. Empty for a
  /// frame decoded from H.265 or taken from a dataset.
  ColorBitstream color_bitstream{};

  /// The colour camera's frame to the world's, in this repo's camera axes
  /// (+Z forward, +Y down): where the sensor sits, from its calibration or,
  /// for a tracked sensor, its own estimate for this frame.
  camera::Mat4d color_to_world = camera::Mat4d(1.0);
  /// The depth camera's frame to the colour camera's: the sensor's own
  /// extrinsic, so the depth camera sits at `color_to_world * depth_to_color`.
  camera::Mat4d depth_to_color = camera::Mat4d(1.0);

  /// Capture time in nanoseconds, on the sensor's clock (`SensorInfo::clock`):
  /// its own, monotonic within one capture session, or the host's
  /// `std::chrono::system_clock`, since its epoch. Zero when the sensor
  /// reports none.
  std::uint64_t timestamp_ns = 0;
  /// The sensor's own count of its frames: a gap is a lost frame, which a
  /// timestamp cannot tell from a late one. Like @ref timestamp_ns, it
  /// counts within one capture session; it may begin again at a restart.
  std::uint64_t sequence = 0;
  /// When the host received the frame, on `std::chrono::system_clock`, in
  /// nanoseconds since its epoch: what a sender's latency counts from. 0 when
  /// the driver does not say, as a recording's frames, which arrive when
  /// polled.
  std::uint64_t arrival_ns = 0;

  /// What @ref depth, @ref color_packed and @ref color_bitstream point into,
  /// held for as long as the frame is; null when they need no owner.
  std::shared_ptr<const void> pixels;

  /// @return `true` if this frame carries colour.
  bool has_color() const noexcept {
    return color_packed != nullptr || color.device != nullptr ||
           color.image[0] != nullptr || color.image[1] != nullptr;
  }
};

}  // namespace volumetric_kit::recon::sensor
