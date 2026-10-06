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
/// Like the capture contract it reaches no Vulkan, so a driver produces it
/// without compiling against a GPU API: a picture a decoder left on the
/// device is named by `core`'s `Buffer` or `Image`, declared here.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/chroma_location.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief How a picture's chroma is laid out.
enum class YuvLayout : std::uint8_t {
  I420,  ///< Cb and Cr each in a plane of its own.
  Nv12,  ///< Cb and Cr interleaved in one plane, Cb first.
};

/// @brief @ref YuvImage::queue_family for device planes that need no
///        ownership transfer: written on the queue family that prepares them,
///        or held in a CONCURRENT buffer. Vulkan's `VK_QUEUE_FAMILY_IGNORED`.
inline constexpr std::uint32_t kQueueFamilyIgnored = ~std::uint32_t{0};
/// @brief @ref YuvImage::queue_family for device planes an API outside Vulkan
///        wrote, such as CUDA through imported memory. Vulkan's
///        `VK_QUEUE_FAMILY_EXTERNAL`.
inline constexpr std::uint32_t kQueueFamilyExternal = ~std::uint32_t{0} - 1;

/// @brief An 8-bit Y'CbCr 4:2:0 picture, on the host or already on the
///        device, and the matrix and range it was coded with.
///
/// Chroma's position is given by @ref YuvImage::chroma_location, defaulting to
/// H.265's left alignment. JPEG uses centred chroma. The planes are Y,
/// then Cb and Cr at half size (rounded up) for I420, or Y then CbCr for
/// NV12, whose chroma rows hold both samples of each pair. They are host
/// memory in @ref plane or, where a decoder left its picture on the GPU,
/// ranges of @ref device that do not overlap or the @ref image planes; one of
/// the three.
///
/// @code
/// YuvImage image;  // a decoder's NV12 picture, left on the device
/// image.layout = YuvLayout::Nv12;
/// image.device = picture_buffer;  // written, and the writer waited on
/// image.offset[0] = 0;
/// image.offset[1] = chroma_at;
/// image.stride[0] = width;
/// image.stride[1] = (width + 1) / 2 * 2;
/// image.queue_family = kQueueFamilyExternal;  // CUDA wrote it
/// image.width = width;
/// image.height = height;
/// @endcode
struct YuvImage {
  YuvLayout layout = YuvLayout::I420;  ///< How the chroma is laid out.
  /// The planes on the host; `plane[2]` is unused for NV12, and so null.
  /// Held by the frame's `RgbdFrame::pixels`.
  const std::uint8_t* plane[3] = {};
  std::size_t stride[3] = {};  ///< Bytes per row of each plane.
  /// Or the planes on the device: a storage buffer on the device the frame
  /// is prepared on, holding plane `p` at byte @ref offset `[p]`.
  ///
  /// Its writer must have **finished** before the frame is prepared -- a
  /// fence waited on, or the CUDA stream synchronized -- since the pass
  /// submits on its own queue and waits on no semaphore; an unfinished write
  /// reads as a torn picture. The frame holds the buffer, so the decoder
  /// cannot reuse it while the frame is prepared, and a prepare whose wait
  /// fails holds it for good, as the device may still read it.
  std::shared_ptr<const core::Buffer> device;
  std::uint64_t offset[3] = {};  ///< Each plane's byte offset in @ref device.
  /// The queue family that wrote @ref device, which the pass takes it over
  /// from before reading: another family of the device, whose writer
  /// released the whole buffer to the pass's family, or
  /// @ref kQueueFamilyExternal. @ref kQueueFamilyIgnored when the pass's own
  /// family wrote it; another Vulkan family's is ignored too for a
  /// CONCURRENT buffer, which needs no transfer.
  std::uint32_t queue_family = kQueueFamilyIgnored;
  /// Or NV12's planes as images on the device, as VideoToolbox's picture
  /// arrives: `image[0]` the luma (`R8_UNORM`), `image[1]` the chroma
  /// (`R8G8_UNORM`, Cb first), each at least the picture's size, chroma
  /// halved and rounded up, and read from its corner. The pass copies them
  /// into its input in its batch, so their writer must have finished, as
  /// for @ref device, and the frame holds them as it does @ref device.
  std::shared_ptr<const core::Image> image[2];
  std::uint32_t width = 0;   ///< Luma width (pixels).
  std::uint32_t height = 0;  ///< Luma height (pixels).
  /// The matrix's red and blue weights: BT.601 is 0.299 and 0.114, BT.709
  /// 0.2126 and 0.0722.
  float kr = 0.299f;
  float kb = 0.114f;       ///< See @ref kr.
  bool full_range = true;  ///< Y in 0..255 rather than 16..235.
  ChromaLocation chroma_location = ChromaLocation::Left;  ///< Sample positions.
};

/// @brief One RGB-D frame as the sensor captured it.
///
/// It holds what it points to, so a consumer may keep several -- a sensor
/// array groups each sensor's frames by trigger before it prepares any: the
/// host pixels through @ref pixels, a device picture through
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
/// frame.color = decoded_picture;             // Y'CbCr, host or device
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

  /// The colour picture; with none of `color.plane[0]`, `color.device` and
  /// `color.image` set, the frame has none. Its size is @ref color_camera's.
  /// Host planes are held by @ref pixels.
  YuvImage color{};
  camera::CameraModel color_camera;  ///< The colour camera, lens included.
  /// What the R'G'B' the matrix gives is encoded as; the pass converts only
  /// from an encoding @ref is_canonical accepts.
  ColorEncoding color_encoding{};

  /// The colour camera's frame to the world's, in this repo's camera axes
  /// (+Z forward, +Y down): where the sensor sits, from its calibration or,
  /// for a tracked sensor, its own estimate for this frame.
  camera::Mat4d color_to_world = camera::Mat4d(1.0);
  /// The depth camera's frame to the colour camera's: the sensor's own
  /// extrinsic, so the depth camera sits at `color_to_world * depth_to_color`.
  camera::Mat4d depth_to_color = camera::Mat4d(1.0);

  /// Device timestamp in nanoseconds; monotonic within one capture session.
  /// Zero when the device reports none.
  std::uint64_t timestamp_ns = 0;
  /// The sensor's own count of its frames: a gap is a lost frame, which a
  /// timestamp cannot tell from a late one.
  std::uint64_t sequence = 0;

  /// What @ref depth and the host colour planes point into, held for as long
  /// as the frame is; null when they need no owner.
  std::shared_ptr<const void> pixels;

  /// @return `true` if this frame carries colour.
  bool has_color() const noexcept {
    return color.plane[0] != nullptr || color.device != nullptr ||
           color.image[0] != nullptr || color.image[1] != nullptr;
  }
};

}  // namespace volumetric_kit::recon::sensor
