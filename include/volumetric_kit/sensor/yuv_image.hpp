// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/yuv_image.hpp
/// @brief A Y'CbCr picture on the device and how it is coded: what a decoder
///        or a platform's capture hands out, and what a frame's colour is.
///
/// The one description of a device picture: a decoder's picture carries it
/// (`DecodedPicture::yuv`), and a frame takes it as its colour
/// (`RgbdFrame::color`) as it is. It names no camera and reaches no Vulkan,
/// so the decoders and the Apple importer describe their pictures with it
/// without linking the camera tier.

#include <cstddef>
#include <cstdint>
#include <memory>

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

/// @brief An 8-bit Y'CbCr 4:2:0 picture on the device, as a hardware decoder
///        or a platform's capture left it, and the matrix and range it was
///        coded with.
///
/// Chroma's position is given by @ref YuvImage::chroma_location, defaulting to
/// H.265's left alignment. JPEG uses centred chroma. The planes are Y,
/// then Cb and Cr at half size (rounded up) for I420, or Y then CbCr for
/// NV12, whose chroma rows hold both samples of each pair. They are ranges of
/// @ref device that do not overlap, or the @ref image planes; one or the
/// other.
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
  /// The planes in a storage buffer on the device the frame is prepared on,
  /// plane `p` at byte @ref offset `[p]`, @ref stride `[p]` bytes a row.
  ///
  /// Its writer must have **finished** before the frame is prepared -- a
  /// fence waited on, or the CUDA stream synchronized -- since the pass
  /// submits on its own queue and waits on no semaphore; an unfinished write
  /// reads as a torn picture. The frame holds the buffer, so the decoder
  /// cannot reuse it while the frame is prepared, and a prepare whose wait
  /// fails holds it for good, as the device may still read it.
  std::shared_ptr<const core::Buffer> device;
  std::uint64_t offset[3] = {};  ///< Each plane's byte offset in @ref device.
  std::size_t stride[3] = {};  ///< Bytes per row of each plane in @ref device.
  /// The queue family that wrote @ref device, which the pass takes it over
  /// from before reading: another family of the device, whose writer
  /// released the whole buffer to the pass's family, or
  /// @ref kQueueFamilyExternal. @ref kQueueFamilyIgnored when the pass's own
  /// family wrote it; another Vulkan family's is ignored too for a
  /// CONCURRENT buffer, which needs no transfer.
  std::uint32_t queue_family = kQueueFamilyIgnored;
  /// Or NV12's planes as images on the device, as Apple's pictures arrive
  /// (`PixelBufferImporter`): `image[0]` the luma (`R8_UNORM`), `image[1]`
  /// the chroma (`R8G8_UNORM`, Cb first), each at least the picture's size,
  /// chroma halved and rounded up, and read from its corner. The pass copies
  /// them into its input in its batch, so their writer must have finished,
  /// as for @ref device, and the frame holds them as it does @ref device.
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

}  // namespace volumetric_kit::recon::sensor
