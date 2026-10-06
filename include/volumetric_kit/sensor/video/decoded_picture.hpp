// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/decoded_picture.hpp
/// @brief A decoded picture, on the device and held by the picture.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/chroma_location.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief How a decoded picture's pixels are laid out.
enum class VideoPixelLayout {
  /// Three 8-bit planes: Y, then U and V at half size, as nvJPEG decodes.
  Yuv420,
  /// Two 8-bit planes: Y, then U and V interleaved at half size, U first, as
  /// NVDEC and VideoToolbox decode.
  Nv12,
};

/// @brief The YCbCr-to-RGB matrix of a coded picture.
enum class VideoColorMatrix {
  Bt601,      ///< BT.601 (SMPTE 170M, BT.470 B/G): SD.
  Bt709,      ///< BT.709: HD.
  Bt2020,     ///< BT.2020 non-constant luminance: UHD.
  Smpte240m,  ///< SMPTE 240M, the interim HD matrix.
  Fcc,        ///< FCC 73.682, the 1953 NTSC matrix.
};

/// @brief A matrix's red and blue luma weights; green's is the rest.
struct YcbcrWeights {
  float kr = 0.0f;  ///< Red's weight in luma.
  float kb = 0.0f;  ///< Blue's weight in luma.
};

/// @return @p matrix's weights, as the standards give them: what the GPU pass
///         converts by.
constexpr YcbcrWeights ycbcr_weights(VideoColorMatrix matrix) noexcept {
  switch (matrix) {
    case VideoColorMatrix::Bt601:
      return {0.299f, 0.114f};
    case VideoColorMatrix::Bt709:
      return {0.2126f, 0.0722f};
    case VideoColorMatrix::Bt2020:
      return {0.2627f, 0.0593f};
    case VideoColorMatrix::Smpte240m:
      return {0.212f, 0.087f};
    case VideoColorMatrix::Fcc:
      return {0.30f, 0.11f};
  }
  return {0.2126f, 0.0722f};
}

/// @brief A stream's YCbCr matrix and range.
struct VideoColorDescription {
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  bool full_range = false;  ///< Y in 0..255 rather than 16..235.
};

/// @brief One decoded picture, on the device: in a buffer (@ref device) or as
///        images (@ref image), never both. The picture holds them.
struct DecodedPicture {
  /// Width in pixels, after the stream's crop: its display width.
  std::uint32_t width = 0;
  /// Height in pixels, after the stream's crop: its display height.
  std::uint32_t height = 0;
  /// The layout the decoder handed it out in.
  VideoPixelLayout layout = VideoPixelLayout::Nv12;
  std::size_t stride[3] = {};  ///< Bytes per row of each plane in @ref device.
  /// In this storage buffer, from NVDEC or nvJPEG (Linux, through CUDA): NV12
  /// from @ref HevcDecoder, Y at `offset[0]` and the chroma at `offset[1]`, or
  /// Yuv420 from @ref JpegDecoder, Y, U and V at `offset[0]` to `offset[2]`;
  /// rows `stride` bytes apart. The decoder reuses the buffer only once
  /// nothing holds it; drop it before the device it is on is destroyed. CUDA
  /// wrote it, so a reader takes it over from `VK_QUEUE_FAMILY_EXTERNAL`
  /// first (`CommandBatch::acquire`), and a `YuvImage` of it carries
  /// `kQueueFamilyExternal`.
  std::shared_ptr<const core::Buffer> device;
  std::uint64_t offset[3] = {};  ///< Each plane's byte offset in @ref device.
  /// Or as images, from VideoToolbox (Apple), for either decoder: NV12,
  /// `image[0]` the luma (`R8_UNORM`) and `image[1]` the chroma
  /// (`R8G8_UNORM`, U first), each at least the picture's size, chroma halved
  /// and rounded up, the picture at their corner. The images hold the
  /// decoder's picture, which VideoToolbox does not reuse while they are
  /// held; drop them before the device they are on is destroyed. VideoToolbox
  /// has finished writing them, and a `YuvImage` of them takes them as its
  /// `image`.
  std::shared_ptr<const core::Image> image[2];
  std::int64_t pts = 0;  ///< The one sent with its access unit.
  /// The matrix and range the picture is coded in, which its consumer
  /// converts by. A stream that declares no matrix takes the decoder's
  /// `Options::unlabelled_color` when it has one; without one, or for a
  /// matrix this type cannot name, it is taken as BT.709 above 576 rows and
  /// BT.601 at or below, as players do.
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  bool full_range = false;  ///< Y in 0..255 rather than 16..235.
  /// The transfer function and primaries the stream declares, as a capture
  /// source declares them (`core/color_space.hpp`); a stream that declares
  /// neither is taken as BT.709. Empty when the stream declares one
  /// @ref ColorEncoding cannot name (HLG, the SD primaries of BT.601, a
  /// gamma curve), so a driver has nothing true to declare and refuses the
  /// picture rather than fusing it through the wrong curve or basis.
  std::optional<ColorEncoding> encoding = ColorEncoding{
      ColorEncoding::Transfer::Bt709, ColorEncoding::Primaries::Bt709};
  /// Chroma sample positions. JPEG is Center; HEVC uses its stream's
  /// declaration, or Left when none is given.
  ChromaLocation chroma_location = ChromaLocation::Left;
};

}  // namespace volumetric_kit::recon::sensor
