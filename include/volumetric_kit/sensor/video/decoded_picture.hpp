// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/decoded_picture.hpp
/// @brief A decoded picture: in host memory, borrowed from its decoder, or on
///        the device, held by the picture.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "volumetric_kit/recon/core/color_space.hpp"

namespace volumetric_kit::recon {
class Buffer;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::sensor {

/// @brief How a decoded picture's pixels are laid out.
enum class VideoPixelLayout {
  /// One plane of R, G, B bytes, converted by @ref DecodedPicture::matrix.
  Rgb24,
  /// Three 8-bit planes as decoded: Y, then U and V at half size.
  Yuv420,
  /// Two 8-bit planes: Y, then U and V interleaved at half size, U first. An
  /// HEVC picture handed out on the device comes this way, as the hardware
  /// decodes it.
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

/// @return @p matrix's weights, as the standards give them: what a converter
///         of its own (a GPU pass) needs in place of swscale's tables.
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

/// @brief One decoded picture. Host planes belong to the decoder and are valid
///        until the decoder's next call; a device picture is held by the
///        picture.
struct DecodedPicture {
  /// Width in pixels, after the stream's crop: its display width.
  std::uint32_t width = 0;
  /// Height in pixels, after the stream's crop: its display height.
  std::uint32_t height = 0;
  /// The layout the decoder was asked for; a device picture's as the
  /// decoder hands it out.
  VideoPixelLayout layout = VideoPixelLayout::Rgb24;
  const std::uint8_t* plane[3] = {};  ///< Rgb24 uses plane[0] only.
  std::size_t stride[3] = {};         ///< Bytes per row of each plane.
  /// On the device instead, for a decoder given a device it decodes on, in
  /// this storage buffer: NV12 from @ref HevcDecoder, Y at `offset[0]` and
  /// the chroma at `offset[1]`, or Yuv420 from @ref JpegDecoder, Y, U and V
  /// at `offset[0]` to `offset[2]`; rows `stride` bytes apart, and @ref plane
  /// empty. The picture holds the buffer, and the decoder reuses it only once
  /// nothing does; drop it before the device it is on is destroyed. CUDA
  /// wrote it, so a reader takes it over from `VK_QUEUE_FAMILY_EXTERNAL`
  /// first (`CommandBatch::acquire`), and a `YuvImage` of it carries
  /// `kQueueFamilyExternal`.
  std::shared_ptr<const Buffer> device;
  std::uint64_t offset[3] = {};  ///< Each plane's byte offset in @ref device.
  std::int64_t pts = 0;          ///< The one sent with its access unit.
  /// The matrix and range the stream declares: what Rgb24 was converted by,
  /// and what a Yuv420 consumer converts by. A stream that declares no matrix
  /// takes the decoder's `Options::unlabelled_color` when it has one. Without
  /// one, or for a matrix swscale has no table for, it is taken as BT.709
  /// above 576 rows and BT.601 at or below, as players do. A stream coded as
  /// RGB is converted to Yuv420 by that same choice.
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
};

}  // namespace volumetric_kit::recon::sensor
