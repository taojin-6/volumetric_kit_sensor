// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/decoded_picture.hpp
/// @brief A decoded picture in host memory, borrowed from its decoder.

#include <cstddef>
#include <cstdint>
#include <optional>

#include "volumetric_kit/recon/core/color_space.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief How a decoded picture's pixels are laid out.
enum class VideoPixelLayout {
  /// One plane of R, G, B bytes, converted by @ref DecodedPicture::matrix.
  Rgb24,
  /// Three 8-bit planes as decoded: Y, then U and V at half size.
  Yuv420,
};

/// @brief The YCbCr-to-RGB matrix of a coded picture.
enum class VideoColorMatrix {
  Bt601,      ///< BT.601 (SMPTE 170M, BT.470 B/G): SD.
  Bt709,      ///< BT.709: HD.
  Bt2020,     ///< BT.2020 non-constant luminance: UHD.
  Smpte240m,  ///< SMPTE 240M, the interim HD matrix.
  Fcc,        ///< FCC 73.682, the 1953 NTSC matrix.
};

/// @brief A stream's YCbCr matrix and range.
struct VideoColorDescription {
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  bool full_range = false;  ///< Y in 0..255 rather than 16..235.
};

/// @brief One decoded picture. Its planes belong to the decoder and are valid
///        until the decoder's next call.
struct DecodedPicture {
  /// Width in pixels, after the stream's crop: its display width.
  std::uint32_t width = 0;
  /// Height in pixels, after the stream's crop: its display height.
  std::uint32_t height = 0;
  /// The layout the decoder was asked for.
  VideoPixelLayout layout = VideoPixelLayout::Rgb24;
  const std::uint8_t* plane[3] = {};  ///< Rgb24 uses plane[0] only.
  std::size_t stride[3] = {};         ///< Bytes per row of each plane.
  std::int64_t pts = 0;               ///< The one sent with its access unit.
  /// The matrix and range the stream declares, or the decoder's
  /// `Options::color` in their place: what Rgb24 was converted by, and what a
  /// Yuv420 consumer converts by. A stream that declares no matrix, or one
  /// swscale has no table for, is taken as BT.709 above 576 rows and BT.601
  /// at or below, as players do. A stream coded as RGB is converted to Yuv420
  /// by that same choice.
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
