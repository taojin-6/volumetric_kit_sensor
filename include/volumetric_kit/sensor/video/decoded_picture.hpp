// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/decoded_picture.hpp
/// @brief A decoded picture in host memory, borrowed from its decoder.

#include <cstddef>
#include <cstdint>

namespace volumetric_kit::recon::sensor {

/// @brief How a decoded picture's pixels are laid out.
enum class VideoPixelLayout {
  /// One plane of R, G, B bytes, converted by @ref DecodedPicture::matrix.
  Rgb24,
  /// Three 8-bit planes as decoded: Y, then U and V at half size.
  Yuv420,
};

/// @brief The YCbCr-to-RGB matrix of a coded picture.
enum class VideoColorMatrix { Bt601, Bt709, Bt2020 };

/// @brief One decoded picture. Its planes belong to the decoder and are valid
///        until the decoder's next call.
struct DecodedPicture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  VideoPixelLayout layout = VideoPixelLayout::Rgb24;
  const std::uint8_t* plane[3] = {};  ///< Rgb24 uses plane[0] only.
  std::size_t stride[3] = {};         ///< Bytes per row of each plane.
  std::int64_t pts = 0;               ///< The one sent with its access unit.
  /// The matrix and range the stream declares: what Rgb24 was converted by,
  /// and what a Yuv420 consumer converts by. A stream that declares no matrix
  /// is taken as BT.709 above 576 rows and BT.601 at or below, as players do.
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  bool full_range = false;  ///< Y in 0..255 rather than 16..235.
};

}  // namespace volumetric_kit::recon::sensor
