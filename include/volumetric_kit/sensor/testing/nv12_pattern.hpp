// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/nv12_pattern.hpp
/// @brief Test support for 8-bit NV12 pictures on the host, with no test
///        framework, codec or GPU: a picture's planes, and the family's
///        test pattern, whose index a reader recovers after lossy coding.
///        Header-only, on the standard library alone; its band's layout is
///        testing/index_band.hpp's.
///
/// testing/nv12_picture.hpp puts a picture on the device.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// const testing::HostNv12 picture = testing::test_pattern(256, 144, frame);
/// // picture.luma: 256 x 144 bytes; picture.chroma: 72 rows of 128 Cb, Cr
/// @endcode

#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/sensor/testing/index_band.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief An 8-bit NV12 picture on the host: its luma rows, then its chroma
///        rows, each plane's rows packed.
struct HostNv12 {
  std::uint32_t width = 0;   ///< Luma width (pixels).
  std::uint32_t height = 0;  ///< Luma height (pixels).
  /// @ref width x @ref height bytes.
  std::vector<std::uint8_t> luma;
  /// (@ref height + 1) / 2 rows of (@ref width + 1) / 2 Cb, Cr pairs, Cb
  /// first: 2 x (@ref width + 1) / 2 bytes a row.
  std::vector<std::uint8_t> chroma;
};

/// @return Picture @p index of the family's test pattern, @p width x
///         @p height: 8-bit codes, with 0 and 255 in them where the picture
///         is at least 32 wide, and, in a picture at least 256 x 16, a
///         different picture at each index below 16384.
///
/// - **The index band**, luma rows 0 to 15 and columns 0 to 255, is 16
///   squares of @ref kIndexSquare x @ref kIndexSquare, left to right. Square
///   0 is 255 and square 1 is 0, the references. Square 2 + b is 255 where
///   bit b of @p index
///   is set and 0 where it is not, for b from 0 to @ref kIndexBits - 1, so
///   the band names `index % 16384`. A reader takes each square's mean away
///   from its edges and compares it with the references' midpoint, which
///   holds after lossy coding and a change of range. The band's chroma
///   (chroma rows 0 to 7, pairs 0 to 127) is 128, neutral. A picture under
///   256 x 16 holds as much of the band as fits.
/// - **Elsewhere** each plane is a triangle wave, smooth, so it codes well,
///   and moving with the index, as a P-picture predicts it: luma
///   `tri(x + 2y + 3 index, 255)`, Cb `64 + tri(2x + y + 2 index, 127)` and
///   Cr `64 + tri(x + 2y + 5 index, 127)` at chroma pair x of chroma row y,
///   where `tri(v, top)` rises from 0 to `top` and falls back over
///   `2 top`.
inline HostNv12 test_pattern(std::uint32_t width, std::uint32_t height,
                             std::uint32_t index) {
  const auto tri = [](std::uint64_t v, std::uint64_t top) {
    const std::uint64_t r = v % (2 * top);
    return static_cast<std::uint8_t>(r <= top ? r : 2 * top - r);
  };
  constexpr std::uint32_t kBandWidth = kIndexSquare * (2 + kIndexBits);
  const std::uint64_t i = index;
  HostNv12 p;
  p.width = width;
  p.height = height;
  p.luma.resize(std::size_t{width} * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      std::uint8_t v = tri(x + 2 * std::uint64_t{y} + 3 * i, 255);
      if (y < kIndexSquare && x < kBandWidth) {
        const std::uint32_t square = x / kIndexSquare;
        const bool set =
            square == 0 || (square >= 2 && ((i >> (square - 2)) & 1));
        v = set ? 255 : 0;
      }
      p.luma[std::size_t{y} * width + x] = v;
    }
  }
  const std::uint32_t pairs = (width + 1) / 2;
  const std::uint32_t rows = (height + 1) / 2;
  p.chroma.resize(std::size_t{pairs} * rows * 2);
  for (std::uint32_t y = 0; y < rows; ++y) {
    for (std::uint32_t x = 0; x < pairs; ++x) {
      std::uint8_t* pair = &p.chroma[(std::size_t{y} * pairs + x) * 2];
      if (y < kIndexSquare / 2 && x < kBandWidth / 2) {
        pair[0] = pair[1] = 128;
        continue;
      }
      pair[0] = static_cast<std::uint8_t>(
          64 + tri(2 * std::uint64_t{x} + y + 2 * i, 127));
      pair[1] = static_cast<std::uint8_t>(
          64 + tri(x + 2 * std::uint64_t{y} + 5 * i, 127));
    }
  }
  return p;
}

/// @return The index the band of a @ref test_pattern picture names, read as
///         @ref test_pattern says, from its luma: each square's mean over
///         its middle 8 x 8 against the references' midpoint, so it holds
///         after lossy coding and a change of range. -1 where the references
///         differ by less than 64, as in a picture that is not the pattern.
/// @param luma    The luma plane's first row.
/// @param stride  Bytes from one luma row to the next.
/// The picture must be at least 256 x 16.
inline long pattern_index(const std::uint8_t* luma, std::size_t stride) {
  const auto mean = [&](std::uint32_t square) {
    unsigned sum = 0;
    for (std::uint32_t y = 4; y < 12; ++y) {
      for (std::uint32_t x = 4; x < 12; ++x) {
        sum += luma[std::size_t{y} * stride + square * kIndexSquare + x];
      }
    }
    return sum / 64.0;
  };
  const double white = mean(0);
  const double black = mean(1);
  if (white - black < 64) return -1;
  long index = 0;
  for (std::uint32_t b = 0; b < kIndexBits; ++b) {
    if (mean(2 + b) > (white + black) / 2) index |= 1L << b;
  }
  return index;
}

}  // namespace volumetric_kit::sensor::testing
