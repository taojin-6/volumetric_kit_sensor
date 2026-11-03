// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/rgbd_pattern.hpp
/// @brief Synthetic depth that carries a frame's index, and the index read
///        back from it. The family's test picture (`test_pattern`,
///        testing/nv12_pattern.hpp) carries the same index, so a test or a
///        stream check sees each frame carry its own picture. Header-only, on
///        the standard library alone, so a source that makes this depth
///        builds with no GPU headers.
///
/// The depth carries the index band (testing/index_band.hpp) at the same
/// pixels of the depth image as the picture carries it in its luma: rows 0
/// to @ref kIndexSquare - 1, 2 + @ref kIndexBits squares of
/// @ref kIndexSquare x @ref kIndexSquare from column 0. A set square is
/// @ref kPatternNearMm and a clear one @ref kPatternFarMm, the plane's own
/// depth. The band sits at the same pixels, not at the same spot of the
/// view: beside a larger picture it covers more of the view in the depth
/// than in the picture, and a frame's depth and picture name one index all
/// the same.
///
/// A reader takes each square's central half, its middle 8 x 8, and every
/// sample there must hold the square's one value, which a lossless depth
/// record keeps. Anything else is no index, so torn depth reads as none
/// rather than as a wrong frame.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// const testing::PatternDepth d = testing::pattern_depth(256, 192, frame);
/// // ... sent, received ...
/// const std::optional<std::uint32_t> index =
///     testing::depth_index(received.data(), 256, 192);  // frame % 16384
/// @endcode

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "volumetric_kit/sensor/testing/index_band.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief A set square's depth in @ref PatternDepth, in millimetres: 1 m.
inline constexpr std::uint16_t kPatternNearMm = 1000;
/// @brief A clear square's depth, and the plane's around the band, in
///        millimetres: 2 m.
inline constexpr std::uint16_t kPatternFarMm = 2000;

/// @brief Synthetic depth that carries a frame index, as a tracked depth
///        sensor (ARKit) delivers its depth: millimetres and a confidence
///        plane.
struct PatternDepth {
  std::uint32_t width = 0;   ///< Pixels.
  std::uint32_t height = 0;  ///< Pixels.
  /// @ref width x @ref height samples in millimetres (`metres_per_unit`
  /// 0.001), rows packed: the band's squares at @ref kPatternNearMm or
  /// @ref kPatternFarMm, the plane around them at @ref kPatternFarMm, and 0
  /// (no return) across the left half of the bottom `height / 8` rows where
  /// the band is not: at 16 rows or fewer the band may cover all of it,
  /// leaving no hole.
  std::vector<std::uint16_t> depth;
  /// One byte a sample, in ARKit's levels: 2 (high) on every square of the
  /// band; elsewhere 0, 1, 2 by column (`x % 3`). Either floor a plane comes
  /// with, 1 or 2 (`RgbdFrame::min_confidence`), then drops part of the
  /// plane and nothing of a square.
  std::vector<std::uint8_t> confidence;
};

/// @return The depth for @p index at @p width x @p height. The band is
///         whole at `kIndexSquare * (2 + kIndexBits)` (256) x
///         @ref kIndexSquare (16) and above, as at ARKit's 256 x 192; a
///         smaller size holds as much of it as fits, as `test_pattern` does,
///         and reads as no index.
inline PatternDepth pattern_depth(std::uint32_t width, std::uint32_t height,
                                  std::uint32_t index) {
  constexpr std::uint32_t kBandWidth = kIndexSquare * (2 + kIndexBits);
  PatternDepth d;
  d.width = width;
  d.height = height;
  d.depth.resize(std::size_t{width} * height);
  d.confidence.resize(d.depth.size());
  const std::uint32_t hole_top = height - height / 8;
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::size_t at = std::size_t{y} * width + x;
      if (y < kIndexSquare && x < kBandWidth) {
        const std::uint32_t square = x / kIndexSquare;
        const bool set =
            square == 0 || (square >= 2 && ((index >> (square - 2)) & 1));
        d.depth[at] = set ? kPatternNearMm : kPatternFarMm;
        d.confidence[at] = 2;
        continue;
      }
      d.depth[at] = y >= hole_top && x < width / 2 ? 0 : kPatternFarMm;
      d.confidence[at] = static_cast<std::uint8_t>(x % 3);
    }
  }
  return d;
}

/// @return `index % 2^kIndexBits` of the index @p depth's band carries,
///         its @p width x @p height samples in millimetres with the rows
///         packed; empty for no samples or a size that does not hold the
///         band, when any square's central half is not all
///         @ref kPatternNearMm or all @ref kPatternFarMm, or when the
///         references are not @ref kPatternNearMm and @ref kPatternFarMm.
inline std::optional<std::uint32_t> depth_index(const std::uint16_t* depth,
                                                std::uint32_t width,
                                                std::uint32_t height) {
  constexpr std::uint32_t kBandWidth = kIndexSquare * (2 + kIndexBits);
  if (depth == nullptr || width < kBandWidth || height < kIndexSquare) {
    return std::nullopt;
  }
  // The one value of a square's central half; 0, never a square's, where
  // its samples differ.
  const auto value = [&](std::uint32_t square) -> std::uint16_t {
    constexpr std::uint32_t lo = kIndexSquare / 4;
    constexpr std::uint32_t hi = kIndexSquare - kIndexSquare / 4;
    const std::uint16_t* first = depth + square * kIndexSquare;
    const std::uint16_t one = first[std::size_t{lo} * width + lo];
    for (std::uint32_t y = lo; y < hi; ++y) {
      for (std::uint32_t x = lo; x < hi; ++x) {
        if (first[std::size_t{y} * width + x] != one) return 0;
      }
    }
    return one;
  };
  if (value(0) != kPatternNearMm || value(1) != kPatternFarMm) {
    return std::nullopt;
  }
  std::uint32_t index = 0;
  for (std::uint32_t b = 0; b < kIndexBits; ++b) {
    const std::uint16_t v = value(2 + b);
    if (v == kPatternNearMm) {
      index |= std::uint32_t{1} << b;
    } else if (v != kPatternFarMm) {
      return std::nullopt;
    }
  }
  return index;
}

}  // namespace volumetric_kit::sensor::testing
