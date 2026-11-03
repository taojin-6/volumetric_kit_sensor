// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/index_band.hpp
/// @brief The index band's layout, which the family's test picture
///        (`test_pattern`, testing/nv12_pattern.hpp) and its test depth
///        (`pattern_depth`, testing/rgbd_pattern.hpp) both carry: rows 0 to
///        @ref kIndexSquare - 1 of an image, 2 + @ref kIndexBits squares of
///        @ref kIndexSquare x @ref kIndexSquare from column 0. Square 0 is
///        set and square 1 clear, the references; square 2 + b holds bit b
///        of the index, so the band names `index % 2^kIndexBits`. Each image
///        says what set and clear are in it. Header-only, on the standard
///        library alone.

#include <cstdint>

namespace volumetric_kit::sensor::testing {

/// @brief The side of one square of the index band, in pixels: 16, a coding
///        block's, so each square of a picture is whole blocks.
inline constexpr std::uint32_t kIndexSquare = 16;
/// @brief The bits of the index the band holds: 14, so it names
///        `index % 16384`.
inline constexpr std::uint32_t kIndexBits = 14;

}  // namespace volumetric_kit::sensor::testing
