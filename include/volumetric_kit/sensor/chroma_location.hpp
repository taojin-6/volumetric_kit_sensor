// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/chroma_location.hpp
/// @brief Chroma sample positions shared by decoders and frame preparation.

#include <array>
#include <cstdint>

namespace volumetric_kit::recon::sensor {

/// @brief Where each 4:2:0 chroma sample sits within its 2x2 luma group.
enum class ChromaLocation : std::uint8_t {
  Left,        ///< Even luma column, halfway between rows (HEVC's default).
  Center,      ///< Halfway between both columns and rows (JPEG).
  TopLeft,     ///< Even luma column and row.
  Top,         ///< Halfway between columns, even row.
  BottomLeft,  ///< Even luma column, odd row.
  Bottom,      ///< Halfway between columns, odd row.
};

/// @brief Locate the first chroma sample relative to the first luma sample.
/// @param location One of the defined @ref ChromaLocation values.
/// @return Horizontal and vertical offsets in luma pixels, whose centres
///         are at integer coordinates. Later chroma samples are 2 pixels apart.
constexpr std::array<float, 2> chroma_offset(ChromaLocation location) noexcept {
  switch (location) {
    case ChromaLocation::Left:
      return {0.0f, 0.5f};
    case ChromaLocation::Center:
      return {0.5f, 0.5f};
    case ChromaLocation::TopLeft:
      return {0.0f, 0.0f};
    case ChromaLocation::Top:
      return {0.5f, 0.0f};
    case ChromaLocation::BottomLeft:
      return {0.0f, 1.0f};
    case ChromaLocation::Bottom:
      return {0.5f, 1.0f};
  }
  return {0.0f, 0.5f};
}

}  // namespace volumetric_kit::recon::sensor
