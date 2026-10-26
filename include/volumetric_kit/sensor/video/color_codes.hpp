// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/color_codes.hpp
/// @brief The colour a video stream declares, as ITU-T H.273 code points.

namespace volumetric_kit::recon::sensor {

/// @brief The ITU-T H.273 code points a stream declares: its YCbCr matrix,
///        transfer and primaries, its range and its chroma siting.
///        Codec-neutral: an H.265 SPS's VUI carries them, and FFmpeg's colour
///        enumerations number them alike. A stream that declares none keeps
///        the defaults: 2 ("unspecified") for the three tables, limited
///        range, and chroma sited left.
struct ColorCodes {
  int matrix = 2;           ///< MatrixCoefficients.
  int transfer = 2;         ///< TransferCharacteristics.
  int primaries = 2;        ///< ColourPrimaries.
  bool full_range = false;  ///< VideoFullRangeFlag.
  int chroma_location = 0;  ///< ChromaSampleLocType, 0 to 5.
};

}  // namespace volumetric_kit::recon::sensor
