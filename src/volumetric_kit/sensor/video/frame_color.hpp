// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The colour a stream declares, as a DecodedPicture describes it: its YCbCr
// matrix and range, transfer and primaries, and chroma siting. Codec-neutral:
// the codes are ITU-T H.273's, which an H.265 SPS's VUI carries and FFmpeg's
// colour enumerations number alike. Internal.

#include <optional>

#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::video {

// The H.273 code points a stream declares; 2 is "unspecified" for the three
// tables, as a stream that declares no colour leaves them.
struct ColorCodes {
  int matrix = 2;           // MatrixCoefficients
  int transfer = 2;         // TransferCharacteristics
  int primaries = 2;        // ColourPrimaries
  bool full_range = false;  // VideoFullRangeFlag: limited unless declared
  int chroma_location = 0;  // ChromaSampleLocType: 0 (left) unless declared
};

/// @return The matrix @p matrix names; an unspecified one by @p height, as
///         @ref DecodedPicture::yuv describes.
VideoColorMatrix resolve_matrix(int matrix, int height) noexcept;

/// @return The encoding @p transfer and @p primaries name, as
///         @ref DecodedPicture::encoding describes.
std::optional<ColorEncoding> resolve_encoding(int transfer,
                                              int primaries) noexcept;

// The matrix's weights, range, encoding and chroma siting @p codes declare,
// set on @p picture, whose height is already set: @p unlabelled_color stands
// in for codes that declare no matrix.
void describe_color(
    const ColorCodes& codes,
    const std::optional<VideoColorDescription>& unlabelled_color,
    DecodedPicture& picture);

}  // namespace volumetric_kit::recon::sensor::video
