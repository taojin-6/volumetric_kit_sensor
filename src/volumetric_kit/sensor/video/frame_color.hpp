// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The colour a stream declares, as a DecodedPicture describes it: its YCbCr
// matrix and range, transfer and primaries, and chroma siting, from the
// H.273 codes of color_codes.hpp, through the public resolvers
// (video/frame_color.hpp); and the codes an encoder states, normalized and
// checked against the one set every backend writes (HevcEncoder::Options::
// color). Internal.

#include <optional>
#include <string>

#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"
#include "volumetric_kit/sensor/video/decoded_picture.hpp"
#include "volumetric_kit/sensor/video/frame_color.hpp"

namespace volumetric_kit::sensor::video {

// The matrix's weights, range, encoding and chroma siting @p codes declare,
// set on @p picture, whose height is already set: @p unlabelled_color stands
// in for codes that declare no matrix.
void describe_color(
    const ColorCodes& codes,
    const std::optional<VideoColorDescription>& unlabelled_color,
    DecodedPicture& picture);

// @p codes with each pair of codes that name one thing given one code
// (H.273): matrix 5 as 6, transfer 6, 14 and 15 as 1.
ColorCodes normalize_codes(const ColorCodes& codes) noexcept;

// Empty when @p normalized is in the encoders' set: matrix 1, 6, 7 or 9;
// transfer 1, 2, 7, 8, 13, 16, 17 or 18; primaries 1, 2, 5, 6, 9, 11, 12 or
// 22; either range; siting 0, or 1 to 5 where @p writes_siting. Else the
// first code outside it, named ("transfer 3"). Every code Core Video names
// with a string of its own, the matrix also one VideoColorMatrix names; 2
// (unspecified) stays out of the matrix, which HevcEncoder::create refuses
// first.
std::string outside_code_set(const ColorCodes& normalized, bool writes_siting);

}  // namespace volumetric_kit::sensor::video
