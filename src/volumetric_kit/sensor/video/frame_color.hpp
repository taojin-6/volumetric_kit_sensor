// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The colour a stream declares, as a DecodedPicture describes it: its YCbCr
// matrix and range, transfer and primaries, and chroma siting, from the
// H.273 codes of color_codes.hpp, through the public resolvers
// (video/frame_color.hpp). Internal.

#include <optional>

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

}  // namespace volumetric_kit::sensor::video
