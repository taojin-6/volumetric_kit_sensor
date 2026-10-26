// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The colour a stream declares, as a DecodedPicture describes it: its YCbCr
// matrix and range, transfer and primaries, and chroma siting, from the
// H.273 codes of color_codes.hpp. Internal.

#include <optional>

#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/video/color_codes.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::video {

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
