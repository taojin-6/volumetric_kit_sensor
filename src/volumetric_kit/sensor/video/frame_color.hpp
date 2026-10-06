// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The colour an FFmpeg frame declares, as a DecodedPicture describes it: its
// YCbCr matrix and range, transfer and primaries, and chroma siting.
// Codec-neutral. Internal.

#include <optional>

#include "ffmpeg.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::video {

/// @return The matrix @p space names; an unspecified one by @p height, as
///         @ref DecodedPicture::matrix describes.
VideoColorMatrix resolve_matrix(AVColorSpace space, int height) noexcept;

/// @return The encoding @p transfer and @p primaries name, as
///         @ref DecodedPicture::encoding describes.
std::optional<ColorEncoding> resolve_encoding(
    AVColorTransferCharacteristic transfer,
    AVColorPrimaries primaries) noexcept;

// The matrix, range, encoding and chroma siting @p frame declares, set on
// @p picture, whose height is already set: @p unlabelled_color stands in for a
// frame that declares no matrix.
void describe_color(
    const AVFrame& frame,
    const std::optional<VideoColorDescription>& unlabelled_color,
    DecodedPicture& picture);

}  // namespace volumetric_kit::recon::sensor::video
