// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file apple/core_video_color.hpp
/// @brief The colour a `CVPixelBuffer`'s tags state, as H.273 code points
///        (Apple only).

#include <CoreVideo/CoreVideo.h>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/apple/export.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"

namespace volumetric_kit::sensor {

/// @brief Read @p pixels' colour tags: matrix, transfer and primaries from
///        its attachments through Core Video's own mapping
///        (`CV*GetIntegerCodePointForString`); range from its pixel format
///        ('420f' full, '420v' video); siting from its top field's chroma
///        location (Left 0, Center 1, TopLeft 2, Top 3, BottomLeft 4,
///        Bottom 5).
///
/// What the codes mean in sensor's terms is video/frame_color.hpp's.
///
/// @code
/// VKC_ASSIGN(ColorCodes codes, color_codes_of(captured_image));
/// if (codes.matrix == 2) codes.matrix = 6;  // untagged: the source's own
/// const YcbcrWeights w = ycbcr_weights(resolve_matrix(codes.matrix, height));
/// @endcode
/// @return The codes; a tag @p pixels lacks keeps @ref ColorCodes' default,
///         and range stays limited for a format other than those two.
///         `Status::Code::InvalidArgument` for no @p pixels, or for a tag
///         Core Video gives no code point for, naming it.
VS_APPLE_API core::Result<ColorCodes> color_codes_of(CVPixelBufferRef pixels);

}  // namespace volumetric_kit::sensor
