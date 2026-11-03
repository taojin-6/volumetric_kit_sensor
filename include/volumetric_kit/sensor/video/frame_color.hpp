// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/frame_color.hpp
/// @brief What H.273 colour codes mean in sensor's terms: the matrix a
///        YuvImage is weighted by, the encoding a frame declares, and the
///        chroma siting. The decoders describe their pictures by it, and
///        capture code its sources' tags.
///
/// The codes are `ColorCodes`' (`video/color_codes.hpp`): an H.265 SPS's VUI
/// declares them (`hevc::Sps::color`), FFmpeg numbers its enumerations alike,
/// and Core Video's tags map onto them. Host code, with no codec or device.
///
/// @code
/// const ColorCodes& codes = sps.color;  // an hevc::parse_sps result
/// const YcbcrWeights w =
///     ycbcr_weights(resolve_matrix(codes.matrix, sps.height()));
/// yuv.kr = w.kr;  // yuv: a picture of the stream's
/// yuv.kb = w.kb;
/// yuv.full_range = codes.full_range;
/// yuv.chroma_location = resolve_chroma_location(codes.chroma_location);
/// const std::optional<ColorEncoding> encoding =
///     resolve_encoding(codes.transfer, codes.primaries);  // empty: refuse
/// @endcode

#include <cstdint>
#include <optional>

#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/video/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @return The matrix @p matrix (MatrixCoefficients) names; for one
///         @ref VideoColorMatrix does not name (2, unspecified, among them),
///         BT.709 above 576 rows and BT.601 at or below, as players guess.
/// @param height  The picture's luma rows.
VS_VIDEO_API VideoColorMatrix resolve_matrix(int matrix,
                                             std::uint32_t height) noexcept;

/// @return The encoding @p transfer (TransferCharacteristics) and
///         @p primaries (ColourPrimaries) name, either unspecified (2) read
///         as BT.709's; empty for one @ref ColorEncoding cannot name (HLG,
///         the SD primaries of BT.601, a gamma curve), so a caller has
///         nothing true to declare.
VS_VIDEO_API std::optional<ColorEncoding> resolve_encoding(
    int transfer, int primaries) noexcept;

/// @return The siting ChromaSampleLocType @p code names (0 to 5); Left for
///         any other, as for a stream that declares none.
VS_VIDEO_API ChromaLocation resolve_chroma_location(int code) noexcept;

}  // namespace volumetric_kit::sensor
