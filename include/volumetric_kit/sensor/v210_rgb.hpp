// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file v210_rgb.hpp
/// @brief A V210 picture's colour on the host: SDI's 10-bit 4:2:2, as a
///        DeckLink card captures it, made 8-bit R'G'B' as the GPU's colour
///        pass makes it, for colour stills (photogrammetry or NeRF, say).
///
/// Pure arithmetic on the host, with no device: `PictureReadback` copies a
/// V210 picture's words to the host (`Planes::All`), and this converts a
/// copy of them, on whatever thread holds it.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @return The bytes a V210 row of @p width pixels takes: 16 each six pixels
///         or part of six, a partial group's words whole (3840: 10,240). A
///         picture's stride is at least this; a DeckLink card pads its rows
///         to whole 128-byte blocks (10,240 too at 3840).
constexpr std::uint64_t v210_row_bytes(std::uint32_t width) noexcept {
  return (std::uint64_t{width} + 5) / 6 * 16;
}

/// @brief Convert a V210 picture on the host to tightly packed 8-bit R'G'B',
///        each pixel as `GpuFramePrep`'s colour pass converts one its lens
///        leaves where it is.
///
/// The pass's arithmetic, in float, step for step (`undistort_color.glsl`,
/// `color_planes.glsl`; DESIGN.md, `sensor`):
/// - each 10-bit sample read as code / 4, on the 8-bit scale, so Y' is
///   `(code / 4 - 16) / 219` and Pb, Pr `(code / 4 - 128) / 224`: BT.709's
///   `(Y - 64) / 876` and `(C - 512) / 896` at 10 bits;
/// - chroma sited across by @ref YuvImage::chroma_location, sampled linearly
///   between its two nearest samples and clamped to the row's ends:
///   co-sited (`Left`, DeckLink's; `TopLeft`, `BottomLeft`), an even pixel
///   takes its own sample and an odd one the mean of the two either side,
///   the last odd one the last sample; centred (`Center`, `Top`, `Bottom`),
///   three quarters of the nearer and a quarter of the farther; the taps and
///   their weight worked out in quarter samples, in integers, so exact at
///   any width (in float they drift past 2^23 pixels across);
/// - R', G' and B' by the picture's @ref YuvImage::kr and @ref YuvImage::kb,
///   each clamped to [0, 1] and rounded once to 8 bits, half up.
/// No transfer function is applied: the bytes are encoded R'G'B', as the
/// camera coded them. On random 10-bit codes the pass's words and these bytes
/// agree within a code, the device's float rounding a channel near a half
/// code the other way now and then (DECISIONS.md, the 2026-11-09 colour
/// stills entry).
///
/// Calls share no state and may run concurrently. 3840x2160 takes about
/// 18 ms on one Apple M5 Max core (Release).
///
/// @code
/// // A DeckLink pair's pictures, read at once and converted later:
/// VKC_ASSIGN(std::vector<PictureView> views,
///            readback->read({&pair.left, &pair.right},
///                           PictureReadback::Planes::All));
/// const PictureView& v = views[0];
/// std::vector<std::uint8_t> words(
///     v.luma, v.luma + (v.height - 1) * v.stride + v210_row_bytes(v.width));
/// YuvImage copy;  // described as the picture is
/// copy.layout = YuvLayout::V210;
/// copy.host = words.data();
/// copy.stride[0] = v.stride;
/// copy.width = v.width;
/// copy.height = v.height;
/// copy.kr = pair.left.kr;
/// copy.kb = pair.left.kb;
/// copy.full_range = v.full_range;
/// copy.chroma_location = pair.left.chroma_location;
/// // Later, on another thread:
/// std::vector<std::uint8_t> rgb(std::size_t{copy.width} * copy.height * 3);
/// VKC_TRY(v210_to_rgb8(copy, rgb.data(), rgb.size()));
/// @endcode
///
/// @param picture    V210 on the host (@ref YuvImage::host, plane 0 at
///                   @ref YuvImage::offset `[0]`, @ref YuvImage::stride `[0]`
///                   bytes a row), its bytes valid for this call; its width
///                   even, a row at least @ref v210_row_bytes, its offset and
///                   stride whole 32-bit words, limited range, as
///                   `YuvLayout::V210` says.
/// @param rgb        Where the pixels go: `width * height` of them, rows top
///                   to bottom, three bytes each, R first, no padding.
/// @param rgb_bytes  The bytes at @p rgb; at least `width * height * 3`.
/// @return OK; `Status::Code::InvalidArgument`, writing nothing, for a
///         picture not in V210, not on the host or also in a buffer or as
///         images, empty, of odd width, with a stride shorter than its rows,
///         an offset or stride off a word, rows past 64 bits, full range, a
///         matrix whose `kr` and `kb` are not positive with a sum below 1,
///         or an unknown chroma location; or for a null @p rgb or fewer than
///         `width * height * 3` @p rgb_bytes; `Status::Code::OutOfMemory`
///         for a host allocation failure (a row's samples, a few times the
///         width in floats).
VS_INTERFACE_API core::Status v210_to_rgb8(const YuvImage& picture,
                                           std::uint8_t* rgb,
                                           std::size_t rgb_bytes);

}  // namespace volumetric_kit::sensor
