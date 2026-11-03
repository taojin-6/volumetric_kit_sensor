// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/hevc_encoder_apple.hpp
/// @brief Apple only: @ref HevcEncoder codes a `CVPixelBuffer` as it is,
///        such as a camera's, with no copy by sensor and no Vulkan device.

#if !defined(__APPLE__)
#error "video/hevc_encoder_apple.hpp is Apple only"
#endif

#include <CoreVideo/CoreVideo.h>

#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/export.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"

namespace volumetric_kit::sensor {

/// @brief Send @p pixels to be coded as it is, an IDR when @p key. Returns
///        at once. It counts toward `HevcEncoder::kInFlight`, and
///        `HevcEncoder::receive` hands back its access unit. A camera's
///        buffer is complete when delivered, so there is no writer to wait
///        for.
///
/// @p pixels is 8-bit NV12 on an IOSurface at the encoder's size: '420f'
/// when the session is full range, '420v' when not. Each colour tag it
/// carries (color_codes_of, apple/core_video_color.hpp) must equal
/// `HevcEncoder::Facts::color`'s; a tag it lacks is taken from that colour.
/// VideoToolbox writes the stream's chroma siting from the buffer, so a
/// buffer with no chroma-location tag is given one where the session's
/// siting is not 0. The encoder retains @p pixels until coded: do not write
/// it after this call. It works with or without `HevcEncoder::Options::
/// device`. A lost session is opened again here, and its first picture must
/// be a key.
/// @return OK. `Status::Code::InvalidArgument` for no @p pixels, another
///         size or format, a tag that differs or that Core Video names no
///         code for, a target taken (`HevcEncoder::target`) and not sent, no
///         key while `HevcEncoder::key_owed`, `HevcEncoder::kInFlight`
///         pictures already in flight, or a moved-from encoder. A lost
///         session's reopening failure, as
///         `HevcEncoder::create`'s. Coding failures come from
///         `HevcEncoder::receive`.
VS_VIDEO_API core::Status send_pixel_buffer(HevcEncoder& encoder,
                                            CVPixelBufferRef pixels, bool key,
                                            std::int64_t pts);

}  // namespace volumetric_kit::sensor
