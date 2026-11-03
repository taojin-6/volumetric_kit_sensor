// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The one place this target includes FFmpeg (NVDEC's path): its C headers,
// owning pointers for the objects it allocates, its error codes as a Status,
// and the colour a frame carries. Internal.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

#include <memory>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"

namespace volumetric_kit::sensor::video {

struct CodecContextFree {
  void operator()(AVCodecContext* c) const noexcept {
    avcodec_free_context(&c);
  }
};
struct FrameFree {
  void operator()(AVFrame* f) const noexcept { av_frame_free(&f); }
};
struct PacketFree {
  void operator()(AVPacket* p) const noexcept { av_packet_free(&p); }
};
struct BufferUnref {
  void operator()(AVBufferRef* b) const noexcept { av_buffer_unref(&b); }
};

using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextFree>;
using FramePtr = std::unique_ptr<AVFrame, FrameFree>;
using PacketPtr = std::unique_ptr<AVPacket, PacketFree>;
using BufferRef = std::unique_ptr<AVBufferRef, BufferUnref>;

/// @return FFmpeg's message for @p err (a negative AVERROR).
std::string ffmpeg_message(int err);

/// @return A status naming @p who, @p what it was doing and FFmpeg's
///         message for @p err: OutOfMemory for `AVERROR(ENOMEM)`, an
///         `Ffmpeg` Backend for `AVERROR_EXTERNAL` (a hardware call failed),
///         else IoError.
core::Status ffmpeg_error(const char* who, const std::string& what, int err);

/// @return An OutOfMemory for an allocation FFmpeg refused.
core::Status ffmpeg_alloc_error(const char* who, const std::string& what);

/// @return The colour @p frame declares, as H.273 codes.
ColorCodes color_codes(const AVFrame& frame) noexcept;

/// @brief @p codes as the colour @p context codes: FFmpeg numbers the
///        matrix, transfer and primaries as H.273 does, and the siting one
///        past it.
void set_color(AVCodecContext& context, const ColorCodes& codes) noexcept;

/// @brief FFmpeg's log level set to ERROR, once a process, by the first
///        coder that opens on it (NVDEC's or NVENC's): FFmpeg has one
///        logger, and its warnings at every picture would flood a caller's.
void quiet_ffmpeg_log() noexcept;

}  // namespace volumetric_kit::sensor::video
