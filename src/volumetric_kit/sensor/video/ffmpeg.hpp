// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The one place this target includes FFmpeg: its C headers, owning pointers
// for the objects it allocates, and its error codes as a Status. Internal.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <string>

#include "volumetric_kit/recon/core/result.hpp"

namespace volumetric_kit::recon::sensor::video {

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
struct SwsFree {
  void operator()(SwsContext* s) const noexcept { sws_freeContext(s); }
};

using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextFree>;
using FramePtr = std::unique_ptr<AVFrame, FrameFree>;
using PacketPtr = std::unique_ptr<AVPacket, PacketFree>;
using BufferRef = std::unique_ptr<AVBufferRef, BufferUnref>;
using SwsContextPtr = std::unique_ptr<SwsContext, SwsFree>;

/// @return An IoError naming @p who, @p what it was doing and FFmpeg's
///         message for @p err (a negative AVERROR).
Status ffmpeg_error(const char* who, const std::string& what, int err);

/// @return An IoError for an allocation FFmpeg refused.
Status ffmpeg_alloc_error(const char* who, const std::string& what);

}  // namespace volumetric_kit::recon::sensor::video
