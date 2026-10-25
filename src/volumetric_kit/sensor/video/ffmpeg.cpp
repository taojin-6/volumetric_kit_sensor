// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "ffmpeg.hpp"

#include <string>

namespace volumetric_kit::recon::sensor::video {

std::string ffmpeg_message(int err) {
  char text[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, text, sizeof(text));
  return text;
}

core::Status ffmpeg_error(const char* who, const std::string& what, int err) {
  const std::string message =
      std::string(who) + ": " + what + ": " + ffmpeg_message(err);
  if (err == AVERROR(ENOMEM)) return core::Status::out_of_memory(message);
  // A hardware call failed inside FFmpeg.
  if (err == AVERROR_EXTERNAL) {
    return core::Status::backend_error(core::Status::Backend::Ffmpeg, err,
                                       message);
  }
  return core::Status::io_error(message);
}

core::Status ffmpeg_alloc_error(const char* who, const std::string& what) {
  return core::Status::out_of_memory(std::string(who) + ": " + what +
                                     ": FFmpeg could not allocate");
}

ColorCodes color_codes(const AVFrame& frame) noexcept {
  ColorCodes codes;
  codes.matrix = frame.colorspace;
  codes.transfer = frame.color_trc;
  codes.primaries = frame.color_primaries;
  codes.full_range = frame.color_range == AVCOL_RANGE_JPEG;
  // FFmpeg numbers the chroma sitings one past H.273, from 0 for none.
  if (frame.chroma_location != AVCHROMA_LOC_UNSPECIFIED) {
    codes.chroma_location = frame.chroma_location - 1;
  }
  return codes;
}

}  // namespace volumetric_kit::recon::sensor::video
