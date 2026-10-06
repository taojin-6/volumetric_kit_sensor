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
  if (err == AVERROR_EXTERNAL) return core::Status::backend_error(err, message);
  return core::Status::io_error(message);
}

core::Status ffmpeg_alloc_error(const char* who, const std::string& what) {
  return core::Status::out_of_memory(std::string(who) + ": " + what +
                                     ": FFmpeg could not allocate");
}

}  // namespace volumetric_kit::recon::sensor::video
