// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "ffmpeg.hpp"

#include <string>

namespace volumetric_kit::recon::sensor::video {

Status ffmpeg_error(const char* who, const std::string& what, int err) {
  char text[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(err, text, sizeof(text));
  return Status::io_error(std::string(who) + ": " + what + ": " + text);
}

Status ffmpeg_alloc_error(const char* who, const std::string& what) {
  return Status::io_error(std::string(who) + ": " + what +
                          ": FFmpeg could not allocate");
}

}  // namespace volumetric_kit::recon::sensor::video
