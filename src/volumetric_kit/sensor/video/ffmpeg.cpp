// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "ffmpeg.hpp"

#include <cerrno>
#include <mutex>
#include <string>

extern "C" {
#include <libavutil/log.h>
}

namespace volumetric_kit::sensor::video {

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

core::Status nvenc_open_failure(const std::string& what, int err) {
  const std::string said = " (" + ffmpeg_message(err) + ")";
  if (err == AVERROR(EINVAL) || err == AVERROR(ENOMEM)) {
    return core::Status::unsupported(
        "HevcEncoder: NVENC did not open " + what + said +
        ": it is at its session limit, busy or out of memory, which another "
        "session closing clears, or this GPU does not take a parameter of it, "
        "its size for one");
  }
  if (err == AVERROR(ENOSYS) || err == AVERROR(ENOENT)) {
    return core::Status::unsupported(
        "HevcEncoder: NVENC on this GPU and driver does not code " + what +
        said +
        ": the GPU has no NVENC or no HEVC on it, the picture is past its "
        "size, or the driver is older than this FFmpeg's NVENC API");
  }
  if (err == -1) {
    return core::Status::unsupported(
        "HevcEncoder: NVIDIA's encode library (libnvidia-encode) did not "
        "load, so no NVENC opens " +
        what + said);
  }
  return ffmpeg_error("HevcEncoder", "opening " + what, err);
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

void set_color(AVCodecContext& context, const ColorCodes& codes) noexcept {
  context.colorspace = static_cast<AVColorSpace>(codes.matrix);
  context.color_trc =
      static_cast<AVColorTransferCharacteristic>(codes.transfer);
  context.color_primaries = static_cast<AVColorPrimaries>(codes.primaries);
  context.color_range = codes.full_range ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
  context.chroma_sample_location =
      static_cast<AVChromaLocation>(codes.chroma_location + 1);
}

void quiet_ffmpeg_log() noexcept {
  static std::once_flag once;
  std::call_once(once, [] { av_log_set_level(AV_LOG_ERROR); });
}

}  // namespace volumetric_kit::sensor::video
