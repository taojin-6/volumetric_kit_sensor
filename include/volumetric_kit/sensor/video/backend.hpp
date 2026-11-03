// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/backend.hpp
/// @brief The hardware this build's video coders run on.

#include <cstdint>

#include "volumetric_kit/sensor/video/export.hpp"

namespace volumetric_kit::sensor {

/// @brief The one backend a build of `volumetric_kit::sensor_video` has,
///        fixed when it is built.
enum class VideoBackend : std::uint8_t {
  /// Apple: @ref HevcDecoder, @ref JpegDecoder and @ref HevcEncoder on
  /// VideoToolbox.
  VideoToolbox,
  /// Linux with `VS_WITH_CUDA`: @ref HevcDecoder on NVDEC through FFmpeg and
  /// @ref JpegDecoder on nvJPEG. @ref HevcEncoder::create refuses
  /// (`Status::Code::Unsupported`) until its NVENC backend lands.
  Nvidia,
};

/// @return This build's backend.
VS_VIDEO_API VideoBackend video_backend() noexcept;

}  // namespace volumetric_kit::sensor
