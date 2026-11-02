// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The calls an iOS app makes to code a camera's pictures, linked into an iOS
// executable by tools/ios_build_check.sh, so a symbol iOS lacks, an FFmpeg
// or CUDA one among them, fails the check. Never run: it opens no session
// unless given a picture.

#include <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <cstdio>

#include "volumetric_kit/sensor/apple/core_video_color.hpp"
#include "volumetric_kit/sensor/testing/hevc_conformance.hpp"
#include "volumetric_kit/sensor/video/backend.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"

namespace sensor = volumetric_kit::sensor;

int main(int argc, char** argv) {
  if (sensor::video_backend() != sensor::VideoBackend::VideoToolbox) return 1;
  // A picture only where one is handed in, which no run of this does.
  auto* picture =
      argc > 99 ? reinterpret_cast<CVPixelBufferRef>(argv[0]) : nullptr;
  if (picture == nullptr) return 0;
  const auto tags = sensor::color_codes_of(picture);
  if (!tags) return 1;
  sensor::HevcEncoder::Options o;
  o.width = static_cast<std::uint32_t>(CVPixelBufferGetWidth(picture));
  o.height = static_cast<std::uint32_t>(CVPixelBufferGetHeight(picture));
  o.color = *tags;
  auto encoder = sensor::HevcEncoder::create(o);
  if (!encoder) return 1;
  if (!sensor::send_pixel_buffer(*encoder, picture, true, 0).ok()) return 1;
  sensor::HevcEncoder::CodedPicture coded;
  const auto got = encoder->receive(coded);
  sensor::testing::HevcEncoderCase c;
  c.options = o;
  c.pixel_buffers = true;
  const auto breach = sensor::testing::hevc_encoder_breach(c);
  std::printf("%zu %d\n", coded.access_unit.size(),
              breach.ok() && breach->empty());
  return got.ok() ? 0 : 1;
}
