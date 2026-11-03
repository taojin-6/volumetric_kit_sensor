// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// vt_pattern_clip: the committed VideoToolbox-coded clip
// (tests/data/hevc/vt_pattern_1920x1440.h265), made by HevcEncoder on this
// Mac's hardware from IOSurface pixel buffers, as a camera's are sent: the
// test pattern's pictures 0 to 11 at 1920x1440, BT.601 full range, sited
// left, QP 30, keys at 0 and 6, each access unit after a delimiter so the
// tests split it (testing/hevc_clip.hpp). The decoder tests decode it on
// NVDEC and VideoToolbox alike.
//
//   vt_pattern_clip <out.h265>

#include <cstdint>
#include <cstdio>
#include <fstream>

#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/testing/nv12_pixel_buffer.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"

namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: vt_pattern_clip <out.h265>\n");
    return 2;
  }
  constexpr std::uint32_t kPictures = 12;
  sensor::HevcEncoder::Options o;
  o.width = 1920;
  o.height = 1440;
  o.qp = 30;
  o.color = {6, 1, 6, true, 0};
  auto encoder = sensor::HevcEncoder::create(o);
  if (!encoder) {
    std::fprintf(stderr, "%s\n", encoder.status().message().c_str());
    return 1;
  }
  std::ofstream out(argv[1], std::ios::binary);
  // An access unit delimiter: AUD_NUT, pic_type 2 (I, P or B).
  const char kDelimiter[] = {0, 0, 0, 1, 0x46, 0x01, 0x50};
  sensor::HevcEncoder::CodedPicture coded;
  for (std::uint32_t i = 0; i < kPictures; ++i) {
    auto pixels = testing::pixel_buffer(
        testing::test_pattern(o.width, o.height, i), o.color);
    if (!pixels) {
      std::fprintf(stderr, "%s\n", pixels.status().message().c_str());
      return 1;
    }
    const auto sent =
        sensor::send_pixel_buffer(*encoder, *pixels, i == 0 || i == 6, i);
    CVPixelBufferRelease(*pixels);
    const auto got = sent.ok() ? encoder->receive(coded)
                               : volumetric_kit::core::Result<bool>(sent);
    if (!got.ok() || !*got) {
      std::fprintf(stderr, "picture %u: %s\n", i,
                   got.ok() ? "none" : got.status().message().c_str());
      return 1;
    }
    out.write(kDelimiter, sizeof kDelimiter);
    out.write(reinterpret_cast<const char*>(coded.access_unit.data()),
              static_cast<std::streamsize>(coded.access_unit.size()));
  }
  std::printf("%s: %u pictures, %s\n", argv[1], kPictures,
              encoder->facts().settings.c_str());
  return out ? 0 : 1;
}
