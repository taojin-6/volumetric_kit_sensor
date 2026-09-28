// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC decoder on a committed clip (tools/make_hevc_fixtures.sh): the
// software decoder against the pattern the clip was made from, each hardware
// back end against software (a conformant decoder is bit-exact), Auto's
// choice, the refusals, and moves.
//
// VR_TEST_HEVC_BACKEND=<name> requires that back end: it must decode here and
// Auto must choose it. CI sets it where a leg promises hardware (cuda on the
// NVIDIA containers, videotoolbox on macOS), so a runner that silently falls
// back to software fails rather than passes.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"
#include "yuv_reference.hpp"

namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::HevcDecoder;
using sensor::VideoDecodeBackend;
using sensor::VideoPixelLayout;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 144;
constexpr int kFrames = 8;

std::int64_t pts_of(int frame) { return 1000 + 33 * frame; }

// The clip's pattern: patch p = (column + frame + 3 * row) mod 8 over 32x72
// luma patches; Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch(int column, int row, int frame) {
  return (column + frame + 3 * row) % 8;
}
int patch_y(int p) { return 40 + 24 * p; }
int patch_u(int p) { return 64 + 16 * ((3 * p) % 8); }
int patch_v(int p) { return 64 + 16 * ((5 * p) % 8); }

// A picture copied out of the decoder, planes packed.
struct Picture {
  sensor::DecodedPicture meta;
  std::vector<std::uint8_t> planes[3];
};

Picture copy(const sensor::DecodedPicture& p) {
  Picture out;
  out.meta = p;
  const bool rgb = p.layout == VideoPixelLayout::Rgb24;
  for (int i = 0; i < (rgb ? 1 : 3); ++i) {
    const std::size_t row =
        rgb ? 3u * p.width : (i == 0 ? p.width : p.width / 2);
    const std::size_t rows = i == 0 ? p.height : p.height / 2;
    for (std::size_t r = 0; r < rows; ++r) {
      const std::uint8_t* line = p.plane[i] + r * p.stride[i];
      out.planes[i].insert(out.planes[i].end(), line, line + row);
    }
    out.meta.plane[i] = nullptr;
  }
  return out;
}

// The clip's access units, split at each access unit delimiter (NAL type 35).
std::vector<std::vector<std::uint8_t>> access_units() {
  std::ifstream in(VR_HEVC_FIXTURE, std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
  std::vector<std::size_t> starts;
  for (std::size_t i = 0; i + 3 < bytes.size(); ++i) {
    if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 &&
        ((bytes[i + 3] >> 1) & 0x3f) == 35) {
      starts.push_back(i > 0 && bytes[i - 1] == 0 ? i - 1 : i);
    }
  }
  std::vector<std::vector<std::uint8_t>> units;
  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t end =
        k + 1 < starts.size() ? starts[k + 1] : bytes.size();
    units.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(starts[k]),
                       bytes.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return units;
}

// Every picture of the clip, taking each as soon as it is ready.
vr::Result<std::vector<Picture>> decode_clip(HevcDecoder& decoder) {
  std::vector<Picture> pictures;
  const auto drain = [&]() -> vr::Status {
    for (;;) {
      VR_ASSIGN(auto picture, decoder.receive());
      if (!picture) return {};
      pictures.push_back(copy(*picture));
    }
  };
  const auto units = access_units();
  for (std::size_t i = 0; i < units.size(); ++i) {
    VR_TRY(decoder.send(units[i].data(), units[i].size(),
                        pts_of(static_cast<int>(i))));
    VR_TRY(drain());
  }
  VR_TRY(decoder.send(nullptr, 0, 0));
  VR_TRY(drain());
  return pictures;
}

vr::Result<std::vector<Picture>> decode_with(VideoDecodeBackend backend,
                                             VideoPixelLayout layout) {
  HevcDecoder::Options options;
  options.backend = backend;
  options.layout = layout;
  VR_ASSIGN(HevcDecoder decoder, HevcDecoder::create(options));
  if (backend != VideoDecodeBackend::Auto && decoder.backend() != backend) {
    return vr::Status::io_error("decoder runs elsewhere");
  }
  return decode_clip(decoder);
}

int check_clip_shape(const std::vector<Picture>& pictures) {
  CHECK(pictures.size() == static_cast<std::size_t>(kFrames));
  for (int f = 0; f < kFrames; ++f) {
    const auto& meta = pictures[static_cast<std::size_t>(f)].meta;
    CHECK(meta.width == kWidth && meta.height == kHeight);
    CHECK(meta.pts == pts_of(f));
    CHECK(meta.matrix == sensor::VideoColorMatrix::Bt709);
    CHECK(!meta.full_range);
  }
  return 0;
}

// Software, Yuv420: every patch's interior against the pattern (qp 4, so
// within a code or two), which also pins the order and pts of the pictures.
int test_software_yuv() {
  auto decoded =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Yuv420);
  if (!decoded)
    std::fprintf(stderr, "%s\n", decoded.status().message().c_str());
  CHECK(decoded.ok());
  if (check_clip_shape(decoded.value()) != 0) return 1;
  int worst = 0;
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.value()[static_cast<std::size_t>(f)];
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int p = patch(col, row, f);
        for (int dy = -8; dy < 8; ++dy) {
          for (int dx = -8; dx < 8; ++dx) {
            const int x = 32 * col + 16 + dx;
            const int y = 72 * row + 36 + dy;
            const int cx = x / 2;
            const int cy = y / 2;
            const int got[3] = {pic.planes[0][y * kWidth + x],
                                pic.planes[1][cy * (kWidth / 2) + cx],
                                pic.planes[2][cy * (kWidth / 2) + cx]};
            const int want[3] = {patch_y(p), patch_u(p), patch_v(p)};
            for (int k = 0; k < 3; ++k) {
              worst = std::max(worst, std::abs(got[k] - want[k]));
            }
          }
        }
      }
    }
  }
  if (worst > 2) std::fprintf(stderr, "worst YUV error %d\n", worst);
  CHECK(worst <= 2);
  return 0;
}

// Software, Rgb24: each patch's centre against BT.709 limited range.
int test_software_rgb() {
  auto decoded =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Rgb24);
  CHECK(decoded.ok());
  if (check_clip_shape(decoded.value()) != 0) return 1;
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.value()[static_cast<std::size_t>(f)];
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int p = patch(col, row, f);
        const auto want =
            yuv_reference::rgb(patch_y(p), patch_u(p), patch_v(p),
                               sensor::VideoColorMatrix::Bt709, false);
        const int x = 32 * col + 16;
        const int y = 72 * row + 36;
        for (int k = 0; k < 3; ++k) {
          const int got = pic.planes[0][3 * (y * kWidth + x) + k];
          if (std::abs(got - want[k]) > 3) {
            std::fprintf(stderr, "frame %d patch %d,%d channel %d: %d vs %d\n",
                         f, col, row, k, got, want[k]);
            CHECK(false);
          }
        }
      }
    }
  }
  return 0;
}

// Each hardware back end: Yuv420 identical to software, Rgb24 within a code.
int test_hardware_matches_software() {
  auto soft_yuv =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Yuv420);
  auto soft_rgb =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Rgb24);
  CHECK(soft_yuv.ok() && soft_rgb.ok());
  for (const VideoDecodeBackend backend : HevcDecoder::hardware_backends()) {
    std::printf("  hardware back end: %s\n", sensor::to_string(backend));
    auto yuv = decode_with(backend, VideoPixelLayout::Yuv420);
    if (!yuv) std::fprintf(stderr, "%s\n", yuv.status().message().c_str());
    CHECK(yuv.ok());
    if (check_clip_shape(yuv.value()) != 0) return 1;
    for (int f = 0; f < kFrames; ++f) {
      for (int i = 0; i < 3; ++i) {
        CHECK(yuv.value()[static_cast<std::size_t>(f)].planes[i] ==
              soft_yuv.value()[static_cast<std::size_t>(f)].planes[i]);
      }
    }
    auto rgb = decode_with(backend, VideoPixelLayout::Rgb24);
    CHECK(rgb.ok());
    if (check_clip_shape(rgb.value()) != 0) return 1;
    int worst = 0;
    for (int f = 0; f < kFrames; ++f) {
      const auto& a = rgb.value()[static_cast<std::size_t>(f)].planes[0];
      const auto& b = soft_rgb.value()[static_cast<std::size_t>(f)].planes[0];
      CHECK(a.size() == b.size());
      for (std::size_t k = 0; k < a.size(); ++k) {
        worst = std::max(worst, std::abs(a[k] - b[k]));
      }
    }
    CHECK(worst <= 1);
  }
  return 0;
}

// Auto takes the first hardware back end, else software; the environment can
// demand which.
int test_auto_choice() {
  const auto hardware = HevcDecoder::hardware_backends();
  for (const VideoDecodeBackend b : hardware) {
    CHECK(b != VideoDecodeBackend::Auto && b != VideoDecodeBackend::Software);
  }
  CHECK(HevcDecoder::hardware_backends() == hardware);  // probed once

  HevcDecoder::Options options;
  auto decoder = HevcDecoder::create(options);
  CHECK(decoder.ok());
  const VideoDecodeBackend chosen = decoder->backend();
  CHECK(chosen ==
        (hardware.empty() ? VideoDecodeBackend::Software : hardware.front()));
  auto pictures = decode_clip(decoder.value());
  CHECK(pictures.ok());
  if (check_clip_shape(pictures.value()) != 0) return 1;
  CHECK(decoder->backend() == chosen);  // the clip is plain 4:2:0 Main
  std::printf("  auto: %s\n", sensor::to_string(chosen));

  const char* required = std::getenv("VR_TEST_HEVC_BACKEND");
  if (required != nullptr && *required != '\0') {
    if (std::strcmp(required, sensor::to_string(chosen)) != 0) {
      std::fprintf(stderr, "VR_TEST_HEVC_BACKEND=%s but auto chose %s\n",
                   required, sensor::to_string(chosen));
      CHECK(false);
    }
  }
  return 0;
}

int test_refusals() {
  HevcDecoder::Options options;
#if defined(__APPLE__)
  options.backend = VideoDecodeBackend::Cuda;
#else
  options.backend = VideoDecodeBackend::VideoToolbox;
#endif
  auto missing = HevcDecoder::create(options);
  CHECK(missing.status().domain() == vr::Status::Code::Unsupported);
  CHECK(missing.status().message().find("available: software") !=
        std::string::npos);

  options.backend = VideoDecodeBackend::Software;
  options.threads = -1;
  CHECK(HevcDecoder::create(options).status().domain() ==
        vr::Status::Code::InvalidArgument);
  options.threads = 0;

  auto decoder = HevcDecoder::create(options);
  CHECK(decoder.ok());
  auto nothing = decoder->receive();  // before any input
  CHECK(nothing.ok() && !nothing.value());
  CHECK(decoder->send(nullptr, 16, 0).domain() ==
        vr::Status::Code::InvalidArgument);
  CHECK(decoder->send(nullptr, 0, 0).ok());
  const std::uint8_t byte = 0;
  CHECK(decoder->send(&byte, 1, 0).domain() ==
        vr::Status::Code::InvalidArgument);  // after the end
  auto drained = decoder->receive();
  CHECK(drained.ok() && !drained.value());
  return 0;
}

int test_moves() {
  HevcDecoder::Options options;
  options.backend = VideoDecodeBackend::Software;
  auto created = HevcDecoder::create(options);
  CHECK(created.ok());
  HevcDecoder a = std::move(created).value();

  HevcDecoder b(std::move(a));
  CHECK(a.backend() == VideoDecodeBackend::Auto);  // NOLINT: moved from
  CHECK(a.send(nullptr, 0, 0).domain() == vr::Status::Code::InvalidArgument);
  CHECK(a.receive().status().domain() == vr::Status::Code::InvalidArgument);
  CHECK(b.backend() == VideoDecodeBackend::Software);

  auto other = HevcDecoder::create(options);
  CHECK(other.ok());
  HevcDecoder c = std::move(other).value();
  c = std::move(b);                                // over a live decoder
  CHECK(b.backend() == VideoDecodeBackend::Auto);  // NOLINT: moved from

  HevcDecoder* alias = &c;
  c = std::move(*alias);  // self-move
  CHECK(c.backend() == VideoDecodeBackend::Software);
  auto pictures = decode_clip(c);
  CHECK(pictures.ok());
  return check_clip_shape(pictures.value());
}

int test_names() {
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::Auto), "auto") == 0);
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::Cuda), "cuda") == 0);
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::VideoToolbox),
                    "videotoolbox") == 0);
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::Vaapi), "vaapi") ==
        0);
  return 0;
}

}  // namespace

int main() {
  if (access_units().size() != static_cast<std::size_t>(kFrames)) {
    std::fprintf(stderr, "FAIL: cannot split %s into %d access units\n",
                 VR_HEVC_FIXTURE, kFrames);
    return 1;
  }
  if (test_names() != 0) return 1;
  if (test_software_yuv() != 0) return 1;
  if (test_software_rgb() != 0) return 1;
  if (test_hardware_matches_software() != 0) return 1;
  if (test_auto_choice() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_moves() != 0) return 1;
  std::puts("sensor_video_hevc: OK");
  return 0;
}
