// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC decoder on committed clips (tools/make_hevc_fixtures.sh), decoded
// on this machine's hardware onto the device and read back: the pattern the
// clip was made from, the colour an unlabelled stream takes, a reset
// mid-stream and after the end, a display window off the coded picture's
// corner, a stream the hardware refuses, the decoders create refuses,
// VideoToolbox's images shared across queue families, NVDEC's decoders
// sharing the GPU's primary CUDA context, the argument checks, and moves. On
// VideoToolbox also: libavcodec's pictures to the bit at either range and
// cropped, Main 10 refused, and a skipped access unit reported.
//
// Where no device path opens the test skips; VR_TEST_HEVC_BACKEND, which CI
// sets on the legs that promise one (NVDEC on the NVIDIA containers,
// VideoToolbox on macOS), makes it fail instead.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "device_picture_readback.hpp"
#include "gpu_test.hpp"
#include "hevc_fixture.hpp"
#include "md5.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"
#include "yuv_reference.hpp"

#if !defined(__APPLE__)
#include <cuda.h>
#include <dlfcn.h>
#endif

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::HevcDecoder;

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 144;
constexpr int kFrames = 8;
constexpr const char* kPatches = VR_HEVC_DATA "/patches_256x144.h265";
constexpr const char* kCropped = VR_HEVC_DATA "/cropped_240x128.h265";
constexpr const char* kRefused = VR_HEVC_DATA "/refused_256x144.h265";
constexpr const char* kUnlabelled = VR_HEVC_DATA "/unlabelled_256x144.h265";
constexpr const char* kFullRange = VR_HEVC_DATA "/fullrange_256x144.h265";
#if defined(__APPLE__)
constexpr const char* kTestsrc2 = VR_HEVC_DATA "/testsrc2_128x64.h265";
constexpr const char* kTestsrc2Full = VR_HEVC_DATA "/testsrc2_full_128x64.h265";
constexpr const char* kTestsrc2Cropped =
    VR_HEVC_DATA "/testsrc2_cropped_120x56.h265";
constexpr const char* kTestsrc2Main10 =
    VR_HEVC_DATA "/testsrc2_main10_128x64.h265";
#endif

std::int64_t pts_of(int frame) { return 1000 + 33 * frame; }

// The clip's pattern: patch p = (column + frame + 3 * row) mod 8 over 32x72
// luma patches; Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch(int column, int row, int frame) {
  return (column + frame + 3 * row) % 8;
}
int patch_y(int p) { return 40 + 24 * p; }
int patch_u(int p) { return 64 + 16 * ((3 * p) % 8); }
int patch_v(int p) { return 64 + 16 * ((5 * p) % 8); }

using Gpu = vr_test::GpuContext;

// The device and allocator the decoders hand their pictures out on.
HevcDecoder::Options decoder_options(const Gpu& gpu) {
  HevcDecoder::Options o;
  o.device = &gpu.device;
  o.allocator = &gpu.allocator;
  return o;
}

// A picture read back from the device, planes packed.
struct Picture {
  sensor::DecodedPicture meta;
  std::vector<std::uint8_t> planes[3];
};

using AccessUnits = std::vector<std::vector<std::uint8_t>>;

// The pictures a clip decoded to, each read back as soon as it is ready, and
// the first error, after which the pictures still waiting are taken too.
struct Decoded {
  std::vector<Picture> pictures;
  vkc::Status status;
};

Decoded decode_clip(HevcDecoder& decoder, const AccessUnits& units,
                    const Gpu& gpu) {
  Decoded out;
  const auto drain = [&]() -> vkc::Status {
    for (;;) {
      VKC_ASSIGN(auto picture, decoder.receive());
      if (!picture) return {};
      Picture p;
      p.meta = *picture;
      vr_test::read_device_picture(picture->yuv, gpu.device, gpu.allocator,
                                   p.planes);
      if (p.planes[0].size() !=
          std::size_t{picture->yuv.width} * picture->yuv.height) {
        return vkc::Status::io_error("reading a picture back");
      }
      out.pictures.push_back(std::move(p));
    }
  };
  for (std::size_t i = 0; i <= units.size() && out.status.ok(); ++i) {
    const bool end = i == units.size();
    out.status =
        decoder.send(end ? nullptr : units[i].data(), end ? 0 : units[i].size(),
                     pts_of(static_cast<int>(i)));
    const vkc::Status drained = drain();
    if (out.status.ok()) out.status = drained;
  }
  return out;
}

Decoded decode(
    const Gpu& gpu, const char* clip,
    std::optional<sensor::VideoColorDescription> unlabelled = std::nullopt) {
  HevcDecoder::Options options = decoder_options(gpu);
  options.unlabelled_color = unlabelled;
  auto decoder = HevcDecoder::create(options);
  if (!decoder) return {{}, decoder.status()};
  return decode_clip(decoder.value(), vr_test::hevc_access_units(clip), gpu);
}

// On the device, where this platform's hardware leaves it: VideoToolbox's
// images, or a buffer CUDA wrote, taken over from outside Vulkan.
int check_on_device(const sensor::DecodedPicture& p) {
  CHECK(p.yuv.layout == sensor::YuvLayout::Nv12);
#if defined(__APPLE__)
  CHECK(p.yuv.image[0] != nullptr && p.yuv.image[1] != nullptr &&
        p.yuv.device == nullptr);
  CHECK(p.yuv.queue_family == sensor::kQueueFamilyIgnored);
#else
  CHECK(p.yuv.device != nullptr && p.yuv.image[0] == nullptr);
  CHECK(p.yuv.queue_family == sensor::kQueueFamilyExternal);
#endif
  return 0;
}

int check_clip_shape(const Decoded& decoded) {
  if (!decoded.status.ok()) {
    std::fprintf(stderr, "%s\n", decoded.status.message().c_str());
  }
  CHECK(decoded.status.ok());
  CHECK(decoded.pictures.size() == static_cast<std::size_t>(kFrames));
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.pictures[static_cast<std::size_t>(f)];
    const auto& meta = pic.meta;
    if (check_on_device(meta) != 0) return 1;
    CHECK(meta.yuv.width == kWidth && meta.yuv.height == kHeight);
    CHECK(meta.pts == pts_of(f));
    CHECK(yuv_reference::coded_in(meta.yuv, sensor::VideoColorMatrix::Bt709,
                                  false));
    CHECK(meta.yuv.chroma_location == sensor::ChromaLocation::Left);
    CHECK(meta.encoding.has_value() &&
          meta.encoding->transfer == vr::ColorEncoding::Transfer::Bt709 &&
          meta.encoding->primaries == vr::ColorEncoding::Primaries::Bt709);
    CHECK(pic.planes[0].size() == static_cast<std::size_t>(kWidth * kHeight));
  }
  return 0;
}

// Every patch's interior against the pattern (qp 4, so within a code or two),
// which also pins the order and pts of the pictures.
int test_pattern(const Gpu& gpu) {
  const Decoded decoded = decode(gpu, kPatches);
  if (check_clip_shape(decoded) != 0) return 1;
  int worst = 0;
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.pictures[static_cast<std::size_t>(f)];
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

// Options::unlabelled_color: the unlabelled clip takes it, reported on each
// picture, the bytes those decoded without it. The labelled clip keeps its
// BT.709 limited, and one labelled full range decodes as such.
int test_unlabelled_color(const Gpu& gpu) {
  const sensor::VideoColorDescription femto{sensor::VideoColorMatrix::Bt601,
                                            true};
  const Decoded plain = decode(gpu, kUnlabelled);
  const Decoded taken = decode(gpu, kUnlabelled, femto);
  CHECK(plain.status.ok() && taken.status.ok());
  CHECK(taken.pictures.size() == static_cast<std::size_t>(kFrames));
  CHECK(taken.pictures.size() == plain.pictures.size());
  for (std::size_t f = 0; f < taken.pictures.size(); ++f) {
    const Picture& p = taken.pictures[f];
    CHECK(yuv_reference::coded_in(p.meta.yuv, sensor::VideoColorMatrix::Bt601,
                                  true));
    for (int i = 0; i < 3; ++i)
      CHECK(p.planes[i] == plain.pictures[f].planes[i]);
  }
  const Decoded patches = decode(gpu, kPatches);
  const Decoded full = decode(gpu, kFullRange);
  CHECK(patches.status.ok() && full.status.ok());
  CHECK(full.pictures.size() == static_cast<std::size_t>(kFrames));
  CHECK(full.pictures.size() == patches.pictures.size());
  for (std::size_t f = 0; f < full.pictures.size(); ++f) {
    const Picture& p = full.pictures[f];
    CHECK(yuv_reference::coded_in(p.meta.yuv, sensor::VideoColorMatrix::Bt709,
                                  true));
    for (int i = 0; i < 3; ++i)
      CHECK(p.planes[i] == patches.pictures[f].planes[i]);
  }
  return check_clip_shape(decode(gpu, kPatches, femto));
}

// reset(): the pictures held for display are dropped, a stream ended with
// size 0 takes data again, and the next key frame decodes as the first.
int test_reset(const Gpu& gpu) {
  const AccessUnits b_frames = vr_test::hevc_access_units(kRefused);
  const AccessUnits patches = vr_test::hevc_access_units(kPatches);
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  int out = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(decoder
              ->send(b_frames[i].data(), b_frames[i].size(),
                     pts_of(static_cast<int>(i)))
              .ok());
    for (;;) {
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!picture.value()) break;
      ++out;
    }
  }
  CHECK(out < 4);  // B-frames: some are held for display
  CHECK(decoder->reset().ok());
  auto dropped = decoder->receive();
  CHECK(dropped.ok() && !dropped.value());
  CHECK(decoder->send(nullptr, 0, 0).ok());
  CHECK(decoder->send(patches[0].data(), patches[0].size(), 0).domain() ==
        vkc::Status::Code::InvalidArgument);  // after the end
  CHECK(decoder->reset().ok());
  return check_clip_shape(decode_clip(decoder.value(), patches, gpu));
}

// The cropped clip is the uncropped one's pixels, 16 right and 16 down, from
// NVDEC. VideoToolbox cannot crop there (it fills a picture of the display
// size from the coded corner), so it refuses the stream, and a reset does not
// undo the refusal.
int test_cropped(const Gpu& gpu) {
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const Decoded cropped =
      decode_clip(decoder.value(), vr_test::hevc_access_units(kCropped), gpu);
#if defined(__APPLE__)
  CHECK(cropped.pictures.empty());
  CHECK(cropped.status.domain() == vkc::Status::Code::Unsupported);
  CHECK(decoder->reset().ok());
  const AccessUnits units = vr_test::hevc_access_units(kCropped);
  CHECK(decoder->send(units[0].data(), units[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
#else
  constexpr int kCropLeft = 16;  // the cropped clip's window
  constexpr int kCropTop = 16;
  CHECK(cropped.status.ok());
  const Decoded full = decode(gpu, kPatches);
  CHECK(full.status.ok());
  CHECK(cropped.pictures.size() == full.pictures.size());
  const int width = kWidth - kCropLeft;
  const int height = kHeight - kCropTop;
  for (std::size_t f = 0; f < cropped.pictures.size(); ++f) {
    const Picture& c = cropped.pictures[f];
    const Picture& u = full.pictures[f];
    if (check_on_device(c.meta) != 0) return 1;
    CHECK(c.meta.yuv.width == static_cast<std::uint32_t>(width));
    CHECK(c.meta.yuv.height == static_cast<std::uint32_t>(height));
    CHECK(c.meta.pts == u.meta.pts);
    for (int i = 0; i < 3; ++i) {
      const int shift = i == 0 ? 0 : 1;
      const int w = width >> shift;
      for (int y = 0; y < height >> shift; ++y) {
        for (int x = 0; x < w; ++x) {
          const int want = u.planes[i][static_cast<std::size_t>(
              ((y + (kCropTop >> shift)) * (kWidth >> shift)) + x +
              (kCropLeft >> shift))];
          CHECK(c.planes[i][static_cast<std::size_t>(y * w + x)] == want);
        }
      }
    }
  }
#endif
  return 0;
}

#if defined(__APPLE__)
// The crop prefix is not proof of a valid SPS. Truncate the committed SPS
// after its crop offsets but before the remaining syntax, retaining its VPS.
// Check recovery both with and without reset, and with the rest of the access
// unit still present. No rejected bytes may latch a permanent refusal.
int test_malformed_cropped_sps(const Gpu& gpu) {
  const AccessUnits cropped = vr_test::hevc_access_units(kCropped);
  const AccessUnits patches = vr_test::hevc_access_units(kPatches);
  CHECK(!cropped.empty());
  const auto& unit = cropped[0];
  std::size_t sps = 0;
  std::size_t end = 0;
  for (std::size_t i = 0; i + 3 < unit.size(); ++i) {
    if (unit[i] != 0 || unit[i + 1] != 0 || unit[i + 2] != 1) continue;
    if (sps != 0) {
      end = i;
      break;
    }
    if (((unit[i + 3] >> 1) & 0x3f) == 33) sps = i + 3;
  }
  CHECK(sps != 0 && end > sps + 30);
  for (const std::size_t kept : {25u, 30u}) {
    for (const bool slices : {false, true}) {
      auto damaged = std::vector<std::uint8_t>(
          unit.begin(), unit.begin() + static_cast<std::ptrdiff_t>(sps + kept));
      if (slices)
        damaged.insert(damaged.end(),
                       unit.begin() + static_cast<std::ptrdiff_t>(end),
                       unit.end());
      for (const bool reset : {false, true}) {
        auto decoder = HevcDecoder::create(decoder_options(gpu));
        CHECK(decoder.ok());
        CHECK(decoder->send(damaged.data(), damaged.size(), 0).domain() ==
              vkc::Status::Code::IoError);
        auto waiting = decoder->receive();
        CHECK(waiting.ok() && !waiting.value());
        if (reset) CHECK(decoder->reset().ok());
        CHECK(check_clip_shape(decode_clip(*decoder, patches, gpu)) == 0);
      }
    }
  }

  // A complete cropped SPS sent without a slice still refuses the stream;
  // resetting must not turn a genuine refusal into an accepted crop.
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  CHECK(decoder->send(unit.data(), end, 0).domain() ==
        vkc::Status::Code::Unsupported);
  CHECK(decoder->reset().ok());
  CHECK(decoder->send(patches[0].data(), patches[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
  return 0;
}

// The MD5 implementation, on RFC 1321's own examples.
int test_md5() {
  const auto md5 = [](const std::string& text) {
    vr_test::Md5 m;
    m.update(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    return m.hex();
  };
  CHECK(md5("") == "d41d8cd98f00b204e9800998ecf8427e");
  CHECK(md5("abc") == "900150983cd24fb0d6963f7d28e17f72");
  CHECK(md5("12345678901234567890123456789012345678901234567890123456789012"
            "345678901234567890") == "57edf4a22be3c955ac49da2e2107b67a");
  return 0;
}

// testsrc2 with B-frames, decoded by VideoToolbox: each clip's pictures, in
// display order, are libavcodec's to the bit (the MD5s
// tools/make_hevc_fixtures.sh prints). With no colour description the clip is
// video range, labelled full range it is full range, each written at its own
// range; the cropped one is its display window, 8 short at the right and the
// bottom.
int test_reference_md5(const Gpu& gpu) {
  const struct {
    const char* clip;
    const char* md5;
    std::uint32_t width;
    std::uint32_t height;
    sensor::VideoColorMatrix matrix;  // BT.601 unlabelled, by the height
    bool full_range;
  } clips[] = {
      {kTestsrc2, "e743e90ae47cb276844495b286c98b0f", 128, 64,
       sensor::VideoColorMatrix::Bt601, false},
      {kTestsrc2Full, "4fe199a7b1bcc80dd1aec3f87b6a9af0", 128, 64,
       sensor::VideoColorMatrix::Bt709, true},
      {kTestsrc2Cropped, "4f676ba80eb5db26c2156fde29f72a11", 120, 56,
       sensor::VideoColorMatrix::Bt601, false},
  };
  for (const auto& c : clips) {
    const Decoded decoded = decode(gpu, c.clip);
    if (!decoded.status.ok()) {
      std::fprintf(stderr, "%s: %s\n", c.clip,
                   decoded.status.message().c_str());
    }
    CHECK(decoded.status.ok());
    CHECK(decoded.pictures.size() == static_cast<std::size_t>(kFrames));
    vr_test::Md5 md5;
    for (const Picture& p : decoded.pictures) {
      if (check_on_device(p.meta) != 0) return 1;
      CHECK(p.meta.yuv.width == c.width && p.meta.yuv.height == c.height);
      CHECK(yuv_reference::coded_in(p.meta.yuv, c.matrix, c.full_range));
      for (const std::vector<std::uint8_t>& plane : p.planes) {
        md5.update(plane.data(), plane.size());
      }
    }
    const std::string got = md5.hex();
    if (got != c.md5) std::fprintf(stderr, "%s: MD5 %s\n", c.clip, got.c_str());
    CHECK(got == c.md5);
  }
  return 0;
}

// Main 10 is refused before any picture comes out, and stays refused.
int test_main10(const Gpu& gpu) {
  const AccessUnits units = vr_test::hevc_access_units(kTestsrc2Main10);
  CHECK(units.size() == static_cast<std::size_t>(kFrames));
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const Decoded decoded = decode_clip(decoder.value(), units, gpu);
  CHECK(decoded.pictures.empty());
  CHECK(decoded.status.domain() == vkc::Status::Code::Unsupported);
  CHECK(decoder->reset().ok());
  CHECK(decoder->send(units[0].data(), units[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
  return 0;
}

// An access unit skipped (frame 1 of the patch clip, IPPP with a key frame
// every 4): the two predicted from it are reported, IoError, rather than
// decoded from a picture the decoder never had, and decoding picks up at the
// key frame, as the clip was made. A decoder that starts at frame 2 reports
// it too.
int test_skipped_access_unit(const Gpu& gpu) {
  const AccessUnits units = vr_test::hevc_access_units(kPatches);
  for (const int first : {0, 2}) {
    auto decoder = HevcDecoder::create(decoder_options(gpu));
    CHECK(decoder.ok());
    std::vector<int> failed;
    std::vector<Picture> shown;
    for (int i = first; i <= kFrames; ++i) {
      if (i == 1) continue;
      const bool end = i == kFrames;
      const vkc::Status sent = decoder->send(
          end ? nullptr : units[static_cast<std::size_t>(i)].data(),
          end ? 0 : units[static_cast<std::size_t>(i)].size(), pts_of(i));
      if (!sent.ok()) {
        CHECK(sent.domain() == vkc::Status::Code::IoError);
        failed.push_back(i);
      }
      for (;;) {
        auto picture = decoder->receive();
        CHECK(picture.ok());
        if (!picture.value()) break;
        Picture p;
        p.meta = *picture.value();
        vr_test::read_device_picture(p.meta.yuv, gpu.device, gpu.allocator,
                                     p.planes);
        shown.push_back(std::move(p));
      }
    }
    CHECK((failed == std::vector<int>{2, 3}));
    const std::vector<int> want_shown = first == 0
                                            ? std::vector<int>{0, 4, 5, 6, 7}
                                            : std::vector<int>{4, 5, 6, 7};
    CHECK(shown.size() == want_shown.size());
    for (std::size_t k = 0; k < shown.size(); ++k) {
      const int f = want_shown[k];
      CHECK(shown[k].meta.pts == pts_of(f));
      CHECK(shown[k].planes[0].size() ==
            static_cast<std::size_t>(kWidth * kHeight));
      const std::size_t y = 36 * kWidth + 16;  // patch (0, 0)'s centre
      CHECK(std::abs(shown[k].planes[0][y] - patch_y(patch(0, 0, f))) <= 2);
    }
  }
  return 0;
}
#endif

// The refused clip: 8 frames with B-frames, then 2 of 4:0:0 grey that no
// hardware path hands out. The pictures decoded before the grey come out, in
// display order, each a distinct pts sent before it; then the refusal, which
// stands after a reset.
int test_refused(const Gpu& gpu) {
  const AccessUnits units = vr_test::hevc_access_units(kRefused);
  CHECK(units.size() == static_cast<std::size_t>(kFrames + 2));
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const Decoded decoded = decode_clip(decoder.value(), units, gpu);
  CHECK(decoded.status.domain() == vkc::Status::Code::Unsupported);
  const auto& got = decoded.pictures;
  CHECK(!got.empty() && got.size() <= static_cast<std::size_t>(kFrames));
  std::vector<std::int64_t> out;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (check_on_device(got[i].meta) != 0) return 1;
    out.push_back(got[i].meta.pts);
    const std::size_t y = 36 * kWidth + 16;  // patch (0, 0)'s centre
    const int want = patch_y(patch(0, 0, static_cast<int>(i)));
    CHECK(std::abs(got[i].planes[0][y] - want) <= 2);
  }
  std::sort(out.begin(), out.end());
  CHECK(std::adjacent_find(out.begin(), out.end()) == out.end());
  CHECK(out.back() < pts_of(kFrames));
  CHECK(decoder->reset().ok());
  CHECK(decoder->send(units[0].data(), units[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
  return 0;
}

// No device, a device the hardware path cannot reach, and on NVDEC no
// allocator: each Unsupported.
int test_create_refusals(const Gpu& gpu) {
  CHECK(HevcDecoder::create({}).status().domain() ==
        vkc::Status::Code::Unsupported);
  vkc::Result<vkc::Device> bare = vr_test::bare_device(gpu);
  CHECK(bare.ok());
  HevcDecoder::Options options = decoder_options(gpu);
  options.device = &bare.value();
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::Unsupported);
#if !defined(__APPLE__)
  options = decoder_options(gpu);
  options.allocator = nullptr;
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::Unsupported);
#endif
  return 0;
}

#if !defined(__APPLE__)
// How many CUDA devices have their primary context active, as libcuda says;
// -1 where it cannot say.
int active_primary_contexts() {
  void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) return -1;
  const auto init = reinterpret_cast<decltype(&cuInit)>(dlsym(lib, "cuInit"));
  const auto count = reinterpret_cast<decltype(&cuDeviceGetCount)>(
      dlsym(lib, "cuDeviceGetCount"));
  const auto get =
      reinterpret_cast<decltype(&cuDeviceGet)>(dlsym(lib, "cuDeviceGet"));
  const auto state = reinterpret_cast<decltype(&cuDevicePrimaryCtxGetState)>(
      dlsym(lib, "cuDevicePrimaryCtxGetState"));
  int devices = 0;
  int active = -1;
  if (init != nullptr && count != nullptr && get != nullptr &&
      state != nullptr && init(0) == CUDA_SUCCESS &&
      count(&devices) == CUDA_SUCCESS) {
    active = 0;
    for (int i = 0; i < devices && active >= 0; ++i) {
      CUdevice device = 0;
      unsigned flags = 0;
      int on = 0;
      if (get(&device, i) != CUDA_SUCCESS ||
          state(device, &flags, &on) != CUDA_SUCCESS) {
        active = -1;
      } else {
        active += on;
      }
    }
  }
  dlclose(lib);
  return active;
}

// The decoders on a GPU share its primary CUDA context, each on a stream of
// its own, and let it go with the last of them: none opens a context of its
// own.
int test_primary_context(const Gpu& gpu) {
  CHECK(active_primary_contexts() == 0);
  {
    auto a = HevcDecoder::create(decoder_options(gpu));
    auto b = HevcDecoder::create(decoder_options(gpu));
    CHECK(a.ok() && b.ok());
    CHECK(active_primary_contexts() == 1);
    const AccessUnits units = vr_test::hevc_access_units(kPatches);
    CHECK(check_clip_shape(decode_clip(a.value(), units, gpu)) == 0);
    CHECK(check_clip_shape(decode_clip(b.value(), units, gpu)) == 0);
  }
  CHECK(active_primary_contexts() == 0);
  return 0;
}
#endif

#if defined(__APPLE__)
// Options::queue_families reach VideoToolbox's images: CONCURRENT across two
// families, every picture as the clip was made. A list the importer refuses
// fails create.
int test_queue_families(const Gpu& gpu) {
  HevcDecoder::Options options = decoder_options(gpu);
  options.queue_family_count = 1;  // and no list
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const std::optional<std::uint32_t> other =
      vr_test::other_queue_family(gpu.device);
  if (!other) return 0;
  const std::uint32_t families[] = {gpu.device.queue_family(), *other};
  options.queue_families = families;
  options.queue_family_count = 2;
  auto decoder = HevcDecoder::create(options);
  CHECK(decoder.ok());
  const Decoded decoded =
      decode_clip(decoder.value(), vr_test::hevc_access_units(kPatches), gpu);
  if (check_clip_shape(decoded) != 0) return 1;
  for (const Picture& p : decoded.pictures) {
    CHECK(p.meta.yuv.image[0]->sharing_mode() == VK_SHARING_MODE_CONCURRENT);
    CHECK(p.meta.yuv.image[1]->sharing_mode() == VK_SHARING_MODE_CONCURRENT);
  }
  return 0;
}
#endif

int test_arguments(const Gpu& gpu) {
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  auto nothing = decoder->receive();  // before any input
  CHECK(nothing.ok() && !nothing.value());
  CHECK(decoder->send(nullptr, 16, 0).domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(decoder->send(nullptr, 0, 0).ok());
  const std::uint8_t byte = 0;
  CHECK(decoder->send(&byte, 1, 0).domain() ==
        vkc::Status::Code::InvalidArgument);  // after the end
  auto drained = decoder->receive();
  CHECK(drained.ok() && !drained.value());
  return 0;
}

int test_moves(const Gpu& gpu) {
  auto created = HevcDecoder::create(decoder_options(gpu));
  CHECK(created.ok());
  HevcDecoder a = std::move(created).value();

  HevcDecoder b(std::move(a));
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is tested
  CHECK(a.send(nullptr, 0, 0).domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.receive().status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.reset().domain() == vkc::Status::Code::InvalidArgument);

  auto other = HevcDecoder::create(decoder_options(gpu));
  CHECK(other.ok());
  HevcDecoder c = std::move(other).value();
  c = std::move(b);  // over a live decoder
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is tested
  CHECK(b.reset().domain() == vkc::Status::Code::InvalidArgument);

  HevcDecoder* alias = &c;
  c = std::move(*alias);  // self-move
  return check_clip_shape(
      decode_clip(c, vr_test::hevc_access_units(kPatches), gpu));
}

int gpu_main(Gpu& gpu) {
  if (auto probe = HevcDecoder::create(decoder_options(gpu)); !probe) {
    if (probe.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", probe.status().message().c_str());
      return 1;
    }
    return vr_test::no_decoder(probe.status().message());
  }

  if (test_pattern(gpu) != 0) return 1;
  if (test_unlabelled_color(gpu) != 0) return 1;
  if (test_reset(gpu) != 0) return 1;
  if (test_cropped(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_malformed_cropped_sps(gpu) != 0) return 1;
  if (test_reference_md5(gpu) != 0) return 1;
  if (test_main10(gpu) != 0) return 1;
  if (test_skipped_access_unit(gpu) != 0) return 1;
#endif
  if (test_refused(gpu) != 0) return 1;
  if (test_create_refusals(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_queue_families(gpu) != 0) return 1;
#else
  if (test_primary_context(gpu) != 0) return 1;
#endif
  if (test_arguments(gpu) != 0) return 1;
  if (test_moves(gpu) != 0) return 1;
  std::puts("sensor_video_hevc: OK");
  return 0;
}

}  // namespace

int main() {
  // Unbuffered, so a crash inside FFmpeg or a driver still shows which clip
  // it was on.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(__APPLE__)
  if (test_md5() != 0) return 1;
#endif
  if (vr_test::hevc_access_units(kPatches).size() !=
      static_cast<std::size_t>(kFrames)) {
    std::fprintf(stderr, "FAIL: cannot split %s into %d access units\n",
                 kPatches, kFrames);
    return 1;
  }
  return vr_test::run_on_gpu(gpu_main);
}
