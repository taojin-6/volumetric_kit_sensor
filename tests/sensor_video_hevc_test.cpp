// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC decoder on committed clips (tools/make_hevc_fixtures.sh), decoded
// on this machine's hardware onto the device, as NV12's two plane images, and
// read back: the pattern the clip was made from, libavcodec's pictures to the
// bit, the colour an unlabelled stream takes, a reset mid-stream and after
// the end, a display window off the coded picture's corner, a held picture
// never reused, frame prep from the images as from the same samples in a
// buffer, a stream the hardware refuses, the decoders create refuses, the
// images shared across queue families, NVDEC's decoders sharing the GPU's
// primary CUDA context, NVDEC's pictures changing size within one decoder,
// the argument checks, and moves. On VideoToolbox also:
// pictures under NVDEC's smallest to the bit at either range and cropped,
// parameter sets that change mid-stream, Main 10 refused, and a skipped
// access unit reported.
//
// Where no device path opens the test skips; VR_TEST_HEVC_BACKEND, which CI
// sets on the legs that promise one (NVDEC on the NVIDIA containers,
// VideoToolbox on macOS), makes it fail instead.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "device_picture_readback.hpp"
#include "hevc_fixture.hpp"
#include "md5.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"
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
// libavcodec's MD5s of the patch clip and, on NVDEC, the cropped one
// (tools/make_hevc_fixtures.sh prints them).
constexpr const char* kPatchesMd5 = "63171f05c92c231ab96a389291ba67fe";
#if !defined(__APPLE__)
constexpr const char* kCroppedMd5 = "9f217594cf6e432f9aa4021534446fbe";
#endif
#if defined(__APPLE__)
constexpr const char* kTestsrc2 = VR_HEVC_DATA "/testsrc2_128x64.h265";
constexpr const char* kTestsrc2Full = VR_HEVC_DATA "/testsrc2_full_128x64.h265";
constexpr const char* kTestsrc2Cropped =
    VR_HEVC_DATA "/testsrc2_cropped_120x56.h265";
constexpr const char* kTestsrc2Main10 =
    VR_HEVC_DATA "/testsrc2_main10_128x64.h265";
// libavcodec's MD5s of the first two (tools/make_hevc_fixtures.sh prints them).
constexpr const char* kTestsrc2Md5 = "e743e90ae47cb276844495b286c98b0f";
constexpr const char* kTestsrc2FullMd5 = "4fe199a7b1bcc80dd1aec3f87b6a9af0";
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

using Gpu = vkc::test::GpuContext;

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

// On the device as NV12's two plane images, R8 and R8G8: VideoToolbox's, or
// ones CUDA wrote, of the picture's size and in GENERAL, which a reader takes
// over from outside Vulkan.
int check_on_device(const sensor::DecodedPicture& p) {
  const sensor::YuvImage& yuv = p.yuv;
  CHECK(yuv.layout == sensor::YuvLayout::Nv12);
  CHECK(yuv.image[0] != nullptr && yuv.image[1] != nullptr &&
        yuv.device == nullptr);
  CHECK(yuv.image[0]->format() == VK_FORMAT_R8_UNORM &&
        yuv.image[1]->format() == VK_FORMAT_R8G8_UNORM);
#if defined(__APPLE__)
  CHECK(yuv.queue_family == sensor::kQueueFamilyIgnored);
#else
  CHECK(yuv.queue_family == sensor::kQueueFamilyExternal);
  CHECK(yuv.image[0]->width() == yuv.width &&
        yuv.image[0]->height() == yuv.height);
  CHECK(yuv.image[1]->width() == (yuv.width + 1) / 2 &&
        yuv.image[1]->height() == (yuv.height + 1) / 2);
  for (const std::shared_ptr<const vkc::Image>& plane : yuv.image) {
    CHECK(plane->layout() == VK_IMAGE_LAYOUT_GENERAL);
    CHECK((plane->usage() & VK_IMAGE_USAGE_SAMPLED_BIT) != 0 &&
          (plane->usage() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0 &&
          plane->view() != VK_NULL_HANDLE);
  }
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
#endif

// The clip decoded whole, every picture on the device and of the size given,
// and its pictures' planes, in display order, to the MD5 given.
int check_md5(const Decoded& decoded, const char* clip, std::uint32_t width,
              std::uint32_t height, const char* md5) {
  if (!decoded.status.ok()) {
    std::fprintf(stderr, "%s: %s\n", clip, decoded.status.message().c_str());
  }
  CHECK(decoded.status.ok());
  CHECK(decoded.pictures.size() == static_cast<std::size_t>(kFrames));
  vr_test::Md5 sum;
  for (const Picture& p : decoded.pictures) {
    if (check_on_device(p.meta) != 0) return 1;
    CHECK(p.meta.yuv.width == width && p.meta.yuv.height == height);
    for (const std::vector<std::uint8_t>& plane : p.planes) {
      sum.update(plane.data(), plane.size());
    }
  }
  const std::string got = sum.hex();
  if (got != md5) std::fprintf(stderr, "%s: MD5 %s\n", clip, got.c_str());
  CHECK(got == md5);
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

// Each clip's pictures, in display order, read back from where the hardware
// left them, are libavcodec's to the bit (the MD5s
// tools/make_hevc_fixtures.sh prints): the patch clip, and its full-range
// copy, whose samples are the same, on either path. On NVDEC the cropped
// clip too, its display window 16 in from the left and the top. On
// VideoToolbox testsrc2 with B-frames, under NVDEC's smallest picture: with
// no colour description the clip is video range, labelled full range it is
// full range, each written at its own range; the cropped one is its display
// window, 8 short at the right and the bottom.
int test_reference_md5(const Gpu& gpu) {
  const struct {
    const char* clip;
    const char* md5;
    std::uint32_t width;
    std::uint32_t height;
    sensor::VideoColorMatrix matrix;  // BT.601 unlabelled, by the height
    bool full_range;
  } clips[] = {
      {kPatches, kPatchesMd5, kWidth, kHeight, sensor::VideoColorMatrix::Bt709,
       false},
      {kFullRange, kPatchesMd5, kWidth, kHeight,
       sensor::VideoColorMatrix::Bt709, true},
#if defined(__APPLE__)
      {kTestsrc2, kTestsrc2Md5, 128, 64, sensor::VideoColorMatrix::Bt601,
       false},
      {kTestsrc2Full, kTestsrc2FullMd5, 128, 64,
       sensor::VideoColorMatrix::Bt709, true},
      {kTestsrc2Cropped, "4f676ba80eb5db26c2156fde29f72a11", 120, 56,
       sensor::VideoColorMatrix::Bt601, false},
#else
      {kCropped, kCroppedMd5, kWidth - 16, kHeight - 16,
       sensor::VideoColorMatrix::Bt709, false},
#endif
  };
  for (const auto& c : clips) {
    const Decoded decoded = decode(gpu, c.clip);
    if (check_md5(decoded, c.clip, c.width, c.height, c.md5) != 0) return 1;
    for (const Picture& p : decoded.pictures) {
      CHECK(yuv_reference::coded_in(p.meta.yuv, c.matrix, c.full_range));
    }
  }
  return 0;
}

#if !defined(__APPLE__)
// One NVDEC decoder hands out pictures of two sizes, the cropped clip's and
// the patch clip's, a reset() before each change: every picture's images are
// its own size, and its samples libavcodec's. A set of images is reused only
// for a picture of its size: the cropped pictures held across the changes
// keep their samples, and a set let go before a picture of the other size is
// never handed out with it, larger or smaller.
int test_size_change(const Gpu& gpu) {
  const AccessUnits small = vr_test::hevc_access_units(kCropped);
  const AccessUnits large = vr_test::hevc_access_units(kPatches);
  constexpr std::uint32_t kSmallWidth = kWidth - 16;
  constexpr std::uint32_t kSmallHeight = kHeight - 16;
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const Decoded held = decode_clip(decoder.value(), small, gpu);
  if (check_md5(held, kCropped, kSmallWidth, kSmallHeight, kCroppedMd5) != 0)
    return 1;
  // Each clip's pictures are let go once checked, so the sets they leave are
  // free, and of the size the next clip does not have.
  for (int round = 0; round < 2; ++round) {
    CHECK(decoder->reset().ok());
    if (check_md5(decode_clip(decoder.value(), large, gpu), kPatches, kWidth,
                  kHeight, kPatchesMd5) != 0)
      return 1;
    CHECK(decoder->reset().ok());
    if (check_md5(decode_clip(decoder.value(), small, gpu), kCropped,
                  kSmallWidth, kSmallHeight, kCroppedMd5) != 0)
      return 1;
  }
  for (const Picture& p : held.pictures) {
    sensor::YuvImage again = p.meta.yuv;
    std::vector<std::uint8_t> planes[3];
    vr_test::read_device_picture(again, gpu.device, gpu.allocator, planes);
    for (int i = 0; i < 3; ++i) CHECK(planes[i] == p.planes[i]);
  }
  return 0;
}
#endif

// A picture held is never reused: the clip decoded with every picture kept,
// each has images of its own, and the first, read again after the last was
// decoded, is still its own. On NVDEC a picture let go is reused: the clip
// decoded with each picture dropped once read takes one set of images.
int test_held_pictures(const Gpu& gpu) {
  const Decoded held = decode(gpu, kPatches);
  if (check_clip_shape(held) != 0) return 1;
  for (int p = 0; p < 2; ++p) {
    std::vector<VkImage> images;
    for (const Picture& picture : held.pictures) {
      images.push_back(picture.meta.yuv.image[p]->handle());
    }
    std::sort(images.begin(), images.end());
    CHECK(std::unique(images.begin(), images.end()) == images.end());
  }
  sensor::YuvImage first = held.pictures[0].meta.yuv;
  std::vector<std::uint8_t> again[3];
  vr_test::read_device_picture(first, gpu.device, gpu.allocator, again);
  for (int i = 0; i < 3; ++i) CHECK(again[i] == held.pictures[0].planes[i]);
#if !defined(__APPLE__)
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const AccessUnits units = vr_test::hevc_access_units(kPatches);
  std::vector<VkImage> luma;
  for (std::size_t i = 0; i <= units.size(); ++i) {
    const bool end = i == units.size();
    CHECK(decoder
              ->send(end ? nullptr : units[i].data(), end ? 0 : units[i].size(),
                     pts_of(static_cast<int>(i)))
              .ok());
    for (;;) {
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!picture.value()) break;
      luma.push_back(picture.value()->yuv.image[0]->handle());
    }
  }
  CHECK(luma.size() == static_cast<std::size_t>(kFrames));
  CHECK(std::count(luma.begin(), luma.end(), luma[0]) ==
        static_cast<std::ptrdiff_t>(luma.size()));
#endif
  return 0;
}

// GpuFramePrep reads each decoded picture where the hardware left it, taking
// NVDEC's over from outside Vulkan, through a lens, to the words the same
// samples give from NV12 planes in a buffer, bit for bit; and a readback
// leaves NVDEC's EXCLUSIVE images named as its own family's.
int test_frame_prep(const Gpu& gpu) {
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  CHECK(prep.ok());
  vr::camera::CameraModel cam;
  cam.size = {kWidth, kHeight};
  cam.intrinsics = {200.0f, 198.0f, 127.3f, 71.6f};
  cam.distortion = {-0.28f, 0.09f, 0.0012f, -0.0009f,
                    -0.01f, 0.05f, 0.0f,    0.0f};
  const std::size_t pixels = std::size_t{kWidth} * kHeight;
  const std::vector<std::uint16_t> depth(pixels, 1500);
  const auto words = [&](const sensor::YuvImage& color) {
    sensor::RgbdFrame f;
    f.depth = depth.data();
    f.depth_camera = cam;
    f.min_depth = 0.1f;
    f.max_depth = 10.0f;
    f.color = color;
    f.color_camera = cam;
    std::vector<std::uint32_t> out;
    auto prepared = prep->prepare(f);
    if (!prepared) {
      std::fprintf(stderr, "%s\n", prepared.status().message().c_str());
      return out;
    }
    auto read = vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                    *prepared->color, pixels);
    if (read) out = std::move(read).value();
    return out;
  };
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const AccessUnits units = vr_test::hevc_access_units(kPatches);
  int prepared = 0;
  for (std::size_t i = 0; i <= units.size(); ++i) {
    const bool end = i == units.size();
    CHECK(decoder
              ->send(end ? nullptr : units[i].data(), end ? 0 : units[i].size(),
                     pts_of(static_cast<int>(i)))
              .ok());
    for (;;) {
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!picture.value()) break;
      sensor::YuvImage& yuv = picture.value()->yuv;
      const std::vector<std::uint32_t> in_place = words(yuv);
      CHECK(in_place.size() == pixels);
      CHECK(std::count_if(in_place.begin(), in_place.end(),
                          [](std::uint32_t w) { return (w >> 24) == 0xFFu; }) >
            static_cast<std::ptrdiff_t>(pixels / 2));
      std::vector<std::uint8_t> planes[3];
      vr_test::read_device_picture(yuv, gpu.device, gpu.allocator, planes);
      CHECK(planes[0].size() == pixels);
#if defined(__APPLE__)
      CHECK(yuv.queue_family == sensor::kQueueFamilyIgnored);
#else
      CHECK(yuv.queue_family == gpu.device.queue_family());
#endif
      // NV12 in a buffer: Y, then Cb and Cr interleaved.
      std::vector<std::uint8_t> bytes(planes[0]);
      const std::size_t chroma_at = bytes.size();
      for (std::size_t k = 0; k < planes[1].size(); ++k) {
        bytes.push_back(planes[1][k]);
        bytes.push_back(planes[2][k]);
      }
      auto buffer = vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                                    bytes.data(), bytes.size());
      CHECK(buffer.ok());
      sensor::YuvImage copied = yuv;
      copied.image[0] = copied.image[1] = nullptr;
      copied.device =
          std::make_shared<const vkc::Buffer>(std::move(buffer).value());
      copied.offset[0] = 0;
      copied.offset[1] = chroma_at;
      copied.stride[0] = copied.stride[1] = kWidth;
      copied.queue_family = sensor::kQueueFamilyIgnored;
      CHECK(words(copied) == in_place);
      ++prepared;
    }
  }
  CHECK(prepared == kFrames);
  return 0;
}

// Options::queue_families reach the images: CONCURRENT across two families,
// every picture as the clip was made, and after a readback needing no
// transfer. A list the decoder refuses fails create.
int test_queue_families(const Gpu& gpu) {
  HevcDecoder::Options options = decoder_options(gpu);
  options.queue_family_count = 1;  // and no list
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const std::uint32_t lacking[] = {
      static_cast<std::uint32_t>(gpu.device.caps().queue_families().size())};
  options.queue_families = lacking;
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const std::optional<std::uint32_t> other =
      vkc::test::other_queue_family(gpu.device);
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
  sensor::YuvImage read = decoded.pictures[0].meta.yuv;
  std::vector<std::uint8_t> planes[3];
  vr_test::read_device_picture(read, gpu.device, gpu.allocator, planes);
  CHECK(planes[0] == decoded.pictures[0].planes[0]);
  CHECK(read.queue_family == sensor::kQueueFamilyIgnored);
  return 0;
}

#if defined(__APPLE__)
// Parameter sets that change mid-stream, at each sequence's IDR, sent to one
// decoder: the patch clip, then labelled full range (its range alone
// changes), then again; then testsrc2 (another size) and testsrc2 at full
// range. Each picture comes out with its own sequence's size and range, in
// display order, its samples the stream's own: the patch clip's pictures,
// whichever range is declared, are the first part's to the bit, and the
// testsrc2 ones libavcodec's.
int test_parameter_change(const Gpu& gpu) {
  const struct {
    const char* clip;
    const char* md5;  // null for the patch clip's
    std::uint32_t width;
    std::uint32_t height;
    sensor::VideoColorMatrix matrix;
    bool full_range;
  } parts[] = {
      {kPatches, nullptr, kWidth, kHeight, sensor::VideoColorMatrix::Bt709,
       false},
      {kFullRange, nullptr, kWidth, kHeight, sensor::VideoColorMatrix::Bt709,
       true},
      {kPatches, nullptr, kWidth, kHeight, sensor::VideoColorMatrix::Bt709,
       false},
      {kTestsrc2, kTestsrc2Md5, 128, 64, sensor::VideoColorMatrix::Bt601,
       false},
      {kTestsrc2Full, kTestsrc2FullMd5, 128, 64,
       sensor::VideoColorMatrix::Bt709, true},
  };
  AccessUnits units;
  for (const auto& part : parts) {
    const AccessUnits more = vr_test::hevc_access_units(part.clip);
    CHECK(more.size() == static_cast<std::size_t>(kFrames));
    units.insert(units.end(), more.begin(), more.end());
  }
  auto decoder = HevcDecoder::create(decoder_options(gpu));
  CHECK(decoder.ok());
  const Decoded decoded = decode_clip(decoder.value(), units, gpu);
  if (!decoded.status.ok()) {
    std::fprintf(stderr, "%s\n", decoded.status.message().c_str());
  }
  CHECK(decoded.status.ok());
  CHECK(decoded.pictures.size() == units.size());
  std::size_t next = 0;
  for (const auto& part : parts) {
    vr_test::Md5 md5;
    for (int f = 0; f < kFrames; ++f, ++next) {
      const Picture& p = decoded.pictures[next];
      if (check_on_device(p.meta) != 0) return 1;
      CHECK(p.meta.yuv.width == part.width && p.meta.yuv.height == part.height);
      CHECK(yuv_reference::coded_in(p.meta.yuv, part.matrix, part.full_range));
      for (const std::vector<std::uint8_t>& plane : p.planes) {
        md5.update(plane.data(), plane.size());
      }
      if (part.md5 == nullptr) {
        CHECK(p.meta.pts == pts_of(static_cast<int>(next)));
        const std::size_t y = 36 * kWidth + 16;  // patch (0, 0)'s centre
        CHECK(std::abs(p.planes[0][y] - patch_y(patch(0, 0, f))) <= 2);
        const Picture& first = decoded.pictures[static_cast<std::size_t>(f)];
        for (int i = 0; i < 3; ++i) CHECK(p.planes[i] == first.planes[i]);
      }
    }
    if (part.md5 != nullptr) CHECK(md5.hex() == part.md5);
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
  if (test_reference_md5(gpu) != 0) return 1;
#if !defined(__APPLE__)
  if (test_size_change(gpu) != 0) return 1;
#endif
  if (test_held_pictures(gpu) != 0) return 1;
  if (test_frame_prep(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_malformed_cropped_sps(gpu) != 0) return 1;
  if (test_parameter_change(gpu) != 0) return 1;
  if (test_main10(gpu) != 0) return 1;
  if (test_skipped_access_unit(gpu) != 0) return 1;
#endif
  if (test_refused(gpu) != 0) return 1;
  if (test_create_refusals(gpu) != 0) return 1;
  if (test_queue_families(gpu) != 0) return 1;
#if !defined(__APPLE__)
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
  if (test_md5() != 0) return 1;
  if (vr_test::hevc_access_units(kPatches).size() !=
      static_cast<std::size_t>(kFrames)) {
    std::fprintf(stderr, "FAIL: cannot split %s into %d access units\n",
                 kPatches, kFrames);
    return 1;
  }
  return vkc::test::run_on_gpu(vr::device_requirements(), gpu_main);
}
