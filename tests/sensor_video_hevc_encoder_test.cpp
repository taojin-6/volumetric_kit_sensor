// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC encoder on this machine's hardware: the encoder rules
// (testing/hevc_conformance.hpp, C1-C13) at the public sizes, at either
// range, BT.601 and BT.709, sitings 0 and 1, QP 1 and 51 and a key QP of its
// own, and on every code in the colour set; the pictures it codes decoded
// back and read on the host, PSNR at QP 22 and the pattern's 0 and 255 and
// index; video_backend(); and on Apple color_codes_of on every string Core
// Video names. A build with no encoder refuses in create.
//
// Where no hardware path opens the GPU part skips; VS_TEST_HEVC_ENCODER,
// which CI sets on the legs that promise one (VideoToolbox on macOS), makes
// it fail instead.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "device_picture_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/testing/hevc_clip.hpp"
#include "volumetric_kit/sensor/testing/hevc_conformance.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/video/backend.hpp"
#include "volumetric_kit/sensor/video/hevc_decoder.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>

#include "volumetric_kit/sensor/apple/core_video_color.hpp"
#include "volumetric_kit/sensor/testing/nv12_pixel_buffer.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"
#endif

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;
using sensor::ColorCodes;
using sensor::HevcEncoder;

namespace {

using Gpu = vkc::test::GpuContext;

int no_encoder(const std::string& why) {
  const bool required = testing::hevc_encoder_promised();
  std::fprintf(stderr, "no hardware HEVC encoder (%s); %s\n", why.c_str(),
               required ? "failing" : "skipping");
  return required ? 1 : vkc::test::kSkipExitCode;
}

int test_backend() {
#if defined(__APPLE__)
  CHECK(sensor::video_backend() == sensor::VideoBackend::VideoToolbox);
#else
  CHECK(sensor::video_backend() == sensor::VideoBackend::Nvidia);
  // No encoder here yet: create says so, whatever the options.
  HevcEncoder::Options o;
  o.width = 512;
  o.height = 512;
  o.color = {1, 1, 1, true, 0};
  const auto made = HevcEncoder::create(o);
  CHECK(!made.ok() &&
        made.status().domain() == vkc::Status::Code::Unsupported &&
        made.status().message().find("no HEVC encoder") != std::string::npos);
#endif
  return 0;
}

#if defined(__APPLE__)
// color_codes_of on every string Core Video names for codes 0 to 23, the six
// sitings, both ranges, an untagged buffer, and the refusals.
int test_color_codes_of() {
  const ColorCodes untagged_full{2, 2, 2, true, 0};
  auto made =
      testing::pixel_buffer(testing::test_pattern(64, 32, 0), untagged_full);
  CHECK(made.ok());
  CVPixelBufferRef p = *made;
  const auto codes = sensor::color_codes_of(p);
  CHECK(codes.ok() && codes->matrix == 2 && codes->transfer == 2 &&
        codes->primaries == 2 && codes->full_range &&
        codes->chroma_location == 0);
  for (int code = 0; code <= 23; ++code) {
    // Each string is taken just before it is used: Core Video makes a "#n"
    // name on the fly and keeps no reference to it.
    const struct {
      CFStringRef key;
      CFStringRef (*name)(int);
      int ColorCodes::* field;
    } tags[3] = {
        {kCVImageBufferYCbCrMatrixKey,
         &CVYCbCrMatrixGetStringForIntegerCodePoint, &ColorCodes::matrix},
        {kCVImageBufferTransferFunctionKey,
         &CVTransferFunctionGetStringForIntegerCodePoint,
         &ColorCodes::transfer},
        {kCVImageBufferColorPrimariesKey,
         &CVColorPrimariesGetStringForIntegerCodePoint,
         &ColorCodes::primaries}};
    for (const auto& t : tags) {
      const CFStringRef value = t.name(code);
      if (value == nullptr) continue;  // 0 and 2 have none
      CVBufferSetAttachment(p, t.key, value, kCVAttachmentMode_ShouldPropagate);
      const auto read = sensor::color_codes_of(p);
      // An IOSurface buffer keeps a tag as its code, and Core Video names
      // some codes back as another string (matrix 11 as primaries' 'DCI_P3'
      // on macOS 27): the codes are what Core Video's own mapping makes of
      // what it keeps, and a string it maps to none is refused.
      CFTypeRef kept = CVBufferCopyAttachment(p, t.key, nullptr);
      if (kept == nullptr) {
        // Not kept at all (matrix 15): the buffer is untagged.
        std::printf("Core Video keeps no tag for code %d\n", code);
        CHECK(read.ok() && (*read).*t.field == 2);
        continue;
      }
      CHECK(CFGetTypeID(kept) == CFStringGetTypeID());
      const bool same = CFEqual(kept, value);
      int back = 0;
      if (t.field == &ColorCodes::matrix) {
        back = CVYCbCrMatrixGetIntegerCodePointForString(
            static_cast<CFStringRef>(kept));
      } else if (t.field == &ColorCodes::transfer) {
        back = CVTransferFunctionGetIntegerCodePointForString(
            static_cast<CFStringRef>(kept));
      } else {
        back = CVColorPrimariesGetIntegerCodePointForString(
            static_cast<CFStringRef>(kept));
      }
      CFRelease(kept);
      if (!same) {
        std::printf("Core Video keeps code %d's tag as another string, %d\n",
                    code, back);
        if (back == 2) {
          CHECK(!read.ok() &&
                read.status().domain() == vkc::Status::Code::InvalidArgument);
        } else {
          CHECK(read.ok() && (*read).*t.field == back);
        }
        CVBufferRemoveAttachment(p, t.key);
        continue;
      }
      if (!read.ok()) {
        std::fprintf(stderr, "code %d: %s\n", code,
                     read.status().message().c_str());
      }
      CHECK(read.ok());
      // Core Video names transfer 6, 14 and 15 with BT.709's string.
      const bool bt709 = t.field == &ColorCodes::transfer &&
                         (code == 6 || code == 14 || code == 15);
      CHECK((*read).*t.field == (bt709 ? 1 : code));
      CVBufferRemoveAttachment(p, t.key);
    }
  }
  const CFStringRef sitings[6] = {kCVImageBufferChromaLocation_Left,
                                  kCVImageBufferChromaLocation_Center,
                                  kCVImageBufferChromaLocation_TopLeft,
                                  kCVImageBufferChromaLocation_Top,
                                  kCVImageBufferChromaLocation_BottomLeft,
                                  kCVImageBufferChromaLocation_Bottom};
  for (int code = 0; code < 6; ++code) {
    CVBufferSetAttachment(p, kCVImageBufferChromaLocationTopFieldKey,
                          sitings[code], kCVAttachmentMode_ShouldPropagate);
    const auto read = sensor::color_codes_of(p);
    CHECK(read.ok() && read->chroma_location == code);
  }
  CVPixelBufferRelease(p);
  // Unknown strings are refused, naming the tag: on a buffer of no
  // IOSurface, which keeps any string it is given.
  CVPixelBufferRef plain = nullptr;
  CHECK(CVPixelBufferCreate(kCFAllocatorDefault, 64, 32,
                            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                            nullptr, &plain) == kCVReturnSuccess);
  CVBufferSetAttachment(plain, kCVImageBufferChromaLocationTopFieldKey,
                        CFSTR("Sideways"), kCVAttachmentMode_ShouldPropagate);
  auto refused = sensor::color_codes_of(plain);
  CHECK(!refused.ok() &&
        refused.status().domain() == vkc::Status::Code::InvalidArgument &&
        refused.status().message().find("Sideways") != std::string::npos);
  CVBufferRemoveAttachment(plain, kCVImageBufferChromaLocationTopFieldKey);
  CVBufferSetAttachment(plain, kCVImageBufferTransferFunctionKey,
                        kCVImageBufferTransferFunction_UseGamma,
                        kCVAttachmentMode_ShouldPropagate);
  refused = sensor::color_codes_of(plain);
  CHECK(!refused.ok() &&
        refused.status().domain() == vkc::Status::Code::InvalidArgument &&
        refused.status().message().find("transfer") != std::string::npos);
  CVBufferRemoveAttachment(plain, kCVImageBufferTransferFunctionKey);
  CVBufferSetAttachment(plain, kCVImageBufferYCbCrMatrixKey,
                        CFSTR("NotAMatrix"), kCVAttachmentMode_ShouldPropagate);
  refused = sensor::color_codes_of(plain);
  CHECK(!refused.ok() &&
        refused.status().message().find("NotAMatrix") != std::string::npos);
  CVPixelBufferRelease(plain);
  // Video range from '420v'; no buffer.
  made = testing::pixel_buffer(testing::test_pattern(64, 32, 0),
                               {6, 1, 6, false, 0});
  CHECK(made.ok());
  const auto video = sensor::color_codes_of(*made);
  CHECK(video.ok() && !video->full_range && video->matrix == 6 &&
        video->transfer == 1 && video->primaries == 6);
  CVPixelBufferRelease(*made);
  CHECK(sensor::color_codes_of(nullptr).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  return 0;
}
#endif

// The rules on one case: 0 when kept.
int breach(const char* what, testing::HevcEncoderCase c, Gpu& gpu) {
  c.device = &gpu.device;
  c.allocator = &gpu.allocator;
  c.pixel_buffers = true;
  const auto broken = testing::hevc_encoder_breach(c);
  if (!broken.ok()) {
    std::fprintf(stderr, "FAIL %s: %s\n", what,
                 broken.status().message().c_str());
    return 1;
  }
  if (!broken->empty()) {
    std::fprintf(stderr, "FAIL %s: %s\n", what, broken->c_str());
    return 1;
  }
  std::printf("rules kept: %s\n", what);
  return 0;
}

testing::HevcEncoderCase make(std::uint32_t w, std::uint32_t h,
                              const ColorCodes& color, int qp = 22) {
  testing::HevcEncoderCase c;
  c.options.width = w;
  c.options.height = h;
  c.options.qp = qp;
  c.options.color = color;
  return c;
}

const ColorCodes kBt709Full{1, 1, 1, true, 0};
const ColorCodes kBt709Video{1, 1, 1, false, 0};
const ColorCodes kBt601Full{6, 1, 6, true, 0};
const ColorCodes kBt601Video{6, 1, 6, false, 0};

// The public sizes and colours (512 square, a 4n+2 width, the strip
// boundary's rows, a camera's 1920x1440), QP at both ends of what
// VideoToolbox codes, and a key QP of its own.
int test_cases(Gpu& gpu) {
  if (breach("512x512 BT.709 full", make(512, 512, kBt709Full), gpu) != 0) {
    return 1;
  }
  auto narrow = make(1922, 1082, kBt709Video);
  narrow.options.key_qp = 26;
  if (breach("1922x1082 BT.709 video, key QP 26", narrow, gpu) != 0) return 1;
  auto tall = make(3840, 4096, kBt709Full, 30);
  tall.pictures = 8;
  if (breach("3840x4096 BT.709 full", tall, gpu) != 0) return 1;
  if (breach("1920x1440 BT.601 full", make(1920, 1440, kBt601Full), gpu) != 0) {
    return 1;
  }
  if (breach("1920x1440 BT.601 video", make(1920, 1440, kBt601Video), gpu) !=
      0) {
    return 1;
  }
  ColorCodes centred = kBt709Full;
  centred.chroma_location = 1;
  if (breach("1920x1440 BT.709 full, sited at the centre",
             make(1920, 1440, centred), gpu) != 0) {
    return 1;
  }
  auto low = make(512, 512, kBt709Full, 1);
  low.options.key_qp = 51;
  if (breach("512x512 QP 1, key QP 51", low, gpu) != 0) return 1;
  auto high = make(512, 512, kBt709Full, 51);
  high.options.key_qp = 1;
  if (breach("512x512 QP 51, key QP 1", high, gpu) != 0) return 1;
  return 0;
}

// Every code in the set, normalization included, and every siting
// VideoToolbox writes: each a short run.
int test_code_set(Gpu& gpu) {
  std::vector<ColorCodes> all;
  for (const int m : {1, 5, 6, 7, 9}) all.push_back({m, 1, 1, true, 0});
  for (const int t : {1, 2, 6, 7, 8, 13, 14, 15, 16, 17, 18}) {
    all.push_back({1, t, 1, true, 0});
  }
  for (const int p : {1, 2, 5, 6, 9, 11, 12, 22}) {
    all.push_back({1, 1, p, false, 0});
  }
  for (int s = 1; s <= 5; ++s) all.push_back({1, 1, 1, true, s});
  for (const ColorCodes& c : all) {
    auto run = make(512, 512, c);
    run.pictures = 4;
    const std::string what = "matrix " + std::to_string(c.matrix) +
                             " transfer " + std::to_string(c.transfer) +
                             " primaries " + std::to_string(c.primaries) +
                             " siting " + std::to_string(c.chroma_location);
    if (breach(what.c_str(), run, gpu) != 0) return 1;
  }
  return 0;
}

#if defined(__APPLE__)
double psnr(const std::vector<std::uint8_t>& a,
            const std::vector<std::uint8_t>& b) {
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  const double mse = sum / static_cast<double>(a.size());
  return mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

// The pattern coded at QP 22 and decoded back: PSNR at least 45 dB in each
// plane, square 0's 255 and square 1's 0 exact, and each picture's index.
int test_quality(Gpu& gpu, const ColorCodes& color) {
  constexpr std::uint32_t kW = 1920;
  constexpr std::uint32_t kH = 1440;
  constexpr std::uint32_t kPictures = 6;
  HevcEncoder::Options o;
  o.width = kW;
  o.height = kH;
  o.qp = 22;
  o.color = color;
  auto encoder = HevcEncoder::create(o);
  CHECK(encoder.ok());
  sensor::HevcDecoder::Options d;
  d.device = &gpu.device;
  d.allocator = &gpu.allocator;
  auto decoder = sensor::HevcDecoder::create(d);
  CHECK(decoder.ok());
  HevcEncoder::CodedPicture coded;
  double worst[3] = {99, 99, 99};
  std::uint32_t out = 0;
  for (std::uint32_t i = 0; i < kPictures; ++i) {
    const testing::HostNv12 want = testing::test_pattern(kW, kH, 40 + i);
    auto pixels = testing::pixel_buffer(want, color);
    CHECK(pixels.ok());
    const vkc::Status sent =
        sensor::send_pixel_buffer(*encoder, *pixels, i == 0, i);
    CVPixelBufferRelease(*pixels);
    CHECK(sent.ok());
    auto got = encoder->receive(coded);
    CHECK(got.ok() && *got);
    CHECK(decoder
              ->send(coded.access_unit.data(), coded.access_unit.size(),
                     coded.pts)
              .ok());
    for (;;) {
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!*picture) break;
      std::vector<std::uint8_t> planes[3];
      vs_test::read_device_picture((*picture)->yuv, gpu.device, gpu.allocator,
                                   planes);
      CHECK(planes[0].size() == std::size_t{kW} * kH);
      const testing::HostNv12 ref = testing::test_pattern(kW, kH, 40 + out);
      std::vector<std::uint8_t> cb, cr;
      for (std::size_t k = 0; k < ref.chroma.size(); k += 2) {
        cb.push_back(ref.chroma[k]);
        cr.push_back(ref.chroma[k + 1]);
      }
      const double p[3] = {psnr(planes[0], ref.luma), psnr(planes[1], cb),
                           psnr(planes[2], cr)};
      for (int k = 0; k < 3; ++k) worst[k] = std::min(worst[k], p[k]);
      for (std::uint32_t y = 0; y < testing::kIndexSquare; ++y) {
        for (std::uint32_t x = 0; x < testing::kIndexSquare; ++x) {
          CHECK(planes[0][std::size_t{y} * kW + x] == 255);
          CHECK(planes[0][std::size_t{y} * kW + testing::kIndexSquare + x] ==
                0);
        }
      }
      CHECK(testing::pattern_index(planes[0].data(), kW) == 40 + out);
      ++out;
    }
  }
  CHECK(out == kPictures);
  std::printf("QP 22, %s range: worst PSNR Y %.2f, Cb %.2f, Cr %.2f dB\n",
              color.full_range ? "full" : "video", worst[0], worst[1],
              worst[2]);
  CHECK(worst[0] >= 45 && worst[1] >= 45 && worst[2] >= 45);
  return 0;
}
#endif

int gpu_main(Gpu& gpu) {
  HevcEncoder::Options probe;
  probe.width = 512;
  probe.height = 512;
  probe.color = kBt709Full;
  if (auto made = HevcEncoder::create(probe); !made) {
    if (made.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", made.status().message().c_str());
      return 1;
    }
    return no_encoder(made.status().message());
  }
  if (test_cases(gpu) != 0) return 1;
  if (test_code_set(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_quality(gpu, kBt601Full) != 0) return 1;
  if (test_quality(gpu, kBt601Video) != 0) return 1;
#endif
  std::puts("sensor_video_hevc_encoder: OK");
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (test_backend() != 0) return 1;
#if defined(__APPLE__)
  if (test_color_codes_of() != 0) return 1;
#else
  // No encoder off Apple yet: nothing to run on the GPU.
  if (!testing::hevc_encoder_promised()) {
    std::puts("sensor_video_hevc_encoder: OK (no encoder in this build)");
    return 0;
  }
#endif
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
