// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC encoder on this machine's hardware: the encoder rules
// (testing/hevc_conformance.hpp, C1-C13b) at the public sizes, at either
// range, BT.601 and BT.709, sitings 0 and 1, QP 1 and 51 and a key QP of its
// own, and on every code in the colour set, through pixel buffers and
// through targets a kernel writes on the device; the pictures it codes
// decoded back and read on the host, PSNR at QP 22 and the pattern's 0 and
// 255 and index, one target written only after a helper thread lets its
// batch run; a device without hevc_encoder_device_requirements() refused;
// video_backend(); and on Apple color_codes_of on every string Core Video
// names. A build with no encoder refuses in create. On every device: the
// test writer (testing/nv12_writer.hpp) stores a picture into plane images
// as it is, and refuses what it cannot write.
//
// Where no hardware path opens the encoder part skips; VS_TEST_HEVC_ENCODER,
// which CI sets on the legs that promise one (VideoToolbox on macOS), makes
// it fail instead.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "device_picture_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/write_image.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/testing/hevc_clip.hpp"
#include "volumetric_kit/sensor/testing/hevc_conformance.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/testing/nv12_writer.hpp"
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

// How a case's pictures go in.
enum class In { PixelBuffers, Targets };

// The rules on one case: 0 when kept.
int breach(const char* what, testing::HevcEncoderCase c, Gpu& gpu,
           In in = In::PixelBuffers) {
  c.device = &gpu.device;
  c.allocator = &gpu.allocator;
  c.pixel_buffers = in == In::PixelBuffers;
  const char* way = c.pixel_buffers ? "pixel buffers" : "targets";
  const auto broken = testing::hevc_encoder_breach(c);
  if (!broken.ok()) {
    std::fprintf(stderr, "FAIL %s, %s: %s\n", what, way,
                 broken.status().message().c_str());
    return 1;
  }
  if (!broken->empty()) {
    std::fprintf(stderr, "FAIL %s, %s: %s\n", what, way, broken->c_str());
    return 1;
  }
  std::printf("rules kept: %s, %s\n", what, way);
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

// The same rules through targets the device writes: the public sizes, both
// ranges and matrices, and a siting VideoToolbox takes from the picture's
// tag.
int test_target_cases(Gpu& gpu) {
  const In t = In::Targets;
  if (breach("512x512 BT.709 full", make(512, 512, kBt709Full), gpu, t) != 0) {
    return 1;
  }
  auto narrow = make(1922, 1082, kBt709Video);
  narrow.options.key_qp = 26;
  if (breach("1922x1082 BT.709 video, key QP 26", narrow, gpu, t) != 0) {
    return 1;
  }
  auto tall = make(3840, 4096, kBt709Full, 30);
  tall.pictures = 8;
  if (breach("3840x4096 BT.709 full", tall, gpu, t) != 0) return 1;
  if (breach("1920x1440 BT.601 full", make(1920, 1440, kBt601Full), gpu, t) !=
      0) {
    return 1;
  }
  ColorCodes centred = kBt601Video;
  centred.chroma_location = 1;
  if (breach("1920x1440 BT.601 video, sited at the centre",
             make(1920, 1440, centred), gpu, t) != 0) {
    return 1;
  }
  return 0;
}

#if defined(__APPLE__)
// A device without what the target's writer needs opens no session on it:
// one that declares Metal's import and not the storage images' formats, and
// one that declares neither.
int test_device_refused(Gpu& gpu) {
  HevcEncoder::Options o;
  o.width = 512;
  o.height = 512;
  o.color = kBt709Full;
  static const char* const kMetal[] = {"VK_EXT_metal_objects"};
  auto metal_only = vs_test::adopted_device(gpu, kMetal, 1);
  CHECK(metal_only.ok());
  o.device = &*metal_only;
  o.allocator = &gpu.allocator;
  auto refused = HevcEncoder::create(o);
  CHECK(!refused.ok() &&
        refused.status().domain() == vkc::Status::Code::Unsupported);
  CHECK(refused.status().message().find("shaderStorageImageExtendedFormats") !=
        std::string::npos);
  auto bare = vs_test::bare_device(gpu);
  CHECK(bare.ok());
  o.device = &*bare;
  refused = HevcEncoder::create(o);
  CHECK(!refused.ok() &&
        refused.status().domain() == vkc::Status::Code::Unsupported);
  // The test's own device has it all.
  o.device = &gpu.device;
  CHECK(HevcEncoder::create(o).ok());
  return 0;
}
#endif

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
// A thread joined as it goes, so a failed check never leaves one running.
struct Helper {
  std::thread thread;
  Helper() = default;
  Helper(const Helper&) = delete;
  Helper& operator=(const Helper&) = delete;
  ~Helper() {
    if (thread.joinable()) thread.join();
  }
};

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
// Through targets, the device writes each picture (testing::Nv12Writer), and
// picture kGated's batch runs only once a helper thread sets a value, after
// its send has returned: it decodes to what that batch wrote, so the coding
// thread coded it only once it was written.
int test_quality(Gpu& gpu, const ColorCodes& color, In in) {
  constexpr std::uint32_t kW = 1920;
  constexpr std::uint32_t kH = 1440;
  constexpr std::uint32_t kPictures = 6;
  constexpr std::uint32_t kGated = 3;
  const bool targets = in == In::Targets;
  HevcEncoder::Options o;
  o.width = kW;
  o.height = kH;
  o.qp = 22;
  o.color = color;
  if (targets) {
    o.device = &gpu.device;
    o.allocator = &gpu.allocator;
  }
  // Before the encoder, so they outlive it: its coding thread waits on
  // `written`, and the batches on `gate`.
  auto writer = testing::Nv12Writer::create(gpu.device);
  CHECK(writer.ok());
  auto written = vkc::TimelineSemaphore::create(gpu.device);
  auto gate = vkc::TimelineSemaphore::create(gpu.device);
  CHECK(written.ok() && gate.ok());
  std::vector<vkc::PendingBatch> pending(kPictures);
  Helper helper;  // sets `gate`: joined before the batches go
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
    if (targets) {
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      auto target = encoder->target(batch);
      CHECK(target.ok());
      CHECK(writer->record_upload(batch, gpu.allocator, want, *target).ok());
      std::vector<vkc::TimelinePoint> waits;
      if (i == kGated) waits.push_back({&*gate, 1});
      auto submitted = batch.submit_async(waits, {{&*written, i + 1}});
      CHECK(submitted.ok());
      pending[i] = std::move(submitted).value();
      if (i == kGated) {
        helper.thread = std::thread([&gate] {
          std::this_thread::sleep_for(std::chrono::milliseconds(300));
          static_cast<void>(gate->signal(1));
        });
      }
      CHECK(encoder->send(i == 0, i, {&*written, i + 1}).ok());
      if (i == kGated) {
        auto at_send = written->value();
        CHECK(at_send.ok() && *at_send < i + 1);  // not yet written
      }
    } else {
      auto pixels = testing::pixel_buffer(want, color);
      CHECK(pixels.ok());
      const vkc::Status sent =
          sensor::send_pixel_buffer(*encoder, *pixels, i == 0, i);
      CVPixelBufferRelease(*pixels);
      CHECK(sent.ok());
    }
    auto got = encoder->receive(coded);
    if (helper.thread.joinable()) helper.thread.join();
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
  std::printf("QP 22, %s range, %s: worst PSNR Y %.2f, Cb %.2f, Cr %.2f dB\n",
              color.full_range ? "full" : "video",
              targets ? "targets" : "pixel buffers", worst[0], worst[1],
              worst[2]);
  CHECK(worst[0] >= 45 && worst[1] >= 45 && worst[2] >= 45);
  return 0;
}
#endif

// A plane image the writer stores into, or not, as @p usage says: 0xEE
// throughout, in GENERAL.
vkc::Result<std::shared_ptr<const vkc::Image>> plane_image(
    Gpu& gpu, std::uint32_t columns, std::uint32_t rows, VkFormat format,
    std::uint32_t texel, VkImageUsageFlags usage) {
  vkc::ImageDesc desc;
  desc.extent = {columns, rows};
  desc.format = format;
  desc.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VKC_ASSIGN(vkc::Image image, gpu.allocator.create_image(desc));
  const std::vector<std::uint8_t> filled(std::size_t{columns} * rows * texel,
                                         0xEE);
  VKC_TRY(vkc::test::write_image(gpu.device, gpu.allocator, image,
                                 filled.data(), filled.size(),
                                 VK_IMAGE_LAYOUT_GENERAL));
  return std::make_shared<const vkc::Image>(std::move(image));
}

// NV12 plane images of @p width x @p height, @p pad columns and rows larger.
vkc::Result<sensor::YuvImage> planes(Gpu& gpu, std::uint32_t width,
                                     std::uint32_t height, std::uint32_t pad,
                                     VkImageUsageFlags usage) {
  sensor::YuvImage yuv;
  yuv.layout = sensor::YuvLayout::Nv12;
  yuv.width = width;
  yuv.height = height;
  VKC_ASSIGN(yuv.image[0], plane_image(gpu, width + pad, height + pad,
                                       VK_FORMAT_R8_UNORM, 1, usage));
  VKC_ASSIGN(yuv.image[1],
             plane_image(gpu, (width + 1) / 2 + pad, (height + 1) / 2 + pad,
                         VK_FORMAT_R8G8_UNORM, 2, usage));
  return yuv;
}

// The test writer, on any device: a picture stored into plane images reads
// back as it is, at even and odd sizes, into images larger than the picture
// too, from their corner; what it cannot write is refused, and a device
// without the storage images' formats has no writer.
int test_writer(Gpu& gpu) {
  auto writer = testing::Nv12Writer::create(gpu.device);
  CHECK(writer.ok());
  const struct {
    std::uint32_t width, height, pad;
  } sizes[] = {{256, 144, 0}, {255, 143, 3}, {1922, 1082, 0}};
  for (const auto& size : sizes) {
    const testing::HostNv12 picture =
        testing::test_pattern(size.width, size.height, 5);
    auto yuv = planes(gpu, size.width, size.height, size.pad,
                      VK_IMAGE_USAGE_STORAGE_BIT);
    CHECK(yuv.ok());
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(writer->record_upload(batch, gpu.allocator, picture, *yuv).ok());
    CHECK(batch.submit().ok());
    std::vector<std::uint8_t> back[3];
    vs_test::read_device_picture(*yuv, gpu.device, gpu.allocator, back);
    CHECK(back[0] == picture.luma);
    CHECK(back[1].size() * 2 == picture.chroma.size());
    for (std::size_t k = 0; k < back[1].size(); ++k) {
      CHECK(back[1][k] == picture.chroma[2 * k]);
      CHECK(back[2][k] == picture.chroma[2 * k + 1]);
    }
  }
  const testing::HostNv12 picture = testing::test_pattern(64, 32, 1);
  const auto refused = [&](const sensor::YuvImage& target,
                           const testing::HostNv12& p) {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    return writer->record_upload(batch, gpu.allocator, p, target).domain() ==
           vkc::Status::Code::InvalidArgument;
  };
  auto read_only = planes(gpu, 64, 32, 0, 0);
  CHECK(read_only.ok());
  CHECK(refused(*read_only, picture));  // no STORAGE usage
  auto stored = planes(gpu, 64, 32, 0, VK_IMAGE_USAGE_STORAGE_BIT);
  CHECK(stored.ok());
  CHECK(refused(*stored, testing::test_pattern(62, 32, 1)));  // another size
  sensor::YuvImage swapped = *stored;
  std::swap(swapped.image[0], swapped.image[1]);
  CHECK(refused(swapped, picture));  // R8G8 as the luma
  sensor::YuvImage larger = *stored;
  larger.width = 66;
  CHECK(refused(larger, testing::test_pattern(66, 32, 1)));  // too small
  {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    auto few = testing::Nv12Writer::upload(batch, gpu.allocator,
                                           testing::test_pattern(32, 32, 1));
    CHECK(few.ok());
    CHECK(writer->record(batch, *few, *stored).domain() ==
          vkc::Status::Code::InvalidArgument);  // planes of a smaller picture
  }
  auto bare = vs_test::bare_device(gpu);
  CHECK(bare.ok());
  CHECK(testing::Nv12Writer::create(*bare).status().domain() ==
        vkc::Status::Code::Unsupported);
  std::puts("the test writer stores pictures as they are");
  return 0;
}

int gpu_main(Gpu& gpu) {
  if (test_writer(gpu) != 0) return 1;
#if !defined(__APPLE__)
  // No encoder off Apple yet: the writer was all there is to run.
  if (!testing::hevc_encoder_promised()) {
    std::puts("sensor_video_hevc_encoder: OK (no encoder in this build)");
    return 0;
  }
#endif
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
  if (test_target_cases(gpu) != 0) return 1;
  if (test_code_set(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_device_refused(gpu) != 0) return 1;
  for (const In in : {In::PixelBuffers, In::Targets}) {
    if (test_quality(gpu, kBt601Full, in) != 0) return 1;
    if (test_quality(gpu, kBt601Video, in) != 0) return 1;
  }
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
#endif
  return vkc::test::run_on_gpu(sensor::hevc_encoder_device_requirements(),
                               gpu_main);
}
