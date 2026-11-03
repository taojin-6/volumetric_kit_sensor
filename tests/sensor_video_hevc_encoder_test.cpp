// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC encoder on this machine's hardware: the encoder rules
// (testing/hevc_conformance.hpp, C1-C13b) on small pictures, every rule once
// through pixel buffers and once through targets a kernel writes on the
// device, and the rules the options change at either range, BT.601 and
// BT.709, sitings 0 and 1, QP 1 and 51 and a key QP of its own, a size the
// coded picture crops, on every code in the colour set, and on one picture
// past level 5.1's size; the pictures it codes
// decoded back and read on the host, PSNR at QP 22 and the pattern's 0 and
// 255 and index, one target's batch held behind a value the test sets after
// its send; a target described by each matrix, range and siting; the images
// a batch still writes kept through the teardown and a reopen, and create's
// first targets waiting for no batch; a device without
// hevc_encoder_device_requirements() refused;
// video_backend(); and on Apple color_codes_of on every string Core Video
// names. A build with no encoder refuses in create. On every device: the
// test writer (testing/nv12_writer.hpp) stores a picture into plane images
// as it is, and refuses what it cannot write.
//
// The pictures are as small as each rule allows and no step sleeps a fixed
// time: each value the test holds is set once the step it holds for is
// seen, or once a wait of ten pictures' time found nothing, so the test
// takes seconds even under ThreadSanitizer. Full-size runs are benchmarks'
// and a device runner's (an app's -self-test).
//
// Where no hardware path opens the encoder part skips; VS_TEST_HEVC_ENCODER,
// which CI sets on the legs that promise one (VideoToolbox on macOS), makes
// it fail instead.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
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
  const bool all = c.rules == testing::HevcEncoderCase::Rules::All;
  const std::string name = std::to_string(c.options.width) + "x" +
                           std::to_string(c.options.height) + " " + what +
                           (c.pixel_buffers ? ", pixel buffers" : ", targets") +
                           (all ? ", every rule" : ", the stream's rules");
  const auto broken = testing::hevc_encoder_breach(c);
  if (!broken.ok()) {
    std::fprintf(stderr, "FAIL %s: %s\n", name.c_str(),
                 broken.status().message().c_str());
    return 1;
  }
  if (!broken->empty()) {
    std::fprintf(stderr, "FAIL %s: %s\n", name.c_str(), broken->c_str());
    return 1;
  }
  std::printf("rules kept: %s\n", name.c_str());
  return 0;
}

// The rules' pictures: as wide as the test pattern's index band (256
// columns), and the same 2 larger each way, a 4n+2 width and rows no
// multiple of 16, which the coded picture crops. A handful of pictures: two
// keys and P-pictures after a key and after a P-picture.
constexpr std::uint32_t kW = 256;
constexpr std::uint32_t kH = 144;
constexpr std::uint32_t kCroppedW = kW + 2;
constexpr std::uint32_t kCroppedH = kH + 2;
constexpr std::uint32_t kPictures = 6;

testing::HevcEncoderCase make(std::uint32_t w, std::uint32_t h,
                              const ColorCodes& color, int qp = 22) {
  testing::HevcEncoderCase c;
  c.options.width = w;
  c.options.height = h;
  c.options.qp = qp;
  c.options.color = color;
  c.pictures = kPictures;
  return c;
}

const ColorCodes kBt709Full{1, 1, 1, true, 0};
const ColorCodes kBt709Video{1, 1, 1, false, 0};
const ColorCodes kBt601Full{6, 1, 6, true, 0};
const ColorCodes kBt601Video{6, 1, 6, false, 0};

// The rules the stream checks, which the session's options change (C2-C6,
// C10, C11), on a case of its own.
testing::HevcEncoderCase stream_only(testing::HevcEncoderCase c) {
  c.rules = testing::HevcEncoderCase::Rules::Stream;
  return c;
}

// Every rule on a camera's colour, BT.601 full. Then the stream's rules,
// which the options change, at the other range, on BT.709 sited at the
// centre, at the cropped size with a key QP of its own, and at QP at both
// ends of what VideoToolbox codes. Then the one case whose size is the
// point: a picture past level 5.1's luma samples (3840x4096), 4 pictures.
int test_cases(Gpu& gpu) {
  if (breach("BT.601 full", make(kW, kH, kBt601Full), gpu) != 0) return 1;
  if (breach("BT.601 video", stream_only(make(kW, kH, kBt601Video)), gpu) !=
      0) {
    return 1;
  }
  ColorCodes centred = kBt709Full;
  centred.chroma_location = 1;
  if (breach("BT.709 full, sited at the centre",
             stream_only(make(kW, kH, centred)), gpu) != 0) {
    return 1;
  }
  auto cropped = stream_only(make(kCroppedW, kCroppedH, kBt709Video));
  cropped.options.key_qp = 26;
  if (breach("BT.709 video, key QP 26", cropped, gpu) != 0) return 1;
  auto low = stream_only(make(kW, kH, kBt709Full, 1));
  low.options.key_qp = 51;
  if (breach("QP 1, key QP 51", low, gpu) != 0) return 1;
  auto high = stream_only(make(kW, kH, kBt709Full, 51));
  high.options.key_qp = 1;
  if (breach("QP 51, key QP 1", high, gpu) != 0) return 1;
  auto tall = stream_only(make(3840, 4096, kBt709Full, 30));
  tall.pictures = 4;
  return breach("BT.709 full", tall, gpu);
}

// The same through targets the device writes: every rule on BT.709 full,
// then the stream's at the cropped size and video range, on BT.601, and at a
// siting VideoToolbox takes from the picture's tag.
int test_target_cases(Gpu& gpu) {
  const In t = In::Targets;
  if (breach("BT.709 full", make(kW, kH, kBt709Full), gpu, t) != 0) return 1;
  auto cropped = stream_only(make(kCroppedW, kCroppedH, kBt709Video));
  cropped.options.key_qp = 26;
  if (breach("BT.709 video, key QP 26", cropped, gpu, t) != 0) return 1;
  if (breach("BT.601 full", stream_only(make(kW, kH, kBt601Full)), gpu, t) !=
      0) {
    return 1;
  }
  ColorCodes centred = kBt601Video;
  centred.chroma_location = 1;
  return breach("BT.601 video, sited at the centre",
                stream_only(make(kW, kH, centred)), gpu, t);
}

#if defined(__APPLE__)
// A device without what the target's writer needs opens no session on it:
// one that declares Metal's import and not the storage images' formats, and
// one that declares neither.
int test_device_refused(Gpu& gpu) {
  HevcEncoder::Options o;
  o.width = kW;
  o.height = kH;
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
// VideoToolbox writes: each the stream's rules, on the fewest pictures they
// take.
int test_code_set(Gpu& gpu) {
  std::vector<ColorCodes> all;
  for (const int m : {1, 5, 6, 7, 9}) all.push_back({m, 1, 1, true, 0});
  for (const int t : {2, 6, 7, 8, 13, 14, 15, 16, 17, 18}) {
    all.push_back({1, t, 1, true, 0});  // transfer 1 is the matrices'
  }
  for (const int p : {1, 2, 5, 6, 9, 11, 12, 22}) {
    all.push_back({1, 1, p, false, 0});
  }
  for (int s = 1; s <= 5; ++s) all.push_back({1, 1, 1, true, s});
  for (const ColorCodes& c : all) {
    auto run = stream_only(make(kW, kH, c));
    run.pictures = 4;
    const std::string what = "matrix " + std::to_string(c.matrix) +
                             " transfer " + std::to_string(c.transfer) +
                             " primaries " + std::to_string(c.primaries) +
                             " siting " + std::to_string(c.chroma_location);
    if (breach(what.c_str(), run, gpu) != 0) return 1;
  }
  return 0;
}

// A target's description on each matrix of the set, at either range, and on
// each siting: the weights H.273 gives the matrix (5 as 6), written here
// apart from the encoder's table.
int test_target_described(Gpu& gpu) {
  const struct {
    int matrix;
    float kr, kb;
  } weights[] = {{1, 0.2126F, 0.0722F},
                 {5, 0.299F, 0.114F},
                 {6, 0.299F, 0.114F},
                 {7, 0.212F, 0.087F},
                 {9, 0.2627F, 0.0593F}};
  const sensor::ChromaLocation sited[] = {
      sensor::ChromaLocation::Left,       sensor::ChromaLocation::Center,
      sensor::ChromaLocation::TopLeft,    sensor::ChromaLocation::Top,
      sensor::ChromaLocation::BottomLeft, sensor::ChromaLocation::Bottom};
  int siting = 0;
  for (const auto& w : weights) {
    for (const bool full : {true, false}) {
      HevcEncoder::Options o;
      o.width = kW;
      o.height = kH;
      o.color = {w.matrix, 1, 1, full, siting};
      o.device = &gpu.device;
      o.allocator = &gpu.allocator;
      auto e = HevcEncoder::create(o);
      CHECK(e.ok());
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      const auto t = e->target(batch);
      CHECK(t.ok());
      CHECK(std::fabs(t->kr - w.kr) < 1e-6F && std::fabs(t->kb - w.kb) < 1e-6F);
      CHECK(t->full_range == full && t->chroma_location == sited[siting]);
      siting = (siting + 1) % 6;
    }
  }
  std::puts("targets described by each matrix, range and siting");
  return 0;
}

#if defined(__APPLE__)
using Clock = std::chrono::steady_clock;

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

// How long what must not happen is waited for: ten times @p path, one
// picture's whole path as just measured, and at least 20 ms.
Clock::duration hold_for(Clock::duration path) {
  return std::max<Clock::duration>(std::chrono::milliseconds(20), 10 * path);
}

std::uint64_t nanoseconds(Clock::duration d) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

// A host timeline the test holds and sets value by value. Should a call
// block behind a held value, as an encoder that waits for a batch it must
// not, a watchdog sets every value kWatchdog after the gate was made, and
// fired() says so: the test fails rather than hangs. Going, it sets every
// value, so no batch it held outlives it waiting.
class Gate {
 public:
  static constexpr auto kWatchdog = std::chrono::seconds(20);

  explicit Gate(vkc::TimelineSemaphore& timeline) : timeline_(timeline) {
    watchdog_ = std::thread([this] {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!going_cv_.wait_for(lock, kWatchdog, [this] { return going_; })) {
        fired_ = true;
        static_cast<void>(timeline_.signal(kEvery));
      }
    });
  }
  Gate(const Gate&) = delete;
  Gate& operator=(const Gate&) = delete;
  ~Gate() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      going_ = true;
      if (!fired_) static_cast<void>(timeline_.signal(kEvery));
    }
    going_cv_.notify_all();
    watchdog_.join();
  }

  // Sets @p value, as long as the watchdog has not.
  void set(std::uint64_t value) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!fired_) static_cast<void>(timeline_.signal(value));
  }
  bool fired() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return fired_;
  }

 private:
  static constexpr std::uint64_t kEvery = 1000;  // past every value held
  vkc::TimelineSemaphore& timeline_;
  mutable std::mutex mutex_;
  std::condition_variable going_cv_;
  bool going_ = false;
  bool fired_ = false;
  std::thread watchdog_;
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
// picture kHeld's batch runs only once the test sets a value, after its send
// has returned and a receive held ten pictures' time found nothing: it
// decodes to what that batch wrote, so the coding thread coded it only once
// it was written.
int test_quality(Gpu& gpu, const ColorCodes& color, In in) {
  constexpr std::uint32_t kCoded = 6;
  constexpr std::uint32_t kHeld = 3;
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
  std::vector<vkc::PendingBatch> pending(kCoded);
  auto encoder = HevcEncoder::create(o);
  CHECK(encoder.ok());
  Gate held(*gate);  // after the encoder: going first, it sets `gate`
  sensor::HevcDecoder::Options d;
  d.device = &gpu.device;
  d.allocator = &gpu.allocator;
  auto decoder = sensor::HevcDecoder::create(d);
  CHECK(decoder.ok());
  HevcEncoder::CodedPicture coded;
  double worst[3] = {99, 99, 99};
  std::uint32_t out = 0;
  Clock::duration path{};  // the slowest picture's, before the held one
  for (std::uint32_t i = 0; i < kCoded; ++i) {
    const testing::HostNv12 want = testing::test_pattern(kW, kH, 40 + i);
    const Clock::time_point start = Clock::now();
    if (targets) {
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      auto target = encoder->target(batch);
      CHECK(target.ok());
      CHECK(writer->record_upload(batch, gpu.allocator, want, *target).ok());
      std::vector<vkc::TimelinePoint> waits;
      if (i == kHeld) waits.push_back({&*gate, 1});
      auto submitted = batch.submit_async(waits, {{&*written, i + 1}});
      CHECK(submitted.ok());
      pending[i] = std::move(submitted).value();
      CHECK(encoder->send(i == 0, i, {&*written, i + 1}).ok());
      if (i == kHeld) {
        auto at_send = written->value();
        CHECK(at_send.ok() && *at_send < i + 1);  // not yet written
        auto early = encoder->receive(coded, nanoseconds(hold_for(path)));
        CHECK(early.ok() && !*early);  // not coded while held
        held.set(1);
        CHECK(!held.fired());
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
    CHECK(got.ok() && *got);
    if (i < kHeld) path = std::max(path, Clock::now() - start);
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
  CHECK(out == kCoded);
  std::printf("QP 22, %s range, %s: worst PSNR Y %.2f, Cb %.2f, Cr %.2f dB\n",
              color.full_range ? "full" : "video",
              targets ? "targets" : "pixel buffers", worst[0], worst[1],
              worst[2]);
  CHECK(worst[0] >= 45 && worst[1] >= 45 && worst[2] >= 45);
  return 0;
}

// A producer's targets wait for no batch, and the images a batch still
// writes outlive the encoder's teardown and a reopen. Four pictures, two in
// flight, each batch waiting for a host value the test holds and sets only
// to receive the picture two before: each target, on a picture create
// readied, returns while the batch of the picture before it is held;
// destroyed with the last two sent and their batches held, the encoder does
// not return within ten pictures' time, nor once the first is set, and
// returns once both are, with the writers done. Lost while a sent target's
// batch is held, the target after the loss opens the session again, that
// picture fails uncoded, and the next codes as the key. The validation layer
// names an image destroyed while a batch still uses it.
int test_writers_outlive(Gpu& gpu) {
  auto writer = testing::Nv12Writer::create(gpu.device);
  CHECK(writer.ok());
  // The batches wait for `gate` in the first part and `lost_gate` in the
  // second, each held by a Gate made after that part's encoder, so a failed
  // check's teardown sets it before the encoder waits for its writers.
  auto written = vkc::TimelineSemaphore::create(gpu.device);
  auto gate = vkc::TimelineSemaphore::create(gpu.device);
  auto lost_gate = vkc::TimelineSemaphore::create(gpu.device);
  CHECK(written.ok() && gate.ok() && lost_gate.ok());
  std::vector<testing::HostNv12> pictures;
  for (std::uint32_t i = 0; i < 4; ++i) {
    pictures.push_back(testing::test_pattern(kW, kH, 60 + i));
  }
  std::vector<vkc::PendingBatch> pending;  // outlive the encoders
  std::uint64_t value = 0;                 // written's last
  // Picture @p i written into a target taken now, by a batch that waits for
  // @p after (none for a null semaphore) and sets written's next.
  const auto write = [&](HevcEncoder& e, std::uint32_t i,
                         vkc::TimelinePoint after) -> vkc::Status {
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    VKC_ASSIGN(const sensor::YuvImage t, e.target(batch));
    VKC_TRY(writer->record_upload(batch, gpu.allocator, pictures[i], t));
    std::vector<vkc::TimelinePoint> waits;
    if (after.semaphore != nullptr) waits.push_back(after);
    VKC_ASSIGN(vkc::PendingBatch p,
               batch.submit_async(waits, {{&*written, value + 1}}));
    pending.push_back(std::move(p));
    ++value;
    return {};
  };
  HevcEncoder::Options o;
  o.width = kW;
  o.height = kH;
  o.color = kBt709Full;
  o.device = &gpu.device;
  o.allocator = &gpu.allocator;
  {
    auto e = HevcEncoder::create(o);
    CHECK(e.ok());
    Gate held(*gate);
    HevcEncoder::CodedPicture coded;
    Clock::duration path{};  // the slowest picture's, from its value set
    for (std::uint32_t i = 0; i < 4; ++i) {
      if (e->in_flight() == HevcEncoder::kInFlight) {
        const Clock::time_point start = Clock::now();
        held.set(i - 1);  // picture i - 2's batch runs
        const auto got = e->receive(coded);
        CHECK(got.ok() && *got);
        path = std::max(path, Clock::now() - start);
      }
      // Taken while picture i - 1's batch waits for value i, held: it did
      // not wait for that batch, or the watchdog set it.
      CHECK(write(*e, i, {&*gate, i + 1}).ok());
      CHECK(!held.fired());
      CHECK(e->send(i == 0, i, {&*written, value}).ok());
    }
    // Destroyed on a thread of its own with the last two sent, their
    // batches held: it waits for their writers.
    std::optional<HevcEncoder> last(std::move(*e));
    std::promise<std::uint64_t> at_return;
    std::future<std::uint64_t> returned = at_return.get_future();
    Helper destroyer;
    destroyer.thread = std::thread([&] {
      last.reset();
      const auto v = written->value();
      at_return.set_value(v.ok() ? *v : 0);
    });
    CHECK(returned.wait_for(hold_for(path)) == std::future_status::timeout);
    held.set(3);
    // The last writer still held: not yet either.
    CHECK(returned.wait_for(hold_for(path)) == std::future_status::timeout);
    held.set(4);
    CHECK(returned.get() >= value);  // the last writer's value, waited for
    CHECK(!held.fired());
  }
  {
    auto e = HevcEncoder::create(o);
    CHECK(e.ok());
    Gate held(*lost_gate);
    CHECK(write(*e, 2, {&*lost_gate, 1}).ok());
    CHECK(e->send(true, 2, {&*written, value}).ok());
    e->inject(HevcEncoder::Fault::Lose);
    // The next target reopens the session while picture 2's batch is held,
    // then waits on the host for that batch, as a surface new to the session
    // does: a helper sets the value once the session is open again, or the
    // target failed.
    std::atomic<bool> taken{false};
    Helper opener;
    opener.thread = std::thread([&] {
      while (e->lost() && !taken) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      held.set(1);
    });
    const vkc::Status reopened = write(*e, 3, {});
    taken = true;
    opener.thread.join();
    CHECK(reopened.ok());
    CHECK(!held.fired() && !e->lost());
    CHECK(e->send(true, 3, {&*written, value}).ok());
    HevcEncoder::CodedPicture coded;
    const auto lost = e->receive(coded);
    CHECK(!lost.ok() && lost.status().domain() == vkc::Status::Code::IoError);
    const auto key = e->receive(coded);
    CHECK(key.ok() && *key && coded.key && coded.pts == 3);
  }
  std::puts("a target's images outlive the teardown and a reopen");
  return 0;
}
#endif

// A plane image the writer stores into, or not, as @p usage says: 0xEE
// throughout, in GENERAL.
vkc::Result<std::shared_ptr<const vkc::Image>> plane_image(
    Gpu& gpu, std::uint32_t columns, std::uint32_t rows, VkFormat format,
    std::uint32_t sample_bytes, VkImageUsageFlags usage) {
  vkc::ImageDesc desc;
  desc.extent = {columns, rows};
  desc.format = format;
  desc.usage = usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VKC_ASSIGN(vkc::Image image, gpu.allocator.create_image(desc));
  const std::vector<std::uint8_t> filled(
      std::size_t{columns} * rows * sample_bytes, 0xEE);
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
// back as it is, at even and odd sizes and a 4n+2 width whose rows end
// within a kernel thread's 4 samples, into images larger than the picture
// too, from their corner; what it cannot write is refused, and a device
// without the storage images' formats has no writer.
int test_writer(Gpu& gpu) {
  auto writer = testing::Nv12Writer::create(gpu.device);
  CHECK(writer.ok());
  const struct {
    std::uint32_t width, height, pad;
  } sizes[] = {{kW, kH, 0}, {kW - 1, kH - 1, 3}, {kCroppedW, kCroppedH, 0}};
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
  probe.width = kW;
  probe.height = kH;
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
  if (test_target_described(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_device_refused(gpu) != 0) return 1;
  if (test_writers_outlive(gpu) != 0) return 1;
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
