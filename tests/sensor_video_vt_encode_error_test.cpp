// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HevcEncoder's public error contract on VideoToolbox with failures at the
// native session and output boundaries: a hardware encoder busy, absent or
// failing to open, one that takes no QP a picture; and outputs that are
// dropped, missing (then handed over as the session is torn down), a
// P-picture where an IDR was asked, or two for one picture; a loss while a
// picture codes; and the coding thread's quality of service. This executable
// compiles the encoder's sources so its VideoToolbox replacements also work in
// a shared build; no fault-injection API is added to the library. Otherwise the
// calls are the real hardware encoder's.

#include <CoreMedia/CoreMedia.h>
#include <VideoToolbox/VideoToolbox.h>
#include <dlfcn.h>
#include <pthread/qos.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"
#include "volumetric_kit/sensor/testing/hevc_clip.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/testing/nv12_pixel_buffer.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;
using sensor::HevcEncoder;
using Code = vkc::Status::Code;

namespace {

// What the replacements do to the next call or output.
OSStatus open_status = noErr;  // VTCompressionSessionCreate's
bool no_base_qp = false;       // SupportsBaseFrameQP read as false
enum class Output { Real, Dropped, Missing, NotSync, Twice };
std::atomic<Output> output{Output::Real};  // read on VideoToolbox's thread
// The next VTCompressionSessionEncodeFrame waits this long before it codes,
// so a picture is coding while the test acts; set once it waits.
std::atomic<bool> slow{false};
std::atomic<bool> slow_waiting{false};
constexpr auto kSlow = std::chrono::milliseconds(50);
// The quality of service of the thread that last called EncodeFrame.
std::atomic<int> encode_qos{QOS_CLASS_UNSPECIFIED};

VTCompressionOutputCallback real_callback = nullptr;

// The missing output, held back and handed to the callback inside the
// session's Invalidate, as VideoToolbox hands over an output still pending
// when a session is torn down.
struct Held {
  void* refcon = nullptr;
  void* frame = nullptr;
  OSStatus status = noErr;
  VTEncodeInfoFlags flags = 0;
  CMSampleBufferRef sample = nullptr;
};
Held held;
std::atomic<bool> holding{false};  // set once held is written

void on_output(void* refcon, void* frame, OSStatus status,
               VTEncodeInfoFlags flags, CMSampleBufferRef sample) {
  const Output now = output.exchange(Output::Real);
  switch (now) {
    case Output::Real:
      break;
    case Output::Dropped:
      real_callback(refcon, frame, noErr, kVTEncodeInfo_FrameDropped, nullptr);
      return;
    case Output::Missing:
      held = {refcon, frame, status, flags, sample};
      if (sample != nullptr) CFRetain(sample);
      holding = true;
      return;
    case Output::NotSync:
      if (sample != nullptr) {
        CFArrayRef all = CMSampleBufferGetSampleAttachmentsArray(sample, true);
        auto first = static_cast<CFMutableDictionaryRef>(
            const_cast<void*>(CFArrayGetValueAtIndex(all, 0)));
        CFDictionarySetValue(first, kCMSampleAttachmentKey_NotSync,
                             kCFBooleanTrue);
      }
      break;
    case Output::Twice:
      real_callback(refcon, frame, status, flags, sample);
      break;
  }
  real_callback(refcon, frame, status, flags, sample);
}

}  // namespace

extern "C" OSStatus VTCompressionSessionCreate(
    CFAllocatorRef allocator, int32_t width, int32_t height,
    CMVideoCodecType codec, CFDictionaryRef specification,
    CFDictionaryRef source_attributes, CFAllocatorRef data_allocator,
    VTCompressionOutputCallback callback, void* refcon,
    VTCompressionSessionRef* session) {
  if (open_status != noErr) {
    *session = nullptr;
    return open_status;
  }
  real_callback = callback;
  const auto real = reinterpret_cast<decltype(&VTCompressionSessionCreate)>(
      dlsym(RTLD_NEXT, "VTCompressionSessionCreate"));
  return real(allocator, width, height, codec, specification, source_attributes,
              data_allocator, &on_output, refcon, session);
}

extern "C" OSStatus VTCompressionSessionEncodeFrame(
    VTCompressionSessionRef session, CVImageBufferRef image, CMTime pts,
    CMTime duration, CFDictionaryRef frame_options, void* frame,
    VTEncodeInfoFlags* info) {
  encode_qos = static_cast<int>(qos_class_self());
  if (slow.exchange(false)) {
    slow_waiting = true;
    std::this_thread::sleep_for(kSlow);
  }
  const auto real =
      reinterpret_cast<decltype(&VTCompressionSessionEncodeFrame)>(
          dlsym(RTLD_NEXT, "VTCompressionSessionEncodeFrame"));
  return real(session, image, pts, duration, frame_options, frame, info);
}

extern "C" void VTCompressionSessionInvalidate(
    VTCompressionSessionRef session) {
  if (holding.exchange(false)) {
    real_callback(held.refcon, held.frame, held.status, held.flags,
                  held.sample);
    if (held.sample != nullptr) CFRelease(held.sample);
  }
  const auto real = reinterpret_cast<decltype(&VTCompressionSessionInvalidate)>(
      dlsym(RTLD_NEXT, "VTCompressionSessionInvalidate"));
  real(session);
}

extern "C" OSStatus VTSessionCopyProperty(VTSessionRef session, CFStringRef key,
                                          CFAllocatorRef allocator,
                                          void* value) {
  if (no_base_qp &&
      CFEqual(key, kVTCompressionPropertyKey_SupportsBaseFrameQP)) {
    *static_cast<CFTypeRef*>(value) = kCFBooleanFalse;
    return noErr;
  }
  const auto real = reinterpret_cast<decltype(&VTSessionCopyProperty)>(
      dlsym(RTLD_NEXT, "VTSessionCopyProperty"));
  return real(session, key, allocator, value);
}

namespace {

HevcEncoder::Options options() {
  HevcEncoder::Options o;
  o.width = 512;
  o.height = 512;
  o.color = {1, 1, 1, true, 0};
  return o;
}

int test_open_failures() {
  const struct {
    OSStatus native;
    Code code;
    const char* says;
  } cases[] = {
      {kVTVideoEncoderNotAvailableNowErr, Code::Unsupported,
       "busy, or at its session limit"},
      {kVTCouldNotFindVideoEncoderErr, Code::Unsupported,
       "no hardware HEVC encoder here opens"},
      {kVTAllocationFailedErr, Code::OutOfMemory, "opening"},
      {kVTVideoEncoderMalfunctionErr, Code::Backend, "opening"},
  };
  for (const auto& c : cases) {
    open_status = c.native;
    const auto made = HevcEncoder::create(options());
    open_status = noErr;
    CHECK(!made.ok() && made.status().domain() == c.code);
    if (made.status().message().find(c.says) == std::string::npos ||
        made.status().message().find("512x512") == std::string::npos) {
      std::fprintf(stderr, "%s\n", made.status().message().c_str());
    }
    CHECK(made.status().message().find(c.says) != std::string::npos);
    CHECK(made.status().message().find("512x512") != std::string::npos);
    if (c.code == Code::Backend) CHECK(made.status().detail() == c.native);
  }
  no_base_qp = true;
  const auto made = HevcEncoder::create(options());
  no_base_qp = false;
  CHECK(!made.ok() && made.status().domain() == Code::Unsupported &&
        made.status().message().find("SupportsBaseFrameQP") !=
            std::string::npos);
  return 0;
}

// One picture sent with @p fault on its output: what receive says.
vkc::Result<bool> coded_with(HevcEncoder& e, Output fault, bool key,
                             std::int64_t pts) {
  auto pixels = testing::pixel_buffer(
      testing::test_pattern(512, 512, static_cast<std::uint32_t>(pts)),
      {1, 1, 1, true, 0});
  if (!pixels.ok()) return pixels.status();
  output = fault;
  const vkc::Status sent = sensor::send_pixel_buffer(e, *pixels, key, pts);
  CVPixelBufferRelease(*pixels);
  if (!sent.ok()) return sent;
  HevcEncoder::CodedPicture coded;
  return e.receive(coded);
}

int test_output_failures() {
  auto e = HevcEncoder::create(options());
  CHECK(e.ok());
  CHECK(coded_with(*e, Output::Real, true, 0).ok());
  // The coding thread calls VideoToolbox at user-interactive QoS.
  CHECK(encode_qos == QOS_CLASS_USER_INTERACTIVE);
  // A drop the hardware reports: IoError, the session kept, a key owed.
  auto r = coded_with(*e, Output::Dropped, false, 1);
  CHECK(!r.ok() && r.status().domain() == Code::IoError &&
        r.status().message().find("dropped") != std::string::npos);
  CHECK(e->key_owed() && !e->lost());
  CHECK(coded_with(*e, Output::Real, true, 2).ok());
  // A P-picture where an IDR was asked: IoError, the session lost.
  r = coded_with(*e, Output::NotSync, true, 3);
  CHECK(!r.ok() && r.status().domain() == Code::IoError &&
        r.status().message().find("asked for an IDR") != std::string::npos);
  CHECK(e->key_owed() && e->lost());
  CHECK(coded_with(*e, Output::Real, true, 4).ok());
  CHECK(!e->lost());
  // Two access units for one picture: the next picture fails, the session
  // lost.
  CHECK(coded_with(*e, Output::Twice, false, 5).ok());
  r = coded_with(*e, Output::Real, false, 6);
  CHECK(!r.ok() && r.status().domain() == Code::IoError &&
        r.status().message().find("two access units") != std::string::npos);
  CHECK(e->lost());
  CHECK(coded_with(*e, Output::Real, true, 7).ok());
  // No access unit: IoError after 2 s, the session lost. The output comes
  // as the key's reopen tears the session down, and lands nowhere.
  r = coded_with(*e, Output::Missing, false, 8);
  CHECK(!r.ok() && r.status().domain() == Code::IoError &&
        r.status().message().find("within 2 s") != std::string::npos);
  CHECK(e->lost() && e->key_owed() && holding);
  CHECK(coded_with(*e, Output::Real, true, 9).ok());
  CHECK(!holding && !e->lost());
  return 0;
}

// A loss while a picture codes, and a key sent at once: the reopen waits
// for that picture, which the old session codes in full, before it closes
// the session; then the key codes on the new one.
int test_reopen_while_coding() {
  auto e = HevcEncoder::create(options());
  CHECK(e.ok());
  CHECK(coded_with(*e, Output::Real, true, 0).ok());
  auto pixels = testing::pixel_buffer(testing::test_pattern(512, 512, 1),
                                      {1, 1, 1, true, 0});
  CHECK(pixels.ok());
  slow = true;
  slow_waiting = false;
  CHECK(sensor::send_pixel_buffer(*e, *pixels, false, 1).ok());
  while (!slow_waiting) std::this_thread::yield();
  e->inject(HevcEncoder::Fault::Lose);
  CHECK(e->lost() && e->key_owed());
  CHECK(sensor::send_pixel_buffer(*e, *pixels, true, 2).ok());
  CVPixelBufferRelease(*pixels);
  CHECK(!e->lost());
  HevcEncoder::CodedPicture coded;
  auto r = e->receive(coded);
  CHECK(r.ok() && *r && coded.pts == 1 && !coded.key);
  r = e->receive(coded);
  CHECK(r.ok() && *r && coded.pts == 2 && coded.key);
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (auto probe = HevcEncoder::create(options()); !probe) {
    const bool required = testing::hevc_encoder_promised();
    std::fprintf(stderr, "no hardware HEVC encoder (%s); %s\n",
                 probe.status().message().c_str(),
                 required ? "failing" : "skipping");
    return required ? 1 : vkc::test::kSkipExitCode;
  }
  if (test_open_failures() != 0) return 1;
  if (test_output_failures() != 0) return 1;
  if (test_reopen_while_coding() != 0) return 1;
  std::puts("sensor_video_vt_encode_error: OK");
  return 0;
}
