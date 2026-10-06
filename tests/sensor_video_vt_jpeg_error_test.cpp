// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Exercise JpegDecoder's public error contract with failures at the native
// session, decode-call and callback boundaries. This executable compiles the
// JPEG sources so its VideoToolbox replacements also work in a shared build;
// no fault-injection API is added to the library. Successful calls use the
// real hardware decoder.

#include <VideoToolbox/VideoToolbox.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "no_device.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"

namespace vkc = volumetric_kit::core;
namespace vr = volumetric_kit::recon;
using vr::sensor::JpegDecoder;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

enum class Phase { Session, Decode, Callback };
Phase phase = Phase::Decode;
OSStatus injected = noErr;
int failures = 0;
VTDecompressionOutputCallbackRecord callback{};

}  // namespace

extern "C" OSStatus VTDecompressionSessionCreate(
    CFAllocatorRef allocator, CMVideoFormatDescriptionRef format,
    CFDictionaryRef specification, CFDictionaryRef attributes,
    const VTDecompressionOutputCallbackRecord* output,
    VTDecompressionSessionRef* session) {
  if (injected != noErr && phase == Phase::Session) {
    ++failures;
    *session = nullptr;
    return injected;
  }
  callback = *output;
  const auto real = reinterpret_cast<decltype(&VTDecompressionSessionCreate)>(
      dlsym(RTLD_NEXT, "VTDecompressionSessionCreate"));
  return real(allocator, format, specification, attributes, output, session);
}

extern "C" OSStatus VTDecompressionSessionDecodeFrame(
    VTDecompressionSessionRef session, CMSampleBufferRef sample,
    VTDecodeFrameFlags flags, void* frame, VTDecodeInfoFlags* info) {
  if (injected != noErr) {
    ++failures;
    if (phase == Phase::Callback) {
      callback.decompressionOutputCallback(callback.decompressionOutputRefCon,
                                           frame, injected, 0, nullptr,
                                           kCMTimeInvalid, kCMTimeInvalid);
      return noErr;
    }
    return injected;
  }
  const auto real =
      reinterpret_cast<decltype(&VTDecompressionSessionDecodeFrame)>(
          dlsym(RTLD_NEXT, "VTDecompressionSessionDecodeFrame"));
  return real(session, sample, flags, frame, info);
}

int main() {
  auto instance = vkc::Instance::create({});
  if (!instance)
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  auto physical = instance->select_physical_device(vr::device_requirements());
  if (!physical)
    return vr_test::no_device("no compute-capable device",
                              physical.status().message());
  auto device =
      vkc::Device::create(*instance, *physical, vr::device_requirements());
  CHECK(device.ok());
  JpegDecoder::Options options;
  options.device = &*device;
  if (auto probe = JpegDecoder::create(options); !probe) {
    CHECK(probe.status().domain() == vkc::Status::Code::Unsupported);
    return vr_test::no_jpeg_decoder(probe.status().message());
  }
  std::ifstream in(VR_JPEG_DATA "/patches_256x144.jpg", std::ios::binary);
  const std::vector<std::uint8_t> jpeg((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
  CHECK(!jpeg.empty());

  using Code = vkc::Status::Code;
  const struct {
    OSStatus native;
    Code expected;
  } errors[] = {
      {kVTParameterErr, Code::IoError},
      {kVTVideoDecoderBadDataErr, Code::IoError},
      {kVTVideoDecoderUnsupportedDataFormatErr, Code::Unsupported},
      {kVTAllocationFailedErr, Code::OutOfMemory},
      {kVTVideoDecoderMalfunctionErr, Code::Backend},
      {kVTVideoDecoderNotAvailableNowErr, Code::Backend},
      {kVTInvalidSessionErr, Code::Backend},
  };
  for (const Phase at : {Phase::Session, Phase::Decode, Phase::Callback}) {
    for (const auto& error : errors) {
      auto decoder = JpegDecoder::create(options);
      CHECK(decoder.ok());
      phase = at;
      injected = error.native;
      failures = 0;
      const auto failed = decoder->decode(jpeg.data(), jpeg.size());
      injected = noErr;
      CHECK(failures == 1);
      const Code expected =
          at == Phase::Session && error.native == kVTParameterErr
              ? Code::Backend
              : error.expected;
      CHECK(failed.status().domain() == expected);
      if (expected == Code::Backend)
        CHECK(failed.status().detail() == error.native);
      // The public decoder does not latch failures; its consumer decides
      // whether to stop. In particular, bad data must cost only that JPEG.
      const auto recovered = decoder->decode(jpeg.data(), jpeg.size());
      CHECK(recovered.ok());
      CHECK(recovered->width == 256 && recovered->height == 144);
      CHECK(recovered->image[0] != nullptr && recovered->image[1] != nullptr);
    }
  }
  std::puts("sensor_video_vt_jpeg_error: OK");
  return 0;
}
