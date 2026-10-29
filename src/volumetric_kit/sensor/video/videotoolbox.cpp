// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "videotoolbox.hpp"

namespace volumetric_kit::sensor::video {
namespace {

// Both the decode call and its callback can fail. Keep the callback's status
// even when it supplies no picture.
struct Decoded {
  CVPixelBufferRef picture = nullptr;
  OSStatus status = noErr;
};

void on_picture(void*, void* frame, OSStatus status, VTDecodeInfoFlags,
                CVImageBufferRef image, CMTime, CMTime) {
  auto& decoded = *static_cast<Decoded*>(frame);
  decoded.status = status;
  if (status == noErr && image != nullptr) {
    decoded.picture = CVPixelBufferRetain(image);
  }
}

// A dictionary to fill, released with its values.
CfPtr<CFMutableDictionaryRef> dictionary() {
  return CfPtr<CFMutableDictionaryRef>(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks));
}

}  // namespace

core::Status vt_error(const char* who, const std::string& what, OSStatus error,
                      bool decoding) {
  const std::string message = std::string(who) + ": " + what +
                              " (VideoToolbox status " + std::to_string(error) +
                              ")";
  switch (error) {
    case kVTParameterErr:
      // The arguments are already checked, but VideoToolbox also returns
      // this for a truncated picture. At session creation it still means
      // the device path failed.
      if (decoding) return core::Status::io_error(message);
      break;
    case kVTVideoDecoderBadDataErr:
    case kVTVideoDecoderReferenceMissingErr:
      return core::Status::io_error(message);
    case kVTVideoDecoderUnsupportedDataFormatErr:
    case kVTCouldNotFindVideoDecoderErr:
      return core::Status::unsupported(message);
    case kVTAllocationFailedErr:
    case kCMFormatDescriptionError_AllocationFailed:
      return core::Status::out_of_memory(message);
    default:
      break;
  }
  return core::Status::backend_error(core::Status::Backend::VideoToolbox, error,
                                     message);
}

core::Result<SessionPtr> open_session(CMVideoFormatDescriptionRef format,
                                      bool full_range, const char* who,
                                      const std::string& what) {
  const auto out = dictionary();
  const std::int32_t pixel =
      full_range ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                 : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  const CfPtr<CFNumberRef> number(
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixel));
  const auto surface = dictionary();
  CFDictionarySetValue(out.get(), kCVPixelBufferPixelFormatTypeKey,
                       number.get());
  CFDictionarySetValue(out.get(), kCVPixelBufferIOSurfacePropertiesKey,
                       surface.get());
  CFDictionarySetValue(out.get(), kCVPixelBufferMetalCompatibilityKey,
                       kCFBooleanTrue);
  const auto spec = dictionary();
  CFDictionarySetValue(
      spec.get(),
      kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
      kCFBooleanTrue);
  const VTDecompressionOutputCallbackRecord callback{&on_picture, nullptr};
  VTDecompressionSessionRef session = nullptr;
  const OSStatus made = VTDecompressionSessionCreate(
      kCFAllocatorDefault, format, spec.get(), out.get(), &callback, &session);
  if (made != noErr) return vt_error(who, "opening " + what, made);
  return SessionPtr(session);
}

core::Result<PixelBufferPtr> decode_sample(VTDecompressionSessionRef session,
                                           CMVideoFormatDescriptionRef format,
                                           const std::uint8_t* data,
                                           std::size_t size, const char* who,
                                           const std::string& what) {
  CMBlockBufferRef block = nullptr;
  if (CMBlockBufferCreateWithMemoryBlock(
          kCFAllocatorDefault, const_cast<std::uint8_t*>(data), size,
          kCFAllocatorNull, nullptr, 0, size, 0, &block) != noErr) {
    return core::Status::out_of_memory(std::string(who) + ": wrapping " + what);
  }
  CMSampleBufferRef sample = nullptr;
  const std::size_t sizes[1] = {size};
  const OSStatus wrapped = CMSampleBufferCreateReady(
      kCFAllocatorDefault, block, format, 1, 0, nullptr, 1, sizes, &sample);
  CFRelease(block);
  if (wrapped != noErr) {
    return core::Status::out_of_memory(std::string(who) + ": wrapping " + what);
  }
  // Decoded before the call returns: no flag asks for it later.
  Decoded decoded;
  OSStatus status =
      VTDecompressionSessionDecodeFrame(session, sample, 0, &decoded, nullptr);
  if (status == noErr) {
    status = VTDecompressionSessionWaitForAsynchronousFrames(session);
  }
  CFRelease(sample);
  PixelBufferPtr picture(decoded.picture);
  if (status == noErr) status = decoded.status;
  if (status != noErr) return vt_error(who, "decoding " + what, status, true);
  return picture;
}

}  // namespace volumetric_kit::sensor::video
