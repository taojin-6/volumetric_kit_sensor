// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "vt_jpeg.hpp"

#include <optional>
#include <string>
#include <utility>

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "vt_pictures.hpp"

namespace volumetric_kit::recon::sensor::video {
namespace {

// A JPEG's frame header: its size, and whether it is baseline or extended
// 8-bit 4:2:0 in three components, the one layout taken here.
struct JpegFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  bool yuv420 = false;
};

// The frame header, read by walking the markers from SOI; empty for bytes
// that reach a scan or their end before one.
std::optional<JpegFrame> read_frame(const std::uint8_t* d, std::size_t n) {
  if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return std::nullopt;
  std::size_t i = 2;
  while (i + 4 <= n) {
    if (d[i] != 0xFF) return std::nullopt;
    const std::uint8_t marker = d[i + 1];
    if (marker == 0xFF) {  // a fill byte
      ++i;
      continue;
    }
    if (marker == 0xD9 || marker == 0xDA) return std::nullopt;
    const std::size_t length = (std::size_t{d[i + 2]} << 8) | d[i + 3];
    if (length < 2 || length > n - i - 2) return std::nullopt;
    // SOF0 to SOF15, less DHT, JPG and DAC, which share the range.
    if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 &&
        marker != 0xCC) {
      const std::uint8_t* s = d + i + 4;
      if (length < 8 || length < 8 + 3 * std::size_t{s[5]}) {
        return std::nullopt;
      }
      JpegFrame frame;
      frame.height = (std::uint32_t{s[1]} << 8) | s[2];
      frame.width = (std::uint32_t{s[3]} << 8) | s[4];
      frame.yuv420 = (marker == 0xC0 || marker == 0xC1) && s[0] == 8 &&
                     s[5] == 3 && s[7] == 0x22 && s[10] == 0x11 &&
                     s[13] == 0x11;
      return frame;
    }
    i += 2 + length;
  }
  return std::nullopt;
}

// Both the decode call and its synchronous callback can fail. Keep the
// callback's status even when it supplies no picture.
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

core::Status vt_error(const char* who, const std::string& what, OSStatus error,
                      bool decoding = false) {
  const std::string message = std::string(who) + ": " + what +
                              " (VideoToolbox status " + std::to_string(error) +
                              ")";
  switch (error) {
    case kVTParameterErr:
      // DecodeFrame's arguments are already checked, but VideoToolbox also
      // returns this for a truncated JPEG. At session creation it still
      // means the device path failed.
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
  return core::Status::backend_error(error, message);
}

// A dictionary to fill, released with its values.
CFMutableDictionaryRef dictionary() {
  return CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                   &kCFTypeDictionaryKeyCallBacks,
                                   &kCFTypeDictionaryValueCallBacks);
}

}  // namespace

core::Result<std::unique_ptr<VtJpeg>> VtJpeg::open(const core::Device& device,
                                                   const char* who) {
  if (!VTIsHardwareDecodeSupported(kCMVideoCodecType_JPEG)) {
    return core::Status::unsupported(
        std::string(who) + ": VideoToolbox has no hardware JPEG decoder");
  }
  std::unique_ptr<VtJpeg> vt(new VtJpeg());
  VKC_ASSIGN(vt->pictures_, VtPictures::create(device, who));
  vt->who_ = who;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  vt->max_extent_ = props.limits.maxImageDimension2D;
  return vt;
}

VtJpeg::~VtJpeg() { stop(); }

void VtJpeg::stop() noexcept {
  if (session_ != nullptr) {
    VTDecompressionSessionInvalidate(session_);
    CFRelease(session_);
    session_ = nullptr;
  }
  if (format_ != nullptr) {
    CFRelease(format_);
    format_ = nullptr;
  }
  width_ = height_ = 0;
}

core::Status VtJpeg::start(std::uint32_t width, std::uint32_t height) {
  stop();
  const OSStatus described = CMVideoFormatDescriptionCreate(
      kCFAllocatorDefault, kCMVideoCodecType_JPEG,
      static_cast<std::int32_t>(width), static_cast<std::int32_t>(height),
      nullptr, &format_);
  if (described != noErr) {
    format_ = nullptr;
    return vt_error(who_, "describing a JPEG", described);
  }
  // NV12 on an IOSurface Metal reads, full range as the JPEG's samples are;
  // on the hardware or not at all.
  CFMutableDictionaryRef out = dictionary();
  const std::int32_t pixel = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
  CFNumberRef number =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixel);
  CFMutableDictionaryRef surface = dictionary();
  CFDictionarySetValue(out, kCVPixelBufferPixelFormatTypeKey, number);
  CFDictionarySetValue(out, kCVPixelBufferIOSurfacePropertiesKey, surface);
  CFDictionarySetValue(out, kCVPixelBufferMetalCompatibilityKey,
                       kCFBooleanTrue);
  CFMutableDictionaryRef spec = dictionary();
  CFDictionarySetValue(
      spec, kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder,
      kCFBooleanTrue);
  const VTDecompressionOutputCallbackRecord callback{&on_picture, nullptr};
  const OSStatus made = VTDecompressionSessionCreate(
      kCFAllocatorDefault, format_, spec, out, &callback, &session_);
  CFRelease(spec);
  CFRelease(surface);
  CFRelease(number);
  CFRelease(out);
  if (made != noErr) {
    session_ = nullptr;
    stop();
    return vt_error(who_,
                    "opening a " + std::to_string(width) + "x" +
                        std::to_string(height) + " JPEG session",
                    made);
  }
  width_ = width;
  height_ = height;
  return {};
}

core::Result<DecodedPicture> VtJpeg::decode(const std::uint8_t* data,
                                            std::size_t size) {
  const std::optional<JpegFrame> frame = read_frame(data, size);
  const auto fail = [this](const std::string& why) {
    return core::Status::io_error(std::string(who_) + ": " + why);
  };
  const auto refuse = [this](const std::string& why) {
    return core::Status::unsupported(std::string(who_) + ": " + why);
  };
  if (!frame) return fail("no JPEG frame header ahead of its scan");
  if (frame->width == 0 || frame->height == 0) return fail("a JPEG of no size");
  if (!frame->yuv420) {
    return refuse("the hardware takes baseline 8-bit 4:2:0 JPEGs only");
  }
  const std::string size_text =
      std::to_string(frame->width) + "x" + std::to_string(frame->height);
  // A plane past the extent cannot be a texture (Metal aborts on one: a
  // 16400-wide JPEG on an M4, whose extent is 16384).
  if (frame->width > max_extent_ || frame->height > max_extent_) {
    return refuse("a " + size_text + " JPEG is larger than the device's " +
                  std::to_string(max_extent_) + "-pixel images");
  }
  if (session_ == nullptr || frame->width != width_ ||
      frame->height != height_) {
    VKC_TRY(start(frame->width, frame->height));
  }

  // The bytes wrapped, not copied: they outlive the synchronous decode.
  CMBlockBufferRef block = nullptr;
  if (CMBlockBufferCreateWithMemoryBlock(
          kCFAllocatorDefault, const_cast<std::uint8_t*>(data), size,
          kCFAllocatorNull, nullptr, 0, size, 0, &block) != noErr) {
    return core::Status::out_of_memory(std::string(who_) + ": wrapping a JPEG");
  }
  CMSampleBufferRef sample = nullptr;
  const std::size_t sizes[1] = {size};
  const OSStatus wrapped = CMSampleBufferCreateReady(
      kCFAllocatorDefault, block, format_, 1, 0, nullptr, 1, sizes, &sample);
  CFRelease(block);
  if (wrapped != noErr) {
    return core::Status::out_of_memory(std::string(who_) + ": wrapping a JPEG");
  }
  Decoded decoded;
  OSStatus status =
      VTDecompressionSessionDecodeFrame(session_, sample, 0, &decoded, nullptr);
  CFRelease(sample);
  if (status == noErr) status = decoded.status;
  // A session that fails is started afresh for the next JPEG.
  if (status != noErr) {
    if (decoded.picture != nullptr) CVPixelBufferRelease(decoded.picture);
    stop();
    return vt_error(who_, "decoding a JPEG", status, true);
  }
  if (decoded.picture == nullptr) return fail("the JPEG does not decode");

  DecodedPicture out;
  const core::Status imported =
      pictures_->import(decoded.picture, frame->width, frame->height, out);
  CVPixelBufferRelease(decoded.picture);  // the images hold their own
  VKC_TRY(imported);
  // JFIF's matrix and range.
  out.matrix = VideoColorMatrix::Bt601;
  out.full_range = true;
  out.chroma_location = ChromaLocation::Center;
  return out;
}

}  // namespace volumetric_kit::recon::sensor::video
