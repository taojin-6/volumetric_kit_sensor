// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "vt_jpeg.hpp"

#include <string>
#include <utility>

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

// Decoding is synchronous, so the picture is handed back through the frame's
// own pointer before the call returns; a JPEG the hardware refuses has none.
void on_picture(void*, void* frame, OSStatus status, VTDecodeInfoFlags,
                CVImageBufferRef image, CMTime, CMTime) {
  if (status == noErr && image != nullptr) {
    *static_cast<CVPixelBufferRef*>(frame) = CVPixelBufferRetain(image);
  }
}

// A dictionary to fill, released with its values.
CFMutableDictionaryRef dictionary() {
  return CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                   &kCFTypeDictionaryKeyCallBacks,
                                   &kCFTypeDictionaryValueCallBacks);
}

}  // namespace

std::unique_ptr<VtJpeg> VtJpeg::open(const Device& device, const char* who) {
  if (!VTIsHardwareDecodeSupported(kCMVideoCodecType_JPEG)) return nullptr;
  auto pictures = VtPictures::create(device, who);
  if (!pictures) return nullptr;
  std::unique_ptr<VtJpeg> vt(new VtJpeg());
  vt->who_ = who;
  vt->pictures_ = std::move(pictures).value();
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

bool VtJpeg::start(std::uint32_t width, std::uint32_t height) {
  stop();
  if (CMVideoFormatDescriptionCreate(
          kCFAllocatorDefault, kCMVideoCodecType_JPEG,
          static_cast<std::int32_t>(width), static_cast<std::int32_t>(height),
          nullptr, &format_) != noErr) {
    format_ = nullptr;
    return false;
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
    return false;
  }
  width_ = width;
  height_ = height;
  return true;
}

Result<std::optional<DecodedPicture>> VtJpeg::decode(const std::uint8_t* data,
                                                     std::size_t size) {
  const std::optional<JpegFrame> frame = read_frame(data, size);
  if (!frame || !frame->yuv420 || frame->width == 0 || frame->height == 0) {
    return std::optional<DecodedPicture>();
  }
  if (session_ == nullptr || frame->width != width_ ||
      frame->height != height_) {
    // A size the hardware took no session for is not asked again each frame.
    if (frame->width == refused_width_ && frame->height == refused_height_) {
      return std::optional<DecodedPicture>();
    }
    if (!start(frame->width, frame->height)) {
      refused_width_ = frame->width;
      refused_height_ = frame->height;
      return std::optional<DecodedPicture>();
    }
  }

  // The bytes wrapped, not copied: they outlive the synchronous decode.
  CMBlockBufferRef block = nullptr;
  if (CMBlockBufferCreateWithMemoryBlock(
          kCFAllocatorDefault, const_cast<std::uint8_t*>(data), size,
          kCFAllocatorNull, nullptr, 0, size, 0, &block) != noErr) {
    return Status::io_error(std::string(who_) + ": wrapping a JPEG");
  }
  CMSampleBufferRef sample = nullptr;
  const std::size_t sizes[1] = {size};
  const OSStatus wrapped = CMSampleBufferCreateReady(
      kCFAllocatorDefault, block, format_, 1, 0, nullptr, 1, sizes, &sample);
  CFRelease(block);
  if (wrapped != noErr) {
    return Status::io_error(std::string(who_) + ": wrapping a JPEG");
  }
  CVPixelBufferRef picture = nullptr;
  const OSStatus decoded =
      VTDecompressionSessionDecodeFrame(session_, sample, 0, &picture, nullptr);
  CFRelease(sample);
  // A session that fails is started afresh for the next JPEG, and this one
  // goes to software, which says what is wrong with it if anything is.
  if (decoded != noErr) {
    if (picture != nullptr) CVPixelBufferRelease(picture);
    stop();
    return std::optional<DecodedPicture>();
  }
  if (picture == nullptr) return std::optional<DecodedPicture>();

  DecodedPicture out;
  const Result<bool> taken =
      pictures_->import(picture, frame->width, frame->height, out);
  CVPixelBufferRelease(picture);  // the images hold their own
  if (!taken) return taken.status();
  if (!taken.value()) return std::optional<DecodedPicture>();
  // JFIF's matrix and range.
  out.matrix = VideoColorMatrix::Bt601;
  out.full_range = true;
  return std::optional<DecodedPicture>(std::move(out));
}

}  // namespace volumetric_kit::recon::sensor::video
