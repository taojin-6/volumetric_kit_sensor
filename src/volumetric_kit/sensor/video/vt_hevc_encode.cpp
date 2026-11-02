// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HevcEncoder's session on VideoToolbox (Apple): one hardware, low-latency
// HEVC compression session, a base QP on every picture, its colour stated
// through Core Video's own code-point strings. In this mode
// VTCompressionSessionEncodeFrame returns once the picture is coded, so
// HevcEncoder's coding thread calls it and waits for the output here.

#include <CoreMedia/CoreMedia.h>
#include <CoreVideo/CoreVideo.h>
#include <VideoToolbox/VideoToolbox.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../apple/core_video_tags.hpp"
#include "hevc_encode_session.hpp"
#include "videotoolbox.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::sensor::video {
namespace {

constexpr const char* kWho = "HevcEncoder";

// How long a picture's output is waited for: far past the tens of
// milliseconds the largest picture takes, so only a stuck session meets it.
constexpr auto kOutputWait = std::chrono::seconds(2);

constexpr std::array<std::uint8_t, 4> kStartCode = {0, 0, 0, 1};

std::string size_text(std::uint32_t width, std::uint32_t height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

CfPtr<CFNumberRef> number(std::int32_t value) {
  return CfPtr<CFNumberRef>(
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &value));
}

std::string fourcc(OSType type) {
  std::string out = "'";
  for (int shift = 24; shift >= 0; shift -= 8) {
    out += static_cast<char>((type >> shift) & 0xff);
  }
  return out + "'";
}

// A session property as the settings line prints it: a flag on or off, a
// number, a string as it is, "unset" where the session reports none.
std::string property_text(VTCompressionSessionRef session, CFStringRef key) {
  CFTypeRef value = nullptr;
  if (VTSessionCopyProperty(session, key, kCFAllocatorDefault, &value) !=
          noErr ||
      value == nullptr) {
    return "unset";
  }
  const CfPtr<CFTypeRef> owned(value);
  if (CFGetTypeID(value) == CFBooleanGetTypeID()) {
    return CFBooleanGetValue(static_cast<CFBooleanRef>(value)) ? "on" : "off";
  }
  if (CFGetTypeID(value) == CFNumberGetTypeID()) {
    double v = 0;
    CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberDoubleType, &v);
    char text[32];
    std::snprintf(text, sizeof text, "%g", v);
    return text;
  }
  return sensor::detail::utf8(value);
}

}  // namespace

const bool HevcEncodeSession::kWritesSiting = true;

struct HevcEncodeSession::Impl {
  HevcEncoder::Options options;
  CompressionPtr session;
  std::uint64_t frames = 0;  // handed to the encoder
  std::vector<std::uint8_t> scratch;

  // The picture encode waits for, written on VideoToolbox's thread.
  std::mutex mutex;
  std::condition_variable ready;
  std::uint64_t waiting = 0;  // its frame; 0 for none
  bool want_key = false;
  bool drop = false;
  std::vector<std::uint8_t>* out = nullptr;
  bool done = false;
  bool extra = false;  // an output for a frame already done, or none waited
  OSStatus status = noErr;
  bool dropped = false;
  core::Status failure;

  static void on_output(void* self, void* frame_ref, OSStatus status,
                        VTEncodeInfoFlags flags, CMSampleBufferRef sample);
  // @p sample's length-prefixed NAL units as Annex B into @p out, a key's
  // parameter sets first.
  core::Status annex_b(CMSampleBufferRef sample, bool want_key,
                       std::vector<std::uint8_t>& out);
  // Sets session property @p key, or says the session refuses it.
  core::Status set(CFStringRef key, CFTypeRef value) const;
};

void HevcEncodeSession::Impl::on_output(void* self, void* frame_ref,
                                        OSStatus status,
                                        VTEncodeInfoFlags flags,
                                        CMSampleBufferRef sample) {
  Impl& s = *static_cast<Impl*>(self);
  const auto frame =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(frame_ref));
  const std::lock_guard<std::mutex> lock(s.mutex);
  if (frame != s.waiting || s.done || s.out == nullptr) {
    // A second access unit for a picture, or one for a picture given up on:
    // the session codes out of step. A drop's notice is no access unit.
    if (status == noErr && sample != nullptr &&
        (flags & kVTEncodeInfo_FrameDropped) == 0) {
      s.extra = true;
    }
    return;
  }
  s.status = status;
  s.dropped = status == noErr && ((flags & kVTEncodeInfo_FrameDropped) != 0 ||
                                  sample == nullptr || s.drop);
  if (status == noErr && !s.dropped) {
    try {
      s.failure = s.annex_b(sample, s.want_key, *s.out);
    } catch (...) {  // nothing may leave VideoToolbox's thread
      s.out->clear();
      s.failure = core::exception_status("HevcEncoder: an access unit",
                                         core::Status::Code::OutOfMemory);
    }
  }
  s.done = true;
  s.ready.notify_all();
}

core::Status HevcEncodeSession::Impl::annex_b(CMSampleBufferRef sample,
                                              bool want_key,
                                              std::vector<std::uint8_t>& out) {
  out.clear();
  bool sync = true;
  CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(sample, 0);
  if (attachments != nullptr && CFArrayGetCount(attachments) > 0) {
    const auto first =
        static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(attachments, 0));
    const void* not_sync = nullptr;
    if (CFDictionaryGetValueIfPresent(first, kCMSampleAttachmentKey_NotSync,
                                      &not_sync) &&
        not_sync != nullptr &&
        CFBooleanGetValue(static_cast<CFBooleanRef>(not_sync))) {
      sync = false;
    }
  }
  if (want_key && !sync) {
    return core::Status::io_error(
        std::string(kWho) +
        ": asked for an IDR, the encoder coded a P-picture");
  }
  CMFormatDescriptionRef format = CMSampleBufferGetFormatDescription(sample);
  std::size_t sets = 0;
  int header = 0;
  OSStatus status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
      format, 0, nullptr, nullptr, &sets, &header);
  if (status != noErr) {
    return vt_error(kWho, "reading the parameter sets", status);
  }
  if (header < 1 || header > 4) {
    return core::Status::io_error(std::string(kWho) +
                                  ": a NAL unit length of " +
                                  std::to_string(header) + " bytes");
  }
  CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample);
  const std::size_t total =
      block != nullptr ? CMBlockBufferGetDataLength(block) : 0;
  if (total == 0) {
    return core::Status::io_error(std::string(kWho) +
                                  ": the encoder gave an empty picture");
  }
  char* data = nullptr;
  std::size_t contiguous = 0;
  const std::uint8_t* bytes = nullptr;
  if (CMBlockBufferGetDataPointer(block, 0, &contiguous, nullptr, &data) ==
          noErr &&
      contiguous == total) {
    bytes = reinterpret_cast<const std::uint8_t*>(data);
  } else {
    scratch.resize(total);
    status = CMBlockBufferCopyDataBytes(block, 0, total, scratch.data());
    if (status != noErr) return vt_error(kWho, "reading a picture", status);
    bytes = scratch.data();
  }
  const auto append = [&out](const std::uint8_t* nal, std::size_t size) {
    out.insert(out.end(), kStartCode.begin(), kStartCode.end());
    out.insert(out.end(), nal, nal + size);
  };
  if (sync) {
    for (std::size_t i = 0; i < sets; ++i) {
      const std::uint8_t* set = nullptr;
      std::size_t size = 0;
      status = CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
          format, i, &set, &size, nullptr, nullptr);
      if (status != noErr) {
        return vt_error(kWho, "reading the parameter sets", status);
      }
      append(set, size);
    }
  }
  const auto length = static_cast<std::size_t>(header);
  for (std::size_t at = 0; at < total;) {
    std::size_t size = 0;
    if (total - at > length) {
      for (std::size_t k = 0; k < length; ++k) {
        size = (size << 8U) | bytes[at + k];
      }
      at += length;
    }
    if (size == 0 || size > total - at) {
      out.clear();
      return core::Status::io_error(std::string(kWho) +
                                    ": a NAL unit runs past its picture");
    }
    append(bytes + at, size);
    at += size;
  }
  return {};
}

core::Status HevcEncodeSession::Impl::set(CFStringRef key,
                                          CFTypeRef value) const {
  const OSStatus status = VTSessionSetProperty(session.get(), key, value);
  if (status == noErr) return {};
  return core::Status::unsupported(
      std::string(kWho) + ": the low-latency session refuses " +
      sensor::detail::utf8(key) + " (VideoToolbox status " +
      std::to_string(status) + ")");
}

HevcEncodeSession::HevcEncodeSession(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

HevcEncodeSession::~HevcEncodeSession() = default;

core::Result<std::unique_ptr<HevcEncodeSession>> HevcEncodeSession::open(
    const HevcEncoder::Options& options, HevcEncoder::Facts& facts) try {
  const int key_qp = options.key_qp.value_or(options.qp);
  if (options.qp == 0 || key_qp == 0) {
    return core::Status::unsupported(
        std::string(kWho) +
        ": VideoToolbox codes QP 1 to 51, and QP 0 as 1 (Options::" +
        (options.qp == 0 ? "qp" : "key_qp") + ")");
  }
  const std::string what = "a hardware, low-latency HEVC session of " +
                           size_text(options.width, options.height);
  auto impl = std::make_unique<Impl>();
  Impl& s = *impl;
  s.options = options;

  // Hardware only, low latency: P-pictures only, with no reordering or
  // look-ahead.
  const auto spec = cf_dictionary();
  CFDictionarySetValue(
      spec.get(),
      kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,
      kCFBooleanTrue);
  CFDictionarySetValue(spec.get(),
                       kVTVideoEncoderSpecification_EnableLowLatencyRateControl,
                       kCFBooleanTrue);
  // Its pool's pictures, and what it is sent: NV12 at the session's range on
  // IOSurfaces Metal reads and writes.
  const OSType pixel_format =
      options.color.full_range
          ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
          : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  const auto pictures = cf_dictionary();
  const auto format_number = number(static_cast<std::int32_t>(pixel_format));
  const auto width_number = number(static_cast<std::int32_t>(options.width));
  const auto height_number = number(static_cast<std::int32_t>(options.height));
  const auto surface = cf_dictionary();
  CFDictionarySetValue(pictures.get(), kCVPixelBufferPixelFormatTypeKey,
                       format_number.get());
  CFDictionarySetValue(pictures.get(), kCVPixelBufferWidthKey,
                       width_number.get());
  CFDictionarySetValue(pictures.get(), kCVPixelBufferHeightKey,
                       height_number.get());
  CFDictionarySetValue(pictures.get(), kCVPixelBufferIOSurfacePropertiesKey,
                       surface.get());
  CFDictionarySetValue(pictures.get(), kCVPixelBufferMetalCompatibilityKey,
                       kCFBooleanTrue);
  VTCompressionSessionRef made = nullptr;
  OSStatus status = VTCompressionSessionCreate(
      kCFAllocatorDefault, static_cast<std::int32_t>(options.width),
      static_cast<std::int32_t>(options.height), kCMVideoCodecType_HEVC,
      spec.get(), pictures.get(), kCFAllocatorDefault, &Impl::on_output,
      impl.get(), &made);
  if (status == kVTVideoEncoderNotAvailableNowErr) {
    return vt_error(kWho,
                    "the hardware HEVC encoder is busy, or at its session "
                    "limit: " +
                        what + " did not open; try once another closes",
                    status);
  }
  if (status == kVTCouldNotFindVideoEncoderErr) {
    return vt_error(kWho,
                    "no hardware HEVC encoder here opens " + what +
                        " (VideoToolbox has one on Apple silicon)",
                    status);
  }
  if (status != noErr) return vt_error(kWho, "opening " + what, status);
  s.session.reset(made);

  // One temporal layer, Main, no B-pictures; the VUI's colour as Core Video
  // names each code, none where it is unspecified (2).
  float one = 1.0F;
  const CfPtr<CFNumberRef> whole(
      CFNumberCreate(kCFAllocatorDefault, kCFNumberFloat32Type, &one));
  VKC_TRY(
      s.set(kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse));
  VKC_TRY(
      s.set(kVTCompressionPropertyKey_BaseLayerFrameRateFraction, whole.get()));
  VKC_TRY(s.set(kVTCompressionPropertyKey_ProfileLevel,
                kVTProfileLevel_HEVC_Main_AutoLevel));
  const ColorCodes& color = options.color;
  VKC_TRY(s.set(kVTCompressionPropertyKey_YCbCrMatrix,
                CVYCbCrMatrixGetStringForIntegerCodePoint(color.matrix)));
  if (color.transfer != 2) {
    VKC_TRY(
        s.set(kVTCompressionPropertyKey_TransferFunction,
              CVTransferFunctionGetStringForIntegerCodePoint(color.transfer)));
  }
  if (color.primaries != 2) {
    VKC_TRY(
        s.set(kVTCompressionPropertyKey_ColorPrimaries,
              CVColorPrimariesGetStringForIntegerCodePoint(color.primaries)));
  }
  status = VTCompressionSessionPrepareToEncodeFrames(s.session.get());
  if (status != noErr) return vt_error(kWho, "preparing " + what, status);

  // A QP on every picture, or no session: its other rate modes take none.
  if (property_text(s.session.get(),
                    kVTCompressionPropertyKey_SupportsBaseFrameQP) != "on") {
    return core::Status::unsupported(
        std::string(kWho) +
        ": the hardware HEVC encoder takes no QP a picture "
        "(SupportsBaseFrameQP) in " +
        what);
  }

  HevcEncoder::Facts read;
  read.encoder =
      property_text(s.session.get(), kVTCompressionPropertyKey_EncoderID);
  read.color = color;
  const auto p = [&s](CFStringRef key) {
    return property_text(s.session.get(), key);
  };
  // What the session reports of the properties set, and the specification
  // as asked (the session reports neither of its keys back).
  read.settings =
      "VideoToolbox HEVC " + size_text(options.width, options.height) +
      ": hardware required, low-latency rate control, profile " +
      p(kVTCompressionPropertyKey_ProfileLevel) + ", reordering " +
      p(kVTCompressionPropertyKey_AllowFrameReordering) +
      ", base layer fraction " +
      p(kVTCompressionPropertyKey_BaseLayerFrameRateFraction) +
      ", base frame QP " + p(kVTCompressionPropertyKey_SupportsBaseFrameQP) +
      " (P " + std::to_string(options.qp) + ", IDR " + std::to_string(key_qp) +
      "), real time " + p(kVTCompressionPropertyKey_RealTime) + ", YCbCr " +
      p(kVTCompressionPropertyKey_YCbCrMatrix) + ", transfer " +
      p(kVTCompressionPropertyKey_TransferFunction) + ", primaries " +
      p(kVTCompressionPropertyKey_ColorPrimaries) + ", " +
      fourcc(pixel_format) + ", siting " +
      std::to_string(color.chroma_location) + " from the buffer";
  facts = std::move(read);
  return std::unique_ptr<HevcEncodeSession>(
      new HevcEncodeSession(std::move(impl)));
} catch (...) {
  return core::exception_status("HevcEncoder::create",
                                core::Status::Code::OutOfMemory);
}

core::Result<bool> HevcEncodeSession::encode(
    EncodePicture& picture, bool key, bool drop,
    std::vector<std::uint8_t>& access_unit) try {
  Impl& s = *impl_;
  access_unit.clear();
  CVPixelBufferRef pixels = picture.pixels.get();
  // VideoToolbox writes the stream's siting from the buffer's tag.
  const int siting = s.options.color.chroma_location;
  if (siting != 0) {
    CFTypeRef tagged = CVBufferCopyAttachment(
        pixels, kCVImageBufferChromaLocationTopFieldKey, nullptr);
    if (tagged != nullptr) {
      CFRelease(tagged);
    } else {
      CFStringRef location = sensor::detail::chroma_location_string(siting);
      CVBufferSetAttachment(pixels, kCVImageBufferChromaLocationTopFieldKey,
                            location, kCVAttachmentMode_ShouldPropagate);
      CVBufferSetAttachment(pixels, kCVImageBufferChromaLocationBottomFieldKey,
                            location, kCVAttachmentMode_ShouldPropagate);
    }
  }
  const int qp = key ? s.options.key_qp.value_or(s.options.qp) : s.options.qp;
  const CfPtr<CFNumberRef> frame_qp = number(qp);
  std::array<const void*, 2> keys = {kVTEncodeFrameOptionKey_BaseFrameQP,
                                     kVTEncodeFrameOptionKey_ForceKeyFrame};
  std::array<const void*, 2> values = {frame_qp.get(),
                                       key ? kCFBooleanTrue : kCFBooleanFalse};
  const CfPtr<CFDictionaryRef> frame_options(CFDictionaryCreate(
      kCFAllocatorDefault, keys.data(), values.data(), 2,
      &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
  if (frame_qp == nullptr || frame_options == nullptr) {
    return core::Status::out_of_memory(std::string(kWho) +
                                       ": a picture's options");
  }
  const std::uint64_t frame = ++s.frames;
  {
    const std::lock_guard<std::mutex> lock(s.mutex);
    if (s.extra) {
      return core::Status::io_error(
          std::string(kWho) +
          ": the encoder gave two access units for one "
          "picture");
    }
    s.waiting = frame;
    s.want_key = key;
    s.drop = drop;
    s.out = &access_unit;
    s.done = false;
    s.status = noErr;
    s.dropped = false;
    s.failure = {};
  }
  // Not under the lock: the output may come on this thread.
  VTEncodeInfoFlags info = 0;
  const OSStatus submitted = VTCompressionSessionEncodeFrame(
      s.session.get(), pixels, CMTimeMake(static_cast<std::int64_t>(frame), 30),
      kCMTimeInvalid, frame_options.get(),
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(frame)), &info);
  std::unique_lock<std::mutex> lock(s.mutex);
  const bool came =
      submitted == noErr && (info & kVTEncodeInfo_FrameDropped) == 0 &&
      s.ready.wait_for(lock, kOutputWait, [&s] { return s.done; });
  const bool done = s.done;
  s.waiting = 0;
  s.out = nullptr;
  if (submitted != noErr) return vt_error(kWho, "coding a picture", submitted);
  if (!came && !done) {
    if ((info & kVTEncodeInfo_FrameDropped) != 0) return false;
    return core::Status::io_error(std::string(kWho) +
                                  ": no access unit came out within 2 s");
  }
  if (s.status != noErr) return vt_error(kWho, "coding a picture", s.status);
  if (s.dropped) {
    access_unit.clear();
    return false;
  }
  if (!s.failure.ok()) return s.failure;
  return true;
} catch (...) {
  return core::exception_status("HevcEncoder: coding a picture",
                                core::Status::Code::OutOfMemory);
}

void HevcEncodeSession::lose() noexcept {
  VTCompressionSessionInvalidate(impl_->session.get());
}

}  // namespace volumetric_kit::sensor::video
