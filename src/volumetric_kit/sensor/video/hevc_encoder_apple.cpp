// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// send_pixel_buffer: the buffer checked against the session, then sent as it
// is through HevcEncoder's own send (hevc_encoder_access.hpp).

#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"

#include <optional>
#include <string>
#include <utility>

#include "../apple/core_video_tags.hpp"
#include "frame_color.hpp"
#include "hevc_encoder_access.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::sensor {
namespace {

core::Status refused(const std::string& why) {
  return core::Status::invalid_argument("send_pixel_buffer: " + why);
}

std::string fourcc(OSType type) {
  std::string out = "'";
  for (int shift = 24; shift >= 0; shift -= 8) {
    out += static_cast<char>((type >> shift) & 0xff);
  }
  return out + "'";
}

// Empty where the codes agree; else which differs.
std::string differs(const char* what, int tagged, int session) {
  if (tagged == session) return {};
  return std::string("its ") + what + " tag is " + std::to_string(tagged) +
         ", the session's " + std::to_string(session);
}

}  // namespace

core::Status send_pixel_buffer(HevcEncoder& encoder, CVPixelBufferRef pixels,
                               bool key, std::int64_t pts) try {
  const HevcEncoder::Options* o = HevcEncoderAccess::options(encoder);
  if (o == nullptr) return refused("the encoder is moved from");
  if (pixels == nullptr) return refused("no pixel buffer");
  if (CVPixelBufferGetIOSurface(pixels) == nullptr) {
    return refused("a pixel buffer not on an IOSurface");
  }
  const OSType want = o->color.full_range
                          ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                          : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  const OSType format = CVPixelBufferGetPixelFormatType(pixels);
  if (format != want) {
    return refused("a pixel buffer of format " + fourcc(format) +
                   ", not the session's " + fourcc(want));
  }
  const std::size_t width = CVPixelBufferGetWidth(pixels);
  const std::size_t height = CVPixelBufferGetHeight(pixels);
  if (width != o->width || height != o->height) {
    return refused("a pixel buffer of " + std::to_string(width) + "x" +
                   std::to_string(height) + ", not the session's " +
                   std::to_string(o->width) + "x" + std::to_string(o->height));
  }
  const detail::CoreVideoTags tags = detail::core_video_tags(pixels);
  if (!tags.unknown.empty()) return refused(tags.unknown);
  // A tag it lacks is the session's; the rest, normalized, must be too.
  ColorCodes tagged = o->color;
  if (tags.matrix) tagged.matrix = *tags.matrix;
  if (tags.transfer) tagged.transfer = *tags.transfer;
  if (tags.primaries) tagged.primaries = *tags.primaries;
  if (tags.chroma_location) tagged.chroma_location = *tags.chroma_location;
  tagged = video::normalize_codes(tagged);
  for (const std::string& d :
       {differs("matrix", tagged.matrix, o->color.matrix),
        differs("transfer", tagged.transfer, o->color.transfer),
        differs("primaries", tagged.primaries, o->color.primaries),
        differs("chroma location", tagged.chroma_location,
                o->color.chroma_location)}) {
    if (!d.empty()) return refused(d);
  }
  // VideoToolbox writes the stream's siting from the buffer's tag, so one
  // that lacks it is tagged with the session's: here, on the caller's
  // thread, before the encoder holds it.
  if (!tags.chroma_location && o->color.chroma_location != 0) {
    const CFStringRef location =
        detail::chroma_location_string(o->color.chroma_location);
    CVBufferSetAttachment(pixels, kCVImageBufferChromaLocationTopFieldKey,
                          location, kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(pixels, kCVImageBufferChromaLocationBottomFieldKey,
                          location, kCVAttachmentMode_ShouldPropagate);
  }
  video::EncodePicture picture;
  picture.pixels.reset(CVPixelBufferRetain(pixels));
  return HevcEncoderAccess::send(encoder, std::move(picture), key, pts);
} catch (...) {
  return core::exception_status("send_pixel_buffer",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor
