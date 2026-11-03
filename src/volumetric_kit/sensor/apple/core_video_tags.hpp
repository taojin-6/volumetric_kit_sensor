// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Core Video's colour tags and H.273 codes, both ways: Core Video's own
// mapping for the matrix, transfer and primaries, and the six
// chroma-location strings. color_codes_of (core_video_color.cpp) reads a
// buffer's tags with it, and the VideoToolbox encoder (sensor_video) checks
// and writes them with it. Header-only, so the two targets share it with no
// symbol between them. Internal.

#include <CoreVideo/CoreVideo.h>

#include <optional>
#include <string>

namespace volumetric_kit::sensor::detail {

// The chroma-location string ChromaSampleLocType @p code names (0 to 5);
// null for any other.
inline CFStringRef chroma_location_string(int code) noexcept {
  switch (code) {
    case 0:
      return kCVImageBufferChromaLocation_Left;
    case 1:
      return kCVImageBufferChromaLocation_Center;
    case 2:
      return kCVImageBufferChromaLocation_TopLeft;
    case 3:
      return kCVImageBufferChromaLocation_Top;
    case 4:
      return kCVImageBufferChromaLocation_BottomLeft;
    case 5:
      return kCVImageBufferChromaLocation_Bottom;
    default:
      return nullptr;
  }
}

// The ChromaSampleLocType @p location names; empty for a string none does.
inline std::optional<int> chroma_location_code(CFStringRef location) noexcept {
  for (int code = 0; code <= 5; ++code) {
    if (CFEqual(location, chroma_location_string(code))) return code;
  }
  return std::nullopt;
}

// @p text as UTF-8; "(not text)" for a value that is no string.
inline std::string utf8(CFTypeRef text) {
  if (text == nullptr || CFGetTypeID(text) != CFStringGetTypeID()) {
    return "(not text)";
  }
  const auto s = static_cast<CFStringRef>(text);
  const CFIndex size = CFStringGetMaximumSizeForEncoding(
                           CFStringGetLength(s), kCFStringEncodingUTF8) +
                       1;
  std::string out(static_cast<std::size_t>(size), '\0');
  if (!CFStringGetCString(s, out.data(), size, kCFStringEncodingUTF8)) {
    return "(not text)";
  }
  out.resize(std::char_traits<char>::length(out.c_str()));
  return out;
}

// What a pixel buffer's colour tags say, each as its code where it has the
// tag; the first tag Core Video names no code for, described, in unknown.
struct CoreVideoTags {
  std::optional<int> matrix;
  std::optional<int> transfer;
  std::optional<int> primaries;
  std::optional<int> chroma_location;  // the top field's
  std::optional<bool> full_range;      // from the format: '420f' or '420v'
  std::string unknown;
};

inline CoreVideoTags core_video_tags(CVPixelBufferRef pixels) {
  CoreVideoTags tags;
  // Each attachment copied, read, and released.
  const auto read = [&](CFStringRef key, const char* what,
                        int (*code_of)(CFStringRef), std::optional<int>& out) {
    CFTypeRef value = CVBufferCopyAttachment(pixels, key, nullptr);
    if (value == nullptr) return;
    // 2 is what Core Video gives a string it does not know: no tag says
    // "unspecified".
    const int code = CFGetTypeID(value) == CFStringGetTypeID()
                         ? code_of(static_cast<CFStringRef>(value))
                         : 2;
    if (code == 2) {
      if (tags.unknown.empty()) {
        tags.unknown = std::string("its ") + what + " tag '" + utf8(value) +
                       "', which Core Video names no code point for";
      }
    } else {
      out = code;
    }
    CFRelease(value);
  };
  read(kCVImageBufferYCbCrMatrixKey, "YCbCr matrix",
       &CVYCbCrMatrixGetIntegerCodePointForString, tags.matrix);
  read(kCVImageBufferTransferFunctionKey, "transfer function",
       &CVTransferFunctionGetIntegerCodePointForString, tags.transfer);
  read(kCVImageBufferColorPrimariesKey, "colour primaries",
       &CVColorPrimariesGetIntegerCodePointForString, tags.primaries);
  if (CFTypeRef location = CVBufferCopyAttachment(
          pixels, kCVImageBufferChromaLocationTopFieldKey, nullptr)) {
    const std::optional<int> code =
        CFGetTypeID(location) == CFStringGetTypeID()
            ? chroma_location_code(static_cast<CFStringRef>(location))
            : std::nullopt;
    if (code) {
      tags.chroma_location = code;
    } else if (tags.unknown.empty()) {
      tags.unknown = "its chroma location tag '" + utf8(location) +
                     "', which names no ChromaSampleLocType";
    }
    CFRelease(location);
  }
  switch (CVPixelBufferGetPixelFormatType(pixels)) {
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
      tags.full_range = true;
      break;
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
      tags.full_range = false;
      break;
    default:
      break;
  }
  return tags;
}

}  // namespace volumetric_kit::sensor::detail
