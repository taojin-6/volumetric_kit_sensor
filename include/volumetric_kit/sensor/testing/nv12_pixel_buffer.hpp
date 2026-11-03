// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/nv12_pixel_buffer.hpp
/// @brief Test support, Apple only: a host NV12 picture as a `CVPixelBuffer`
///        on an IOSurface, as a camera hands one out, tagged with its colour
///        through Core Video's own strings. Header-only, on Core Video.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// const ColorCodes bt601_full{6, 1, 6, true, 0};
/// VKC_ASSIGN(CVPixelBufferRef pixels,
///            testing::pixel_buffer(testing::test_pattern(1920, 1440, i),
///                                  bt601_full));
/// VKC_TRY(send_pixel_buffer(encoder, pixels, key, pts));
/// CVPixelBufferRelease(pixels);
/// @endcode

#if !defined(__APPLE__)
#error "testing/nv12_pixel_buffer.hpp is Apple only"
#endif

#include <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"

namespace volumetric_kit::sensor::testing {

/// @return @p picture as an 8-bit NV12 `CVPixelBuffer` on an IOSurface,
///         Metal-compatible: '420f' where @p color is full range, '420v'
///         where not, tagged with @p color's matrix, transfer and primaries
///         as Core Video names them (no tag for 2), and with no
///         chroma-location tag. The caller releases it
///         (`CVPixelBufferRelease`). `Status::Code::OutOfMemory` where Core
///         Video makes or locks no buffer; `Status::Code::InvalidArgument`
///         for planes of other lengths than its size's.
inline core::Result<CVPixelBufferRef> pixel_buffer(const HostNv12& picture,
                                                   const ColorCodes& color) {
  const std::uint32_t w = picture.width;
  const std::uint32_t h = picture.height;
  const std::size_t chroma_row = 2 * std::size_t{(w + 1) / 2};
  if (w == 0 || h == 0 || picture.luma.size() != std::size_t{w} * h ||
      picture.chroma.size() != chroma_row * ((h + 1) / 2)) {
    return core::Status::invalid_argument(
        "pixel_buffer: an NV12 picture of no size or with planes of other "
        "lengths");
  }
  CFMutableDictionaryRef attrs = CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFMutableDictionaryRef surface = CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CVPixelBufferRef out = nullptr;
  CVReturn made = kCVReturnError;
  if (attrs != nullptr && surface != nullptr) {
    CFDictionarySetValue(attrs, kCVPixelBufferIOSurfacePropertiesKey, surface);
    CFDictionarySetValue(attrs, kCVPixelBufferMetalCompatibilityKey,
                         kCFBooleanTrue);
    made = CVPixelBufferCreate(
        kCFAllocatorDefault, w, h,
        color.full_range ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                         : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        attrs, &out);
  }
  if (surface != nullptr) CFRelease(surface);
  if (attrs != nullptr) CFRelease(attrs);
  if (made != kCVReturnSuccess || out == nullptr) {
    return core::Status::out_of_memory("pixel_buffer: no " + std::to_string(w) +
                                       "x" + std::to_string(h) +
                                       " pixel buffer");
  }
  if (CVPixelBufferLockBaseAddress(out, 0) != kCVReturnSuccess) {
    CVPixelBufferRelease(out);
    return core::Status::out_of_memory(
        "pixel_buffer: Core Video does not lock the buffer");
  }
  const std::uint8_t* planes[2] = {picture.luma.data(), picture.chroma.data()};
  const std::size_t row_bytes[2] = {w, chroma_row};
  const std::uint32_t rows[2] = {h, (h + 1) / 2};
  for (std::size_t k = 0; k < 2; ++k) {
    auto* base =
        static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(out, k));
    const std::size_t stride = CVPixelBufferGetBytesPerRowOfPlane(out, k);
    for (std::uint32_t r = 0; r < rows[k]; ++r) {
      std::copy_n(planes[k] + r * row_bytes[k], row_bytes[k],
                  base + r * stride);
    }
  }
  CVPixelBufferUnlockBaseAddress(out, 0);
  const auto tag = [out](CFStringRef key, int code, CFStringRef value) {
    if (code != 2 && value != nullptr) {
      CVBufferSetAttachment(out, key, value, kCVAttachmentMode_ShouldPropagate);
    }
  };
  tag(kCVImageBufferYCbCrMatrixKey, color.matrix,
      CVYCbCrMatrixGetStringForIntegerCodePoint(color.matrix));
  tag(kCVImageBufferTransferFunctionKey, color.transfer,
      CVTransferFunctionGetStringForIntegerCodePoint(color.transfer));
  tag(kCVImageBufferColorPrimariesKey, color.primaries,
      CVColorPrimariesGetStringForIntegerCodePoint(color.primaries));
  return out;
}

}  // namespace volumetric_kit::sensor::testing
