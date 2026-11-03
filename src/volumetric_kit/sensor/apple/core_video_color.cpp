// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/apple/core_video_color.hpp"

#include <string>

#include "core_video_tags.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::sensor {

core::Result<ColorCodes> color_codes_of(CVPixelBufferRef pixels) try {
  if (pixels == nullptr) {
    return core::Status::invalid_argument("color_codes_of: no pixel buffer");
  }
  const detail::CoreVideoTags tags = detail::core_video_tags(pixels);
  if (!tags.unknown.empty()) {
    return core::Status::invalid_argument("color_codes_of: " + tags.unknown);
  }
  ColorCodes codes;
  if (tags.matrix) codes.matrix = *tags.matrix;
  if (tags.transfer) codes.transfer = *tags.transfer;
  if (tags.primaries) codes.primaries = *tags.primaries;
  if (tags.chroma_location) codes.chroma_location = *tags.chroma_location;
  if (tags.full_range) codes.full_range = *tags.full_range;
  return codes;
} catch (...) {
  return core::exception_status("color_codes_of",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor
