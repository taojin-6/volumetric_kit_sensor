// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// VideoToolbox's decoder sessions, the one place the HEVC and JPEG decoders
// (Apple) open them and decode through them: NV12 on IOSurfaces Metal reads,
// which PixelBufferImporter hands to Vulkan as images, on the hardware
// decoder or not at all. Internal.

#include <VideoToolbox/VideoToolbox.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

#include "volumetric_kit/core/base/result.hpp"

namespace volumetric_kit::sensor::video {

// A Core Foundation object, released with its last owner.
struct CfRelease {
  void operator()(CFTypeRef ref) const noexcept { CFRelease(ref); }
};
template <class Ref>
using CfPtr = std::unique_ptr<std::remove_pointer_t<Ref>, CfRelease>;
using PixelBufferPtr = CfPtr<CVPixelBufferRef>;
using FormatPtr = CfPtr<CMVideoFormatDescriptionRef>;

// A session, invalidated and released with its owner.
struct SessionClose {
  void operator()(VTDecompressionSessionRef session) const noexcept {
    VTDecompressionSessionInvalidate(session);
    CFRelease(session);
  }
};
using SessionPtr =
    std::unique_ptr<std::remove_pointer_t<VTDecompressionSessionRef>,
                    SessionClose>;

// A VideoToolbox failure as the decoders report it, @p who and @p what naming
// it: bad data IoError (kVTParameterErr too while @p decoding, which a
// truncated picture returns), a format or decoder the hardware lacks
// Unsupported, an allocation OutOfMemory, and anything else a Backend error
// keeping its code.
core::Status vt_error(const char* who, const std::string& what, OSStatus error,
                      bool decoding = false);

// A session decoding @p format on the hardware decoder, with no software
// fallback, into NV12 at full or video range as @p full_range says, on
// Metal-compatible IOSurfaces. @p what names it in errors.
core::Result<SessionPtr> open_session(CMVideoFormatDescriptionRef format,
                                      bool full_range, const char* who,
                                      const std::string& what);

// @p size bytes at @p data decoded on @p session as one sample of @p format,
// before this returns: its picture, which is null if the decoder gave none.
// The bytes are wrapped, not copied, and borrowed for the call only.
core::Result<PixelBufferPtr> decode_sample(VTDecompressionSessionRef session,
                                           CMVideoFormatDescriptionRef format,
                                           const std::uint8_t* data,
                                           std::size_t size, const char* who,
                                           const std::string& what);

}  // namespace volumetric_kit::sensor::video
