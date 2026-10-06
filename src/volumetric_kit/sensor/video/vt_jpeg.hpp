// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// JPEGs through VideoToolbox's hardware decoder (Apple), into NV12 on an
// IOSurface whose planes VtPictures hands to Vulkan as images, so an MJPEG
// camera's picture never reaches the host.

#include <VideoToolbox/VideoToolbox.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::video {

class VtPictures;

class VtJpeg {
 public:
  // Unsupported where VideoToolbox has no hardware JPEG decoder or @p device
  // imports no Metal textures. @p who names the decoder in errors.
  static core::Result<std::unique_ptr<VtJpeg>> open(const core::Device& device,
                                                    const char* who);
  ~VtJpeg();
  VtJpeg(const VtJpeg&) = delete;
  VtJpeg& operator=(const VtJpeg&) = delete;

  // The JPEG as NV12 images, with the codes JpegDecoder::decode documents:
  // IoError for bytes that do not decode, Unsupported for a JPEG the hardware
  // does not take -- not baseline 8-bit 4:2:0, past the device's image
  // extent, or a size it does not support -- and OutOfMemory or Backend
  // if the device path fails.
  core::Result<DecodedPicture> decode(const std::uint8_t* data,
                                      std::size_t size);

 private:
  VtJpeg() = default;
  // A session for JPEGs of this size, preserving the platform's failure code.
  core::Status start(std::uint32_t width, std::uint32_t height);
  void stop() noexcept;

  const char* who_ = nullptr;
  std::unique_ptr<VtPictures> pictures_;
  std::uint32_t max_extent_ = 0;  // the device's maxImageDimension2D
  CMVideoFormatDescriptionRef format_ = nullptr;
  VTDecompressionSessionRef session_ = nullptr;
  std::uint32_t width_ = 0;  // the session's size
  std::uint32_t height_ = 0;
};

}  // namespace volumetric_kit::recon::sensor::video
