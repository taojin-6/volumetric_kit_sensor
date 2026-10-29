// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// JPEGs through VideoToolbox's hardware decoder (Apple), into NV12 on an
// IOSurface whose planes PixelBufferImporter hands to Vulkan as images, so an
// MJPEG camera's picture never reaches the host.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "videotoolbox.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/recon/sensor/apple/pixel_buffer_importer.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::video {

class VtJpeg {
 public:
  // Unsupported where VideoToolbox has no hardware JPEG decoder or @p device
  // imports no Metal textures. @p who names the decoder in errors. The
  // pictures are shared with @p queue_families, as PixelBufferImporter::create
  // takes them, and InvalidArgument is its refusal of them.
  static core::Result<std::unique_ptr<VtJpeg>> open(
      const core::Device& device, const char* who,
      const std::uint32_t* queue_families, std::uint32_t queue_family_count);
  VtJpeg(const VtJpeg&) = delete;
  VtJpeg& operator=(const VtJpeg&) = delete;

  // The JPEG as NV12 images, its colour left to JpegDecoder::decode, with the
  // codes it documents:
  // IoError for bytes that do not decode, Unsupported for a JPEG the hardware
  // does not take -- not baseline 8-bit 4:2:0, past the device's image
  // extent, or a size it does not support -- and OutOfMemory or Backend
  // if the device path fails.
  core::Result<DecodedPicture> decode(const std::uint8_t* data,
                                      std::size_t size);

 private:
  explicit VtJpeg(PixelBufferImporter pictures) noexcept
      : pictures_(std::move(pictures)) {}
  // A session for JPEGs of this size, preserving the platform's failure code.
  core::Status start(std::uint32_t width, std::uint32_t height);
  void stop() noexcept;

  const char* who_ = nullptr;
  PixelBufferImporter pictures_;
  std::uint32_t max_extent_ = 0;  // the device's maxImageDimension2D
  FormatPtr format_;
  SessionPtr session_;
  std::uint32_t width_ = 0;  // the session's size
  std::uint32_t height_ = 0;
};

}  // namespace volumetric_kit::recon::sensor::video
