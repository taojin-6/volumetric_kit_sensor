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
#include <optional>

#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon {
class Device;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::sensor::video {

class VtPictures;

class VtJpeg {
 public:
  // Null where VideoToolbox has no hardware JPEG decoder or @p device imports
  // no Metal textures. @p who names the decoder in errors.
  static std::unique_ptr<VtJpeg> open(const Device& device, const char* who);
  ~VtJpeg();
  VtJpeg(const VtJpeg&) = delete;
  VtJpeg& operator=(const VtJpeg&) = delete;

  // The JPEG as NV12 images; empty for one this does not take -- not
  // baseline 8-bit 4:2:0, or refused by the hardware -- which goes to
  // software instead. An error means the device path failed.
  Result<std::optional<DecodedPicture>> decode(const std::uint8_t* data,
                                               std::size_t size);

 private:
  VtJpeg() = default;
  // A session for JPEGs of this size; false where the hardware takes none.
  bool start(std::uint32_t width, std::uint32_t height);
  void stop() noexcept;

  const char* who_ = nullptr;
  std::unique_ptr<VtPictures> pictures_;
  CMVideoFormatDescriptionRef format_ = nullptr;
  VTDecompressionSessionRef session_ = nullptr;
  std::uint32_t width_ = 0;  // the session's size
  std::uint32_t height_ = 0;
  std::uint32_t refused_width_ = 0;  // the last size no session opened for
  std::uint32_t refused_height_ = 0;
};

}  // namespace volumetric_kit::recon::sensor::video
