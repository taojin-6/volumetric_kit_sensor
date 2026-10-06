// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/jpeg_decoder.hpp
/// @brief A baseline JPEG decoder for MJPEG colour, on the GPU's hardware,
///        handing out pictures on the device only.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"
#include "volumetric_kit/recon/sensor/video/export.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief Decodes one baseline 8-bit 4:2:0 JPEG at a time, as an MJPEG camera
///        sends them, on the hardware of the device it is given, and hands
///        the picture out there.
///
/// In a build with VR_WITH_CUDA (Linux), on an NVIDIA GPU with a hardware
/// JPEG engine, nvJPEG decodes straight into a Vulkan buffer CUDA has
/// imported: I420 (`Yuv420`). libcuda and libnvjpeg are loaded at run time.
/// On Apple, VideoToolbox's hardware decoder decodes into NV12 on an
/// IOSurface, whose two planes are handed out as images. The picture is
/// BT.601 full range as JFIF defines it, its chroma centred.
///
/// @warning Not thread-safe: use from one thread.
class VR_SENSOR_VIDEO_API JpegDecoder {
 public:
  /// @brief How @ref create opens a decoder.
  struct Options {
    /// The device the pictures are handed out on, whose GPU decodes them: one
    /// that imports Metal textures on Apple, one that exports memory and is a
    /// CUDA device elsewhere. Required. Borrowed: it must outlive the decoder
    /// and every picture on it.
    const core::Device* device = nullptr;
    /// The allocator nvJPEG's picture buffers are made through: exported
    /// device-only memory, counted against its heap's budget like any other
    /// allocation. Required with nvJPEG; VideoToolbox's pictures need none.
    /// Borrowed: it must outlive the decoder and every picture on it.
    core::Allocator* allocator = nullptr;
  };

  /// @return The decoder; `Status::Code::Unsupported` if there is no device
  ///         path: no @ref Options::device (or, for nvJPEG, no
  ///         @ref Options::allocator), a device whose GPU has no hardware JPEG
  ///         decoder this build reaches, or libcuda or libnvjpeg that do not
  ///         load; or `Status::Code::Backend` if CUDA or nvJPEG fails.
  static core::Result<JpegDecoder> create(const Options& options);

  ~JpegDecoder();
  JpegDecoder(JpegDecoder&& other) noexcept;
  JpegDecoder& operator=(JpegDecoder&& other) noexcept;
  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;

  /// @brief Decode one JPEG.
  /// @param data  The JPEG's bytes, SOI to EOI; read during the call only.
  /// @param size  Their count.
  /// @return The picture on the device: I420 in a buffer
  ///         (@ref DecodedPicture::device) from nvJPEG, written by CUDA, so a
  ///         reader takes it over from `VK_QUEUE_FAMILY_EXTERNAL`; or NV12
  ///         images (@ref DecodedPicture::image) from VideoToolbox. Or:
  ///         - `Status::Code::IoError` for bytes that do not decode;
  ///         - `Status::Code::Unsupported` for a JPEG the hardware cannot
  ///           decode: not baseline 8-bit 4:2:0 in three components (4:2:2,
  ///           say), or larger than it takes;
  ///         - `Status::Code::Backend` or `Status::Code::OutOfMemory` if the
  ///           device path fails;
  ///         - `Status::Code::InvalidArgument` for no bytes or a moved-from
  ///           decoder.
  core::Result<DecodedPicture> decode(const std::uint8_t* data,
                                      std::size_t size);

 private:
  struct Impl;
  explicit JpegDecoder(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
