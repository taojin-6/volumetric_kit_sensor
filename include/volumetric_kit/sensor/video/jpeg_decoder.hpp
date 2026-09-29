// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/jpeg_decoder.hpp
/// @brief A baseline JPEG decoder for MJPEG colour: on the GPU with nvJPEG,
///        where the picture stays, else in software to host planes.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"
#include "volumetric_kit/recon/sensor/video/export.hpp"

namespace volumetric_kit::recon {
class Device;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::sensor {

/// @brief What a @ref JpegDecoder decodes on.
enum class JpegDecodeBackend {
  Software,  ///< FFmpeg's decoder, to host planes.
  /// nvJPEG on the GPU's hardware JPEG engine, and on its cores for a JPEG
  /// the engine refuses (past 16384 pixels a side).
  NvjpegHardware,
  NvjpegGpu,     ///< nvJPEG on the GPU's cores, Huffman decoding included.
  VideoToolbox,  ///< VideoToolbox's hardware JPEG decoder, on Apple.
};

/// @return @p backend's name: `software`, `nvjpeg-hardware`, `nvjpeg-gpu`,
///         `videotoolbox`.
VR_SENSOR_VIDEO_API const char* to_string(JpegDecodeBackend backend) noexcept;

/// @brief Decodes one baseline JPEG at a time, as an MJPEG camera sends them.
///
/// Given a device on an NVIDIA GPU, in a build with VR_WITH_CUDA (Linux),
/// nvJPEG decodes a baseline 8-bit 4:2:0 JPEG straight into a Vulkan buffer
/// CUDA has imported, and the picture stays on the GPU: its hardware JPEG
/// engine where the GPU has one, and its cores for the rest. libcuda and
/// libnvjpeg are loaded at run time, so without them the decoder runs in
/// software. On Apple, given a device that imports Metal textures,
/// VideoToolbox's hardware decoder takes the same JPEGs into NV12 on an
/// IOSurface, whose two planes are handed out as images and never reach the
/// host. Anything else decodes in software to host planes: another
/// subsampling, which is converted to 4:2:0, a device on another GPU, or no
/// device at all.
/// The picture is BT.601 full range as JFIF defines it: I420 (`Yuv420`), or
/// NV12 images from VideoToolbox.
///
/// @warning Not thread-safe: use from one thread.
class VR_SENSOR_VIDEO_API JpegDecoder {
 public:
  /// @brief How @ref create opens a decoder.
  struct Options {
    /// The device to decode on and hand pictures out on, as @ref
    /// JpegDecoder says; null decodes in software, and so does every JPEG
    /// after the device path fails once. Borrowed: it must outlive the
    /// decoder and every picture on it.
    const Device* device = nullptr;
    /// Set FFmpeg's log level to ERROR. Process-wide: FFmpeg has one logger.
    bool configure_ffmpeg_logging = true;
  };

  /// @return The decoder; or @ref Status::Code::IoError if FFmpeg's decoder
  ///         does not open. A device nvJPEG cannot use is not an error: the
  ///         decoder runs in software instead, as @ref backend says.
  static Result<JpegDecoder> create(const Options& options);

  ~JpegDecoder();
  JpegDecoder(JpegDecoder&& other) noexcept;
  JpegDecoder& operator=(JpegDecoder&& other) noexcept;
  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;

  /// @brief Decode one JPEG.
  /// @param data  The JPEG's bytes, SOI to EOI; read during the call only.
  /// @param size  Their count.
  /// @return The picture. I420 on the device (@ref DecodedPicture::device),
  ///         held by the picture and written by CUDA, so a reader takes it
  ///         over from `VK_QUEUE_FAMILY_EXTERNAL`; NV12 as images
  ///         (@ref DecodedPicture::image), held by the picture, from
  ///         VideoToolbox; or I420 host planes, valid until the next call. @ref
  ///         Status::Code::InvalidArgument for no bytes, more than 2 GiB of
  ///         them, a JPEG in a pixel format swscale cannot read, or a
  ///         moved-from decoder;
  ///         @ref Status::Code::IoError for bytes that do not decode.
  Result<DecodedPicture> decode(const std::uint8_t* data, std::size_t size);

  /// @return What this decoder decodes a 4:2:0 JPEG on: Software once the
  ///         device path has failed.
  JpegDecodeBackend backend() const noexcept;

 private:
  struct Impl;
  explicit JpegDecoder(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
