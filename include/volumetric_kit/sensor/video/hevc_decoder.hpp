// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/hevc_decoder.hpp
/// @brief An H.265/HEVC decoder on the GPU's hardware, handing out pictures on
///        the device only.
///
/// Nothing camera-specific: bytes in, pictures out. No FFmpeg type appears
/// here; linking still needs FFmpeg's libraries, which the target carries.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"
#include "volumetric_kit/recon/sensor/video/export.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief Decodes an H.265 elementary stream (Annex B), one access unit at a
///        time in decode order, on the hardware of the device it is given:
///        VideoToolbox on Apple, NVDEC through FFmpeg in a build with
///        VR_WITH_CUDA (Linux). Pictures come out in display order, NV12, on
///        that device (@ref DecodedPicture), and never cross to the host.
///
/// Every access unit must be sent: each frame is predicted from earlier ones,
/// so skipping one corrupts the pictures after it until the next key frame.
/// Drop pictures after @ref receive instead.
///
/// @warning Not thread-safe: use from one thread.
class VR_SENSOR_VIDEO_API HevcDecoder {
 public:
  /// @brief How @ref create opens a decoder.
  struct Options {
    /// The matrix and range to decode a stream by when it declares no matrix,
    /// for a source known to code one it does not name: the Femto Mega writes
    /// no colour description and codes BT.601 full range. The range goes with
    /// the matrix, since FFmpeg reads a stream that declares neither as
    /// limited. A stream that declares a matrix is decoded as it declares.
    /// Empty: an unlabelled stream is guessed at (@ref DecodedPicture::yuv).
    std::optional<VideoColorDescription> unlabelled_color;
    /// Set FFmpeg's log level to ERROR. Process-wide: FFmpeg has one logger.
    bool configure_ffmpeg_logging = true;
    /// The device the pictures are handed out on, whose GPU decodes them: one
    /// that imports Metal textures on Apple, one that exports memory and is a
    /// CUDA device elsewhere. Required. Borrowed: it must outlive the decoder
    /// and every picture on it.
    const core::Device* device = nullptr;
    /// The allocator NVDEC's picture buffers are made through: exported
    /// device-only memory, counted against its heap's budget like any other
    /// allocation. Required with NVDEC; VideoToolbox's pictures need none.
    /// Borrowed: it must outlive the decoder and every picture on it.
    core::Allocator* allocator = nullptr;
  };

  /// @return The decoder; `Status::Code::Unsupported` if there is no device
  ///         path: no @ref Options::device (or, for NVDEC, no
  ///         @ref Options::allocator), a device whose GPU this build's
  ///         hardware path cannot reach, or an FFmpeg without that path; or
  ///         `Status::Code::IoError`, `Status::Code::OutOfMemory` or
  ///         `Status::Code::Backend` if FFmpeg or CUDA fails.
  static core::Result<HevcDecoder> create(const Options& options);

  HevcDecoder(HevcDecoder&& other) noexcept;
  HevcDecoder& operator=(HevcDecoder&& other) noexcept;
  ~HevcDecoder();

  /// @brief Hand the decoder one access unit; `size == 0` ends the stream,
  ///        after which @ref receive drains the pictures still held. Take
  ///        every ready picture with @ref receive before the next send.
  /// @return OK; `Status::Code::IoError` if the data cannot be decoded, which
  ///         costs the pictures up to the next key frame;
  ///         `Status::Code::Unsupported`, from then on, once the stream is one
  ///         the hardware cannot decode or hand out (not 8-bit 4:2:0, or on
  ///         VideoToolbox cropped at the left or top);
  ///         `Status::Code::Backend` or `Status::Code::OutOfMemory` if the
  ///         hardware decoder fails; or
  ///         `Status::Code::InvalidArgument` on a moved-from decoder, a null
  ///         @p data with a size, data after the end, or pictures left
  ///         waiting.
  core::Status send(const std::uint8_t* data, std::size_t size,
                    std::int64_t pts);

  /// @brief Take the next decoded picture, if one is ready.
  /// @return The picture; empty when the decoder needs more input or has
  ///         drained; `Status::Code::Unsupported`, as from @ref send, once the
  ///         pictures decoded before a refusal are taken;
  ///         `Status::Code::IoError` if decoding failed;
  ///         `Status::Code::Backend` or `Status::Code::OutOfMemory` if the
  ///         hardware decoder failed or the picture could not be handed over
  ///         on the device; or
  ///         `Status::Code::InvalidArgument` on a moved-from decoder.
  core::Result<std::optional<DecodedPicture>> receive();

  /// @brief Start the stream afresh, as after a seek: the pictures still held
  ///        are dropped, the end of the stream is undone, and the next key
  ///        frame is decoded as the stream's first -- so the pictures that
  ///        lead a CRA key frame, predicted from before it, are skipped
  ///        rather than decoded from references the decoder never had. For a
  ///        caller that lost access units. A refusal
  ///        (`Status::Code::Unsupported`) stands.
  /// @return OK; `Status::Code::InvalidArgument` on a moved-from decoder.
  core::Status reset();

 private:
  struct Impl;
  explicit HevcDecoder(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
