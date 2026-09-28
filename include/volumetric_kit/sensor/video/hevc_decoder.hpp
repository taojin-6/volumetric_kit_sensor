// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/hevc_decoder.hpp
/// @brief An H.265/HEVC decoder over FFmpeg, on the hardware the machine has.
///
/// Nothing camera-specific: bytes in, pictures out. No FFmpeg type appears
/// here; linking still needs FFmpeg's libraries, which the target carries.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"
#include "volumetric_kit/recon/sensor/video/export.hpp"
#include "volumetric_kit/recon/sensor/video/video_backend.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief Decodes an H.265 elementary stream (Annex B), one access unit at a
///        time in decode order; pictures come out in display order.
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
    /// @ref VideoDecodeBackend::Auto takes the first of
    /// @ref hardware_backends that opens, else software, and moves to
    /// software if that hardware refuses the stream (a 4:4:4 one, say, or
    /// one VideoToolbox cannot crop). A named back end is never swapped for
    /// another.
    VideoDecodeBackend backend = VideoDecodeBackend::Auto;
    /// What @ref receive hands out: RGB bytes, or the Y, U and V planes.
    VideoPixelLayout layout = VideoPixelLayout::Rgb24;
    /// Decoding threads; 0 lets FFmpeg choose. In software each thread holds
    /// back a picture, so a live stream that wants the newest picture
    /// soonest sets 1. The hardware back ends decode on the device, and use
    /// them only if Auto moves the stream to software, and then as slice
    /// threads: they hold nothing back, but help only a stream coded in
    /// wavefronts (WPP, as x265 codes by default).
    int threads = 0;
    /// The matrix and range to decode by in place of the stream's, for a
    /// source that labels its stream wrongly or not at all: the Femto Mega
    /// writes no colour description and codes BT.601 full range. Empty: as
    /// the stream declares (@ref DecodedPicture::matrix).
    std::optional<VideoColorDescription> color;
    /// Set FFmpeg's log level to ERROR. Process-wide: FFmpeg has one logger.
    bool configure_ffmpeg_logging = true;
  };

  /// @return The hardware back ends that decode HEVC here, in the order
  ///         @ref VideoDecodeBackend::Auto tries them: VideoToolbox on Apple;
  ///         Cuda, then Vaapi on Linux, so an NVIDIA GPU is
  ///         chosen over an integrated one; Cuda, then D3d11va, on Windows.
  ///         Each is listed only if it opens and decodes HEVC: VideoToolbox
  ///         as VTIsHardwareDecodeSupported answers, the others by decoding a
  ///         built-in clip. Found once per process with FFmpeg's log silenced
  ///         (a back end that is not there says so at ERROR). Auto stops at
  ///         the first that decodes, so it tries only as many as it needs.
  static std::vector<VideoDecodeBackend> hardware_backends();

  /// @return The decoder; @ref Status::Code::Unsupported for a named back end
  ///         not in @ref hardware_backends (the message lists those that
  ///         are); @ref Status::Code::InvalidArgument for a negative thread
  ///         count; or @ref Status::Code::IoError if FFmpeg fails.
  static Result<HevcDecoder> create(const Options& options);

  HevcDecoder(HevcDecoder&& other) noexcept;
  HevcDecoder& operator=(HevcDecoder&& other) noexcept;
  ~HevcDecoder();

  /// @return Where decoding runs now; @ref VideoDecodeBackend::Auto on a
  ///         moved-from decoder.
  VideoDecodeBackend backend() const noexcept;

  /// @brief Hand the decoder one access unit; `size == 0` ends the stream,
  ///        after which @ref receive drains the pictures still held. Take
  ///        every ready picture with @ref receive before the next send.
  /// @return OK; @ref Status::Code::Unsupported once a named hardware back
  ///         end meets a stream it cannot decode or crop, which ends the
  ///         stream; @ref Status::Code::IoError if the data cannot be decoded;
  ///         @ref Status::Code::InvalidArgument on a moved-from decoder, a
  ///         null @p data with a size, data after the end, or pictures left
  ///         waiting.
  Status send(const std::uint8_t* data, std::size_t size, std::int64_t pts);

  /// @brief Take the next decoded picture, if one is ready.
  /// @return The picture, valid until the next call on this decoder; empty
  ///         when the decoder needs more input or has drained;
  ///         @ref Status::Code::Unsupported, as from @ref send, once the
  ///         pictures decoded before a refusal are taken;
  ///         @ref Status::Code::IoError if decoding failed; or
  ///         @ref Status::Code::InvalidArgument on a moved-from decoder.
  Result<std::optional<DecodedPicture>> receive();

 private:
  struct Impl;
  explicit HevcDecoder(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
