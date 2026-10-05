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
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
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
    /// The matrix and range to decode a stream by when it declares no matrix,
    /// for a source known to code one it does not name: the Femto Mega writes
    /// no colour description and codes BT.601 full range. The range goes with
    /// the matrix, since FFmpeg reads a stream that declares neither as
    /// limited. A stream that declares a matrix is decoded as it declares.
    /// Empty: an unlabelled stream is guessed at (@ref DecodedPicture::matrix).
    std::optional<VideoColorDescription> unlabelled_color;
    /// Set FFmpeg's log level to ERROR. Process-wide: FFmpeg has one logger.
    bool configure_ffmpeg_logging = true;
    /// A device to hand pictures out on when the back end decodes on its GPU:
    /// Cuda, in a build with VR_WITH_CUDA, on a device that exports memory,
    /// or VideoToolbox, on a device that imports Metal textures. Such a
    /// picture stays on the GPU as NV12: in a buffer it holds (@ref
    /// DecodedPicture::device) from Cuda, as images it holds (@ref
    /// DecodedPicture::image) from VideoToolbox. Any other comes to the host
    /// as @ref layout says, and so does every picture after the device path
    /// fails once. A device path that does not open, fails, or is left when
    /// Auto moves to software is said once, as a warning through core's log
    /// handler.
    /// Borrowed: it must outlive the decoder and every picture on it.
    const core::Device* device = nullptr;
    /// The allocator the Cuda back end makes its picture buffers through:
    /// exported device-only memory, counted against its heap's budget like
    /// any other allocation. Needed with @ref device for Cuda; VideoToolbox's
    /// pictures need none. Borrowed: it must outlive the decoder and every
    /// picture on it.
    core::Allocator* allocator = nullptr;
    /// Whose decoder this is (a camera, say), put ahead of its warnings so a
    /// program running several can tell them apart. Empty puts nothing.
    std::string label;
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

  /// @return The decoder; `Status::Code::Unsupported` for a named back end
  ///         not in @ref hardware_backends (the message lists those that
  ///         are); `Status::Code::InvalidArgument` for a negative thread
  ///         count or an Nv12 layout, which only a device picture comes as;
  ///         or `Status::Code::IoError` if FFmpeg fails.
  static core::Result<HevcDecoder> create(const Options& options);

  HevcDecoder(HevcDecoder&& other) noexcept;
  HevcDecoder& operator=(HevcDecoder&& other) noexcept;
  ~HevcDecoder();

  /// @return Where decoding runs now; @ref VideoDecodeBackend::Auto on a
  ///         moved-from decoder.
  VideoDecodeBackend backend() const noexcept;

  /// @brief Hand the decoder one access unit; `size == 0` ends the stream,
  ///        after which @ref receive drains the pictures still held. Take
  ///        every ready picture with @ref receive before the next send.
  /// @return OK; `Status::Code::Unsupported` once a named hardware back
  ///         end meets a stream it cannot decode or crop, which ends the
  ///         stream; `Status::Code::IoError` if the data cannot be decoded;
  ///         `Status::Code::InvalidArgument` on a moved-from decoder, a
  ///         null @p data with a size, data after the end, or pictures left
  ///         waiting.
  core::Status send(const std::uint8_t* data, std::size_t size,
                    std::int64_t pts);

  /// @brief Take the next decoded picture, if one is ready.
  /// @return The picture, valid until the next call on this decoder; empty
  ///         when the decoder needs more input or has drained;
  ///         `Status::Code::Unsupported`, as from @ref send, once the
  ///         pictures decoded before a refusal are taken;
  ///         `Status::Code::IoError` if decoding failed; or
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
