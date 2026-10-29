// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file orbbec/orbbec_stream.hpp
/// @brief What an Orbbec camera streams and reports: the streams an
///        @ref volumetric_kit::sensor::OrbbecSensor is opened with,
///        what the camera says about itself, and its counters.
///
/// No Orbbec SDK type appears here, so a consumer includes this header
/// without the SDK's headers. The measurements behind the frame path are in
/// the 2026-09-26 and 2026-09-28 decisions.

#include <cstdint>
#include <optional>
#include <string>

#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/orbbec/orbbec_sync_config.hpp"

namespace volumetric_kit::sensor {

/// @brief The size of @ref OrbbecStreamOptions::color_queue_families: the
///        core's `BufferDesc::kMaxQueueFamilies`, mirrored so this header
///        includes no Vulkan.
inline constexpr std::uint32_t kMaxColorQueueFamilies = 4;

/// @brief How the colour stream crosses the wire. Either way the camera's
///        colour is decoded on a thread per camera, onto the GPU by the
///        hardware decoder.
enum class OrbbecColorCodec {
  /// Motion JPEG, decoded by sensor/video's `JpegDecoder`: nvJPEG or
  /// VideoToolbox. About 185 Mbit/s at 4K, each frame coded on its own: for
  /// calibration captures, which want the least compressed frames, and
  /// diagnostics.
  Mjpeg,
  /// H.265, decoded by sensor/video's `HevcDecoder`: NVDEC or VideoToolbox.
  /// About 21 Mbit/s at 720p and at 4K. The default.
  Hevc,
};

/// @brief What the camera said about itself when it was opened.
struct OrbbecDeviceInfo {
  std::string name;              ///< Model, e.g. "Orbbec Femto Mega".
  std::string serial;            ///< Serial number, e.g. "CL2A141000N".
  std::string firmware_version;  ///< Firmware version string.
  std::string connection_type;   ///< "Ethernet", "USB3.1", ...
  std::string ip_address;        ///< Empty for a USB camera.
  OrbbecSyncMode sync_mode = OrbbecSyncMode::Other;  ///< Its rig role.
  /// The colour camera's lens model at the opened mode, as the SDK reports
  /// it: the model the factory calibration's coefficients are in, before
  /// `SensorInfo::color` holds them as OpenCV's rational one. Named in
  /// recon's kebab case for the SDK's model, one of those an open accepts:
  /// "none", "brown-conrady" or "brown-conrady-k6".
  std::string color_lens_model;
  std::string depth_lens_model;  ///< The same for the depth camera.
};

/// @brief One camera's counters, which a caller reads to see whether it
///        keeps up with the camera. Reset at each start.
struct OrbbecStreamStats {
  /// Frame sets the camera delivered since the start: synchronised depth +
  /// colour pairs, and for H.265 every colour frame, whether or not its
  /// depth came (see @ref lost).
  std::uint64_t received = 0;
  /// Pairs handed out.
  std::uint64_t delivered = 0;
  /// Pairs replaced by a newer one before a poll or the JPEG decoder took
  /// them.
  std::uint64_t dropped = 0;
  /// Pairs a poll took but could not hand out, skipped or refused.
  std::uint64_t failed = 0;
  /// Pairs that will not be handed out. For H.265: a colour frame that came
  /// without its depth (decoded, since the frames after it are predicted
  /// from it, and dropped), and the colour that did not decode -- after a
  /// gap in the stream (a frame lost on the network, or empty), a decode
  /// error or a decoder falling two seconds behind, until the next key
  /// frame, and before the first. For MJPEG, a JPEG that did not decode and
  /// a pair missing a frame. Each is counted once, so
  /// `delivered + dropped + failed + lost <= received`; the difference is a
  /// pair still pending, discarded by a stop, or arriving after the colour
  /// decoder failed.
  std::uint64_t lost = 0;
};

/// @brief The streams a camera is opened with.
///
/// Frames are handed out as the camera captured them: raw depth and the
/// decoded colour, each with its own camera's factory model, for
/// `sensor/utils`'s GPU pass to undistort and convert. Nothing on the host
/// undistorts, registers or converts.
struct OrbbecStreamOptions {
  /// Depth stream mode. The default is the Femto Mega's narrow-field
  /// unbinned mode; 320x288, 512x512 and 1024x1024 (15 fps) are the others.
  std::uint32_t depth_width = 640;
  std::uint32_t depth_height = 576;  ///< See @ref depth_width.
  /// Colour stream size.
  std::uint32_t color_width = 3840;
  std::uint32_t color_height = 2160;  ///< See @ref color_width.
  /// Frame rate of both streams; the camera pairs them only at one rate.
  /// The Femto Mega offers 3840x2160 at 25 fps at most.
  std::uint32_t fps = 25;
  /// How colour crosses the wire.
  OrbbecColorCodec color_codec = OrbbecColorCodec::Hevc;
  /// The colour camera's exposure, which open writes where the camera differs
  /// and reads back: a manual exposure (auto exposure off), or auto exposure
  /// when empty, the default. Auto white balance is switched on either way.
  /// Always written, since the camera keeps what was last written to it, in
  /// another process too: a camera left as it was would keep an earlier
  /// run's exposure. Exposure and gain above 0.
  std::optional<camera::ColorControls> color_controls;
  /// Reject depth nearer than this (metres); stamped on every frame as the
  /// fusion gate. Above 0, as the GPU pass requires.
  float min_depth = 0.25f;
  /// Reject depth farther than this (metres).
  float max_depth = 5.0f;
  /// The device the colour is decoded onto and the GPU pass prepares the
  /// frames on: NVDEC's or nvJPEG's picture (VS_WITH_CUDA), or VideoToolbox's,
  /// stays there, and a frame's colour is that picture (`YuvImage::device` or
  /// `YuvImage::image`), so it never crosses to the host. Required: without
  /// it, the start returns `Status::Code::Unsupported`. Borrowed: it must
  /// outlive the camera and every frame on it.
  const core::Device* device = nullptr;
  /// The allocator NVDEC's and nvJPEG's pictures are made through (exported
  /// device-only memory): required with them, as @ref device is.
  /// VideoToolbox's need none. Borrowed: it must outlive the camera and every
  /// frame on it.
  core::Allocator* allocator = nullptr;
  /// The queue families that will read the colour pictures themselves
  /// (`YuvImage::image`), as @ref PixelBufferImporter::create takes them:
  /// @ref device's own and, say, a renderer's that draws them. The twin of
  /// `GpuFramePrepConfig::color_queue_families`, for the pictures rather than
  /// the GPU pass's output; nvJPEG's buffer ignores them. Held by value,
  /// since each start opens the decoders afresh.
  std::uint32_t color_queue_families[kMaxColorQueueFamilies] = {};
  /// Entries in @ref color_queue_families; more than
  /// @ref kMaxColorQueueFamilies is refused when the camera opens.
  std::uint32_t color_queue_family_count = 0;
};

}  // namespace volumetric_kit::sensor
