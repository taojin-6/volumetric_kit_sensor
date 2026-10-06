// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/orbbec/orbbec_stream.hpp
/// @brief What an Orbbec camera streams and reports: the streams it is opened
///        with, what it says about itself, and its counters -- the same for
///        an @ref volumetric_kit::recon::sensor::OrbbecSensor and each camera
///        of an @ref volumetric_kit::recon::sensor::OrbbecRig.
///
/// No Orbbec SDK type appears here, so a consumer includes this header
/// without the SDK's headers. The measurements behind the frame path are in
/// the 2026-09-26 and 2026-09-28 decisions.

#include <cstdint>
#include <string>

#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief How the colour stream crosses the wire. Either way the camera's
///        colour is decoded on a thread per camera, onto the GPU where the
///        hardware decoder takes it.
enum class OrbbecColorCodec {
  /// Motion JPEG, decoded by sensor/video's `JpegDecoder`: nvJPEG or
  /// VideoToolbox. About 185 Mbit/s at 4K.
  Mjpeg,
  /// H.265, decoded by sensor/video's `HevcDecoder`: NVDEC or VideoToolbox.
  /// About 21 Mbit/s at 720p and at 4K.
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
  /// pair still pending or discarded by a stop.
  std::uint64_t lost = 0;
  /// Of @ref delivered, the frames handed out with their colour on the host
  /// although the stream was opened onto a device
  /// (`OrbbecStreamOptions::device`): the decoder could not keep the picture
  /// there, or its device path failed and every later picture followed. Each
  /// costs a 4K frame's 12 MB across the bus on a discrete GPU, so a run that
  /// should stay on the device reads 0 here.
  std::uint64_t host_pictures = 0;
};

/// @brief The streams a camera is opened with -- the same for every camera of
///        an @ref OrbbecRig.
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
  std::uint32_t color_width = 1280;
  std::uint32_t color_height = 720;  ///< See @ref color_width.
  /// Frame rate of both streams; the camera pairs them only at one rate.
  std::uint32_t fps = 30;
  /// How colour crosses the wire.
  OrbbecColorCodec color_codec = OrbbecColorCodec::Mjpeg;
  /// Reject depth nearer than this (metres); stamped on every frame as the
  /// fusion gate. Above 0, as the GPU pass requires.
  float min_depth = 0.25f;
  /// Reject depth farther than this (metres).
  float max_depth = 5.0f;
  /// The device the GPU pass prepares the frames on. A picture the hardware
  /// decoder leaves there -- NVDEC's or nvJPEG's (VR_WITH_CUDA), or
  /// VideoToolbox's -- stays there, and a frame's colour is that picture
  /// (`YuvImage::device` or `YuvImage::image`), so it never crosses to the
  /// host. Null, or a decode elsewhere, gives host planes. Borrowed: it must
  /// outlive the camera and every frame on it.
  const core::Device* device = nullptr;
  /// With @ref device, the allocator NVDEC's and nvJPEG's pictures are made
  /// through (exported device-only memory); without it, their pictures come
  /// to the host. VideoToolbox's need none. Borrowed: it must outlive the
  /// camera and every frame on it.
  core::Allocator* allocator = nullptr;
};

}  // namespace volumetric_kit::recon::sensor
