// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file wire/rgbd_wire.hpp
/// @brief An RGB-D sensor's frames as bytes, for a recording or a live link:
///        what the sensor is, once, then each frame's depth half and colour
///        half, a record each, read back in place.
///
/// The bytes are the frame as its driver handed it out (`RgbdFrame`): the
/// depth in the sensor's units and its confidence plane as captured, with no
/// delta and no compression; each camera with its lens; the pose and the
/// extrinsic; the units, the range and the noise. A container carries them
/// and reads none of them. Grid's `.vrst` session (`TrackKind::RgbdDepth`,
/// `TrackKind::RgbdColor`) also names the sensor, its clock, the rate, and
/// each record's time, number and set, so none of those is here: a reader
/// sets `RgbdFrame::timestamp_ns` from a record's time and
/// `RgbdFrame::sequence` from its number. A track's clock 0 (the host's) is
/// `ClockDomain::Host`, any other `ClockDomain::Device`. A writer gives each
/// sensor on `ClockDomain::Device` a clock number of its own above 0, shared
/// by its depth and picture tracks, since two sensors' device clocks do not
/// compare.
///
/// Host code, with no device: nothing here reads or copies a depth sample or
/// a picture. Every field is little-endian, each f32 and f64 bit for bit, and
/// the library builds only for a little-endian host.
///
/// **Version 1.** A track's params begin with the version, and every record
/// of the track is laid out as that version says. Reserved bytes are written
/// 0 and ignored. A field an old reader may ignore takes reserved bytes or
/// comes after the params' last; one it must not ignore takes the next
/// version, which this reader refuses.
///
/// **A camera**, @ref kCameraBytes (`camera::CameraModel`):
/// @code
///   offset  size  field
///        0     4  width (u32)
///        4     4  height (u32)
///        8    32  fx, fy, cx, cy (f64)
///       40    64  k1, k2, p1, p2, k3, k4, k5, k6 (f64)
/// @endcode
/// **A transform** is 16 f64, column-major: 128 bytes.
///
/// **The sensor**, the depth track's params (@ref encode_sensor):
/// @ref kSensorParamsBytes, then the model's bytes; a reader ignores bytes
/// after them.
/// @code
///   offset  size  field
///        0     2  version (1)
///        2     1  role: 0 free run, 1 primary, 2 secondary (SyncRole)
///        3     1  pose: 0 fixed, 1 tracked (PoseSource)
///        4     2  flags: bit 0 a colour camera, bit 1 colour controls
///        6     2  reserved
///        8   104  the depth camera, the factory model
///      112   104  the colour camera, the factory model; zero without one
///      216   128  depth_to_color, the factory extrinsic
///      344     4  the colour controls' exposure (u32); 0 without
///      348     4  their gain (u32); 0 without
///      352     4  M, the model's bytes (u32)
///      356     M  the model (UTF-8)
/// @endcode
///
/// **A depth record** (@ref encode_depth): the prefix, the rows, then the
/// confidence plane when the flags say so, for a depth camera of w x h.
/// Every depth record is a key.
/// @code
///   offset   size  field
///        0    104  depth_camera, the frame's
///      104    128  color_to_world
///      232    128  depth_to_color
///      360      4  metres_per_unit (f32)
///      364      4  min_depth (f32, metres)
///      368      4  max_depth (f32, metres)
///      372     12  depth_noise: a, b, c (f32)
///      384      1  flags: bit 0 a confidence plane
///      385      1  min_confidence: 1 or more with a plane, 0 without
///      386      6  reserved
///      392   2 wh  the depth: h rows of w u16, rows packed; 0 no return
///   392+2wh    wh  with a plane: one u8 level a sample, in the depth's order
/// @endcode
///
/// **The picture track's params** (@ref encode_color_params),
/// @ref kColorParamsBytes; a reader ignores bytes after them: u16 version
/// (1), u16 codec (@ref ColorCodec).
///
/// **A colour record** (@ref encode_color): the prefix, then the picture as
/// the track's codec codes it. A key is a picture decoding can start at.
/// @code
///   offset  size  field
///        0   104  color_camera, the frame's
///      104     1  matrix (H.273 MatrixCoefficients)
///      105     1  transfer (TransferCharacteristics)
///      106     1  primaries (ColourPrimaries)
///      107     1  flags: bit 0 full range
///      108     1  chroma siting (ChromaSampleLocType, 0 to 5)
///      109     3  reserved
///      112    12  exposure: duration_s, iso, bias_ev (f32); 0 unknown
///      124     4  reserved
///      128     n  the coded picture, n >= 1: for HEVC one Annex B access
///                 unit, a key's parameter sets first
/// @endcode
///
/// **The decode checks the layout, not the values.** It refuses bytes that
/// are not a record of the sensor it is given: a size, a camera of another
/// size than `SensorInfo` names, a flag it does not know, a plane without a
/// floor. It passes every value as written; the code that uses one checks
/// it, as `GpuFramePrep` checks the units, the range, the cameras and the
/// poses. A version it does not read is `Unsupported`, read before the rest,
/// so a reader skips the track.
///
/// @code
/// namespace wire = volumetric_kit::sensor::wire;
/// // Writing, once: the depth track's params.
/// VKC_ASSIGN(const std::string params, wire::encode_sensor(sensor.info()));
/// // Each frame: its depth record, in up to three runs, none of them copied.
/// VKC_ASSIGN(const wire::DepthRecord d,
///            wire::encode_depth(sensor.info(), frame));
/// // ... appended as d.prefix, then d.depth, then d.confidence ...
///
/// // Reading: the sensor, then each depth record, held by `bytes`.
/// VKC_ASSIGN(const SensorInfo info,
///            wire::decode_sensor(params, id, ClockDomain::Host, 10));
/// VKC_ASSIGN(RgbdFrame frame,
///            wire::decode_depth(info, bytes->data(), bytes->size(), bytes));
/// frame.timestamp_ns = pts;  // the container's
/// frame.sequence = seq;
/// @endcode

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"
#include "volumetric_kit/sensor/wire/export.hpp"

namespace volumetric_kit::sensor::wire {

/// @brief The wire form's version: what @ref encode_sensor and
///        @ref encode_color_params write, and the only one this library reads.
inline constexpr std::uint16_t kVersion = 1;
/// @brief A camera's bytes.
inline constexpr std::size_t kCameraBytes = 104;
/// @brief The sensor's params before the model's bytes.
inline constexpr std::size_t kSensorParamsBytes = 356;
/// @brief A depth record's prefix: a multiple of 8, so the rows start where
///        the payload's alignment holds.
inline constexpr std::size_t kDepthPrefixBytes = 392;
/// @brief The picture track's params.
inline constexpr std::size_t kColorParamsBytes = 4;
/// @brief A colour record's prefix.
inline constexpr std::size_t kColorPrefixBytes = 128;

/// @brief How a picture track codes its pictures.
enum class ColorCodec : std::uint16_t {
  /// H.265 Main, 8-bit 4:2:0: one Annex B access unit a record, a key an
  /// IRAP picture with its VPS, SPS and PPS.
  HevcMain8 = 1,
};

/// @brief What @p info says, as the depth track's params: all of it but its
///        id, clock and rate, which the track's own fields hold.
/// @param info  A sensor's, as it opened: it names a depth camera.
/// @return The params, @p info as given: a value the decode passes is
///         written as it is; `Status::Code::Unsupported` for a sensor with
///         no depth camera, which this version does not carry;
///         `Status::Code::InvalidArgument` for a role or a pose this version
///         does not name, or a model of 2^32 bytes or more;
///         `Status::Code::OutOfMemory`. Messages read
///         `"wire::encode_sensor: <reason>"`, as each function's read its own
///         name.
VS_WIRE_API core::Result<std::string> encode_sensor(const SensorInfo& info);

/// @brief The sensor a depth track's params describe.
/// @param params  The track's params; bytes after the model's are ignored.
/// @param id      The track's sensor id: `SensorInfo::id`.
/// @param clock   Its clock: `ClockDomain::Host` for the host's clock.
/// @param fps     Its rate, rounded to frames a second; 0 when unknown.
/// @return The sensor; `Status::Code::InvalidArgument` for fewer than
///         @ref kSensorParamsBytes bytes or fewer than the model needs, a
///         flag, a role or a pose this version does not name;
///         `Status::Code::Unsupported` for another version, so a reader
///         skips the track; `Status::Code::OutOfMemory`.
VS_WIRE_API core::Result<SensorInfo> decode_sensor(std::string_view params,
                                                   std::string id,
                                                   ClockDomain clock,
                                                   std::uint32_t fps);

/// @brief A frame's depth record, as runs that a container appends in this
///        order: the prefix, then the frame's own rows and plane, borrowed
///        from it (valid while the frame holds them), not copied.
struct DepthRecord {
  std::array<std::uint8_t, kDepthPrefixBytes> prefix{};  ///< Reserved bytes 0.
  const std::uint8_t* depth = nullptr;  ///< `RgbdFrame::depth`'s bytes.
  std::size_t depth_bytes = 0;          ///< 2 w h.
  /// `RgbdFrame::depth_confidence`; null without a plane.
  const std::uint8_t* confidence = nullptr;
  std::size_t confidence_bytes = 0;  ///< w h with a plane; 0 without.
};

/// @brief @p frame 's depth half as a record of @p info 's sensor.
/// @return The record; `Status::Code::InvalidArgument` for a frame without
///         depth, with both host and device depth, whose depth camera is not
///         @p info 's size or too large for a record's bytes to be counted,
///         with a plane beside no host depth or at a floor of 0, or with a
///         floor and no plane, or for an @p info without a depth camera;
///         `Status::Code::Unsupported` for depth on the device only
///         (`RgbdFrame::depth_device`), which a writer reads back first;
///         `Status::Code::OutOfMemory`.
VS_WIRE_API core::Result<DepthRecord> encode_depth(const SensorInfo& info,
                                                   const RgbdFrame& frame);

/// @brief A depth record read in place: a frame whose depth half the record
///        states, its `depth` and `depth_confidence` pointing into
///        @p payload, its `pixels` @p owner. Nothing is read or copied of the
///        rows or the plane.
///
/// The frame's colour, `timestamp_ns`, `sequence` and `arrival_ns` are left
/// at their defaults, for the reader to set.
/// @param info     The track's sensor (@ref decode_sensor).
/// @param payload  The record's payload, at an even address (a vector's data
///                 is).
/// @param size     Its bytes.
/// @param owner    What holds @p payload: the frame's `pixels`.
/// @return The frame; `Status::Code::InvalidArgument` for an @p info without
///         a depth camera, a null or odd @p payload, fewer bytes than the
///         prefix, a depth camera that is not @p info 's size, a flag this
///         version does not name, a plane at a floor of 0 or a floor without
///         a plane, or a size other than the layout's;
///         `Status::Code::OutOfMemory`.
VS_WIRE_API core::Result<RgbdFrame> decode_depth(
    const SensorInfo& info, const std::uint8_t* payload, std::size_t size,
    std::shared_ptr<const void> owner);

/// @return The picture track's params for @p codec: @ref kColorParamsBytes
///         bytes. A codec's number is its value, so @p codec is written as
///         given, where @ref encode_sensor maps a role and a pose to wire
///         numbers and refuses one it has none for; params of a codec this
///         version does not name are @ref decode_color_params 's to refuse,
///         `Unsupported`.
VS_WIRE_API std::string encode_color_params(ColorCodec codec) noexcept;

/// @return The picture track's codec; `Status::Code::InvalidArgument` for
///         fewer than @ref kColorParamsBytes bytes;
///         `Status::Code::Unsupported` for another version or a codec this
///         version does not name, so a reader skips the track;
///         `Status::Code::OutOfMemory`.
VS_WIRE_API core::Result<ColorCodec> decode_color_params(
    std::string_view params);

/// @brief @p frame 's colour half, less the coded picture, as a colour
///        record's prefix: its colour camera, @p codes and its exposure.
/// @param codes  The colour the coded picture's stream states, which its
///               encoder also writes into the stream (an HEVC VUI).
/// @return The prefix, reserved bytes 0; `Status::Code::InvalidArgument`
///         for an @p info without a colour camera, a frame without colour or
///         whose colour camera is not @p info 's size, a matrix, transfer or
///         primaries outside 0 to 255, or a siting outside 0 to 5;
///         `Status::Code::OutOfMemory`.
VS_WIRE_API core::Result<std::array<std::uint8_t, kColorPrefixBytes>>
encode_color(const SensorInfo& info, const RgbdFrame& frame,
             const ColorCodes& codes);

/// @brief A colour record read in place.
struct ColorRecord {
  camera::CameraModel camera;  ///< The frame's colour camera.
  ColorCodes codes;            ///< The colour its picture is coded in.
  ColorExposure exposure;      ///< As the camera reported it; 0 unknown.
  /// The coded picture, in the payload: valid while it is.
  const std::uint8_t* picture = nullptr;
  std::size_t picture_bytes = 0;  ///< Above 0.
};

/// @return The record, read in place; `Status::Code::InvalidArgument` for a
///         null @p payload, no byte after the prefix, a colour camera that is
///         not @p info 's size or an @p info without one, a flag this
///         version does not name, or a siting above 5;
///         `Status::Code::OutOfMemory`.
VS_WIRE_API core::Result<ColorRecord> decode_color(const SensorInfo& info,
                                                   const std::uint8_t* payload,
                                                   std::size_t size);

}  // namespace volumetric_kit::sensor::wire
