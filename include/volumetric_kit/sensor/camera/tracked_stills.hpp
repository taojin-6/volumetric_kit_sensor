// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/tracked_stills.hpp
/// @brief A tracked sensor's stills of one capture session, each with the
///        pose its session records for it: what the sensor's recorder writes
///        beside the session and calib's `register_tracked` reads.
///
/// Version 1, in the directory of the stills it names:
///
/// @code{.json}
/// {"format": "volumetric_kit.tracked_stills", "version": 1,
///  "sensor": "ipad-pro-m5",
///  "session": "8f3c0a7e5b2d4e6f9a1b0c2d3e4f5a6b", "world_epoch": 0,
///  "stills": [
///    {"image": "still_000.png", "timestamp_ns": 1761957000000000000,
///     "camera": {"source": "factory", "width": 1920, "height": 1440,
///                "intrinsics": {"fx": 1596.2, "fy": 1596.2,
///                               "cx": 959.1, "cy": 719.6}},
///     "cam_to_tracking": {"rvec": [2.2, 0.1, -0.3],
///                         "tvec": [0.1, 0.2, 1.4]}}]}
/// @endcode
///
/// - Each still is one of the session's recorded frames: its picture's luma,
///   an 8-bit grey PNG at the picture's size, full range. `image` is its
///   file's name, in this file's directory, unique as bytes.
/// - `camera` is the array file's camera block (camera/array_calibration.hpp):
///   the picture's model, `distortion` absent for a pinhole. The writer
///   writes the block as the array file does, `distortion` included, and
///   `source` as `"factory"`: version 1 keeps no source, its stills' models
///   being their SDK's. The reader reads either source and keeps the model.
/// - `cam_to_tracking` is the frame's pose as its session's records carry it
///   (its `RgbdFrame::color_to_world`, after any correction the recorder
///   makes to every pose it records), so the registration solved from these
///   stills places that session's frames. It has the array file's
///   `depth_to_color` form: `x_tracking = R(rvec) x_camera + tvec`, the
///   camera in this package's axes (+Z forward, +Y down; ARKit's poses
///   through `cv_from_gl_camera` first).
/// - `session` and `world_epoch` are camera/tracked_registration.hpp's: the
///   capture session the frames came in, as 32 hex digits, byte 0 first,
///   and the device's world within it.
/// - `timestamp_ns` is the frame's time, as its session's records carry it.
/// - Every number reads back to the same double. Keys this reader does not
///   know are ignored, and a later `version` is refused.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/export.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace volumetric_kit::sensor::camera {

/// @brief One still: its file, its frame's time, its model and its pose.
struct TrackedStill {
  /// Its file's name, in the stills file's directory: non-empty UTF-8, not
  /// `.` or `..`, with no '/' and no NUL.
  std::string image;
  std::uint64_t timestamp_ns = 0;      ///< Its frame's time.
  CameraModel model;                   ///< The picture's model.
  Mat4d cam_to_tracking = Mat4d(1.0);  ///< Its frame's recorded pose; rigid.
};

/// @brief A tracked sensor's stills of one session's world epoch.
///
/// @code
/// VKC_ASSIGN(const TrackedStills stills, read_tracked_stills("stills.json"));
/// for (const TrackedStill& s : stills.stills) load(s.image, s.model);
/// @endcode
struct TrackedStills {
  std::string sensor;  ///< The sensor's id, as its session's track names it.
  std::array<std::uint8_t, 16> session{};  ///< Never all zero.
  std::uint16_t world_epoch = 0;           ///< The device's world.
  std::vector<TrackedStill> stills;        ///< At least one, in time order.
};

/// @brief The checks stills pass to be read or written.
/// @param stills  The stills.
/// @return OK; `Status::Code::InvalidArgument` naming what is wrong: a sensor
///         id that is empty or not UTF-8, an all-zero session, no still, or a
///         still with a name that is empty, `.` or `..`, has a '/' or a NUL,
///         is not UTF-8 or was used before, a time not after the one before,
///         a model @ref check_camera_model refuses or a pose not rigid
///         (@ref check_rigid); or `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Status validate_tracked_stills(const TrackedStills& stills);

/// @brief Parse a stills document.
/// @param json  The document's text.
/// @return The stills; `Status::Code::InvalidArgument` as
///         @ref parse_tracked_registration names it (not JSON, another
///         format, no `version`, a field missing or of the wrong type, a
///         session that is not 32 hex digits, an epoch past 65535), or for
///         stills @ref validate_tracked_stills refuses;
///         `Status::Code::Unsupported` for a later version or a lens model
///         other than `"rational"`; or `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Result<TrackedStills> parse_tracked_stills(
    std::string_view json);

/// @brief Read and parse the stills file at @p path.
/// @param path  The file.
/// @return As @ref parse_tracked_stills, the message prefixed with @p path;
///         or `core::read_file`'s error when the file cannot be read
///         (`Status::Code::NotFound` for a missing one).
VS_CAMERA_API core::Result<TrackedStills> read_tracked_stills(
    const std::string& path);

/// @brief Format @p stills as a version-1 document, which
///        @ref parse_tracked_stills reads back to the same stills, poses to
///        the Rodrigues round-off.
/// @param stills  The stills.
/// @return The document; `Status::Code::InvalidArgument` for stills
///         @ref validate_tracked_stills refuses; or
///         `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Result<std::string> format_tracked_stills(
    const TrackedStills& stills);

/// @brief Write @p stills to @p path as @ref format_tracked_stills formats
///        them, through `core::write_file`, so a failed write leaves
///        whatever was at @p path.
/// @param path    The file.
/// @param stills  The stills.
/// @param mode    `WriteMode::Replace` to replace a file at @p path
///                atomically, or `WriteMode::CreateNew` to refuse one.
/// @return OK; `Status::Code::InvalidArgument` for stills
///         @ref validate_tracked_stills refuses, or
///         `Status::Code::OutOfMemory`, before the file is touched; or
///         `core::write_file`'s error.
VS_CAMERA_API core::Status write_tracked_stills(const std::string& path,
                                                const TrackedStills& stills,
                                                core::WriteMode mode);

}  // namespace volumetric_kit::sensor::camera
