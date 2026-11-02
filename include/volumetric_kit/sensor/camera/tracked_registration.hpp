// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/tracked_registration.hpp
/// @brief A tracked sensor's world registered into a sensor array's world for
///        one capture session: what calib's `register_tracked` writes and a
///        tracked source reads when it opens.
///
/// A tracked sensor (ARKit) poses every frame in a world of its own. It
/// starts that world afresh each session and moves it at each reset. This
/// file holds the one rigid transform that places that world in an array's,
/// so a frame posed `cam_to_tracking` sits at
/// `tracking_to_world * cam_to_tracking` in the array's world.
///
/// Version 1:
///
/// @code{.json}
/// {"format": "volumetric_kit.tracked_registration", "version": 1,
///  "sensor": "ipad-pro-m5",
///  "session": "8f3c0a7e5b2d4e6f9a1b0c2d3e4f5a6b", "world_epoch": 0,
///  "world": {"apriltag": {"family": "tag36h11", "id": 0, "size_m": 0.15}},
///  "tracking_to_world": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]},
///  "fit": {"rms_px": 0.61, "views": 6, "edge_m": 0.1471}}
/// @endcode
///
/// - `tracking_to_world` has the array file's `depth_to_color` form
///   (camera/array_calibration.hpp): `x_world = R(rvec) x_tracking + tvec`
///   (radians, metres).
/// - `session` is the id of the capture session the poses came in (a
///   `.vrst` session's), as 32 hex digits, byte 0 first, written lowercase
///   and read in either case. `world_epoch` counts the device's world
///   resets within that session. The transform holds for that session's
///   epoch alone. A reader refuses frames of another session or epoch, and
///   refuses the registration for an array of another world.
/// - `world` is the array file's world block: one sensor or one tag
///   (@ref WorldFrame).
/// - `fit` is how well the solve held, as the solver states it: the tag
///   fit's RMS reprojection error, the stills it used, and the tag's mean
///   edge as those stills placed it. Against the tag's `size_m`, the edge
///   shows the tracking's scale error.
/// - Every number is written in a form that reads back to the same double.
///   Keys this reader does not know are ignored, and a later `version` is
///   refused.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/export.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace volumetric_kit::sensor::camera {

/// @brief One tracked sensor's world, registered into an array's world for
///        one session's world epoch.
///
/// @code
/// VKC_ASSIGN(const TrackedRegistration reg,
///            read_tracked_registration("take.registration.json"));
/// const Mat4d color_to_world = reg.tracking_to_world * cam_to_tracking;
/// @endcode
struct TrackedRegistration {
  /// The tracked sensor's id, as its session's track names it.
  std::string sensor;
  /// The capture session's id. Never all zero, which a session uses for
  /// none.
  std::array<std::uint8_t, 16> session{};
  /// The device's world this holds for: its count of world resets within
  /// @ref session.
  std::uint16_t world_epoch = 0;
  /// The array's world it registers into: one sensor or one tag.
  WorldFrame world;
  /// The tracking world to the array's world. Rigid.
  Mat4d tracking_to_world = Mat4d(1.0);
  /// The tag fit's RMS reprojection error, in pixels: finite, at least 0.
  double rms_px = 0.0;
  /// The stills the fit used: at least 1.
  std::uint32_t views = 0;
  /// The tag's mean edge as the stills placed it, in metres: finite, above 0.
  double edge_m = 0.0;
};

/// @brief The checks a registration passes to be read or written.
/// @param registration  The registration.
/// @return OK; `Status::Code::InvalidArgument` naming what is wrong: a sensor
///         id that is empty or not UTF-8, an all-zero session, a world that
///         is not exactly one sensor (a non-empty UTF-8 id) or one tag (as
///         @ref validate_array_calibration checks a tag), a transform that is
///         not rigid (@ref check_rigid), an RMS that is not finite and at
///         least 0, no views, or an edge that is not finite and above 0; or
///         `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Status validate_tracked_registration(
    const TrackedRegistration& registration);

/// @brief Parse a registration document.
/// @param json  The document's text.
/// @return The registration; `Status::Code::InvalidArgument` naming what is
///         wrong (not JSON, another format, no `version`, a field missing or
///         of the wrong type, a session that is not 32 hex digits, an epoch
///         past 65535, or a registration
///         @ref validate_tracked_registration refuses);
///         `Status::Code::Unsupported` for a later version or a world that is
///         neither a sensor nor a tag; or `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Result<TrackedRegistration> parse_tracked_registration(
    std::string_view json);

/// @brief Read and parse the registration file at @p path.
/// @param path  The file.
/// @return As @ref parse_tracked_registration, the message prefixed with
///         @p path; or `core::read_file`'s error when the file cannot be read
///         (`Status::Code::NotFound` for a missing one).
VS_CAMERA_API core::Result<TrackedRegistration> read_tracked_registration(
    const std::string& path);

/// @brief Format @p registration as a version-1 document, every number in a
///        form that reads back to the same double, so
///        @ref parse_tracked_registration returns the same fit exactly and
///        the same transform to the Rodrigues round-off.
/// @param registration  The registration.
/// @return The document; `Status::Code::InvalidArgument` for a registration
///         @ref validate_tracked_registration refuses; or
///         `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Result<std::string> format_tracked_registration(
    const TrackedRegistration& registration);

/// @brief Write @p registration to @p path as
///        @ref format_tracked_registration formats it, through
///        `core::write_file`, so a failed write leaves whatever was at
///        @p path.
/// @param path          The file.
/// @param registration  The registration.
/// @param mode          `WriteMode::Replace` to replace a file at @p path
///                      atomically, or `WriteMode::CreateNew` to refuse one.
/// @return OK; `Status::Code::InvalidArgument` for a registration
///         @ref validate_tracked_registration refuses, or
///         `Status::Code::OutOfMemory`, before the file is touched; or
///         `core::write_file`'s error.
VS_CAMERA_API core::Status write_tracked_registration(
    const std::string& path, const TrackedRegistration& registration,
    core::WriteMode mode);

}  // namespace volumetric_kit::sensor::camera
