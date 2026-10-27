// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/array_calibration.hpp
/// @brief A sensor array's calibration file: where each sensor sits, its
///        cameras' models and their corrections, as calib writes it and a
///        capture reads it.
///
/// Version 2, which @ref write_array_calibration writes for a calibration
/// without a correction:
///
/// @code{.json}
/// {"format": "volumetric_kit.array_calibration", "version": 2,
///  "world": {"sensor": "CL2A141000N"},
///  "device_calibration": {
///    "CL2A141000N": {
///      "pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]},
///      "color": {"source": "factory", "width": 3840, "height": 2160,
///                "intrinsics": {"fx": 2239.5, "fy": 2239.0,
///                               "cx": 1913.2, "cy": 1039.2},
///                "distortion": {"model": "rational", "k1": 0.0754,
///                               "k2": -0.1053, "p1": -0.0003, "p2": 0.0003,
///                               "k3": 0.0436, "k4": 0, "k5": 0, "k6": 0}},
///      "depth": {...as "color"...},
///      "depth_to_color": {"rvec": [...], "tvec": [...]}}}}
/// @endcode
///
/// - Sensors are keyed by id, the serial a capture finds them by, under
///   version 1's `device_calibration` key, so a version-1 reader still finds
///   the poses. Every field of a sensor is optional: a file poses sensors,
///   records their lenses, or both. A factory record is unposed.
/// - `pose` is the sensor's **colour camera's** OpenCV extrinsic, world to
///   camera: `x_camera = R(rvec) x_world + tvec` (radians, metres).
///   `depth_to_color`, beside `depth` only, is the same form:
///   `x_color = R(rvec) x_depth + tvec`.
/// - A camera's `width` and `height` are the image its intrinsics are in
///   pixels of. `distortion` may be absent, which reads as a pinhole lens;
///   the writer always writes it. Its `model` is `"rational"`, OpenCV's
///   eight coefficients, the one this reader knows. `source` says whether
///   the model is the sensor's own (`"factory"`) or solved from images
///   (`"calibrated"`).
/// - `world`, optional, is the world frame: `{"sensor": id}`, the colour
///   camera of a sensor of the array, which is posed at the origin; or
///   `{"apriltag": {"family": "tag36h11", "id": 0, "size_m": 0.15}}`, a
///   printed tag (@ref AprilTagWorld). A world of another kind is refused.
/// - Numbers are written in a form that reads back to the same double. Keys
///   this reader does not know are ignored; a later `version` is refused.
///
/// Version 3 is version 2 with a correction: a sensor's `depth_offset_m`
/// (@ref SensorCalibration::depth_offset_m), beside its other fields:
///
/// @code{.json}
///    "CL2A141000N": {"pose": {...}, "depth": {...}, "depth_offset_m": 0.00417}
/// @endcode
///
/// The writer writes version 3 only for a calibration that carries a
/// correction, so an uncorrected one is written as version 2, byte for byte
/// as before, and a reader that knows only version 2 refuses a corrected file
/// rather than run it uncorrected. Both are read; a correction in a file that
/// says version 2 is refused, since such a reader would ignore it.
///
/// Version 1, the family's earlier layout, has no `format` or `version`: a
/// `device_calibration` object of sensors by serial, each with a `pose`. It
/// is read for the poses alone, every other key ignored, and never written.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/export.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"

namespace volumetric_kit::recon::camera {

/// @brief Where a camera's model came from.
enum class IntrinsicsSource : std::uint8_t {
  Factory,     ///< The sensor's own, as its SDK reports it.
  Calibrated,  ///< Solved from images, by calib.
};

/// @brief One camera of a sensor: its model, and where it came from.
struct CameraCalibration {
  CameraModel model;  ///< Valid (@ref check_camera_model).
  IntrinsicsSource source = IntrinsicsSource::Factory;  ///< Its provenance.
};

/// @brief One sensor of the array.
///
/// @code
/// VKC_ASSIGN(const ArrayCalibration array,
///            read_array_calibration("array_calibration.json"));
/// const SensorCalibration* primary = find_sensor(array, "CL2A141000N");
/// if (primary != nullptr && primary->color_to_world) {
///   const Vec3d centre((*primary->color_to_world)[3]);
/// }
/// @endcode
struct SensorCalibration {
  std::string id;  ///< How a capture finds the sensor: its serial.
  /// The colour camera's frame to the world's: the inverse of the file's
  /// `pose`. Empty where the file does not pose the sensor.
  std::optional<Mat4d> color_to_world;
  std::optional<CameraCalibration> color;  ///< If the file records it.
  std::optional<CameraCalibration> depth;  ///< If the file records it.
  /// The depth camera's frame to the colour camera's; only beside
  /// @ref depth.
  std::optional<Mat4d> depth_to_color;
  /// Metres the depth reads too far (+) or too near (-) along its axis,
  /// subtracted from every return when the sensor's frames are prepared
  /// (`sensor::GpuFramePrepConfig::depth_offset_m`); 0, the default, where
  /// the file has none. Finite, and at most 0.05 m either way. Written as
  /// `"depth_offset_m"`, a version-3 field, only when it is not 0.
  double depth_offset_m = 0.0;
};

/// @brief A printed AprilTag whose frame is the world.
///
/// The axes are OpenCV's marker frame, not libapriltag's (x right, y down,
/// z into the tag): the origin at the tag's centre, x from its top-left
/// corner to its top-right, y from its bottom-left corner to its top-left,
/// and z out of the printed face. The corners are the tag's own, in the
/// order OpenCV's detector returns them, however the tag is turned in an
/// image. A tag laid face up on the floor puts z up and the floor at z = 0.
struct AprilTagWorld {
  /// The tag's family, as the AprilTag library names it (`"tag36h11"`).
  std::string family;
  int id = 0;           ///< The tag's id in its family.
  double size_m = 0.0;  ///< The black square's edge, in metres.
};

/// @brief What the array's world frame is: a sensor, a tag, or unsaid.
struct WorldFrame {
  /// The sensor whose colour camera is the world, posed at the origin;
  /// empty where the world is not a sensor.
  std::string sensor;
  /// The tag whose frame is the world, which poses no sensor at the origin;
  /// empty where the world is not a tag.
  std::optional<AprilTagWorld> apriltag;
};

/// @brief A sensor array's calibration: its world, and its sensors.
struct ArrayCalibration {
  WorldFrame world;                        ///< What the world frame is.
  std::vector<SensorCalibration> sensors;  ///< In id order once read.
};

/// @brief Find a sensor by its id.
/// @param array  A calibration.
/// @param id     A sensor's id.
/// @return The sensor of @p array with @p id, or null; valid while @p array
///         is unchanged.
VR_CAMERA_API const SensorCalibration* find_sensor(
    const ArrayCalibration& array, std::string_view id);

/// @brief The checks a calibration passes to be read or written.
///
/// At least one sensor; each id non-empty, UTF-8 and unique; each pose and
/// `depth_to_color` rigid (@ref check_rigid), and `depth_to_color` only
/// beside a depth camera; each camera model valid (@ref check_camera_model);
/// each `depth_offset_m` finite and at most 0.05 m either way;
/// and at most one world: a sensor that is in the array and posed at the
/// origin, or a tag with a non-empty UTF-8 family, an id of 0 or more and a
/// finite size above 0.
/// @param array  The calibration.
/// @return OK; `Status::Code::InvalidArgument` naming what is wrong; or
///         `Status::Code::OutOfMemory`.
VR_CAMERA_API core::Status validate_array_calibration(
    const ArrayCalibration& array);

/// @brief Parse a calibration document, version 3, 2 or 1.
/// @param json  The document's text.
/// @return The calibration, its sensors in id order;
///         `Status::Code::InvalidArgument` naming what is wrong (not JSON,
///         another format, a field of the wrong type, a `version` without a
///         `format`, a correction in a version-2 document, a version-1
///         sensor without a pose, or a calibration
///         @ref validate_array_calibration refuses);
///         `Status::Code::Unsupported` for a later version, a distortion
///         model other than `"rational"`, or a world that is neither a
///         sensor nor a tag; or `Status::Code::OutOfMemory`.
VR_CAMERA_API core::Result<ArrayCalibration> parse_array_calibration(
    std::string_view json);

/// @brief Read and parse the calibration file at @p path.
/// @param path  The file.
/// @return As @ref parse_array_calibration, the message prefixed with
///         @p path; or `core::read_file`'s error when the file cannot be read
///         (`Status::Code::NotFound` for a missing one).
VR_CAMERA_API core::Result<ArrayCalibration> read_array_calibration(
    const std::string& path);

/// @brief Format @p array as a version-3 document if it carries a correction
///        (a sensor's `depth_offset_m` other than 0), and as a version-2 one
///        if not.
///
/// Sensors in id order, every number in a form that reads back to the same
/// double (at most 17 significant digits), so @ref parse_array_calibration
/// returns the same models exactly, and the same poses to the Rodrigues
/// conversion's round-off.
/// @param array  The calibration.
/// @return The document; `Status::Code::InvalidArgument` for a calibration
///         @ref validate_array_calibration refuses; or
///         `Status::Code::OutOfMemory`.
VR_CAMERA_API core::Result<std::string> format_array_calibration(
    const ArrayCalibration& array);

/// @brief Write @p array to @p path as @ref format_array_calibration formats
///        it, through `core::write_file`, so a failed write leaves what was
///        at @p path.
/// @param path   The file.
/// @param array  The calibration.
/// @param mode   `WriteMode::Replace` to replace a file at @p path
///               atomically, or `WriteMode::CreateNew` to refuse one.
/// @return OK; `Status::Code::InvalidArgument` for a calibration
///         @ref validate_array_calibration refuses, or
///         `Status::Code::OutOfMemory`, before the file is touched; or
///         `core::write_file`'s error.
VR_CAMERA_API core::Status write_array_calibration(
    const std::string& path, const ArrayCalibration& array,
    core::WriteMode mode);

}  // namespace volumetric_kit::recon::camera
