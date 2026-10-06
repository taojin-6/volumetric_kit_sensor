// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/array_calibration.hpp
/// @brief A sensor array's calibration file: where each sensor sits, and its
///        cameras' models, as calib writes it and a capture reads it.
///
/// Version 2 of the file, the one this writes:
///
/// @code{.json}
/// {"format": "volumetric_kit.array_calibration", "version": 2,
///  "world": {"sensor": "CL2A141000N"},
///  "sensors": {
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
/// - A sensor is keyed by its id, the serial a capture finds it by. Every
///   field of it is optional, so a file can pose sensors, record their lenses,
///   or both.
/// - `pose` is the sensor's **colour camera's** OpenCV extrinsic, the
///   world-to-camera `x_camera = R(rvec) x_world + tvec` (radians, metres).
///   The depth camera sits at `depth_to_color` from it, the same form:
///   `x_color = R(rvec) x_depth + tvec`.
/// - A camera's `width` and `height` are the image its intrinsics are in
///   pixels of. `distortion` is optional (absent is a pinhole lens); its
///   `model` is `"rational"`, OpenCV's eight coefficients, the one this
///   reader knows. `source` says whether the intrinsics are the sensor's own
///   (`"factory"`) or calib's (`"calibrated"`).
/// - `world`, also optional, names what the world frame is: a sensor's colour
///   camera (`{"sensor": id}`, which must then be posed at the origin), or an
///   AprilTag (`{"apriltag": {"family": "tag36h11", "id": 0, "size_m":
///   0.22}}`).
/// - Keys this reader does not know are ignored, so a later field does not
///   break it; an incompatible change bumps `version`, which it refuses.
///
/// Version 1, the family's earlier layout (a `device_calibration` object of
/// colour cameras by serial, each with a required `pose`), is still read: its
/// intrinsics and distortion become the colour camera when the intrinsics
/// record `width` and `height`, and are dropped otherwise, as is
/// `optimal_intrinsics`. It is never written.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/export.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"

namespace volumetric_kit::recon::camera {

/// @brief Where a camera's intrinsics and lens came from.
enum class IntrinsicsSource : std::uint8_t {
  Factory,     ///< The sensor's own calibration, as its SDK reports it.
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
};

/// @brief An AprilTag whose frame is the world: libapriltag's axes, x right,
///        y down and z into the tag, at its centre.
struct AprilTagOrigin {
  std::string family;    ///< The tag family, e.g. `"tag36h11"`.
  std::uint32_t id = 0;  ///< The tag's id in its family.
  double size_m = 0.0;   ///< The tag's edge, in metres; positive.
};

/// @brief What the array's world frame is.
struct WorldFrame {
  /// @brief Which kind of frame it is.
  enum class Kind : std::uint8_t {
    Unspecified,  ///< The file does not say.
    Sensor,       ///< A sensor's colour camera: @ref sensor.
    AprilTag,     ///< A tag in the scene: @ref tag.
  };
  Kind kind = Kind::Unspecified;  ///< Which of the fields below applies.
  std::string sensor;             ///< For `Kind::Sensor`: the sensor's id.
  AprilTagOrigin tag;             ///< For `Kind::AprilTag`: the tag.
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
/// `depth_to_color` rigid (@ref check_rigid), and `depth_to_color` only beside
/// a depth camera; each camera model valid (@ref check_camera_model); a world
/// sensor that is present and posed at the origin, or an AprilTag with a
/// family and a positive size.
/// @param array  The calibration.
/// @return OK; or `Status::Code::InvalidArgument` naming the sensor and what
///         is wrong.
VR_CAMERA_API core::Status validate_array_calibration(
    const ArrayCalibration& array);

/// @brief Parse a calibration document, version 2 or version 1.
/// @param json  The document's text.
/// @return The calibration, its sensors in id order; or
///         `Status::Code::InvalidArgument` naming what is wrong (not JSON,
///         another format, a field of the wrong type, a version-1 camera
///         without a pose, or a calibration
///         @ref validate_array_calibration refuses); or
///         `Status::Code::Unsupported` for a later version or a distortion
///         model other than `"rational"`.
VR_CAMERA_API core::Result<ArrayCalibration> parse_array_calibration(
    std::string_view json);

/// @brief Read and parse the calibration file at @p path.
/// @param path  The file.
/// @return As @ref parse_array_calibration, the message prefixed with
///         @p path; or `Status::Code::IoError` if the file cannot be read.
VR_CAMERA_API core::Result<ArrayCalibration> read_array_calibration(
    const std::string& path);

/// @brief Format @p array as a version-2 document.
///
/// Every number is written in the shortest form that reads back to the same
/// double, so @ref parse_array_calibration returns the same models, and the
/// same poses to within the Rodrigues conversion's round-off.
/// @param array  The calibration.
/// @return The document; or `Status::Code::InvalidArgument` for a calibration
///         @ref validate_array_calibration refuses.
VR_CAMERA_API core::Result<std::string> format_array_calibration(
    const ArrayCalibration& array);

/// @brief Write @p array to @p path as @ref format_array_calibration formats
///        it.
///
/// The document goes to a temporary file beside @p path (beside its target,
/// for a symbolic link), is synced, and is renamed over it, so a failed write
/// or a crash leaves the calibration that was there. A replaced file keeps
/// its permissions.
/// @param path   The file, created or replaced.
/// @param array  The calibration.
/// @return OK; `Status::Code::InvalidArgument` for a calibration
///         @ref validate_array_calibration refuses, before the file is
///         touched; or `Status::Code::IoError` if it cannot be written.
VR_CAMERA_API core::Status write_array_calibration(
    const std::string& path, const ArrayCalibration& array);

}  // namespace volumetric_kit::recon::camera
