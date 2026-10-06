// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/array_calibration.hpp
/// @brief A sensor array's calibration file: where each sensor sits.
///
/// The family's layout: a `device_calibration` object of sensors by serial,
/// each with its colour camera's `pose`, OpenCV's world-to-camera extrinsic
/// `x_camera = R(rvec) x_world + tvec` (radians, metres).
///
/// @code{.json}
/// {"device_calibration": {
///    "CL2A141000N": {"pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]}}}}
/// @endcode
///
/// Every other key is ignored, a sensor's lens fields included: a capture
/// takes its lenses from the sensor.

#include <string>
#include <string_view>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/export.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"

namespace volumetric_kit::recon::camera {

/// @brief One sensor of the array.
///
/// @code
/// VKC_ASSIGN(const ArrayCalibration array,
///            read_array_calibration("array_calibration.json"));
/// const SensorCalibration* primary = find_sensor(array, "CL2A141000N");
/// if (primary != nullptr) {
///   const Vec3d centre(primary->color_to_world[3]);
/// }
/// @endcode
struct SensorCalibration {
  std::string id;  ///< How a capture finds the sensor: its serial.
  /// The colour camera's frame to the world's: the inverse of the file's
  /// `pose`.
  Mat4d color_to_world{1.0};
};

/// @brief A sensor array's calibration: its sensors.
struct ArrayCalibration {
  std::vector<SensorCalibration> sensors;  ///< In id order once read.
};

/// @brief Find a sensor by its id.
/// @param array  A calibration.
/// @param id     A sensor's id.
/// @return The sensor of @p array with @p id, or null; valid while @p array
///         is unchanged.
VR_CAMERA_API const SensorCalibration* find_sensor(
    const ArrayCalibration& array, std::string_view id);

/// @brief The checks a calibration passes to be read: at least one sensor,
///        each id non-empty and unique, and each pose rigid
///        (@ref check_rigid).
/// @param array  The calibration.
/// @return OK; or `Status::Code::InvalidArgument` naming the sensor and what
///         is wrong.
VR_CAMERA_API core::Status validate_array_calibration(
    const ArrayCalibration& array);

/// @brief Parse a calibration document.
/// @param json  The document's text.
/// @return The calibration, its sensors in id order; or
///         `Status::Code::InvalidArgument` naming what is wrong (not JSON, no
///         `device_calibration` object, a sensor without a pose, or a
///         calibration @ref validate_array_calibration refuses).
VR_CAMERA_API core::Result<ArrayCalibration> parse_array_calibration(
    std::string_view json);

/// @brief Read and parse the calibration file at @p path.
/// @param path  The file.
/// @return As @ref parse_array_calibration, the message prefixed with
///         @p path; or `Status::Code::IoError` if the file cannot be read.
VR_CAMERA_API core::Result<ArrayCalibration> read_array_calibration(
    const std::string& path);

}  // namespace volumetric_kit::recon::camera
