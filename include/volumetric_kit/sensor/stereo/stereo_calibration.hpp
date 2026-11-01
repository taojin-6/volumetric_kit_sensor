// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stereo/stereo_calibration.hpp
/// @brief A stereo pair from a sensor array's calibration file: two of its
///        cameras, each an entry of its own.

#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/stereo/export.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief The pair two entries of a calibration file make, as
///        `StereoRectifier::create` takes it.
///
/// Each camera of the pair is a sensor of the file, keyed by the camera's
/// name, posed and with its colour camera's model: a pair of cine cameras,
/// say. The pair's cameras are those models, and
/// `left_to_right = inverse(right.color_to_world) * left.color_to_world`.
///
/// @code
/// VKC_ASSIGN(const camera::ArrayCalibration file,
///            camera::read_array_calibration("array_calibration.json"));
/// VKC_ASSIGN(const StereoCalibration pair,
///            stereo_calibration(file, "bm-1", "bm-2"));
/// @endcode
/// @param file      The calibration.
/// @param left_id   The left camera's id.
/// @param right_id  The right camera's id.
/// @return The pair; `Status::Code::InvalidArgument` for a file that
///         `camera::validate_array_calibration` refuses, the same id twice,
///         an id not in @p file, unposed or without a colour model (each
///         named), cameras at one place, or colour models of different
///         sizes; or `Status::Code::OutOfMemory`.
VS_STEREO_API core::Result<StereoCalibration> stereo_calibration(
    const camera::ArrayCalibration& file, std::string_view left_id,
    std::string_view right_id);

}  // namespace volumetric_kit::sensor
