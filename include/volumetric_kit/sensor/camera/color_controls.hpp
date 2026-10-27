// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/color_controls.hpp
/// @brief A colour camera's manual exposure, as a rig's calibration records
///        it and a driver writes it to the camera.

#include <cstdint>

namespace volumetric_kit::recon::camera {

/// @brief A colour camera's manual exposure: auto exposure off, at this
///        exposure time and gain, with auto white balance on.
///
/// A rig's cameras share one (`ArrayCalibration::color_controls`), so their
/// images are exposed alike; a driver writes it when it opens a camera and
/// reports what it read back (`sensor::SensorInfo::color_controls`).
///
/// @code
/// camera::ColorControls lock;
/// lock.exposure = 167;  // about 16.7 ms on a Femto Mega
/// lock.gain = 60;
/// @endcode
struct ColorControls {
  /// The exposure time, in the camera's units (about 100 us on a Femto
  /// Mega); above 0.
  std::uint32_t exposure = 0;
  /// The sensor's gain, in the camera's units; above 0.
  std::uint32_t gain = 0;
};

/// @return Whether @p a and @p b set the same exposure and gain.
inline bool operator==(const ColorControls& a, const ColorControls& b) {
  return a.exposure == b.exposure && a.gain == b.gain;
}

/// @return Whether @p a and @p b differ in exposure or gain.
inline bool operator!=(const ColorControls& a, const ColorControls& b) {
  return !(a == b);
}

}  // namespace volumetric_kit::recon::camera
