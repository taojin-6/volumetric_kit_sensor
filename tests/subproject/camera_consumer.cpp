// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A sibling's use of the camera tier: it includes each of the tier's public
// headers, so each must compile in a camera-only build, and checks a camera
// model and a lens's fold, so it links sensor_camera and the core's base tier.

#include <limits>

#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"
#include "volumetric_kit/sensor/camera/stereo_rectification.hpp"
#include "volumetric_kit/sensor/camera/tracked_registration.hpp"
#include "volumetric_kit/sensor/camera/tracked_stills.hpp"

int main() {
  namespace camera = volumetric_kit::sensor::camera;
  const camera::CameraModel depth{{640, 576}, {504.0, 504.0, 320.0, 288.0}, {}};
  const bool never_folds =
      volumetric_kit::sensor::fold_radius2(depth.distortion) ==
      std::numeric_limits<float>::max();
  return camera::check_camera_model(depth).ok() && never_folds ? 0 : 1;
}
