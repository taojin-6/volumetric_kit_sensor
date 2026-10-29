// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A sibling's use of the camera tier: it includes each of the tier's public
// headers, so each must compile in a camera-only build, and checks a camera
// model, so it links sensor_camera and the core's base tier.

#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"

int main() {
  namespace camera = volumetric_kit::sensor::camera;
  const camera::CameraModel depth{{640, 576}, {504.0, 504.0, 320.0, 288.0}, {}};
  return camera::check_camera_model(depth).ok() ? 0 : 1;
}
