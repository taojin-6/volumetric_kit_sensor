// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A sibling's use of the camera tier: it includes each of the tier's public
// headers, so each must compile from a camera-only install, and checks a camera
// model, so it links recon_camera and the core's base tier.

#include "volumetric_kit/recon/camera/array_calibration.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/camera/projection.hpp"

int main() {
  namespace camera = volumetric_kit::recon::camera;
  const camera::CameraModel depth{{640, 576}, {504.0, 504.0, 320.0, 288.0}, {}};
  return camera::check_camera_model(depth).ok() ? 0 : 1;
}
