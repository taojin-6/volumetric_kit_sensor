// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/camera/camera_model.hpp"

#include <cmath>
#include <string>

namespace volumetric_kit::recon::camera {

namespace {

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("camera model: " + what);
}

}  // namespace

core::Status check_camera_model(const CameraModel& camera) {
  if (camera.size.width == 0 || camera.size.height == 0) {
    return bad("image size has a zero side");
  }
  const PinholeIntrinsics& k = camera.intrinsics;
  if (!(std::isfinite(k.fx) && k.fx > 0.0 && std::isfinite(k.fy) &&
        k.fy > 0.0)) {
    return bad("focal lengths must be finite and positive");
  }
  if (!(std::isfinite(k.cx) && std::isfinite(k.cy))) {
    return bad("principal point is not finite");
  }
  const RationalDistortion& d = camera.distortion;
  for (const double c : {d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6}) {
    if (!std::isfinite(c)) return bad("distortion is not finite");
  }
  return {};
}

}  // namespace volumetric_kit::recon::camera
