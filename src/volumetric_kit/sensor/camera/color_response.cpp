// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/camera/color_response.hpp"

#include <cmath>
#include <string>

namespace volumetric_kit::sensor::camera {

namespace {

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("color response: " + what);
}

}  // namespace

core::Status validate(const ColorResponse& response) {
  if (response.is_identity()) return {};
  for (const double g : response.gain) {
    if (!std::isfinite(g) || !(g > 0.0)) {
      return bad("a gain is not finite above 0");
    }
  }
  if (!std::isfinite(response.a) || !std::isfinite(response.b)) {
    return bad("the falloff is not finite");
  }
  if (!std::isfinite(response.r_max) || !(response.r_max > 0.0)) {
    return bad("r_max is not finite above 0");
  }
  // F(q) = 1 + a q + b q^2 over q in [0, r_max^2]: F(0) is 1, so it is above
  // 0 throughout unless it is at the far end or at the parabola's vertex.
  const double q_max = response.r_max * response.r_max;
  const auto f = [&](double q) {
    return 1.0 + response.a * q + response.b * q * q;
  };
  bool positive = f(q_max) > 0.0;
  if (response.b > 0.0) {
    const double vertex = -response.a / (2.0 * response.b);
    if (vertex > 0.0 && vertex < q_max) positive = positive && f(vertex) > 0.0;
  }
  if (!positive) return bad("the falloff F is not above 0 up to r_max");
  return {};
}

}  // namespace volumetric_kit::sensor::camera
