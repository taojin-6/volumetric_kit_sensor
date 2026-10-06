// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/projection.hpp
/// @brief OpenCV's rational lens model as a scalar template: the one
///        implementation every precision uses.
///
/// Generic over the scalar so the same expressions serve `float` in the
/// sensor tier, whose GLSL lens mirrors this order of operations, and
/// `double` as the tests' reference. Nothing is checked here, so a caller
/// validates its model first (`check_camera_model`).
///
/// The distortion coefficients are OpenCV's eight, in its order: `k1, k2, p1,
/// p2, k3, k4, k5, k6`. The radial factor is `(1 + k1 r^2 + k2 r^4 + k3 r^6) /
/// (1 + k4 r^2 + k5 r^4 + k6 r^6)`; with `k4 = k5 = k6 = 0` it is the
/// five-coefficient Brown-Conrady model.

namespace volumetric_kit::recon::camera {

/// @brief Where the lens images a point: normalized pinhole coordinates
///        (`x = X / Z`) to the normalized coordinates it lands at.
///
/// OpenCV's forward model, as `cv::projectPoints` applies it.
/// @tparam T   The scalar: `float` or `double`.
/// @param d    The eight distortion coefficients, in OpenCV's order.
/// @param x    The normalized point's x.
/// @param y    The normalized point's y.
/// @param xd   Receives the distorted x.
/// @param yd   Receives the distorted y.
template <typename T>
inline void distort_rational(const T* d, const T& x, const T& y, T* xd, T* yd) {
  const T r2 = x * x + y * y;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;
  const T radial = (T(1) + d[0] * r2 + d[1] * r4 + d[4] * r6) /
                   (T(1) + d[5] * r2 + d[6] * r4 + d[7] * r6);
  const T xy = x * y;
  *xd = x * radial + T(2) * d[2] * xy + d[3] * (r2 + T(2) * x * x);
  *yd = y * radial + d[2] * (r2 + T(2) * y * y) + T(2) * d[3] * xy;
}

}  // namespace volumetric_kit::recon::camera
