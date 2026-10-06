// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/camera/geometry.hpp"

#include <cmath>

namespace volumetric_kit::recon::camera {

namespace {

// How far a rotation may be from orthonormal: refuses a matrix that is not
// one, and accepts one rounded through float.
constexpr double kOrthonormalTolerance = 1e-6;
constexpr double kBottomRowTolerance = 1e-12;

}  // namespace

core::Status check_rigid(const Mat4d& transform) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (!std::isfinite(transform[c][r])) {
        return core::Status::invalid_argument("transform is not finite");
      }
    }
  }
  if (std::fabs(transform[0][3]) > kBottomRowTolerance ||
      std::fabs(transform[1][3]) > kBottomRowTolerance ||
      std::fabs(transform[2][3]) > kBottomRowTolerance ||
      std::fabs(transform[3][3] - 1.0) > kBottomRowTolerance) {
    return core::Status::invalid_argument(
        "transform's bottom row is not [0, 0, 0, 1]");
  }
  const Mat3d rotation(transform);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double d = glm::dot(rotation[a], rotation[b]);
      if (std::fabs(d - (a == b ? 1.0 : 0.0)) > kOrthonormalTolerance) {
        return core::Status::invalid_argument(
            "transform's rotation is not orthonormal");
      }
    }
  }
  if (!(glm::determinant(rotation) > 0.0)) {
    return core::Status::invalid_argument(
        "transform's rotation is a reflection");
  }
  return {};
}

Mat4d rigid_inverse(const Mat4d& transform) noexcept {
  const Mat3d rotation_t = glm::transpose(Mat3d(transform));
  Mat4d inverse(rotation_t);
  inverse[3] = glm::dvec4(-(rotation_t * Vec3d(transform[3])), 1.0);
  return inverse;
}

Mat3d nearest_rotation(const Mat3d& m) noexcept {
  Mat3d r = m;
  for (int i = 0; i < 30; ++i) {
    const Mat3d next = 0.5 * (r + glm::transpose(glm::inverse(r)));
    double change = 0.0;
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        change = std::max(change, std::fabs(next[c][k] - r[c][k]));
      }
    }
    r = next;
    if (change < 1e-15) break;
  }
  return r;
}

Mat3d rotation_from_rodrigues(const Vec3d& rvec) noexcept {
  const double theta = glm::length(rvec);
  Mat3d r(1.0);
  if (theta < 1e-12) {  // first order: I + [r]x
    r[1][0] = -rvec.z;
    r[2][0] = rvec.y;
    r[0][1] = rvec.z;
    r[2][1] = -rvec.x;
    r[0][2] = -rvec.y;
    r[1][2] = rvec.x;
    return r;
  }
  const Vec3d k = rvec / theta;
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double t = 1.0 - c;
  const double kx[3][3] = {
      {0.0, -k.z, k.y}, {k.z, 0.0, -k.x}, {-k.y, k.x, 0.0}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      r[j][i] = (i == j ? c : 0.0) + t * k[i] * k[j] + s * kx[i][j];
    }
  }
  return r;
}

Mat4d matrix_from_rodrigues(const RodriguesTransform& transform) noexcept {
  Mat4d m(rotation_from_rodrigues(transform.rvec));
  m[3] = glm::dvec4(transform.tvec, 1.0);
  return m;
}

}  // namespace volumetric_kit::recon::camera
