// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The camera tier's geometry: what check_rigid refuses, the rigid inverse,
// Rodrigues against OpenCV's matrix and back to its vector at every angle,
// exact_rigid on a transform as float leaves one (made exact, deterministic,
// the polar factor of a shear, at and past its tolerances), and OpenCV's
// extrinsic turned into a camera's pose and back. Host-only.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <glm/gtc/constants.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::sensor::camera;

namespace {

const double kPi = glm::pi<double>();

bool near(const camera::Mat4d& a, const camera::Mat4d& b, double eps) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (std::fabs(a[c][r] - b[c][r]) > eps) return false;
    }
  }
  return true;
}

bool refused(const camera::Mat4d& m) {
  const vkc::Status s = camera::check_rigid(m);
  return s.domain() == vkc::Status::Code::InvalidArgument;
}

int test_check_rigid() {
  CHECK(camera::check_rigid(camera::Mat4d(1.0)).ok());
  camera::Mat4d general(camera::rotation_from_rodrigues({0.3, -0.8, 0.52}));
  general[3] = glm::dvec4(1.25, -2.5, 3.75e-3, 1.0);
  CHECK(camera::check_rigid(general).ok());
  // A pose that went through float on its way here is still rigid.
  CHECK(camera::check_rigid(camera::Mat4d(glm::mat4(general))).ok());

  CHECK(refused(camera::Mat4d(2.0)));  // a scale
  camera::Mat4d mirror(1.0);
  mirror[2][2] = -1.0;
  CHECK(refused(mirror));
  camera::Mat4d nan(1.0);
  nan[3][0] = std::nan("");
  CHECK(refused(nan));
  camera::Mat4d projective(1.0);
  projective[2][3] = 0.5;
  CHECK(refused(projective));
  return 0;
}

int test_rigid_inverse() {
  camera::Mat4d m(camera::rotation_from_rodrigues({0.3, -0.8, 0.52}));
  m[3] = glm::dvec4(1.25, -2.5, 3.75e-3, 1.0);
  CHECK(near(m * camera::rigid_inverse(m), camera::Mat4d(1.0), 1e-15));
  CHECK(near(camera::rigid_inverse(m) * m, camera::Mat4d(1.0), 1e-15));
  return 0;
}

int test_rodrigues() {
  // cv::Rodrigues of (0.3, -0.8, 0.52), OpenCV 5.0.0, row by row.
  const double cv[3][3] = {
      {0.58150540754602609, -0.54785729747347389, -0.60141819277420561},
      {0.32720988493183562, 0.8343305677499866, -0.44365098322992569},
      {0.74483901092627047, 0.061195468157752758, 0.6644320600929251}};
  const camera::Mat3d r = camera::rotation_from_rodrigues({0.3, -0.8, 0.52});
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      CHECK(std::fabs(r[col][row] - cv[row][col]) < 1e-15);
    }
  }
  return 0;
}

int test_rodrigues_inverse() {
  // Each angle's rotation back to its vector, about axes that are not a
  // coordinate axis and about x: cv::Rodrigues returns exactly zero below
  // about 1e-5 rad, and sin(theta) vanishes toward pi.
  const camera::Vec3d axes[] = {glm::normalize(camera::Vec3d(0.3, 0.5, 0.8)),
                                glm::normalize(camera::Vec3d(0.2, 0.9, -0.4)),
                                glm::normalize(camera::Vec3d(1.0, 1.0, 0.0)),
                                camera::Vec3d(1.0, 0.0, 0.0)};
  for (const double theta :
       {0.0, 1e-9, 1e-6, 1e-5, 0.3, 1.0, kPi / 2, 2.5, kPi - 1e-6, kPi}) {
    for (const camera::Vec3d& axis : axes) {
      const camera::Vec3d rvec = theta * axis;
      const camera::Mat3d rotation = camera::rotation_from_rodrigues(rvec);
      const camera::Vec3d back = camera::rodrigues_from_rotation(rotation);
      CHECK(near(camera::Mat4d(camera::rotation_from_rodrigues(back)),
                 camera::Mat4d(rotation), 1e-14));
      // The vector to round-off relative to its angle; at exactly pi, -rvec
      // is the same rotation, and which comes back is arbitrary.
      const double error = theta == kPi ? std::min(glm::length(back - rvec),
                                                   glm::length(back + rvec))
                                        : glm::length(back - rvec);
      CHECK(error <= 1e-14 * theta);
    }
  }
  // The vector is the shortest: a turn past pi comes back as one under it.
  const camera::Vec3d over = (kPi + 0.5) * axes[0];
  const camera::Vec3d under =
      camera::rodrigues_from_rotation(camera::rotation_from_rodrigues(over));
  CHECK(glm::length(under - (kPi - 0.5) * -axes[0]) < 1e-12);
  return 0;
}

int test_nearest_rotation() {
  const camera::Mat3d r = camera::rotation_from_rodrigues({0.11, -0.02, 0.03});
  CHECK(near(camera::nearest_rotation(r), r, 1e-15));
  // A rotation with a row scaled, as an SDK's factory extrinsic arrives:
  // the nearest rotation is the rotation itself.
  camera::Mat3d scaled = r;
  for (int c = 0; c < 3; ++c) scaled[c][0] *= 0.9939;
  CHECK(near(camera::nearest_rotation(scaled), r, 1e-12));
  camera::Mat4d m(camera::nearest_rotation(scaled));
  CHECK(camera::check_rigid(m).ok());
  return 0;
}

// The largest entry's distance of R^T R from the identity's.
double orthonormal_offset(const camera::Mat4d& m) {
  const camera::Mat3d r(m);
  double offset = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      offset = std::max(offset,
                        std::fabs(glm::dot(r[a], r[b]) - (a == b ? 1.0 : 0.0)));
    }
  }
  return offset;
}

bool refused_exact(const camera::Mat4d& m, const char* why) {
  const vkc::Result<camera::Mat4d> r = camera::exact_rigid(m);
  if (r.ok() || r.status().domain() != vkc::Status::Code::InvalidArgument ||
      r.status().message() != why) {
    std::fprintf(stderr, "expected \"%s\", got \"%s\"\n", why,
                 r.ok() ? "OK" : r.status().message().c_str());
    return false;
  }
  return true;
}

int test_exact_rigid() {
  camera::Mat4d general(camera::rotation_from_rodrigues({0.3, -0.8, 0.52}));
  general[3] = glm::dvec4(1.25, -2.5, 3.75e-3, 1.0);
  // Rigid to double's rounding already (a calibration's): bit for bit.
  const vkc::Result<camera::Mat4d> kept = camera::exact_rigid(general);
  CHECK(kept.ok() && *kept == general);

  // As a float device leaves it: rounded to float, the corner 3 ulp past 1,
  // a row entry 1e-7, a column stretched 4e-7 off orthonormal.
  camera::Mat4d rounded = camera::Mat4d(glm::mat4(general));
  rounded[3][3] = double(
      std::nextafter(std::nextafter(std::nextafter(1.0f, 2.0f), 2.0f), 2.0f));
  rounded[1][3] = double(1.0e-7f);
  for (int r = 0; r < 3; ++r) rounded[0][r] *= 1.0 + 2.0e-7;
  CHECK(orthonormal_offset(rounded) > 3e-7);
  CHECK(!camera::check_rigid(rounded).ok());
  const vkc::Result<camera::Mat4d> exact = camera::exact_rigid(rounded);
  CHECK(exact.ok());
  const camera::Mat4d& m = *exact;
  CHECK(m[0][3] == 0.0 && m[1][3] == 0.0 && m[2][3] == 0.0 && m[3][3] == 1.0);
  CHECK(orthonormal_offset(m) < 1e-12);
  CHECK(glm::determinant(camera::Mat3d(m)) > 0.0);
  CHECK(camera::check_rigid(m).ok());
  CHECK(m[3] == glm::dvec4(glm::dvec3(rounded[3]), 1.0));  // translation kept
  CHECK(near(m, general, 1e-6));  // the rotation nearest, not another one
  // Deterministic: the same transform, the same bits; and exact is a fixed
  // point.
  const vkc::Result<camera::Mat4d> again = camera::exact_rigid(rounded);
  CHECK(again.ok() && *again == m);
  const vkc::Result<camera::Mat4d> fixed = camera::exact_rigid(m);
  CHECK(fixed.ok() && *fixed == m);

  // A shear, column 0 leaning 4e-5 toward column 1: the rotation is the
  // polar factor, so R^T M is symmetric; Gram-Schmidt's would leave it
  // triangular, 4e-5 off symmetric.
  camera::Mat4d sheared = general;
  sheared[0] += 4e-5 * sheared[1];
  const vkc::Result<camera::Mat4d> polar = camera::exact_rigid(sheared);
  CHECK(polar.ok());
  CHECK(orthonormal_offset(*polar) < 1e-12);
  const camera::Mat3d factor =
      glm::transpose(camera::Mat3d(*polar)) * camera::Mat3d(sheared);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      CHECK(std::fabs(factor[a][b] - factor[b][a]) < 1e-12);
    }
  }

  // At the tolerances, taken; past them, refused, each named.
  camera::Mat4d row = general;
  row[3][3] = 1.0 + 0.9e-6;
  row[0][3] = -0.9e-6;
  CHECK(camera::exact_rigid(row).ok());
  row[0][3] = -1.1e-6;
  CHECK(refused_exact(
      row, "transform's bottom row is more than 1e-6 off [0, 0, 0, 1]"));
  camera::Mat4d drifted = general;
  for (int r = 0; r < 3; ++r) drifted[2][r] *= 1.0 + 0.45e-4;  // 0.9e-4 off
  const vkc::Result<camera::Mat4d> pulled = camera::exact_rigid(drifted);
  CHECK(pulled.ok() && orthonormal_offset(*pulled) < 1e-12);
  CHECK(near(*pulled, general, 1e-12));
  for (int r = 0; r < 3; ++r) drifted[2][r] *= 1.0 + 0.1e-4;  // 1.1e-4 off
  CHECK(refused_exact(
      drifted, "transform's rotation is more than 1e-4 off orthonormal"));
  camera::Mat4d mirror = general;
  mirror[2] = -mirror[2];
  CHECK(refused_exact(mirror, "transform's rotation is a reflection"));
  camera::Mat4d nan = general;
  nan[3][1] = std::nan("");
  CHECK(refused_exact(nan, "transform is not finite"));
  return 0;
}

int test_extrinsic_is_world_to_camera() {
  // OpenCV's extrinsic of a camera turned 90 degrees about +Y, tvec (0, 0, 2):
  // x_camera = R x_world + t, so the camera sits at -R^T t = (2, 0, 0), and a
  // point 1 m ahead of it is at (1, 0, 0), since R^T turns +Z into -X.
  const camera::RodriguesTransform extrinsic{{0.0, kPi / 2, 0.0},
                                             {0.0, 0.0, 2.0}};
  const camera::Mat4d camera_to_world =
      camera::rigid_inverse(camera::matrix_from_rodrigues(extrinsic));
  const glm::dvec4 centre = camera_to_world[3];
  CHECK(glm::length(camera::Vec3d(centre) - camera::Vec3d(2.0, 0.0, 0.0)) <
        1e-15);
  const glm::dvec4 ahead = camera_to_world * glm::dvec4(0.0, 0.0, 1.0, 1.0);
  CHECK(glm::length(camera::Vec3d(ahead) - camera::Vec3d(1.0, 0.0, 0.0)) <
        1e-15);

  // And back to the same Rodrigues form.
  const camera::RodriguesTransform back =
      camera::rodrigues_from_matrix(camera::rigid_inverse(camera_to_world));
  CHECK(glm::length(back.rvec - extrinsic.rvec) < 1e-15);
  CHECK(glm::length(back.tvec - extrinsic.tvec) < 1e-15);
  return 0;
}

}  // namespace

int main() {
  if (test_check_rigid() != 0) return 1;
  if (test_rigid_inverse() != 0) return 1;
  if (test_rodrigues() != 0) return 1;
  if (test_rodrigues_inverse() != 0) return 1;
  if (test_nearest_rotation() != 0) return 1;
  if (test_exact_rigid() != 0) return 1;
  if (test_extrinsic_is_world_to_camera() != 0) return 1;
  std::printf("camera geometry tests passed\n");
  return 0;
}
