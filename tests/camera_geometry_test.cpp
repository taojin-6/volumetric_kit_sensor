// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The camera tier's geometry: what check_rigid refuses, the rigid inverse,
// Rodrigues against OpenCV's matrix and through its edge cases (no turn, a
// tiny one, turns of pi and just under, where sin vanishes), and OpenCV's
// extrinsic turned into a camera's pose. Host-only.

#include <cmath>
#include <cstdio>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr double kPi = 3.14159265358979323846;

bool near(const camera::Mat3d& a, const camera::Mat3d& b, double eps) {
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      if (std::fabs(a[c][r] - b[c][r]) > eps) return false;
    }
  }
  return true;
}

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

  // Each rotation back to its vector and out again, at the edge cases.
  const camera::Vec3d axis_a = glm::normalize(camera::Vec3d(0.3, 0.5, 0.8));
  const camera::Vec3d axis_b = glm::normalize(camera::Vec3d(1.0, 1.0, 0.0));
  const camera::Vec3d axis_c = glm::normalize(camera::Vec3d(0.2, 0.9, -0.4));
  for (const camera::Vec3d rvec :
       {camera::Vec3d(0.0), 1e-7 * axis_a, camera::Vec3d(0.3, -0.8, 0.52),
        camera::Vec3d(kPi, 0.0, 0.0), kPi * axis_b, 3.1415 * axis_c}) {
    const camera::Mat3d rotation = camera::rotation_from_rodrigues(rvec);
    const camera::Vec3d back = camera::rodrigues_from_rotation(rotation);
    CHECK(near(camera::rotation_from_rodrigues(back), rotation, 1e-12));
    // At exactly pi the axis's sign is arbitrary; elsewhere the vector is.
    if (std::fabs(glm::length(rvec) - kPi) > 1e-12) {
      CHECK(glm::length(back - rvec) < 1e-12);
    }
  }
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
  if (test_extrinsic_is_world_to_camera() != 0) return 1;
  std::printf("camera geometry tests passed\n");
  return 0;
}
