// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The camera model: its checks, and the lens template against OpenCV's own
// numbers in double and against the sensor tier's float lens. Host-only.

#include <cmath>
#include <cstdio>
#include <limits>

#include "test_check.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/camera/projection.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;

namespace {

// A Femto Mega's 4K colour camera, its factory lens (CL2A141000G).
const camera::CameraModel kFemto4k{
    {3840, 2160},
    {2239.48193, 2238.95605, 1891.20874, 1036.8446},
    {0.0753582269, -0.105282806, -0.000271808181, 0.000335, 0.0436422117, 0.0,
     0.0, 0.0}};

bool invalid(const vkc::Status& s) {
  return s.domain() == vkc::Status::Code::InvalidArgument;
}

camera::Vec2d distort(const camera::RationalDistortion& d, camera::Vec2d p) {
  const double c[8] = {d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6};
  camera::Vec2d out;
  camera::distort_rational(c, p.x, p.y, &out.x, &out.y);
  return out;
}

int test_checks() {
  CHECK(camera::check_camera_model(kFemto4k).ok());
  camera::CameraModel m = kFemto4k;
  m.size.width = 0;
  CHECK(invalid(camera::check_camera_model(m)));
  m = kFemto4k;
  m.intrinsics.fx = 0.0;
  CHECK(invalid(camera::check_camera_model(m)));
  m = kFemto4k;
  m.intrinsics.fy = -1.0;
  CHECK(invalid(camera::check_camera_model(m)));
  m = kFemto4k;
  m.intrinsics.cx = std::nan("");
  CHECK(invalid(camera::check_camera_model(m)));
  m = kFemto4k;
  m.distortion.k6 = std::numeric_limits<double>::infinity();
  CHECK(invalid(camera::check_camera_model(m)));
  return 0;
}

int test_lens_matches_opencv() {
  // cv::projectPoints, OpenCV 5.0.0, with an identity camera matrix and an
  // eight-coefficient lens, at points out to r ~ 0.9.
  const camera::RationalDistortion d{0.42,  -0.31, 0.0013, -0.0021,
                                     0.087, 0.39,  -0.12,  0.051};
  const struct {
    camera::Vec2d in, out;
  } normalized[] = {
      {{0.3, -0.2}, {0.29941270929159258, -0.19962147286106177}},
      {{-0.55, 0.4}, {-0.54180871681926501, 0.39393758950491992}},
      {{0.7, 0.55}, {0.65411586826839641, 0.51628605721088305}},
  };
  for (const auto& c : normalized) {
    const camera::Vec2d got = distort(d, c.in);
    CHECK(std::fabs(got.x - c.out.x) < 1e-12);
    CHECK(std::fabs(got.y - c.out.y) < 1e-12);
  }

  // Each term alone, by hand at (0.5, 0), where r^2 = 0.25.
  camera::RationalDistortion k1;
  k1.k1 = 0.1;
  CHECK(std::fabs(distort(k1, {0.5, 0.0}).x - 0.5 * 1.025) < 1e-15);
  camera::RationalDistortion k4 = k1;
  k4.k4 = 0.1;  // the denominator cancels the numerator
  CHECK(distort(k4, {0.5, 0.0}).x == 0.5);
  return 0;
}

int test_float_lens_is_the_template() {
  // In float, as the GPU pass's lens.glsl evaluates it, the template agrees
  // with the double model to float precision.
  const float f[8] = {0.42f,  -0.31f, 0.0013f, -0.0021f,
                      0.087f, 0.39f,  -0.12f,  0.051f};
  camera::RationalDistortion d;
  d.k1 = f[0];
  d.k2 = f[1];
  d.p1 = f[2];
  d.p2 = f[3];
  d.k3 = f[4];
  d.k4 = f[5];
  d.k5 = f[6];
  d.k6 = f[7];
  const float points[3][2] = {{0.3f, -0.2f}, {-0.55f, 0.4f}, {0.7f, 0.55f}};
  for (const auto& p : points) {
    float x = 0.0f;
    float y = 0.0f;
    camera::distort_rational(f, p[0], p[1], &x, &y);
    const camera::Vec2d want = distort(d, {p[0], p[1]});
    CHECK(std::fabs(x - want.x) < 1e-6);
    CHECK(std::fabs(y - want.y) < 1e-6);
  }
  return 0;
}

}  // namespace

int main() {
  if (test_checks() != 0) return 1;
  if (test_lens_matches_opencv() != 0) return 1;
  if (test_float_lens_is_the_template() != 0) return 1;
  std::printf("camera model tests passed\n");
  return 0;
}
