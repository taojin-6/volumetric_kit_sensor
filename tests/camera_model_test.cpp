// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The camera model: its checks, rescaling about pixel centres, projection
// against OpenCV's own numbers, projection and unprojection to 1e-9 px inside
// the lens's invertible radius and refusal past it, and the scalar templates
// run with a dual number, as a solver differentiates through them. Host-only.

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/projection.hpp"
#include "volumetric_kit/recon/sensor/lens.hpp"

namespace vkc = volumetric_kit::core;
namespace vr = volumetric_kit::recon;
namespace camera = volumetric_kit::recon::camera;
namespace sensor = volumetric_kit::recon::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// A Femto Mega's 4K colour camera, its factory lens (CL2A141000G).
const camera::CameraModel kFemto4k{
    {3840, 2160},
    {2239.48193, 2238.95605, 1891.20874, 1036.8446},
    {0.0753582269, -0.105282806, -0.000271808181, 0.000335, 0.0436422117, 0.0,
     0.0, 0.0}};

bool invalid(const vkc::Status& s) {
  return s.domain() == vkc::Status::Code::InvalidArgument;
}

camera::CameraProjection projection_of(const camera::CameraModel& m) {
  return camera::CameraProjection::create(m).value();
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
  m.distortion.k6 = kInf;
  CHECK(invalid(camera::check_camera_model(m)));
  // A projection checks the model too.
  CHECK(invalid(camera::CameraProjection::create(m).status()));
  return 0;
}

int test_scaling() {
  // A principal point at the centre of a 3840x2160 image stays at the centre
  // of a 1920x1080 one: (W - 1) / 2 in both.
  const camera::CameraModel centred{
      {3840, 2160}, {2239.5, 2239.0, 1919.5, 1079.5}, {}};
  const auto half = camera::scale_camera_model(centred, {1920, 1080});
  CHECK(half.ok());
  CHECK(half.value().size.width == 1920 && half.value().size.height == 1080);
  CHECK(half.value().intrinsics.fx == 1119.75);
  CHECK(half.value().intrinsics.fy == 1119.5);
  CHECK(half.value().intrinsics.cx == 959.5);
  CHECK(half.value().intrinsics.cy == 539.5);

  // The lens is unchanged, and scaling back restores the start.
  const auto at_720p = camera::scale_camera_model(kFemto4k, {1280, 720});
  CHECK(at_720p.ok());
  CHECK(at_720p.value().distortion.k3 == kFemto4k.distortion.k3);
  const auto back = camera::scale_camera_model(at_720p.value(), {3840, 2160});
  CHECK(back.ok());
  CHECK(std::fabs(back.value().intrinsics.cx - kFemto4k.intrinsics.cx) < 1e-9);
  CHECK(std::fabs(back.value().intrinsics.cy - kFemto4k.intrinsics.cy) < 1e-9);

  // A 4:3 mode crops a 16:9 sensor rather than scaling it.
  CHECK(invalid(camera::scale_camera_model(kFemto4k, {1280, 960}).status()));
  CHECK(invalid(camera::scale_camera_model(kFemto4k, {0, 720}).status()));
  return 0;
}

int test_projection_matches_opencv() {
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
    const camera::Vec2d got = camera::distort_normalized(d, c.in);
    CHECK(std::fabs(got.x - c.out.x) < 1e-12);
    CHECK(std::fabs(got.y - c.out.y) < 1e-12);
  }

  // cv::projectPoints through the Femto camera, at points in its frame.
  const camera::CameraProjection femto = projection_of(kFemto4k);
  const struct {
    camera::Vec3d point;
    camera::Vec2d pixel;
  } pixels[] = {
      {{0.2, -0.1, 2.0}, {2115.3946917319745, 924.77502678161159}},
      {{-1.1, 0.6, 1.5}, {224.08835514493734, 1945.8302883786889}},
      {{0.9, 0.5, 1.0}, {3935.5409863296645, 2171.2312630499546}},
  };
  for (const auto& c : pixels) {
    const auto got = femto.project(c.point);
    CHECK(got.ok());
    CHECK(std::fabs(got.value().x - c.pixel.x) < 1e-9);
    CHECK(std::fabs(got.value().y - c.pixel.y) < 1e-9);
  }

  // Each term alone, by hand at (0.5, 0), where r^2 = 0.25.
  camera::RationalDistortion k1;
  k1.k1 = 0.1;
  CHECK(std::fabs(camera::distort_normalized(k1, {0.5, 0.0}).x - 0.5 * 1.025) <
        1e-15);
  camera::RationalDistortion k4 = k1;
  k4.k4 = 0.1;  // the denominator cancels the numerator
  CHECK(camera::distort_normalized(k4, {0.5, 0.0}).x == 0.5);

  CHECK(invalid(femto.project({0.0, 0.0, 0.0}).status()));
  CHECK(invalid(femto.project({0.0, 0.0, -1.0}).status()));
  CHECK(invalid(femto.project({kInf, 0.0, 1.0}).status()));
  return 0;
}

int test_float_lens_is_the_template() {
  // The sensor tier's float lens, which its GLSL mirrors, is the same
  // expressions: it agrees with the double model to float precision.
  const sensor::LensDistortion f{0.42f,  -0.31f, 0.0013f, -0.0021f,
                                 0.087f, 0.39f,  -0.12f,  0.051f};
  const camera::RationalDistortion d{double{f.k1}, double{f.k2}, double{f.p1},
                                     double{f.p2}, double{f.k3}, double{f.k4},
                                     double{f.k5}, double{f.k6}};
  for (const vr::Vec2f p : {vr::Vec2f(0.3f, -0.2f), vr::Vec2f(-0.55f, 0.4f),
                            vr::Vec2f(0.7f, 0.55f)}) {
    const vr::Vec2f got = sensor::distort_normalized(f, p);
    const camera::Vec2d want = camera::distort_normalized(d, {p.x, p.y});
    CHECK(std::fabs(got.x - want.x) < 1e-6);
    CHECK(std::fabs(got.y - want.y) < 1e-6);
  }
  return 0;
}

int test_unproject_round_trips() {
  // Every 4th pixel of the 4K image, corners included, back to the ray it
  // came from to within 1e-9 px.
  const camera::CameraProjection femto = projection_of(kFemto4k);
  for (int y = 0; y <= 2160; y += 4) {
    for (int x = 0; x <= 3840; x += 4) {
      const camera::Vec2d pixel(x - 0.5, y - 0.5);
      const auto ray = femto.unproject(pixel);
      CHECK(ray.ok());
      const auto back = femto.project({ray.value().x, ray.value().y, 1.0});
      CHECK(back.ok());
      CHECK(std::hypot(back.value().x - pixel.x, back.value().y - pixel.y) <=
            1e-9);
    }
  }
  // No distortion inverts in closed form.
  camera::CameraModel pinhole = kFemto4k;
  pinhole.distortion = {};
  const auto ray =
      projection_of(pinhole).unproject({2239.48193 + 1891.20874, 0.0});
  CHECK(ray.ok() && std::fabs(ray.value().x - 1.0) < 1e-12);
  CHECK(invalid(femto.unproject({std::nan(""), 0.0}).status()));
  return 0;
}

int test_invertible_radius() {
  // No distortion never folds.
  CHECK(camera::invertible_radius2({}) == kInf);
  // Barrel distortion r (1 + k1 r^2) stops growing where 1 + 3 k1 r^2 = 0.
  camera::RationalDistortion barrel;
  barrel.k1 = -0.3;
  CHECK(std::fabs(camera::invertible_radius2(barrel) - 1.0 / 0.9) < 1e-12);
  // A denominator 1 - r^2 reaches zero at r^2 = 1 while the radius still
  // grows ((1 + r^2) / (1 - r^2)^2 > 0): the pole is the limit.
  camera::RationalDistortion pole;
  pole.k4 = -1.0;
  CHECK(std::fabs(camera::invertible_radius2(pole) - 1.0) < 1e-12);

  // Past the fold, a pixel's rays fold back, and unproject refuses rather
  // than return either; project refuses the points out there.
  const camera::CameraProjection folding =
      projection_of({{1000, 1000}, {500, 500, 499.5, 499.5}, barrel});
  // The fold's own image: r = sqrt(1/0.9) distorts to r (1 - 0.3/0.9).
  const double r_fold = std::sqrt(1.0 / 0.9);
  const double rd_fold = r_fold * (1.0 - 0.3 / 0.9);
  const auto inside =
      folding.unproject({499.5 + 500.0 * 0.95 * rd_fold, 499.5});
  CHECK(inside.ok() && inside.value().x < r_fold);
  const auto past = folding.unproject({499.5 + 500.0 * 1.05 * rd_fold, 499.5});
  CHECK(!past.ok() && past.status().domain() == vkc::Status::Code::Numerical);
  CHECK(folding.project({0.99 * r_fold, 0.0, 1.0}).ok());
  CHECK(invalid(folding.project({1.01 * r_fold, 0.0, 1.0}).status()));
  // Past the pole the radial factor turns negative: x = 1.5 would image
  // left of centre, so project refuses it.
  const camera::CameraProjection poled =
      projection_of({{1000, 1000}, {500, 500, 499.5, 499.5}, pole});
  CHECK(poled.project({0.9, 0.0, 1.0}).ok());
  CHECK(invalid(poled.project({1.5, 0.0, 1.0}).status()));
  // A lens that folds at r = 1 and unfolds at r^2 = 2 (1 - 1.5 s + 0.5 s^2 =
  // 0) images x = 1.6 only from r ~ 2.2, past the fold: Newton must not step
  // out to that ray, and unproject refuses. x = 0.58 has a ray inside.
  camera::RationalDistortion unfolding;
  unfolding.k1 = -0.5;
  unfolding.k2 = 0.1;
  CHECK(std::fabs(camera::invertible_radius2(unfolding) - 1.0) < 1e-12);
  const camera::CameraProjection wavy =
      projection_of({{1000, 1000}, {500, 500, 499.5, 499.5}, unfolding});
  const auto outside = wavy.unproject({499.5 + 500.0 * 1.6, 499.5});
  CHECK(!outside.ok() &&
        outside.status().domain() == vkc::Status::Code::Numerical);
  const auto within = wavy.unproject({499.5 + 500.0 * 0.58, 499.5});
  CHECK(within.ok() && within.value().x < 1.0);
  // Past the unfold the distorted radius grows again, onto pixels that rays
  // inside already reach: project refuses those points too.
  CHECK(invalid(wavy.project({1.6, 0.0, 1.0}).status()));
  return 0;
}

// A forward-mode dual number, as a solver's Jet is: the templates must take
// a scalar that is not a built-in type, and their derivative must be the
// model's.
struct Dual {
  double v = 0.0;
  double d = 0.0;  // d/d(fx)
  Dual() = default;
  explicit Dual(double value, double derivative = 0.0)
      : v(value), d(derivative) {}
};
Dual operator+(Dual a, Dual b) { return Dual(a.v + b.v, a.d + b.d); }
Dual operator*(Dual a, Dual b) {
  return Dual(a.v * b.v, a.d * b.v + a.v * b.d);
}
Dual operator/(Dual a, Dual b) {
  return Dual(a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v));
}

int test_templates_take_a_dual_number() {
  const camera::PinholeIntrinsics& k = kFemto4k.intrinsics;
  const std::array<double, 8> c = camera::coefficients(kFemto4k.distortion);
  const Dual intrinsics[4] = {Dual(k.fx, 1.0), Dual(k.fy), Dual(k.cx),
                              Dual(k.cy)};
  Dual d[8];
  for (int i = 0; i < 8; ++i) d[i] = Dual(c[static_cast<std::size_t>(i)]);
  const Dual point[3] = {Dual(0.2), Dual(-0.1), Dual(2.0)};
  Dual uv[2];
  camera::project_rational(intrinsics, d, point, uv);
  const auto pixel = projection_of(kFemto4k).project({0.2, -0.1, 2.0});
  CHECK(pixel.ok());
  // Equal up to the compiler's freedom to fuse a multiply-add.
  CHECK(std::fabs(uv[0].v - pixel.value().x) < 1e-9);
  CHECK(std::fabs(uv[1].v - pixel.value().y) < 1e-9);
  // u = fx * xd + cx, so du/dfx is the distorted x, and v does not move.
  const camera::Vec2d xd =
      camera::distort_normalized(kFemto4k.distortion, {0.1, -0.05});
  CHECK(std::fabs(uv[0].d - xd.x) < 1e-15);
  CHECK(uv[1].d == 0.0);
  return 0;
}

}  // namespace

int main() {
  if (test_checks() != 0) return 1;
  if (test_scaling() != 0) return 1;
  if (test_projection_matches_opencv() != 0) return 1;
  if (test_float_lens_is_the_template() != 0) return 1;
  if (test_unproject_round_trips() != 0) return 1;
  if (test_invertible_radius() != 0) return 1;
  if (test_templates_take_a_dual_number() != 0) return 1;
  std::printf("camera model tests passed\n");
  return 0;
}
