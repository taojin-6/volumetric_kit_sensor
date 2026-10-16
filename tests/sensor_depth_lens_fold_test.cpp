// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep keeps the depth camera's intrinsics, so its pinhole image is
// the lens's view cropped to that K and resampled at the nearest pixel
// (DECISIONS.md, 2026-10-16). That holds only while every output pixel lies
// inside the lens's monotonic radius: past it distort_rational folds, two
// output pixels read one raw pixel, and depth shows mirrored. Scanning
// distort_rational forward along each border pixel's ray, on the host:
//   - a Femto Mega's NFOV depth camera keeps every same-K output pixel inside
//     the radius;
//   - a pinhole lens never folds;
//   - a strong barrel lens folds inside its image, and the scan finds it, so
//     the first check cannot pass by finding no fold at all.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "test_check.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/projection.hpp"

namespace camera = volumetric_kit::recon::camera;

namespace {

// A scan's step and reach, in normalized radius.
constexpr double kStep = 5e-4;
constexpr double kReach = 3.0;

// The radius along the ray through (x, y) at which `lens` first stops
// carrying a point outward: the distorted point's distance along the ray
// fails to grow over a step. A sign flip or a pole of the rational model
// stops it too. kReach when the lens does not fold before it.
double fold_radius(const camera::RationalDistortion& lens, double x, double y) {
  const double k[8] = {lens.k1, lens.k2, lens.p1, lens.p2,
                       lens.k3, lens.k4, lens.k5, lens.k6};
  const double norm = std::hypot(x, y);
  const double dx = x / norm, dy = y / norm;
  double last = 0.0;
  const int steps = static_cast<int>(kReach / kStep);
  for (int i = 1; i <= steps; ++i) {
    const double r = i * kStep;
    double xd = 0.0, yd = 0.0;
    camera::distort_rational(k, r * dx, r * dy, &xd, &yd);
    const double along = xd * dx + yd * dy;
    if (!(along > last)) return r - kStep;
    last = along;
  }
  return kReach;
}

// The same-K output of `cam`: its pixels' largest normalized radius, and the
// smallest radius at which the lens folds along any border pixel's ray. The
// pass samples integer pixel centres, so the corners are (0, 0) and
// (w - 1, h - 1). Every pixel lies on a ray from the principal point (inside
// the image) to the border, so `widest < fold` keeps all of them inside.
struct SameK {
  double widest = 0.0;
  double fold = kReach;
};

SameK same_k(const camera::CameraModel& cam) {
  const double fx = cam.intrinsics.fx, fy = cam.intrinsics.fy;
  const double cx = cam.intrinsics.cx, cy = cam.intrinsics.cy;
  const std::uint32_t w = cam.size.width, h = cam.size.height;
  SameK out;
  const auto pixel = [&](std::uint32_t u, std::uint32_t v) {
    const double x = (u - cx) / fx, y = (v - cy) / fy;
    out.widest = std::max(out.widest, std::hypot(x, y));
    if (u == 0 || v == 0 || u == w - 1 || v == h - 1) {
      out.fold = std::min(out.fold, fold_radius(cam.distortion, x, y));
    }
  };
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) pixel(u, v);
  }
  return out;
}

// A Femto Mega's NFOV unbinned depth camera, 640 x 576, as recon records one
// (camera_array_calibration_test's factory record): the rational lens with
// CL2A141000N's k1 and k6 as the SDK reported them (DECISIONS.md,
// 2026-09-26: k1 29.29, k6 3.77), its other terms that record's.
// TODO: the SDK's full report of CL2A141000N's depth lens, and the rig's
// three others, by hand on the rig; that report was not kept.
camera::CameraModel femto_nfov() {
  camera::CameraModel c;
  c.size = {640, 576};
  c.intrinsics = {504.2, 504.3, 321.1, 330.6};
  c.distortion = {29.29, 0.1 / 7.0, 1e-5, -2e-5, 0.0, 29.6, 12.0 / 7.0, 3.77};
  return c;
}

double degrees(double radians) { return radians * 180.0 / 3.14159265358979; }

int test_femto() {
  const camera::CameraModel cam = femto_nfov();
  CHECK(camera::check_camera_model(cam).ok());
  const SameK k = same_k(cam);
  const double fx = cam.intrinsics.fx, fy = cam.intrinsics.fy;
  const double cx = cam.intrinsics.cx, cy = cam.intrinsics.cy;
  std::printf(
      "  Femto Mega NFOV: output %.1f x %.1f degrees, widest pixel at r = "
      "%.3f, fold at r = %.3f\n",
      degrees(std::atan((639.5 - cx) / fx) + std::atan((cx + 0.5) / fx)),
      degrees(std::atan((575.5 - cy) / fy) + std::atan((cy + 0.5) / fy)),
      k.widest, k.fold);
  CHECK(k.widest < k.fold);
  return 0;
}

int test_pinhole() {
  camera::CameraModel cam = femto_nfov();
  cam.distortion = {};
  const SameK k = same_k(cam);
  CHECK(k.fold == kReach && k.widest < k.fold);
  return 0;
}

// x (1 - 0.6 r^2) stops growing at r = 1 / sqrt(1.8), about 0.745, inside the
// image's corners at about 0.914.
int test_folding_lens() {
  camera::CameraModel cam = femto_nfov();
  cam.distortion = {-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const SameK k = same_k(cam);
  CHECK(std::fabs(k.fold - 1.0 / std::sqrt(1.8)) < 2.0 * kStep);
  CHECK(!(k.widest < k.fold));
  return 0;
}

}  // namespace

int main() {
  if (test_femto() != 0) return 1;
  if (test_pinhole() != 0) return 1;
  if (test_folding_lens() != 0) return 1;
  std::puts("sensor_depth_lens_fold: OK");
  return 0;
}
