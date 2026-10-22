// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep keeps the depth camera's intrinsics, so its pinhole image is
// the lens's view cropped to that K and resampled at the nearest pixel
// (DECISIONS.md, 2026-10-16). That holds only while every output pixel lies
// inside the lens's monotonic radius: past it distort_rational folds, two
// output pixels read one raw pixel, and depth shows mirrored. Scanning
// distort_rational forward along each border pixel's ray, on the host:
//   - a stand-in for a Femto Mega's NFOV depth lens, held to what is on record
//     of CL2A141000N's, keeps every same-K output pixel inside the radius;
//   - a pinhole lens never folds;
//   - a strong barrel lens folds inside its image, and the scan finds it, so
//     the first check cannot pass by finding no fold at all.
// The stand-in is not an SDK report, and where a lens folds depends on the
// terms not on record, so a rig's own lenses are a hand check (the TODO at
// femto_stand_in).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "test_check.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/projection.hpp"

namespace camera = volumetric_kit::recon::camera;

namespace {

// A scan's march, the step it is resolved to, and its reach, in normalized
// radius.
constexpr double kMarch = 0.01;
constexpr double kStep = 5e-4;
constexpr double kReach = 3.0;

// How far `lens` carries the point at radius r along the unit ray (dx, dy):
// the distorted point's distance along that ray.
double along(const camera::RationalDistortion& lens, double dx, double dy,
             double r) {
  const double k[8] = {lens.k1, lens.k2, lens.p1, lens.p2,
                       lens.k3, lens.k4, lens.k5, lens.k6};
  double xd = 0.0, yd = 0.0;
  camera::distort_rational(k, r * dx, r * dy, &xd, &yd);
  return xd * dx + yd * dy;
}

// The radius along the ray through (x, y) at which `lens` first stops
// carrying a point outward: the distorted point's distance along the ray
// fails to grow over a step. Marched out at kMarch until it fails to grow
// (a sign flip or a pole of the rational model stops it too), then bisected
// in the two marches before for where it stops growing over kStep. kReach
// when the lens does not fold before it.
double fold_radius(const camera::RationalDistortion& lens, double x, double y) {
  const double norm = std::hypot(x, y);
  const double dx = x / norm, dy = y / norm;
  const auto grows = [&](double r) {
    return along(lens, dx, dy, r + kStep) > along(lens, dx, dy, r);
  };
  double last = 0.0;
  const int marches = static_cast<int>(std::lround(kReach / kMarch));
  for (int i = 1; i <= marches; ++i) {
    const double r = i * kMarch;
    const double a = along(lens, dx, dy, r);
    if (!(a > last)) {
      double lo = std::max(0.0, r - 2.0 * kMarch), hi = r;
      for (int k = 0; k < 60; ++k) {
        const double mid = 0.5 * (lo + hi);
        (grows(mid) ? lo : hi) = mid;
      }
      return lo;
    }
    last = a;
  }
  return kReach;
}

// The radius along the ray through the raw point (xd, yd) that `lens` carries
// to it: marched out until the distorted distance reaches |(xd, yd)|, then
// bisected. NaN when the lens stops carrying points outward first, so that no
// ray reaches the raw point. The undistorted point is taken on the raw
// point's ray, which the tangential terms move it off by far less than a
// pixel.
double undistorted_radius(const camera::RationalDistortion& lens, double xd,
                          double yd) {
  const double target = std::hypot(xd, yd);
  if (target == 0.0) return 0.0;
  const double dx = xd / target, dy = yd / target;
  double lo = 0.0, last = 0.0;
  for (double hi = kMarch; hi <= kReach; hi += kMarch) {
    const double a = along(lens, dx, dy, hi);
    if (!(a > last)) break;
    if (a >= target) {
      for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        (along(lens, dx, dy, mid) < target ? lo : hi) = mid;
      }
      return 0.5 * (lo + hi);
    }
    lo = hi;
    last = a;
  }
  return std::numeric_limits<double>::quiet_NaN();
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

// How far undistorting moves `cam`'s raw pixels: each raw pixel against its
// pinhole position at the same K, in pixels. The mean is over every eighth
// pixel; the largest and `reached` take in the last row and column too (so
// all four corners), which would tilt the mean towards the edge. `reached` is
// false if some raw pixel has no ray reaching it.
struct Correction {
  bool reached = true;
  double mean_px = 0.0;
  double max_px = 0.0;
};

// Every eighth index below n, and the last.
std::vector<std::uint32_t> every_eighth(std::uint32_t n) {
  std::vector<std::uint32_t> out;
  for (std::uint32_t i = 0; i < n; i += 8) out.push_back(i);
  if (out.back() != n - 1) out.push_back(n - 1);
  return out;
}

Correction correction(const camera::CameraModel& cam) {
  const double fx = cam.intrinsics.fx, fy = cam.intrinsics.fy;
  const double cx = cam.intrinsics.cx, cy = cam.intrinsics.cy;
  Correction out;
  double sum = 0.0;
  std::size_t n = 0;
  for (const std::uint32_t v : every_eighth(cam.size.height)) {
    for (const std::uint32_t u : every_eighth(cam.size.width)) {
      const double xd = (u - cx) / fx, yd = (v - cy) / fy;
      const double rd = std::hypot(xd, yd);
      const double ru = undistorted_radius(cam.distortion, xd, yd);
      if (!std::isfinite(ru)) {
        out.reached = false;
        continue;
      }
      const double s = rd > 0.0 ? ru / rd - 1.0 : 0.0;
      const double moved = std::hypot(xd * s * fx, yd * s * fy);
      if (u % 8 == 0 && v % 8 == 0) {
        sum += moved;
        ++n;
      }
      out.max_px = std::max(out.max_px, moved);
    }
  }
  out.mean_px = n == 0 ? 0.0 : sum / static_cast<double>(n);
  return out;
}

// A stand-in for a Femto Mega's NFOV unbinned depth camera, 640 x 576: not an
// SDK report. k1 29.29 and k6 3.77 are CL2A141000N's, the only terms of its
// depth lens on record (DECISIONS.md, 2026-09-28, GPU pre-processing). The
// others are calib's Femto-like test lens (session_writer_test), and K is
// recon's calibration-file test's; test_stand_in holds the lens to the rest of
// the record.
// TODO: each rig camera's depth lens as the SDK reports it (OrbbecSensor reads
// it at open; a calib session's factory_calibration.json holds it), checked
// by test_femto in place of the stand-in.
camera::CameraModel femto_stand_in() {
  camera::CameraModel c;
  c.size = {640, 576};
  c.intrinsics = {504.2, 504.3, 321.1, 330.6};
  c.distortion = {29.29, 14.52, 3.1e-05, -2.2e-05, 0.61, 29.63, 23.99, 3.77};
  return c;
}

double degrees(double radians) { return radians * 180.0 / 3.14159265358979; }

// The stand-in is a lens CL2A141000N could have: one whose rays reach its
// whole raw image, and which moves raw pixels as undistorting CL2A141000N's
// depth was measured to, within 10 % (DECISIONS.md, 2026-09-26, the Orbbec
// driver: 31.5 px mean, 220.9 px at the edge).
int test_stand_in() {
  const camera::CameraModel cam = femto_stand_in();
  CHECK(camera::check_camera_model(cam).ok());
  const Correction c = correction(cam);
  const double fx = cam.intrinsics.fx, fy = cam.intrinsics.fy;
  const double cx = cam.intrinsics.cx, cy = cam.intrinsics.cy;
  const auto raw_angle = [&](double xd, double yd) {
    return std::atan(undistorted_radius(cam.distortion, xd, yd));
  };
  std::printf(
      "  stand-in: raw %.1f x %.1f degrees, undistorting moves a raw pixel "
      "%.1f px on average, %.1f px at most\n",
      degrees(raw_angle((-0.5 - cx) / fx, 0.0) +
              raw_angle((639.5 - cx) / fx, 0.0)),
      degrees(raw_angle(0.0, (-0.5 - cy) / fy) +
              raw_angle(0.0, (575.5 - cy) / fy)),
      c.mean_px, c.max_px);
  CHECK(c.reached);
  CHECK(std::fabs(c.mean_px - 31.5) < 0.1 * 31.5);
  CHECK(std::fabs(c.max_px - 220.9) < 0.1 * 220.9);
  return 0;
}

int test_femto() {
  const camera::CameraModel cam = femto_stand_in();
  const SameK k = same_k(cam);
  const double fx = cam.intrinsics.fx, fy = cam.intrinsics.fy;
  const double cx = cam.intrinsics.cx, cy = cam.intrinsics.cy;
  std::printf(
      "  stand-in: output %.1f x %.1f degrees, widest pixel at r = %.3f, "
      "fold at r = %.3f (%.0f: none)\n",
      degrees(std::atan((639.5 - cx) / fx) + std::atan((cx + 0.5) / fx)),
      degrees(std::atan((575.5 - cy) / fy) + std::atan((cy + 0.5) / fy)),
      k.widest, k.fold, kReach);
  CHECK(k.widest < k.fold);
  return 0;
}

int test_pinhole() {
  camera::CameraModel cam = femto_stand_in();
  cam.distortion = {};
  const SameK k = same_k(cam);
  CHECK(k.fold == kReach && k.widest < k.fold);
  const Correction c = correction(cam);
  CHECK(c.reached && c.max_px < 1e-6);
  return 0;
}

// x (1 - 0.6 r^2) stops growing at r = 1 / sqrt(1.8), about 0.745, inside the
// image's corners at about 0.914. Its raw corners, past the most it carries a
// point (0.745 (1 - 0.6 / 1.8), about 0.497), no ray reaches.
int test_folding_lens() {
  camera::CameraModel cam = femto_stand_in();
  cam.distortion = {-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const SameK k = same_k(cam);
  CHECK(std::fabs(k.fold - 1.0 / std::sqrt(1.8)) < 2.0 * kStep);
  CHECK(!(k.widest < k.fold));
  CHECK(!correction(cam).reached);
  return 0;
}

}  // namespace

int main() {
  if (test_stand_in() != 0) return 1;
  if (test_femto() != 0) return 1;
  if (test_pinhole() != 0) return 1;
  if (test_folding_lens() != 0) return 1;
  std::puts("sensor_depth_lens_fold: OK");
  return 0;
}
