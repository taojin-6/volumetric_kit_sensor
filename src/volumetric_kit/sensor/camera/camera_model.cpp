// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/camera/camera_model.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace volumetric_kit::recon::camera {

namespace {

// unproject() stops once project() maps its ray within this of the pixel: far
// finer than any detector, yet ~100x above double rounding at the widest
// radii.
constexpr double kUnprojectTolerancePx = 1e-9;
constexpr int kUnprojectMaxIterations = 50;
constexpr int kMaxStepHalvings = 30;

// invertible_radius2()'s scan: r^2 from kScanStart, in steps of kScanRatio,
// out to r = 1000, past which no pinhole lens images anything (89.94 degrees
// off-axis).
constexpr double kScanStart = 1e-4;
constexpr double kScanRatio = 1.02;
constexpr double kScanEnd = 1e6;

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("camera model: " + what);
}

// The radial factor's numerator and denominator at s = r^2, and their
// derivatives in s.
struct Radial {
  double n, d, dn, dd;
};

Radial radial_terms(const RationalDistortion& k, double s) {
  return {1.0 + s * (k.k1 + s * (k.k2 + s * k.k3)),
          1.0 + s * (k.k4 + s * (k.k5 + s * k.k6)),
          k.k1 + s * (2.0 * k.k2 + 3.0 * s * k.k3),
          k.k4 + s * (2.0 * k.k5 + 3.0 * s * k.k6)};
}

// Whether the distorted radius r N(s) / D(s) still grows at s = r^2, with the
// denominator positive. Its derivative in r is (N D + 2 s (N' D - N D')) /
// D^2, so only the numerator's sign matters.
bool invertible_at(const RationalDistortion& k, double s) {
  const Radial t = radial_terms(k, s);
  return t.d > 0.0 && t.n * t.d + 2.0 * s * (t.dn * t.d - t.n * t.dd) > 0.0;
}

// The Jacobian of distort_normalized() in the undistorted point, row-major.
std::array<double, 4> distort_jacobian(const RationalDistortion& k,
                                       const Vec2d& u) {
  const double x = u.x;
  const double y = u.y;
  const Radial t = radial_terms(k, x * x + y * y);
  const double radial = t.n / t.d;
  const double dradial = (t.dn * t.d - t.n * t.dd) / (t.d * t.d);
  const double cross = 2.0 * x * y * dradial + 2.0 * k.p1 * x + 2.0 * k.p2 * y;
  return {radial + 2.0 * x * x * dradial + 2.0 * k.p1 * y + 6.0 * k.p2 * x,
          cross, cross,
          radial + 2.0 * y * y * dradial + 6.0 * k.p1 * y + 2.0 * k.p2 * x};
}

bool finite(const Vec2d& v) { return std::isfinite(v.x) && std::isfinite(v.y); }

// The pixel a normalized point images at: project()'s arithmetic, which
// unproject() measures its residual with.
Vec2d project_normalized(const CameraModel& camera, const Vec2d& normalized) {
  const PinholeIntrinsics& k = camera.intrinsics;
  const double intrinsics[4] = {k.fx, k.fy, k.cx, k.cy};
  const std::array<double, 8> d = coefficients(camera.distortion);
  const double p[3] = {normalized.x, normalized.y, 1.0};
  double uv[2];
  project_rational(intrinsics, d.data(), p, uv);
  return {uv[0], uv[1]};
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
  for (const double c : coefficients(camera.distortion)) {
    if (!std::isfinite(c)) return bad("distortion is not finite");
  }
  return {};
}

core::Result<CameraModel> scale_camera_model(const CameraModel& camera,
                                             ImageSize size) {
  VKC_TRY(check_camera_model(camera));
  if (size.width == 0 || size.height == 0) {
    return bad("cannot scale to a size with a zero side");
  }
  const ImageSize& from = camera.size;
  if (std::uint64_t{size.width} * from.height !=
      std::uint64_t{size.height} * from.width) {
    return bad(std::to_string(from.width) + "x" + std::to_string(from.height) +
               " and " + std::to_string(size.width) + "x" +
               std::to_string(size.height) +
               " differ in aspect ratio, so one is not a scale of the other");
  }
  const double sx = static_cast<double>(size.width) / from.width;
  const double sy = static_cast<double>(size.height) / from.height;
  CameraModel scaled = camera;
  scaled.size = size;
  scaled.intrinsics.fx = camera.intrinsics.fx * sx;
  scaled.intrinsics.fy = camera.intrinsics.fy * sy;
  scaled.intrinsics.cx = (camera.intrinsics.cx + 0.5) * sx - 0.5;
  scaled.intrinsics.cy = (camera.intrinsics.cy + 0.5) * sy - 0.5;
  return scaled;
}

double invertible_radius2(const RationalDistortion& d) noexcept {
  double good = 0.0;  // the radius is invertible at s = 0
  for (double s = kScanStart; s <= kScanEnd; s *= kScanRatio) {
    if (invertible_at(d, s)) {
      good = s;
      continue;
    }
    double bad_s = s;
    for (int i = 0; i < 200; ++i) {
      const double mid = 0.5 * (good + bad_s);
      if (mid <= good || mid >= bad_s) break;
      (invertible_at(d, mid) ? good : bad_s) = mid;
    }
    return good;
  }
  return std::numeric_limits<double>::infinity();
}

core::Result<CameraProjection> CameraProjection::create(
    const CameraModel& camera) {
  VKC_TRY(check_camera_model(camera));
  return CameraProjection(camera, invertible_radius2(camera.distortion));
}

core::Result<Vec2d> CameraProjection::project(const Vec3d& point_camera) const {
  if (!(std::isfinite(point_camera.x) && std::isfinite(point_camera.y) &&
        std::isfinite(point_camera.z))) {
    return core::Status::invalid_argument("project: the point is not finite");
  }
  if (!(point_camera.z > 0.0)) {
    return core::Status::invalid_argument(
        "project: the point is not in front of the camera (z <= 0)");
  }
  const Vec2d normalized(point_camera.x / point_camera.z,
                         point_camera.y / point_camera.z);
  if (!(glm::dot(normalized, normalized) < invertible_r2_)) {
    return core::Status::invalid_argument(
        "project: the point is past the lens's invertible radius");
  }
  const Vec2d pixel = project_normalized(camera_, normalized);
  if (!finite(pixel)) {
    return core::Status::numerical(
        "project: overflowed; the point is too far off-axis");
  }
  return pixel;
}

core::Result<Vec2d> CameraProjection::unproject(const Vec2d& pixel) const {
  if (!finite(pixel)) {
    return core::Status::invalid_argument("unproject: the pixel is not finite");
  }
  const PinholeIntrinsics& k = camera_.intrinsics;
  const RationalDistortion& lens = camera_.distortion;
  // The residual is in pixels, as project() computes them, so the tolerance
  // bounds the round trip exactly and means the same at any focal length.
  const auto residual_at = [&](const Vec2d& u) {
    return project_normalized(camera_, u) - pixel;
  };

  // Newton's method on distort(u) = the pixel's normalized point, from that
  // point (exact for no distortion). Every iterate stays inside the
  // invertible radius, so the search cannot settle on a folded-back ray that
  // lands on the pixel too.
  Vec2d u((pixel.x - k.cx) / k.fx, (pixel.y - k.cy) / k.fy);
  const double u2 = glm::dot(u, u);
  if (u2 >= invertible_r2_) u *= std::sqrt(0.5 * invertible_r2_ / u2);
  Vec2d residual = residual_at(u);
  double error = std::hypot(residual.x, residual.y);
  for (int iteration = 0; iteration < kUnprojectMaxIterations; ++iteration) {
    if (error <= kUnprojectTolerancePx) return u;
    const std::array<double, 4> j = distort_jacobian(lens, u);
    const double det = j[0] * j[3] - j[1] * j[2];
    if (!(det > 0.0)) break;  // not locally invertible here
    // Solve j * step = -residual, in normalized units (2x2, by the adjugate).
    const Vec2d r(residual.x / k.fx, residual.y / k.fy);
    const Vec2d step((-j[3] * r.x + j[1] * r.y) / det,
                     (j[2] * r.x - j[0] * r.y) / det);
    // Backtrack until the step reduces the error without leaving the radius.
    bool improved = false;
    double t = 1.0;
    for (int halving = 0; halving < kMaxStepHalvings; ++halving, t *= 0.5) {
      const Vec2d candidate = u + t * step;
      if (glm::dot(candidate, candidate) >= invertible_r2_) continue;
      const Vec2d candidate_residual = residual_at(candidate);
      const double candidate_error =
          std::hypot(candidate_residual.x, candidate_residual.y);
      if (candidate_error < error) {
        u = candidate;
        residual = candidate_residual;
        error = candidate_error;
        improved = true;
        break;
      }
    }
    if (!improved) break;
  }
  if (error <= kUnprojectTolerancePx) return u;
  return core::Status::numerical(
      "unproject: no ray inside the lens's invertible radius lands on this "
      "pixel");
}

}  // namespace volumetric_kit::recon::camera
