// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/camera/stereo_rectification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <string>

namespace volumetric_kit::sensor {
namespace {

using camera::CameraModel;
using camera::Mat3d;
using camera::Mat4d;
using camera::Vec2d;
using camera::Vec3d;

constexpr const char* kWho = "rectify_stereo";

core::Status refuse(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

// ---------------------------------------------------------------------------
// OpenCV's stereoRectify (calib3d's stereo_geom.cpp, CALIB_ZERO_DISPARITY),
// step for step, in double.

// OpenCV's undistortPoints for one pixel of `c`, with its default criteria:
// normalized, five fixed-point iterations of the lens's inverse, then through
// `rr` (P's 3x3 times R) as a homography when given.
Vec2d undistort_point(const CameraModel& c, Vec2d pixel, const Mat3d* rr) {
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  const double x0 = (pixel.x - k.cx) / k.fx;
  const double y0 = (pixel.y - k.cy) / k.fy;
  double x = x0;
  double y = y0;
  for (int j = 0; j < 5; ++j) {
    const double r2 = x * x + y * y;
    const double icdist = (1 + ((d.k6 * r2 + d.k5) * r2 + d.k4) * r2) /
                          (1 + ((d.k3 * r2 + d.k2) * r2 + d.k1) * r2);
    if (icdist < 0) {  // OpenCV's regression_14583: past the lens's fold
      x = x0;
      y = y0;
      break;
    }
    const double dx = 2 * d.p1 * x * y + d.p2 * (r2 + 2 * x * x);
    const double dy = d.p1 * (r2 + 2 * y * y) + 2 * d.p2 * x * y;
    x = (x0 - dx) * icdist;
    y = (y0 - dy) * icdist;
  }
  if (rr == nullptr) return Vec2d(x, y);
  const Vec3d h = *rr * Vec3d(x, y, 1.0);
  return Vec2d(h.x / h.z, h.y / h.z);
}

// A rectangle, OpenCV's Rect_<double>.
struct Rect {
  double x = 0, y = 0, width = 0, height = 0;
};

// OpenCV's getUndistortRectangles: the border of a 9 x 9 grid over the image,
// undistorted and taken through `r` and `p` (a pinhole of focal length `f`
// and principal point `cc`), and the rectangles inside and around it.
void undistort_rectangles(const CameraModel& c, const Mat3d& r, double f,
                          Vec2d cc, Rect* inner, Rect* outer) {
  constexpr int kN = 9;
  Mat3d p(0.0);
  p[0][0] = f;
  p[1][1] = f;
  p[2][0] = cc.x;
  p[2][1] = cc.y;
  p[2][2] = 1.0;
  const Mat3d rr = p * r;
  const double step_x = (c.size.width - 1) / static_cast<double>(kN - 1);
  const double step_y = (c.size.height - 1) / static_cast<double>(kN - 1);
  double ix0 = -std::numeric_limits<float>::max();
  double ix1 = std::numeric_limits<float>::max();
  double iy0 = -std::numeric_limits<float>::max();
  double iy1 = std::numeric_limits<float>::max();
  double ox0 = std::numeric_limits<float>::max();
  double ox1 = -std::numeric_limits<float>::max();
  double oy0 = std::numeric_limits<float>::max();
  double oy1 = -std::numeric_limits<float>::max();
  for (int y = 0; y < kN; ++y) {
    for (int x = 0; x < kN; ++x) {
      if (x != 0 && x != kN - 1 && y != 0 && y != kN - 1) continue;
      const Vec2d q = undistort_point(c, Vec2d(x * step_x, y * step_y), &rr);
      ox0 = std::min(ox0, q.x);
      ox1 = std::max(ox1, q.x);
      oy0 = std::min(oy0, q.y);
      oy1 = std::max(oy1, q.y);
      if (x == 0) ix0 = std::max(ix0, q.x);
      if (x == kN - 1) ix1 = std::min(ix1, q.x);
      if (y == 0) iy0 = std::max(iy0, q.y);
      if (y == kN - 1) iy1 = std::min(iy1, q.y);
    }
  }
  *inner = Rect{ix0, iy0, ix1 - ix0, iy1 - iy0};
  *outer = Rect{ox0, oy0, ox1 - ox0, oy1 - oy0};
}

Mat4d rotation4(const Mat3d& r) {
  Mat4d m(r);
  m[3][3] = 1.0;
  return m;
}

// Whether the pair is rectified already: no rotation, no lens, one camera
// model, the right camera along +x alone, each within round-off.
bool already_rectified(const StereoCalibration& c) {
  constexpr double kEps = 1e-9;
  const Mat3d r(c.left_to_right);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::fabs(r[i][j] - (i == j ? 1.0 : 0.0)) > kEps) return false;
    }
  }
  const auto no_lens = [](const camera::RationalDistortion& d) {
    return d.k1 == 0 && d.k2 == 0 && d.p1 == 0 && d.p2 == 0 && d.k3 == 0 &&
           d.k4 == 0 && d.k5 == 0 && d.k6 == 0;
  };
  const camera::PinholeIntrinsics& a = c.left.intrinsics;
  const camera::PinholeIntrinsics& b = c.right.intrinsics;
  const Vec3d t(c.left_to_right[3]);
  return no_lens(c.left.distortion) && no_lens(c.right.distortion) &&
         std::fabs(a.fx - b.fx) <= kEps && std::fabs(a.fy - b.fy) <= kEps &&
         std::fabs(a.cx - b.cx) <= kEps && std::fabs(a.cy - b.cy) <= kEps &&
         std::fabs(t.y) <= kEps && std::fabs(t.z) <= kEps && t.x < 0;
}

core::Status check_camera(const char* what, const CameraModel& c) {
  const core::Status valid = camera::check_camera_model(c);
  if (!valid.ok()) {
    return refuse(std::string("the ") + what + " camera: " + valid.message());
  }
  return {};
}

}  // namespace

core::Result<StereoRectification> rectify_stereo(const StereoCalibration& c,
                                                 camera::ImageSize size,
                                                 double alpha) try {
  VKC_TRY(check_camera("left", c.left));
  VKC_TRY(check_camera("right", c.right));
  if (c.left.size.width != c.right.size.width ||
      c.left.size.height != c.right.size.height) {
    return refuse("the two cameras' sizes differ");
  }
  const core::Status rigid = camera::check_rigid(c.left_to_right);
  if (!rigid.ok()) return refuse("left_to_right: " + rigid.message());
  if (!(std::isfinite(alpha) && alpha >= 0.0 && alpha <= 1.0)) {
    return refuse("alpha must be in [0, 1]");
  }
  if ((size.width == 0) != (size.height == 0)) {
    return refuse("the rectified size has one side 0");
  }

  const camera::ImageSize image = c.left.size;
  if (size.width == 0) size = image;
  const double nx = image.width;
  const double ny = image.height;
  StereoRectification out;
  out.rig.camera.size = size;

  if (already_rectified(c)) {
    // Resampled only: the camera scaled by one factor, so its pixels keep
    // their shape, about the centres of the picture and the view (pixel
    // centres at integers, so an image spans [-0.5, w - 0.5]). Alpha picks
    // the factor as for any pair: 0 fills the view, cropping the picture's
    // longer side; 1 fits the whole picture, bordering the view's.
    const double sx = size.width / nx;
    const double sy = size.height / ny;
    const double s = std::max(sx, sy) * (1 - alpha) + std::min(sx, sy) * alpha;
    const camera::PinholeIntrinsics& k = c.left.intrinsics;
    out.rig.camera.intrinsics = {
        k.fx * s, k.fy * s, (size.width - 1) / 2.0 + (k.cx - (nx - 1) / 2) * s,
        (size.height - 1) / 2.0 + (k.cy - (ny - 1) / 2) * s};
    out.rig.baseline = -c.left_to_right[3][0];
    return out;
  }

  // The relative rotation split in half: each camera turned half-way.
  const Mat3d r(c.left_to_right);
  const Vec3d translation(c.left_to_right[3]);
  const Mat3d r_r = camera::rotation_from_rodrigues(
      -0.5 * camera::rodrigues_from_rotation(r));
  Vec3d t = r_r * translation;
  const int idx = std::fabs(t.x) > std::fabs(t.y) ? 0 : 1;
  const double cc = t[idx];
  const double nt = glm::length(t);
  if (!(nt > 0.0)) return refuse("the two cameras are at one place");
  if (idx != 0) {
    // TODO(sensor): a vertical pair, which OpenCV rectifies along y.
    return refuse(
        "the pair is vertical: the right camera must be along the "
        "left one's x axis");
  }
  if (cc > 0) {
    return refuse("the right camera is on the left one's -x side: swap them");
  }
  // Then the baseline turned onto -x of the right camera's offset, about the
  // axis that takes it there.
  Vec3d uu(0.0);
  uu[idx] = cc > 0 ? 1 : -1;
  Vec3d ww = glm::cross(t, uu);
  const double nw = glm::length(ww);
  if (nw > 0.0) ww *= std::acos(std::fabs(cc) / nt) / nw;
  const Mat3d w_r = camera::rotation_from_rodrigues(ww);
  const Mat3d r1 = w_r * glm::transpose(r_r);
  const Mat3d r2 = w_r * r_r;
  t = r2 * translation;

  // One focal length for both views: the vertical ones, scaled to the
  // rectified height.
  const double ratio_y = static_cast<double>(size.height) / ny / 2;
  double fc = (c.left.intrinsics.fy + c.right.intrinsics.fy) * ratio_y;
  // Each principal point: where the image's corners, undistorted and turned,
  // centre.
  Vec2d cc_new[2];
  for (int k = 0; k < 2; ++k) {
    const CameraModel& cam = k == 0 ? c.left : c.right;
    const Mat3d& rk = k == 0 ? r1 : r2;
    Vec2d sum(0.0);
    for (int i = 0; i < 4; ++i) {
      const Vec2d corner((i % 2) * (nx - 1), (i < 2 ? 0 : 1) * (ny - 1));
      const Vec2d n = undistort_point(cam, corner, nullptr);
      const Vec3d q = rk * Vec3d(n.x, n.y, 1.0);
      sum += Vec2d(fc * q.x / q.z, fc * q.y / q.z);
    }
    cc_new[k] = Vec2d((nx - 1) / 2, (ny - 1) / 2) - sum / 4.0;
  }
  // CALIB_ZERO_DISPARITY: one principal point for both.
  const Vec2d c0 = (cc_new[0] + cc_new[1]) * 0.5;

  // The scale alpha chooses: s0 shows only pixels both views see, s1 every
  // source pixel.
  Rect inner[2], outer[2];
  undistort_rectangles(c.left, r1, fc, c0, &inner[0], &outer[0]);
  undistort_rectangles(c.right, r2, fc, c0, &inner[1], &outer[1]);
  const double cx = size.width * c0.x / nx;
  const double cy = size.height * c0.y / ny;
  const double w1 = size.width - 1.0;
  const double h1 = size.height - 1.0;
  double s0 = -std::numeric_limits<double>::max();
  double s1 = std::numeric_limits<double>::max();
  for (int k = 0; k < 2; ++k) {
    const Rect& i = inner[k];
    const Rect& o = outer[k];
    s0 = std::max({s0, cx / (c0.x - i.x), cy / (c0.y - i.y),
                   (w1 - cx) / (i.x + i.width - c0.x),
                   (h1 - cy) / (i.y + i.height - c0.y)});
    s1 = std::min({s1, cx / (c0.x - o.x), cy / (c0.y - o.y),
                   (w1 - cx) / (o.x + o.width - c0.x),
                   (h1 - cy) / (o.y + o.height - c0.y)});
  }
  const double s = s0 * (1 - alpha) + s1 * alpha;
  fc *= s;
  if (!(std::isfinite(fc) && fc > 0 && std::isfinite(cx) &&
        std::isfinite(cy))) {
    return refuse(
        "the rectification is degenerate: the lenses fold or the "
        "cameras look too far apart");
  }
  out.rig.camera.intrinsics = {fc, fc, cx, cy};
  out.rig.baseline = -t.x;
  out.rig.rectified_to_left = rotation4(glm::transpose(r1));
  out.right_to_rectified = r2;
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory(std::string(kWho) +
                                     ": out of host memory");
}

float fold_radius2(const camera::RationalDistortion& d) {
  const auto folds = [&d](double angle) {
    const double t = std::tan(angle);
    const double s = t * t;
    const double n = 1 + s * (d.k1 + s * (d.k2 + s * d.k3));
    const double dn = 1 + s * (d.k4 + s * (d.k5 + s * d.k6));
    const double n1 = d.k1 + s * (2 * d.k2 + 3 * s * d.k3);
    const double d1 = d.k4 + s * (2 * d.k5 + 3 * s * d.k6);
    return !(dn > 0 && n * dn + 2 * s * (n1 * dn - n * d1) > 0);
  };
  constexpr int kSteps = 9000;
  const double step = std::acos(0.0) / kSteps;
  for (int i = 1; i < kSteps; ++i) {
    if (!folds(i * step)) continue;
    double lo = (i - 1) * step;
    double hi = i * step;
    for (int k = 0; k < 60; ++k) {
      const double mid = 0.5 * (lo + hi);
      (folds(mid) ? hi : lo) = mid;
    }
    const double t = std::tan(lo);
    return static_cast<float>(
        std::min<double>(t * t, std::numeric_limits<float>::max()));
  }
  return std::numeric_limits<float>::max();
}

}  // namespace volumetric_kit::sensor
