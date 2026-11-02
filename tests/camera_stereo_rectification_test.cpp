// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The camera tier's stereo pair (camera/stereo_rectification.hpp), on the
// host, so camera-only builds run it too:
//   - stereo_calibration: a pair from two entries of a calibration file, the
//     left camera's point in the right one, to round-off, named either way
//     round, and each refusal, named;
//   - rectify_stereo: synthetic correspondences -- one point's rays in the
//     two cameras -- land on one row of the views within 1e-9 px, and
//     fx * baseline / z apart, for a toed-in pair, the Blackmagic pair's
//     provisional calibration and a converging pair, at alpha 0, 0.5 and 1,
//     resized or not; R2 is the right camera's own rotation (R1 times the
//     inverse of the pair's, the baseline turned onto -x), not R1; a pair
//     rectified already is only resampled, both rotations the identity and
//     its camera scaled by one factor about the centres; the file's pair
//     rectifies, its baseline the cameras' distance; coincident, vertical and
//     swapped pairs, the file's named the wrong way round among them, and
//     each bad input are refused, InvalidArgument, named;
//   - fold_radius2: a lens with k1 < 0 alone folds at r^2 = -1/(3 k1), and
//     one with k4 < 0 alone at its denominator's pole, -1/k4; a pinhole and
//     the provisional calibration's lens never fold (float's largest); a 2024
//     fit of a Blackmagic lens folds at 74.8 degrees off the axis.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "stereo_pair_file.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/camera/stereo_rectification.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

using camera::CameraModel;
using camera::Mat3d;
using camera::Mat4d;
using camera::Vec3d;
using stereo_pair_file::left_to_right;
using stereo_pair_file::lens;
using stereo_pair_file::pair_file;

// The lens of the Blackmagic pair's provisional calibration (2026-11-01):
// the 2026-09-29 fit of one camera at 1080p, scaled to 2160p, on both
// cameras.
CameraModel provisional_lens() {
  CameraModel c;
  c.size = {3840, 2160};
  c.intrinsics = {1702.410047823579, 1702.1183368985696, 1917.379808460028,
                  1108.2673697604462};
  c.distortion = {0.7635753673346596,     0.1339021352077378,
                  -7.126331072400135e-05, 4.5662431317182075e-05,
                  0.0032974798211048597,  1.1293583015706812,
                  0.3232813210032048,     0.02266887261491428};
  return c;
}

// That calibration's right camera, as its pose: the left camera's point in
// it, the left camera the world.
Mat4d provisional_left_to_right() {
  return camera::matrix_from_rodrigues(
      {Vec3d(-0.015865302441396372, -0.0046460690322987, -0.002866660807177224),
       Vec3d(-0.3994849048218282, 0.01892731759163713, 0.0073189800027165745)});
}

// A 2024 fit of a Blackmagic camera's lens (bm-2 of the old four-camera rig,
// 3840x2160), whose rational model folds before the image corners.
camera::RationalDistortion fit_2024_bm2() {
  return {0.43858459246177117,    -0.008806271952411146,
          0.0001088146725150302,  -0.00011317892942162059,
          -0.0010377761285881846, 0.8032619209291925,
          0.06157382758196919,    -0.005978906562434272};
}

double worst_difference(const Mat4d& a, const Mat4d& b) {
  double worst = 0.0;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      worst = std::fmax(worst, std::fabs(a[c][r] - b[c][r]));
    }
  }
  return worst;
}

double worst_difference(const Mat3d& a, const Mat3d& b) {
  return worst_difference(Mat4d(a), Mat4d(b));
}

bool same_model(const CameraModel& a, const CameraModel& b) {
  return a.size.width == b.size.width && a.size.height == b.size.height &&
         a.intrinsics.fx == b.intrinsics.fx &&
         a.intrinsics.fy == b.intrinsics.fy &&
         a.intrinsics.cx == b.intrinsics.cx &&
         a.intrinsics.cy == b.intrinsics.cy &&
         a.distortion.k1 == b.distortion.k1 &&
         a.distortion.k2 == b.distortion.k2 &&
         a.distortion.p1 == b.distortion.p1 &&
         a.distortion.k4 == b.distortion.k4;
}

// Whether `made` is refused as InvalidArgument, naming `why`.
template <typename T>
bool refused(const vkc::Result<T>& made, const char* why) {
  if (made.ok() ||
      made.status().domain() != vkc::Status::Code::InvalidArgument ||
      made.status().message().find(why) == std::string::npos) {
    std::fprintf(stderr, "expected \"%s\": %s\n", why,
                 made.ok() ? "accepted" : made.status().message().c_str());
    return false;
  }
  return true;
}

// --- stereo_calibration ------------------------------------------------------

// The pair is the two entries' colour models and the left camera's point in
// the right one, named either way round; and each refusal.
int test_calibration() {
  const camera::ArrayCalibration file = pair_file();
  CHECK(camera::validate_array_calibration(file).ok());
  auto made = sensor::stereo_calibration(file, "bm-1", "bm-2");
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  CHECK(same_model(made->left, lens(0.0)) &&
        same_model(made->right, lens(3.5)));
  const double forward = worst_difference(made->left_to_right, left_to_right());
  auto swapped = sensor::stereo_calibration(file, "bm-2", "bm-1");
  CHECK(swapped.ok());
  CHECK(same_model(swapped->left, lens(3.5)));
  const double back = worst_difference(swapped->left_to_right,
                                       camera::rigid_inverse(left_to_right()));
  std::printf("  left_to_right %.1e from the pair's, %.1e named back\n",
              forward, back);
  CHECK(forward < 1e-12 && back < 1e-12);

  CHECK(refused(
      sensor::stereo_calibration(camera::ArrayCalibration{}, "bm-1", "bm-2"),
      "stereo_calibration: array calibration: no sensors"));
  CHECK(refused(sensor::stereo_calibration(file, "bm-1", "bm-1"),
                "are both \"bm-1\""));
  CHECK(refused(sensor::stereo_calibration(file, "bm-1", "bm-3"),
                "the right camera \"bm-3\" is not in the file"));
  camera::ArrayCalibration bad = file;
  bad.sensors[0].color_to_world.reset();
  CHECK(refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"),
                "the left camera \"bm-1\" is unposed"));
  bad = file;
  bad.sensors[1].color.reset();
  CHECK(refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"),
                "the right camera \"bm-2\" has no colour model"));
  CHECK(refused(sensor::stereo_calibration(file, "bm-1", "femto"),
                "the right camera \"femto\" has no colour model"));
  bad = file;
  bad.sensors[1].color->model.size = {1920, 1080};
  CHECK(refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"),
                "sizes differ: 3840x2160 and 1920x1080"));
  // Turned, but at the left camera's place.
  bad = file;
  Mat4d turned = *bad.sensors[0].color_to_world *
                 camera::matrix_from_rodrigues({Vec3d(0, 0.1, 0), Vec3d(0.0)});
  turned[3] = (*bad.sensors[0].color_to_world)[3];
  bad.sensors[1].color_to_world = turned;
  CHECK(
      refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"), "at one place"));
  std::puts("stereo_calibration: the pair either way round, each refusal");
  return 0;
}

// --- rectify_stereo ----------------------------------------------------------

struct PairCase {
  const char* name;
  sensor::StereoCalibration pair;
};

std::vector<PairCase> pair_cases() {
  std::vector<PairCase> cases;
  {
    sensor::StereoCalibration p;
    p.left = lens(0.0);
    p.right = lens(3.5);
    p.left_to_right = left_to_right();
    cases.push_back({"the file's pair, toed in and tilted", p});
  }
  {
    sensor::StereoCalibration p;
    p.left = p.right = provisional_lens();
    p.left_to_right = provisional_left_to_right();
    cases.push_back({"the Blackmagic pair's provisional calibration", p});
  }
  {
    // Barrel lenses, the cameras turned 6 degrees toward each other.
    sensor::StereoCalibration p;
    p.left = {{800, 600},
              {420.0, 421.0, 401.0, 297.0},
              {-0.21, 0.05, 0.001, -0.0007, -0.004, 0.02, 0.0, 0.0}};
    p.right = {{800, 600},
               {418.5, 419.25, 396.5, 303.75},
               {-0.2, 0.045, -0.0005, 0.0009, -0.003, 0.015, 0.0, 0.0}};
    p.left_to_right = camera::matrix_from_rodrigues(
        {Vec3d(0.015, 0.105, -0.012), Vec3d(-0.15, 0.003, 0.008)});
    cases.push_back({"barrel lenses, converging", p});
  }
  return cases;
}

// Each point's rays in the two cameras, turned into the views by R1 and R2,
// project onto one row of the rig's camera, fx * baseline / z apart; and R2
// is the right camera's own rotation: R1 times the inverse of the pair's,
// taking the right camera's offset onto -x.
int test_rows() {
  double worst_row = 0.0, worst_disparity = 0.0, worst_r2 = 0.0,
         worst_offset = 0.0, least_r1_r2 = 1e9;
  int points = 0;
  for (const PairCase& c : pair_cases()) {
    for (const camera::ImageSize size :
         {camera::ImageSize{960, 544}, camera::ImageSize{}}) {
      for (const double alpha : {0.0, 0.5, 1.0}) {
        auto made = sensor::rectify_stereo(c.pair, size, alpha);
        if (!made) {
          std::fprintf(stderr, "%s: %s\n", c.name,
                       made.status().message().c_str());
        }
        CHECK(made.ok());
        const sensor::StereoRig& rig = made->rig;
        const camera::ImageSize want =
            size.width != 0 ? size : c.pair.left.size;
        CHECK(rig.camera.size.width == want.width &&
              rig.camera.size.height == want.height);
        CHECK(camera::check_camera_model(rig.camera).ok());
        CHECK(rig.baseline > 0.0);
        CHECK(camera::check_rigid(rig.rectified_to_left).ok());
        CHECK(rig.rectified_to_left[3] == glm::dvec4(0, 0, 0, 1));
        const Mat3d r1 = glm::transpose(Mat3d(rig.rectified_to_left));
        const Mat3d& r2 = made->right_to_rectified;
        CHECK(camera::check_rigid(Mat4d(r2)).ok());
        const Mat3d r(c.pair.left_to_right);
        const Vec3d t(c.pair.left_to_right[3]);
        worst_r2 = std::fmax(worst_r2, worst_difference(r2 * r, r1));
        worst_offset = std::fmax(
            worst_offset, glm::length(r2 * t - Vec3d(-rig.baseline, 0, 0)));
        least_r1_r2 = std::fmin(least_r1_r2, worst_difference(r1, r2));

        const camera::PinholeIntrinsics& k = rig.camera.intrinsics;
        for (const double z : {0.8, 2.0, 5.0}) {
          for (int j = -4; j <= 4; ++j) {
            for (int i = -6; i <= 6; ++i) {
              const Vec3d p(z * 0.12 * i, z * 0.1 * j, z);  // the left camera's
              const Vec3d q = r * p + t;                    // the right one's
              const Vec3d a = r1 * p;
              const Vec3d b = r2 * q;
              const double ul = k.fx * a.x / a.z + k.cx;
              const double vl = k.fy * a.y / a.z + k.cy;
              const double ur = k.fx * b.x / b.z + k.cx;
              const double vr = k.fy * b.y / b.z + k.cy;
              worst_row = std::fmax(worst_row, std::fabs(vl - vr));
              worst_disparity =
                  std::fmax(worst_disparity,
                            std::fabs((ul - ur) - k.fx * rig.baseline / a.z));
              ++points;
            }
          }
        }
      }
    }
  }
  std::printf(
      "  %d correspondences: rows within %.2g px, disparity within %.2g px "
      "of fx * baseline / z; R2 * R within %.2g of R1, R2 * T within %.2g m "
      "of -x, R1 and R2 at least %.3f apart\n",
      points, worst_row, worst_disparity, worst_r2, worst_offset, least_r1_r2);
  CHECK(worst_row < 1e-9);
  CHECK(worst_disparity < 1e-9);
  CHECK(worst_r2 < 1e-12);
  CHECK(worst_offset < 1e-12);
  CHECK(least_r1_r2 > 1e-3);  // so R2 taken as R1 would show
  std::puts("rectify_stereo: rows aligned, R2 the right camera's own");
  return 0;
}

// A pair rectified already: both rotations the identity, its camera scaled
// by one factor about the centres, cropped at alpha 0 and fitted at 1.
int test_already_rectified() {
  const CameraModel ir{{1280, 800}, {640.5, 640.5, 637.25, 401.0}, {}};
  sensor::StereoCalibration pair;
  pair.left = pair.right = ir;
  pair.left_to_right[3] = glm::dvec4(-0.095, 0.0, 0.0, 1.0);

  auto same = sensor::rectify_stereo(pair, {}, 0.0);
  CHECK(same.ok());
  const camera::PinholeIntrinsics& own = same->rig.camera.intrinsics;
  CHECK(same->rig.camera.size.width == 1280 &&
        same->rig.camera.size.height == 800);
  CHECK(own.fx == 640.5 && own.fy == 640.5 && own.cx == 637.25 &&
        own.cy == 401.0);
  CHECK(same->rig.rectified_to_left == Mat4d(1.0));
  CHECK(same->right_to_rectified == Mat3d(1.0));
  CHECK(same->rig.baseline == 0.095);

  // To 960x544: 0.75 across, 0.68 down.
  for (const double alpha : {0.0, 1.0}) {
    auto scaled = sensor::rectify_stereo(pair, {960, 544}, alpha);
    CHECK(scaled.ok());
    const camera::PinholeIntrinsics& k = scaled->rig.camera.intrinsics;
    const double s = alpha == 0.0 ? 0.75 : 0.68;
    std::printf(
        "  rectified already, to 960x544 at alpha %.0f: fx %.6f, "
        "cx %.6f, cy %.6f\n",
        alpha, k.fx, k.cx, k.cy);
    CHECK(std::fabs(k.fx - 640.5 * s) < 1e-12 && k.fx == k.fy);
    CHECK(std::fabs(k.cx - (479.5 + (637.25 - 639.5) * s)) < 1e-12);
    CHECK(std::fabs(k.cy - (271.5 + (401.0 - 399.5) * s)) < 1e-12);
    CHECK(scaled->rig.rectified_to_left == Mat4d(1.0));
    CHECK(scaled->right_to_rectified == Mat3d(1.0));
    CHECK(scaled->rig.baseline == 0.095);
  }
  std::puts("rectify_stereo: a pair rectified already only resampled");
  return 0;
}

// The file's pair rectifies, its baseline the cameras' distance apart; what
// rectify_stereo refuses, each named.
int test_refusals() {
  const camera::ArrayCalibration file = pair_file();
  auto pair = sensor::stereo_calibration(file, "bm-1", "bm-2");
  CHECK(pair.ok());
  auto made = sensor::rectify_stereo(*pair, {960, 544}, 0.0);
  CHECK(made.ok());
  const double distance = glm::length(Vec3d(left_to_right()[3]));
  std::printf(
      "  the file's pair: baseline %.9f m (cameras %.9f m apart), "
      "fx %.2f\n",
      made->rig.baseline, distance, made->rig.camera.intrinsics.fx);
  CHECK(std::fabs(made->rig.baseline - distance) < 1e-9);

  // Named the wrong way round, the right camera is on the left one's -x
  // side: a caller tries the pair reversed.
  auto reversed = sensor::stereo_calibration(file, "bm-2", "bm-1");
  CHECK(reversed.ok());
  CHECK(refused(sensor::rectify_stereo(*reversed, {960, 544}, 0.0),
                "rectify_stereo: the right camera is on the left one's -x"));

  const sensor::StereoCalibration good = *pair;
  const camera::ImageSize size{960, 544};
  sensor::StereoCalibration bad = good;
  bad.left_to_right = Mat4d(1.0);
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "at one place"));
  bad.left_to_right[3] = glm::dvec4(0.0, -0.1, 0.0, 1.0);
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "vertical"));
  bad.left_to_right[3] = glm::dvec4(0.095, 0.0, 0.001, 1.0);
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "-x side"));
  // A pair that would be rectified already but for the right camera on -x.
  bad.left = bad.right =
      CameraModel{{1280, 800}, {640.5, 640.5, 637.25, 401.0}, {}};
  bad.left_to_right[3] = glm::dvec4(0.095, 0.0, 0.0, 1.0);
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "-x side"));

  CHECK(refused(sensor::rectify_stereo(sensor::StereoCalibration{}, size, 0.0),
                "the left camera: camera model: image size has a zero side"));
  bad = good;
  bad.left.intrinsics.fx = 0;
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "the left camera"));
  bad = good;
  bad.right.distortion.k2 = std::nan("");
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "the right camera"));
  bad = good;
  bad.right.size = {1920, 1080};
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "sizes differ"));
  bad = good;
  bad.left_to_right[0][0] = 2.0;
  CHECK(refused(sensor::rectify_stereo(bad, size, 0.0), "left_to_right"));
  for (const double alpha : {-0.1, 1.5, std::nan("")}) {
    CHECK(refused(sensor::rectify_stereo(good, size, alpha), "alpha"));
  }
  CHECK(refused(sensor::rectify_stereo(good, {960, 0}, 0.0), "one side 0"));
  std::puts("rectify_stereo: the file's pair, and each refusal named");
  return 0;
}

// --- fold_radius2 ------------------------------------------------------------

double degrees(float r2) {
  return std::atan(std::sqrt(static_cast<double>(r2))) * 180.0 /
         3.14159265358979323846;
}

int test_fold() {
  constexpr float kNever = std::numeric_limits<float>::max();
  // k1 alone: r (1 + k1 r^2) stops growing at 1 + 3 k1 r^2 = 0.
  double worst = 0.0;
  for (const double k1 : {-0.3, -0.1, -0.03, -0.01}) {
    camera::RationalDistortion d;
    d.k1 = k1;
    const double want = -1.0 / (3.0 * k1);
    const float got = sensor::fold_radius2(d);
    std::printf("  k1 %.2f: folds at r^2 %.7f (want %.7f), %.2f degrees\n", k1,
                static_cast<double>(got), want, degrees(got));
    worst = std::fmax(worst, std::fabs(got - want) / want);
  }
  CHECK(worst < 1e-6);
  // k4 alone: r / (1 + k4 r^2) grows all the way to its pole, 1 + k4 r^2 = 0.
  {
    camera::RationalDistortion d;
    d.k4 = -0.2;
    const float got = sensor::fold_radius2(d);
    std::printf("  k4 -0.20: folds at its pole, r^2 %.7f (want 5)\n",
                static_cast<double>(got));
    CHECK(std::fabs(got - 5.0) < 5e-6);
  }
  CHECK(sensor::fold_radius2(camera::RationalDistortion{}) == kNever);
  CHECK(sensor::fold_radius2(provisional_lens().distortion) == kNever);
  // Tangential terms play no part.
  camera::RationalDistortion tangential = provisional_lens().distortion;
  tangential.p1 = 0.3;
  tangential.p2 = -0.3;
  CHECK(sensor::fold_radius2(tangential) == kNever);
  const float bm2 = sensor::fold_radius2(fit_2024_bm2());
  std::printf(
      "  the 2024 fit folds at %.3f degrees; the provisional lens "
      "never does\n",
      degrees(bm2));
  CHECK(std::fabs(degrees(bm2) - 74.8) < 0.05);
  std::puts("fold_radius2: k1's turn, k4's pole, a lens that never folds");
  return 0;
}

}  // namespace

int main() {
  if (test_calibration() != 0) return 1;
  if (test_rows() != 0) return 1;
  if (test_already_rectified() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_fold() != 0) return 1;
  std::puts("camera stereo rectification tests passed");
  return 0;
}
