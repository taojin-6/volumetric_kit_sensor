// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StereoRectifier on synthetic calibrated pairs, no camera:
//   - its rectification is OpenCV's stereoRectify (Bouguet's, zero
//     disparity), to the numbers tools/stereo_rectify_reference.py pinned:
//     both rotations, the camera and the baseline, for mild and strong lenses,
//     toed-in and converging pairs, alpha 0, 0.5 and 1, resized or not;
//   - blobs on a plane, imaged through two lensed cameras, rotated and
//     offset, by an independent host renderer (each camera pixel's ray found
//     by inverting the lens iteratively, where the pass runs it forward),
//     come out on one row in both views, within 0.05 px, where the rig
//     projects them;
//   - each view matches a host reference of the pass's remap within a code:
//     luma, YUYV, NV12 and I420, as luma bytes and as R'G'B' words, the
//     coverage byte exact, and limited-range luma;
//   - the same pictures in device buffers give the same views, bit for bit;
//   - past a lens's fold the pass writes 0: on the strong barrel pair at
//     alpha 1, 5-10% of each view's rays are past it, where the lens model
//     would put them back inside the picture, mirrored, and every one is 0,
//     the fold found independently (the review's host replica);
//   - pictures in device buffers are taken over from their writer's family,
//     VK_QUEUE_FAMILY_EXTERNAL's or another of the device's, once a buffer:
//     a side-by-side frame's one buffer, a colour picture in it too, records
//     one transfer, and pictures in one buffer that name two families are
//     refused;
//   - a pair rectified already only resamples: at its own size its luma
//     comes back byte for byte, and at another its camera is scaled by one
//     factor, the picture cropped at alpha 0 and fitted at alpha 1;
//   - the depth view for GpuFramePrep (depth_info): depth in the rectified
//     left camera, colour the left camera or the third one, and the
//     source's id, model, clock, rate and colour exposure;
//   - pairs used in turn have the same two buffers, a Vulkan reader keeping
//     them or not; a pair kept past the next keeps its views, and a view no
//     pair holds is written again; where the device exports memory the
//     views are exported buffers handed out from VK_QUEUE_FAMILY_EXTERNAL;
//   - what it refuses, at create and per frame;
//   - the pass's time at 1280x720 to 960x544, both views, printed.
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"
#include "volumetric_kit/sensor/chroma_location.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"
#include "yuv_reference.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

using camera::CameraModel;
using camera::Mat3d;
using camera::Mat4d;
using camera::Vec2d;
using camera::Vec3d;
using Rectifier = sensor::StereoRectifier;

vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

// --- OpenCV's numbers -------------------------------------------------------

struct OpenCvCase {
  const char* name;
  CameraModel left, right;
  double rvec[3], tvec[3];  // left_to_right, as OpenCV's R (Rodrigues) and T
  camera::ImageSize size;
  double alpha;
  double r1[9], r2[9], p1[12], p2[12];  // row-major, as OpenCV prints them
};

// tools/stereo_rectify_reference.py, OpenCV 5.0.0
const OpenCvCase kOpenCvCases[] = {
    {"mild lenses, toed in, to 960x544, alpha 0",
     {{1280, 720},
      {652.5, 651.75, 641.25, 358.5},
      {-0.052, 0.031, 0.0007, -0.0004, -0.0061, 0.012, 0.0021, -0.0009}},
     {{1280, 720},
      {649.25, 650.5, 636.75, 362.25},
      {-0.047, 0.026, -0.0003, 0.0006, -0.0048, 0.009, 0.0, 0.0}},
     {0.0123, -0.0347, 0.0061},
     {-0.09503, 0.0012, -0.0021},
     {960, 544},
     0.0,
     {0.9997488061688111, -0.01866291464067818, -0.012410486741139221,
      0.018584449851300255, 0.9998067590793931, -0.006408020200098485,
      0.012527680861266353, 0.006175768476483172, 0.99990245379035},
     {0.999676262151829, -0.01262350325772873, 0.022091130701028836,
      0.01248423827298086, 0.9999013899773131, 0.006430716614845072,
      -0.02217013046626626, -0.0061528438090947615, 0.9997352788754479},
     {491.30709188830133, 0.0, 475.7912893295288, 0.0, 0.0, 491.30709188830133,
      272.30182562934027, 0.0, 0.0, 0.0, 1.0, 0.0},
     {491.30709188830133, 0.0, 475.7912893295288, -46.704032805226554, 0.0,
      491.30709188830133, 272.30182562934027, 0.0, 0.0, 0.0, 1.0, 0.0}},
    {"mild lenses, toed in, to 960x544, alpha 1",
     {{1280, 720},
      {652.5, 651.75, 641.25, 358.5},
      {-0.052, 0.031, 0.0007, -0.0004, -0.0061, 0.012, 0.0021, -0.0009}},
     {{1280, 720},
      {649.25, 650.5, 636.75, 362.25},
      {-0.047, 0.026, -0.0003, 0.0006, -0.0048, 0.009, 0.0, 0.0}},
     {0.0123, -0.0347, 0.0061},
     {-0.09503, 0.0012, -0.0021},
     {960, 544},
     1.0,
     {0.9997488061688111, -0.01866291464067818, -0.012410486741139221,
      0.018584449851300255, 0.9998067590793931, -0.006408020200098485,
      0.012527680861266353, 0.006175768476483172, 0.99990245379035},
     {0.999676262151829, -0.01262350325772873, 0.022091130701028836,
      0.01248423827298086, 0.9999013899773131, 0.006430716614845072,
      -0.02217013046626626, -0.0061528438090947615, 0.9997352788754479},
     {443.941841005151, 0.0, 475.7912893295288, 0.0, 0.0, 443.941841005151,
      272.30182562934027, 0.0, 0.0, 0.0, 1.0, 0.0},
     {443.941841005151, 0.0, 475.7912893295288, -42.20145535906714, 0.0,
      443.941841005151, 272.30182562934027, 0.0, 0.0, 0.0, 1.0, 0.0}},
    {"mild lenses, toed in, own size, alpha 0.5",
     {{1280, 720},
      {652.5, 651.75, 641.25, 358.5},
      {-0.052, 0.031, 0.0007, -0.0004, -0.0061, 0.012, 0.0021, -0.0009}},
     {{1280, 720},
      {649.25, 650.5, 636.75, 362.25},
      {-0.047, 0.026, -0.0003, 0.0006, -0.0048, 0.009, 0.0, 0.0}},
     {0.0123, -0.0347, 0.0061},
     {-0.09503, 0.0012, -0.0021},
     {0, 0},
     0.5,
     {0.9997488061688111, -0.01866291464067818, -0.012410486741139221,
      0.018584449851300255, 0.9998067590793931, -0.006408020200098485,
      0.012527680861266353, 0.006175768476483172, 0.99990245379035},
     {0.999676262151829, -0.01262350325772873, 0.022091130701028836,
      0.01248423827298086, 0.9999013899773131, 0.006430716614845072,
      -0.02217013046626626, -0.0061528438090947615, 0.9997352788754479},
     {621.8226368993827, 0.0, 632.734619140625, 0.0, 0.0, 621.8226368993827,
      360.690486907959, 0.0, 0.0, 0.0, 1.0, 0.0},
     {621.8226368993827, 0.0, 632.734619140625, -59.11094163359617, 0.0,
      621.8226368993827, 360.690486907959, 0.0, 0.0, 0.0, 1.0, 0.0}},
    {"strong barrel lenses, converging, to 640x480, alpha 0",
     {{800, 600},
      {420.0, 421.0, 401.0, 297.0},
      {-0.28, 0.09, 0.0012, -0.0009, -0.01, 0.05, 0.0, 0.0}},
     {{800, 600},
      {418.5, 419.25, 396.5, 303.75},
      {-0.26, 0.08, -0.0006, 0.0011, -0.008, 0.04, 0.0, 0.0}},
     {0.02, 0.11, -0.015},
     {-0.12, 0.004, 0.009},
     {640, 480},
     0.0,
     {0.99919041629922, -0.018628261579352816, 0.03565809650448065,
      0.018960912876812504, 0.9997795954164821, -0.009013565983181118,
      -0.035482330231674096, 0.009682378808249669, 0.9993233989965131},
     {0.9966488651850164, -0.033221628839500875, -0.07474866488887648,
      0.03391965042199359, 0.9993918370204781, 0.008087855839697757,
      0.07443451377330855, -0.010596200926956203, 0.9971696062783135},
     {283.97960082016266, 0.0, 326.25732421875, 0.0, 0.0, 283.97960082016266,
      245.94703063964843, 0.0, 0.0, 0.0, 1.0, 0.0},
     {283.97960082016266, 0.0, 326.25732421875, -34.19213455091169, 0.0,
      283.97960082016266, 245.94703063964843, 0.0, 0.0, 0.0, 1.0, 0.0}},
    {"strong barrel lenses, converging, to 640x480, alpha 1",
     {{800, 600},
      {420.0, 421.0, 401.0, 297.0},
      {-0.28, 0.09, 0.0012, -0.0009, -0.01, 0.05, 0.0, 0.0}},
     {{800, 600},
      {418.5, 419.25, 396.5, 303.75},
      {-0.26, 0.08, -0.0006, 0.0011, -0.008, 0.04, 0.0, 0.0}},
     {0.02, 0.11, -0.015},
     {-0.12, 0.004, 0.009},
     {640, 480},
     1.0,
     {0.99919041629922, -0.018628261579352816, 0.03565809650448065,
      0.018960912876812504, 0.9997795954164821, -0.009013565983181118,
      -0.035482330231674096, 0.009682378808249669, 0.9993233989965131},
     {0.9966488651850164, -0.033221628839500875, -0.07474866488887648,
      0.03391965042199359, 0.9993918370204781, 0.008087855839697757,
      0.07443451377330855, -0.010596200926956203, 0.9971696062783135},
     {146.02081799106335, 0.0, 326.25732421875, 0.0, 0.0, 146.02081799106335,
      245.94703063964843, 0.0, 0.0, 0.0, 1.0, 0.0},
     {146.02081799106335, 0.0, 326.25732421875, -17.581415853691613, 0.0,
      146.02081799106335, 245.94703063964843, 0.0, 0.0, 0.0, 1.0, 0.0}},
};

// A row-major 3x3, as OpenCV prints one, in GLM's columns.
Mat3d from_rows(const double* m) {
  Mat3d out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) out[c][r] = m[r * 3 + c];
  }
  return out;
}

sensor::StereoInfo info_of(const CameraModel& left, const CameraModel& right,
                           const double* rvec, const double* tvec) {
  sensor::StereoInfo info;
  info.calibration.left = left;
  info.calibration.right = right;
  info.calibration.left_to_right = camera::matrix_from_rodrigues(
      {Vec3d(rvec[0], rvec[1], rvec[2]), Vec3d(tvec[0], tvec[1], tvec[2])});
  return info;
}

// Each view's rotation from its source camera into the rectified one, from
// the rig and the calibration alone: R1 undoes rectified_to_left, and the
// right view shares the left one's orientation.
Mat3d source_to_rect(const sensor::StereoRig& rig,
                     const sensor::StereoCalibration& c, int view) {
  const Mat3d r1 = glm::transpose(Mat3d(rig.rectified_to_left));
  return view == 0 ? r1 : r1 * glm::transpose(Mat3d(c.left_to_right));
}

std::unique_ptr<Rectifier> make(const sensor::StereoInfo& info,
                                const Rectifier::Options& options) {
  auto made = Rectifier::create(*g_device, *g_allocator, info, options);
  if (!made) {
    std::fprintf(stderr, "create: %s\n", made.status().message().c_str());
    return nullptr;
  }
  return std::move(made).value();
}

// Bouguet's rectification is OpenCV's: R1 and R2 to round-off, the camera
// (P1 and P2, one principal point) and the baseline (-P2's Tx / f).
int test_opencv() {
  double worst_r = 0, worst_f = 0, worst_c = 0, worst_b = 0;
  for (const OpenCvCase& c : kOpenCvCases) {
    const sensor::StereoInfo info = info_of(c.left, c.right, c.rvec, c.tvec);
    const auto rectifier = make(
        info, {c.size, static_cast<float>(c.alpha), Rectifier::Output::Luma8});
    CHECK(rectifier != nullptr);
    const sensor::StereoRig& rig = rectifier->rig();
    for (int v = 0; v < 2; ++v) {
      const Mat3d got = source_to_rect(rig, info.calibration, v);
      const Mat3d want = from_rows(v == 0 ? c.r1 : c.r2);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          worst_r = std::max(worst_r, std::fabs(got[i][j] - want[i][j]));
        }
      }
    }
    const camera::ImageSize size = c.size.width != 0 ? c.size : c.left.size;
    CHECK(rig.camera.size.width == size.width &&
          rig.camera.size.height == size.height);
    const camera::RationalDistortion& d = rig.camera.distortion;
    CHECK(d.k1 == 0 && d.k2 == 0 && d.p1 == 0 && d.p2 == 0 && d.k3 == 0 &&
          d.k4 == 0 && d.k5 == 0 && d.k6 == 0);
    CHECK(rig.rectified_to_left[3] == glm::dvec4(0, 0, 0, 1));
    const camera::PinholeIntrinsics& k = rig.camera.intrinsics;
    for (const double* p : {c.p1, c.p2}) {
      worst_f =
          std::max({worst_f, std::fabs(k.fx - p[0]), std::fabs(k.fy - p[5])});
      worst_c =
          std::max({worst_c, std::fabs(k.cx - p[2]), std::fabs(k.cy - p[6])});
    }
    worst_b = std::max(worst_b, std::fabs(rig.baseline + c.p2[3] / c.p2[0]));
  }
  std::printf(
      "  OpenCV's stereoRectify, %zu pairs: rotations within %.2g, focal "
      "length %.2g px, principal point %.2g px, baseline %.2g m\n",
      sizeof(kOpenCvCases) / sizeof(kOpenCvCases[0]), worst_r, worst_f, worst_c,
      worst_b);
  // OpenCV undistorts the corners in float, so its principal point and,
  // through alpha's scale, its focal length carry float's round-off.
  CHECK(worst_r < 1e-12);
  CHECK(worst_f < 1e-4);
  CHECK(worst_c < 1e-4);
  CHECK(worst_b < 1e-12);
  return 0;
}

// --- A scene, imaged through each camera ------------------------------------

// A plane in front of the pair, in the left camera's frame, through `origin`
// and spanned by `e1` and `e2`, with Gaussian blobs on a grid on it.
struct Scene {
  Vec3d origin{0.0, 0.0, 1.2};
  Vec3d normal{0.0};
  Vec3d e1{0.0}, e2{0.0};
  double sigma = 0.006;   // a blob's radius (metres)
  double spacing = 0.05;  // between blobs
  int half1 = 18, half2 = 10;
  static constexpr double kBackground = 30.0;
  static constexpr double kPeak = 200.0;

  Scene() {
    normal = glm::normalize(Vec3d(0.08, -0.05, 1.0));
    e1 = glm::normalize(glm::cross(Vec3d(0.0, 1.0, 0.0), normal));
    e2 = glm::cross(normal, e1);
  }

  std::vector<Vec3d> blobs() const {
    std::vector<Vec3d> out;
    for (int j = -half2; j <= half2; ++j) {
      for (int i = -half1; i <= half1; ++i) {
        out.push_back(origin + (i * spacing) * e1 + (j * spacing) * e2);
      }
    }
    return out;
  }

  // Where the ray from `from` along `dir` meets the plane, in its own
  // coordinates; false where it does not, ahead.
  bool hit(const Vec3d& from, const Vec3d& dir, Vec2d* at) const {
    const double along = glm::dot(dir, normal);
    if (std::fabs(along) < 1e-12) return false;
    const double t = glm::dot(origin - from, normal) / along;
    if (t <= 0) return false;
    const Vec3d x = from + t * dir - origin;
    *at = Vec2d(glm::dot(x, e1), glm::dot(x, e2));
    return true;
  }

  // The luma at plane point `p`: the nearest blob over the background.
  double luma(Vec2d p) const {
    const double i =
        std::clamp(std::round(p.x / spacing), -1.0 * half1, 1.0 * half1);
    const double j =
        std::clamp(std::round(p.y / spacing), -1.0 * half2, 1.0 * half2);
    const Vec2d d = p - Vec2d(i * spacing, j * spacing);
    return kBackground +
           kPeak * std::exp(-glm::dot(d, d) / (2 * sigma * sigma));
  }

  // R'G'B' at plane point `p`, each in 0..1: smooth, and different in each.
  std::array<double, 3> rgb(Vec2d p) const {
    return {0.5 + 0.4 * std::sin(p.x * 21.0 + 0.3),
            0.5 + 0.4 * std::cos(p.y * 17.0 - 0.2),
            0.5 + 0.4 * std::sin((p.x + p.y) * 13.0)};
  }
};

// The normalized ray camera `c`'s pixel (u, v) sees: the lens inverted by
// fixed-point iteration, independently of the pass's forward model.
Vec2d undistort(const CameraModel& c, double u, double v) {
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  const double coeffs[8] = {d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6};
  const Vec2d target((u - k.cx) / k.fx, (v - k.cy) / k.fy);
  Vec2d p = target;
  for (int i = 0; i < 40; ++i) {
    Vec2d q;
    camera::distort_rational(coeffs, p.x, p.y, &q.x, &q.y);
    p += target - q;
  }
  return p;
}

using Points = std::vector<std::optional<Vec2d>>;

// A picture of the scene through camera `c`, which sits at `cam_to_left` in
// the left camera's frame: per pixel, the plane point its ray meets, or none.
// Each camera's is worked out once and kept, as the tests image one pair.
const Points& image_points(const Scene& scene, const CameraModel& c,
                           const Mat4d& cam_to_left) {
  struct Kept {
    CameraModel camera;
    Mat4d pose;
    Points points;
  };
  static std::vector<std::unique_ptr<Kept>> kept;
  for (const auto& k : kept) {
    if (std::memcmp(&k->camera, &c, sizeof(c)) == 0 && k->pose == cam_to_left) {
      return k->points;
    }
  }
  const Mat3d r(cam_to_left);
  const Vec3d from(cam_to_left[3]);
  Points out(std::size_t{c.size.width} * c.size.height);
  for (std::uint32_t v = 0; v < c.size.height; ++v) {
    for (std::uint32_t u = 0; u < c.size.width; ++u) {
      const Vec2d n = undistort(c, u, v);
      Vec2d at;
      if (scene.hit(from, r * Vec3d(n.x, n.y, 1.0), &at)) {
        out[std::size_t{v} * c.size.width + u] = at;
      }
    }
  }
  kept.push_back(std::make_unique<Kept>(Kept{c, cam_to_left, std::move(out)}));
  return kept.back()->points;
}

std::uint8_t to_code(double v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
}

// A picture's luma plane and the planes of its colour, as the layouts lay
// them out, with the YuvImage that names them on the host. Rows `pad` bytes
// longer than the picture's, the rest junk.
struct Picture {
  std::uint32_t w = 0, h = 0;
  std::vector<std::uint8_t> y, cb, cr;  // full-size planes, for the reference
  std::vector<std::uint8_t> bytes;      // the layout's planes, as handed over
  sensor::YuvImage image;
};

constexpr std::uint32_t kPad = 12;

Picture luma_picture(const Scene& scene, const CameraModel& c,
                     const Mat4d& cam_to_left) {
  const Points& points = image_points(scene, c, cam_to_left);
  Picture p;
  p.w = c.size.width;
  p.h = c.size.height;
  p.y.resize(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    p.y[i] = to_code(points[i] ? scene.luma(*points[i]) : 0.0);
  }
  const std::size_t stride = p.w + kPad;
  p.bytes.assign(3 + stride * p.h, 0xCD);
  for (std::uint32_t r = 0; r < p.h; ++r) {
    std::memcpy(&p.bytes[3 + r * stride], &p.y[std::size_t{r} * p.w], p.w);
  }
  p.image.layout = sensor::YuvLayout::Luma;
  p.image.host = p.bytes.data();
  p.image.offset[0] = 3;
  p.image.stride[0] = stride;
  p.image.width = p.w;
  p.image.height = p.h;
  p.image.full_range = true;
  return p;
}

// The scene's colour through camera `c` as Y'CbCr (BT.601, full range), in
// `layout`, chroma sited left: point samples at each chroma site.
Picture color_picture(const Scene& scene, const CameraModel& c,
                      const Mat4d& cam_to_left, sensor::YuvLayout layout) {
  const Points& points = image_points(scene, c, cam_to_left);
  Picture p;
  p.w = c.size.width;
  p.h = c.size.height;
  p.y.resize(points.size());
  p.cb.resize(points.size());
  p.cr.resize(points.size());
  const double kr = 0.299, kb = 0.114, kg = 1 - kr - kb;
  for (std::size_t i = 0; i < points.size(); ++i) {
    std::array<double, 3> rgb = {0, 0, 0};
    if (points[i]) rgb = scene.rgb(*points[i]);
    const double luma = kr * rgb[0] + kg * rgb[1] + kb * rgb[2];
    p.y[i] = to_code(luma * 255);
    p.cb[i] = to_code(128 + 255 * (rgb[2] - luma) / (2 * (1 - kb)));
    p.cr[i] = to_code(128 + 255 * (rgb[0] - luma) / (2 * (1 - kr)));
  }
  const std::uint32_t cw = (p.w + 1) / 2, ch = (p.h + 1) / 2;
  sensor::YuvImage& im = p.image;
  im.layout = layout;
  im.width = p.w;
  im.height = p.h;
  im.kr = 0.299f;
  im.kb = 0.114f;
  im.full_range = true;
  im.chroma_location = sensor::ChromaLocation::Left;
  const auto at = [&](const std::vector<std::uint8_t>& plane, std::uint32_t x,
                      std::uint32_t r) {
    return plane[std::size_t{std::min(r, p.h - 1)} * p.w +
                 std::min(x, p.w - 1)];
  };
  switch (layout) {
    case sensor::YuvLayout::Yuyv: {
      const std::size_t stride = 2 * std::size_t{p.w} + kPad;
      p.bytes.assign(1 + stride * p.h, 0xCD);
      for (std::uint32_t r = 0; r < p.h; ++r) {
        for (std::uint32_t i = 0; i < p.w / 2; ++i) {
          std::uint8_t* o = &p.bytes[1 + r * stride + 4 * std::size_t{i}];
          o[0] = at(p.y, 2 * i, r);
          o[1] = at(p.cb, 2 * i, r);
          o[2] = at(p.y, 2 * i + 1, r);
          o[3] = at(p.cr, 2 * i, r);
        }
      }
      im.offset[0] = 1;
      im.stride[0] = stride;
      break;
    }
    case sensor::YuvLayout::Nv12: {
      const std::size_t ys = p.w + kPad, cs = 2 * std::size_t{cw} + kPad;
      p.bytes.assign(ys * p.h + cs * ch, 0xCD);
      for (std::uint32_t r = 0; r < p.h; ++r) {
        std::memcpy(&p.bytes[r * ys], &p.y[std::size_t{r} * p.w], p.w);
      }
      for (std::uint32_t r = 0; r < ch; ++r) {
        for (std::uint32_t i = 0; i < cw; ++i) {
          p.bytes[ys * p.h + r * cs + 2 * i] = at(p.cb, 2 * i, 2 * r);
          p.bytes[ys * p.h + r * cs + 2 * i + 1] = at(p.cr, 2 * i, 2 * r);
        }
      }
      im.offset[0] = 0;
      im.offset[1] = ys * p.h;
      im.stride[0] = ys;
      im.stride[1] = cs;
      break;
    }
    case sensor::YuvLayout::I420: {
      const std::size_t ys = p.w + kPad, cs = cw + kPad;
      p.bytes.assign(ys * p.h + 2 * cs * ch, 0xCD);
      for (std::uint32_t r = 0; r < p.h; ++r) {
        std::memcpy(&p.bytes[r * ys], &p.y[std::size_t{r} * p.w], p.w);
      }
      for (std::uint32_t r = 0; r < ch; ++r) {
        for (std::uint32_t i = 0; i < cw; ++i) {
          p.bytes[ys * p.h + r * cs + i] = at(p.cb, 2 * i, 2 * r);
          p.bytes[ys * p.h + cs * ch + r * cs + i] = at(p.cr, 2 * i, 2 * r);
        }
      }
      im.offset[0] = 0;
      im.offset[1] = ys * p.h;
      im.offset[2] = ys * p.h + cs * ch;
      im.stride[0] = ys;
      im.stride[1] = im.stride[2] = cs;
      break;
    }
    case sensor::YuvLayout::Luma:
      break;
  }
  im.host = p.bytes.data();
  return p;
}

// `bytes` in a device buffer of their own, EXCLUSIVE, and left as a writer
// leaves it: released to VK_QUEUE_FAMILY_EXTERNAL, as CUDA's writes come, for
// `to` kQueueFamilyExternal, or kept, for kQueueFamilyIgnored. Null on
// failure.
std::shared_ptr<const vkc::Buffer> device_bytes(
    std::vector<std::uint8_t> bytes,
    std::uint32_t to = sensor::kQueueFamilyIgnored) {
  bytes.resize((bytes.size() + 3) & ~std::size_t{3}, 0xEE);
  auto made = vkc::device_storage_buffer(*g_allocator, bytes.size());
  if (!made.ok()) return nullptr;
  vkc::CommandBatch batch(*g_device, *g_allocator);
  if (!batch.upload(made.value(), 0, bytes.data(), bytes.size()).ok() ||
      !batch.release(made.value(), to).ok() || !batch.submit().ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

// The same planes in a device buffer, bound where they are, `lead` bytes in,
// released by their writer to `to`, which the picture names.
sensor::YuvImage on_device(const Picture& p, std::size_t lead,
                           std::uint32_t to = sensor::kQueueFamilyIgnored) {
  std::vector<std::uint8_t> bytes(lead, 0xEE);
  bytes.insert(bytes.end(), p.bytes.begin(), p.bytes.end());
  sensor::YuvImage im = p.image;
  im.host = nullptr;
  for (std::uint64_t& o : im.offset) o += lead;
  im.device = device_bytes(std::move(bytes), to);
  im.queue_family = to;
  return im;
}

sensor::StereoFrame frame_of(const Picture& left, const Picture& right) {
  sensor::StereoFrame f;
  f.left = left.image;
  f.right = right.image;
  f.timestamp_ns = 4321;
  f.sequence = 7;
  return f;
}

// --- Reading the views, and the host reference ------------------------------

// A view's bytes, read as a Vulkan reader takes them: over from the pair's
// queue family and, with `hand_back`, handed back to it; without, kept, as
// PictureReadback and GpuFramePrep keep what they take.
std::vector<std::uint8_t> read_view(const sensor::RectifiedPair& pair,
                                    const vkc::ExportedBuffer& view,
                                    std::size_t bytes, bool hand_back = true) {
  vkc::CommandBatch batch(*g_device, *g_allocator);
  std::vector<std::uint8_t> out(bytes);
  if (!batch.acquire(view.buffer, pair.queue_family).ok() ||
      !batch.readback(view.buffer, 0, bytes, out.data()).ok() ||
      (hand_back && !batch.release(view.buffer, pair.queue_family).ok()) ||
      !batch.submit().ok()) {
    std::fprintf(stderr, "read_view failed\n");
    return {};
  }
  return out;
}

std::size_t view_bytes(const sensor::StereoRig& rig, Rectifier::Output o) {
  const std::size_t pixels =
      std::size_t{rig.camera.size.width} * rig.camera.size.height;
  return o == Rectifier::Output::Luma8 ? (pixels + 3) & ~std::size_t{3}
                                       : 4 * pixels;
}

// The lens's fold as the review's host replica finds it, independently of
// the rectifier: the normalized radius at which r * radial(r^2) first stops
// growing, marched out 1e-4 at a time (so up to 1e-4 past it); infinity for
// none within r = 100.
double replica_fold(const camera::RationalDistortion& d) {
  double prev = 0.0;
  for (int i = 1; i <= 1000000; ++i) {
    const double r = i * 1e-4;
    const double r2 = r * r;
    const double v = r *
                     (1 + d.k1 * r2 + d.k2 * r2 * r2 + d.k3 * r2 * r2 * r2) /
                     (1 + d.k4 * r2 + d.k5 * r2 * r2 + d.k6 * r2 * r2 * r2);
    if (!(v > prev)) return r;
    prev = v;
  }
  return std::numeric_limits<double>::infinity();
}

// The pass's remap for one view, in float as the kernel computes it: each
// rectified pixel's source position, from the rig and the calibration, and
// the source lens's fold.
struct Remap {
  float m[3][3];  // rect_to_source, m[column][row]
  float fx, fy, cx, cy;
  std::uint32_t w, h;
  float k[8];
  float sfx, sfy, scx, scy;
  std::uint32_t sw, sh;
  double fold;  // replica_fold's

  Remap(const sensor::StereoRig& rig, const sensor::StereoCalibration& c,
        int view) {
    const Mat3d back = glm::transpose(source_to_rect(rig, c, view));
    for (int col = 0; col < 3; ++col) {
      for (int row = 0; row < 3; ++row) {
        m[col][row] = static_cast<float>(back[col][row]);
      }
    }
    const auto f = [](double v) { return static_cast<float>(v); };
    fx = f(rig.camera.intrinsics.fx);
    fy = f(rig.camera.intrinsics.fy);
    cx = f(rig.camera.intrinsics.cx);
    cy = f(rig.camera.intrinsics.cy);
    w = rig.camera.size.width;
    h = rig.camera.size.height;
    const CameraModel& s = view == 0 ? c.left : c.right;
    const camera::RationalDistortion& d = s.distortion;
    const float kk[8] = {f(d.k1), f(d.k2), f(d.p1), f(d.p2),
                         f(d.k3), f(d.k4), f(d.k5), f(d.k6)};
    std::copy(kk, kk + 8, k);
    sfx = f(s.intrinsics.fx);
    sfy = f(s.intrinsics.fy);
    scx = f(s.intrinsics.cx);
    scy = f(s.intrinsics.cy);
    sw = s.size.width;
    sh = s.size.height;
    fold = replica_fold(d);
  }

  // Where the lens model puts rectified pixel (u, v) in the source, and the
  // ray's normalized radius there, `*r`; false where the ray is behind the
  // camera or lands outside the picture. The fold is not applied.
  bool lands(std::uint32_t u, std::uint32_t v, float* sx, float* sy,
             float* r) const {
    const float x = (static_cast<float>(u) - cx) / fx;
    const float y = (static_cast<float>(v) - cy) / fy;
    const float rx = m[0][0] * x + m[1][0] * y + m[2][0];
    const float ry = m[0][1] * x + m[1][1] * y + m[2][1];
    const float rz = m[0][2] * x + m[1][2] * y + m[2][2];
    *r = 0.0f;
    if (rz <= 0.0f) return false;
    const float px = rx / rz, py = ry / rz;
    *r = std::sqrt(px * px + py * py);
    float xd = 0, yd = 0;
    camera::distort_rational(k, px, py, &xd, &yd);
    *sx = xd * sfx + scx;
    *sy = yd * sfy + scy;
    return *sx >= -0.5f && *sy >= -0.5f && *sx <= sw - 0.5f && *sy <= sh - 0.5f;
  }

  // Where rectified pixel (u, v) samples the source; false where the pass
  // writes 0: behind the camera, past the fold or outside the picture. `*r`
  // as lands'.
  bool source(std::uint32_t u, std::uint32_t v, float* sx, float* sy,
              float* r) const {
    return lands(u, v, sx, sy, r) && !(*r > fold);
  }

  // Whether a ray at radius `r` is so near the fold that the rectifier's and
  // the replica's may put it on either side (the replica's is up to 1e-4
  // past it, and float rounds the radius).
  bool near_fold(float r) const { return std::fabs(r - fold) < 1e-3; }

  // Whether (sx, sy) is so near the picture's edge that the device's and
  // the host's rounding may put it on either side.
  bool near_edge(float sx, float sy) const {
    const auto near = [](float a, float b) { return std::fabs(a - b) < 1e-3f; };
    return near(sx, -0.5f) || near(sy, -0.5f) || near(sx, sw - 0.5f) ||
           near(sy, sh - 0.5f);
  }
};

// The kernel's bilinear sample of a w x h plane at (x, y), clamped.
float bilinear(const std::vector<std::uint8_t>& plane, std::uint32_t pw,
               std::uint32_t w, std::uint32_t h, float x, float y) {
  x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
  y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
  const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(x));
  const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(y));
  const std::uint32_t x1 = std::min(x0 + 1, w - 1),
                      y1 = std::min(y0 + 1, h - 1);
  const float fx = x - x0, fy = y - y0;
  const auto at = [&](std::uint32_t xx, std::uint32_t yy) {
    return static_cast<float>(plane[std::size_t{yy} * pw + xx]);
  };
  const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
  const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
  return top + (bottom - top) * fy;
}

// The luma code the pass writes for luma `y` sampled from a picture of that
// range: Y', full range out.
int luma_code(float y, bool full_range) {
  const float level = full_range ? y / 255.0f : (y - 16.0f) / 219.0f;
  return static_cast<int>(std::lround(std::clamp(level, 0.0f, 1.0f) * 255.0f));
}

int channel_diff(std::uint32_t a, std::uint32_t b) {
  int worst = 0;
  for (int s = 0; s < 24; s += 8) {
    worst = std::max(worst, std::abs(static_cast<int>((a >> s) & 0xFF) -
                                     static_cast<int>((b >> s) & 0xFF)));
  }
  return worst;
}

// A view against the host reference of the remap: luma codes, or R'G'B'
// words from `picture`'s own planes as its layout subsamples them, by its
// range. Returns the worst code off, -1 for a coverage mismatch; `compared`
// counts pixels.
int compare_view(const std::vector<std::uint8_t>& got, const Remap& remap,
                 const Picture& picture, Rectifier::Output output,
                 std::size_t* compared) {
  const sensor::YuvLayout layout = picture.image.layout;
  const bool full = picture.image.full_range;
  const std::uint32_t cw = (picture.w + 1) / 2, ch = (picture.h + 1) / 2;
  // The chroma planes as the layout holds them, at their own size.
  std::vector<std::uint8_t> cb, cr;
  std::uint32_t pcw = picture.w, pch = picture.h;
  if (layout != sensor::YuvLayout::Luma) {
    pcw = cw;
    pch = layout == sensor::YuvLayout::Yuyv ? picture.h : ch;
    cb.resize(std::size_t{pcw} * pch);
    cr.resize(cb.size());
    for (std::uint32_t r = 0; r < pch; ++r) {
      for (std::uint32_t i = 0; i < pcw; ++i) {
        const std::uint32_t sr = std::min(
            layout == sensor::YuvLayout::Yuyv ? r : 2 * r, picture.h - 1);
        const std::size_t from =
            std::size_t{sr} * picture.w + std::min(2 * i, picture.w - 1);
        cb[std::size_t{r} * pcw + i] = picture.cb[from];
        cr[std::size_t{r} * pcw + i] = picture.cr[from];
      }
    }
  }
  int worst = 0;
  *compared = 0;
  for (std::uint32_t v = 0; v < remap.h; ++v) {
    for (std::uint32_t u = 0; u < remap.w; ++u) {
      const std::size_t i = std::size_t{v} * remap.w + u;
      float sx = 0, sy = 0, r = 0;
      const bool inside = remap.source(u, v, &sx, &sy, &r);
      if (remap.near_edge(sx, sy) || remap.near_fold(r)) continue;
      const float luma =
          inside ? bilinear(picture.y, picture.w, picture.w, picture.h, sx, sy)
                 : 0.0f;
      if (output == Rectifier::Output::Luma8) {
        const int want = inside ? luma_code(luma, full) : 0;
        worst = std::max(worst, std::abs(want - static_cast<int>(got[i])));
        ++*compared;
        continue;
      }
      std::uint32_t word = 0;
      std::memcpy(&word, &got[4 * i], 4);
      if (!inside) {
        if (word != 0) return -1;
        continue;
      }
      if ((word >> 24) != 0xFF) return -1;
      std::uint32_t want = 0;
      if (layout == sensor::YuvLayout::Luma) {
        const std::uint32_t g =
            static_cast<std::uint32_t>(luma_code(luma, full));
        want = g | (g << 8) | (g << 16);
      } else {
        // Chroma sited left: its first sample at luma's (0, 0.5), and on
        // every row for YUYV.
        const float cx = (sx - 0.0f) * 0.5f;
        const float cy =
            layout == sensor::YuvLayout::Yuyv ? sy : (sy - 0.5f) * 0.5f;
        const float u_c = bilinear(cb, pcw, pcw, pch, cx, cy);
        const float v_c = bilinear(cr, pcw, pcw, pch, cx, cy);
        const std::array<int, 3> rgb = yuv_reference::rgb(
            luma, u_c, v_c, sensor::VideoColorMatrix::Bt601, full);
        want = static_cast<std::uint32_t>(rgb[0]) |
               (static_cast<std::uint32_t>(rgb[1]) << 8) |
               (static_cast<std::uint32_t>(rgb[2]) << 16);
      }
      worst = std::max(worst, channel_diff(word, want));
      ++*compared;
    }
  }
  return worst;
}

// A blob's centroid in a luma view, around (u, v), over the background; false
// where its window leaves the view or meets an uncovered pixel.
bool centroid(const std::vector<std::uint8_t>& view, std::uint32_t w,
              std::uint32_t h, double u, double v, double radius, Vec2d* at) {
  const int r = static_cast<int>(std::ceil(radius));
  const int cu = static_cast<int>(std::lround(u));
  const int cv = static_cast<int>(std::lround(v));
  if (cu - r < 0 || cv - r < 0 || cu + r >= static_cast<int>(w) ||
      cv + r >= static_cast<int>(h)) {
    return false;
  }
  double sum = 0, sx = 0, sy = 0;
  for (int y = cv - r; y <= cv + r; ++y) {
    for (int x = cu - r; x <= cu + r; ++x) {
      const double code = view[std::size_t(y) * w + x];
      if (code == 0) return false;
      const double weight = std::max(0.0, code - Scene::kBackground);
      sum += weight;
      sx += weight * x;
      sy += weight * y;
    }
  }
  if (sum <= 0) return false;
  *at = Vec2d(sx / sum, sy / sum);
  return true;
}

// The pair the scene and the timing use: the pinned mild pair, at a RealSense
// IR pair's 1280x720, to FFS's 960x544.
sensor::StereoInfo mild_pair() {
  const OpenCvCase& c = kOpenCvCases[0];
  return info_of(c.left, c.right, c.rvec, c.tvec);
}

// Blobs on a plane come out on one row in both views, where the rig projects
// them, and each view is the host reference's remap of its picture.
int test_rows_align() {
  const sensor::StereoInfo info = mild_pair();
  const auto rectifier =
      make(info, {{960, 544}, 0.0f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr);
  const sensor::StereoRig& rig = rectifier->rig();
  const Scene scene;
  const Mat4d right_to_left =
      camera::rigid_inverse(info.calibration.left_to_right);
  const Picture left = luma_picture(scene, info.calibration.left, Mat4d(1.0));
  const Picture right =
      luma_picture(scene, info.calibration.right, right_to_left);
  auto pair = rectifier->rectify(frame_of(left, right));
  if (!pair) std::fprintf(stderr, "%s\n", pair.status().message().c_str());
  CHECK(pair.ok());
  CHECK(pair->source.timestamp_ns == 4321 && pair->source.sequence == 7);
  const std::size_t bytes = view_bytes(rig, Rectifier::Output::Luma8);
  const std::vector<std::uint8_t> views[2] = {
      read_view(*pair, *pair->left, bytes),
      read_view(*pair, *pair->right, bytes)};
  CHECK(views[0].size() == bytes && views[1].size() == bytes);

  // Each view is the remap the rig describes, within a code.
  for (int v = 0; v < 2; ++v) {
    std::size_t compared = 0;
    const int worst = compare_view(views[v], Remap(rig, info.calibration, v),
                                   v == 0 ? left : right,
                                   Rectifier::Output::Luma8, &compared);
    std::printf("  %s view against the host remap: %zu pixels, worst %d\n",
                v == 0 ? "left" : "right", compared, worst);
    CHECK(compared > bytes / 2);
    CHECK(worst >= 0 && worst <= 1);
  }

  // Each blob's centroid in both views: on one row, where the rig puts it.
  const Mat3d r1 = source_to_rect(rig, info.calibration, 0);
  const Mat3d r2 = source_to_rect(rig, info.calibration, 1);
  const Mat3d rel(info.calibration.left_to_right);
  const Vec3d t(info.calibration.left_to_right[3]);
  const camera::PinholeIntrinsics& k = rig.camera.intrinsics;
  double worst_row = 0, worst_place = 0, sum_row2 = 0;
  int measured = 0;
  for (const Vec3d& b : scene.blobs()) {
    const Vec3d in[2] = {r1 * b, r2 * (rel * b + t)};
    Vec2d at[2];
    bool seen = true;
    for (int v = 0; v < 2 && seen; ++v) {
      const Vec2d want(k.fx * in[v].x / in[v].z + k.cx,
                       k.fy * in[v].y / in[v].z + k.cy);
      const double radius = 4 * scene.sigma * k.fx / in[v].z;
      seen = centroid(views[v], rig.camera.size.width, rig.camera.size.height,
                      want.x, want.y, radius, &at[v]);
      if (seen) worst_place = std::max(worst_place, glm::length(at[v] - want));
    }
    if (!seen) continue;
    const double dy = at[0].y - at[1].y;
    worst_row = std::max(worst_row, std::fabs(dy));
    sum_row2 += dy * dy;
    ++measured;
  }
  std::printf(
      "  %d blobs in both views: rows apart by %.4f px at worst, %.4f px "
      "rms; %.4f px at worst from where the rig projects them\n",
      measured, worst_row, std::sqrt(sum_row2 / std::max(measured, 1)),
      worst_place);
  CHECK(measured > 200);
  CHECK(worst_row < 0.05);
  CHECK(worst_place < 0.1);
  return 0;
}

// YUYV, NV12 and I420 from the host, and luma in either range, as R'G'B'
// words and as luma bytes match the host reference within a code, the
// coverage byte exact; the same planes in device buffers give the same
// words, bit for bit.
int test_layouts() {
  const sensor::StereoInfo info = mild_pair();
  const Scene scene;
  const Mat4d right_to_left =
      camera::rigid_inverse(info.calibration.left_to_right);
  // A smaller rectified size than the pair's, so both sides of the remap
  // are exercised: resampled and undistorted.
  const auto rectifier =
      make(info, {{640, 360}, 0.25f, Rectifier::Output::Rgba8});
  const auto luma8 = make(info, {{640, 360}, 0.25f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr && luma8 != nullptr);
  const sensor::StereoRig& rig = rectifier->rig();
  const std::size_t bytes = view_bytes(rig, Rectifier::Output::Rgba8);
  const std::size_t luma_bytes = view_bytes(rig, Rectifier::Output::Luma8);
  struct Form {
    sensor::YuvLayout layout;
    bool full_range;
  };
  for (const Form form : {Form{sensor::YuvLayout::Yuyv, true},
                          Form{sensor::YuvLayout::Nv12, true},
                          Form{sensor::YuvLayout::I420, true},
                          Form{sensor::YuvLayout::Luma, true},
                          Form{sensor::YuvLayout::Luma, false}}) {
    const sensor::YuvLayout layout = form.layout;
    Picture left =
        layout == sensor::YuvLayout::Luma
            ? luma_picture(scene, info.calibration.left, Mat4d(1.0))
            : color_picture(scene, info.calibration.left, Mat4d(1.0), layout);
    Picture right =
        layout == sensor::YuvLayout::Luma
            ? luma_picture(scene, info.calibration.right, right_to_left)
            : color_picture(scene, info.calibration.right, right_to_left,
                            layout);
    left.image.full_range = right.image.full_range = form.full_range;
    std::vector<std::uint8_t> host_views[2];
    std::vector<std::uint8_t> luma_views[2];
    {
      auto pair = rectifier->rectify(frame_of(left, right));
      if (!pair) std::fprintf(stderr, "%s\n", pair.status().message().c_str());
      CHECK(pair.ok());
      host_views[0] = read_view(*pair, *pair->left, bytes);
      host_views[1] = read_view(*pair, *pair->right, bytes);
      auto luma_pair = luma8->rectify(frame_of(left, right));
      CHECK(luma_pair.ok());
      luma_views[0] = read_view(*luma_pair, *luma_pair->left, luma_bytes);
      luma_views[1] = read_view(*luma_pair, *luma_pair->right, luma_bytes);
    }
    for (int v = 0; v < 2; ++v) {
      const Remap remap(rig, info.calibration, v);
      std::size_t compared = 0, luma_compared = 0;
      const int worst =
          compare_view(host_views[v], remap, v == 0 ? left : right,
                       Rectifier::Output::Rgba8, &compared);
      const int luma_worst =
          compare_view(luma_views[v], remap, v == 0 ? left : right,
                       Rectifier::Output::Luma8, &luma_compared);
      std::printf(
          "  layout %d, %s range, %s view: as R'G'B' %zu pixels, worst %d; "
          "as luma %zu pixels, worst %d\n",
          static_cast<int>(layout), form.full_range ? "full" : "limited",
          v == 0 ? "left" : "right", compared, worst, luma_compared,
          luma_worst);
      CHECK(compared > bytes / 8);
      CHECK(worst >= 0 && worst <= 1);
      CHECK(luma_compared > luma_bytes / 2);
      CHECK(luma_worst >= 0 && luma_worst <= 1);
    }
    // The same planes in device buffers, from an offset the binding rounds.
    sensor::StereoFrame dev = frame_of(left, right);
    dev.left = on_device(left, 300);
    dev.right = on_device(right, 8);
    CHECK(dev.left.device != nullptr && dev.right.device != nullptr);
    auto from_device = rectifier->rectify(dev);
    if (!from_device) {
      std::fprintf(stderr, "%s\n", from_device.status().message().c_str());
    }
    CHECK(from_device.ok());
    CHECK(from_device->source.left.queue_family == sensor::kQueueFamilyIgnored);
    CHECK(read_view(*from_device, *from_device->left, bytes) == host_views[0]);
    CHECK(read_view(*from_device, *from_device->right, bytes) == host_views[1]);
  }
  return 0;
}

// A pair rectified already -- no rotation, no lens, one camera, the right
// one along +x -- is resampled only: at its own size its luma comes back byte
// for byte; at another its camera is scaled by one factor about the centres,
// so its pixels stay square: at alpha 0 the larger factor, the picture
// cropped to fill the view; at alpha 1 the smaller, all of it in the view.
int test_already_rectified() {
  const CameraModel ir{{1280, 720}, {657.25, 657.25, 651.375, 359.0}, {}};
  sensor::StereoInfo info;
  info.calibration.left = info.calibration.right = ir;
  info.calibration.left_to_right[3] = glm::dvec4(-0.095, 0.0, 0.0, 1.0);
  const auto same = make(info, {{}, 0.0f, Rectifier::Output::Luma8});
  CHECK(same != nullptr);
  const sensor::StereoRig& rig = same->rig();
  CHECK(rig.camera.size.width == 1280 && rig.camera.size.height == 720);
  CHECK(rig.camera.intrinsics.fx == ir.intrinsics.fx &&
        rig.camera.intrinsics.fy == ir.intrinsics.fy &&
        rig.camera.intrinsics.cx == ir.intrinsics.cx &&
        rig.camera.intrinsics.cy == ir.intrinsics.cy);
  CHECK(rig.rectified_to_left == Mat4d(1.0));
  CHECK(rig.baseline == 0.095);

  // Random luma, rows padded: the view is the picture, rows packed.
  Picture pics[2];
  std::uint32_t x = 0x2545F491u;
  for (Picture& p : pics) {
    p.w = 1280;
    p.h = 720;
    p.y.resize(std::size_t{p.w} * p.h);
    for (std::uint8_t& b : p.y) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      b = static_cast<std::uint8_t>(x >> 24);
    }
    const std::size_t stride = p.w + kPad;
    p.bytes.assign(stride * p.h, 0xCD);
    for (std::uint32_t r = 0; r < p.h; ++r) {
      std::memcpy(&p.bytes[r * stride], &p.y[std::size_t{r} * p.w], p.w);
    }
    p.image.layout = sensor::YuvLayout::Luma;
    p.image.host = p.bytes.data();
    p.image.stride[0] = stride;
    p.image.width = p.w;
    p.image.height = p.h;
  }
  auto pair = same->rectify(frame_of(pics[0], pics[1]));
  CHECK(pair.ok());
  CHECK(read_view(*pair, *pair->left, pics[0].y.size()) == pics[0].y);
  CHECK(read_view(*pair, *pair->right, pics[1].y.size()) == pics[1].y);

  // To FFS's size, 4:3 against 16:9 nearly: 960 / 1280 across, 544 / 720
  // down. One factor, the picture's centre on the view's.
  for (const float alpha : {0.0f, 1.0f}) {
    const auto scaled =
        make(info, {{960, 544}, alpha, Rectifier::Output::Luma8});
    CHECK(scaled != nullptr);
    const camera::PinholeIntrinsics& k = scaled->rig().camera.intrinsics;
    const double s = alpha == 0.0f ? 544.0 / 720 : 960.0 / 1280;
    CHECK(k.fx == k.fy);
    CHECK(std::fabs(k.fx - 657.25 * s) < 1e-12);
    CHECK(std::fabs(k.cx - (479.5 + (651.375 - 639.5) * s)) < 1e-12);
    CHECK(std::fabs(k.cy - (271.5 + (359.0 - 359.5) * s)) < 1e-12);
    CHECK(scaled->rig().rectified_to_left == Mat4d(1.0));
    auto resampled = scaled->rectify(frame_of(pics[0], pics[1]));
    CHECK(resampled.ok());
    const std::size_t bytes =
        view_bytes(scaled->rig(), Rectifier::Output::Luma8);
    const Remap remap(scaled->rig(), info.calibration, 0);
    std::size_t compared = 0;
    const int worst =
        compare_view(read_view(*resampled, *resampled->left, bytes), remap,
                     pics[0], Rectifier::Output::Luma8, &compared);
    // Cropped, every view pixel shows the picture; fitted, a band at the
    // top and the bottom does not, and every picture column shows.
    std::size_t covered = 0;
    float first_x = 1e9f, last_x = -1e9f;
    for (std::uint32_t v = 0; v < 544; ++v) {
      for (std::uint32_t u = 0; u < 960; ++u) {
        float sx = 0, sy = 0, r = 0;
        if (!remap.source(u, v, &sx, &sy, &r)) continue;
        ++covered;
        first_x = std::min(first_x, sx);
        last_x = std::max(last_x, sx);
      }
    }
    std::printf(
        "  rectified already, to 960x544 at alpha %.0f: fx %.4f, fy %.4f; "
        "%zu of %u pixels covered, columns %.2f to %.2f of 1280; worst %d\n",
        alpha, k.fx, k.fy, covered, 960u * 544u, first_x, last_x, worst);
    CHECK(compared == 960u * 544u);
    CHECK(worst >= 0 && worst <= 1);
    if (alpha == 0.0f) {
      CHECK(covered == 960u * 544u);
      CHECK(first_x > 4.0f && last_x < 1275.0f);  // cropped across
    } else {
      CHECK(covered == 960u * 540u);  // two rows bordered above and below
      CHECK(first_x < 0.5f && last_x > 1278.5f);  // every column, edge to edge
    }
  }
  return 0;
}

// The depth view a matcher's depth is prepared with: the rectified left
// camera; the colour camera, the left one or a third; and between them. What
// the source is -- id, model, clock, rate, colour exposure -- is its own.
int test_depth_info() {
  sensor::StereoInfo info = mild_pair();
  info.id = "123456789012";
  info.model = "Intel RealSense D455";
  info.clock = sensor::ClockDomain::Host;
  info.fps = 30;
  info.color_controls = camera::ColorControls{167, 60};
  const auto own = make(info, {{960, 544}, 0.0f, Rectifier::Output::Luma8});
  CHECK(own != nullptr);
  const sensor::SensorInfo& d = own->depth_info();
  CHECK(d.depth && d.color);
  CHECK(d.depth->intrinsics.fx == own->rig().camera.intrinsics.fx &&
        d.depth->size.width == 960);
  CHECK(d.color->intrinsics.fx == info.calibration.left.intrinsics.fx &&
        d.color->distortion.k1 == info.calibration.left.distortion.k1);
  CHECK(d.depth_to_color == own->rig().rectified_to_left);
  CHECK(d.id == info.id && d.model == info.model);
  CHECK(d.clock == sensor::ClockDomain::Host && d.fps == 30);
  CHECK(d.color_controls && *d.color_controls == *info.color_controls);
  CHECK(d.role == sensor::SyncRole::FreeRun &&
        d.pose == sensor::PoseSource::Fixed);

  info.color = CameraModel{{1280, 800}, {640.5, 640.25, 642.0, 401.5}, {}};
  info.left_to_color = Mat4d(1.0);
  info.left_to_color[3] = glm::dvec4(-0.059, 0.0001, 0.0004, 1.0);
  const auto third = make(info, {{960, 544}, 0.0f, Rectifier::Output::Luma8});
  CHECK(third != nullptr);
  const sensor::SensorInfo& e = third->depth_info();
  CHECK(e.color && e.color->size.height == 800);
  CHECK(e.depth_to_color ==
        info.left_to_color * third->rig().rectified_to_left);
  return 0;
}

// Pairs used in turn -- rectify, read, drop -- have the same two buffers,
// with nothing allocated, whether a Vulkan reader hands the views back or
// keeps them (the next pass takes them without a transfer). A pair kept past
// the next keeps its views, which the next does not write, and the next has
// new ones, their descriptors open where exported; a view no pair holds is
// written again. Exported where the device exports.
int test_pairs_hold_views() {
  const sensor::StereoInfo info = mild_pair();
  const auto rectifier =
      make(info, {{480, 272}, 0.0f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr);
  const Scene scene;
  const Picture left = luma_picture(scene, info.calibration.left, Mat4d(1.0));
  const Picture right =
      luma_picture(scene, info.calibration.right,
                   camera::rigid_inverse(info.calibration.left_to_right));
  Picture dark = left;
  for (std::uint8_t& b : dark.bytes) b = static_cast<std::uint8_t>(b / 2);
  dark.image.host = dark.bytes.data();
  const std::size_t bytes =
      view_bytes(rectifier->rig(), Rectifier::Output::Luma8);

  // In turn, the left picture bright and dark by turns, the views read and
  // kept or handed back by turns.
  {
    std::vector<std::uint8_t> want[2];
    VkBuffer handles[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::uint64_t staged = 0;  // staging buffers made after the first pair
    for (int i = 0; i < 6; ++i) {
      const std::uint64_t was = g_device->counters().staging_buffers;
      auto pair = rectifier->rectify(frame_of(i % 2 == 0 ? left : dark, right));
      CHECK(pair.ok());
      if (i > 0) staged += g_device->counters().staging_buffers - was;
      const std::vector<std::uint8_t> got =
          read_view(*pair, *pair->left, bytes, i % 4 < 2);
      CHECK(read_view(*pair, *pair->right, bytes, i % 4 < 2).size() == bytes);
      if (i < 2) want[i] = got;
      CHECK(got == want[i % 2]);
      if (i == 0) {
        handles[0] = pair->left->buffer.handle();
        handles[1] = pair->right->buffer.handle();
      }
      CHECK(pair->left->buffer.handle() == handles[0] &&
            pair->right->buffer.handle() == handles[1]);
    }
    CHECK(want[0] != want[1]);
    CHECK(staged == 0);
  }

  std::vector<std::uint8_t> a_left;
  const vkc::ExportedBuffer* b_left = nullptr;
  {
    auto a = rectifier->rectify(frame_of(left, right));
    CHECK(a.ok());
    a_left = read_view(*a, *a->left, bytes);
    if (g_device->exports_memory()) {
      CHECK(a->queue_family == sensor::kQueueFamilyExternal);
      CHECK(a->left->fd.valid() && a->right->fd.valid());
      CHECK(a->left->memory_size >= a->left->buffer.size());
    } else {
      CHECK(a->queue_family == sensor::kQueueFamilyIgnored);
    }
    std::printf("  views %s\n", g_device->exports_memory()
                                    ? "exported, from VK_QUEUE_FAMILY_EXTERNAL"
                                    : "not exported (the device exports none)");
    auto b = rectifier->rectify(frame_of(dark, right));
    CHECK(b.ok());
    CHECK(a->left.get() != b->left.get() && a->right.get() != b->right.get());
    // New buffers: an importer knows them by their open descriptors.
    if (g_device->exports_memory()) {
      CHECK(b->left->fd.valid() && b->right->fd.valid());
    }
    CHECK(read_view(*a, *a->left, bytes) == a_left);
    CHECK(read_view(*b, *b->left, bytes) != a_left);
    b_left = b->left.get();
  }
  auto c = rectifier->rectify(frame_of(left, right));
  CHECK(c.ok());
  CHECK(c->left.get() == b_left);
  CHECK(read_view(*c, *c->left, bytes) == a_left);
  return 0;
}

// A smooth luma picture of `c`'s size that never reads code 0, so a covered
// luma pixel is never 0.
Picture smooth_picture(const CameraModel& c, double phase) {
  Picture p;
  p.w = c.size.width;
  p.h = c.size.height;
  p.y.resize(std::size_t{p.w} * p.h);
  for (std::uint32_t r = 0; r < p.h; ++r) {
    for (std::uint32_t x = 0; x < p.w; ++x) {
      p.y[std::size_t{r} * p.w + x] = to_code(
          128 + 100 * std::sin(0.05 * x + phase) * std::cos(0.04 * r - phase));
    }
  }
  p.bytes = p.y;
  p.image.layout = sensor::YuvLayout::Luma;
  p.image.host = p.bytes.data();
  p.image.stride[0] = p.w;
  p.image.width = p.w;
  p.image.height = p.h;
  return p;
}

// The strong barrel pair OpenCV's numbers pin, 800x600 to 640x480. Its
// lenses fold -- r * radial(r^2) stops growing, at r = 2.05 for the left
// one, which images no ray past r = 1.14, and 2.20 for the right -- and at
// alpha 1 a tenth of the left view's rays and a twentieth of the right's lie
// past the fold, where the lens model turns back and would land them inside
// the picture, mirrored (32,417 of the left view's 307,200, by the review's
// host replica in double). The pass writes each 0, coverage included,
// as the host reference cut at the replica's fold has it, and the rest within
// a code of it, as luma bytes and as R'G'B' words; at alpha 0 no ray reaches
// the fold.
int test_fold() {
  const OpenCvCase& c = kOpenCvCases[3];
  const sensor::StereoInfo info = info_of(c.left, c.right, c.rvec, c.tvec);
  const Picture pics[2] = {smooth_picture(c.left, 0.3),
                           smooth_picture(c.right, 1.1)};
  for (const float alpha : {0.0f, 1.0f}) {
    for (const Rectifier::Output output :
         {Rectifier::Output::Rgba8, Rectifier::Output::Luma8}) {
      const bool luma = output == Rectifier::Output::Luma8;
      const auto rectifier = make(info, {{640, 480}, alpha, output});
      CHECK(rectifier != nullptr);
      const sensor::StereoRig& rig = rectifier->rig();
      auto pair = rectifier->rectify(frame_of(pics[0], pics[1]));
      CHECK(pair.ok());
      const std::size_t bytes = view_bytes(rig, output);
      for (int v = 0; v < 2; ++v) {
        const Remap remap(rig, info.calibration, v);
        const std::vector<std::uint8_t> view =
            read_view(*pair, v == 0 ? *pair->left : *pair->right, bytes);
        std::size_t compared = 0;
        const int worst = compare_view(view, remap, pics[v], output, &compared);
        // The rays past the fold that the lens model lands inside.
        std::size_t past = 0, written = 0;
        for (std::uint32_t y = 0; y < remap.h; ++y) {
          for (std::uint32_t x = 0; x < remap.w; ++x) {
            float sx = 0, sy = 0, r = 0;
            if (!remap.lands(x, y, &sx, &sy, &r) || !(r > remap.fold) ||
                remap.near_fold(r)) {
              continue;
            }
            ++past;
            const std::size_t i = std::size_t{y} * remap.w + x;
            std::uint32_t value = view[i];
            if (!luma) std::memcpy(&value, &view[4 * i], 4);
            if (value != 0) ++written;
          }
        }
        const std::size_t pixels = std::size_t{remap.w} * remap.h;
        std::printf(
            "  strong barrel lenses, alpha %.0f, %s view as %s: folds at r = "
            "%.4f; %zu rays past it land inside, %zu written; %zu pixels, "
            "worst %d\n",
            alpha, v == 0 ? "left" : "right", luma ? "luma" : "R'G'B'",
            remap.fold, past, written, compared, worst);
        CHECK(worst >= 0 && worst <= 1);
        CHECK(written == 0);
        CHECK(compared > pixels / 4);
        if (alpha == 0.0f) {
          CHECK(past == 0);
        } else {
          CHECK(past > (v == 0 ? pixels / 10 : pixels / 20));
        }
      }
    }
  }
  return 0;
}

// A side-by-side frame's one plane, as a ZED's picture holds it:
// row r is `l`'s row r, then `r`'s, then kPad bytes; `*stride` its bytes.
std::vector<std::uint8_t> side_by_side_rows(const Picture& l, const Picture& r,
                                            std::size_t* stride) {
  *stride = 2 * std::size_t{l.w} + kPad;
  std::vector<std::uint8_t> rows(*stride * l.h, 0xCD);
  for (std::uint32_t y = 0; y < l.h; ++y) {
    std::memcpy(&rows[y * *stride], &l.y[std::size_t{y} * l.w], l.w);
    std::memcpy(&rows[y * *stride + l.w], &r.y[std::size_t{y} * r.w], r.w);
  }
  return rows;
}

// The frame those rows in `buffer` make, every picture naming `family`: the
// left and right luma pictures, and a colour picture in the buffer too.
sensor::StereoFrame side_by_side_frame(
    const Picture& l, const Picture& r,
    const std::shared_ptr<const vkc::Buffer>& buffer, std::size_t stride,
    std::uint32_t family) {
  sensor::StereoFrame f = frame_of(l, r);
  for (sensor::YuvImage* image : {&f.left, &f.right, &f.color}) {
    *image = l.image;
    image->host = nullptr;
    image->device = buffer;
    image->offset[0] = 0;
    image->stride[0] = stride;
    image->queue_family = family;
  }
  f.right.offset[0] = l.w;
  return f;
}

// Pictures in device buffers are taken over from the family their writer
// released them to, each buffer once (DeviceCounters::barriers, a barrier a
// transfer), and give the host pictures' views:
//   - each picture in a buffer of its own, from VK_QUEUE_FAMILY_EXTERNAL, as
//     CUDA's writes come: two transfers more than from the pass's own family;
//   - a side-by-side frame's one buffer, released once, a colour picture in
//     it too: one transfer more, and RectifiedPair::source's pictures read
//     the pass's family, while a colour picture in a buffer of its own keeps
//     its writer's;
//   - the same buffer CONCURRENT with another of the device's families,
//     which wrote it: none;
//   - pictures in one buffer that name two families, the colour one's
//     included, are refused before any work.
int test_taken_over() {
  const sensor::StereoInfo info = mild_pair();
  const auto rectifier =
      make(info, {{480, 272}, 0.0f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr);
  const Scene scene;
  const Picture left = luma_picture(scene, info.calibration.left, Mat4d(1.0));
  const Picture right =
      luma_picture(scene, info.calibration.right,
                   camera::rigid_inverse(info.calibration.left_to_right));
  const std::size_t bytes =
      view_bytes(rectifier->rig(), Rectifier::Output::Luma8);
  std::vector<std::uint8_t> want[2];
  {
    auto pair = rectifier->rectify(frame_of(left, right));
    CHECK(pair.ok());
    want[0] = read_view(*pair, *pair->left, bytes);
    want[1] = read_view(*pair, *pair->right, bytes);
  }
  // `f` rectified: whether its views are the host pictures', the barriers
  // its batch recorded, and its source.
  std::uint64_t barriers = 0;
  sensor::StereoFrame source;
  const auto same = [&](const sensor::StereoFrame& f) {
    const std::uint64_t was = g_device->counters().barriers;
    auto pair = rectifier->rectify(f);
    if (!pair) {
      std::fprintf(stderr, "%s\n", pair.status().message().c_str());
      return false;
    }
    barriers = g_device->counters().barriers - was;
    source = pair->source;
    return read_view(*pair, *pair->left, bytes) == want[0] &&
           read_view(*pair, *pair->right, bytes) == want[1];
  };

  sensor::StereoFrame apart = frame_of(left, right);
  apart.left = on_device(left, 0);
  apart.right = on_device(right, 4);
  CHECK(same(apart));
  const std::uint64_t apart_own = barriers;
  apart.left = on_device(left, 0, sensor::kQueueFamilyExternal);
  apart.right = on_device(right, 4, sensor::kQueueFamilyExternal);
  CHECK(same(apart));
  CHECK(barriers == apart_own + 2);
  CHECK(source.left.queue_family == sensor::kQueueFamilyIgnored &&
        source.right.queue_family == sensor::kQueueFamilyIgnored);

  std::size_t stride = 0;
  const std::vector<std::uint8_t> rows =
      side_by_side_rows(left, right, &stride);
  const auto side_by_side = [&](std::uint32_t family) {
    return side_by_side_frame(left, right, device_bytes(rows, family), stride,
                              family);
  };
  CHECK(same(side_by_side(sensor::kQueueFamilyIgnored)));
  const std::uint64_t together_own = barriers;
  const sensor::StereoFrame together =
      side_by_side(sensor::kQueueFamilyExternal);
  CHECK(same(together));
  CHECK(barriers == together_own + 1);
  CHECK(source.left.queue_family == sensor::kQueueFamilyIgnored &&
        source.right.queue_family == sensor::kQueueFamilyIgnored &&
        source.color.queue_family == sensor::kQueueFamilyIgnored);
  CHECK(together.left.queue_family == sensor::kQueueFamilyExternal);
  sensor::StereoFrame own_color = side_by_side(sensor::kQueueFamilyExternal);
  own_color.color = on_device(left, 0, sensor::kQueueFamilyExternal);
  CHECK(same(own_color));
  CHECK(barriers == together_own + 1);
  CHECK(source.color.queue_family == sensor::kQueueFamilyExternal);
  // Written on another of the device's families into a buffer CONCURRENT
  // with the pass's: shared already, so taken with no transfer.
  const std::uint32_t own = g_device->queue_family();
  if (g_device->caps().queue_families().size() > 1) {
    const std::uint32_t shared[2] = {own, own == 0 ? 1u : 0u};
    std::vector<std::uint8_t> words = rows;
    words.resize((words.size() + 3) & ~std::size_t{3}, 0xEE);
    auto made =
        vkc::device_storage_buffer(*g_allocator, words.size(), 0, shared, 2);
    CHECK(made.ok());
    vkc::CommandBatch batch(*g_device, *g_allocator);
    CHECK(batch.upload(made.value(), 0, words.data(), words.size()).ok() &&
          batch.submit().ok());
    const auto concurrent =
        std::make_shared<const vkc::Buffer>(std::move(made).value());
    CHECK(same(side_by_side_frame(left, right, concurrent, stride, shared[1])));
    CHECK(barriers == together_own);
    CHECK(source.left.queue_family == sensor::kQueueFamilyIgnored);
  }

  const auto refused = [&](const sensor::StereoFrame& f, const char* why) {
    const std::uint64_t submits = g_device->counters().submits;
    const vkc::Status s = rectifier->rectify(f).status();
    if (s.domain() != vkc::Status::Code::InvalidArgument ||
        s.message().find(why) == std::string::npos ||
        g_device->counters().submits != submits) {
      std::fprintf(stderr, "rectify, expected \"%s\": %s\n", why,
                   s.message().c_str());
      return false;
    }
    return true;
  };
  sensor::StereoFrame split = side_by_side(sensor::kQueueFamilyExternal);
  split.right.queue_family = sensor::kQueueFamilyIgnored;
  CHECK(refused(split, "in one buffer, taken over once"));
  split = side_by_side(sensor::kQueueFamilyExternal);
  split.color.queue_family = sensor::kQueueFamilyIgnored;
  CHECK(refused(split, "colour picture shares a buffer"));
  CHECK(same(side_by_side(sensor::kQueueFamilyExternal)));
  std::printf(
      "  device pictures taken over: apart from VK_QUEUE_FAMILY_EXTERNAL "
      "%llu barriers (%llu from the pass's family), side by side %llu "
      "(%llu)\n",
      static_cast<unsigned long long>(apart_own + 2),
      static_cast<unsigned long long>(apart_own),
      static_cast<unsigned long long>(together_own + 1),
      static_cast<unsigned long long>(together_own));
  return 0;
}

// What create and rectify refuse, before any work.
int test_refusals() {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  const sensor::StereoInfo good = mild_pair();
  const Rectifier::Options options{{480, 272}, 0.0f, Rectifier::Output::Luma8};
  const auto refused = [&](const sensor::StereoInfo& info,
                           const Rectifier::Options& o, const char* why) {
    auto made = Rectifier::create(*g_device, *g_allocator, info, o);
    if (made.ok() || made.status().domain() != invalid ||
        made.status().message().find(why) == std::string::npos) {
      std::fprintf(stderr, "create, expected \"%s\": %s\n", why,
                   made.ok() ? "made" : made.status().message().c_str());
      return false;
    }
    return true;
  };
  sensor::StereoInfo bad = good;
  bad.calibration.left.intrinsics.fx = 0;
  CHECK(refused(bad, options, "the left camera"));
  bad = good;
  bad.calibration.right.distortion.k2 = std::nan("");
  CHECK(refused(bad, options, "the right camera"));
  bad = good;
  bad.calibration.right.size = {1280, 800};
  CHECK(refused(bad, options, "sizes differ"));
  bad = good;
  bad.calibration.left_to_right[0][0] = 2.0;
  CHECK(refused(bad, options, "left_to_right"));
  bad = good;
  bad.color = CameraModel{};
  CHECK(refused(bad, options, "the colour camera"));
  bad = good;
  bad.left_to_color[1][1] = -1.0;
  CHECK(refused(bad, options, "left_to_color"));
  bad = good;
  bad.calibration.left_to_right = Mat4d(1.0);
  CHECK(refused(bad, options, "at one place"));
  bad.calibration.left_to_right[3] = glm::dvec4(0.0, -0.1, 0.0, 1.0);
  CHECK(refused(bad, options, "vertical"));
  bad.calibration.left_to_right[3] = glm::dvec4(0.095, 0.0, 0.001, 1.0);
  CHECK(refused(bad, options, "-x side"));
  for (const float alpha : {-0.1f, 1.5f, std::nanf("")}) {
    CHECK(
        refused(good, {{480, 272}, alpha, Rectifier::Output::Luma8}, "alpha"));
  }
  CHECK(
      refused(good, {{480, 0}, 0.0f, Rectifier::Output::Luma8}, "one side 0"));
  CHECK(refused(good, {{65536, 65536}, 0.0f, Rectifier::Output::Rgba8},
                "single dispatch"));

  const auto rectifier = make(good, options);
  CHECK(rectifier != nullptr);
  const Scene scene;
  const Picture left = luma_picture(scene, good.calibration.left, Mat4d(1.0));
  const Picture right =
      luma_picture(scene, good.calibration.right,
                   camera::rigid_inverse(good.calibration.left_to_right));
  const auto frame_refused = [&](const sensor::StereoFrame& f,
                                 const char* why) {
    const vkc::Status s = rectifier->rectify(f).status();
    if (s.domain() != invalid || s.message().find(why) == std::string::npos) {
      std::fprintf(stderr, "rectify, expected \"%s\": %s\n", why,
                   s.message().c_str());
      return false;
    }
    return true;
  };
  sensor::StereoFrame f = frame_of(left, right);
  f.right = sensor::YuvImage{};
  CHECK(frame_refused(f, "the right picture has no planes"));
  f = frame_of(left, right);
  f.left.width = 640;
  CHECK(frame_refused(f, "the left picture is 640x720"));
  f = frame_of(left, right);
  f.left.layout = static_cast<sensor::YuvLayout>(4);
  CHECK(frame_refused(f, "unknown colour layout"));
  f = frame_of(left, right);
  f.left.stride[0] = 100;
  CHECK(frame_refused(f, "stride is shorter"));
  // A picture with chroma needs a known siting and a matrix; luma alone,
  // grey, needs neither, so a mono driver may leave them unset.
  const Picture yuyv_left = color_picture(scene, good.calibration.left,
                                          Mat4d(1.0), sensor::YuvLayout::Yuyv);
  const Picture yuyv_right =
      color_picture(scene, good.calibration.right,
                    camera::rigid_inverse(good.calibration.left_to_right),
                    sensor::YuvLayout::Yuyv);
  f = frame_of(yuyv_left, yuyv_right);
  f.left.chroma_location = static_cast<sensor::ChromaLocation>(200);
  CHECK(frame_refused(f, "unknown chroma location"));
  f = frame_of(yuyv_left, yuyv_right);
  f.left.kr = 0.9f;
  CHECK(frame_refused(f, "matrix's kr and kb"));
  f = frame_of(left, right);
  f.left.chroma_location = static_cast<sensor::ChromaLocation>(200);
  f.left.kr = f.left.kb = 0.0f;
  f.right.kr = f.right.kb = std::nanf("");
  CHECK(rectifier->rectify(f).ok());
  f = frame_of(left, right);
  f.right = on_device(right, 0);
  CHECK(f.right.device != nullptr);
  f.right.queue_family =
      static_cast<std::uint32_t>(g_device->caps().queue_families().size());
  CHECK(frame_refused(f, "which the device does not have"));
  f = frame_of(left, right);
  f.left.image[0] = std::make_shared<const vkc::Image>();
  CHECK(frame_refused(f, "are images"));
  // The rectifier still works after every refusal.
  CHECK(rectifier->rectify(frame_of(left, right)).ok());
  return 0;
}

// The pass at a RealSense IR pair's 1280x720 to FFS's 960x544, both views:
// device time from the batch's timestamps, host time around rectify.
int test_timing() {
#ifdef VKC_TEST_SANITIZED
  constexpr int kRuns = 3;
#else
  constexpr int kRuns = 200;
#endif
  const sensor::StereoInfo info = mild_pair();
  const Scene scene;
  const Mat4d right_to_left =
      camera::rigid_inverse(info.calibration.left_to_right);
  const Picture luma[2] = {
      luma_picture(scene, info.calibration.left, Mat4d(1.0)),
      luma_picture(scene, info.calibration.right, right_to_left)};
  const Picture yuyv[2] = {
      color_picture(scene, info.calibration.left, Mat4d(1.0),
                    sensor::YuvLayout::Yuyv),
      color_picture(scene, info.calibration.right, right_to_left,
                    sensor::YuvLayout::Yuyv)};
  struct Case {
    const char* name;
    const Picture* pictures;
    bool device;
    Rectifier::Output output;
  };
  const Case cases[] = {
      {"luma from the host -> Luma8", luma, false, Rectifier::Output::Luma8},
      {"luma in device buffers -> Luma8", luma, true, Rectifier::Output::Luma8},
      {"luma from the host -> Rgba8", luma, false, Rectifier::Output::Rgba8},
      {"YUYV from the host -> Rgba8", yuyv, false, Rectifier::Output::Rgba8},
  };
  for (const Case& c : cases) {
    const auto rectifier = make(info, {{960, 544}, 0.0f, c.output});
    CHECK(rectifier != nullptr);
    sensor::StereoFrame f = frame_of(c.pictures[0], c.pictures[1]);
    if (c.device) {
      f.left = on_device(c.pictures[0], 0);
      f.right = on_device(c.pictures[1], 0);
    }
    CHECK(rectifier->rectify(f).ok());  // warm
    vkc::StageMetrics metrics;
    for (int i = 0; i < kRuns; ++i) {
      CHECK(rectifier->rectify(f, &metrics).ok());
    }
    for (const vkc::StageRow& row : metrics.rows()) {
      if (std::strcmp(row.name, "stereo rectify") != 0) continue;
      std::printf(
          "  1280x720 -> 960x544, both views, %s: %.3f ms device, %.3f ms "
          "host (%.3f blocked) a pair, over %d\n",
          c.name, row.gpu_ms / kRuns, row.cpu_ms / kRuns, row.wait_ms / kRuns,
          kRuns);
    }
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  std::printf("sensor_stereo_rectifier on %s\n",
              gpu.physical.properties().deviceName);
  if (test_opencv() != 0) return 1;
  if (test_rows_align() != 0) return 1;
  if (test_layouts() != 0) return 1;
  if (test_already_rectified() != 0) return 1;
  if (test_depth_info() != 0) return 1;
  if (test_pairs_hold_views() != 0) return 1;
  if (test_fold() != 0) return 1;
  if (test_taken_over() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_timing() != 0) return 1;
  std::puts("sensor_stereo_rectifier: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
