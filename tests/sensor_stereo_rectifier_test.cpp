// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StereoRectifier on synthetic calibrated pairs, no camera (its
// rectification is rectify_stereo's, which camera_stereo_rectification_test
// pins to OpenCV's numbers):
//   - blobs on a plane, imaged through two lensed cameras, rotated and
//     offset, by an independent host renderer (each camera pixel's ray found
//     by inverting the lens iteratively, where the pass runs it forward),
//     come out on one row in both views, within 0.05 px, where the rig
//     projects them;
//   - each view matches a host reference of the pass's remap within a code,
//     its area prefilter included (each pixel's footprint from where the
//     points half a pixel either side of it land, 5 x 5 taps over it where
//     its longer side is over 1.5 source pixels): luma, YUYV, NV12 and I420,
//     as luma bytes and as R'G'B' words, the coverage byte exact, and
//     limited-range luma, at every third pixel each way (odd and even
//     columns and rows alike);
//   - the prefilter: a D455's 1.33:1 takes one tap, as the one-tap
//     reference has it byte for byte but at ties, half way between codes; at
//     4.5:1 a checker grating past the view's Nyquist reads flat within 2
//     levels, and step edges stay within 0.03 view pixels at every phase;
//     through a near-equidistant 4K lens, a side edge's 1.5 x 3 footprint
//     keeps rings that a centre-sized box would flatten (these two not
//     under AddressSanitizer, where they would run on 4K pictures the
//     rectify path layouts and fold already check);
//   - the same pictures in device buffers give the same views, bit for bit;
//   - V210 of a YUYV pair's codes times 4 gives YUYV's views bit for bit,
//     both outputs, from the host and from device buffers;
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
//   - a pair holds its frame's left picture but lets the right one go, its
//     own owner (StereoFrame::right_pixels) or its device buffer;
//   - what it refuses, at create and per frame.
// With --timing (sensor_stereo_rectifier_timing, labelled manual: run by
// hand, never in CI) it prints the pass's time instead, at 1280x720 to
// 960x544, and at 3840x2160 V210 to 960x544 through the 4K lens, every pixel
// averaged, both views.
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

#include "stereo_rectify_reference.hpp"
#include "v210.hpp"
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
using stereo_rectify_reference::kOpenCvCases;
using stereo_rectify_reference::OpenCvCase;
using Rectifier = sensor::StereoRectifier;

vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

// Under AddressSanitizer alone (the core's VKC_TEST_SANITIZED is
// ThreadSanitizer's too).
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define VS_TEST_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(VS_TEST_ASAN)
#define VS_TEST_ASAN 1
#endif
#ifdef VS_TEST_ASAN
constexpr bool kAddressSanitizer = true;
#else
constexpr bool kAddressSanitizer = false;
#endif

// --- The pinned pairs -------------------------------------------------------

// A pinned pair's source (stereo_rectify_reference.hpp), as a driver
// reports it.
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
    case sensor::YuvLayout::V210:  // as_v210's, from YUYV's
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

  // Where the lens model puts point (u, v) of the rectified view in the
  // source, and the ray's normalized radius there, `*r`; false where the ray
  // is behind the camera. The fold is not applied.
  bool lens_at(float u, float v, float* sx, float* sy, float* r) const {
    const float x = (u - cx) / fx;
    const float y = (v - cy) / fy;
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
    return true;
  }

  // Where the lens model puts rectified pixel (u, v) in the source, and the
  // ray's normalized radius there, `*r`; false where the ray is behind the
  // camera or lands outside the picture. The fold is not applied.
  bool lands(std::uint32_t u, std::uint32_t v, float* sx, float* sy,
             float* r) const {
    return lens_at(static_cast<float>(u), static_cast<float>(v), sx, sy, r) &&
           *sx >= -0.5f && *sy >= -0.5f && *sx <= sw - 0.5f && *sy <= sh - 0.5f;
  }

  // Rectified pixel (u, v)'s footprint in the source, the spec's, worked
  // out apart from the pass: where the points half a pixel either side of
  // it land, across (`du`) and down (`dv`). 1 where the pass averages the
  // pixel over it -- every neighbour's ray in front and inside the fold,
  // the longer side over 1.5 source pixels -- 0 where it takes one tap, -1
  // where the device and the host may decide either way (a side within
  // 1e-3 of 1.5, a neighbour's ray near the fold).
  int footprint(std::uint32_t u, std::uint32_t v, Vec2d* du, Vec2d* dv) const {
    const float fu = static_cast<float>(u), fv = static_cast<float>(v);
    const float at[4][2] = {
        {fu - 0.5f, fv}, {fu + 0.5f, fv}, {fu, fv - 0.5f}, {fu, fv + 0.5f}};
    Vec2d s[4];
    for (int i = 0; i < 4; ++i) {
      float sx = 0, sy = 0, r = 0;
      if (!lens_at(at[i][0], at[i][1], &sx, &sy, &r)) return 0;
      if (near_fold(r)) return -1;
      if (r > fold) return 0;
      s[i] = Vec2d(sx, sy);
    }
    *du = s[1] - s[0];
    *dv = s[3] - s[2];
    const double longer = std::max(glm::length(*du), glm::length(*dv));
    if (std::fabs(longer - 1.5) < 1e-3) return -1;
    return longer > 1.5 ? 1 : 0;
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

// The pass's taps a footprint side, the spec's: enough for a 4K picture's
// 4.5:1 shrink to FFS's 960 x 544.
constexpr int kTaps = 5;

// The mean of `sample` (a source point to its luma, Cb and Cr) over a
// footprint (`du`, `dv`) about `s`: `taps` x `taps` points, each at the
// centre of one of as many equal cells, weighted alike.
template <typename Sample>
std::array<float, 3> area_mean(const Sample& sample, Vec2d s, Vec2d du,
                               Vec2d dv, int taps) {
  std::array<double, 3> sum = {0, 0, 0};
  for (int j = 0; j < taps; ++j) {
    for (int i = 0; i < taps; ++i) {
      const double a = (i + 0.5) / taps - 0.5, b = (j + 0.5) / taps - 0.5;
      const Vec2d p = s + du * a + dv * b;
      const std::array<float, 3> got =
          sample(static_cast<float>(p.x), static_cast<float>(p.y));
      for (int c = 0; c < 3; ++c) sum[c] += got[c];
    }
  }
  const double n = static_cast<double>(taps) * taps;
  return {static_cast<float>(sum[0] / n), static_cast<float>(sum[1] / n),
          static_cast<float>(sum[2] / n)};
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
// range, at every `step`th pixel each way from (0, 0). Returns the worst
// code off, -1 for a coverage mismatch; `compared` counts pixels.
int compare_view(const std::vector<std::uint8_t>& got, const Remap& remap,
                 const Picture& picture, Rectifier::Output output,
                 std::size_t* compared, std::uint32_t step = 1) {
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
  // Luma, Cb and Cr at source point (x, y), each at its siting (chroma
  // sited left: its first sample at luma's (0, 0.5), and on every row for
  // YUYV).
  const auto sample = [&](float x, float y) {
    std::array<float, 3> out = {
        bilinear(picture.y, picture.w, picture.w, picture.h, x, y), 0.0f, 0.0f};
    if (layout != sensor::YuvLayout::Luma) {
      const float cx = x * 0.5f;
      const float cy =
          layout == sensor::YuvLayout::Yuyv ? y : (y - 0.5f) * 0.5f;
      out[1] = bilinear(cb, pcw, pcw, pch, cx, cy);
      out[2] = bilinear(cr, pcw, pcw, pch, cx, cy);
    }
    return out;
  };
  int worst = 0;
  *compared = 0;
  for (std::uint32_t v = 0; v < remap.h; v += step) {
    for (std::uint32_t u = 0; u < remap.w; u += step) {
      const std::size_t i = std::size_t{v} * remap.w + u;
      float sx = 0, sy = 0, r = 0;
      const bool inside = remap.source(u, v, &sx, &sy, &r);
      if (remap.near_edge(sx, sy) || remap.near_fold(r)) continue;
      Vec2d du, dv;
      const int area = inside ? remap.footprint(u, v, &du, &dv) : 0;
      if (area < 0) continue;
      const std::array<float, 3> ycc =
          !inside     ? std::array<float, 3>{}
          : area == 0 ? sample(sx, sy)
                      : area_mean(sample, Vec2d(sx, sy), du, dv, kTaps);
      const float luma = ycc[0];
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
        const std::array<int, 3> rgb = yuv_reference::rgb(
            luma, ycc[1], ycc[2], sensor::VideoColorMatrix::Bt601, full);
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
// coverage byte exact, at every third pixel each way: odd and even columns
// and rows alike, which a chroma pair's two luma columns and a 4:2:0 plane's
// two rows differ by, and the reference's 25 taps a pixel the case's cost;
// the same planes in device buffers give the same words, bit for bit, at
// every pixel.
int test_layouts() {
  constexpr std::uint32_t kStep = 3;
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
  // The pixels compare_view reaches at kStep.
  const std::size_t reached =
      std::size_t{(rig.camera.size.width + kStep - 1) / kStep} *
      ((rig.camera.size.height + kStep - 1) / kStep);
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
                       Rectifier::Output::Rgba8, &compared, kStep);
      const int luma_worst =
          compare_view(luma_views[v], remap, v == 0 ? left : right,
                       Rectifier::Output::Luma8, &luma_compared, kStep);
      std::printf(
          "  layout %d, %s range, %s view: as R'G'B' %zu pixels, worst %d; "
          "as luma %zu pixels, worst %d\n",
          static_cast<int>(layout), form.full_range ? "full" : "limited",
          v == 0 ? "left" : "right", compared, worst, luma_compared,
          luma_worst);
      CHECK(compared > reached / 2);
      CHECK(worst >= 0 && worst <= 1);
      CHECK(luma_compared > reached / 2);
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

// `p`, a YUYV picture, as V210 of its codes times 4: the same samples, rows
// padded, from byte 4.
Picture as_v210(const Picture& p) {
  const std::uint32_t cw = (p.w + 1) / 2;
  std::vector<std::uint8_t> cb(std::size_t{cw} * p.h), cr(cb.size());
  for (std::uint32_t r = 0; r < p.h; ++r) {
    for (std::uint32_t i = 0; i < cw; ++i) {
      const std::size_t from =
          std::size_t{r} * p.w + std::min(2 * i, p.w - 1);  // its site
      cb[std::size_t{r} * cw + i] = p.cb[from];
      cr[std::size_t{r} * cw + i] = p.cr[from];
    }
  }
  Picture out = p;
  const std::size_t lead = 4, stride = v210::row_bytes(p.w) + 64;
  out.bytes = v210::pack(v210::widen(p.y), v210::widen(cb), v210::widen(cr),
                         p.w, p.h, lead, stride);
  out.image.layout = sensor::YuvLayout::V210;
  out.image.host = out.bytes.data();
  out.image.offset[0] = lead;
  out.image.stride[0] = stride;
  return out;
}

// V210 of a YUYV pair's codes times 4, both limited range as SDI's is, gives
// YUYV's views bit for bit, as R'G'B' words and as luma bytes, from the host
// and from device buffers: the pass reads V210 through the colour pass's
// sampler, at 1280 wide (213 groups and two pixels of another).
int test_v210() {
  const sensor::StereoInfo info = mild_pair();
  const Scene scene;
  const Mat4d right_to_left =
      camera::rigid_inverse(info.calibration.left_to_right);
  Picture yuyv[2] = {color_picture(scene, info.calibration.left, Mat4d(1.0),
                                   sensor::YuvLayout::Yuyv),
                     color_picture(scene, info.calibration.right, right_to_left,
                                   sensor::YuvLayout::Yuyv)};
  for (Picture& p : yuyv) {
    p.image.kr = 0.2126f;
    p.image.kb = 0.0722f;
    p.image.full_range = false;
  }
  const Picture v210[2] = {as_v210(yuyv[0]), as_v210(yuyv[1])};
  CHECK(info.calibration.left.size.width % 6 != 0);
  for (const Rectifier::Output output :
       {Rectifier::Output::Rgba8, Rectifier::Output::Luma8}) {
    const auto rectifier = make(info, {{640, 360}, 0.25f, output});
    CHECK(rectifier != nullptr);
    const std::size_t bytes = view_bytes(rectifier->rig(), output);
    const auto views = [&](const sensor::StereoFrame& f) {
      std::array<std::vector<std::uint8_t>, 2> out;
      auto pair = rectifier->rectify(f);
      if (!pair) {
        std::fprintf(stderr, "%s\n", pair.status().message().c_str());
        return out;
      }
      out[0] = read_view(*pair, *pair->left, bytes);
      out[1] = read_view(*pair, *pair->right, bytes);
      return out;
    };
    const auto want = views(frame_of(yuyv[0], yuyv[1]));
    CHECK(want[0].size() == bytes && want[1].size() == bytes);
    sensor::StereoFrame dev = frame_of(v210[0], v210[1]);
    dev.left = on_device(v210[0], 300);
    dev.right = on_device(v210[1], 8);
    CHECK(dev.left.device != nullptr && dev.right.device != nullptr);
    const bool host_same = views(frame_of(v210[0], v210[1])) == want;
    const bool device_same = views(dev) == want;
    const auto lit = std::count_if(want[0].begin(), want[0].end(),
                                   [](std::uint8_t b) { return b != 0; });
    std::printf(
        "  v210 against YUYV, as %s: %td of %zu left bytes lit; host %s, "
        "device %s\n",
        output == Rectifier::Output::Luma8 ? "luma" : "R'G'B'", lit, bytes,
        host_same ? "same" : "DIFFERENT", device_same ? "same" : "DIFFERENT");
    CHECK(lit > static_cast<std::ptrdiff_t>(bytes / 2));
    CHECK(host_same && device_same);
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
  info.model = "RealSense D455";
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

// The right picture is let go once read: a pair holds its frame's left
// picture and `pixels`, but not the right picture's own owner
// (StereoFrame::right_pixels) or its device buffer, which go with the frame
// passed in; the views are as ever.
int test_right_let_go() {
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

  // Host pictures, each with an owner of its own.
  std::weak_ptr<const void> left_owner, right_owner;
  {
    auto pair = [&] {
      sensor::StereoFrame f = frame_of(left, right);
      const auto l =
          std::make_shared<const std::vector<std::uint8_t>>(left.bytes);
      const auto r =
          std::make_shared<const std::vector<std::uint8_t>>(right.bytes);
      f.left.host = l->data();
      f.right.host = r->data();
      f.pixels = l;
      f.right_pixels = r;
      left_owner = f.pixels;
      right_owner = f.right_pixels;
      return rectifier->rectify(f);
    }();
    CHECK(pair.ok());
    CHECK(right_owner.expired() && !left_owner.expired());
    CHECK(pair->source.pixels == left_owner.lock());
    CHECK(pair->source.right_pixels == nullptr);
    CHECK(pair->source.right.host == nullptr &&
          pair->source.right.device == nullptr);
    CHECK(pair->source.left.host != nullptr);
    CHECK(read_view(*pair, *pair->left, bytes) == want[0] &&
          read_view(*pair, *pair->right, bytes) == want[1]);
  }
  CHECK(left_owner.expired());

  // Device pictures: the right buffer goes with the frame, the left stays.
  std::weak_ptr<const vkc::Buffer> left_buffer, right_buffer;
  {
    auto pair = [&] {
      sensor::StereoFrame f = frame_of(left, right);
      f.left = on_device(left, 0);
      f.right = on_device(right, 4);
      left_buffer = f.left.device;
      right_buffer = f.right.device;
      return rectifier->rectify(f);
    }();
    CHECK(pair.ok());
    CHECK(!left_buffer.expired() && right_buffer.expired());
    CHECK(pair->source.left.device == left_buffer.lock());
    CHECK(pair->source.right.device == nullptr);
    CHECK(read_view(*pair, *pair->left, bytes) == want[0] &&
          read_view(*pair, *pair->right, bytes) == want[1]);
  }
  std::puts("  the right picture let go once read, host and device");
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

// --- The area prefilter -----------------------------------------------------

constexpr double kPi = 3.14159265358979323846;

// A w x h luma picture, full range, of `level(x, y)` at each pixel, rows
// kPad bytes longer.
template <typename Level>
Picture luma_of(std::uint32_t w, std::uint32_t h, const Level& level) {
  Picture p;
  p.w = w;
  p.h = h;
  p.y.resize(std::size_t{w} * h);
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t x = 0; x < w; ++x) {
      p.y[std::size_t{r} * w + x] = to_code(level(x, r));
    }
  }
  const std::size_t stride = w + kPad;
  p.bytes.assign(stride * h, 0xCD);
  for (std::uint32_t r = 0; r < h; ++r) {
    std::memcpy(&p.bytes[r * stride], &p.y[std::size_t{r} * w], w);
  }
  p.image.layout = sensor::YuvLayout::Luma;
  p.image.host = p.bytes.data();
  p.image.stride[0] = stride;
  p.image.width = w;
  p.image.height = h;
  p.image.full_range = true;
  return p;
}

// A pair rectified already: one pinhole camera, w x h, of focal length f,
// the right one 0.1 m along +x.
sensor::StereoInfo pinhole_pair(std::uint32_t w, std::uint32_t h, double f) {
  sensor::StereoInfo info;
  info.calibration.left = info.calibration.right =
      CameraModel{{w, h}, {f, f, (w - 1) / 2.0, (h - 1) / 2.0}, {}};
  info.calibration.left_to_right[3] = glm::dvec4(-0.1, 0.0, 0.0, 1.0);
  return info;
}

// A near-equidistant lens on a 4K picture, r_d about atan(r) to 3% out to
// 55 degrees (atan(r) / r's Pade [2/2] in r^2), which never folds: a
// fisheye such as a 4K stereo pair's, whose view at 960x544 shrinks the
// picture 4.8:1 at the centre and about 1.5 radially by 3 tangentially at
// its side edges.
CameraModel near_equidistant() {
  CameraModel c{{3840, 2160}, {1500.0, 1500.0, 1919.5, 1079.5}, {}};
  c.distortion.k1 = 735.0 / 945;
  c.distortion.k2 = 64.0 / 945;
  c.distortion.k4 = 1050.0 / 945;
  c.distortion.k5 = 225.0 / 945;
  return c;
}

// Rings about camera `c`'s principal point, `period` source pixels apart
// radially, +-64 levels about 128.
Picture rings_of(const CameraModel& c, double period) {
  const double cx = c.intrinsics.cx, cy = c.intrinsics.cy;
  return luma_of(c.size.width, c.size.height,
                 [&](std::uint32_t x, std::uint32_t y) {
                   const double r = std::hypot(x - cx, y - cy);
                   return 128 + 64 * std::cos(2 * kPi * r / period);
                 });
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t h = 14695981039346656037ull;
  for (const std::uint8_t b : bytes) {
    h ^= b;
    h *= 1099511628211ull;
  }
  return h;
}

// Where the pixels a view's footprint averages are one tap -- a D455's
// infrared pair, 1280x800, to FFS's 960x544, 1.33:1 -- random luma comes
// back as the one-tap reference has it, byte for byte but at ties, half way
// between two codes, which float rounds either way; its hash, printed,
// compares builds by hand. The luma spans 32 codes: averaged, it would move
// pixels by several codes, and float's error in where a sample lands moves
// it by under 0.02 of a code.
int prefilter_one_tap() {
  const sensor::StereoInfo info = pinhole_pair(1280, 800, 640.0);
  const auto rectifier =
      make(info, {{960, 544}, 0.0f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr);
  const sensor::StereoRig& rig = rectifier->rig();
  std::uint32_t state = 0x9E3779B9u;
  const Picture pic =
      luma_of(1280, 800, [&state](std::uint32_t, std::uint32_t) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return 112.0 + static_cast<double>(state >> 27);
      });
  auto pair = rectifier->rectify(frame_of(pic, pic));
  CHECK(pair.ok());
  const std::vector<std::uint8_t> view =
      read_view(*pair, *pair->left, view_bytes(rig, Rectifier::Output::Luma8));
  CHECK(view.size() >= 960u * 544u);
  const Remap remap(rig, info.calibration, 0);
  // Ties: a sample within 0.02 of half way between two codes, which float
  // may round either way. Vulkan lets a shader's division be 2.5 ulps off,
  // so the device and the host may place a sample a few float steps apart
  // (6e-4 px is five at x 1280); neighbours at most 31 codes apart turn that
  // into under 0.02 of a code.
  std::size_t off = 0, ties = 0, total = 0;
  int worst = 0;
  double longest = 0;
  for (std::uint32_t v = 0; v < 544; ++v) {
    for (std::uint32_t u = 0; u < 960; ++u) {
      float sx = 0, sy = 0, r = 0;
      CHECK(remap.source(u, v, &sx, &sy, &r));
      Vec2d du, dv;
      CHECK(remap.footprint(u, v, &du, &dv) == 0);
      longest = std::max({longest, glm::length(du), glm::length(dv)});
      const float y = bilinear(pic.y, pic.w, pic.w, pic.h, sx, sy);
      const int want = luma_code(y, true);
      const int got = view[std::size_t{v} * 960 + u];
      worst = std::max(worst, std::abs(want - got));
      const bool tie = std::fabs(y - std::floor(y) - 0.5f) < 0.02f;
      ties += tie ? 1 : 0;
      off += !tie && want != got ? 1 : 0;
      ++total;
    }
  }
  std::printf(
      "  a D455's 1280x800 to 960x544: footprint %.3f source px at most, one "
      "tap; of %zu pixels, %zu off the one-tap reference away from %zu "
      "ties, worst %d; view FNV-1a %016llx\n",
      longest, total, off, ties, worst,
      static_cast<unsigned long long>(fnv1a(view)));
  CHECK(longest < 1.4);
  CHECK(worst <= 1);
  CHECK(off == 0);
  return 0;
}

// At 4.5:1 -- a pair rectified already, 4320x2448 to 960x544 -- a checker
// grating of period 5 source pixels (1.1 view pixels: past the view's
// Nyquist in both axes) reads flat, within 2 levels of its mean, where one
// tap a pixel aliases it to tens; and a step edge stays where the rig puts
// it, within 0.03 view pixels, at each of six sub-pixel phases, across and
// down (its place read off the area under the view across it, which a box
// keeps), where one tap a pixel moves it by up to a third of a pixel.
int prefilter_shrink() {
  const sensor::StereoInfo info = pinhole_pair(4320, 2448, 2000.0);
  const auto rectifier =
      make(info, {{960, 544}, 0.0f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr);
  const sensor::StereoRig& rig = rectifier->rig();
  const camera::PinholeIntrinsics& k = rig.camera.intrinsics;
  CHECK(std::fabs(k.fx * 4.5 - 2000.0) < 1e-9);
  const std::size_t bytes = view_bytes(rig, Rectifier::Output::Luma8);
  const Remap remap(rig, info.calibration, 0);
  const auto view_of = [&](const Picture& p) {
    auto pair = rectifier->rectify(frame_of(p, p));
    return pair.ok() ? read_view(*pair, *pair->left, bytes)
                     : std::vector<std::uint8_t>{};
  };
  const auto one_tap = [&](const Picture& p, std::uint32_t u, std::uint32_t v) {
    float sx = 0, sy = 0, r = 0;
    remap.source(u, v, &sx, &sy, &r);
    return luma_code(bilinear(p.y, p.w, p.w, p.h, sx, sy), true);
  };

  const Picture checker =
      luma_of(4320, 2448, [](std::uint32_t x, std::uint32_t y) {
        return 128 +
               64 * std::cos(2 * kPi * x / 5.0) * std::cos(2 * kPi * y / 5.0);
      });
  const std::vector<std::uint8_t> flat = view_of(checker);
  CHECK(flat.size() == bytes);
  std::size_t compared = 0;
  const int worst =
      compare_view(flat, remap, checker, Rectifier::Output::Luma8, &compared);
  int ripple = 0, aliased = 0;
  for (std::uint32_t v = 0; v < 544; ++v) {
    for (std::uint32_t u = 0; u < 960; ++u) {
      ripple = std::max(ripple, std::abs(flat[std::size_t{v} * 960 + u] - 128));
      aliased = std::max(aliased, std::abs(one_tap(checker, u, v) - 128));
    }
  }
  std::printf(
      "  4320x2448 to 960x544 (4.5:1), a checker grating of period 5 source "
      "px, +-64: within %d levels of its mean (one tap a pixel: %d); %zu "
      "pixels against the reference, worst %d\n",
      ripple, aliased, compared, worst);
  CHECK(worst >= 0 && worst <= 1);
  CHECK(compared > 960u * 544u * 9 / 10);
  CHECK(ripple <= 2);
  CHECK(aliased >= 20);

  // Six bands of rows, each a step across at its own column, one source
  // pixel (two ninths of a view pixel) apart; then six bands of columns,
  // each a step down at its own row.
  const Picture across =
      luma_of(4320, 2448, [](std::uint32_t x, std::uint32_t y) {
        return x < 2000 + y / 408 ? 64.0 : 192.0;
      });
  const Picture down =
      luma_of(4320, 2448, [](std::uint32_t x, std::uint32_t y) {
        return y < 1200 + x / 720 ? 64.0 : 192.0;
      });
  const camera::PinholeIntrinsics& sk = info.calibration.left.intrinsics;
  double worst_step = 0, worst_one = 0;
  for (int axis = 0; axis < 2; ++axis) {
    const Picture& pic = axis == 0 ? across : down;
    const std::vector<std::uint8_t> view = view_of(pic);
    CHECK(view.size() == bytes);
    const double band_size = axis == 0 ? 408.0 : 720.0;
    for (int band = 0; band < 6; ++band) {
      // The step in the source (the bilinear ramp's middle, half a pixel
      // before its first bright sample), and in the view.
      const double edge = (axis == 0 ? 2000.0 : 1200.0) + band - 0.5;
      const double at = axis == 0 ? (edge - sk.cx) / sk.fx * k.fx + k.cx
                                  : (edge - sk.cy) / sk.fy * k.fy + k.cy;
      const int first = static_cast<int>(std::floor(at)) - 6;
      const int last = first + 12;
      double err = 0, err_one = 0;
      int lines = 0;
      for (std::uint32_t l = 0; l < (axis == 0 ? 544u : 960u); ++l) {
        // The line's place across the bands, in the source; only lines
        // well inside the band.
        const double c = axis == 0 ? (l - k.cy) / k.fy * sk.fy + sk.cy
                                   : (l - k.cx) / k.fx * sk.fx + sk.cx;
        if (c < band * band_size + 100 || c > (band + 1) * band_size - 100) {
          continue;
        }
        double area = 0, area_one = 0;
        for (int t = first; t <= last; ++t) {
          const std::uint32_t u = axis == 0 ? static_cast<std::uint32_t>(t) : l;
          const std::uint32_t v = axis == 0 ? l : static_cast<std::uint32_t>(t);
          area += (view[std::size_t{v} * 960 + u] - 64.0) / 128.0;
          area_one += (one_tap(pic, u, v) - 64.0) / 128.0;
        }
        err += last + 0.5 - area - at;
        err_one += last + 0.5 - area_one - at;
        ++lines;
      }
      CHECK(lines > 10);
      worst_step = std::max(worst_step, std::fabs(err / lines));
      worst_one = std::max(worst_one, std::fabs(err_one / lines));
    }
  }
  std::printf(
      "  step edges at 4.5:1, six phases across and six down: within %.4f "
      "view px of where the rig puts them (one tap a pixel: %.3f)\n",
      worst_step, worst_one);
  CHECK(worst_step < 0.03);
  return 0;
}

// Each pixel's own footprint, not the centre's: through the near-equidistant
// lens, 3840x2160 to 960x544, the footprint is about 4.8 source pixels at
// the centre and 1.5 across by 3 down at the side edge. Rings whose period
// is the centre's footprint average out near the centre (past the view's
// Nyquist there) and keep their contrast at the side edge (below its own,
// 3 view pixels a ring), where a centre-sized box would flatten them too;
// the view matches the host reference within a code.
int prefilter_anisotropic() {
  sensor::StereoInfo info;
  info.calibration.left = info.calibration.right = near_equidistant();
  info.calibration.left_to_right[3] = glm::dvec4(-0.4, 0.0, 0.0, 1.0);
  const auto rectifier =
      make(info, {{960, 544}, 0.0f, Rectifier::Output::Luma8});
  CHECK(rectifier != nullptr);
  const sensor::StereoRig& rig = rectifier->rig();
  const Remap remap(rig, info.calibration, 0);
  const auto cu =
      static_cast<std::uint32_t>(std::lround(rig.camera.intrinsics.cx));
  const auto cv =
      static_cast<std::uint32_t>(std::lround(rig.camera.intrinsics.cy));
  Vec2d cdu, cdv, edu, edv;
  CHECK(remap.footprint(cu, cv, &cdu, &cdv) == 1);
  CHECK(remap.footprint(959, cv, &edu, &edv) == 1);
  const double centre = glm::length(cdu);
  std::printf(
      "  a near-equidistant lens, 3840x2160 to 960x544 (fx %.1f): footprint "
      "%.2f x %.2f source px at the centre, %.2f x %.2f at the side edge "
      "(across x down)\n",
      rig.camera.intrinsics.fx, centre, glm::length(cdv), glm::length(edu),
      glm::length(edv));
  CHECK(centre > 4.4 && glm::length(cdv) > 4.4);
  CHECK(glm::length(edu) < 1.7 && glm::length(edv) > 2.7);

  const Picture rings = rings_of(info.calibration.left, centre);
  auto pair = rectifier->rectify(frame_of(rings, rings));
  CHECK(pair.ok());
  const std::vector<std::uint8_t> view =
      read_view(*pair, *pair->left, view_bytes(rig, Rectifier::Output::Luma8));
  CHECK(view.size() >= 960u * 544u);
  std::size_t compared = 0;
  const int worst =
      compare_view(view, remap, rings, Rectifier::Output::Luma8, &compared);
  // Half the spread of levels along the middle row, from `from` to `to`.
  const auto contrast = [](const auto& level, std::uint32_t from,
                           std::uint32_t to) {
    double lo = 255, hi = 0;
    for (std::uint32_t u = from; u <= to; ++u) {
      lo = std::min(lo, level(u));
      hi = std::max(hi, level(u));
    }
    return (hi - lo) / 2;
  };
  const auto in_view = [&](std::uint32_t u) {
    return static_cast<double>(view[std::size_t{cv} * 960 + u]);
  };
  // What a box of the centre's footprint, across and down, would give.
  const auto centre_box = [&](std::uint32_t u) {
    float sx = 0, sy = 0, r = 0;
    remap.source(u, cv, &sx, &sy, &r);
    const auto luma = [&](float x, float y) {
      return std::array<float, 3>{
          bilinear(rings.y, rings.w, rings.w, rings.h, x, y), 0.0f, 0.0f};
    };
    const std::array<float, 3> mean = area_mean(
        luma, Vec2d(sx, sy), Vec2d(centre, 0.0), Vec2d(0.0, centre), kTaps);
    return static_cast<double>(luma_code(mean[0], true));
  };
  const double kept = contrast(in_view, 928, 959);
  const double boxed = contrast(centre_box, 928, 959);
  const double middle = contrast(in_view, cu + 20, cu + 50);
  std::printf(
      "  rings of the centre's period, +-64: at the side edge +-%.1f kept (a "
      "centre-sized box: +-%.1f); near the centre +-%.1f; %zu pixels against "
      "the reference, worst %d\n",
      kept, boxed, middle, compared, worst);
  CHECK(worst >= 0 && worst <= 1);
  CHECK(compared > 960u * 544u * 9 / 10);
  CHECK(kept >= 32);
  CHECK(boxed <= 10);
  CHECK(middle <= 6);
  return 0;
}

// The 4.5:1 shrink and the 4K lens's footprints run on every leg but
// AddressSanitizer's: there they would rectify 4K pictures down a path that
// layouts and fold check already, each planted bug they catch failing those
// too (DECISIONS.md, the 2026-11-04 test budget entry).
int test_prefilter() {
  if (prefilter_one_tap() != 0) return 1;
  if (kAddressSanitizer) {
    std::puts(
        "  the 4.5:1 shrink and the 4K lens's footprints: not under "
        "AddressSanitizer");
    return 0;
  }
  if (prefilter_shrink() != 0) return 1;
  if (prefilter_anisotropic() != 0) return 1;
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
        source.right.device == nullptr);

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
        source.right.device == nullptr &&
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
  // One past the last.
  f.left.layout = static_cast<sensor::YuvLayout>(
      static_cast<int>(sensor::YuvLayout::V210) + 1);
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
// device time from the batch's timestamps, host time around rectify. Run by
// hand (--timing), on an otherwise idle GPU: it asserts nothing the other
// cases do not.
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

  // A 4K stereo camera's: 3840x2160 V210 in device buffers, through the
  // near-equidistant lens, to 960x544 R'G'B', every pixel averaged.
  sensor::StereoInfo wide;
  wide.calibration.left = wide.calibration.right = near_equidistant();
  wide.calibration.left_to_right[3] = glm::dvec4(-0.4, 0.0, 0.0, 1.0);
  const auto rectifier =
      make(wide, {{960, 544}, 0.0f, Rectifier::Output::Rgba8});
  CHECK(rectifier != nullptr);
  Picture rings = rings_of(wide.calibration.left, 4.8);
  rings.cb.assign(rings.y.size(), 128);
  rings.cr.assign(rings.y.size(), 128);
  rings.image.kr = 0.2126f;
  rings.image.kb = 0.0722f;
  rings.image.full_range = false;
  rings.image.chroma_location = sensor::ChromaLocation::Left;
  const Picture v210 = as_v210(rings);
  sensor::StereoFrame f = frame_of(v210, v210);
  f.left = on_device(v210, 0);
  f.right = on_device(v210, 0);
  CHECK(f.left.device != nullptr && f.right.device != nullptr);
  CHECK(rectifier->rectify(f).ok());  // warm
  vkc::StageMetrics metrics;
  for (int i = 0; i < kRuns; ++i) {
    CHECK(rectifier->rectify(f, &metrics).ok());
  }
  for (const vkc::StageRow& row : metrics.rows()) {
    if (std::strcmp(row.name, "stereo rectify") != 0) continue;
    std::printf(
        "  3840x2160 V210 in device buffers -> 960x544 Rgba8, both views, "
        "averaged: %.3f ms device, %.3f ms host a pair, over %d\n",
        row.gpu_ms / kRuns, row.cpu_ms / kRuns, kRuns);
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  std::printf("sensor_stereo_rectifier on %s\n",
              gpu.physical.properties().deviceName);
  if (test_rows_align() != 0) return 1;
  if (test_layouts() != 0) return 1;
  if (test_v210() != 0) return 1;
  if (test_already_rectified() != 0) return 1;
  if (test_depth_info() != 0) return 1;
  if (test_pairs_hold_views() != 0) return 1;
  if (test_right_let_go() != 0) return 1;
  if (test_fold() != 0) return 1;
  if (test_prefilter() != 0) return 1;
  if (test_taken_over() != 0) return 1;
  if (test_refusals() != 0) return 1;
  std::puts("sensor_stereo_rectifier: OK");
  return 0;
}

int timing_main(vkc::test::GpuContext& gpu) {
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  std::printf("sensor_stereo_rectifier --timing on %s\n",
              gpu.physical.properties().deviceName);
  if (test_timing() != 0) return 1;
  std::puts("sensor_stereo_rectifier --timing: OK");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--timing") {
    return vkc::test::run_on_gpu(sensor::device_requirements(), timing_main);
  }
  if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--timing]\n", argv[0]);
    return 1;
  }
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
