// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep's depth outlier filter (GpuFramePrepConfig::depth_outliers) on
// synthetic frames, no camera:
//   - the filtered depth is the unfiltered depth where a host reference keeps
//     the return and 0 elsewhere, bit for bit, through a pinhole and through
//     a lens: the reference counts every return within a window four times
//     the pass's bound, in float, each operation rounded once, as the pass
//     computes;
//   - on a scene of a slanted wall, a box in front of it, a curtain of flying
//     pixels between the two, a one-pixel cable, isolated returns and holes,
//     it removes the curtain, the cable and the isolated returns and keeps
//     the wall and the box;
//   - on a slab of scattered returns, at the slab's median count, where a
//     window a pixel short flips decisions, it still matches;
//   - on a plane facing the camera, where every return's decision rests on
//     neighbours at almost exactly its radius, it still matches: an operation
//     fused or regrouped, or a <= for the <, flips decisions there;
//   - at the largest radius, its window past the image on every side, it
//     still matches;
//   - it runs after the colour-view test, on the depth that test keeps;
//   - several cameras in one batch, of different sizes, and a pass whose
//     frames grow, each match their reference;
//   - off by default, and its refusals at create.
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

constexpr float kScale = 0.001f;  // metres per unit

vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

camera::CameraModel pinhole(std::uint32_t w = 128, std::uint32_t h = 96) {
  camera::CameraModel c;
  c.size = {w, h};
  c.intrinsics = {110.0, 108.5, 0.497 * w, 0.494 * h};
  return c;
}

// A barrel lens: the filter sees the undistorted image's pinhole camera.
camera::CameraModel lensed(std::uint32_t w, std::uint32_t h) {
  camera::CameraModel c = pinhole(w, h);
  c.distortion = {-0.12, 0.03, 0.0008, -0.0005, 0.0, 0.0, 0.0, 0.0};
  return c;
}

// A hash of pixel i in [0, 1).
float unit(std::uint32_t i) {
  std::uint32_t x = i * 2654435761u + 0x9E3779B9u;
  x ^= x >> 15;
  x *= 0x2C1B3C6Du;
  x ^= x >> 12;
  return static_cast<float>(x >> 8) / 16777216.0f;
}

// Where each part of scene() lies, by pixel.
enum class Part : std::uint8_t { Hole, Wall, Box, Curtain, Cable, Isolated };

struct Scene {
  std::vector<std::uint16_t> raw;  // millimetres
  std::vector<Part> part;
};

// A wall 2.5 m away turned about the vertical, a box 1.2 m away, a curtain of
// flying pixels two columns wide down the box's right edge at random depths
// between the two (each farther than its radius from both), a one-pixel cable
// 1.6 m away, an isolated return 0.5-1 m away, nearer than anything else,
// every 97th pixel, and a hole every 41st.
Scene scene(const camera::CameraModel& c) {
  const std::uint32_t w = c.size.width, h = c.size.height;
  Scene s;
  s.raw.resize(std::size_t{w} * h);
  s.part.resize(s.raw.size());
  const std::uint32_t bx0 = w / 3, bx1 = w / 3 + w / 4;
  const std::uint32_t by0 = h / 3, by1 = h / 3 + h / 3;
  const std::uint32_t cable = w * 4 / 5;
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) {
      const std::size_t i = std::size_t{v} * w + u;
      const double xn = (u - c.intrinsics.cx) / c.intrinsics.fx;
      const double wall = 2.5 / (1.0 - 0.6 * xn);
      double z = wall;
      Part p = Part::Wall;
      if (v >= by0 && v < by1 && u >= bx0 && u < bx1) {
        z = 1.2;
        p = Part::Box;
      } else if (v >= by0 && v < by1 && (u == bx1 || u == bx1 + 1)) {
        z = 1.2 +
            (wall - 1.2) * (0.15 + 0.6 * unit(static_cast<std::uint32_t>(i)));
        p = Part::Curtain;
      } else if (u == cable && v >= h / 10 && v < h - h / 10) {
        z = 1.6;
        p = Part::Cable;
      }
      if (i % 97 == 50) {
        z = 0.5 + 0.5 * unit(static_cast<std::uint32_t>(i) + 7u);
        p = Part::Isolated;
      }
      if (i % 41 == 20) {
        z = 0.0;
        p = Part::Hole;
      }
      s.raw[i] = static_cast<std::uint16_t>(std::lround(z * 1000.0));
      s.part[i] = p;
    }
  }
  return s;
}

// A slab of returns scattered 1.5 m +- 12 cm: about the radius at 1.5 m at
// 8 px, so a return's neighbours fill its ball unevenly, out to its sphere.
std::vector<std::uint16_t> slab(const camera::CameraModel& c) {
  std::vector<std::uint16_t> raw(std::size_t{c.size.width} * c.size.height);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const float z = 1.5f + 0.24f * (unit(static_cast<std::uint32_t>(i)) - 0.5f);
    raw[i] = static_cast<std::uint16_t>(std::lround(z * 1000.0f));
  }
  return raw;
}

sensor::RgbdFrame frame_of(const std::vector<std::uint16_t>& raw,
                           const camera::CameraModel& cam) {
  sensor::RgbdFrame f;
  f.depth = raw.data();
  f.metres_per_unit = kScale;
  f.depth_camera = cam;
  f.min_depth = 0.1f;
  f.max_depth = 10.0f;
  return f;
}

std::vector<float> depth_of(const sensor::DeviceFrame& f) {
  auto out = vkc::test::read_back<float>(
      *g_device, *g_allocator, *f.depth,
      std::size_t{f.depth_camera.width} * f.depth_camera.height);
  if (!out) {
    std::fprintf(stderr, "read back: %s\n", out.status().message().c_str());
    return {};
  }
  return std::move(out).value();
}

// `frame` prepared by `prep`, its depth read back; empty on a failure.
std::vector<float> prepared(sensor::GpuFramePrep& prep,
                            const sensor::RgbdFrame& frame) {
  auto out = prep.prepare(frame);
  if (!out) {
    std::fprintf(stderr, "prepare: %s\n", out.status().message().c_str());
    return {};
  }
  return depth_of(out.value());
}

sensor::GpuFramePrepConfig filtering(float radius_px,
                                     std::uint32_t min_points) {
  sensor::GpuFramePrepConfig config;
  config.depth_outliers.enabled = true;
  config.depth_outliers.radius_px = radius_px;
  config.depth_outliers.min_points = min_points;
  return config;
}

// The pass's camera and radius in float, as it narrows them: the
// reciprocals and radius_px / fx from the doubles.
struct Rule {
  float fx, fy, cx, cy, inv_fx, inv_fy, radius_per_m;
};

Rule rule_of(const camera::CameraModel& c, float radius_px) {
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const auto f = [](double v) { return static_cast<float>(v); };
  return Rule{f(k.fx),
              f(k.fy),
              f(k.cx),
              f(k.cy),
              f(1.0 / k.fx),
              f(1.0 / k.fy),
              f(double{radius_px} / k.fx)};
}

// The point at pixel (u, v), z deep, as the pass computes it, each operation
// rounded once: the test is built with contraction off.
void point(const Rule& k, int u, int v, float z, float* p) {
  const float du = static_cast<float>(u) - k.cx;
  const float nx = du * k.inv_fx;
  p[0] = nx * z;
  const float dv = static_cast<float>(v) - k.cy;
  const float ny = dv * k.inv_fy;
  p[1] = ny * z;
  p[2] = z;
}

// Every return's count of returns strictly within its radius, itself
// included, over a window of +-kReach px, past the pass's bound: four times
// it at 8 px, and past its at most 38 px at the largest radius on test_cap's
// image; 0 for no return.
constexpr int kReach = 40;

std::vector<std::uint32_t> counts(const std::vector<float>& depth,
                                  const camera::CameraModel& c,
                                  float radius_px) {
  const Rule k = rule_of(c, radius_px);
  const int w = static_cast<int>(c.size.width);
  const int h = static_cast<int>(c.size.height);
  std::vector<std::uint32_t> n(depth.size(), 0);
  for (int v = 0; v < h; ++v) {
    for (int u = 0; u < w; ++u) {
      const float z = depth[static_cast<std::size_t>(v * w + u)];
      if (!(z > 0.0f)) continue;
      const float r = k.radius_per_m * z;
      const float r2 = r * r;
      float p[3];
      point(k, u, v, z, p);
      std::uint32_t count = 0;
      for (int y = std::max(v - kReach, 0); y <= std::min(v + kReach, h - 1);
           ++y) {
        for (int x = std::max(u - kReach, 0); x <= std::min(u + kReach, w - 1);
             ++x) {
          const float zj = depth[static_cast<std::size_t>(y * w + x)];
          if (!(zj > 0.0f)) continue;
          float q[3];
          point(k, x, y, zj, q);
          const float dx = q[0] - p[0];
          const float dy = q[1] - p[1];
          const float dz = q[2] - p[2];
          float d2 = dx * dx;
          const float y2 = dy * dy;
          d2 = d2 + y2;
          const float z2 = dz * dz;
          d2 = d2 + z2;
          count += d2 < r2 ? 1u : 0u;
        }
      }
      n[static_cast<std::size_t>(v * w + u)] = count;
    }
  }
  return n;
}

// How many pixels of `filtered` differ from `unfiltered` kept where the count
// reaches `min_points` and 0 elsewhere.
std::size_t mismatches(const std::vector<float>& filtered,
                       const std::vector<float>& unfiltered,
                       const std::vector<std::uint32_t>& n,
                       std::uint32_t min_points) {
  std::size_t off = 0;
  for (std::size_t i = 0; i < unfiltered.size(); ++i) {
    const float want = n[i] >= min_points ? unfiltered[i] : 0.0f;
    off += filtered[i] != want;  // bit for bit; 0 == 0 for any sign
  }
  return off;
}

std::uint32_t median_of(std::vector<std::uint32_t> n) {
  std::nth_element(n.begin(), n.begin() + n.size() / 2, n.end());
  return n[n.size() / 2];
}

// The scene through `cam`: the filtered depth against the reference on the
// unfiltered depth, and, with `parts`, each part removed or kept as it should
// be (through a pinhole, where each pixel shows its own raw pixel).
int check_scene(vkc::Device& device, vkc::Allocator& allocator,
                const camera::CameraModel& cam, const char* what, bool parts) {
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  auto filter =
      sensor::GpuFramePrep::create(device, allocator, filtering(8.0f, 20));
  CHECK(plain.ok() && filter.ok());
  const Scene s = scene(cam);
  const sensor::RgbdFrame f = frame_of(s.raw, cam);
  const std::vector<float> before = prepared(plain.value(), f);
  const std::vector<float> after = prepared(filter.value(), f);
  CHECK(!before.empty() && after.size() == before.size());
  const std::vector<std::uint32_t> n = counts(before, cam, 8.0f);
  const std::size_t off = mismatches(after, before, n, 20);
  // Kept, by part, of the returns the undistortion left.
  std::size_t kept[6] = {}, all[6] = {};
  const std::uint32_t w = cam.size.width;
  for (std::size_t i = 0; i < before.size(); ++i) {
    if (!(before[i] > 0.0f)) continue;
    const auto p = static_cast<std::size_t>(s.part[i]);
    all[p] += 1;
    kept[p] += after[i] > 0.0f;
  }
  std::printf(
      "  %s: %zu off; kept wall %zu/%zu, box %zu/%zu, curtain %zu/%zu, "
      "cable %zu/%zu, isolated %zu/%zu\n",
      what, off, kept[1], all[1], kept[2], all[2], kept[3], all[3], kept[4],
      all[4], kept[5], all[5]);
  CHECK(off == 0);
  if (!parts) {
    CHECK(kept[1] + kept[2] > (all[1] + all[2]) * 9 / 10);
    return 0;
  }
  CHECK(all[3] > 40 && all[4] > 40 && all[5] > 40);
  CHECK(kept[1] > all[1] * 95 / 100 && kept[2] > all[2] * 90 / 100);
  CHECK(kept[3] < all[3] / 10 && kept[4] < all[4] / 10 && kept[5] == 0);
  // A wall pixel far from every edge is always kept.
  const std::size_t inner = std::size_t{8} * w + 8;
  CHECK(s.part[inner] == Part::Wall && after[inner] == before[inner]);
  return 0;
}

int test_scene(vkc::Device& device, vkc::Allocator& allocator) {
  if (check_scene(device, allocator, pinhole(), "pinhole", true) != 0) {
    return 1;
  }
  // Odd, and through a lens.
  return check_scene(device, allocator, lensed(131, 97), "lens", false);
}

// Scattered returns at their median count: a window short by a pixel would
// flip some.
int test_slab(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cam = pinhole(96, 80);
  const std::vector<std::uint16_t> raw = slab(cam);
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  CHECK(plain.ok());
  const std::vector<float> before = prepared(plain.value(), frame_of(raw, cam));
  CHECK(!before.empty());
  const std::vector<std::uint32_t> n = counts(before, cam, 8.0f);
  const std::uint32_t median = median_of(n);
  std::size_t near = 0;
  for (const std::uint32_t c : n) {
    near += c + 2 >= median && c <= median + 2;
  }
  auto filter =
      sensor::GpuFramePrep::create(device, allocator, filtering(8.0f, median));
  CHECK(filter.ok());
  const std::vector<float> after = prepared(filter.value(), frame_of(raw, cam));
  CHECK(after.size() == before.size());
  const std::size_t off = mismatches(after, before, n, median);
  std::printf("  slab: median count %u, %zu of %zu within 2 of it, %zu off\n",
              median, near, n.size(), off);
  CHECK(off == 0);
  CHECK(median > 20 && near > n.size() / 50);
  return 0;
}

// Ties: a plane facing the camera, fx = fy and the principal point on a
// pixel, so a return's four axis neighbours 8 px out lie at almost exactly its
// radius (exactly, at the principal point) and every other one clear of it, as
// 64 is a sum of two squares only as 8^2 + 0^2. Kept at the disc's count
// strictly inside, 193, plus one, a return is kept only where one of its ties
// counts.
int test_ties(vkc::Device& device, vkc::Allocator& allocator) {
  camera::CameraModel cam;
  cam.size = {128, 96};
  cam.intrinsics = {110.0, 110.0, 64.0, 48.0};
  const std::vector<std::uint16_t> raw(std::size_t{128} * 96, 1500);
  constexpr std::uint32_t kInside = 193;
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  auto filter = sensor::GpuFramePrep::create(device, allocator,
                                             filtering(8.0f, kInside + 1));
  CHECK(plain.ok() && filter.ok());
  const sensor::RgbdFrame f = frame_of(raw, cam);
  const std::vector<float> before = prepared(plain.value(), f);
  const std::vector<float> after = prepared(filter.value(), f);
  CHECK(!before.empty() && after.size() == before.size());
  const std::vector<std::uint32_t> n = counts(before, cam, 8.0f);
  const std::size_t off = mismatches(after, before, n, kInside + 1);
  // Away from the borders: the returns whose ties decide, each way.
  std::size_t kept = 0, removed = 0;
  for (std::uint32_t v = 16; v < 80; ++v) {
    for (std::uint32_t u = 16; u < 112; ++u) {
      const std::size_t i = std::size_t{v} * 128 + u;
      CHECK(n[i] >= kInside && n[i] <= kInside + 4);
      (n[i] > kInside ? kept : removed) += 1;
    }
  }
  std::printf("  ties: %zu kept, %zu removed by their ties, %zu off\n", kept,
              removed, off);
  CHECK(off == 0);
  CHECK(kept > 100 && removed > 100);
  // At the principal point the ties lie at exactly the radius, outside it.
  CHECK(n[std::size_t{48} * 128 + 64] == kInside);
  return 0;
}

// The largest radius, on an image its window passes on every side, at the
// scene's median count.
int test_cap(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cam = pinhole(64, 48);
  constexpr float kRadius = sensor::DepthOutlierFilter::kMaxRadiusPx;
  const Scene s = scene(cam);
  const sensor::RgbdFrame f = frame_of(s.raw, cam);
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  CHECK(plain.ok());
  const std::vector<float> before = prepared(plain.value(), f);
  CHECK(!before.empty());
  const std::vector<std::uint32_t> n = counts(before, cam, kRadius);
  const std::uint32_t median = median_of(n);
  auto filter = sensor::GpuFramePrep::create(device, allocator,
                                             filtering(kRadius, median));
  CHECK(filter.ok());
  const std::vector<float> after = prepared(filter.value(), f);
  CHECK(after.size() == before.size());
  std::size_t kept = 0, returns = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    returns += before[i] > 0.0f;
    kept += after[i] > 0.0f;
  }
  const std::size_t off = mismatches(after, before, n, median);
  std::printf("  cap: median count %u, %zu of %zu kept, %zu off\n", median,
              kept, returns, off);
  CHECK(off == 0);
  CHECK(median > 100 && kept > returns / 4 && kept < returns * 3 / 4);
  return 0;
}

// After the colour-view test: a pincushion colour camera's corners map
// outside its picture, so depth there is zeroed before the filter, which
// counts it as no return.
int test_after_color(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cam = pinhole();
  camera::CameraModel color = cam;
  color.distortion.k1 = 0.3;
  const Scene s = scene(cam);
  const std::vector<std::uint32_t> words(s.raw.size(), 0x00808080u);
  sensor::RgbdFrame f = frame_of(s.raw, cam);
  f.color_packed = words.data();
  f.color_camera = color;
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  auto filter =
      sensor::GpuFramePrep::create(device, allocator, filtering(8.0f, 20));
  CHECK(plain.ok() && filter.ok());
  const std::vector<float> before = prepared(plain.value(), f);
  const std::vector<float> after = prepared(filter.value(), f);
  CHECK(!before.empty() && after.size() == before.size());
  std::size_t masked = 0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    masked += s.raw[i] != 0 && before[i] == 0.0f;
  }
  const std::size_t off =
      mismatches(after, before, counts(before, cam, 8.0f), 20);
  std::printf("  after the colour view: %zu masked first, %zu off\n", masked,
              off);
  CHECK(masked > 200 && off == 0);
  return 0;
}

// Several cameras in one batch, each its own size and buffers, and one pass
// whose frames grow and shrink.
int test_batch(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cams[3] = {pinhole(), lensed(131, 97),
                                       pinhole(64, 48)};
  std::vector<Scene> scenes;
  std::vector<sensor::GpuFramePrep> plain, filter;
  std::vector<std::optional<sensor::RgbdFrame>> frames;
  for (const camera::CameraModel& c : cams) {
    scenes.push_back(scene(c));
    auto p = sensor::GpuFramePrep::create(device, allocator);
    auto q =
        sensor::GpuFramePrep::create(device, allocator, filtering(6.0f, 20));
    CHECK(p.ok() && q.ok());
    plain.push_back(std::move(p).value());
    filter.push_back(std::move(q).value());
  }
  for (std::size_t i = 0; i < 3; ++i) {
    frames.emplace_back(frame_of(scenes[i].raw, cams[i]));
  }
  auto before = sensor::GpuFramePrep::prepare_batch(plain, frames);
  auto after = sensor::GpuFramePrep::prepare_batch(filter, frames);
  CHECK(before.ok() && after.ok());
  const std::vector<float> first = depth_of(*(*after)[0]);
  for (std::size_t i = 0; i < 3; ++i) {
    const std::vector<float> b = depth_of(*(*before)[i]);
    const std::vector<float> a = depth_of(*(*after)[i]);
    CHECK(!b.empty() && a.size() == b.size());
    const std::size_t off = mismatches(a, b, counts(b, cams[i], 6.0f), 20);
    std::printf("  batch camera %zu: %zu off\n", i, off);
    CHECK(off == 0);
  }
  // One pass, small, large, small: its buffer grows and is reused.
  for (const std::size_t i : {2, 1, 0, 2}) {
    const sensor::RgbdFrame f = frame_of(scenes[i].raw, cams[i]);
    const std::vector<float> b = prepared(plain[0], f);
    const std::vector<float> a = prepared(filter[0], f);
    CHECK(!b.empty() && a.size() == b.size());
    CHECK(mismatches(a, b, counts(b, cams[i], 6.0f), 20) == 0);
  }
  // The batch's frame, held meanwhile, kept its own depth.
  CHECK(depth_of(*(*after)[0]) == first);
  return 0;
}

// Off by default; refused at create only when enabled.
int test_config(vkc::Device& device, vkc::Allocator& allocator) {
  const sensor::GpuFramePrepConfig defaults;
  CHECK(!defaults.depth_outliers.enabled);
  CHECK(defaults.depth_outliers.radius_px == 8.0f);
  CHECK(defaults.depth_outliers.min_points == 20u);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {nan, inf, -1.0f, 0.0f, 0.99f, 32.01f}) {
    const auto made =
        sensor::GpuFramePrep::create(device, allocator, filtering(bad, 20));
    CHECK(made.status().domain() == vkc::Status::Code::InvalidArgument);
    sensor::GpuFramePrepConfig off = filtering(bad, 20);
    off.depth_outliers.enabled = false;
    CHECK(sensor::GpuFramePrep::create(device, allocator, off).ok());
  }
  for (const std::uint32_t bad : {0u, 1u}) {
    const auto made =
        sensor::GpuFramePrep::create(device, allocator, filtering(8.0f, bad));
    CHECK(made.status().domain() == vkc::Status::Code::InvalidArgument);
  }
  CHECK(
      sensor::GpuFramePrep::create(device, allocator, filtering(1.0f, 2)).ok());
  CHECK(sensor::GpuFramePrep::create(
            device, allocator,
            filtering(sensor::DepthOutlierFilter::kMaxRadiusPx, 4000))
            .ok());
  // A moved pass keeps its filter.
  auto made =
      sensor::GpuFramePrep::create(device, allocator, filtering(8.0f, 20));
  CHECK(made.ok());
  sensor::GpuFramePrep moved = std::move(made).value();
  const Scene s = scene(pinhole());
  const std::vector<float> d = prepared(moved, frame_of(s.raw, pinhole()));
  CHECK(!d.empty() && d[50] == 0.0f);  // pixel 50 is an isolated return
  CHECK(s.part[50] == Part::Isolated);
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  if (test_scene(gpu.device, gpu.allocator) != 0) return 1;
  if (test_slab(gpu.device, gpu.allocator) != 0) return 1;
  if (test_ties(gpu.device, gpu.allocator) != 0) return 1;
  if (test_cap(gpu.device, gpu.allocator) != 0) return 1;
  if (test_after_color(gpu.device, gpu.allocator) != 0) return 1;
  if (test_batch(gpu.device, gpu.allocator) != 0) return 1;
  if (test_config(gpu.device, gpu.allocator) != 0) return 1;
  std::puts("sensor_gpu_frame_prep_outliers: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
