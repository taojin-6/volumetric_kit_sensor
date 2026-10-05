// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep on synthetic frames, no camera:
//   - a pinhole lens leaves depth exactly raw * scale, and colour the matrix's
//     inverse of a forward-converted RGB, within a code;
//   - a real lens undistorts, not distorts: a pattern drawn in the pinhole
//     image, then imaged through the lens by an independent iterative
//     inversion, comes back where it was drawn;
//   - both passes match a host reference of the same sampling, colour's
//     coverage byte included;
//   - the frames it refuses, before any work;
//   - a frame kept past the next one keeps its buffers' contents;
//   - its colour output is shared with the queue families its config names;
//   - with depth_within_color, depth survives only where colour recorded it;
//   - its output fuses through the device-input overloads.
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "test_image.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/lens.hpp"
#include "volumetric_kit/recon/sensor/raw_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"
#include "volumetric_kit/recon/tsdf/tsdf_integrator.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
namespace vol = volumetric_kit::recon::volume;
namespace tsdf = volumetric_kit::recon::tsdf;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 240;
constexpr float kScale = 0.001f;  // metres per unit

// Where the passes' device-local outputs are read back through.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

sensor::LensCamera pinhole() {
  sensor::LensCamera c;
  c.fx = 260.0f;
  c.fy = 258.0f;
  c.cx = 161.3f;
  c.cy = 118.7f;
  c.width = kWidth;
  c.height = kHeight;
  return c;
}

// A strong barrel lens with a little tangential and rational distortion.
sensor::LensCamera lensed() {
  sensor::LensCamera c = pinhole();
  c.lens.k1 = -0.28f;
  c.lens.k2 = 0.09f;
  c.lens.p1 = 0.0012f;
  c.lens.p2 = -0.0009f;
  c.lens.k3 = -0.01f;
  c.lens.k4 = 0.05f;
  return c;
}

// Where pinhole pixel (u, v) is imaged, by the host model.
vr::Vec2f source_pixel(const sensor::LensCamera& c, float u, float v) {
  const vr::Vec2f d = sensor::distort_normalized(
      c.lens, vr::Vec2f((u - c.cx) / c.fx, (v - c.cy) / c.fy));
  return vr::Vec2f(d.x * c.fx + c.cx, d.y * c.fy + c.cy);
}

// The pinhole point a captured pixel shows: the lens inverted by fixed-point
// iteration, independently of the passes' forward sampling.
vr::Vec2f undistort_pixel(const sensor::LensCamera& c, float us, float vs) {
  const vr::Vec2f target((us - c.cx) / c.fx, (vs - c.cy) / c.fy);
  vr::Vec2f p = target;
  for (int i = 0; i < 50; ++i) {
    p += target - sensor::distort_normalized(c.lens, p);
  }
  return vr::Vec2f(p.x * c.fx + c.cx, p.y * c.fy + c.cy);
}

// A checkerboard over the pinhole image, 20 px squares: 0 or 1.
int checker(vr::Vec2f pinhole_px) {
  return (static_cast<int>(std::floor(pinhole_px.x / 20.0f)) +
          static_cast<int>(std::floor(pinhole_px.y / 20.0f))) &
         1;
}

// How far a pinhole pixel is from the nearest checker edge, in pixels.
float edge_distance(vr::Vec2f p) {
  const float fx = std::fabs(p.x / 20.0f - std::round(p.x / 20.0f)) * 20.0f;
  const float fy = std::fabs(p.y / 20.0f - std::round(p.y / 20.0f)) * 20.0f;
  return std::min(fx, fy);
}

struct Planes {
  std::vector<std::uint8_t> y, cb, cr;
  std::uint32_t w = 0, h = 0;
  std::uint32_t cw = 0, ch = 0;
  sensor::YuvImage image(float kr, float kb, bool full) {
    sensor::YuvImage im;
    im.plane[0] = y.data();
    im.plane[1] = cb.data();
    im.plane[2] = cr.data();
    im.stride[0] = w;
    im.stride[1] = cw;
    im.stride[2] = cw;
    im.width = w;
    im.height = h;
    im.kr = kr;
    im.kb = kb;
    im.full_range = full;
    return im;
  }
};

Planes make_planes(std::uint32_t w = kWidth, std::uint32_t h = kHeight) {
  Planes p;
  p.w = w;
  p.h = h;
  p.cw = (w + 1) / 2;
  p.ch = (h + 1) / 2;
  p.y.resize(std::size_t{w} * h);
  p.cb.resize(std::size_t{p.cw} * p.ch);
  p.cr.resize(std::size_t{p.cw} * p.ch);
  return p;
}

// Y'CbCr from R'G'B' in [0, 1]: the forward matrix, independent of the pass's
// inverse.
void forward(float r, float g, float b, float kr, float kb, bool full,
             std::uint8_t* y, std::uint8_t* cb, std::uint8_t* cr) {
  const float luma = kr * r + (1.0f - kr - kb) * g + kb * b;
  const float pb = (b - luma) / (2.0f * (1.0f - kb));
  const float pr = (r - luma) / (2.0f * (1.0f - kr));
  const auto q = [](float v) {
    return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 255.0f)));
  };
  *y = q(full ? luma * 255.0f : 16.0f + luma * 219.0f);
  *cb = q(128.0f + pb * (full ? 255.0f : 224.0f));
  *cr = q(128.0f + pr * (full ? 255.0f : 224.0f));
}

// The pass's colour sampling, on the host: bilinear luma and chroma at their
// sitings, the inverse matrix, round to a code; coverage 0xFF in the high
// byte, and a 0 word where the lens maps outside the picture.
std::uint32_t reference_color(const Planes& p, const sensor::LensCamera& c,
                              float kr, float kb, bool full, float u, float v) {
  const vr::Vec2f s = source_pixel(c, u, v);
  if (!(s.x >= -0.5f && s.y >= -0.5f && s.x <= p.w - 0.5f &&
        s.y <= p.h - 0.5f)) {
    return 0;
  }
  const auto bilinear = [](const std::vector<std::uint8_t>& plane,
                           std::uint32_t w, std::uint32_t h, float x, float y) {
    x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
    const auto x0 = static_cast<std::uint32_t>(x);
    const auto y0 = static_cast<std::uint32_t>(y);
    const std::uint32_t x1 = std::min(x0 + 1, w - 1);
    const std::uint32_t y1 = std::min(y0 + 1, h - 1);
    const float fx = x - x0, fy = y - y0;
    const auto t = [&](std::uint32_t a, std::uint32_t b) {
      return static_cast<float>(plane[std::size_t{b} * w + a]);
    };
    const float top = t(x0, y0) + (t(x1, y0) - t(x0, y0)) * fx;
    const float bottom = t(x0, y1) + (t(x1, y1) - t(x0, y1)) * fx;
    return top + (bottom - top) * fy;
  };
  const float yv = bilinear(p.y, p.w, p.h, s.x, s.y);
  const float cbv = bilinear(p.cb, p.cw, p.ch, s.x * 0.5f, (s.y - 0.5f) * 0.5f);
  const float crv = bilinear(p.cr, p.cw, p.ch, s.x * 0.5f, (s.y - 0.5f) * 0.5f);
  const float luma = full ? yv / 255.0f : (yv - 16.0f) / 219.0f;
  const float pb = (cbv - 128.0f) / (full ? 255.0f : 224.0f);
  const float pr = (crv - 128.0f) / (full ? 255.0f : 224.0f);
  const float r = luma + 2.0f * (1.0f - kr) * pr;
  const float b = luma + 2.0f * (1.0f - kb) * pb;
  const float g = (luma - kr * r - kb * b) / (1.0f - kr - kb);
  const auto code = [](float x) {
    return static_cast<std::uint32_t>(std::clamp(x, 0.0f, 1.0f) * 255.0f +
                                      0.5f);
  };
  return code(r) | (code(g) << 8) | (code(b) << 16) | 0xFF000000u;
}

int channel_diff(std::uint32_t a, std::uint32_t b) {
  int worst = 0;
  for (int k = 0; k < 3; ++k) {
    const int ca = static_cast<int>((a >> (8 * k)) & 0xFF);
    const int cb = static_cast<int>((b >> (8 * k)) & 0xFF);
    worst = std::max(worst, std::abs(ca - cb));
  }
  return worst;
}

sensor::RawFrame frame_of(const std::vector<std::uint16_t>& depth,
                          const sensor::LensCamera& depth_cam) {
  sensor::RawFrame f;
  f.depth = depth.data();
  f.metres_per_unit = kScale;
  f.depth_camera = depth_cam;
  f.min_depth = 0.1f;
  f.max_depth = 10.0f;
  f.timestamp_ns = 1234;
  return f;
}

// A prepared frame's outputs, read back; empty if the copy failed.
template <typename T>
std::vector<T> read(const vkc::Buffer& buffer, std::size_t count) {
  auto out = vr_test::read_back<T>(*g_device, *g_allocator, buffer, count);
  if (!out) {
    std::fprintf(stderr, "read back: %s\n", out.status().message().c_str());
    return {};
  }
  return std::move(out).value();
}
std::vector<float> depth_of(const sensor::DeviceFrame& f) {
  return read<float>(*f.depth,
                     std::size_t{f.depth_camera.width} * f.depth_camera.height);
}
std::vector<std::uint32_t> color_of(const sensor::DeviceFrame& f) {
  return read<std::uint32_t>(
      *f.color, std::size_t{f.color_camera.width} * f.color_camera.height);
}

// Pinhole: depth is raw * scale exactly, and a colour survives the forward
// matrix and the pass's inverse to within a code, in both matrices and
// ranges.
int test_pinhole(sensor::GpuFramePrep& prep) {
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>((i * 7919u) % 6000u);
  }
  struct Case {
    float kr, kb;
    bool full;
  };
  for (const Case m :
       {Case{0.299f, 0.114f, true}, Case{0.2126f, 0.0722f, false}}) {
    Planes p = make_planes();
    // 8x8 solid tiles, one colour each, so chroma is constant under a tile.
    const auto rgb_of = [](std::uint32_t tx, std::uint32_t ty) {
      return vr::Vec3f(((tx * 37u) % 11u) / 10.0f, ((ty * 53u) % 13u) / 12.0f,
                       (((tx + ty) * 29u) % 7u) / 6.0f);
    };
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        const vr::Vec3f c = rgb_of(x / 8, y / 8);
        std::uint8_t yy, cb, cr;
        forward(c.x, c.y, c.z, m.kr, m.kb, m.full, &yy, &cb, &cr);
        p.y[std::size_t{y} * kWidth + x] = yy;
        if (x % 2 == 0 && y % 2 == 0) {
          p.cb[std::size_t{y / 2} * p.cw + x / 2] = cb;
          p.cr[std::size_t{y / 2} * p.cw + x / 2] = cr;
        }
      }
    }
    sensor::RawFrame f = frame_of(raw, pinhole());
    f.color = p.image(m.kr, m.kb, m.full);
    f.color_camera = pinhole();
    auto out = prep.prepare(f);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    CHECK(out.ok());
    CHECK(out->has_color() && out->timestamp_ns == 1234);
    CHECK(out->depth_camera.fx == 260.0f && out->depth_camera.width == kWidth);
    CHECK(out->depth_camera.min_depth == 0.1f);
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
      CHECK(d[i] == static_cast<float>(raw[i]) * kScale);
    }
    // Tile interiors: 2 px in, so no bilinear tap or chroma sample reaches
    // the next tile.
    const std::vector<std::uint32_t> c = color_of(out.value());
    CHECK(c.size() == raw.size());
    int worst = 0;
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        if (x % 8 < 2 || x % 8 > 5 || y % 8 < 2 || y % 8 > 5) continue;
        const vr::Vec3f want = rgb_of(x / 8, y / 8);
        const std::uint32_t packed =
            static_cast<std::uint32_t>(std::lround(want.x * 255)) |
            (static_cast<std::uint32_t>(std::lround(want.y * 255)) << 8) |
            (static_cast<std::uint32_t>(std::lround(want.z * 255)) << 16);
        worst = std::max(worst,
                         channel_diff(c[std::size_t{y} * kWidth + x], packed));
      }
    }
    if (worst > 2) std::fprintf(stderr, "pinhole colour off by %d\n", worst);
    CHECK(worst <= 2);
  }
  return 0;
}

// A real lens: a checkerboard drawn in the pinhole image and imaged through
// the lens (by inverting it) comes back where it was drawn, away from the
// squares' edges. Distorting instead of undistorting would move it by up to
// tens of pixels here.
int test_undistorts(sensor::GpuFramePrep& prep) {
  const sensor::LensCamera cam = lensed();
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  Planes p = make_planes();
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const int k = checker(
          undistort_pixel(cam, static_cast<float>(u), static_cast<float>(v)));
      raw[std::size_t{v} * kWidth + u] = k ? 2000 : 1000;
      p.y[std::size_t{v} * kWidth + u] = k ? 220 : 30;
    }
  }
  std::fill(p.cb.begin(), p.cb.end(), 128);
  std::fill(p.cr.begin(), p.cr.end(), 128);
  sensor::RawFrame f = frame_of(raw, cam);
  f.color = p.image(0.299f, 0.114f, true);
  f.color_camera = cam;
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const std::vector<float> d = depth_of(out.value());
  const std::vector<std::uint32_t> c = color_of(out.value());
  CHECK(d.size() == raw.size() && c.size() == raw.size());
  int checked = 0, wrong_depth = 0, wrong_color = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const vr::Vec2f px(static_cast<float>(u), static_cast<float>(v));
      const vr::Vec2f s = source_pixel(cam, px.x, px.y);
      if (edge_distance(px) < 2.0f || s.x < 1 || s.y < 1 || s.x > kWidth - 2 ||
          s.y > kHeight - 2) {
        continue;
      }
      ++checked;
      const int k = checker(px);
      if (d[std::size_t{v} * kWidth + u] != (k ? 2.0f : 1.0f)) ++wrong_depth;
      const int luma = static_cast<int>(c[std::size_t{v} * kWidth + u] & 0xFF);
      if (std::abs(luma - (k ? 220 : 30)) > 3) ++wrong_color;
    }
  }
  std::printf("  undistorted: %d pixels checked, %d depth and %d colour off\n",
              checked, wrong_depth, wrong_color);
  CHECK(checked > 40000);
  CHECK(wrong_depth <= checked / 1000);
  CHECK(wrong_color <= checked / 1000);
  // The corners map outside the captured image under this barrel lens? No:
  // barrel pulls them in. A pixel whose source is outside is 0 and black.
  return 0;
}

// A picture with structure in every plane, so a misread plane shows.
Planes patterned(std::uint32_t w, std::uint32_t h) {
  Planes p = make_planes(w, h);
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      p.y[std::size_t{y} * w + x] =
          static_cast<std::uint8_t>(128 + 100 * std::sin(0.05 * x + 0.03 * y));
    }
  }
  for (std::uint32_t y = 0; y < p.ch; ++y) {
    for (std::uint32_t x = 0; x < p.cw; ++x) {
      p.cb[std::size_t{y} * p.cw + x] =
          static_cast<std::uint8_t>(128 + 60 * std::cos(0.07 * x));
      p.cr[std::size_t{y} * p.cw + x] =
          static_cast<std::uint8_t>(128 + 60 * std::sin(0.09 * y));
    }
  }
  return p;
}

// Both passes against the host reference of the same sampling, on smooth
// planes and random depth, at w x h. `padded` gives the planes rows longer
// than their width, full of junk, as a decoder can hand them out.
int test_matches_reference(sensor::GpuFramePrep& prep, std::uint32_t w,
                           std::uint32_t h, bool padded) {
  sensor::LensCamera cam = lensed();
  cam.width = w;
  cam.height = h;
  std::vector<std::uint16_t> raw(std::size_t{w} * h);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>(500 + (i * 2654435761u) % 4000u);
  }
  Planes p = patterned(w, h);
  sensor::RawFrame f = frame_of(raw, cam);
  f.color = p.image(0.2126f, 0.0722f, false);
  f.color_camera = cam;
  std::vector<std::uint8_t> strided[3];
  if (padded) {
    const std::vector<std::uint8_t>* tight[3] = {&p.y, &p.cb, &p.cr};
    const std::uint32_t pw[3] = {w, p.cw, p.cw};
    const std::uint32_t ph[3] = {h, p.ch, p.ch};
    const std::uint32_t pad[3] = {7, 5, 3};
    for (int k = 0; k < 3; ++k) {
      const std::uint32_t stride = pw[k] + pad[k];
      strided[k].assign(std::size_t{stride} * ph[k], 0xAB);
      for (std::uint32_t y = 0; y < ph[k]; ++y) {
        std::memcpy(strided[k].data() + std::size_t{y} * stride,
                    tight[k]->data() + std::size_t{y} * pw[k], pw[k]);
      }
      f.color.plane[k] = strided[k].data();
      f.color.stride[k] = stride;
    }
  }
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const std::vector<float> d = depth_of(out.value());
  const std::vector<std::uint32_t> c = color_of(out.value());
  CHECK(d.size() == raw.size() && c.size() == raw.size());
  int depth_off = 0, color_worst = 0, color_off = 0, coverage_off = 0;
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) {
      const std::size_t i = std::size_t{v} * w + u;
      const vr::Vec2f s =
          source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
      const float rx = std::floor(s.x + 0.5f), ry = std::floor(s.y + 0.5f);
      float want = 0.0f;
      if (rx >= 0 && ry >= 0 && rx < w && ry < h) {
        want = static_cast<float>(raw[static_cast<std::size_t>(ry) * w +
                                      static_cast<std::size_t>(rx)]) *
               kScale;
      }
      if (d[i] != want) ++depth_off;
      const std::uint32_t ref =
          reference_color(p, cam, 0.2126f, 0.0722f, false,
                          static_cast<float>(u), static_cast<float>(v));
      if ((c[i] >> 24) != (ref >> 24)) ++coverage_off;
      const int diff = channel_diff(c[i], ref);
      color_worst = std::max(color_worst, diff);
      if (diff > 0) ++color_off;
    }
  }
  std::printf(
      "  reference %ux%u%s: depth %d differ, colour %d differ, worst %d\n", w,
      h, padded ? " padded" : "", depth_off, color_off, color_worst);
  // A source coordinate within float round-off of a pixel boundary may round
  // the other way on the device; nothing else may differ.
  CHECK(depth_off <= 20);
  CHECK(color_worst <= 1);
  CHECK(coverage_off == 0);
  return 0;
}

// One picture, many ways: host I420, host NV12 with tight and padded rows,
// device planes in either layout at odd offsets with padded rows, far
// enough into their buffer that the binding starts past byte 0, taken over
// from outside Vulkan or not, and with Cb and Cr rows of different lengths,
// and NV12's planes as images larger than the picture. They are only
// different addresses for the same samples, so all come out identical.
// Planes the pass cannot read are refused, before any work.
// Every siting samples the same continuous chroma ramps. At luma (8, 8),
// Y, Cb and Cr are all 128, so the prepared pixel must be neutral grey.
int test_chroma_locations(sensor::GpuFramePrep& prep) {
  using Location = sensor::ChromaLocation;
  struct Case {
    Location location;
    int dx;
    int dy;
  };
  const Case cases[] = {
      {Location::Left, 0, 5},        {Location::Center, 5, 5},
      {Location::TopLeft, 0, 0},     {Location::Top, 5, 0},
      {Location::BottomLeft, 0, 10}, {Location::Bottom, 5, 10}};
  auto cam = pinhole();
  cam.width = cam.height = 16;
  cam.fx = cam.fy = 16.0f;
  cam.cx = cam.cy = 8.0f;
  std::vector<std::uint16_t> raw(16 * 16, 1000);
  Planes p = make_planes(16, 16);
  std::fill(p.y.begin(), p.y.end(), 128);
  auto f = frame_of(raw, cam);
  f.color_camera = cam;
  for (const auto& c : cases) {
    for (std::uint32_t y = 0; y < 8; ++y) {
      for (std::uint32_t x = 0; x < 8; ++x) {
        p.cb[y * 8 + x] = static_cast<std::uint8_t>(48 + 20 * x + c.dx);
        p.cr[y * 8 + x] = static_cast<std::uint8_t>(48 + 20 * y + c.dy);
      }
    }
    f.color = p.image(0.299f, 0.114f, true);
    f.color.chroma_location = c.location;
    auto prepared = prep.prepare(f);
    CHECK(prepared.ok());
    const auto color = color_of(prepared.value());
    CHECK(color.size() == raw.size());
    CHECK(color[8 * 16 + 8] == 0xFF808080u);
  }
  return 0;
}

int test_layouts(sensor::GpuFramePrep& prep, sensor::ChromaLocation location) {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  const std::uint32_t w = kWidth + 1, h = kHeight + 1;
  sensor::LensCamera cam = lensed();
  cam.width = w;
  cam.height = h;
  std::vector<std::uint16_t> raw(std::size_t{w} * h, 1500);
  Planes p = patterned(w, h);
  sensor::RawFrame f = frame_of(raw, cam);
  f.color_camera = cam;
  sensor::YuvImage i420 = p.image(0.2126f, 0.0722f, false);
  i420.chroma_location = location;
  f.color = i420;
  auto base = prep.prepare(f);
  CHECK(base.ok());
  const std::vector<std::uint32_t> want = color_of(base.value());
  CHECK(want.size() == std::size_t{w} * h);
  const auto same = [&](const sensor::YuvImage& image) {
    f.color = image;
    auto prepared = prep.prepare(f);
    return prepared.ok() && color_of(prepared.value()) == want;
  };
  const auto refused = [&](const sensor::YuvImage& image,
                           const char* why = "") {
    f.color = image;
    const vkc::Status status = prep.prepare(f).status();
    return status.domain() == invalid &&
           status.message().find(why) != std::string::npos;
  };

  // NV12: each chroma pair interleaved, Cb first, every row padded.
  const std::size_t y_stride = w + 10, c_stride = 2 * p.cw + 6;
  std::vector<std::uint8_t> y_rows(y_stride * h, 0xCD);
  std::vector<std::uint8_t> cbcr(c_stride * p.ch, 0xCD);
  std::vector<std::uint8_t> cbcr_tight(2 * std::size_t{p.cw} * p.ch);
  for (std::uint32_t y = 0; y < h; ++y) {
    std::memcpy(&y_rows[y * y_stride], &p.y[std::size_t{y} * w], w);
  }
  for (std::uint32_t y = 0; y < p.ch; ++y) {
    for (std::uint32_t x = 0; x < p.cw; ++x) {
      for (std::uint32_t c = 0; c < 2; ++c) {
        const std::uint8_t v =
            (c == 0 ? p.cb : p.cr)[std::size_t{y} * p.cw + x];
        cbcr[y * c_stride + 2 * x + c] = v;
        cbcr_tight[(std::size_t{y} * p.cw + x) * 2 + c] = v;
      }
    }
  }
  sensor::YuvImage nv12 = i420;
  nv12.layout = sensor::YuvLayout::Nv12;
  nv12.plane[0] = y_rows.data();
  nv12.plane[1] = cbcr.data();
  nv12.plane[2] = nullptr;
  nv12.stride[0] = y_stride;
  nv12.stride[1] = c_stride;
  CHECK(same(nv12));
  // Tight rows: each plane is one copy into the staging.
  sensor::YuvImage nv12_tight = nv12;
  nv12_tight.plane[0] = p.y.data();
  nv12_tight.plane[1] = cbcr_tight.data();
  nv12_tight.stride[0] = w;
  nv12_tight.stride[1] = 2 * std::size_t{p.cw};
  CHECK(same(nv12_tight));

  // Device planes: one buffer holds the padded NV12 planes and the I420 ones,
  // each plane at an odd offset, all past a lead longer than any binding
  // alignment, so reading a plane from where it sits in the buffer rather
  // than in the binding misreads it. Cr's rows are longer than Cb's.
  const std::size_t c_i420 = p.cw + 3, cr_i420 = p.cw + 8;
  std::vector<std::uint8_t> blob(4096 + 5, 0xEE);
  const auto place = [&blob](const std::uint8_t* rows, std::size_t stride,
                             std::size_t count, std::size_t row_bytes,
                             std::size_t in_stride) {
    blob.resize(blob.size() + 3, 0xEE);  // odd padding before each plane
    const std::size_t at = blob.size();
    blob.resize(at + stride * count, 0xEE);
    for (std::size_t r = 0; r < count; ++r) {
      std::memcpy(&blob[at + r * stride], rows + r * in_stride, row_bytes);
    }
    return static_cast<std::uint64_t>(at);
  };
  const std::uint64_t nv12_y = place(y_rows.data(), y_stride, h, w, y_stride);
  const std::uint64_t nv12_c =
      place(cbcr.data(), c_stride, p.ch, 2 * p.cw, c_stride);
  const std::uint64_t i420_y = place(p.y.data(), y_stride, h, w, w);
  const std::uint64_t i420_b = place(p.cb.data(), c_i420, p.ch, p.cw, p.cw);
  const std::uint64_t i420_r = place(p.cr.data(), cr_i420, p.ch, p.cw, p.cw);
  blob.resize((blob.size() + 3) & ~std::size_t{3}, 0xEE);
  const auto on_device = [](const std::vector<std::uint8_t>& bytes)
      -> std::shared_ptr<const vkc::Buffer> {
    auto made = vkc::device_storage_buffer(*g_allocator, bytes.size());
    if (!made.ok() ||
        !vr_test::write_back(*g_device, *g_allocator, made.value(), bytes)
             .ok()) {
      return nullptr;
    }
    return std::make_shared<const vkc::Buffer>(std::move(made).value());
  };
  const auto planes = on_device(blob);
  CHECK(planes != nullptr);

  sensor::YuvImage dev = i420;
  dev.plane[0] = dev.plane[1] = dev.plane[2] = nullptr;
  dev.device = planes;
  dev.layout = sensor::YuvLayout::Nv12;
  dev.offset[0] = nv12_y;
  dev.offset[1] = nv12_c;
  dev.stride[0] = y_stride;
  dev.stride[1] = c_stride;
  CHECK(same(dev));
  // Taken over from CUDA, or named as the pass's own family: the same picture.
  sensor::YuvImage taken = dev;
  taken.queue_family = sensor::kQueueFamilyExternal;
  CHECK(same(taken));
  taken.queue_family = g_device->queue_family();
  CHECK(same(taken));

  sensor::YuvImage dev_i420 = dev;
  dev_i420.layout = sensor::YuvLayout::I420;
  dev_i420.offset[0] = i420_y;
  dev_i420.offset[1] = i420_b;
  dev_i420.offset[2] = i420_r;
  dev_i420.stride[1] = c_i420;
  dev_i420.stride[2] = cr_i420;
  CHECK(same(dev_i420));

  // The kernel reads whole words: an NV12 picture whose chroma ends two bytes
  // into its buffer's last word is refused while that word runs past the
  // buffer, and read once the buffer holds it, planes where they were.
  std::vector<std::uint8_t> ragged(y_rows);
  const std::size_t c_bytes = c_stride * (p.ch - 1) + 2 * std::size_t{p.cw};
  std::size_t ragged_c = ragged.size();
  while ((ragged_c + c_bytes) % 4 != 2) ++ragged_c;
  ragged.resize(ragged_c, 0xEE);
  ragged.insert(ragged.end(), cbcr.begin(),
                cbcr.begin() + static_cast<std::ptrdiff_t>(c_bytes));
  sensor::YuvImage word = dev;
  word.offset[0] = 0;
  word.offset[1] = ragged_c;
  word.device = on_device(ragged);
  CHECK(word.device != nullptr && word.device->size() % 4 == 2);
  CHECK(refused(word));
  ragged.resize(ragged.size() + 2, 0xEE);
  word.device = on_device(ragged);
  CHECK(word.device != nullptr);
  CHECK(same(word));

  // Images, each wider and taller than its plane, the rest of it junk: the
  // pass copies the picture from their corner.
  const auto image_of = [](VkFormat format, std::uint32_t texel,
                           std::uint32_t iw, std::uint32_t ih,
                           const std::uint8_t* rows, std::uint32_t row_bytes,
                           std::uint32_t count, VkImageUsageFlags usage) {
    std::vector<std::uint8_t> texels(std::size_t{iw} * ih * texel, 0xEE);
    for (std::uint32_t r = 0; r < count; ++r) {
      std::memcpy(&texels[std::size_t{r} * iw * texel],
                  rows + std::size_t{r} * row_bytes, row_bytes);
    }
    auto made = test_image::make(*g_device, *g_allocator, format, iw, ih,
                                 texels, usage);
    return made.ok()
               ? std::make_shared<const vkc::Image>(std::move(made).value())
               : nullptr;
  };
  const VkImageUsageFlags src = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  const auto luma =
      image_of(VK_FORMAT_R8_UNORM, 1, w + 3, h + 2, p.y.data(), w, h, src);
  const auto chroma = image_of(VK_FORMAT_R8G8_UNORM, 2, p.cw + 1, p.ch + 1,
                               cbcr_tight.data(), 2 * p.cw, p.ch, src);
  CHECK(luma != nullptr && chroma != nullptr);
  sensor::YuvImage images = i420;
  images.plane[0] = images.plane[1] = images.plane[2] = nullptr;
  images.stride[0] = images.stride[1] = images.stride[2] = 0;
  images.layout = sensor::YuvLayout::Nv12;
  images.image[0] = luma;
  images.image[1] = chroma;
  CHECK(same(images));

  // Refused as images: beside host or device planes; I420; either one
  // missing; the two swapped; one smaller than its plane; one the pass
  // cannot copy, for its usage or its layout.
  sensor::YuvImage bad_images = images;
  bad_images.plane[0] = p.y.data();
  CHECK(refused(bad_images, "one of the three"));
  bad_images = images;
  bad_images.device = planes;
  CHECK(refused(bad_images, "one of the three"));
  bad_images = images;
  bad_images.layout = sensor::YuvLayout::I420;
  CHECK(refused(bad_images, "NV12's two"));
  bad_images = images;
  bad_images.image[1] = nullptr;
  CHECK(refused(bad_images, "NV12's two"));
  bad_images = images;
  bad_images.image[0] = nullptr;
  CHECK(refused(bad_images, "NV12's two"));
  bad_images = images;
  bad_images.image[0] = chroma;
  bad_images.image[1] = luma;
  CHECK(refused(bad_images, "at least the picture's size"));
  bad_images = images;
  bad_images.image[0] =
      image_of(VK_FORMAT_R8_UNORM, 1, w - 1, h, p.y.data(), w - 1, h, src);
  CHECK(refused(bad_images, "at least the picture's size"));
  bad_images = images;
  bad_images.image[1] = image_of(VK_FORMAT_R8G8_UNORM, 2, p.cw, p.ch,
                                 cbcr_tight.data(), 2 * p.cw, p.ch, 0);
  CHECK(refused(bad_images, "TRANSFER_SRC"));
  bad_images = images;
  vkc::ImageInfo shader_read = chroma->info();
  shader_read.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  bad_images.image[1] =
      std::make_shared<const vkc::Image>(shader_read, nullptr);  // borrowed
  CHECK(refused(bad_images, "TRANSFER_SRC_OPTIMAL"));

  // Refused: host and device planes at once, even a stale third one; an NV12
  // host picture with a third plane; a plane past the buffer; planes that
  // overlap, as I420's do with its offsets left at zero; a row shorter than
  // its picture; a buffer that is empty or the kernel cannot bind; and a
  // queue family the device does not have.
  sensor::YuvImage bad = dev;
  bad.plane[0] = p.y.data();
  CHECK(refused(bad));
  bad = dev_i420;
  bad.plane[2] = p.cr.data();
  CHECK(refused(bad));
  bad = nv12;
  bad.plane[2] = p.cr.data();
  CHECK(refused(bad));
  bad = dev;
  bad.offset[1] = planes->size() - c_stride;  // the chroma's last rows past it
  CHECK(refused(bad));
  bad = dev_i420;
  bad.offset[2] = i420_b;
  CHECK(refused(bad));
  bad = dev_i420;
  bad.offset[0] = bad.offset[1] = bad.offset[2] = 0;
  CHECK(refused(bad));
  bad = dev;
  bad.stride[0] = w - 1;
  CHECK(refused(bad));
  bad = dev;
  bad.device = std::make_shared<const vkc::Buffer>();
  CHECK(refused(bad, "is empty"));
  vkc::BufferDesc desc;
  desc.size = planes->size();
  desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto unbindable = g_allocator->create_buffer(desc);
  CHECK(unbindable.ok());
  bad = dev;
  bad.device =
      std::make_shared<const vkc::Buffer>(std::move(unbindable).value());
  CHECK(refused(bad, "not a storage buffer"));
  std::uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_device->physical_device(),
                                           &families, nullptr);
  bad = dev;
  bad.queue_family = families;
  f.color = bad;
  vkc::StageMetrics metrics;
  CHECK(prep.prepare(f, &metrics).status().domain() == invalid);
  CHECK(metrics.rows().size() == 1 && !metrics.rows()[0].has_gpu);
  std::printf(
      "  layouts: host I420 and NV12, device I420 and NV12, and NV12 images "
      "agree, from byte %llu of their buffer\n",
      static_cast<unsigned long long>(nv12_y));
  return 0;
}

// prepare_set runs each camera's pass on its own thread. Each frame comes out
// as a pass of its own alone makes it, an empty slot stays empty, a refused
// frame fails the set, and too few passes are refused.
int test_prepare_set(vkc::Device& device, vkc::Allocator& allocator) {
  constexpr std::size_t kCams = 4;
  std::vector<sensor::GpuFramePrep> preps;
  for (std::size_t c = 0; c < kCams; ++c) {
    auto made = sensor::GpuFramePrep::create(device, allocator);
    CHECK(made.ok());
    preps.push_back(std::move(made).value());
  }
  auto alone = sensor::GpuFramePrep::create(device, allocator);
  CHECK(alone.ok());
  const sensor::LensCamera cam = lensed();
  std::vector<std::vector<std::uint16_t>> raws(kCams);
  std::vector<Planes> planes(kCams);
  std::vector<std::optional<sensor::RawFrame>> frames(kCams);
  for (std::size_t c = 0; c < kCams; ++c) {
    raws[c].resize(std::size_t{kWidth} * kHeight);
    for (std::size_t i = 0; i < raws[c].size(); ++i) {
      raws[c][i] = static_cast<std::uint16_t>(500 + 300 * c + i % 97);
    }
    planes[c] = make_planes();
    for (std::size_t i = 0; i < planes[c].y.size(); ++i) {
      planes[c].y[i] = static_cast<std::uint8_t>(40 * c + i % 150);
    }
    std::fill(planes[c].cb.begin(), planes[c].cb.end(),
              static_cast<std::uint8_t>(100 + 10 * c));
    std::fill(planes[c].cr.begin(), planes[c].cr.end(),
              static_cast<std::uint8_t>(150 - 10 * c));
    frames[c] = frame_of(raws[c], cam);
    frames[c]->color = planes[c].image(0.2126f, 0.0722f, false);
    frames[c]->color_camera = cam;
  }
  frames[2].reset();  // a camera whose frame never arrived

  for (int round = 0; round < 10; ++round) {
    auto set = sensor::prepare_set(preps, frames);
    CHECK(set.ok());
    CHECK(set.value().size() == kCams && !set.value()[2]);
    for (std::size_t c = 0; c < kCams; ++c) {
      if (!frames[c]) continue;
      CHECK(set.value()[c].has_value());
      auto one = alone.value().prepare(*frames[c]);
      CHECK(one.ok());
      CHECK(depth_of(*set.value()[c]) == depth_of(one.value()));
      CHECK(color_of(*set.value()[c]) == color_of(one.value()));
    }
  }

  std::vector<std::optional<sensor::RawFrame>> refused = frames;
  refused[1]->depth = nullptr;
  CHECK(sensor::prepare_set(preps, refused).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  std::vector<sensor::GpuFramePrep> too_few;
  too_few.push_back(std::move(alone).value());
  CHECK(sensor::prepare_set(too_few, frames).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  return 0;
}

// A pincushion lens maps the pinhole image's corners outside the captured
// picture. There colour is a 0 word, coverage and all, so that fusion skips
// it rather than fusing black; everywhere else its coverage byte is 0xFF,
// black included.
int test_coverage(sensor::GpuFramePrep& prep) {
  sensor::LensCamera cam = pinhole();
  cam.lens.k1 = 0.3f;
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes p = make_planes();  // all zero: black at full range
  std::fill(p.cb.begin(), p.cb.end(), 128);
  std::fill(p.cr.begin(), p.cr.end(), 128);
  sensor::RawFrame f = frame_of(raw, cam);
  f.color = p.image(0.299f, 0.114f, true);
  f.color_camera = cam;
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const std::vector<std::uint32_t> c = color_of(out.value());
  CHECK(c.size() == raw.size());
  int outside = 0, inside = 0, wrong = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const std::uint32_t want =
          reference_color(p, cam, 0.299f, 0.114f, true, static_cast<float>(u),
                          static_cast<float>(v));
      const std::uint32_t got = c[std::size_t{v} * kWidth + u];
      (want == 0 ? outside : inside) += 1;
      if (want == 0 ? got != 0 : got != 0xFF000000u) ++wrong;
    }
  }
  std::printf("  coverage: %d pixels outside the picture, %d inside, %d off\n",
              outside, inside, wrong);
  CHECK(outside > 1000 && inside > 50000);
  CHECK(wrong <= 20);  // within round-off of the picture's edge
  CHECK(c[0] == 0u);   // a corner
  CHECK(c[std::size_t{kHeight / 2} * kWidth + kWidth / 2] == 0xFF000000u);
  return 0;
}

int test_refusals(sensor::GpuFramePrep& prep) {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes p = make_planes();

  sensor::RawFrame f = frame_of(raw, pinhole());
  f.depth = nullptr;
  CHECK(prep.prepare(f).status().domain() == invalid);
  f = frame_of(raw, pinhole());
  f.depth_camera.fx = 0.0f;
  CHECK(prep.prepare(f).status().domain() == invalid);
  f = frame_of(raw, pinhole());
  f.depth_camera.lens.k1 = NAN;
  CHECK(prep.prepare(f).status().domain() == invalid);
  f = frame_of(raw, pinhole());
  f.metres_per_unit = 0.0f;
  CHECK(prep.prepare(f).status().domain() == invalid);
  // The depth gate: 0 is "no return", so a range from 0 is refused, as are
  // RawFrame's unset zeros, an empty range and a NaN.
  const float ranges[][2] = {{0.0f, 5.0f}, {0.0f, 0.0f}, {2.0f, 2.0f},
                             {3.0f, 1.0f}, {NAN, 5.0f},  {0.1f, INFINITY}};
  for (const auto& range : ranges) {
    f = frame_of(raw, pinhole());
    f.min_depth = range[0];
    f.max_depth = range[1];
    CHECK(prep.prepare(f).status().domain() == invalid);
  }

  f = frame_of(raw, pinhole());
  f.color = p.image(0.299f, 0.114f, true);
  f.color_camera = pinhole();
  f.color_camera.width = kWidth / 2;  // the picture disagrees
  CHECK(prep.prepare(f).status().domain() == invalid);
  f.color_camera = pinhole();
  f.color.stride[1] = 4;  // shorter than a chroma row
  CHECK(prep.prepare(f).status().domain() == invalid);
  f.color = p.image(0.6f, 0.5f, true);  // kr + kb past 1
  CHECK(prep.prepare(f).status().domain() == invalid);
  f.color = p.image(0.299f, 0.114f, true);
  f.color.chroma_location = static_cast<sensor::ChromaLocation>(255);
  CHECK(prep.prepare(f).status().domain() == invalid);
  f.color = p.image(0.299f, 0.114f, true);
  f.color_encoding.transfer = vr::ColorEncoding::Transfer::Bt2020Pq;
  CHECK(prep.prepare(f).status().domain() == vkc::Status::Code::Unsupported);
  f.color_encoding = {};
  CHECK(prep.prepare(f).ok());

  // A colour half refused costs the depth half nothing either: both are
  // checked before anything is uploaded, so no pass is dispatched.
  vkc::StageMetrics metrics;
  f.color.stride[1] = 4;
  CHECK(prep.prepare(f, &metrics).status().domain() == invalid);
  CHECK(metrics.rows().size() == 1);
  CHECK(!metrics.rows()[0].has_gpu);
  return 0;
}

// A frame kept past the next prepare keeps its buffers' contents, and that
// prepare writes to new ones; once no frame holds them, they are reused.
int test_frames_hold_buffers(sensor::GpuFramePrep& prep) {
  std::vector<std::uint16_t> near(std::size_t{kWidth} * kHeight, 1000);
  std::vector<std::uint16_t> far(near.size(), 3000);
  const vkc::Buffer* held = nullptr;
  {
    auto first = prep.prepare(frame_of(near, pinhole()));
    CHECK(first.ok());
    auto second = prep.prepare(frame_of(far, pinhole()));
    CHECK(second.ok());
    CHECK(first->depth != second->depth);
    const std::vector<float> a = depth_of(first.value());
    const std::vector<float> b = depth_of(second.value());
    CHECK(a.size() == near.size() && b.size() == near.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i] == 1000.0f * kScale && b[i] == 3000.0f * kScale);
    }
    held = second->depth.get();
  }
  auto third = prep.prepare(frame_of(near, pinhole()));
  CHECK(third.ok() && third->depth.get() == held);
  return 0;
}

// One pass under `config`: a frame with colour prepares, its colour output
// carries `want` as its sharing mode, and its depth stays EXCLUSIVE.
int prepared_sharing(vkc::Device& device, vkc::Allocator& allocator,
                     const sensor::GpuFramePrepConfig& config,
                     const sensor::RawFrame& frame, VkSharingMode want) {
  auto prep = sensor::GpuFramePrep::create(device, allocator, config);
  if (!prep) std::fprintf(stderr, "%s\n", prep.status().message().c_str());
  CHECK(prep.ok());
  auto out = prep->prepare(frame);
  if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
  CHECK(out.ok() && out->has_color());
  CHECK(out->depth->sharing_mode() == VK_SHARING_MODE_EXCLUSIVE);
  CHECK(out->color->sharing_mode() == want);
  const std::vector<float> d = depth_of(out.value());
  CHECK(d.size() == std::size_t{kWidth} * kHeight && d[0] == 1000.0f * kScale);
  return 0;
}

// The colour output is shared with the queue families the config names, for a
// consumer on another queue: a second family makes it CONCURRENT, the pass's
// own family named twice collapses to EXCLUSIVE, as does no config, and a
// count past the array is refused. Depth is EXCLUSIVE throughout. The second
// family is skipped on a device that has only one.
int test_queue_families(vkc::Device& device, vkc::Allocator& allocator) {
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes planes = make_planes();
  sensor::RawFrame f = frame_of(raw, pinhole());
  f.color = planes.image(0.299f, 0.114f, true);
  f.color_camera = pinhole();

  const std::uint32_t own = device.queue_family();
  if (prepared_sharing(device, allocator, {}, f, VK_SHARING_MODE_EXCLUSIVE) !=
      0) {
    return 1;
  }
  sensor::GpuFramePrepConfig twice;
  twice.color_queue_families[0] = own;
  twice.color_queue_families[1] = own;
  twice.color_queue_family_count = 2;
  if (prepared_sharing(device, allocator, twice, f,
                       VK_SHARING_MODE_EXCLUSIVE) != 0) {
    return 1;
  }
  std::uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device.physical_device(),
                                           &family_count, nullptr);
  if (family_count > 1) {
    sensor::GpuFramePrepConfig two = twice;
    two.color_queue_families[1] = own == 0 ? 1 : 0;
    if (prepared_sharing(device, allocator, two, f,
                         VK_SHARING_MODE_CONCURRENT) != 0) {
      return 1;
    }
  } else {
    std::printf("  one queue family: CONCURRENT outputs not exercised\n");
  }
  sensor::GpuFramePrepConfig too_many;
  too_many.color_queue_family_count = vkc::BufferDesc::kMaxQueueFamilies + 1;
  CHECK(sensor::GpuFramePrep::create(device, allocator, too_many)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  return 0;
}

// GpuFramePrepConfig::depth_within_color keeps depth only where the colour
// camera recorded it.
//   - Same intrinsics and pose, a pincushion colour lens: each depth pixel
//     lands on the colour pixel it shares, so depth must be zeroed exactly
//     where the colour's coverage byte is 0 -- the corners the lens missed.
//   - A narrower colour camera 10 cm to the right: the depth kept is the
//     region a host projection puts inside its image. A mask that ignored the
//     pose would keep a region 39 px to one side of it.
//   - Off, and on a frame without colour, every depth pixel is kept.
int test_depth_within_color(vkc::Device& device, vkc::Allocator& allocator) {
  sensor::GpuFramePrepConfig on;
  on.depth_within_color = true;
  auto masked = sensor::GpuFramePrep::create(device, allocator, on);
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  CHECK(masked.ok() && plain.ok());
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes planes = make_planes();

  sensor::LensCamera pincushion = pinhole();
  pincushion.lens.k1 = 0.3f;
  sensor::RawFrame same = frame_of(raw, pinhole());
  same.color = planes.image(0.299f, 0.114f, true);
  same.color_camera = pincushion;
  auto out = masked->prepare(same);
  if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
  CHECK(out.ok());
  const std::vector<float> d = depth_of(out.value());
  const std::vector<std::uint32_t> c = color_of(out.value());
  CHECK(d.size() == raw.size() && c.size() == raw.size());
  std::size_t kept = 0, zeroed = 0, wrong = 0;
  for (std::size_t i = 0; i < d.size(); ++i) {
    const bool covered = (c[i] >> 24) != 0u;
    (covered ? kept : zeroed) += 1;
    if (covered ? d[i] != 1000.0f * kScale : d[i] != 0.0f) ++wrong;
  }
  std::printf("  depth within colour: %zu kept, %zu zeroed, %zu off\n", kept,
              zeroed, wrong);
  CHECK(wrong == 0);
  CHECK(zeroed > 1000 && kept > 50000);

  // Narrower, and 10 cm to the right: the point moves by -0.1 m in x.
  sensor::LensCamera narrow = pinhole();
  narrow.fx *= 1.5f;
  narrow.fy *= 1.5f;
  sensor::RawFrame aside = frame_of(raw, pinhole());
  aside.color = planes.image(0.299f, 0.114f, true);
  aside.color_camera = narrow;
  aside.color_cam_to_world[3] = vr::Vec4f(0.1f, 0.0f, 0.0f, 1.0f);
  out = masked->prepare(aside);
  CHECK(out.ok());
  const std::vector<float> e = depth_of(out.value());
  const sensor::LensCamera dc = pinhole();
  std::size_t inside = 0, off = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const float x = (static_cast<float>(u) - dc.cx) / dc.fx - 0.1f;
      const float y = (static_cast<float>(v) - dc.cy) / dc.fy;
      const float pu = narrow.fx * x + narrow.cx + 0.5f;
      const float pv = narrow.fy * y + narrow.cy + 0.5f;
      // Within a hair of a rounding edge either answer is the float's.
      const float edge = std::fmin(std::fabs(pu - std::round(pu)),
                                   std::fabs(pv - std::round(pv)));
      if (edge < 1e-3f) continue;
      const bool in = std::floor(pu) >= 0.0f && std::floor(pu) < kWidth &&
                      std::floor(pv) >= 0.0f && std::floor(pv) < kHeight;
      inside += in ? 1 : 0;
      const float got = e[std::size_t{v} * kWidth + u];
      if (in ? got != 1000.0f * kScale : got != 0.0f) ++off;
    }
  }
  CHECK(off == 0);
  CHECK(inside > 1000 && inside < std::size_t{kWidth} * kHeight / 2);

  // Off, and a frame with no colour: nothing zeroed.
  const sensor::RawFrame depth_only = frame_of(raw, pinhole());
  for (sensor::GpuFramePrep* check : {&plain.value(), &masked.value()}) {
    auto all = check->prepare(check == &plain.value() ? aside : depth_only);
    CHECK(all.ok());
    for (const float z : depth_of(all.value())) CHECK(z == 1000.0f * kScale);
  }
  return 0;
}

// Allocate `depth`'s band, retrying rounds that only lost bucket-lock races,
// as examples/common/fuse_frame.hpp does: adjacent pixels dilate into one
// block, and a round can hand back such failures over a map far from full.
int allocate(vol::VoxelBlockGrid& grid, const vkc::Buffer& depth,
             const vr::DepthCameraParams& camera) {
  for (int round = 0; round < 5; ++round) {
    vol::AllocFailures why;
    auto failed = grid.map().allocate_from_depth(depth, camera, &why);
    CHECK(failed.ok() && !why.capacity_limited());
    if (failed.value() == 0) return 0;
  }
  std::fprintf(stderr, "FAIL: allocation kept losing lock races\n");
  return 1;
}

// The pass's output straight into the device-input fusion overloads.
int test_fuses(vkc::Device& device, vkc::Allocator& allocator,
               sensor::GpuFramePrep& prep) {
  // A small depth camera, 80x60, beside the full colour one: the two sizes
  // differ, as a sensor's do. A tilted surface, 0.7-1.2 m, spreads the band
  // over many blocks. (A 320x240 flat wall here once ran past NVIDIA's 7 s
  // watchdog in CI, Xid 109, contending for the hash map's bucket locks while
  // they were host-visible; PR #81 moved them into device memory.)
  sensor::LensCamera depth_cam = lensed();
  depth_cam.fx /= 4.0f;
  depth_cam.fy /= 4.0f;
  depth_cam.cx = (depth_cam.cx + 0.5f) / 4.0f - 0.5f;
  depth_cam.cy = (depth_cam.cy + 0.5f) / 4.0f - 0.5f;
  depth_cam.width = kWidth / 4;
  depth_cam.height = kHeight / 4;
  std::vector<std::uint16_t> raw(std::size_t{depth_cam.width} *
                                 depth_cam.height);
  for (std::uint32_t v = 0; v < depth_cam.height; ++v) {
    for (std::uint32_t u = 0; u < depth_cam.width; ++u) {
      raw[std::size_t{v} * depth_cam.width + u] =
          static_cast<std::uint16_t>(700 + 5 * u + 3 * v);
    }
  }
  Planes p = make_planes();
  std::fill(p.y.begin(), p.y.end(), 180);
  std::fill(p.cb.begin(), p.cb.end(), 100);
  std::fill(p.cr.begin(), p.cr.end(), 150);
  sensor::RawFrame f = frame_of(raw, depth_cam);
  f.color = p.image(0.299f, 0.114f, true);
  f.color_camera = pinhole();
  auto out = prep.prepare(f);
  CHECK(out.ok());

  vol::VoxelGridParams grid{};
  grid.voxel_size = 0.01f;
  grid.block_size = 8;
  grid.voxels_per_block = 512;
  grid.trunc_dist = 0.04f;
  grid.bucket_size = 8;
  grid.num_buckets = 2048;
  grid.num_blocks = 2048 * 8;
  grid.max_chain = 128;
  const vol::AttributeSpec attrs[] = {{"tsdf", sizeof(float)},
                                      {"weight", sizeof(float)},
                                      {"color", sizeof(std::uint32_t)}};
  auto vbg = vol::VoxelBlockGrid::create(device, allocator, grid, attrs, 3);
  auto integrator = tsdf::TsdfIntegrator::create(device, allocator);
  CHECK(vbg.ok() && integrator.ok());
  CHECK(allocate(vbg.value(), *out->depth, out->depth_camera) == 0);
  tsdf::ColorFrame color{};
  color.buffer = out->color.get();
  color.cam = out->color_camera;
  color.encoding = out->color_encoding;
  color.coverage_in_alpha = true;
  CHECK(integrator
            ->integrate(vbg.value(), *out->depth, out->depth_camera, 5.0f,
                        tsdf::IntegrationMode::Classic, &color)
            .ok());
  auto active = vbg->map().compact_active_blocks();
  CHECK(active.ok() && active->size() > 20);
  return 0;
}

}  // namespace

int main() {
  vkc::Result<vkc::Instance> instance = vkc::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vkc::Result<vkc::PhysicalDeviceInfo> gpu =
      instance.value().select_physical_device(vr::device_requirements());
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vkc::Result<vkc::Device> device = vkc::Device::create(
      instance.value(), gpu.value(), vr::device_requirements());
  CHECK(device.ok());
  vkc::Result<vkc::Allocator> allocator =
      vkc::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  g_device = &device.value();
  g_allocator = &allocator.value();
  auto prep = sensor::GpuFramePrep::create(device.value(), allocator.value());
  if (!prep) std::fprintf(stderr, "%s\n", prep.status().message().c_str());
  CHECK(prep.ok());

  if (test_pinhole(prep.value()) != 0) return 1;
  if (test_undistorts(prep.value()) != 0) return 1;
  if (test_matches_reference(prep.value(), kWidth, kHeight, false) != 0) {
    return 1;
  }
  // Odd, so the chroma planes' offsets are rounded up to a word.
  if (test_matches_reference(prep.value(), kWidth + 1, kHeight + 1, true) !=
      0) {
    return 1;
  }
  if (test_chroma_locations(prep.value()) != 0) return 1;
  for (auto location :
       {sensor::ChromaLocation::Left, sensor::ChromaLocation::Center}) {
    if (test_layouts(prep.value(), location) != 0) return 1;
  }
  if (test_coverage(prep.value()) != 0) return 1;
  if (test_refusals(prep.value()) != 0) return 1;
  if (test_frames_hold_buffers(prep.value()) != 0) return 1;
  if (test_fuses(device.value(), allocator.value(), prep.value()) != 0) {
    return 1;
  }
  if (test_prepare_set(device.value(), allocator.value()) != 0) return 1;
  if (test_queue_families(device.value(), allocator.value()) != 0) return 1;
  if (test_depth_within_color(device.value(), allocator.value()) != 0) return 1;

  sensor::GpuFramePrep moved = std::move(prep).value();
  CHECK(moved.valid());
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1);
  sensor::GpuFramePrep other = std::move(moved);
  CHECK(!moved.valid());  // NOLINT: moved from
  CHECK(moved.prepare(frame_of(raw, pinhole())).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(other.prepare(frame_of(raw, pinhole())).ok());
  sensor::GpuFramePrep* alias = &other;
  other = std::move(*alias);  // self-move
  CHECK(other.valid());
  // Move-assigned over a live pass, one that has buffers and a frame out.
  auto live = sensor::GpuFramePrep::create(device.value(), allocator.value());
  CHECK(live.ok());
  auto kept = live->prepare(frame_of(raw, pinhole()));
  CHECK(kept.ok());
  live.value() = std::move(other);
  CHECK(live->valid() && !other.valid());  // NOLINT: moved from
  CHECK(live->prepare(frame_of(raw, pinhole())).ok());
  CHECK(depth_of(kept.value()).size() == raw.size());  // outlives its pass

  std::puts("sensor_gpu_frame_prep: OK");
  return 0;
}
