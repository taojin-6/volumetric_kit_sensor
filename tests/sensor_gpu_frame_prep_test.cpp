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
//   - its output fuses through the device-input overloads.
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/stage_metrics.hpp"
#include "volumetric_kit/recon/sensor/lens.hpp"
#include "volumetric_kit/recon/sensor/raw_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"
#include "volumetric_kit/recon/tsdf/tsdf_integrator.hpp"
#include "volumetric_kit/recon/volume/voxel_block_grid.hpp"
#include "volumetric_kit/recon/volume/voxel_hash_map.hpp"

namespace vr = volumetric_kit::recon;
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
vr::Device* g_device = nullptr;
vr::Allocator* g_allocator = nullptr;

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
std::vector<T> read(const vr::Buffer& buffer, std::size_t count) {
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

// Both passes against the host reference of the same sampling, on smooth
// planes and random depth, at w x h. `padded` gives the planes rows longer
// than their width, full of junk, as a decoder can hand them out.
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

// One picture, four ways: host I420, host NV12 with padded rows, and device
// planes in either layout at odd offsets with padded rows. They are only
// different addresses for the same samples, so all four come out identical.
// Device planes the pass cannot read are refused.
int test_layouts(sensor::GpuFramePrep& prep) {
  const auto invalid = vr::Status::Code::InvalidArgument;
  const std::uint32_t w = kWidth + 1, h = kHeight + 1;
  sensor::LensCamera cam = lensed();
  cam.width = w;
  cam.height = h;
  std::vector<std::uint16_t> raw(std::size_t{w} * h, 1500);
  Planes p = patterned(w, h);
  sensor::RawFrame f = frame_of(raw, cam);
  f.color_camera = cam;
  const sensor::YuvImage i420 = p.image(0.2126f, 0.0722f, false);
  f.color = i420;
  auto base = prep.prepare(f);
  CHECK(base.ok());
  const std::vector<std::uint32_t> want = color_of(base.value());
  CHECK(want.size() == std::size_t{w} * h);

  // NV12: each chroma pair interleaved, Cb first, every row padded.
  const std::size_t y_stride = w + 10, c_stride = 2 * p.cw + 6;
  std::vector<std::uint8_t> y_rows(y_stride * h, 0xCD);
  std::vector<std::uint8_t> cbcr(c_stride * p.ch, 0xCD);
  for (std::uint32_t y = 0; y < h; ++y) {
    std::memcpy(&y_rows[y * y_stride], &p.y[std::size_t{y} * w], w);
  }
  for (std::uint32_t y = 0; y < p.ch; ++y) {
    for (std::uint32_t x = 0; x < p.cw; ++x) {
      cbcr[y * c_stride + 2 * x] = p.cb[std::size_t{y} * p.cw + x];
      cbcr[y * c_stride + 2 * x + 1] = p.cr[std::size_t{y} * p.cw + x];
    }
  }
  sensor::YuvImage nv12 = i420;
  nv12.layout = sensor::YuvLayout::Nv12;
  nv12.plane[0] = y_rows.data();
  nv12.plane[1] = cbcr.data();
  nv12.plane[2] = nullptr;
  nv12.stride[0] = y_stride;
  nv12.stride[1] = c_stride;
  f.color = nv12;
  auto host_nv12 = prep.prepare(f);
  CHECK(host_nv12.ok());
  CHECK(color_of(host_nv12.value()) == want);

  // Device planes: one buffer holds the padded NV12 planes and the I420 ones,
  // each plane at an odd offset.
  const std::size_t c_i420 = p.cw + 3;
  std::vector<std::uint8_t> blob;
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
  const std::uint64_t i420_r = place(p.cr.data(), c_i420, p.ch, p.cw, p.cw);
  blob.resize((blob.size() + 3) & ~std::size_t{3}, 0xEE);
  auto made = vr::device_storage_buffer(*g_allocator, blob.size());
  CHECK(made.ok());
  CHECK(vr_test::write_back(*g_device, *g_allocator, made.value(), blob).ok());
  const auto planes =
      std::make_shared<const vr::Buffer>(std::move(made).value());

  sensor::YuvImage dev = i420;
  dev.plane[0] = dev.plane[1] = dev.plane[2] = nullptr;
  dev.device = planes;
  dev.layout = sensor::YuvLayout::Nv12;
  dev.offset[0] = nv12_y;
  dev.offset[1] = nv12_c;
  dev.stride[0] = y_stride;
  dev.stride[1] = c_stride;
  f.color = dev;
  auto dev_nv12 = prep.prepare(f);
  CHECK(dev_nv12.ok());
  CHECK(color_of(dev_nv12.value()) == want);

  sensor::YuvImage dev_i420 = dev;
  dev_i420.layout = sensor::YuvLayout::I420;
  dev_i420.offset[0] = i420_y;
  dev_i420.offset[1] = i420_b;
  dev_i420.offset[2] = i420_r;
  dev_i420.stride[1] = dev_i420.stride[2] = c_i420;
  f.color = dev_i420;
  auto dev_planar = prep.prepare(f);
  CHECK(dev_planar.ok());
  CHECK(color_of(dev_planar.value()) == want);

  // Refused: host and device planes at once, a plane past the buffer, Cb and
  // Cr rows of different lengths, a row shorter than its picture, and a
  // buffer the kernel cannot bind.
  sensor::YuvImage bad = dev;
  bad.plane[0] = p.y.data();
  f.color = bad;
  CHECK(prep.prepare(f).status().domain() == invalid);
  bad = dev;
  bad.offset[1] = planes->size() - c_stride;  // the chroma's last rows past it
  f.color = bad;
  CHECK(prep.prepare(f).status().domain() == invalid);
  bad = dev_i420;
  bad.stride[2] = c_i420 + 1;
  f.color = bad;
  CHECK(prep.prepare(f).status().domain() == invalid);
  bad = dev;
  bad.stride[0] = w - 1;
  f.color = bad;
  CHECK(prep.prepare(f).status().domain() == invalid);
  vr::BufferDesc desc;
  desc.size = planes->size();
  desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto unbindable = g_allocator->create_buffer(desc);
  CHECK(unbindable.ok());
  bad = dev;
  bad.device =
      std::make_shared<const vr::Buffer>(std::move(unbindable).value());
  f.color = bad;
  CHECK(prep.prepare(f).status().domain() == invalid);
  std::printf("  layouts: host I420 and NV12, device I420 and NV12 agree\n");
  return 0;
}

// prepare_set runs each camera's pass on its own thread. Each frame comes out
// as a pass of its own alone makes it, an empty slot stays empty, a refused
// frame fails the set, and too few passes are refused.
int test_prepare_set(vr::Device& device, vr::Allocator& allocator) {
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
        vr::Status::Code::InvalidArgument);
  std::vector<sensor::GpuFramePrep> too_few;
  too_few.push_back(std::move(alone).value());
  CHECK(sensor::prepare_set(too_few, frames).status().domain() ==
        vr::Status::Code::InvalidArgument);
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
  const auto invalid = vr::Status::Code::InvalidArgument;
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
  f.color_encoding.transfer = vr::ColorEncoding::Transfer::Bt2020Pq;
  CHECK(prep.prepare(f).status().domain() == vr::Status::Code::Unsupported);
  f.color_encoding = {};
  CHECK(prep.prepare(f).ok());

  // A colour half refused costs the depth half nothing either: both are
  // checked before anything is uploaded, so no pass is dispatched.
  vr::StageMetrics metrics;
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
  const vr::Buffer* held = nullptr;
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

// Allocate `depth`'s band, retrying rounds that only lost bucket-lock races,
// as examples/common/fuse_frame.hpp does: adjacent pixels dilate into one
// block, and a round can hand back such failures over a map far from full.
int allocate(vol::VoxelBlockGrid& grid, const vr::Buffer& depth,
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
int test_fuses(vr::Device& device, vr::Allocator& allocator,
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
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) {
    std::fprintf(stderr, "no Vulkan instance (%s); skipping\n",
                 instance.status().message().c_str());
    return 0;
  }
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) {
    std::fprintf(stderr, "no compute-capable device (%s); skipping\n",
                 gpu.status().message().c_str());
    return 0;
  }
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  CHECK(device.ok());
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
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
  if (test_layouts(prep.value()) != 0) return 1;
  if (test_coverage(prep.value()) != 0) return 1;
  if (test_refusals(prep.value()) != 0) return 1;
  if (test_frames_hold_buffers(prep.value()) != 0) return 1;
  if (test_fuses(device.value(), allocator.value(), prep.value()) != 0) {
    return 1;
  }
  if (test_prepare_set(device.value(), allocator.value()) != 0) return 1;

  sensor::GpuFramePrep moved = std::move(prep).value();
  CHECK(moved.valid());
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1);
  sensor::GpuFramePrep other = std::move(moved);
  CHECK(!moved.valid());  // NOLINT: moved from
  CHECK(moved.prepare(frame_of(raw, pinhole())).status().domain() ==
        vr::Status::Code::InvalidArgument);
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
