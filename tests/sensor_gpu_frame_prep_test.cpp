// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep on synthetic frames, no camera:
//   - a pinhole lens leaves depth exactly raw * scale, and colour the matrix's
//     inverse of a forward-converted RGB, within a code;
//   - a real lens undistorts, not distorts: a pattern drawn in the pinhole
//     image, then imaged through the lens by an independent iterative
//     inversion, comes back where it was drawn;
//   - both passes match a host reference of the same sampling;
//   - the frames it refuses;
//   - its output fuses through the device-input overloads.
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/result.hpp"
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
  std::uint32_t cw = 0, ch = 0;
  sensor::YuvImage image(float kr, float kb, bool full) {
    sensor::YuvImage im;
    im.plane[0] = y.data();
    im.plane[1] = cb.data();
    im.plane[2] = cr.data();
    im.stride[0] = kWidth;
    im.stride[1] = cw;
    im.stride[2] = cw;
    im.width = kWidth;
    im.height = kHeight;
    im.kr = kr;
    im.kb = kb;
    im.full_range = full;
    return im;
  }
};

Planes make_planes() {
  Planes p;
  p.cw = (kWidth + 1) / 2;
  p.ch = (kHeight + 1) / 2;
  p.y.resize(std::size_t{kWidth} * kHeight);
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
// sitings, the inverse matrix, round to a code.
std::uint32_t reference_color(const Planes& p, const sensor::LensCamera& c,
                              float kr, float kb, bool full, float u, float v) {
  const vr::Vec2f s = source_pixel(c, u, v);
  if (!(s.x >= -0.5f && s.y >= -0.5f && s.x <= kWidth - 0.5f &&
        s.y <= kHeight - 0.5f)) {
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
  const float yv = bilinear(p.y, kWidth, kHeight, s.x, s.y);
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
  return code(r) | (code(g) << 8) | (code(b) << 16);
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

const float* depth_of(const sensor::DeviceFrame& f) {
  return static_cast<const float*>(f.depth->mapped());
}
const std::uint32_t* color_of(const sensor::DeviceFrame& f) {
  return static_cast<const std::uint32_t*>(f.color->mapped());
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
    const float* d = depth_of(out.value());
    for (std::size_t i = 0; i < raw.size(); ++i) {
      CHECK(d[i] == static_cast<float>(raw[i]) * kScale);
    }
    // Tile interiors: 2 px in, so no bilinear tap or chroma sample reaches
    // the next tile.
    const std::uint32_t* c = color_of(out.value());
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
  const float* d = depth_of(out.value());
  const std::uint32_t* c = color_of(out.value());
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
// planes and random depth.
int test_matches_reference(sensor::GpuFramePrep& prep) {
  const sensor::LensCamera cam = lensed();
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>(500 + (i * 2654435761u) % 4000u);
  }
  Planes p = make_planes();
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      p.y[std::size_t{y} * kWidth + x] =
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
  sensor::RawFrame f = frame_of(raw, cam);
  f.color = p.image(0.2126f, 0.0722f, false);
  f.color_camera = cam;
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const float* d = depth_of(out.value());
  const std::uint32_t* c = color_of(out.value());
  int depth_off = 0, color_worst = 0, color_off = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const std::size_t i = std::size_t{v} * kWidth + u;
      const vr::Vec2f s =
          source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
      const float rx = std::floor(s.x + 0.5f), ry = std::floor(s.y + 0.5f);
      float want = 0.0f;
      if (rx >= 0 && ry >= 0 && rx < kWidth && ry < kHeight) {
        want = static_cast<float>(raw[static_cast<std::size_t>(ry) * kWidth +
                                      static_cast<std::size_t>(rx)]) *
               kScale;
      }
      if (d[i] != want) ++depth_off;
      const int diff = channel_diff(
          c[i], reference_color(p, cam, 0.2126f, 0.0722f, false,
                                static_cast<float>(u), static_cast<float>(v)));
      color_worst = std::max(color_worst, diff);
      if (diff > 0) ++color_off;
    }
  }
  std::printf(
      "  reference: depth %d of %u differ, colour %d differ, worst %d\n",
      depth_off, kWidth * kHeight, color_off, color_worst);
  // A source coordinate within float round-off of a pixel boundary may round
  // the other way on the device; nothing else may differ.
  CHECK(depth_off <= 20);
  CHECK(color_worst <= 1);
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
  // differ, as a sensor's do. Small because the hash map's bucket locks are
  // host-visible until PR #81, and on NVIDIA 320x240 pixels contending for
  // them ran past the 7 s watchdog in CI (Xid 109). A tilted surface,
  // 0.7-1.2 m, spreads the band over many blocks.
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
  color.buffer = out->color;
  color.cam = out->color_camera;
  color.encoding = out->color_encoding;
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
  auto prep = sensor::GpuFramePrep::create(device.value(), allocator.value());
  if (!prep) std::fprintf(stderr, "%s\n", prep.status().message().c_str());
  CHECK(prep.ok());

  if (test_pinhole(prep.value()) != 0) return 1;
  if (test_undistorts(prep.value()) != 0) return 1;
  if (test_matches_reference(prep.value()) != 0) return 1;
  if (test_refusals(prep.value()) != 0) return 1;
  if (test_fuses(device.value(), allocator.value(), prep.value()) != 0) {
    return 1;
  }

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

  std::puts("sensor_gpu_frame_prep: OK");
  return 0;
}
