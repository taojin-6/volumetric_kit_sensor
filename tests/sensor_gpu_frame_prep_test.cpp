// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep on synthetic frames, no camera:
//   - a pinhole lens leaves depth exactly raw * scale, and colour the matrix's
//     inverse of a forward-converted RGB, within a code;
//   - a depth offset leaves depth exactly raw * scale less the offset, for
//     either sign, 0 where that leaves no return, and a hole a hole;
//   - a real lens undistorts, not distorts: a pattern drawn in the pinhole
//     image, then imaged through the lens by an independent iterative
//     inversion, comes back where it was drawn;
//   - depth mixes the four raw pixels around its source where they are one
//     surface: a tilted plane comes out as the plane, within 0.1 mm and with
//     no plateau, while a step past kBlendSpanM, or a pixel with no return
//     among the four, keeps the nearest raw pixel's depth, bit for bit;
//   - both passes match a host reference of the same sampling, colour's
//     coverage byte included;
//   - packed R'G'B' words from the host come through a pinhole as they went
//     in, and through a lens as the reference samples them;
//   - NV12's planes as images larger than the picture, sampled in place in
//     either layout the pass takes, taken over from outside Vulkan or not,
//     and I420 and NV12 planes on the host, give the words the same samples
//     in a buffer give, bit for bit;
//   - a camera's YUYV on the host matches the host reference of its 4:2:2
//     sampling through a lens, its words in a buffer give the same, and flat
//     colours come out as librealsense's own YUY2 conversion makes them;
//   - the frames it refuses, before any work;
//   - a frame kept past the next one keeps its buffers' contents;
//   - the driver's depth noise is handed on unchanged, alone and in a set,
//     each camera its own, with nothing else in the frame changed, and a
//     frame that states none hands on none;
//   - its depth and colour outputs are shared with the queue families its
//     config names for each, and read on another family as on its own;
//   - a frame with colour keeps depth only where colour recorded it, at the
//     corrected and mixed depth, and a frame without colour keeps all of it;
//   - and colour only inside its depth camera's field of view, however the
//     two cameras differ, and depth only inside the colour camera's, where
//     it lands on colour that is kept;
//   - depth on the device, at any even byte of its buffer, taken over or
//     not, alone or in a set beside host depth, gives the words the same
//     depth from the host gives, bit for bit, and holds the pass's slot;
//   - a buffer or image from another queue family is taken over, one barrier
//     each, and a buffer several depths and pictures share, once.
// Fusing its output is recon's test (recon_tsdf_prepared_frame).
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "test_image.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/two_family_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 240;
constexpr float kScale = 0.001f;  // metres per unit

// Where the passes' device-local outputs are read back through.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

// Every pass here has the depth-outlier filter off, so its depth is the depth
// pass's alone; the filter's test is sensor_gpu_frame_prep_outliers.
sensor::GpuFramePrepConfig unfiltered() {
  sensor::GpuFramePrepConfig config;
  config.depth_outliers.enabled = false;
  return config;
}

// The references below compute in float, as the passes do: each value
// narrowed from the camera's model, as the pass narrows it. The cameras are
// built from float literals, so the narrowing is exact.
float f32(double v) { return static_cast<float>(v); }

// The lens in float: camera::distort_rational, which lens.glsl mirrors.
sensor::Vec2f distort(const camera::RationalDistortion& d, sensor::Vec2f p) {
  const float k[8] = {f32(d.k1), f32(d.k2), f32(d.p1), f32(d.p2),
                      f32(d.k3), f32(d.k4), f32(d.k5), f32(d.k6)};
  sensor::Vec2f out;
  camera::distort_rational(k, p.x, p.y, &out.x, &out.y);
  return out;
}

camera::CameraModel pinhole() {
  camera::CameraModel c;
  c.size = {kWidth, kHeight};
  c.intrinsics = {260.0f, 258.0f, 161.3f, 118.7f};
  return c;
}

// A strong barrel lens with a little tangential and rational distortion.
camera::CameraModel lensed() {
  camera::CameraModel c = pinhole();
  c.distortion = {-0.28f, 0.09f, 0.0012f, -0.0009f, -0.01f, 0.05f, 0.0f, 0.0f};
  return c;
}

// Where pinhole pixel (u, v) is imaged, by the host model.
sensor::Vec2f source_pixel(const camera::CameraModel& c, float u, float v) {
  const float fx = f32(c.intrinsics.fx), fy = f32(c.intrinsics.fy);
  const float cx = f32(c.intrinsics.cx), cy = f32(c.intrinsics.cy);
  const sensor::Vec2f d =
      distort(c.distortion, sensor::Vec2f((u - cx) / fx, (v - cy) / fy));
  return sensor::Vec2f(d.x * fx + cx, d.y * fy + cy);
}

// The pinhole point a captured pixel shows: the lens inverted by fixed-point
// iteration, independently of the passes' forward sampling.
sensor::Vec2f undistort_pixel(const camera::CameraModel& c, float us,
                              float vs) {
  const float fx = f32(c.intrinsics.fx), fy = f32(c.intrinsics.fy);
  const float cx = f32(c.intrinsics.cx), cy = f32(c.intrinsics.cy);
  const sensor::Vec2f target((us - cx) / fx, (vs - cy) / fy);
  sensor::Vec2f p = target;
  for (int i = 0; i < 50; ++i) {
    p += target - distort(c.distortion, p);
  }
  return sensor::Vec2f(p.x * fx + cx, p.y * fy + cy);
}

// A checkerboard over the pinhole image, 20 px squares: 0 or 1.
int checker(sensor::Vec2f pinhole_px) {
  return (static_cast<int>(std::floor(pinhole_px.x / 20.0f)) +
          static_cast<int>(std::floor(pinhole_px.y / 20.0f))) &
         1;
}

// How far a pinhole pixel is from the nearest checker edge, in pixels.
float edge_distance(sensor::Vec2f p) {
  const float fx = std::fabs(p.x / 20.0f - std::round(p.x / 20.0f)) * 20.0f;
  const float fy = std::fabs(p.y / 20.0f - std::round(p.y / 20.0f)) * 20.0f;
  return std::min(fx, fy);
}

// `bytes` in a device storage buffer, as a decoder leaves its picture; null
// if it cannot be made.
std::shared_ptr<const vkc::Buffer> on_device(
    const std::vector<std::uint8_t>& bytes) {
  auto made = vkc::device_storage_buffer(*g_allocator, bytes.size());
  if (!made.ok() ||
      !vkc::test::write_back(*g_device, *g_allocator, made.value(), bytes)
           .ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

struct Planes {
  std::vector<std::uint8_t> y, cb, cr;
  std::uint32_t w = 0, h = 0;
  std::uint32_t cw = 0, ch = 0;
  // The picture as a decoder leaves it: I420 in a device buffer, each plane
  // after the last, its rows `pad[k]` bytes longer than its width and the
  // rest junk. Without colour if the buffer cannot be made.
  sensor::YuvImage image(float kr, float kb, bool full,
                         const std::uint32_t* pad = nullptr) const {
    const std::vector<std::uint8_t>* planes[3] = {&y, &cb, &cr};
    const std::uint32_t pw[3] = {w, cw, cw};
    const std::uint32_t ph[3] = {h, ch, ch};
    sensor::YuvImage im;
    std::vector<std::uint8_t> bytes;
    for (int k = 0; k < 3; ++k) {
      im.offset[k] = bytes.size();
      im.stride[k] = pw[k] + (pad != nullptr ? pad[k] : 0);
      bytes.resize(bytes.size() + im.stride[k] * ph[k], 0xAB);
      for (std::uint32_t r = 0; r < ph[k]; ++r) {
        std::memcpy(&bytes[im.offset[k] + r * im.stride[k]],
                    planes[k]->data() + std::size_t{r} * pw[k], pw[k]);
      }
    }
    bytes.resize((bytes.size() + 3) & ~std::size_t{3}, 0xAB);  // whole words
    im.device = on_device(bytes);
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

// Bilinear over a w x h byte plane, centres at integer coordinates, clamped
// to the edge, as the colour pass samples a plane.
float bilinear(const std::vector<std::uint8_t>& plane, std::uint32_t w,
               std::uint32_t h, float x, float y) {
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
}

// A value in [0, 1] as an 8-bit code, rounded as the colour pass rounds it.
std::uint32_t code(float x) {
  return static_cast<std::uint32_t>(std::clamp(x, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Sampled Y'CbCr through the pass's inverse matrix, each channel rounded to a
// code, with full coverage.
std::uint32_t reference_word(float yv, float cbv, float crv, float kr, float kb,
                             bool full) {
  const float luma = full ? yv / 255.0f : (yv - 16.0f) / 219.0f;
  const float pb = (cbv - 128.0f) / (full ? 255.0f : 224.0f);
  const float pr = (crv - 128.0f) / (full ? 255.0f : 224.0f);
  const float r = luma + 2.0f * (1.0f - kr) * pr;
  const float b = luma + 2.0f * (1.0f - kb) * pb;
  const float g = (luma - kr * r - kb * b) / (1.0f - kr - kb);
  return code(r) | (code(g) << 8) | (code(b) << 16) | 0xFF000000u;
}

// The pass's colour sampling, on the host: bilinear luma and chroma at their
// sitings, the inverse matrix, round to a code; coverage 0xFF in the high
// byte, and a 0 word where the lens maps outside the picture.
std::uint32_t reference_color(const Planes& p, const camera::CameraModel& c,
                              float kr, float kb, bool full, float u, float v) {
  const sensor::Vec2f s = source_pixel(c, u, v);
  if (!(s.x >= -0.5f && s.y >= -0.5f && s.x <= p.w - 0.5f &&
        s.y <= p.h - 0.5f)) {
    return 0;
  }
  const float yv = bilinear(p.y, p.w, p.h, s.x, s.y);
  const float cbv = bilinear(p.cb, p.cw, p.ch, s.x * 0.5f, (s.y - 0.5f) * 0.5f);
  const float crv = bilinear(p.cr, p.cw, p.ch, s.x * 0.5f, (s.y - 0.5f) * 0.5f);
  return reference_word(yv, cbv, crv, kr, kb, full);
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

// undistort_depth.comp's kBlendSpanM: the widest depth range across a block of
// four raw pixels that the depth pass still mixes.
constexpr float kBlendSpanM = 0.04f;

// The depth pass's sample at pinhole pixel (u, v), before the colour-view
// mask, on the host: each raw pixel scaled and less the offset, 0 no return;
// the bilinear mix of the 2x2 block around the lens's source position where
// all four return within kBlendSpanM, else the nearest raw pixel; 0 where the
// nearest pixel lies outside the image.
struct DepthSample {
  float z = 0.0f;        // what the pass writes, unmasked
  float nearest = 0.0f;  // the nearest raw pixel, as the pass sampled before
  bool mixed = false;    // z is the block's mix
  bool block = false;    // the block lies in the image
  float lo = 0.0f, hi = 0.0f;  // the block's range, when it does
};
DepthSample reference_depth(const std::vector<std::uint16_t>& raw,
                            const camera::CameraModel& cam, float scale,
                            float offset, std::uint32_t u, std::uint32_t v) {
  const std::uint32_t w = cam.size.width, h = cam.size.height;
  const auto metres = [&](float x, float y) {
    const float z = static_cast<float>(raw[static_cast<std::size_t>(y) * w +
                                           static_cast<std::size_t>(x)]) *
                    scale;
    return z > 0.0f ? std::max(z - offset, 0.0f) : 0.0f;
  };
  DepthSample out;
  const sensor::Vec2f s =
      source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
  const float rx = std::floor(s.x + 0.5f), ry = std::floor(s.y + 0.5f);
  if (!(rx >= 0 && ry >= 0 && rx < w && ry < h)) return out;
  out.nearest = out.z = metres(rx, ry);
  const float ox = std::floor(s.x), oy = std::floor(s.y);
  if (!(ox >= 0 && oy >= 0 && ox + 1 < w && oy + 1 < h)) return out;
  const float d00 = metres(ox, oy), d10 = metres(ox + 1, oy);
  const float d01 = metres(ox, oy + 1), d11 = metres(ox + 1, oy + 1);
  out.block = true;
  out.lo = std::min(std::min(d00, d10), std::min(d01, d11));
  out.hi = std::max(std::max(d00, d10), std::max(d01, d11));
  if (out.lo > 0.0f && out.hi - out.lo <= kBlendSpanM) {
    const float fx = s.x - ox, fy = s.y - oy;
    const float top = d00 + (d10 - d00) * fx;
    const float bottom = d01 + (d11 - d01) * fx;
    out.z = top + (bottom - top) * fy;
    out.mixed = true;
  }
  return out;
}

// Whether the device's mix matches the reference's. The operations are the
// same, but each side rounds the lens its own way, fusing multiply-adds or
// not, so the device's source position may sit up to about 1e-4 px from the
// host's: the mix may move by that fraction of the block's range, beside a
// few float steps at 3 m.
bool same_mix(float device, const DepthSample& host) {
  return std::fabs(device - host.z) <= 1e-6f + 1e-4f * (host.hi - host.lo);
}

// Counts a pixel whose depth `bad` says is wrong, printing the first few.
void count_wrong(bool bad, const char* test, std::uint32_t u, std::uint32_t v,
                 float got, const DepthSample& want, int* wrong) {
  if (!bad) return;
  if (++*wrong <= 3) {
    std::fprintf(stderr,
                 "  %s: pixel (%u, %u) is %.9g, the reference %.9g (nearest "
                 "%.9g, %s, block %.9g to %.9g)\n",
                 test, u, v, static_cast<double>(got),
                 static_cast<double>(want.z), static_cast<double>(want.nearest),
                 want.mixed ? "mixed" : "not mixed",
                 static_cast<double>(want.lo), static_cast<double>(want.hi));
  }
}

// Whether a source coordinate lies within float round-off of where the
// nearest pixel or the block changes, where the device and the host may
// each round it their own way.
bool near_rounding_edge(sensor::Vec2f s) {
  const auto edge = [](float x) {
    const float half = std::fabs(x - std::floor(x) - 0.5f);
    const float whole = std::fabs(x - std::round(x));
    return std::fmin(half, whole) < 1e-3f;
  };
  return edge(s.x) || edge(s.y);
}

// `count` packed words as a dataset decodes them, random in every byte: the
// high one is junk the pass must ignore.
std::vector<std::uint32_t> packed_words(std::size_t count, std::uint32_t seed) {
  std::vector<std::uint32_t> words(count);
  std::uint32_t x = seed;
  for (std::uint32_t& word : words) {
    x ^= x << 13;  // xorshift32
    x ^= x >> 17;
    x ^= x << 5;
    word = x;
  }
  return words;
}

sensor::RgbdFrame frame_of(const std::vector<std::uint16_t>& depth,
                           const camera::CameraModel& depth_cam) {
  sensor::RgbdFrame f;
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
  auto out = vkc::test::read_back<T>(*g_device, *g_allocator, buffer, count);
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

// The pipeline barriers preparing `frames` records, `preps[i]` preparing
// `frames[i]`: one more for each buffer or image it takes over from another
// queue family (DeviceCounters::barriers). A refused set counts as ~0.
std::uint64_t barriers_of(
    std::vector<sensor::GpuFramePrep>& preps,
    const std::vector<std::optional<sensor::RgbdFrame>>& frames) {
  const std::uint64_t was = g_device->counters().barriers;
  auto set = sensor::GpuFramePrep::prepare_batch(preps, frames);
  if (!set) {
    std::fprintf(stderr, "%s\n", set.status().message().c_str());
    return ~std::uint64_t{0};
  }
  return g_device->counters().barriers - was;
}
std::uint64_t barriers_of(sensor::GpuFramePrep& prep,
                          const sensor::RgbdFrame& frame) {
  const std::uint64_t was = g_device->counters().barriers;
  auto prepared = prep.prepare(frame);
  if (!prepared) {
    std::fprintf(stderr, "%s\n", prepared.status().message().c_str());
    return ~std::uint64_t{0};
  }
  return g_device->counters().barriers - was;
}

// What the pass needs of a plane image: SAMPLED usage, and with it a view.
// No TRANSFER_SRC: no copy reads it.
constexpr VkImageUsageFlags kSampled = VK_IMAGE_USAGE_SAMPLED_BIT;

// test_image::with_corner on the test's device.
std::shared_ptr<const vkc::Image> image_of(
    VkFormat format, std::uint32_t texel, std::uint32_t iw, std::uint32_t ih,
    const std::uint8_t* rows, std::uint32_t row_bytes, std::uint32_t count,
    VkImageUsageFlags usage, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
  return test_image::with_corner(*g_device, *g_allocator, format, texel, iw, ih,
                                 rows, row_bytes, count, usage, layout);
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
      return sensor::Vec3f(((tx * 37u) % 11u) / 10.0f,
                           ((ty * 53u) % 13u) / 12.0f,
                           (((tx + ty) * 29u) % 7u) / 6.0f);
    };
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        const sensor::Vec3f c = rgb_of(x / 8, y / 8);
        std::uint8_t yy, cb, cr;
        forward(c.x, c.y, c.z, m.kr, m.kb, m.full, &yy, &cb, &cr);
        p.y[std::size_t{y} * kWidth + x] = yy;
        if (x % 2 == 0 && y % 2 == 0) {
          p.cb[std::size_t{y / 2} * p.cw + x / 2] = cb;
          p.cr[std::size_t{y / 2} * p.cw + x / 2] = cr;
        }
      }
    }
    sensor::RgbdFrame f = frame_of(raw, pinhole());
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
        const sensor::Vec3f want = rgb_of(x / 8, y / 8);
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

// A power-of-two depth scale: raw * scale is exact, so the one rounding left
// in a corrected return is the subtraction's, whether or not the compiler
// fuses the two.
constexpr float kExactScale = 1.0f / 1024.0f;

// GpuFramePrepConfig::depth_offset_m: every return less the offset, exactly,
// for a sensor that reads too far (+4.17 mm), one that reads too near
// (-0.96 mm), and an offset equal to a return (4 units); a return the offset
// takes to 0 or below is no return, and no return stays one whatever the
// sign. A non-finite offset is refused at create.
int test_depth_offset(vkc::Device& device, vkc::Allocator& allocator) {
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>((i * 7919u) % 6000u);  // 0 among them
  }
  // Returns at, near and below each offset, and a hole.
  for (std::uint16_t u = 0; u <= 6; ++u) raw[u] = u;
  for (const float offset : {0.00417f, -0.00096f, 4.0f * kExactScale}) {
    sensor::GpuFramePrepConfig config = unfiltered();
    config.depth_offset_m = offset;
    auto prep = sensor::GpuFramePrep::create(device, allocator, config);
    if (!prep) std::fprintf(stderr, "%s\n", prep.status().message().c_str());
    CHECK(prep.ok());
    sensor::RgbdFrame f = frame_of(raw, pinhole());
    f.metres_per_unit = kExactScale;
    auto out = prep->prepare(f);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    CHECK(out.ok());
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == raw.size());
    std::size_t holes = 0, removed = 0, below = 0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const float z = static_cast<float>(raw[i]) * kExactScale;
      if (raw[i] == 0) {
        CHECK(d[i] == 0.0f);  // a hole stays one, for either sign
        ++holes;
        continue;
      }
      CHECK(d[i] == std::max(z - offset, 0.0f));
      removed += d[i] == 0.0f ? 1 : 0;
      below += z <= offset ? 1 : 0;
    }
    // Only a positive offset removes returns: those at or below it.
    CHECK(holes > 0 && removed == below);
    CHECK(offset > 0.0f ? removed >= 4 : removed == 0);
    // 4 units is at or below each positive offset, 5 above it.
    CHECK(d[4] == (offset > 0.0f ? 0.0f : 4.0f * kExactScale - offset));
    CHECK(d[5] == 5.0f * kExactScale - offset && d[5] > 0.0f);
  }
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {std::nanf(""), inf, -inf}) {
    sensor::GpuFramePrepConfig config;
    config.depth_offset_m = bad;
    const vkc::Status refused =
        sensor::GpuFramePrep::create(device, allocator, config).status();
    CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
    CHECK(refused.message().rfind("GpuFramePrep::create: depth_offset_m", 0) ==
          0);
  }
  return 0;
}

// A real lens: a checkerboard drawn in the pinhole image and imaged through
// the lens (by inverting it) comes back where it was drawn, away from the
// squares' edges. Distorting instead of undistorting would move it by up to
// tens of pixels here.
int test_undistorts(sensor::GpuFramePrep& prep) {
  const camera::CameraModel cam = lensed();
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
  sensor::RgbdFrame f = frame_of(raw, cam);
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
      const sensor::Vec2f px(static_cast<float>(u), static_cast<float>(v));
      const sensor::Vec2f s = source_pixel(cam, px.x, px.y);
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

// Depth in 0.1 mm units, fine enough that rounding to them stays well inside
// the plane test's 0.1 mm.
constexpr float kFineScale = 1e-4f;

// A plane 0.9-1.8 m away, turned about 24 degrees from facing the camera: its
// depth at normalized pinhole point (x, y).
float plane_depth(float x, float y) {
  return 1.2f / (1.0f + 0.4f * x + 0.2f * y);
}

// The plane as `cam` records it, in kFineScale units: each raw pixel holds
// the depth at the pinhole point the lens inversion puts it at.
std::vector<std::uint16_t> plane_raw(const camera::CameraModel& cam) {
  const std::uint32_t w = cam.size.width, h = cam.size.height;
  const float fx = f32(cam.intrinsics.fx), fy = f32(cam.intrinsics.fy);
  const float cx = f32(cam.intrinsics.cx), cy = f32(cam.intrinsics.cy);
  std::vector<std::uint16_t> raw(std::size_t{w} * h);
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) {
      const sensor::Vec2f p =
          undistort_pixel(cam, static_cast<float>(u), static_cast<float>(v));
      raw[std::size_t{v} * w + u] = static_cast<std::uint16_t>(std::lround(
          plane_depth((p.x - cx) / fx, (p.y - cy) / fy) / kFineScale));
    }
  }
  return raw;
}

// A tilted plane through the barrel lens comes out as the plane: within
// 0.1 mm of its depth at every pixel, the units' rounding and the mix's error
// together, and falling strictly along every row, with no plateau. The
// nearest raw pixel alone is millimetres off and repeats a raw pixel across
// thousands of neighbours, the terraces fusion shades as bands.
int test_tilted_plane(sensor::GpuFramePrep& prep) {
  const camera::CameraModel cam = lensed();
  const std::vector<std::uint16_t> raw = plane_raw(cam);
  sensor::RgbdFrame f = frame_of(raw, cam);
  f.metres_per_unit = kFineScale;
  auto out = prep.prepare(f);
  if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
  CHECK(out.ok());
  const std::vector<float> d = depth_of(out.value());
  CHECK(d.size() == raw.size());
  const float fx = f32(cam.intrinsics.fx), fy = f32(cam.intrinsics.fy);
  const float cx = f32(cam.intrinsics.cx), cy = f32(cam.intrinsics.cy);
  int checked = 0, mixed = 0, plateaus = 0, rises = 0, nearest_plateaus = 0;
  float worst = 0.0f, nearest_worst = 0.0f;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    bool previous = false;  // the pixel to the left was checked
    float previous_nearest = 0.0f;
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const std::size_t i = std::size_t{v} * kWidth + u;
      const DepthSample ref = reference_depth(raw, cam, kFineScale, 0.0f, u, v);
      if (!ref.block) {
        previous = false;
        continue;
      }
      ++checked;
      mixed += ref.mixed ? 1 : 0;
      const float truth = plane_depth((static_cast<float>(u) - cx) / fx,
                                      (static_cast<float>(v) - cy) / fy);
      worst = std::max(worst, std::fabs(d[i] - truth));
      nearest_worst = std::max(nearest_worst, std::fabs(ref.nearest - truth));
      if (previous) {
        plateaus += d[i] == d[i - 1] ? 1 : 0;
        rises += d[i] > d[i - 1] ? 1 : 0;
        nearest_plateaus += ref.nearest == previous_nearest ? 1 : 0;
      }
      previous = true;
      previous_nearest = ref.nearest;
    }
  }
  std::printf(
      "  tilted plane: %d pixels, %d mixed, worst %.3f mm (nearest %.3f mm), "
      "%d plateaus (nearest %d)\n",
      checked, mixed, worst * 1e3f, nearest_worst * 1e3f, plateaus,
      nearest_plateaus);
  CHECK(checked > 70000 && mixed == checked);
  CHECK(worst <= 1e-4f);
  CHECK(plateaus == 0 && rises == 0);
  // What the nearest pixel would have given, which these checks refuse.
  CHECK(nearest_worst > 2e-3f && nearest_plateaus > 1000);
  return 0;
}

// Two flat surfaces 1 m away meet along a slanted line of the raw image,
// stepping back by 3 cm, under kBlendSpanM, or 5 cm, over it, seen through the
// barrel lens. Where a pixel's block straddles the line, the smaller step is
// mixed, to a depth between the two, and the larger never is: the pixel is
// the nearest raw pixel's depth, bit for bit, one surface or the other. Every
// other block is flat and keeps its surface's depth exactly.
int test_depth_steps(sensor::GpuFramePrep& prep) {
  const camera::CameraModel cam = lensed();
  for (const std::uint16_t step : {std::uint16_t{30}, std::uint16_t{50}}) {
    const bool edge = static_cast<float>(step) * kScale > kBlendSpanM;
    std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        raw[std::size_t{y} * kWidth + x] =
            static_cast<std::uint16_t>(2 * x + y < 400 ? 1000 : 1000 + step);
      }
    }
    auto out = prep.prepare(frame_of(raw, cam));
    CHECK(out.ok());
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == raw.size());
    const float near_z = 1000.0f * kScale;
    const float far_z = static_cast<float>(1000 + step) * kScale;
    int straddling = 0, between = 0, wrong = 0, skipped = 0;
    for (std::uint32_t v = 0; v < kHeight; ++v) {
      for (std::uint32_t u = 0; u < kWidth; ++u) {
        const std::size_t i = std::size_t{v} * kWidth + u;
        if (near_rounding_edge(source_pixel(cam, static_cast<float>(u),
                                            static_cast<float>(v)))) {
          ++skipped;
          continue;
        }
        const DepthSample ref = reference_depth(raw, cam, kScale, 0.0f, u, v);
        between += d[i] > near_z && d[i] < far_z ? 1 : 0;
        if (!ref.block || ref.lo == ref.hi) {
          count_wrong(d[i] != ref.z, "step", u, v, d[i], ref, &wrong);
          continue;
        }
        ++straddling;
        count_wrong(edge ? ref.mixed || d[i] != ref.nearest
                         : !ref.mixed || !same_mix(d[i], ref),
                    "step", u, v, d[i], ref, &wrong);
      }
    }
    std::printf(
        "  %u mm step: %d blocks straddle it, %d pixels between, %d wrong, "
        "%d at a rounding edge\n",
        static_cast<unsigned>(step), straddling, between, wrong, skipped);
    CHECK(wrong == 0);
    CHECK(straddling > 200 && skipped < 1000);
    CHECK(edge ? between == 0 : between > straddling / 2);
  }
  return 0;
}

// A tap with no return is never mixed. A plane 2.5-3.2 cm away, near enough
// that 0 lies within kBlendSpanM of every return, so only the no-return rule
// keeps a hole out of the mix, with holes (0) and weak returns (2 mm)
// scattered through it. Every block with a hole is the nearest raw pixel's
// depth, bit for bit: 0 where that pixel is the hole, a return where not; the
// rest mix.
// A weak return is a return, and mixed, until a 4 mm depth offset takes it
// to 0, when it is a hole too.
int test_depth_holes(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cam = lensed();
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      std::uint16_t z = static_cast<std::uint16_t>(250 + (x + y) / 8);
      if ((x * 7 + y * 13) % 23 == 0) {
        z = 0;
      } else if ((x * 5 + y * 11) % 29 == 0) {
        z = 20;
      }
      raw[std::size_t{y} * kWidth + x] = z;
    }
  }
  for (const float offset : {0.0f, 0.004f}) {
    sensor::GpuFramePrepConfig config = unfiltered();
    config.depth_offset_m = offset;
    auto prep = sensor::GpuFramePrep::create(device, allocator, config);
    CHECK(prep.ok());
    sensor::RgbdFrame f = frame_of(raw, cam);
    f.metres_per_unit = kFineScale;
    f.min_depth = 0.001f;
    auto out = prep->prepare(f);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    CHECK(out.ok());
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == raw.size());
    int holed = 0, hole_nearest = 0, weak_mixed = 0, wrong = 0;
    for (std::uint32_t v = 0; v < kHeight; ++v) {
      for (std::uint32_t u = 0; u < kWidth; ++u) {
        const std::size_t i = std::size_t{v} * kWidth + u;
        if (near_rounding_edge(source_pixel(cam, static_cast<float>(u),
                                            static_cast<float>(v)))) {
          continue;
        }
        const DepthSample ref =
            reference_depth(raw, cam, kFineScale, offset, u, v);
        if (!ref.block) continue;
        if (ref.lo == 0.0f) {
          ++holed;
          hole_nearest += ref.nearest == 0.0f ? 1 : 0;
          count_wrong(ref.mixed || d[i] != ref.nearest, "hole", u, v, d[i], ref,
                      &wrong);
          continue;
        }
        count_wrong(!ref.mixed || !same_mix(d[i], ref), "hole", u, v, d[i], ref,
                    &wrong);
        weak_mixed += ref.lo < 0.005f ? 1 : 0;
      }
    }
    std::printf(
        "  holes, offset %.0f mm: %d blocks with a hole (%d the nearest "
        "pixel), %d weak returns mixed, %d wrong\n",
        static_cast<double>(offset) * 1e3, holed, hole_nearest, weak_mixed,
        wrong);
    CHECK(wrong == 0);
    CHECK(holed > 5000 && hole_nearest > 1000 && hole_nearest < holed);
    CHECK(offset > 0.0f ? weak_mixed == 0 : weak_mixed > 3000);
  }
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

// `p` as NV12 plane images, as VideoToolbox hands them out, which the pass
// samples; BT.709 limited range.
sensor::YuvImage nv12_images(const Planes& p) {
  std::vector<std::uint8_t> cbcr(2 * p.cb.size());
  for (std::size_t i = 0; i < p.cb.size(); ++i) {
    cbcr[2 * i] = p.cb[i];
    cbcr[2 * i + 1] = p.cr[i];
  }
  sensor::YuvImage images;
  images.layout = sensor::YuvLayout::Nv12;
  images.width = p.w;
  images.height = p.h;
  images.kr = 0.2126f;
  images.kb = 0.0722f;
  images.full_range = false;
  images.image[0] =
      image_of(VK_FORMAT_R8_UNORM, 1, p.w, p.h, p.y.data(), p.w, p.h, kSampled);
  images.image[1] = image_of(VK_FORMAT_R8G8_UNORM, 2, p.cw, p.ch, cbcr.data(),
                             2 * p.cw, p.ch, kSampled);
  return images;
}

// A w x h pinhole frame, its depth held by the frame, its colour patterned
// NV12 plane images (nv12_images).
sensor::RgbdFrame image_frame(std::uint32_t w, std::uint32_t h) {
  camera::CameraModel cam = pinhole();
  cam.size = {w, h};
  auto raw = std::make_shared<std::vector<std::uint16_t>>(std::size_t{w} * h,
                                                          std::uint16_t{1500});
  sensor::RgbdFrame f = frame_of(*raw, cam);
  f.pixels = raw;
  f.color_camera = cam;
  f.color = nv12_images(patterned(w, h));
  return f;
}

// Both passes against the host reference of the same sampling, on smooth
// planes and random depth, at w x h. `padded` gives the planes rows longer
// than their width, full of junk, as a decoder can hand them out. Depth
// alternates 16 px squares of a slope, which the pass mixes, and random
// returns, which it samples at the nearest pixel.
int test_matches_reference(sensor::GpuFramePrep& prep, std::uint32_t w,
                           std::uint32_t h, bool padded) {
  camera::CameraModel cam = lensed();
  cam.size = {w, h};
  std::vector<std::uint16_t> raw(std::size_t{w} * h);
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      const std::size_t i = std::size_t{y} * w + x;
      raw[i] = static_cast<std::uint16_t>(((x / 16 + y / 16) & 1u) != 0
                                              ? 500 + (i * 2654435761u) % 4000u
                                              : 1500 + 3 * x + 2 * y);
    }
  }
  Planes p = patterned(w, h);
  sensor::RgbdFrame f = frame_of(raw, cam);
  const std::uint32_t pad[3] = {7, 5, 3};
  f.color = p.image(0.2126f, 0.0722f, false, padded ? pad : nullptr);
  f.color_camera = cam;
  CHECK(f.has_color());
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const std::vector<float> d = depth_of(out.value());
  const std::vector<std::uint32_t> c = color_of(out.value());
  CHECK(d.size() == raw.size() && c.size() == raw.size());
  int depth_off = 0, mixed = 0, color_worst = 0, color_off = 0,
      coverage_off = 0;
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) {
      const std::size_t i = std::size_t{v} * w + u;
      const DepthSample want = reference_depth(raw, cam, kScale, 0.0f, u, v);
      mixed += want.mixed ? 1 : 0;
      if (want.mixed ? !same_mix(d[i], want) : d[i] != want.z) ++depth_off;
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
      "  reference %ux%u%s: depth %d differ (%d mixed), colour %d differ, "
      "worst %d\n",
      w, h, padded ? " padded" : "", depth_off, mixed, color_off, color_worst);
  // A source coordinate within float round-off of a pixel boundary may round
  // the other way on the device; nothing else may differ.
  CHECK(depth_off <= 20);
  CHECK(mixed > static_cast<int>(w * h / 4));
  CHECK(color_worst <= 1);
  CHECK(coverage_off == 0);
  return 0;
}

// Packed R'G'B' words on the host, as a dataset decodes them. Through a
// pinhole each comes back as it went in, its junk high byte replaced by full
// coverage, beside depth that is still raw * scale; through a lens, R, G and B
// are each sampled as the host reference samples a plane, and a pixel the lens
// maps outside the picture is a 0 word, as Y'CbCr colour's is.
int test_packed(sensor::GpuFramePrep& prep) {
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>((i * 7919u) % 6000u);
  }
  const std::vector<std::uint32_t> words = packed_words(raw.size(), 1);
  sensor::RgbdFrame f = frame_of(raw, pinhole());
  f.color_packed = words.data();
  f.color_camera = pinhole();
  CHECK(f.has_color());
  auto out = prep.prepare(f);
  if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
  CHECK(out.ok() && out->has_color());
  const std::vector<float> d = depth_of(out.value());
  const std::vector<std::uint32_t> c = color_of(out.value());
  CHECK(d.size() == raw.size() && c.size() == raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    CHECK(d[i] == static_cast<float>(raw[i]) * kScale);
    CHECK(c[i] == ((words[i] & 0xFFFFFFu) | 0xFF000000u));
  }

  camera::CameraModel cam = pinhole();
  cam.distortion.k1 = 0.3f;  // pincushion: the corners map outside
  f.color_camera = cam;
  out = prep.prepare(f);
  CHECK(out.ok());
  const std::vector<std::uint32_t> lens = color_of(out.value());
  CHECK(lens.size() == words.size());
  std::vector<std::uint8_t> channel[3];
  for (int k = 0; k < 3; ++k) {
    channel[k].resize(words.size());
    for (std::size_t i = 0; i < words.size(); ++i) {
      channel[k][i] = static_cast<std::uint8_t>(words[i] >> (8 * k));
    }
  }
  int worst = 0, outside = 0, coverage_off = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      const sensor::Vec2f s =
          source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
      std::uint32_t ref = 0;
      if (s.x >= -0.5f && s.y >= -0.5f && s.x <= kWidth - 0.5f &&
          s.y <= kHeight - 0.5f) {
        ref = 0xFF000000u;
        for (int k = 0; k < 3; ++k) {
          const float sample = bilinear(channel[k], kWidth, kHeight, s.x, s.y);
          ref |= code(sample / 255.0f) << (8 * k);
        }
      }
      const std::uint32_t got = lens[std::size_t{v} * kWidth + u];
      if (ref == 0) ++outside;
      if (ref == 0 ? got != 0 : (got >> 24) != 0xFFu) {
        ++coverage_off;
      } else if (ref != 0) {
        worst = std::max(worst, channel_diff(got, ref));
      }
    }
  }
  std::printf(
      "  packed through a lens: %d outside, worst %d, %d coverage off\n",
      outside, worst, coverage_off);
  CHECK(outside > 1000);
  CHECK(worst <= 1);
  CHECK(coverage_off <= 20);  // within round-off of the picture's edge
  return 0;
}

// One picture, many ways: planes in a buffer in either layout at odd offsets
// with padded rows, far enough into their buffer that the binding starts past
// byte 0, taken over from outside Vulkan or not, and with Cb and Cr rows of
// different lengths, and NV12's planes as images larger than the picture. They
// are only different addresses for the same samples, so all come out identical.
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
  cam.size = {16, 16};
  cam.intrinsics = {16.0f, 16.0f, 8.0f, 8.0f};
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
  camera::CameraModel cam = lensed();
  cam.size = {w, h};
  std::vector<std::uint16_t> raw(std::size_t{w} * h, 1500);
  Planes p = patterned(w, h);
  sensor::RgbdFrame f = frame_of(raw, cam);
  f.color_camera = cam;
  sensor::YuvImage i420 = p.image(0.2126f, 0.0722f, false);
  CHECK(i420.device != nullptr);
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
  const auto planes = on_device(blob);
  CHECK(planes != nullptr);

  sensor::YuvImage dev = i420;
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
  // And taken over: from CUDA, one barrier more than from the pass's own
  // family, the planes' buffer's acquire.
  f.color = dev;
  const std::uint64_t own_barriers = barriers_of(prep, f);
  f.color = taken;
  CHECK(barriers_of(prep, f) == own_barriers + 1);
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
  // The same planes on the host, as a camera's SDK hands them over: staged
  // beside the depth, from the first plane's first byte, and read alike.
  for (const sensor::YuvImage& form : {dev, dev_i420}) {
    sensor::YuvImage host = form;
    host.device = nullptr;
    host.queue_family = sensor::kQueueFamilyIgnored;
    host.host = blob.data();
    CHECK(same(host));
    // Beside planes in a buffer, or images, it is refused.
    host.device = planes;
    CHECK(refused(host, "one of them"));
  }
  // Host planes from byte 0 to the last byte 64 bits count, whose span a
  // word's rounding would wrap to nothing, are refused before any copy.
  sensor::YuvImage far = dev_i420;
  far.device = nullptr;
  far.host = blob.data();
  far.offset[0] = 0;
  far.offset[2] = std::numeric_limits<std::uint64_t>::max() -
                  ((std::uint64_t{p.ch} - 1) * cr_i420 + p.cw);
  CHECK(refused(far, "more than 4 GiB"));

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
  // pass samples the picture from their corner, in place, and its words are
  // the buffer's, bit for bit.
  const auto luma =
      image_of(VK_FORMAT_R8_UNORM, 1, w + 3, h + 2, p.y.data(), w, h, kSampled);
  const auto chroma = image_of(VK_FORMAT_R8G8_UNORM, 2, p.cw + 1, p.ch + 1,
                               cbcr_tight.data(), 2 * p.cw, p.ch, kSampled);
  CHECK(luma != nullptr && chroma != nullptr);
  sensor::YuvImage images = i420;
  images.device = nullptr;
  images.offset[0] = images.offset[1] = images.offset[2] = 0;
  images.stride[0] = images.stride[1] = images.stride[2] = 0;
  images.layout = sensor::YuvLayout::Nv12;
  images.image[0] = luma;
  images.image[1] = chroma;
  CHECK(same(images));
  // And in SHADER_READ_ONLY_OPTIMAL, the other layout the pass samples in,
  // which its sets name.
  const VkImageLayout read_only = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  sensor::YuvImage read_only_images = images;
  read_only_images.image[0] = image_of(VK_FORMAT_R8_UNORM, 1, w + 3, h + 2,
                                       p.y.data(), w, h, kSampled, read_only);
  read_only_images.image[1] =
      image_of(VK_FORMAT_R8G8_UNORM, 2, p.cw + 1, p.ch + 1, cbcr_tight.data(),
               2 * p.cw, p.ch, kSampled, read_only);
  CHECK(read_only_images.image[0] != nullptr &&
        read_only_images.image[1] != nullptr);
  CHECK(read_only_images.image[0]->layout() == read_only);
  CHECK(same(read_only_images));
  // Taken over from outside Vulkan, as NVDEC's arrive, in either layout, or
  // named as the pass's own family: the same picture.
  for (const sensor::YuvImage& form : {images, read_only_images}) {
    sensor::YuvImage taken_images = form;
    taken_images.queue_family = sensor::kQueueFamilyExternal;
    CHECK(same(taken_images));
    CHECK(taken_images.image[0]->layout() == form.image[0]->layout());
    // And taken over: two barriers more than from the pass's own family,
    // an acquire for each image.
    f.color = form;
    const std::uint64_t own_images = barriers_of(prep, f);
    f.color = taken_images;
    CHECK(barriers_of(prep, f) == own_images + 2);
    taken_images.queue_family = g_device->queue_family();
    CHECK(same(taken_images));
  }

  // Refused as images: beside planes in a buffer; I420; either one missing;
  // the two swapped; one smaller than its plane; one the pass cannot sample,
  // for its usage, its missing view, its layout or its shape.
  sensor::YuvImage bad_images = images;
  bad_images.device = planes;
  CHECK(refused(bad_images, "one of them"));
  bad_images = images;
  bad_images.host = blob.data();
  CHECK(refused(bad_images, "one of them"));
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
      image_of(VK_FORMAT_R8_UNORM, 1, w - 1, h, p.y.data(), w - 1, h, kSampled);
  CHECK(refused(bad_images, "at least the picture's size"));
  // What a copy needs, which the pass does not: TRANSFER_SRC, no view.
  bad_images = images;
  bad_images.image[1] =
      image_of(VK_FORMAT_R8G8_UNORM, 2, p.cw, p.ch, cbcr_tight.data(), 2 * p.cw,
               p.ch, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
  CHECK(bad_images.image[1] != nullptr);
  CHECK(refused(bad_images, "SAMPLED usage and a view"));
  bad_images = images;
  vkc::ImageInfo viewless = chroma->info();
  viewless.view = VK_NULL_HANDLE;
  bad_images.image[1] =
      std::make_shared<const vkc::Image>(viewless, nullptr);  // borrowed
  CHECK(refused(bad_images, "SAMPLED usage and a view"));
  // A view without SAMPLED usage, as the allocator's storage images have.
  bad_images = images;
  vkc::ImageInfo unsampled = chroma->info();
  unsampled.usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  CHECK(unsampled.view != VK_NULL_HANDLE);
  bad_images.image[1] =
      std::make_shared<const vkc::Image>(unsampled, nullptr);  // borrowed
  CHECK(refused(bad_images, "SAMPLED usage and a view"));
  bad_images = images;
  vkc::ImageInfo copied = chroma->info();
  copied.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  bad_images.image[1] =
      std::make_shared<const vkc::Image>(copied, nullptr);  // borrowed
  CHECK(refused(bad_images, "GENERAL or SHADER_READ_ONLY_OPTIMAL"));
  // Either plane in a shape whose view the pass's sampler2D does not match:
  // a 3D image of depth 1, two layers, or four samples.
  const auto reshaped = [&](int plane, void (*shape)(vkc::ImageInfo&)) {
    vkc::ImageInfo info = images.image[plane]->info();
    shape(info);
    sensor::YuvImage out = images;
    out.image[plane] =
        std::make_shared<const vkc::Image>(info, nullptr);  // borrowed
    return out;
  };
  // EXCLUSIVE images another family wrote, which cross only from outside
  // Vulkan, and images from a family the device lacks.
  if (const auto other = vkc::test::other_queue_family(*g_device)) {
    bad_images = images;
    bad_images.queue_family = *other;
    CHECK(refused(bad_images, "cross only from outside Vulkan"));
  }
  bad_images = images;
  bad_images.queue_family =
      static_cast<std::uint32_t>(g_device->caps().queue_families().size());
  CHECK(refused(bad_images, "which the device does not have"));
  for (int plane = 0; plane < 2; ++plane) {
    CHECK(refused(
        reshaped(plane, [](vkc::ImageInfo& i) { i.type = VK_IMAGE_TYPE_3D; }),
        "2D, of one layer and one sample"));
    CHECK(
        refused(reshaped(plane, [](vkc::ImageInfo& i) { i.array_layers = 2; }),
                "2D, of one layer and one sample"));
    CHECK(refused(
        reshaped(plane,
                 [](vkc::ImageInfo& i) { i.samples = VK_SAMPLE_COUNT_4_BIT; }),
        "2D, of one layer and one sample"));
  }

  // Refused: a plane past the buffer; planes that overlap, as I420's do with
  // its offsets left at zero; a stride shorter than the picture's rows; a
  // buffer that is empty or the kernel cannot bind; and a queue family the
  // device does not have.
  sensor::YuvImage bad = dev;
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
  CHECK(refused(bad, "stride is shorter"));
  bad = dev;
  bad.layout = static_cast<sensor::YuvLayout>(4);
  CHECK(refused(bad, "unknown colour layout"));
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
  // Refused with the frame's checks, before anything is recorded or timed.
  vkc::StageMetrics metrics;
  CHECK(prep.prepare(f, &metrics).status().domain() == invalid);
  CHECK(metrics.rows().empty());
  std::printf(
      "  layouts: I420 and NV12 in a buffer and NV12 images agree, from byte "
      "%llu of their buffer\n",
      static_cast<unsigned long long>(nv12_y));
  return 0;
}

// librealsense's YUY2 to RGB8 (BT.601, limited range, in integers), what its
// RGB8 stream would have been; the pass's float conversion is within 1 code
// of it over the whole legal range.
std::uint32_t librealsense_rgb(int y, int u, int v) {
  const int c = y - 16, d = u - 128, e = v - 128;
  const auto clamp = [](int x) {
    return static_cast<std::uint32_t>(std::clamp(x, 0, 255));
  };
  return clamp((298 * c + 409 * e + 128) >> 8) |
         (clamp((298 * c - 100 * d - 208 * e + 128) >> 8) << 8) |
         (clamp((298 * c + 516 * d + 128) >> 8) << 16) | 0xFF000000u;
}

// A camera's YUYV on the host, as librealsense hands a D400's colour over:
//   - through a lens, luma at every pixel and the 4:2:2 chroma, on every row
//     and sited left or centred across, match the host reference of that
//     sampling, rows padded, the picture at an odd offset from its pointer
//     and its buffer ending at its last byte, so a copy past it shows under
//     the sanitizers;
//   - its words in a device buffer give the same, bit for bit;
//   - its bytes need last only until record returns: freed before the batch
//     runs, the frame comes out the same;
//   - flat colours through a pinhole come out as librealsense's own YUY2
//     conversion makes them, within the 1 code its integers round to plus
//     the device's float rounding;
//   - YUYV of odd width, as images, or with rows shorter than its pixels is
//     refused.
int test_yuyv(sensor::GpuFramePrep& prep) {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  const std::uint32_t w = kWidth, h = kHeight, cw = kWidth / 2;
  std::vector<std::uint8_t> y(std::size_t{w} * h), cb(std::size_t{cw} * h),
      cr(std::size_t{cw} * h);
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t x = 0; x < w; ++x) {
      y[std::size_t{r} * w + x] =
          static_cast<std::uint8_t>(128 + 100 * std::sin(0.05 * x + 0.03 * r));
    }
    for (std::uint32_t x = 0; x < cw; ++x) {
      cb[std::size_t{r} * cw + x] =
          static_cast<std::uint8_t>(128 + 60 * std::cos(0.07 * x + 0.02 * r));
      cr[std::size_t{r} * cw + x] =
          static_cast<std::uint8_t>(128 + 60 * std::sin(0.09 * r + 0.01 * x));
    }
  }
  // Y0 Cb Y1 Cr, rows 6 bytes longer than the picture's, junk between them,
  // from byte 3 to the last row's last pixel, an odd count of bytes.
  const std::size_t lead = 3, stride = 2 * std::size_t{w} + 6;
  std::vector<std::uint8_t> bytes(lead + stride * (h - 1) + 2 * std::size_t{w},
                                  0xCD);
  CHECK(bytes.size() % 4 != 0);
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t i = 0; i < cw; ++i) {
      std::uint8_t* at = &bytes[lead + r * stride + 4 * std::size_t{i}];
      at[0] = y[std::size_t{r} * w + 2 * i];
      at[1] = cb[std::size_t{r} * cw + i];
      at[2] = y[std::size_t{r} * w + 2 * i + 1];
      at[3] = cr[std::size_t{r} * cw + i];
    }
  }
  sensor::YuvImage yuyv;
  yuyv.layout = sensor::YuvLayout::Yuyv;
  yuyv.host = bytes.data();
  yuyv.offset[0] = lead;
  yuyv.stride[0] = stride;
  yuyv.width = w;
  yuyv.height = h;
  yuyv.kr = 0.299f;
  yuyv.kb = 0.114f;
  yuyv.full_range = false;

  camera::CameraModel cam = lensed();
  cam.size = {w, h};
  std::vector<std::uint16_t> raw(std::size_t{w} * h, 1500);
  sensor::RgbdFrame f = frame_of(raw, cam);
  f.color_camera = cam;
  std::vector<std::uint32_t> got;  // sited left, as the D400's is
  for (const auto location :
       {sensor::ChromaLocation::Center, sensor::ChromaLocation::Left}) {
    f.color = yuyv;
    f.color.chroma_location = location;
    CHECK(f.has_color());
    auto out = prep.prepare(f);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    CHECK(out.ok());
    got = color_of(out.value());
    CHECK(got.size() == raw.size());
    // Chroma's first sample from luma's, across; on every row, down.
    const float across = sensor::chroma_offset(location)[0];
    int worst = 0, coverage_off = 0, inside = 0;
    for (std::uint32_t v = 0; v < h; ++v) {
      for (std::uint32_t u = 0; u < w; ++u) {
        const sensor::Vec2f s =
            source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
        std::uint32_t ref = 0;
        if (s.x >= -0.5f && s.y >= -0.5f && s.x <= w - 0.5f &&
            s.y <= h - 0.5f) {
          const float cx = (s.x - across) * 0.5f;
          ref = reference_word(
              bilinear(y, w, h, s.x, s.y), bilinear(cb, cw, h, cx, s.y),
              bilinear(cr, cw, h, cx, s.y), 0.299f, 0.114f, false);
          ++inside;
        }
        const std::uint32_t c = got[std::size_t{v} * w + u];
        if ((c >> 24) != (ref >> 24)) ++coverage_off;
        worst = std::max(worst, channel_diff(c, ref));
      }
    }
    std::printf(
        "  yuyv through a lens, chroma %s: %d inside, worst %d, %d coverage "
        "off\n",
        location == sensor::ChromaLocation::Left ? "left" : "centred", inside,
        worst, coverage_off);
    CHECK(inside > static_cast<int>(w * h / 2));
    CHECK(worst <= 1);
    CHECK(coverage_off == 0);
  }

  // The same words in a device buffer, as a capture could leave them.
  std::vector<std::uint8_t> words(bytes);
  words.resize((words.size() + 3) & ~std::size_t{3}, 0xCD);
  sensor::YuvImage dev = yuyv;
  dev.host = nullptr;
  dev.device = on_device(words);
  CHECK(dev.device != nullptr);
  f.color = dev;
  auto from_device = prep.prepare(f);
  CHECK(from_device.ok() && color_of(from_device.value()) == got);

  // Recorded from a copy of the bytes, which goes before the batch runs.
  {
    auto copy = std::make_unique<std::vector<std::uint8_t>>(bytes);
    f.color = yuyv;
    f.color.host = copy->data();
    vkc::CommandBatch batch(*g_device, *g_allocator);
    auto recorded = prep.record(batch, f);
    CHECK(recorded.ok() && recorded->has_color());
    f.color = yuyv;
    copy.reset();
    std::vector<std::uint32_t> color(raw.size());
    CHECK(batch
              .readback(*recorded->color, 0,
                        color.size() * sizeof(std::uint32_t), color.data())
              .ok());
    CHECK(batch.submit().ok());
    CHECK(color == got);
  }

  // Flat colours, one per 8 x 8 block, through a pinhole: every pixel of a
  // block's inner 4 x 4 samples its own colour alone.
  const int ys[] = {16, 60, 128, 200, 235};
  const int cs[][2] = {{128, 128}, {16, 128}, {240, 128}, {128, 16},
                       {128, 240}, {54, 34},  {90, 200},  {200, 60}};
  camera::CameraModel flat_cam = pinhole();
  const std::uint32_t blocks_x = kWidth / 8;
  std::vector<std::uint8_t> flat(stride * h, 0xCD);
  std::vector<std::uint32_t> want_rs;
  for (std::uint32_t b = 0; b < (kWidth / 8) * (kHeight / 8); ++b) {
    const int yv = ys[b % 5];
    const int u = cs[(b / 5) % 8][0], v = cs[(b / 5) % 8][1];
    want_rs.push_back(librealsense_rgb(yv, u, v));
    const std::uint32_t bx = b % blocks_x, by = b / blocks_x;
    for (std::uint32_t r = by * 8; r < by * 8 + 8; ++r) {
      for (std::uint32_t i = bx * 4; i < bx * 4 + 4; ++i) {
        std::uint8_t* at = &flat[r * stride + 4 * std::size_t{i}];
        at[0] = at[2] = static_cast<std::uint8_t>(yv);
        at[1] = static_cast<std::uint8_t>(u);
        at[3] = static_cast<std::uint8_t>(v);
      }
    }
  }
  sensor::RgbdFrame g = frame_of(raw, flat_cam);
  g.color = yuyv;
  g.color.host = flat.data();
  g.color.offset[0] = 0;
  g.color_camera = flat_cam;
  auto flat_out = prep.prepare(g);
  CHECK(flat_out.ok());
  const std::vector<std::uint32_t> flat_got = color_of(flat_out.value());
  CHECK(flat_got.size() == raw.size());
  int rs_worst = 0, rs_exact = 0, rs_count = 0;
  for (std::uint32_t b = 0; b < want_rs.size(); ++b) {
    const std::uint32_t bx = b % blocks_x, by = b / blocks_x;
    for (std::uint32_t r = by * 8 + 2; r < by * 8 + 6; ++r) {
      for (std::uint32_t x = bx * 8 + 2; x < bx * 8 + 6; ++x) {
        const int d =
            channel_diff(flat_got[std::size_t{r} * kWidth + x], want_rs[b]);
        rs_worst = std::max(rs_worst, d);
        rs_exact += d == 0 ? 1 : 0;
        ++rs_count;
      }
    }
  }
  std::printf(
      "  yuyv flat colours against librealsense's YUY2: %d of %d exact, "
      "worst %d\n",
      rs_exact, rs_count, rs_worst);
  CHECK(rs_worst <= 2);
  CHECK(rs_exact > rs_count / 2);

  const auto refused = [&](const sensor::YuvImage& image, const char* why) {
    f.color = image;
    const vkc::Status status = prep.prepare(f).status();
    if (status.domain() != invalid ||
        status.message().find(why) == std::string::npos) {
      std::fprintf(stderr, "%s\n", status.message().c_str());
      return false;
    }
    return true;
  };
  sensor::YuvImage bad = yuyv;
  bad.width = w - 1;
  f.color_camera.size.width = w - 1;
  CHECK(refused(bad, "width must be even"));
  f.color_camera = cam;
  bad = yuyv;
  bad.stride[0] = 2 * std::size_t{w} - 1;
  CHECK(refused(bad, "stride is shorter"));
  bad = yuyv;
  bad.host = nullptr;
  const auto luma =
      image_of(VK_FORMAT_R8_UNORM, 1, w, h, y.data(), w, h, kSampled);
  CHECK(luma != nullptr);
  bad.image[0] = bad.image[1] = luma;
  CHECK(refused(bad, "NV12's two"));
  return 0;
}

// Luma alone, a mono camera's, from the host, rows padded and from byte 1,
// through the barrel lens: grey, each word's R, G and B the luma the host
// reference samples (its range expanded where limited), within a code, the
// coverage byte exact; the same bytes in a device buffer give the same words,
// and so does a picture whose matrix and siting are unset. As images it is
// refused.
int test_luma(sensor::GpuFramePrep& prep) {
  const std::uint32_t w = kWidth, h = kHeight;
  std::vector<std::uint8_t> y(std::size_t{w} * h);
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t x = 0; x < w; ++x) {
      y[std::size_t{r} * w + x] =
          static_cast<std::uint8_t>(128 + 110 * std::sin(0.06 * x - 0.04 * r));
    }
  }
  const std::size_t lead = 1, stride = std::size_t{w} + 5;
  std::vector<std::uint8_t> bytes(lead + stride * h, 0xCD);
  for (std::uint32_t r = 0; r < h; ++r) {
    std::memcpy(&bytes[lead + r * stride], &y[std::size_t{r} * w], w);
  }
  sensor::YuvImage mono;
  mono.layout = sensor::YuvLayout::Luma;
  mono.host = bytes.data();
  mono.offset[0] = lead;
  mono.stride[0] = stride;
  mono.width = w;
  mono.height = h;

  const camera::CameraModel cam = lensed();
  std::vector<std::uint16_t> raw(std::size_t{w} * h, 1500);
  sensor::RgbdFrame f = frame_of(raw, cam);
  f.color_camera = cam;
  for (const bool full : {true, false}) {
    f.color = mono;
    f.color.full_range = full;
    auto out = prep.prepare(f);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    CHECK(out.ok());
    const std::vector<std::uint32_t> got = color_of(out.value());
    CHECK(got.size() == raw.size());
    int worst = 0, coverage_off = 0, not_grey = 0, inside = 0;
    for (std::uint32_t v = 0; v < h; ++v) {
      for (std::uint32_t u = 0; u < w; ++u) {
        const sensor::Vec2f s =
            source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
        std::uint32_t ref = 0;
        if (s.x >= -0.5f && s.y >= -0.5f && s.x <= w - 0.5f &&
            s.y <= h - 0.5f) {
          const float luma = bilinear(y, w, h, s.x, s.y);
          const std::uint32_t g =
              code(full ? luma / 255.0f : (luma - 16.0f) / 219.0f);
          ref = g | (g << 8) | (g << 16) | 0xFF000000u;
          ++inside;
        }
        const std::uint32_t c = got[std::size_t{v} * w + u];
        if ((c >> 24) != (ref >> 24)) ++coverage_off;
        if ((c & 0xFF) != ((c >> 8) & 0xFF) ||
            (c & 0xFF) != ((c >> 16) & 0xFF)) {
          ++not_grey;
        }
        worst = std::max(worst, channel_diff(c, ref));
      }
    }
    std::printf(
        "  luma alone through a lens, %s range: %d inside, worst %d, %d "
        "coverage off, %d not grey\n",
        full ? "full" : "limited", inside, worst, coverage_off, not_grey);
    CHECK(inside > static_cast<int>(w * h / 2));
    CHECK(worst <= 1 && coverage_off == 0 && not_grey == 0);
    if (!full) continue;

    // The same bytes in a device buffer.
    std::vector<std::uint8_t> words(bytes);
    words.resize((words.size() + 3) & ~std::size_t{3}, 0xCD);
    sensor::YuvImage dev = f.color;
    dev.host = nullptr;
    dev.device = on_device(words);
    CHECK(dev.device != nullptr);
    sensor::RgbdFrame g = f;
    g.color = dev;
    auto from_device = prep.prepare(g);
    CHECK(from_device.ok() && color_of(from_device.value()) == got);

    // Grey reads no matrix and no siting, so a mono driver may leave them
    // unset.
    sensor::RgbdFrame unset = f;
    unset.color.kr = unset.color.kb = 0.0f;
    unset.color.chroma_location = static_cast<sensor::ChromaLocation>(255);
    auto grey = prep.prepare(unset);
    if (!grey) std::fprintf(stderr, "%s\n", grey.status().message().c_str());
    CHECK(grey.ok() && color_of(grey.value()) == got);
  }

  sensor::YuvImage bad = mono;
  bad.host = nullptr;
  const auto plane =
      image_of(VK_FORMAT_R8_UNORM, 1, w, h, y.data(), w, h, kSampled);
  CHECK(plane != nullptr);
  bad.image[0] = bad.image[1] = plane;
  f.color = bad;
  const vkc::Status status = prep.prepare(f).status();
  CHECK(status.domain() == vkc::Status::Code::InvalidArgument &&
        status.message().find("NV12's two") != std::string::npos);
  return 0;
}

// prepare_batch prepares every camera's frame in one batch, each as a pass of
// its own makes it alone, whichever way its colour arrives -- planes in a
// buffer, taken over from outside Vulkan or not, NV12's planes as images, or
// packed words on the host, staged beside the depth -- with depth kept within
// colour; an empty slot stays empty. A set with a
// refused frame is refused before any work, with that frame's refusal, and
// too few passes are refused.
int test_prepare_batch(vkc::Device& device, vkc::Allocator& allocator) {
  constexpr std::size_t kCams = 5;
  std::vector<sensor::GpuFramePrep> preps;
  for (std::size_t c = 0; c < kCams; ++c) {
    auto made = sensor::GpuFramePrep::create(device, allocator, unfiltered());
    CHECK(made.ok());
    preps.push_back(std::move(made).value());
  }
  auto alone = sensor::GpuFramePrep::create(device, allocator, unfiltered());
  CHECK(alone.ok());
  // A pincushion colour lens misses the corners, which the mask zeroes.
  camera::CameraModel pincushion = pinhole();
  pincushion.distortion.k1 = 0.3f;
  std::vector<std::vector<std::uint16_t>> raws(kCams);
  std::vector<Planes> planes(kCams);
  std::vector<std::optional<sensor::RgbdFrame>> frames(kCams);
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
    frames[c] = frame_of(raws[c], lensed());
    frames[c]->color = planes[c].image(0.2126f, 0.0722f, false);
    frames[c]->color_camera = pincushion;
    // Each camera's own noise; camera 0 states none.
    if (c > 0) {
      frames[c]->depth_noise = {0.0f, 0.001f * static_cast<float>(c), 0.0f};
    }
  }
  frames[2].reset();  // a camera whose frame never arrived
  for (const auto& f : frames) CHECK(!f || f->has_color());

  // Camera 1's planes written by CUDA.
  frames[1]->color.queue_family = sensor::kQueueFamilyExternal;
  // Camera 3's as NV12 plane images, as VideoToolbox hands them out.
  const Planes& p3 = planes[3];
  std::vector<std::uint8_t> cbcr(2 * p3.cb.size());
  for (std::size_t i = 0; i < p3.cb.size(); ++i) {
    cbcr[2 * i] = p3.cb[i];
    cbcr[2 * i + 1] = p3.cr[i];
  }
  sensor::YuvImage& images = frames[3]->color;
  images.device = nullptr;
  images.offset[0] = images.offset[1] = images.offset[2] = 0;
  images.stride[0] = images.stride[1] = images.stride[2] = 0;
  images.layout = sensor::YuvLayout::Nv12;
  images.image[0] = image_of(VK_FORMAT_R8_UNORM, 1, p3.w, p3.h, p3.y.data(),
                             p3.w, p3.h, kSampled);
  images.image[1] = image_of(VK_FORMAT_R8G8_UNORM, 2, p3.cw, p3.ch, cbcr.data(),
                             2 * p3.cw, p3.ch, kSampled);
  CHECK(images.image[0] != nullptr && images.image[1] != nullptr);
  // Camera 4's as packed words on the host, as a dataset decodes them.
  const std::vector<std::uint32_t> words = packed_words(raws[4].size(), 3);
  frames[4]->color = {};
  frames[4]->color_packed = words.data();
  CHECK(frames[4]->has_color());

  for (int round = 0; round < 3; ++round) {
    auto set = sensor::GpuFramePrep::prepare_batch(preps, frames);
    if (!set) std::fprintf(stderr, "%s\n", set.status().message().c_str());
    CHECK(set.ok());
    CHECK(set.value().size() == kCams && !set.value()[2]);
    for (std::size_t c = 0; c < kCams; ++c) {
      if (!frames[c]) continue;
      CHECK(set.value()[c].has_value());
      auto one = alone.value().prepare(*frames[c]);
      CHECK(one.ok());
      const std::vector<float> depth = depth_of(*set.value()[c]);
      const std::vector<std::uint32_t> color = color_of(*set.value()[c]);
      CHECK(depth.size() == raws[c].size() && depth == depth_of(one.value()));
      CHECK(color.size() == raws[c].size() && color == color_of(one.value()));
      CHECK(std::count(depth.begin(), depth.end(), 0.0f) > 0);  // masked
      CHECK(set.value()[c]->depth_noise == frames[c]->depth_noise);
      CHECK(one->depth_noise == frames[c]->depth_noise);
      CHECK(set.value()[c]->depth_noise.known() == (c > 0));
    }
  }
  // A set with no frame is nothing to do.
  std::vector<std::optional<sensor::RgbdFrame>> empty(kCams);
  auto none = sensor::GpuFramePrep::prepare_batch(preps, empty);
  CHECK(none.ok() && none.value().size() == kCams && !none.value()[0]);

  // Refused before any work: no row is timed, though one is asked for.
  std::vector<std::optional<sensor::RgbdFrame>> refused = frames;
  refused[3]->depth = nullptr;
  vkc::StageMetrics metrics;
  CHECK(sensor::GpuFramePrep::prepare_batch(preps, refused, &metrics)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  CHECK(metrics.rows().empty());
  refused = frames;
  refused[3]->color_encoding.transfer =
      sensor::ColorEncoding::Transfer::Bt2020Pq;
  CHECK(sensor::GpuFramePrep::prepare_batch(preps, refused).status().domain() ==
        vkc::Status::Code::Unsupported);
  std::vector<sensor::GpuFramePrep> too_few;
  too_few.push_back(std::move(alone).value());
  CHECK(
      sensor::GpuFramePrep::prepare_batch(too_few, frames).status().domain() ==
      vkc::Status::Code::InvalidArgument);
  return 0;
}

// The driver's depth noise, as GpuFramePrep::finish hands it on: unchanged,
// term by term, with every other field and both buffers' words as the same
// frame stating none gives; a frame that states none hands on unknown, as a
// default RgbdFrame states. The set form is test_prepare_batch's.
int test_depth_noise(sensor::GpuFramePrep& prep) {
  CHECK(sensor::RgbdFrame{}.depth_noise == sensor::DepthNoise{});
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>(700 + (i * 31u) % 900u);
  }
  Planes p = make_planes();
  sensor::RgbdFrame f = frame_of(raw, lensed());
  f.color = p.image(0.2126f, 0.0722f, false);
  f.color_camera = pinhole();
  f.color_to_world[3] = glm::dvec4(0.5, -0.25, 2.0, 1.0);
  auto none = prep.prepare(f);
  CHECK(none.ok());
  CHECK(none->depth_noise == sensor::DepthNoise{});
  CHECK(!none->depth_noise.known());
  const std::vector<float> depth = depth_of(none.value());
  const std::vector<std::uint32_t> color = color_of(none.value());

  // Three distinct terms, so a swapped or dropped one shows.
  const sensor::DepthNoise stated{0.0011f, 0.00095f, 0.0021f};
  f.depth_noise = stated;
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const sensor::DeviceFrame& d = out.value();
  CHECK(d.depth_noise == stated);
  CHECK(d.depth_noise.a == 0.0011f && d.depth_noise.b == 0.00095f &&
        d.depth_noise.c == 0.0021f);
  // Nothing else changes.
  const sensor::DeviceFrame& n = none.value();
  CHECK(d.depth_camera.fx == n.depth_camera.fx &&
        d.depth_camera.fy == n.depth_camera.fy &&
        d.depth_camera.cx == n.depth_camera.cx &&
        d.depth_camera.cy == n.depth_camera.cy &&
        d.depth_camera.min_depth == n.depth_camera.min_depth &&
        d.depth_camera.max_depth == n.depth_camera.max_depth &&
        d.depth_camera.width == n.depth_camera.width &&
        d.depth_camera.height == n.depth_camera.height &&
        d.depth_camera.cam_to_world == n.depth_camera.cam_to_world);
  CHECK(d.color_camera.fx == n.color_camera.fx &&
        d.color_camera.fy == n.color_camera.fy &&
        d.color_camera.cx == n.color_camera.cx &&
        d.color_camera.cy == n.color_camera.cy &&
        d.color_camera.width == n.color_camera.width &&
        d.color_camera.height == n.color_camera.height &&
        d.color_camera.cam_to_world == n.color_camera.cam_to_world);
  CHECK(d.color_encoding == n.color_encoding);
  CHECK(d.timestamp_ns == n.timestamp_ns && d.has_color());
  CHECK(depth_of(d) == depth && color_of(d) == color);

  // A model the pass does not judge: what the driver stated, NaN and all.
  f.depth_noise = {std::numeric_limits<float>::quiet_NaN(), -1.0f, 0.0f};
  auto odd = prep.prepare(f);
  CHECK(odd.ok());
  CHECK(std::isnan(odd->depth_noise.a) && odd->depth_noise.b == -1.0f &&
        odd->depth_noise.c == 0.0f);
  return 0;
}

// The record form: a frame recorded into a caller's batch, read back in that
// batch, holds what prepare gives, and a set what prepare_batch gives; packed
// colour after a depth whose bytes the binding alignment does not divide
// lands exactly; a refused frame records nothing; a pass whose last batch is
// unresolved refuses, changing no buffer that batch uses, even for a larger
// frame; and once warm a frame allocates nothing to move bytes,
// and a prepare is one submit.
int test_record(vkc::Device& device, vkc::Allocator& allocator) {
  auto reference =
      sensor::GpuFramePrep::create(device, allocator, unfiltered());
  auto made = sensor::GpuFramePrep::create(device, allocator, unfiltered());
  CHECK(reference.ok() && made.ok());
  sensor::GpuFramePrep& prep = made.value();
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>((i * 7919u) % 6000u);
  }
  const std::vector<std::uint32_t> words = packed_words(raw.size(), 7);
  sensor::RgbdFrame f = frame_of(raw, lensed());
  f.color_packed = words.data();
  f.color_camera = lensed();
  auto want = reference->prepare(f);
  CHECK(want.ok());
  const std::vector<float> want_depth = depth_of(want.value());
  const std::vector<std::uint32_t> want_color = color_of(want.value());
  {
    vkc::CommandBatch batch(device, allocator);
    auto got = prep.record(batch, f);
    CHECK(got.ok() && got->has_color());
    std::vector<float> depth(raw.size());
    std::vector<std::uint32_t> color(raw.size());
    CHECK(batch
              .readback(*got->depth, 0, depth.size() * sizeof(float),
                        depth.data())
              .ok());
    CHECK(batch
              .readback(*got->color, 0, color.size() * sizeof(std::uint32_t),
                        color.data())
              .ok());
    CHECK(batch.submit().ok());
    CHECK(depth == want_depth && color == want_color);
  }

  // A set, a camera missing, against prepare_batch.
  std::vector<sensor::GpuFramePrep> preps;
  std::vector<sensor::GpuFramePrep> blocking;
  for (int c = 0; c < 3; ++c) {
    auto a = sensor::GpuFramePrep::create(device, allocator, unfiltered());
    auto b = sensor::GpuFramePrep::create(device, allocator, unfiltered());
    CHECK(a.ok() && b.ok());
    preps.push_back(std::move(a).value());
    blocking.push_back(std::move(b).value());
  }
  std::vector<std::optional<sensor::RgbdFrame>> frames{
      f, std::nullopt, frame_of(raw, pinhole())};
  auto blocked = sensor::GpuFramePrep::prepare_batch(blocking, frames);
  CHECK(blocked.ok());
  {
    vkc::CommandBatch batch(device, allocator);
    auto set = sensor::GpuFramePrep::record_set(batch, preps, frames);
    CHECK(set.ok() && set->size() == 3 && !(*set)[1]);
    CHECK(batch.submit().ok());
    for (const std::size_t c : {std::size_t{0}, std::size_t{2}}) {
      CHECK(depth_of(*(*set)[c]) == depth_of(*(*blocked)[c]));
    }
    CHECK(color_of(*(*set)[0]) == color_of(*(*blocked)[0]));
  }

  // 3x3 depth is 18 bytes, a word-padded 20, so the packed colour after it
  // starts at the binding alignment instead.
  camera::CameraModel tiny = pinhole();
  tiny.size = {3, 3};
  tiny.intrinsics = {4.0f, 4.0f, 1.0f, 1.0f};
  const std::vector<std::uint16_t> tiny_raw{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<std::uint32_t> tiny_words = packed_words(9, 11);
  sensor::RgbdFrame small = frame_of(tiny_raw, tiny);
  small.color_packed = tiny_words.data();
  small.color_camera = tiny;
  auto exact = prep.prepare(small);
  CHECK(exact.ok());
  const std::vector<float> small_depth = depth_of(exact.value());
  const std::vector<std::uint32_t> small_color = color_of(exact.value());
  CHECK(small_depth.size() == 9 && small_color.size() == 9);
  for (std::size_t i = 0; i < 9; ++i) {
    CHECK(small_depth[i] == static_cast<float>(tiny_raw[i]) * kScale);
    CHECK(small_color[i] == ((tiny_words[i] & 0xFFFFFFu) | 0xFF000000u));
  }

  // A refused frame records nothing: the batch submits nothing.
  {
    sensor::RgbdFrame refused = f;
    refused.depth = nullptr;
    vkc::CommandBatch batch(device, allocator);
    const vkc::DeviceCounters was = device.counters();
    CHECK(prep.record(batch, refused).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(batch.submit().ok());
    CHECK(device.counters().submits == was.submits);
  }

  // A batch recorded and unresolved holds the pass's slot: a record before
  // it resolves refuses, with nothing recorded, and after it records again.
  {
    vkc::CommandBatch first(device, allocator);
    vkc::CommandBatch second(device, allocator);
    auto a = prep.record(first, f);
    CHECK(a.ok());
    CHECK(prep.record(second, f).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(first.submit().ok() && second.submit().ok());
  }
  CHECK(prep.prepare(f).ok());

  // That refusal changes no buffer the unresolved batch uses: a small frame
  // recorded with its outputs' readback and its frame dropped, so nothing
  // holds those outputs but the batch, then a larger one refused, which would
  // grow them; the first batch still reads what prepare gives.
  {
    auto fresh = sensor::GpuFramePrep::create(device, allocator, unfiltered());
    CHECK(fresh.ok());
    const sensor::RgbdFrame small_images = image_frame(16, 16);
    const sensor::RgbdFrame large_images = image_frame(kWidth, kHeight);
    auto want_small = reference->prepare(small_images);
    CHECK(want_small.ok());
    const std::vector<float> small_want_depth = depth_of(want_small.value());
    const std::vector<std::uint32_t> small_want_color =
        color_of(want_small.value());
    std::vector<float> depth(small_want_depth.size());
    std::vector<std::uint32_t> color(small_want_color.size());
    vkc::CommandBatch first(device, allocator);
    vkc::CommandBatch second(device, allocator);
    {
      auto a = fresh->record(first, small_images);
      CHECK(a.ok() && a->has_color());
      CHECK(first
                .readback(*a->depth, 0, depth.size() * sizeof(float),
                          depth.data())
                .ok());
      CHECK(first
                .readback(*a->color, 0, color.size() * sizeof(std::uint32_t),
                          color.data())
                .ok());
    }  // the frame and its outputs' leases go
    CHECK(fresh->record(second, large_images).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    CHECK(first.submit().ok() && second.submit().ok());
    CHECK(depth == small_want_depth && color == small_want_color);
    CHECK(fresh->prepare(large_images).ok());
  }

  // Warm: a hundred frames allocate nothing that moves bytes, each prepare
  // one submit.
  CHECK(prep.prepare(f).ok());
  const vkc::DeviceCounters warm = device.counters();
  for (int i = 0; i < 100; ++i) CHECK(prep.prepare(f).ok());
  const vkc::DeviceCounters now = device.counters();
  CHECK(now.staging_buffers == warm.staging_buffers);
  CHECK(now.submits - warm.submits == 100);
  return 0;
}

// A set's timed commands share one window, grown with the array: a dozen
// cameras, four commands each, still report the set's device time.
int test_batch_timing(vkc::Device& device, vkc::Allocator& allocator) {
  constexpr std::size_t kCams = 12;
  std::vector<sensor::GpuFramePrep> preps;
  for (std::size_t c = 0; c < kCams; ++c) {
    auto made = sensor::GpuFramePrep::create(device, allocator, unfiltered());
    CHECK(made.ok());
    preps.push_back(std::move(made).value());
  }
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes planes = make_planes();
  sensor::RgbdFrame frame = frame_of(raw, pinhole());
  frame.color = planes.image(0.2126f, 0.0722f, false);
  frame.color_camera = pinhole();
  const std::vector<std::optional<sensor::RgbdFrame>> frames(kCams, frame);
  vkc::StageMetrics one;
  CHECK(preps[0].prepare(frame, &one).ok());
  if (one.rows().size() != 1 || !one.rows()[0].has_gpu) {
    std::printf("  batch timing: skipped, the device times nothing\n");
    return 0;
  }
  vkc::StageMetrics metrics;
  CHECK(sensor::GpuFramePrep::prepare_batch(preps, frames, &metrics).ok());
  CHECK(metrics.rows().size() == 1 && metrics.rows()[0].has_gpu);
  return 0;
}

// w x h raw depth for the device-depth test: 16 px squares of a slope, which
// the pass mixes, and of random returns, which it samples at the nearest
// pixel, with a hole every 37 pixels.
std::vector<std::uint16_t> mixed_raw(std::uint32_t w, std::uint32_t h) {
  std::vector<std::uint16_t> raw(std::size_t{w} * h);
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      const std::size_t i = std::size_t{y} * w + x;
      raw[i] = i % 37 == 0 ? 0
               : ((x / 16 + y / 16) & 1u) != 0
                   ? static_cast<std::uint16_t>(500 + (i * 2654435761u) % 4000u)
                   : static_cast<std::uint16_t>(1500 + 3 * x + 2 * y);
    }
  }
  return raw;
}

// `raw`'s samples in a device storage buffer from byte `at`, after `at` junk
// bytes and before `tail` more, as a GPU stage leaves its depth; null if it
// cannot be made. The junk reads as 61 m, so a sample read from the wrong
// place shows.
std::shared_ptr<const vkc::Buffer> depth_buffer(
    const std::vector<std::uint16_t>& raw, std::size_t at, std::size_t tail) {
  std::vector<std::uint8_t> bytes(
      at + raw.size() * sizeof(std::uint16_t) + tail, 0xEE);
  std::memcpy(bytes.data() + at, raw.data(),
              raw.size() * sizeof(std::uint16_t));
  return on_device(bytes);
}

// `host`, its depth moved to the device: `buffer` from byte `at`.
sensor::RgbdFrame on_device_depth(sensor::RgbdFrame host,
                                  std::shared_ptr<const vkc::Buffer> buffer,
                                  std::uint64_t at) {
  host.depth = nullptr;
  host.depth_device.buffer = std::move(buffer);
  host.depth_device.offset = at;
  return host;
}

// A prepared frame's depth as the words the pass wrote, for a comparison bit
// for bit.
std::vector<std::uint32_t> depth_words(const sensor::DeviceFrame& f) {
  return read<std::uint32_t>(
      *f.depth, std::size_t{f.depth_camera.width} * f.depth_camera.height);
}

// Depth on the device, as a GPU stage leaves it (RgbdFrame::depth_device), is
// read where it is, and gives the words the same samples from the host give,
// bit for bit, through the barrel lens, mixed, at an edge and at a hole, less
// a depth offset:
//   - from byte 0 of a buffer that ends at the last sample;
//   - deep in a bigger buffer, junk before and after, at offsets no binding
//     may start at: an odd count of samples in, so the image starts in the
//     second half of a word, and an even count;
//   - an odd count of samples that ends two bytes into a word, refused while
//     that word runs past the buffer and read once the buffer holds it, or
//     ending a word when it starts two bytes in;
//   - through a pinhole, as stereo depth comes, read to the last sample's
//     word, at even and odd counts and offsets, the buffer ending there;
//   - taken over from outside Vulkan (CUDA), one barrier more, or named as
//     the pass's own family;
//   - beside packed colour from the host, masked by the colour's view;
//   - in a set beside frames whose depth is on the host, one with planes in a
//     buffer, prepared or recorded, as each prepares alone;
//   - in one buffer with the frame's planes, or with other cameras' depth,
//     taken over once, and refused where they name different writers.
// A frame with nothing on the host still holds the pass's slot, so a record
// before its batch resolves is refused; warm, it allocates nothing to move
// bytes; and the frames it refuses, before any work.
int test_device_depth(vkc::Device& device, vkc::Allocator& allocator) {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  sensor::GpuFramePrepConfig config = unfiltered();
  config.depth_offset_m = 0.0123f;
  auto made = sensor::GpuFramePrep::create(device, allocator, config);
  auto made_reference = sensor::GpuFramePrep::create(device, allocator, config);
  CHECK(made.ok() && made_reference.ok());
  sensor::GpuFramePrep& prep = made.value();
  sensor::GpuFramePrep& reference = made_reference.value();

  const camera::CameraModel cam = lensed();
  const std::vector<std::uint16_t> raw = mixed_raw(kWidth, kHeight);
  const std::size_t bytes = raw.size() * sizeof(std::uint16_t);
  const sensor::RgbdFrame host = frame_of(raw, cam);
  auto want_frame = reference.prepare(host);
  CHECK(want_frame.ok());
  const std::vector<std::uint32_t> want = depth_words(want_frame.value());
  CHECK(want.size() == raw.size());
  std::size_t holes = 0;
  for (const std::uint32_t w : want) holes += w == 0u ? 1 : 0;
  CHECK(holes > raw.size() / 100 && holes < raw.size() / 2);
  const auto same = [&](const sensor::RgbdFrame& f) {
    auto got = prep.prepare(f);
    if (!got) std::fprintf(stderr, "%s\n", got.status().message().c_str());
    return got.ok() && depth_words(got.value()) == want;
  };
  const auto refused = [&](const sensor::RgbdFrame& f, const char* why) {
    vkc::StageMetrics metrics;
    const vkc::Status status = prep.prepare(f, &metrics).status();
    if (status.message().find(why) == std::string::npos) {
      std::fprintf(stderr, "  refused for \"%s\", not \"%s\"\n",
                   status.message().c_str(), why);
    }
    return status.domain() == invalid &&
           status.message().find(why) != std::string::npos &&
           metrics.rows().empty();
  };

  // From byte 0, the buffer ending at the last sample (a whole word).
  const auto exact = depth_buffer(raw, 0, 0);
  CHECK(exact != nullptr && exact->size() == bytes);
  const sensor::RgbdFrame at_zero = on_device_depth(host, exact, 0);
  CHECK(at_zero.has_depth() && !at_zero.has_color());
  CHECK(same(at_zero));
  // Deep in a bigger buffer: 4102 is 2051 samples in, odd, and 4100 even;
  // neither is a multiple of 16, the binding alignment of the RTX 4090 and
  // lavapipe, and 4102 is not even a word's.
  constexpr std::size_t kDeep = 4102;
  const auto deep = depth_buffer(raw, kDeep, 10);
  const auto deep_even = depth_buffer(raw, kDeep - 2, 6);
  CHECK(deep != nullptr && deep_even != nullptr);
  const sensor::RgbdFrame deep_frame = on_device_depth(host, deep, kDeep);
  CHECK(same(deep_frame));
  CHECK(same(on_device_depth(host, deep_even, kDeep - 2)));
  // Taken over from CUDA, or named as the pass's own family: the same depth.
  sensor::RgbdFrame taken = deep_frame;
  taken.depth_device.queue_family = sensor::kQueueFamilyExternal;
  CHECK(same(taken));
  // And taken over: from CUDA, one barrier more than from the pass's own
  // family, the depth buffer's acquire.
  const std::uint64_t own_barriers = barriers_of(prep, deep_frame);
  CHECK(barriers_of(prep, taken) == own_barriers + 1);
  taken.depth_device.queue_family = device.queue_family();
  CHECK(same(taken));

  // An odd count of samples, 321 x 241: from byte 4096 they end two bytes
  // into a word, which the kernel reads whole, so a buffer that ends with
  // them is refused, and one two bytes longer is read; from 4098 they end a
  // word, and the buffer may end with them.
  {
    camera::CameraModel odd_cam = cam;
    odd_cam.size = {kWidth + 1, kHeight + 1};
    const std::vector<std::uint16_t> odd_raw =
        mixed_raw(kWidth + 1, kHeight + 1);
    CHECK(odd_raw.size() % 2 == 1);
    const sensor::RgbdFrame odd_host = frame_of(odd_raw, odd_cam);
    auto odd_want = reference.prepare(odd_host);
    CHECK(odd_want.ok());
    const std::vector<std::uint32_t> odd_words = depth_words(odd_want.value());
    const auto odd_same = [&](const sensor::RgbdFrame& f) {
      auto got = prep.prepare(f);
      if (!got) std::fprintf(stderr, "%s\n", got.status().message().c_str());
      return got.ok() && depth_words(got.value()) == odd_words;
    };
    const auto short_word = depth_buffer(odd_raw, 4096, 0);
    CHECK(short_word != nullptr && short_word->size() % 4 == 2);
    CHECK(refused(on_device_depth(odd_host, short_word, 4096),
                  "the binding reads"));
    CHECK(odd_same(
        on_device_depth(odd_host, depth_buffer(odd_raw, 4096, 2), 4096)));
    const auto ends_word = depth_buffer(odd_raw, 4098, 0);
    CHECK(ends_word != nullptr && ends_word->size() % 4 == 0);
    CHECK(odd_same(on_device_depth(odd_host, ends_word, 4098)));
  }

  // Through a pinhole, as stereo matching's rectified depth comes, the pass
  // reads every raw pixel, the word holding the last sample included, which
  // the barrel lens above never reaches: 320 x 240 from byte 0 and from byte
  // 4102 (the image starting in a word's second half), and 321 x 241 from
  // bytes 4098 and 4096, each in a buffer that ends at the samples' last
  // word. The last sample is a return, so a binding cut short, which reads
  // it as no return, shows.
  for (const std::uint32_t pad : {0u, 1u}) {
    camera::CameraModel flat = pinhole();
    flat.size = {kWidth + pad, kHeight + pad};
    const std::vector<std::uint16_t> flat_raw =
        mixed_raw(flat.size.width, flat.size.height);
    const std::size_t flat_bytes = flat_raw.size() * sizeof(std::uint16_t);
    const sensor::RgbdFrame flat_host = frame_of(flat_raw, flat);
    auto flat_want = reference.prepare(flat_host);
    CHECK(flat_want.ok());
    const std::vector<std::uint32_t> flat_words =
        depth_words(flat_want.value());
    CHECK(flat_raw.back() != 0 && flat_words.size() == flat_raw.size() &&
          flat_words.back() != 0u);
    const std::vector<std::size_t> offsets =
        pad == 0 ? std::vector<std::size_t>{0, kDeep}
                 : std::vector<std::size_t>{kDeep - 4, kDeep - 6};
    for (const std::size_t at : offsets) {
      const std::size_t end = (at + flat_bytes + 3) / 4 * 4;
      const auto buffer = depth_buffer(flat_raw, at, end - at - flat_bytes);
      CHECK(buffer != nullptr && buffer->size() == end);
      auto got = prep.prepare(on_device_depth(flat_host, buffer, at));
      if (!got) std::fprintf(stderr, "%s\n", got.status().message().c_str());
      CHECK(got.ok() && depth_words(got.value()) == flat_words);
    }
  }

  // Beside packed colour from the host, the colour camera's pincushion lens
  // missing the corners: depth masked by the colour's view, and the colour,
  // as the same frame with its depth on the host gives them.
  camera::CameraModel pincushion = pinhole();
  pincushion.distortion.k1 = 0.3f;
  const std::vector<std::uint32_t> words = packed_words(raw.size(), 5);
  sensor::RgbdFrame host_colored = host;
  host_colored.color_packed = words.data();
  host_colored.color_camera = pincushion;
  auto colored_want = reference.prepare(host_colored);
  CHECK(colored_want.ok());
  const std::vector<std::uint32_t> colored_depth =
      depth_words(colored_want.value());
  const std::vector<std::uint32_t> colored_color =
      color_of(colored_want.value());
  CHECK(colored_depth != want);  // the mask zeroed the corners
  {
    auto got = prep.prepare(on_device_depth(host_colored, deep, kDeep));
    CHECK(got.ok());
    CHECK(depth_words(got.value()) == colored_depth);
    CHECK(color_of(got.value()) == colored_color);
  }

  // A set: host depth with packed colour; device depth from CUDA, no colour;
  // a camera missing; device depth deep in its buffer with I420 planes in a
  // buffer; host depth alone. Prepared together, and recorded into a
  // caller's batch, each is what its pass gives it alone.
  {
    Planes planes = patterned(kWidth, kHeight);
    sensor::RgbdFrame host_planes = host;
    host_planes.color = planes.image(0.2126f, 0.0722f, false);
    host_planes.color_camera = pincushion;
    CHECK(host_planes.color.device != nullptr);
    sensor::RgbdFrame from_cuda = at_zero;
    from_cuda.depth_device.queue_family = sensor::kQueueFamilyExternal;
    const std::vector<std::optional<sensor::RgbdFrame>> mixed{
        host_colored, from_cuda, std::nullopt,
        on_device_depth(host_planes, deep, kDeep), host};
    // What each gives alone, its depth on the host.
    const std::vector<std::optional<sensor::RgbdFrame>> alone{
        host_colored, host, std::nullopt, host_planes, host};
    std::vector<sensor::GpuFramePrep> preps;
    std::vector<sensor::GpuFramePrep> recording;
    for (std::size_t c = 0; c < mixed.size(); ++c) {
      auto a = sensor::GpuFramePrep::create(device, allocator, config);
      auto b = sensor::GpuFramePrep::create(device, allocator, config);
      CHECK(a.ok() && b.ok());
      preps.push_back(std::move(a).value());
      recording.push_back(std::move(b).value());
    }
    for (int round = 0; round < 2; ++round) {
      auto set = sensor::GpuFramePrep::prepare_batch(preps, mixed);
      if (!set) std::fprintf(stderr, "%s\n", set.status().message().c_str());
      CHECK(set.ok() && set->size() == mixed.size() && !(*set)[2]);
      vkc::CommandBatch batch(device, allocator);
      auto recorded = sensor::GpuFramePrep::record_set(batch, recording, mixed);
      CHECK(recorded.ok() && recorded->size() == mixed.size());
      CHECK(batch.submit().ok());
      for (std::size_t c = 0; c < mixed.size(); ++c) {
        if (!mixed[c]) continue;
        auto one = reference.prepare(*alone[c]);
        CHECK(one.ok());
        const std::vector<std::uint32_t> d = depth_words(one.value());
        CHECK(depth_words(*(*set)[c]) == d);
        CHECK(depth_words(*(*recorded)[c]) == d);
        CHECK(mixed[c]->has_color() == (*set)[c]->has_color());
        if (mixed[c]->has_color()) {
          const std::vector<std::uint32_t> col = color_of(one.value());
          CHECK(color_of(*(*set)[c]) == col);
          CHECK(color_of(*(*recorded)[c]) == col);
        }
      }
    }
    // One frame refused refuses the set before any work.
    std::vector<std::optional<sensor::RgbdFrame>> bad = mixed;
    bad[1]->depth_device.offset = 1;
    vkc::StageMetrics metrics;
    CHECK(sensor::GpuFramePrep::prepare_batch(preps, bad, &metrics)
              .status()
              .domain() == invalid);
    CHECK(metrics.rows().empty());
  }

  // One buffer named by several depths and pictures is taken over once: a
  // frame's depth and its I420 planes in one buffer, and three cameras'
  // depth in one buffer, from CUDA, record one barrier more than from the
  // pass's own family, and give what separate buffers give. References to
  // one buffer that name different writers are refused, before any work;
  // the pass's own family and kQueueFamilyIgnored are one writer.
  {
    Planes planes = patterned(kWidth, kHeight);
    sensor::RgbdFrame apart = host;
    apart.color = planes.image(0.2126f, 0.0722f, false);
    apart.color_camera = pincushion;
    CHECK(apart.color.device != nullptr);
    auto apart_want = reference.prepare(apart);
    CHECK(apart_want.ok());
    // The depth from byte 0, then the planes as planes.image lays them out.
    std::vector<std::uint8_t> blob(bytes);
    std::memcpy(blob.data(), raw.data(), bytes);
    for (const std::vector<std::uint8_t>* plane :
         {&planes.y, &planes.cb, &planes.cr}) {
      blob.insert(blob.end(), plane->begin(), plane->end());
    }
    blob.resize((blob.size() + 3) & ~std::size_t{3}, 0xEE);
    const auto shared = on_device(blob);
    CHECK(shared != nullptr);
    sensor::RgbdFrame together = on_device_depth(apart, shared, 0);
    together.color.device = shared;
    for (std::uint64_t& at : together.color.offset) at += bytes;
    sensor::RgbdFrame together_cuda = together;
    together_cuda.depth_device.queue_family = sensor::kQueueFamilyExternal;
    together_cuda.color.queue_family = sensor::kQueueFamilyExternal;
    {
      auto got = prep.prepare(together_cuda);
      if (!got) std::fprintf(stderr, "%s\n", got.status().message().c_str());
      CHECK(got.ok());
      CHECK(depth_words(got.value()) == depth_words(apart_want.value()));
      CHECK(color_of(got.value()) == color_of(apart_want.value()));
    }
    const std::uint64_t together_own = barriers_of(prep, together);
    CHECK(barriers_of(prep, together_cuda) == together_own + 1);
    sensor::RgbdFrame split = together_cuda;
    split.color.queue_family = sensor::kQueueFamilyIgnored;
    CHECK(refused(split, "one buffer is named"));
    split.color.queue_family = device.queue_family();
    split.depth_device.queue_family = sensor::kQueueFamilyIgnored;
    CHECK(prep.prepare(split).ok());

    // Three cameras' depth, 6 bytes apart, two of them where no binding may
    // start, each its own depth.
    constexpr std::size_t kCams = 3;
    const std::size_t stride = bytes + 6;
    std::vector<std::vector<std::uint16_t>> raws(kCams, raw);
    std::vector<std::uint8_t> all(kCams * stride, 0xEE);
    std::vector<std::optional<sensor::RgbdFrame>> own_set(kCams);
    std::vector<std::optional<sensor::RgbdFrame>> cuda_set(kCams);
    std::vector<sensor::GpuFramePrep> preps;
    for (std::size_t c = 0; c < kCams; ++c) {
      for (std::uint16_t& s : raws[c]) {
        if (s != 0) s = static_cast<std::uint16_t>(s + 100 * c);
      }
      std::memcpy(all.data() + c * stride, raws[c].data(), bytes);
      auto made_pass = sensor::GpuFramePrep::create(device, allocator, config);
      CHECK(made_pass.ok());
      preps.push_back(std::move(made_pass).value());
    }
    const auto one = on_device(all);
    CHECK(one != nullptr);
    for (std::size_t c = 0; c < kCams; ++c) {
      own_set[c] = on_device_depth(frame_of(raws[c], cam), one, c * stride);
      cuda_set[c] = own_set[c];
      cuda_set[c]->depth_device.queue_family = sensor::kQueueFamilyExternal;
    }
    auto set = sensor::GpuFramePrep::prepare_batch(preps, cuda_set);
    if (!set) std::fprintf(stderr, "%s\n", set.status().message().c_str());
    CHECK(set.ok() && set->size() == kCams);
    for (std::size_t c = 0; c < kCams; ++c) {
      auto alone = reference.prepare(frame_of(raws[c], cam));
      CHECK(alone.ok());
      CHECK(depth_words(*(*set)[c]) == depth_words(alone.value()));
    }
    const std::uint64_t set_own = barriers_of(preps, own_set);
    CHECK(barriers_of(preps, cuda_set) == set_own + 1);
    std::vector<std::optional<sensor::RgbdFrame>> split_set = cuda_set;
    split_set[1]->depth_device.queue_family = device.queue_family();
    vkc::StageMetrics metrics;
    const vkc::Status status =
        sensor::GpuFramePrep::prepare_batch(preps, split_set, &metrics)
            .status();
    CHECK(status.domain() == invalid &&
          status.message().find("one buffer is named") != std::string::npos);
    CHECK(metrics.rows().empty());
  }

  // Nothing on the host, yet the frame's batch holds the pass's slot: a
  // record before it resolves is refused, with nothing recorded, and after it
  // records again.
  {
    auto fresh = sensor::GpuFramePrep::create(device, allocator, config);
    CHECK(fresh.ok());
    vkc::CommandBatch first(device, allocator);
    vkc::CommandBatch second(device, allocator);
    CHECK(fresh->record(first, deep_frame).ok());
    const vkc::DeviceCounters was = device.counters();
    CHECK(fresh->record(second, deep_frame).status().domain() == invalid);
    CHECK(first.submit().ok() && second.submit().ok());
    CHECK(device.counters().submits - was.submits == 1);  // the first only
    CHECK(same(deep_frame));
  }

  // Warm: frames whose depth is on the device allocate nothing to move
  // bytes, each prepare one submit.
  CHECK(prep.prepare(deep_frame).ok());
  const vkc::DeviceCounters warm = device.counters();
  for (int i = 0; i < 50; ++i) CHECK(prep.prepare(deep_frame).ok());
  const vkc::DeviceCounters now = device.counters();
  CHECK(now.staging_buffers == warm.staging_buffers);
  CHECK(now.submits - warm.submits == 50);

  // Refused, before any work: depth on the host and the device; an empty
  // buffer, or one the kernel cannot bind; an odd offset; samples past the
  // buffer, by a sample or from an offset past it; and a queue family the
  // device does not have.
  sensor::RgbdFrame bad = deep_frame;
  bad.depth = raw.data();
  CHECK(refused(bad, "not both"));
  bad = deep_frame;
  bad.depth_device.buffer = std::make_shared<const vkc::Buffer>();
  CHECK(refused(bad, "is empty"));
  vkc::BufferDesc desc;
  desc.size = deep->size();
  desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  auto unbindable = allocator.create_buffer(desc);
  CHECK(unbindable.ok());
  bad.depth_device.buffer =
      std::make_shared<const vkc::Buffer>(std::move(unbindable).value());
  CHECK(refused(bad, "not a storage buffer"));
  bad = deep_frame;
  bad.depth_device.offset = kDeep + 1;
  CHECK(refused(bad, "is odd"));
  bad = deep_frame;
  bad.depth_device.offset = deep->size() - bytes + 2;  // a sample past it
  CHECK(refused(bad, "runs past its buffer"));
  bad.depth_device.offset = deep->size() + 2;
  CHECK(refused(bad, "runs past its buffer"));
  bad = deep_frame;
  bad.depth_device.queue_family =
      static_cast<std::uint32_t>(device.caps().queue_families().size());
  CHECK(refused(bad, "which the device does not have"));
  // The same depth through the frame's own camera, but no depth at all.
  bad = deep_frame;
  bad.depth_device = {};
  CHECK(!bad.has_depth());
  CHECK(refused(bad, "has no depth"));
  std::printf(
      "  device depth: from bytes 0, %zu and %zu of its buffer, bit for bit "
      "as from the host (%zu holes)\n",
      kDeep - 2, kDeep, holes);
  return 0;
}

// A pincushion lens maps the pinhole image's corners outside the captured
// picture. There colour is a 0 word, coverage and all, so that fusion skips
// it rather than fusing black; everywhere else its coverage byte is 0xFF,
// black included.
int test_coverage(sensor::GpuFramePrep& prep) {
  camera::CameraModel cam = pinhole();
  cam.distortion.k1 = 0.3f;
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes p = make_planes();  // all zero: black at full range
  std::fill(p.cb.begin(), p.cb.end(), 128);
  std::fill(p.cr.begin(), p.cr.end(), 128);
  sensor::RgbdFrame f = frame_of(raw, cam);
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

// The output cameras are posed from the frame's: the colour camera at
// color_to_world, the depth camera through the sensor's extrinsic. A pose
// that is not rigid is refused, and so is a double model past float's range,
// which the passes would read as infinite.
int test_poses(sensor::GpuFramePrep& prep) {
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1500);
  Planes p = make_planes();
  sensor::RgbdFrame f = frame_of(raw, pinhole());
  f.color = p.image(0.299f, 0.114f, true);
  f.color_camera = pinhole();
  f.color_to_world =
      camera::Mat4d(camera::rotation_from_rodrigues({0.1, -0.2, 0.3}));
  f.color_to_world[3] = glm::dvec4(1.0, 2.0, -0.5, 1.0);
  f.depth_to_color =
      camera::Mat4d(camera::rotation_from_rodrigues({0.002, 0.001, 0.0}));
  f.depth_to_color[3] = glm::dvec4(-0.032, -0.002, 0.004, 1.0);
  auto out = prep.prepare(f);
  CHECK(out.ok());
  const sensor::Mat4f color(f.color_to_world);
  const sensor::Mat4f depth(f.color_to_world * f.depth_to_color);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      CHECK(std::fabs(out->color_camera.cam_to_world[c][r] - color[c][r]) <
            1e-6f);
      CHECK(std::fabs(out->depth_camera.cam_to_world[c][r] - depth[c][r]) <
            1e-6f);
    }
  }

  const auto invalid = vkc::Status::Code::InvalidArgument;
  sensor::RgbdFrame bad = f;
  bad.color_to_world = camera::Mat4d(2.0);
  CHECK(prep.prepare(bad).status().domain() == invalid);
  bad = f;
  bad.depth_to_color[0][0] = 1.5;
  CHECK(prep.prepare(bad).status().domain() == invalid);
  bad = f;
  bad.depth_camera.intrinsics.fx = 1e40;  // finite in double, not in float
  CHECK(prep.prepare(bad).status().domain() == invalid);
  return 0;
}

int test_refusals(sensor::GpuFramePrep& prep) {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes p = make_planes();

  sensor::RgbdFrame f = frame_of(raw, pinhole());
  f.depth = nullptr;
  CHECK(prep.prepare(f).status().domain() == invalid);
  f = frame_of(raw, pinhole());
  f.depth_camera.intrinsics.fx = 0.0;
  CHECK(prep.prepare(f).status().domain() == invalid);
  f = frame_of(raw, pinhole());
  f.depth_camera.distortion.k1 = NAN;
  CHECK(prep.prepare(f).status().domain() == invalid);
  f = frame_of(raw, pinhole());
  f.metres_per_unit = 0.0f;
  CHECK(prep.prepare(f).status().domain() == invalid);
  // The depth gate: 0 is "no return", so a range from 0 is refused, as are
  // RgbdFrame's unset zeros, an empty range and a NaN.
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
  f.color_camera.size.width = kWidth / 2;  // the picture disagrees
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
  f.color_encoding.transfer = sensor::ColorEncoding::Transfer::Bt2020Pq;
  CHECK(prep.prepare(f).status().domain() == vkc::Status::Code::Unsupported);
  f.color_encoding = {};
  CHECK(prep.prepare(f).ok());

  // A colour half refused costs the depth half nothing either: both are
  // checked before anything is uploaded, so no row is timed, as for a set.
  vkc::StageMetrics metrics;
  f.color.stride[1] = 4;
  CHECK(prep.prepare(f, &metrics).status().domain() == invalid);
  CHECK(metrics.rows().empty());

  // Packed words beside Y'CbCr planes, which dispatches nothing either, and
  // in an encoding the pass does not convert.
  const std::vector<std::uint32_t> words = packed_words(raw.size(), 2);
  f = frame_of(raw, pinhole());
  f.color_packed = words.data();
  f.color_camera = pinhole();
  CHECK(prep.prepare(f).ok());
  f.color = p.image(0.299f, 0.114f, true);
  vkc::StageMetrics both;
  const vkc::Status refused = prep.prepare(f, &both).status();
  CHECK(refused.domain() == invalid &&
        refused.message().find("not both") != std::string::npos);
  CHECK(both.rows().empty());
  f.color = {};
  f.color_encoding.transfer = sensor::ColorEncoding::Transfer::Bt2020Pq;
  CHECK(prep.prepare(f).status().domain() == vkc::Status::Code::Unsupported);
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

// One pass under `config`: a frame with colour prepares, and its depth and
// colour outputs carry `depth` and `color` as their sharing modes, as do the
// outputs the next frame reuses.
int prepared_sharing(vkc::Device& device, vkc::Allocator& allocator,
                     const sensor::GpuFramePrepConfig& config,
                     const sensor::RgbdFrame& frame, VkSharingMode depth,
                     VkSharingMode color) {
  auto prep = sensor::GpuFramePrep::create(device, allocator, config);
  if (!prep) std::fprintf(stderr, "%s\n", prep.status().message().c_str());
  CHECK(prep.ok());
  const vkc::Buffer* held = nullptr;
  for (int pass = 0; pass < 2; ++pass) {
    auto out = prep->prepare(frame);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    CHECK(out.ok() && out->has_color());
    CHECK(out->depth->sharing_mode() == depth);
    CHECK(out->color->sharing_mode() == color);
    // Storage and transfer usage, no vertex usage: a consumer reads it as a
    // storage buffer, or copies from it.
    CHECK((out->depth->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0);
    CHECK((out->depth->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0);
    CHECK((out->depth->usage() & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) == 0);
    CHECK(held == nullptr || out->depth.get() == held);  // reused
    held = out->depth.get();
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == std::size_t{kWidth} * kHeight &&
          d[0] == 1000.0f * kScale);
  }
  return 0;
}

// Each output is shared with the queue families the config names for it, for
// a consumer on another queue: no family, one, or the pass's own named twice
// leaves it EXCLUSIVE, and a second family makes it CONCURRENT. Depth and
// colour are independent: naming one's families leaves the other EXCLUSIVE.
// A count past the array, or a family the device lacks, is refused for either.
// The second family is skipped on a device that has only one.
int test_queue_families(vkc::Device& device, vkc::Allocator& allocator) {
  constexpr VkSharingMode kExclusive = VK_SHARING_MODE_EXCLUSIVE;
  constexpr VkSharingMode kConcurrent = VK_SHARING_MODE_CONCURRENT;
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes planes = make_planes();
  sensor::RgbdFrame f = frame_of(raw, pinhole());
  f.color = planes.image(0.299f, 0.114f, true);
  f.color_camera = pinhole();

  const std::uint32_t own = device.queue_family();
  const auto named = [](std::initializer_list<std::uint32_t> depth,
                        std::initializer_list<std::uint32_t> color) {
    sensor::GpuFramePrepConfig c = unfiltered();
    std::copy(depth.begin(), depth.end(), c.depth_queue_families);
    c.depth_queue_family_count = static_cast<std::uint32_t>(depth.size());
    std::copy(color.begin(), color.end(), c.color_queue_families);
    c.color_queue_family_count = static_cast<std::uint32_t>(color.size());
    return c;
  };
  struct Case {
    sensor::GpuFramePrepConfig config;
    VkSharingMode depth, color;
  };
  std::vector<Case> cases = {
      {named({}, {}), kExclusive, kExclusive},
      {named({own}, {}), kExclusive, kExclusive},
      {named({own, own}, {own, own}), kExclusive, kExclusive},
  };
  if (const std::optional<std::uint32_t> other =
          vkc::test::other_queue_family(device)) {
    cases.push_back({named({own, *other}, {}), kConcurrent, kExclusive});
    cases.push_back({named({}, {own, *other}), kExclusive, kConcurrent});
    cases.push_back({named({*other, own, *other, own}, {own, *other}),
                     kConcurrent, kConcurrent});
  } else {
    std::printf("  one queue family: CONCURRENT outputs not exercised\n");
  }
  for (const Case& c : cases) {
    if (prepared_sharing(device, allocator, c.config, f, c.depth, c.color) !=
        0) {
      return 1;
    }
  }
  // Refused at create, naming the output and why: a count past the array, and
  // a family the device lacks -- for colour too, which no frame would
  // otherwise reach while the frames carry none.
  const auto device_families =
      static_cast<std::uint32_t>(device.caps().queue_families().size());
  struct Refusal {
    sensor::GpuFramePrepConfig config;
    const char* why;
  };
  for (const bool depth : {true, false}) {
    const std::string who =
        std::string("GpuFramePrep::create, ") + (depth ? "depth" : "colour");
    sensor::GpuFramePrepConfig too_many;
    (depth ? too_many.depth_queue_family_count
           : too_many.color_queue_family_count) =
        vkc::BufferDesc::kMaxQueueFamilies + 1;
    const Refusal refusals[] = {
        {too_many, ": queue_family_count"},
        {depth ? named({own, device_families}, {})
               : named({}, {own, device_families}),
         ": queue family"},
    };
    for (const Refusal& r : refusals) {
      const vkc::Status refused =
          sensor::GpuFramePrep::create(device, allocator, r.config).status();
      CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
      CHECK(refused.message().rfind(who + r.why, 0) == 0);
    }
  }
  return 0;
}

// What depth_queue_families is for: on one device shared by recon and a
// renderer on another queue family, carved as SharedDevice carves them, sets
// of 1, 3 and 5 frames are prepared on recon's family in one batch each, and
// every frame's depth and colour, read on the renderer's family, are what a
// read on recon's family gets. Nothing depends on the count of frames. The
// validation layer checks the CONCURRENT create info but not which family
// owns an EXCLUSIVE buffer, so the recorded sharing mode is what shows the
// read is legal. Skipped where the device puts both on one family.
int test_cross_family() {
  auto made =
      vkc::test::TwoFamilyDevice::create(sensor::device_requirements(), {});
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  vkc::test::TwoFamilyDevice& two = **made;
  if (!two.crosses_families()) {
    std::printf("  cross-family read skipped: %s\n",
                two.shared().summary().c_str());
    return 0;
  }
  sensor::GpuFramePrepConfig config = unfiltered();
  for (std::uint32_t i = 0; i < vkc::test::TwoFamilyDevice::kFamilyCount; ++i) {
    config.depth_queue_families[i] = two.families()[i];
    config.color_queue_families[i] = two.families()[i];
  }
  config.depth_queue_family_count = vkc::test::TwoFamilyDevice::kFamilyCount;
  config.color_queue_family_count = vkc::test::TwoFamilyDevice::kFamilyCount;
  constexpr std::uint32_t kW = 64, kH = 48;
  const std::size_t n = std::size_t{kW} * kH;
  camera::CameraModel cam;
  cam.size = {kW, kH};
  cam.intrinsics = {52.0, 51.5, 31.6, 23.8};
  cam.distortion = {-0.2, 0.05, 0.001, -0.001, 0.0, 0.03, 0.0, 0.0};
  for (const std::size_t count :
       {std::size_t{1}, std::size_t{3}, std::size_t{5}}) {
    std::vector<sensor::GpuFramePrep> preps;
    std::vector<std::vector<std::uint16_t>> raws(count);
    std::vector<std::vector<std::uint32_t>> words(count);
    std::vector<std::optional<sensor::RgbdFrame>> frames(count);
    for (std::size_t c = 0; c < count; ++c) {
      auto prep = sensor::GpuFramePrep::create(two.compute(),
                                               two.compute_allocator(), config);
      CHECK(prep.ok());
      preps.push_back(std::move(prep).value());
      raws[c].resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        raws[c][i] = static_cast<std::uint16_t>(500 + 97 * c + i % 251);
      }
      words[c] = packed_words(n, 0x9E3779B9u + static_cast<std::uint32_t>(c));
      sensor::RgbdFrame& f = frames[c].emplace(frame_of(raws[c], cam));
      f.color_packed = words[c].data();
      f.color_camera = cam;
    }
    auto prepared = sensor::GpuFramePrep::prepare_batch(preps, frames);
    if (!prepared) {
      std::fprintf(stderr, "%s\n", prepared.status().message().c_str());
    }
    CHECK(prepared.ok() && prepared->size() == count);
    for (std::size_t c = 0; c < count; ++c) {
      const sensor::DeviceFrame& f = *(*prepared)[c];
      CHECK(f.depth->sharing_mode() == VK_SHARING_MODE_CONCURRENT);
      CHECK(f.color->sharing_mode() == VK_SHARING_MODE_CONCURRENT);
      auto on_compute = vkc::test::read_back<float>(
          two.compute(), two.compute_allocator(), *f.depth, n);
      auto on_graphics = vkc::test::read_back<float>(
          two.graphics(), two.graphics_allocator(), *f.depth, n);
      CHECK(on_compute.ok() && on_graphics.ok());
      CHECK(*on_graphics == *on_compute);
      // Not a buffer of zeros: the pinhole centre reads its raw pixel.
      const std::size_t centre = std::size_t{24} * kW + 32;
      CHECK((*on_compute)[centre] > 0.0f);
      auto color_compute = vkc::test::read_back<std::uint32_t>(
          two.compute(), two.compute_allocator(), *f.color, n);
      auto color_graphics = vkc::test::read_back<std::uint32_t>(
          two.graphics(), two.graphics_allocator(), *f.color, n);
      CHECK(color_compute.ok() && color_graphics.ok());
      CHECK(*color_graphics == *color_compute);
    }
  }
  std::printf("  cross-family read: 1, 3 and 5 frames, depth and colour\n");
  return 0;
}

// Whether coordinate `p` rounds, half up, into [0, n); `*edge` is set within
// a hair of where the rounding changes, where either answer is the float's.
bool rounds_into(float p, std::uint32_t n, bool* edge) {
  const float r = p + 0.5f;
  if (std::fabs(r - std::round(r)) < 1e-3f) *edge = true;
  return std::floor(r) >= 0.0f && std::floor(r) < static_cast<float>(n);
}

// Whether source coordinate `s` lies in the colour pass's bounds for an
// n-pixel axis, [-0.5, n - 0.5]; `*edge` is set within a hair of either end.
bool within_picture(float s, std::uint32_t n, bool* edge) {
  const float hi = static_cast<float>(n) - 0.5f;
  if (std::fabs(s + 0.5f) < 1e-3f || std::fabs(s - hi) < 1e-3f) *edge = true;
  return s >= -0.5f && s <= hi;
}

// Whether the depth pass covers point `q` of depth camera `d`'s frame, on the
// host: in front of it, on a pinhole pixel that rounds into the picture, whose
// source in the captured image does too.
bool depth_covers(const camera::CameraModel& d, sensor::Vec3f q, bool* edge) {
  if (!(q.z > 0.0f)) return false;
  const float u = f32(d.intrinsics.fx) * (q.x / q.z) + f32(d.intrinsics.cx);
  const float v = f32(d.intrinsics.fy) * (q.y / q.z) + f32(d.intrinsics.cy);
  if (!(rounds_into(u, d.size.width, edge) &&
        rounds_into(v, d.size.height, edge))) {
    return false;
  }
  const sensor::Vec2f s =
      source_pixel(d, std::floor(u + 0.5f), std::floor(v + 0.5f));
  return rounds_into(s.x, d.size.width, edge) &&
         rounds_into(s.y, d.size.height, edge);
}

// Colour pixel (u, v)'s ray direction in `f`'s colour camera's frame.
sensor::Vec3f color_ray(const sensor::RgbdFrame& f, std::uint32_t u,
                        std::uint32_t v) {
  const camera::PinholeIntrinsics& k = f.color_camera.intrinsics;
  return sensor::Vec3f((static_cast<float>(u) - f32(k.cx)) / f32(k.fx),
                       (static_cast<float>(v) - f32(k.cy)) / f32(k.fy), 1.0f);
}

// Whether `f`'s depth camera's field of view holds colour pixel (u, v)'s ray,
// on the host: its direction, turned into the depth camera's frame by the
// rotation of rigid_inverse(depth_to_color), the baseline left out, is
// covered by the depth pass.
bool depth_sees(const sensor::RgbdFrame& f, std::uint32_t u, std::uint32_t v,
                bool* edge) {
  const sensor::Mat4f to_depth(camera::rigid_inverse(f.depth_to_color));
  return depth_covers(f.depth_camera,
                      sensor::Mat3f(to_depth) * color_ray(f, u, v), edge);
}

// Whether the depth camera covers colour pixel (u, v)'s ray where it lies `z`
// metres deep for the depth camera, a point the baseline moves: what a test
// that kept the baseline would ask.
bool depth_covers_at(const sensor::RgbdFrame& f, std::uint32_t u,
                     std::uint32_t v, float z, bool* edge) {
  const sensor::Mat4f to_depth(camera::rigid_inverse(f.depth_to_color));
  const sensor::Vec3f dir = sensor::Mat3f(to_depth) * color_ray(f, u, v);
  const sensor::Vec3f origin(to_depth[3]);
  const float t = (z - origin.z) / dir.z;
  if (!(t > 0.0f)) return false;
  return depth_covers(f.depth_camera, origin + t * dir, edge);
}

// Whether the colour pass's lens finds `f`'s colour pixel (u, v) in the
// picture, on the host.
bool recorded(const sensor::RgbdFrame& f, std::uint32_t u, std::uint32_t v,
              bool* edge) {
  const camera::CameraModel& c = f.color_camera;
  const sensor::Vec2f s =
      source_pixel(c, static_cast<float>(u), static_cast<float>(v));
  return within_picture(s.x, c.size.width, edge) &&
         within_picture(s.y, c.size.height, edge);
}

// The colour pixel point or direction `q` of `f`'s colour camera's frame
// rounds to, on the host, in (*cu, *cv); false where `q` is behind the camera
// or rounds outside the picture.
bool rounds_to(const sensor::RgbdFrame& f, sensor::Vec3f q, std::uint32_t* cu,
               std::uint32_t* cv, bool* edge) {
  if (!(q.z > 0.0f)) return false;
  const camera::CameraModel& c = f.color_camera;
  const float pu = f32(c.intrinsics.fx) * (q.x / q.z) + f32(c.intrinsics.cx);
  const float pv = f32(c.intrinsics.fy) * (q.y / q.z) + f32(c.intrinsics.cy);
  if (!(rounds_into(pu, c.size.width, edge) &&
        rounds_into(pv, c.size.height, edge))) {
    return false;
  }
  *cu = static_cast<std::uint32_t>(std::floor(pu + 0.5f));
  *cv = static_cast<std::uint32_t>(std::floor(pv + 0.5f));
  return true;
}

// The colour pixel `f`'s depth pixel (u, v), `z` metres deep, lands on, on
// the host, as rounds_to.
bool lands_on(const sensor::RgbdFrame& f, std::uint32_t u, std::uint32_t v,
              float z, std::uint32_t* cu, std::uint32_t* cv, bool* edge) {
  const camera::PinholeIntrinsics& d = f.depth_camera.intrinsics;
  const sensor::Vec4f p((static_cast<float>(u) - f32(d.cx)) * z / f32(d.fx),
                        (static_cast<float>(v) - f32(d.cy)) * z / f32(d.fy), z,
                        1.0f);
  return rounds_to(f, sensor::Vec3f(sensor::Mat4f(f.depth_to_color) * p), cu,
                   cv, edge);
}

// The colour pixel `f`'s depth pixel (u, v)'s direction, turned into the
// colour camera's frame by depth_to_color's rotation, the baseline left out,
// points at, on the host, as rounds_to.
bool points_at(const sensor::RgbdFrame& f, std::uint32_t u, std::uint32_t v,
               std::uint32_t* cu, std::uint32_t* cv, bool* edge) {
  const camera::PinholeIntrinsics& d = f.depth_camera.intrinsics;
  const sensor::Vec3f dir((static_cast<float>(u) - f32(d.cx)) / f32(d.fx),
                          (static_cast<float>(v) - f32(d.cy)) / f32(d.fy),
                          1.0f);
  return rounds_to(f, sensor::Mat3f(sensor::Mat4f(f.depth_to_color)) * dir, cu,
                   cv, edge);
}

// A frame with colour keeps depth only where its colour camera recorded it.
//   - Same intrinsics and pose, a pincushion colour lens: each depth pixel
//     lands on the colour pixel it shares, so depth must be zeroed exactly
//     where the colour's coverage byte is 0 -- the corners the lens missed.
//   - A narrower colour camera 10 cm to the right, inside the depth camera's
//     view, so all the colour is kept: the depth kept is where a host
//     projection puts both its point and its direction, the baseline left
//     out, inside the colour image. The point sets the left edge and the
//     direction the right, 26 columns inside where the other would.
//   - The same cameras with a depth offset: the mask tests the corrected
//     depth, where one testing the raw depth gets pixels wrong in every row
//     along the edge the point sets.
//   - A colour principal point a fraction of a pixel off: the image's edges
//     are where the rounded colour pixel leaves it, not the pixel itself.
//   - A frame without colour keeps every depth pixel, the same cameras'
//     included.
int test_depth_within_color(vkc::Device& device, vkc::Allocator& allocator) {
  auto masked = sensor::GpuFramePrep::create(device, allocator, unfiltered());
  CHECK(masked.ok());
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight, 1000);
  Planes planes = make_planes();

  camera::CameraModel pincushion = pinhole();
  pincushion.distortion.k1 = 0.3f;
  sensor::RgbdFrame same = frame_of(raw, pinhole());
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
  camera::CameraModel narrow = pinhole();
  narrow.intrinsics.fx *= 1.5;
  narrow.intrinsics.fy *= 1.5;
  sensor::RgbdFrame aside = frame_of(raw, pinhole());
  aside.color = planes.image(0.299f, 0.114f, true);
  aside.color_camera = narrow;
  // The colour camera 10 cm along +x; the depth camera stays at the origin.
  aside.color_to_world[3] = glm::dvec4(0.1, 0.0, 0.0, 1.0);
  aside.depth_to_color[3] = glm::dvec4(-0.1, 0.0, 0.0, 1.0);
  out = masked->prepare(aside);
  CHECK(out.ok());
  // The depth camera's view holds every colour pixel's ray, so all the
  // colour is kept.
  for (const std::uint32_t word : color_of(out.value())) {
    CHECK((word >> 24) == 0xFFu);
  }
  const std::vector<float> e = depth_of(out.value());
  const camera::PinholeIntrinsics dc = pinhole().intrinsics;
  const camera::PinholeIntrinsics nc = narrow.intrinsics;
  // Whether depth pixel (u, v)'s point at depth z lands in the colour image,
  // the point 0.1 m to the left for the colour camera; at an infinite z, its
  // direction. *edge is set within a hair of a rounding edge, where either
  // answer is the float's.
  const auto lands_in = [&](std::uint32_t u, std::uint32_t v, float z,
                            bool* edge) {
    const float x =
        (static_cast<float>(u) - f32(dc.cx)) / f32(dc.fx) - 0.1f / z;
    const float y = (static_cast<float>(v) - f32(dc.cy)) / f32(dc.fy);
    const float pu = f32(nc.fx) * x + f32(nc.cx) + 0.5f;
    const float pv = f32(nc.fy) * y + f32(nc.cy) + 0.5f;
    *edge = std::fmin(std::fabs(pu - std::round(pu)),
                      std::fabs(pv - std::round(pv))) < 1e-3f;
    return std::floor(pu) >= 0.0f && std::floor(pu) < kWidth &&
           std::floor(pv) >= 0.0f && std::floor(pv) < kHeight;
  };
  // Whether depth pixel (u, v) at depth z is kept: its point and its
  // direction both land in the colour image (all of which is kept).
  const auto kept_at = [&](std::uint32_t u, std::uint32_t v, float z,
                           bool* edge) {
    bool point_edge = false, direction_edge = false;
    const bool point = lands_in(u, v, z, &point_edge);
    const bool direction = lands_in(u, v, INFINITY, &direction_edge);
    *edge = point_edge || direction_edge;
    return point && direction;
  };
  std::size_t inside = 0, off = 0, point_alone = 0, direction_alone = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      bool edge = false, point_edge = false, direction_edge = false;
      const float z = 1000.0f * kScale;
      const bool in = kept_at(u, v, z, &edge);
      const bool point = lands_in(u, v, z, &point_edge);
      const bool direction = lands_in(u, v, INFINITY, &direction_edge);
      if (edge || point_edge || direction_edge) continue;
      inside += in ? 1 : 0;
      point_alone += point != in ? 1 : 0;
      direction_alone += direction != in ? 1 : 0;
      const float got = e[std::size_t{v} * kWidth + u];
      if (in ? got != z : got != 0.0f) ++off;
    }
  }
  std::printf(
      "  depth within colour, aside: %zu kept, %zu off; the point alone gets "
      "%zu wrong, the direction alone %zu\n",
      inside, off, point_alone, direction_alone);
  CHECK(off == 0);
  CHECK(inside > 1000 && inside < std::size_t{kWidth} * kHeight / 2);
  CHECK(point_alone > 10 * kHeight && direction_alone > 10 * kHeight);

  // The mask tests the corrected depth: the same cameras, a return at 1 m
  // that reads 5 cm too far. At 0.95 m the point lands about 2 px further
  // left in the colour camera (390 px x 0.1 m x (1 / 0.95 - 1)), so a mask
  // that tested the raw depth would get a column wrong at the side of the
  // kept region the point sets, in every row it spans.
  sensor::GpuFramePrepConfig offset = unfiltered();
  offset.depth_offset_m = 0.05f;
  auto offset_masked = sensor::GpuFramePrep::create(device, allocator, offset);
  CHECK(offset_masked.ok());
  const std::vector<std::uint16_t> metre(raw.size(), 1024);
  sensor::RgbdFrame distant = frame_of(metre, pinhole());
  distant.metres_per_unit = kExactScale;
  distant.color = planes.image(0.299f, 0.114f, true);
  distant.color_camera = narrow;
  distant.color_to_world[3] = glm::dvec4(0.1, 0.0, 0.0, 1.0);
  distant.depth_to_color[3] = glm::dvec4(-0.1, 0.0, 0.0, 1.0);
  out = offset_masked->prepare(distant);
  CHECK(out.ok());
  const std::vector<float> corrected_depth = depth_of(out.value());
  const float corrected = 1.0f - 0.05f;
  std::size_t off_corrected = 0, raw_would_differ = 0;
  for (std::uint32_t v = 0; v < kHeight; ++v) {
    for (std::uint32_t u = 0; u < kWidth; ++u) {
      bool edge = false, raw_edge = false;
      const bool in = kept_at(u, v, corrected, &edge);
      const bool raw_in = kept_at(u, v, 1.0f, &raw_edge);
      if (edge || raw_edge) continue;
      raw_would_differ += in != raw_in ? 1 : 0;
      const float got = corrected_depth[std::size_t{v} * kWidth + u];
      if (in ? got != corrected : got != 0.0f) ++off_corrected;
    }
  }
  std::printf(
      "  depth within colour, corrected: %zu off, %zu a raw mask "
      "gets wrong\n",
      off_corrected, raw_would_differ);
  CHECK(off_corrected == 0);
  CHECK(raw_would_differ > kHeight / 2);

  // The bounds are the rounded colour pixel's, as the colour pass marks
  // coverage, not the pixel's own: a colour principal point `shift` px off the
  // depth camera's puts depth pixel u at colour u + shift. At -0.25 the first
  // row and column land at -0.25 and round into the image, so they are kept;
  // at +0.6 the last land at size - 0.4 and round out of it, so they go.
  for (const float shift : {-0.25f, 0.6f}) {
    camera::CameraModel shifted = pinhole();
    shifted.intrinsics.cx += shift;
    shifted.intrinsics.cy += shift;
    sensor::RgbdFrame nudged = frame_of(raw, pinhole());
    nudged.color = planes.image(0.299f, 0.114f, true);
    nudged.color_camera = shifted;
    out = masked->prepare(nudged);
    CHECK(out.ok());
    const std::vector<float> g = depth_of(out.value());
    const auto rounds_in = [shift](std::uint32_t i, std::uint32_t n) {
      const float r = std::floor(static_cast<float>(i) + shift + 0.5f);
      return r >= 0.0f && r < static_cast<float>(n);
    };
    std::size_t wrong_edge = 0;
    for (std::uint32_t v = 0; v < kHeight; ++v) {
      for (std::uint32_t u = 0; u < kWidth; ++u) {
        const bool in = rounds_in(u, kWidth) && rounds_in(v, kHeight);
        const float got = g[std::size_t{v} * kWidth + u];
        if (in ? got != 1000.0f * kScale : got != 0.0f) ++wrong_edge;
      }
    }
    CHECK(wrong_edge == 0);
  }

  // The mask tests the depth the pass writes, mixed: raw columns alternating
  // 0.5 and 0.535 m through the barrel lens (the same intrinsics), so every
  // block mixes, up to 17.5 mm from its nearest pixel's depth, and the
  // narrower colour camera 10 cm aside. Each pixel is the depth the same
  // frame without colour gives where the host puts that depth's point and its
  // direction in the colour image, and 0 where not. A mask that tested the
  // nearest pixel's depth would decide pixels along the side the point sets
  // differently, about 100 of them.
  {
    const camera::CameraModel lens = lensed();
    std::vector<std::uint16_t> stripes(std::size_t{kWidth} * kHeight);
    for (std::size_t i = 0; i < stripes.size(); ++i) {
      stripes[i] = (i % kWidth) % 2 == 0 ? 5000 : 5350;
    }
    sensor::RgbdFrame striped = frame_of(stripes, lens);
    striped.metres_per_unit = kFineScale;
    auto all = masked->prepare(striped);  // no colour yet: nothing masked
    CHECK(all.ok());
    const std::vector<float> unmasked = depth_of(all.value());
    striped.color = planes.image(0.299f, 0.114f, true);
    striped.color_camera = narrow;
    striped.color_to_world[3] = glm::dvec4(0.1, 0.0, 0.0, 1.0);
    striped.depth_to_color[3] = glm::dvec4(-0.1, 0.0, 0.0, 1.0);
    out = masked->prepare(striped);
    CHECK(out.ok());
    for (const std::uint32_t word : color_of(out.value())) {
      CHECK((word >> 24) == 0xFFu);
    }
    const std::vector<float> m = depth_of(out.value());
    CHECK(m.size() == unmasked.size());
    std::size_t kept_mixed = 0, off_mixed = 0, nearest_would_differ = 0;
    for (std::uint32_t v = 0; v < kHeight; ++v) {
      for (std::uint32_t u = 0; u < kWidth; ++u) {
        const std::size_t i = std::size_t{v} * kWidth + u;
        bool edge = false, nearest_edge = false;
        const bool in = kept_at(u, v, unmasked[i], &edge);
        if (edge || unmasked[i] == 0.0f) continue;
        kept_mixed += in ? 1 : 0;
        if (in ? m[i] != unmasked[i] : m[i] != 0.0f) ++off_mixed;
        const DepthSample ref =
            reference_depth(stripes, lens, kFineScale, 0.0f, u, v);
        const bool nearest_in = kept_at(u, v, ref.nearest, &nearest_edge);
        nearest_would_differ += !nearest_edge && nearest_in != in ? 1 : 0;
      }
    }
    std::printf(
        "  depth within colour, mixed: %zu kept, %zu off, %zu a "
        "nearest-pixel mask gets wrong\n",
        kept_mixed, off_mixed, nearest_would_differ);
    CHECK(off_mixed == 0);
    CHECK(kept_mixed > 1000 && kept_mixed < std::size_t{kWidth} * kHeight / 2);
    CHECK(nearest_would_differ > 50);
  }

  // The aside frame without its colour: nothing zeroed.
  sensor::RgbdFrame depth_only = aside;
  depth_only.color = {};
  CHECK(!depth_only.has_color());
  auto all = masked->prepare(depth_only);
  CHECK(all.ok());
  for (const float z : depth_of(all.value())) CHECK(z == 1000.0f * kScale);
  return 0;
}

// A frame with colour keeps only what both its cameras see, by their fields
// of view: colour where its ray's direction lies inside the depth camera's
// view, the baseline left out, a 0 word elsewhere, and depth where its
// direction lies inside the colour camera's view and its point lands on
// colour that is kept. Each case is held to the host's tests at every pixel
// off a rounding edge, the kept colour to the words the same picture gives
// beside a depth camera that sees all of it (the colour camera's own, in its
// place), and every kept depth pixel to colour with its coverage byte set:
//   - a narrower depth camera 5 cm to the right: colour dropped around the
//     depth camera's view, where a test of the ray's point at 0.25 m or 1 m,
//     which the baseline moves, would decide hundreds of pixels otherwise;
//     depth zeroed at its right, where the colour its point lands on is
//     dropped; for planes in a buffer and NV12 plane images alike, and the
//     same words and depth whatever the frame's depth range;
//   - colour narrower than depth, 2 cm aside: every colour pixel kept, and
//     depth zeroed where its direction leaves the colour view, a few columns
//     more than its point would;
//   - a pincushion depth lens beside the same camera undistorted: colour
//     dropped where the depth lens misses the picture, the corners;
//   - colour wider than depth, as a RealSense D455's (90 x 65 degrees against
//     87 x 58), 1.5 cm apart: all four edges dropped; with the depth on the
//     device, the same words and depth;
//   - a Femto Mega's cameras, scaled down, the depth camera tilted 6.3 degrees
//     below the colour's axis: depth loses a band at the top and a wider one
//     at the bottom, outside the colour camera's view, and its left side,
//     where the colour its point lands on 1 m away is dropped; colour loses
//     its sides, outside the depth camera's view.
// A frame with colour and no depth is refused, so no colour goes unmasked.
int test_color_within_depth(sensor::GpuFramePrep& prep) {
  const std::vector<std::uint16_t> raw(std::size_t{320} * 288, 1000);  // 1 m
  const float metre = 1000.0f * kScale;
  struct Crop {
    // colour: kept, dropped, wrong, and those the ray's point 0.25 m or 1 m
    // deep for the depth camera would decide otherwise
    std::size_t kept = 0, dropped = 0, off = 0, at_near = 0, at_metre = 0;
    // depth: kept; zeroed where its colour is dropped; zeroed where its point
    // lands on the colour picture but its direction does not; wrong; and kept
    // on a colour pixel whose coverage byte is 0
    std::size_t depth_kept = 0, cropped = 0, aimed_out = 0, depth_off = 0,
                stranded = 0;
    bool held() const { return off == 0 && depth_off == 0 && stranded == 0; }
  };
  // `f` prepared with colour `color`, against the host's tests; its colour in
  // `words` and its depth in `depth`.
  const auto crop_of = [&](sensor::RgbdFrame f, const sensor::YuvImage& color,
                           std::vector<std::uint32_t>* words,
                           std::vector<float>* depth) {
    Crop c;
    f.color = color;
    sensor::RgbdFrame whole = f;
    whole.depth_camera = f.color_camera;
    whole.depth_to_color = camera::Mat4d(1.0);
    auto all = prep.prepare(whole);
    auto out = prep.prepare(f);
    if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
    if (!all.ok() || !out.ok()) {
      c.off = ~std::size_t{0};
      return c;
    }
    const std::vector<std::uint32_t> unmasked = color_of(all.value());
    *words = color_of(out.value());
    *depth = depth_of(out.value());
    const std::uint32_t w = f.color_camera.size.width;
    for (std::uint32_t v = 0; v < f.color_camera.size.height; ++v) {
      for (std::uint32_t u = 0; u < w; ++u) {
        const std::size_t i = std::size_t{v} * w + u;
        if ((unmasked[i] >> 24) != 0xFFu) ++c.off;  // the whole picture kept
        bool edge = false;
        const bool in = depth_sees(f, u, v, &edge);
        const bool near_in = depth_covers_at(f, u, v, 0.25f, &edge);
        const bool metre_in = depth_covers_at(f, u, v, metre, &edge);
        if (edge) continue;
        (in ? c.kept : c.dropped) += 1;
        if ((*words)[i] != (in ? unmasked[i] : 0u)) ++c.off;
        c.at_near += near_in != in ? 1 : 0;
        c.at_metre += metre_in != in ? 1 : 0;
      }
    }
    const camera::CameraModel& d = f.depth_camera;
    for (std::uint32_t v = 0; v < d.size.height; ++v) {
      for (std::uint32_t u = 0; u < d.size.width; ++u) {
        const float got = (*depth)[std::size_t{v} * d.size.width + u];
        std::uint32_t cu = 0, cv = 0, au = 0, av = 0;
        bool edge = false;
        if (got > 0.0f) {  // the device's own depth and colour agree
          const bool on = lands_on(f, u, v, got, &cu, &cv, &edge);
          if (!edge &&
              !(on && ((*words)[std::size_t{cv} * w + cu] >> 24) == 0xFFu)) {
            ++c.stranded;
          }
        }
        // Whether the lens returns the pixel, its point lands on a pixel the
        // colour camera records, its direction does too, and the colour pass
        // keeps the pixel its point lands on.
        edge = false;
        const sensor::Vec2f s =
            source_pixel(d, static_cast<float>(u), static_cast<float>(v));
        const bool returned = rounds_into(s.x, d.size.width, &edge) &&
                              rounds_into(s.y, d.size.height, &edge);
        const bool lands = returned &&
                           lands_on(f, u, v, metre, &cu, &cv, &edge) &&
                           recorded(f, cu, cv, &edge);
        const bool aimed = returned && points_at(f, u, v, &au, &av, &edge) &&
                           recorded(f, au, av, &edge);
        const bool keep = lands && aimed && depth_sees(f, cu, cv, &edge);
        if (edge) continue;
        c.depth_kept += keep ? 1 : 0;
        c.cropped += lands && aimed && !keep ? 1 : 0;
        c.aimed_out += lands && !aimed ? 1 : 0;
        if (got != (keep ? metre : 0.0f)) ++c.depth_off;
      }
    }
    return c;
  };
  const auto report = [](const char* name, const Crop& c) {
    std::printf(
        "  colour within depth, %s: colour %zu kept, %zu dropped, %zu off; "
        "depth %zu kept, %zu zeroed under dropped colour, %zu outside the "
        "colour view, %zu off, %zu kept on dropped colour\n",
        name, c.kept, c.dropped, c.off, c.depth_kept, c.cropped, c.aimed_out,
        c.depth_off, c.stranded);
  };
  const auto colored = [&](const camera::CameraModel& depth,
                           const camera::CameraModel& color, double right) {
    sensor::RgbdFrame f = frame_of(raw, depth);
    f.color_camera = color;
    f.depth_to_color[3] = glm::dvec4(right, 0.0, 0.0, 1.0);
    return f;
  };
  const auto camera_of = [](std::uint32_t w, std::uint32_t h, double f,
                            double cx, double cy) {
    camera::CameraModel c;
    c.size = {w, h};
    c.intrinsics = {f, f, cx, cy};
    return c;
  };
  const auto word = [](const std::vector<std::uint32_t>& words, std::uint32_t w,
                       std::uint32_t u, std::uint32_t v) {
    return words[std::size_t{v} * w + u];
  };
  std::vector<std::uint32_t> words;
  std::vector<float> depth;

  // The depth camera narrower, 5 cm to the right.
  {
    camera::CameraModel narrower = pinhole();
    narrower.intrinsics.fx *= 1.25;
    narrower.intrinsics.fy *= 1.25;
    sensor::RgbdFrame f = colored(narrower, pinhole(), 0.05);
    const Planes planes = patterned(kWidth, kHeight);
    for (const sensor::YuvImage& color :
         {planes.image(0.2126f, 0.0722f, false), nv12_images(planes)}) {
      const Crop c = crop_of(f, color, &words, &depth);
      report(color.image[0] != nullptr ? "images" : "buffer", c);
      std::printf(
          "    the ray's point 0.25 m deep gets %zu wrong, 1 m deep %zu\n",
          c.at_near, c.at_metre);
      CHECK(c.held());
      CHECK(c.kept > 20000 && c.dropped > 20000);
      CHECK(c.at_near > kHeight && c.at_metre > kHeight);
      CHECK(c.depth_kept > 20000 && c.cropped > kHeight);
    }
    // The depth range has no part in the overlap.
    f.color = planes.image(0.2126f, 0.0722f, false);
    auto wide = prep.prepare(f);
    CHECK(wide.ok());
    words = color_of(wide.value());
    depth = depth_of(wide.value());
    f.min_depth = 0.5f;
    f.max_depth = 4.0f;
    auto ranged = prep.prepare(f);
    CHECK(ranged.ok() && color_of(ranged.value()) == words &&
          depth_of(ranged.value()) == depth);
  }

  // Colour narrower than depth, 2 cm aside: no colour dropped, and depth
  // where its direction leaves the colour view, at one side a few columns
  // inside where its point does.
  {
    camera::CameraModel narrow = pinhole();
    narrow.intrinsics.fx *= 1.5;
    narrow.intrinsics.fy *= 1.5;
    const sensor::RgbdFrame f = colored(pinhole(), narrow, 0.02);
    const Crop c =
        crop_of(f, patterned(kWidth, kHeight).image(0.2126f, 0.0722f, false),
                &words, &depth);
    report("narrower colour", c);
    CHECK(c.held() && c.dropped == 0 && c.cropped == 0);
    CHECK(c.aimed_out > kHeight);
  }

  // A pincushion depth lens, the colour camera the same one undistorted, in
  // the same place: colour is dropped exactly where the depth lens misses the
  // picture.
  {
    camera::CameraModel pincushion = pinhole();
    pincushion.distortion.k1 = 0.3f;
    const sensor::RgbdFrame f = colored(pincushion, pinhole(), 0.0);
    const Crop c =
        crop_of(f, patterned(kWidth, kHeight).image(0.2126f, 0.0722f, false),
                &words, &depth);
    report("depth lens", c);
    CHECK(c.held() && c.dropped > 1000 && c.kept > 50000);
    CHECK(word(words, kWidth, 0, 0) == 0u);
    CHECK((word(words, kWidth, kWidth / 2, kHeight / 2) >> 24) == 0xFFu);
  }

  // Colour wider than depth, as a RealSense D455's: 87 x 56 degrees of depth
  // at 320 x 180, 90 x 64 of colour at 320 x 200.
  {
    const camera::CameraModel d455 = camera_of(320, 180, 168.6, 159.5, 89.5);
    const camera::CameraModel color = camera_of(320, 200, 160.0, 159.5, 99.5);
    const sensor::RgbdFrame f = colored(d455, color, -0.015);
    const sensor::YuvImage picture =
        patterned(320, 200).image(0.2126f, 0.0722f, false);
    const Crop c = crop_of(f, picture, &words, &depth);
    report("wider colour", c);
    CHECK(c.held());
    CHECK(word(words, 320, 0, 100) == 0u && word(words, 320, 319, 100) == 0u);
    CHECK(word(words, 320, 160, 0) == 0u && word(words, 320, 160, 199) == 0u);
    CHECK((word(words, 320, 160, 100) >> 24) == 0xFFu);
    // The depth on the device: the same colour and depth.
    sensor::RgbdFrame device = on_device_depth(
        f,
        depth_buffer(
            std::vector<std::uint16_t>(raw.begin(), raw.begin() + 320 * 180), 0,
            0),
        0);
    device.color = picture;
    auto got = prep.prepare(device);
    CHECK(got.ok() && color_of(got.value()) == words &&
          depth_of(got.value()) == depth);
  }

  // A Femto Mega's cameras, scaled down: depth 640 x 576 at f 504 by 2,
  // colour 3840 x 2160 at f 2239 by 12, from the cameras' intrinsics, the
  // colour camera 32 mm to the depth camera's right, and the depth camera
  // turned 0.11 rad down about x, as the factory calibrations have it.
  {
    const camera::CameraModel femto = camera_of(320, 288, 252.0, 159.5, 167.0);
    const camera::CameraModel color = camera_of(320, 180, 186.6, 159.5, 86.6);
    sensor::RgbdFrame f = colored(femto, color, -0.032);
    f.depth_to_color =
        camera::Mat4d(camera::rotation_from_rodrigues({-0.11, 0.0, 0.0}));
    f.depth_to_color[3] = glm::dvec4(-0.032, -0.002, 0.004, 1.0);
    const Crop c = crop_of(
        f, patterned(320, 180).image(0.2126f, 0.0722f, false), &words, &depth);
    report("Femto Mega", c);
    CHECK(c.held() && c.cropped > 1000);
    CHECK(word(words, 320, 0, 90) == 0u && word(words, 320, 319, 90) == 0u);
    CHECK((word(words, 320, 160, 0) >> 24) == 0xFFu);
    CHECK((word(words, 320, 160, 179) >> 24) == 0xFFu);
    // The depth: whole rows zeroed at the top and the bottom, and the left
    // side of the rows between, the baseline's parallax 1 m away.
    const auto zero_row = [&](std::uint32_t v) {
      const auto row = depth.begin() + std::ptrdiff_t{v} * 320;
      return std::all_of(row, row + 320, [](float z) { return z == 0.0f; });
    };
    std::uint32_t top = 0, bottom = 0, left = 0;
    while (top < 288 && zero_row(top)) ++top;
    while (bottom < 288 && zero_row(287 - bottom)) ++bottom;
    const float* mid = depth.data() + std::size_t{144} * 320;
    while (left < 320 && mid[left] == 0.0f) ++left;
    std::printf(
        "    depth's top %u and bottom %u rows zeroed, and row 144's first "
        "%u\n",
        top, bottom, left);
    CHECK(top > 10 && top < 18 && bottom > 22 && bottom < 32);
    CHECK(left > 4 && left < 12 && mid[160] == metre && mid[319] == metre);
  }

  // Colour with no depth.
  sensor::RgbdFrame color_only = frame_of(raw, pinhole());
  color_only.depth = nullptr;
  color_only.color = make_planes().image(0.299f, 0.114f, true);
  color_only.color_camera = pinhole();
  CHECK(prep.prepare(color_only).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  auto prep =
      sensor::GpuFramePrep::create(gpu.device, gpu.allocator, unfiltered());
  if (!prep) std::fprintf(stderr, "%s\n", prep.status().message().c_str());
  CHECK(prep.ok());

  if (test_pinhole(prep.value()) != 0) return 1;
  if (test_depth_offset(gpu.device, gpu.allocator) != 0) return 1;
  if (test_undistorts(prep.value()) != 0) return 1;
  if (test_tilted_plane(prep.value()) != 0) return 1;
  if (test_depth_steps(prep.value()) != 0) return 1;
  if (test_depth_holes(gpu.device, gpu.allocator) != 0) return 1;
  if (test_matches_reference(prep.value(), kWidth, kHeight, false) != 0) {
    return 1;
  }
  // Odd, so the chroma planes' offsets are rounded up to a word.
  if (test_matches_reference(prep.value(), kWidth + 1, kHeight + 1, true) !=
      0) {
    return 1;
  }
  if (test_packed(prep.value()) != 0) return 1;
  if (test_chroma_locations(prep.value()) != 0) return 1;
  for (auto location :
       {sensor::ChromaLocation::Left, sensor::ChromaLocation::Center}) {
    if (test_layouts(prep.value(), location) != 0) return 1;
  }
  if (test_yuyv(prep.value()) != 0) return 1;
  if (test_luma(prep.value()) != 0) return 1;
  if (test_coverage(prep.value()) != 0) return 1;
  if (test_poses(prep.value()) != 0) return 1;
  if (test_refusals(prep.value()) != 0) return 1;
  if (test_frames_hold_buffers(prep.value()) != 0) return 1;
  if (test_depth_noise(prep.value()) != 0) return 1;
  if (test_prepare_batch(gpu.device, gpu.allocator) != 0) return 1;
  if (test_record(gpu.device, gpu.allocator) != 0) return 1;
  if (test_batch_timing(gpu.device, gpu.allocator) != 0) return 1;
  if (test_queue_families(gpu.device, gpu.allocator) != 0) return 1;
  if (test_cross_family() != 0) return 1;
  if (test_depth_within_color(gpu.device, gpu.allocator) != 0) return 1;
  if (test_color_within_depth(prep.value()) != 0) return 1;
  if (test_device_depth(gpu.device, gpu.allocator) != 0) return 1;

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
  auto live =
      sensor::GpuFramePrep::create(gpu.device, gpu.allocator, unfiltered());
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

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
