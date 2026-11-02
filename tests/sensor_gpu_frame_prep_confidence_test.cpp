// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep's confidence gate (RgbdFrame::depth_confidence and
// min_confidence) on synthetic frames, no camera:
//   - through a pinhole, a sample is kept exactly where its level reaches the
//     floor and is 0 elsewhere: at ARKit's levels and floors, and at every
//     floor from 1 to 255 over every level, a level-0 sample dropped at each;
//   - the gate is per raw sample, before the depth offset, the mix, the
//     colour-view test and the outlier filter: a gated frame through a lens
//     prepares bit for bit as the same frame with its gated samples zeroed on
//     the host, where gated samples lie within the mix's span of their kept
//     neighbours (so a gate after the mix would blend them in) or 9 m away;
//   - a frame without a plane, with a plane all at the floor, and with a
//     plane at levels 0 and 1 and a floor of 0 prepare bit for bit alike;
//   - a gated frame with host colour, its depth and plane ending part-way
//     through a word, keeps its colour as without the plane, and its depth as
//     the zeroed frame's;
//   - gated and ungated frames in one set, recorded or prepared, each as
//     alone;
//   - a plane beside device depth, and a floor without a plane, refused
//     before anything is recorded.
// Runs on the real driver; exits 0 (skip) where no device is present.

#include <cstdint>
#include <cstdio>
#include <cstring>
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
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

constexpr float kScale = 0.001f;  // metres per unit: millimetres

// ARKit's levels.
constexpr std::uint8_t kLow = 0, kMedium = 1, kHigh = 2;

vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

camera::CameraModel pinhole(std::uint32_t w, std::uint32_t h) {
  camera::CameraModel c;
  c.size = {w, h};
  c.intrinsics = {0.83 * w, 0.83 * w, 0.497 * w, 0.494 * h};
  return c;
}

// A barrel lens, so each pinhole pixel samples its raw image part-way
// between raw pixels.
camera::CameraModel lensed(std::uint32_t w, std::uint32_t h) {
  camera::CameraModel c = pinhole(w, h);
  c.distortion = {-0.21, 0.06, 0.0011, -0.0007, 0.0, 0.0, 0.0, 0.0};
  return c;
}

// A hash of i in [0, 1).
float unit(std::uint32_t i) {
  std::uint32_t x = i * 2654435761u + 0x9E3779B9u;
  x ^= x >> 15;
  x *= 0x2C1B3C6Du;
  x ^= x >> 12;
  return static_cast<float>(x >> 8) / 16777216.0f;
}

// A plane slanted across the view, 1.2 m at the left to 1.8 m at the right
// and 0.2 m deeper at the bottom: neighbours differ by a few millimetres,
// well within the mix's 4 cm, so every block of kept samples is mixed.
std::vector<std::uint16_t> slanted(const camera::CameraModel& c) {
  const std::uint32_t w = c.size.width, h = c.size.height;
  std::vector<std::uint16_t> raw(std::size_t{w} * h);
  for (std::uint32_t v = 0; v < h; ++v) {
    for (std::uint32_t u = 0; u < w; ++u) {
      const double z = 1200.0 + 600.0 * u / w + 200.0 * v / h;
      raw[std::size_t{v} * w + u] = static_cast<std::uint16_t>(z);
    }
  }
  return raw;
}

// Levels for `count` samples by hash: about 30% medium, 5% low, the rest
// high.
std::vector<std::uint8_t> scattered_levels(std::size_t count,
                                           std::uint32_t seed) {
  std::vector<std::uint8_t> level(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float r = unit(static_cast<std::uint32_t>(i) * 3u + seed);
    level[i] = r < 0.05f ? kLow : r < 0.35f ? kMedium : kHigh;
  }
  return level;
}

// `raw` with every sample below `floor` zeroed on the host, as the gate
// should leave it.
std::vector<std::uint16_t> zeroed(std::vector<std::uint16_t> raw,
                                  const std::vector<std::uint8_t>& level,
                                  std::uint8_t floor) {
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (level[i] < floor) raw[i] = 0;
  }
  return raw;
}

// `count` packed words, random in every byte.
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

// `f` gated by `level` at `floor`.
sensor::RgbdFrame gated(sensor::RgbdFrame f,
                        const std::vector<std::uint8_t>& level,
                        std::uint8_t floor) {
  f.depth_confidence = level.data();
  f.min_confidence = floor;
  return f;
}

// The pass's config with the outlier filter off, where a test holds depth to
// the depth pass alone.
sensor::GpuFramePrepConfig unfiltered() {
  sensor::GpuFramePrepConfig config;
  config.depth_outliers.enabled = false;
  return config;
}

template <typename T>
std::vector<T> read(const vkc::Buffer& buffer, std::size_t count) {
  auto out = vkc::test::read_back<T>(*g_device, *g_allocator, buffer, count);
  if (!out) {
    std::fprintf(stderr, "read back: %s\n", out.status().message().c_str());
    return {};
  }
  return std::move(out).value();
}

// A prepared frame's depth, as floats and as the words the pass wrote, for a
// comparison bit for bit; and its colour words.
std::vector<float> depth_of(const sensor::DeviceFrame& f) {
  return read<float>(*f.depth,
                     std::size_t{f.depth_camera.width} * f.depth_camera.height);
}
std::vector<std::uint32_t> depth_words(const sensor::DeviceFrame& f) {
  return read<std::uint32_t>(
      *f.depth, std::size_t{f.depth_camera.width} * f.depth_camera.height);
}
std::vector<std::uint32_t> color_words(const sensor::DeviceFrame& f) {
  return read<std::uint32_t>(
      *f.color, std::size_t{f.color_camera.width} * f.color_camera.height);
}

// `frame` prepared by `prep`; its depth words and, with colour, its colour
// words. Both empty on a failure.
struct Prepared {
  std::vector<std::uint32_t> depth, color;
};
Prepared prepared(sensor::GpuFramePrep& prep, const sensor::RgbdFrame& frame) {
  auto out = prep.prepare(frame);
  if (!out) {
    std::fprintf(stderr, "prepare: %s\n", out.status().message().c_str());
    return {};
  }
  Prepared p;
  p.depth = depth_words(out.value());
  if (out->has_color()) p.color = color_words(out.value());
  return p;
}

std::size_t differing(const std::vector<std::uint32_t>& a,
                      const std::vector<std::uint32_t>& b) {
  if (a.size() != b.size()) return ~std::size_t{0};
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

// Through a pinhole each pixel is its own raw sample, so the gate is exact:
// 1.5 m where the level reaches the floor, 0 elsewhere. ARKit's levels, by
// column, at its floors and one past them; then levels 0-255, scattered, at
// every floor from 1 to 255, where a level-0 sample is never kept.
int test_gate(vkc::Device& device, vkc::Allocator& allocator) {
  auto made = sensor::GpuFramePrep::create(device, allocator, unfiltered());
  CHECK(made.ok());
  sensor::GpuFramePrep& prep = made.value();
  constexpr std::uint32_t kW = 64, kH = 48;
  const camera::CameraModel cam = pinhole(kW, kH);
  const std::vector<std::uint16_t> raw(std::size_t{kW} * kH, 1500);
  const float kept = static_cast<float>(raw[0]) * kScale;
  std::vector<std::uint8_t> arkit(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    arkit[i] = static_cast<std::uint8_t>((i % kW) % 3);
  }
  const sensor::RgbdFrame plain = frame_of(raw, cam);
  for (const std::uint8_t floor : {std::uint8_t{1}, kHigh, std::uint8_t{3}}) {
    auto out = prep.prepare(gated(plain, arkit, floor));
    CHECK(out.ok());
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == raw.size());
    std::size_t off = 0, n_kept = 0, want_kept = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
      const bool keep = arkit[i] >= floor;
      off += d[i] != (keep ? kept : 0.0f);
      n_kept += d[i] != 0.0f;
      want_kept += keep;
    }
    std::printf("  ARKit's levels at floor %u: %zu of %zu kept, %zu off\n",
                floor, n_kept, d.size(), off);
    CHECK(off == 0 && n_kept == want_kept);
  }

  // Every level, each pixel's i * 37 mod 256 (37 is odd, so every 256
  // consecutive pixels hold each level once), at every floor.
  std::vector<std::uint8_t> every(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    every[i] = static_cast<std::uint8_t>((i * 37u) % 256u);
  }
  std::size_t off = 0, low_kept = 0, lows = 0;
  for (std::uint32_t floor = 1; floor <= 255; ++floor) {
    auto out =
        prep.prepare(gated(plain, every, static_cast<std::uint8_t>(floor)));
    CHECK(out.ok());
    const std::vector<float> d = depth_of(out.value());
    CHECK(d.size() == raw.size());
    for (std::size_t i = 0; i < d.size(); ++i) {
      off += d[i] != (every[i] >= floor ? kept : 0.0f);
      if (every[i] == kLow) {
        lows += 1;
        low_kept += d[i] != 0.0f;
      }
    }
  }
  std::printf(
      "  every floor 1-255 over levels 0-255: %zu off, %zu of %zu "
      "level-0 samples kept\n",
      off, low_kept, lows);
  CHECK(off == 0 && low_kept == 0 && lows == 255 * (raw.size() / 256));

  // Warm, gated frames allocate nothing that moves bytes, each prepare one
  // submit.
  const sensor::RgbdFrame f = gated(plain, arkit, kHigh);
  CHECK(prep.prepare(f).ok());
  const vkc::DeviceCounters warm = device.counters();
  for (int i = 0; i < 20; ++i) CHECK(prep.prepare(f).ok());
  const vkc::DeviceCounters now = device.counters();
  CHECK(now.staging_buffers == warm.staging_buffers);
  CHECK(now.submits - warm.submits == 20);
  return 0;
}

// Per raw sample, before anything else reads it: through a lens, on a slanted
// plane, a gated frame prepares bit for bit as the same frame with its gated
// samples zeroed on the host and no plane. Most gated samples lie on the
// plane, within the mix's span of their kept neighbours, so a gate on the
// mixed value would blend them in; a tenth hold 9 m, which must never show.
// With the outlier filter off, then on beside a depth offset and colour,
// whose view test and filter must see the gated samples as holes.
int test_before_mix(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cam = lensed(160, 120);
  std::vector<std::uint16_t> raw = slanted(cam);
  const std::vector<std::uint8_t> level = scattered_levels(raw.size(), 1);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto k = static_cast<std::uint32_t>(i);
    if (level[i] < kHigh && unit(k * 5u + 2u) < 0.1f) raw[i] = 9000;
    if (i % 53 == 7) raw[i] = 0;  // the sensor's own holes
  }
  const std::vector<std::uint16_t> want_raw = zeroed(raw, level, kHigh);
  const std::vector<std::uint32_t> words = packed_words(raw.size(), 5);
  camera::CameraModel color = pinhole(160, 120);
  color.intrinsics.fx *= 1.15;  // a narrower view, so the view test cuts
  color.intrinsics.fy *= 1.15;

  sensor::GpuFramePrepConfig filtered;  // the default: the filter on
  filtered.depth_offset_m = 0.012f;
  const struct {
    const char* what;
    sensor::GpuFramePrepConfig config;
    bool with_color;
  } cases[] = {{"filter off", unfiltered(), false},
               {"offset, colour, filter on", filtered, true}};
  for (const auto& c : cases) {
    auto a = sensor::GpuFramePrep::create(device, allocator, c.config);
    auto b = sensor::GpuFramePrep::create(device, allocator, c.config);
    CHECK(a.ok() && b.ok());
    sensor::RgbdFrame f = frame_of(raw, cam);
    sensor::RgbdFrame want_f = frame_of(want_raw, cam);
    if (c.with_color) {
      for (sensor::RgbdFrame* x : {&f, &want_f}) {
        x->color_packed = words.data();
        x->color_camera = color;
      }
    }
    const Prepared got = prepared(a.value(), gated(f, level, kHigh));
    const Prepared want = prepared(b.value(), want_f);
    const Prepared ungated = prepared(b.value(), f);
    CHECK(!got.depth.empty() && !want.depth.empty() && !ungated.depth.empty());
    const std::size_t off = differing(got.depth, want.depth);
    const std::size_t changed = differing(ungated.depth, want.depth);
    std::size_t far = 0, returns = 0;
    for (const std::uint32_t w : got.depth) {
      float z;
      std::memcpy(&z, &w, sizeof z);
      far += z > 5.0f;
      returns += z > 0.0f;
    }
    std::printf(
        "  %s: %zu off; the gate changes %zu of %zu, %zu returns, "
        "%zu past 5 m\n",
        c.what, off, changed, got.depth.size(), returns, far);
    CHECK(off == 0);
    CHECK(got.color == want.color);
    CHECK(changed > got.depth.size() / 4 && returns > got.depth.size() / 3);
    CHECK(far == 0);
  }
  return 0;
}

// Bit-identical: one frame through a lens, with holes, an edge, colour, a
// depth offset and the outlier filter, prepared without a plane, with a plane
// all high at a floor of 2, and with a plane of levels 0 and 1 at a floor of
// 0, which is read nowhere.
int test_unchanged(vkc::Device& device, vkc::Allocator& allocator) {
  sensor::GpuFramePrepConfig config;
  config.depth_offset_m = 0.008f;
  auto made = sensor::GpuFramePrep::create(device, allocator, config);
  CHECK(made.ok());
  sensor::GpuFramePrep& prep = made.value();
  const camera::CameraModel cam = lensed(160, 120);
  std::vector<std::uint16_t> raw = slanted(cam);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const std::size_t u = i % 160, v = i / 160;
    if (u > 60 && u < 100 && v > 40 && v < 80) raw[i] = 700;  // a box
    if (i % 29 == 3) raw[i] = 0;
  }
  const std::vector<std::uint32_t> words = packed_words(raw.size(), 9);
  sensor::RgbdFrame f = frame_of(raw, cam);
  f.color_packed = words.data();
  f.color_camera = lensed(160, 120);
  const std::vector<std::uint8_t> high(raw.size(), kHigh);
  std::vector<std::uint8_t> low(raw.size());
  for (std::size_t i = 0; i < low.size(); ++i) {
    low[i] = static_cast<std::uint8_t>(i % 2);
  }
  const Prepared none = prepared(prep, f);
  const Prepared all_high = prepared(prep, gated(f, high, kHigh));
  const Prepared floor_0 = prepared(prep, gated(f, low, 0));
  CHECK(!none.depth.empty() && !none.color.empty());
  std::size_t holes = 0;
  for (const std::uint32_t w : none.depth) holes += w == 0u;
  std::printf("  unchanged: %zu and %zu words off, %zu holes of %zu\n",
              differing(none.depth, all_high.depth),
              differing(none.depth, floor_0.depth), holes, none.depth.size());
  CHECK(all_high.depth == none.depth && floor_0.depth == none.depth);
  CHECK(all_high.color == none.color && floor_0.color == none.color);
  CHECK(holes > none.depth.size() / 50 && holes < none.depth.size() / 2);
  return 0;
}

// The slot: host colour goes after the plane. At sizes whose depth (2 bytes a
// sample) and plane (1) end part-way through a word, and one where both end
// on a word, a gated frame's colour is the colour without the plane, and its
// depth the zeroed frame's.
int test_slot(vkc::Device& device, vkc::Allocator& allocator) {
  auto made = sensor::GpuFramePrep::create(device, allocator, unfiltered());
  auto made_reference =
      sensor::GpuFramePrep::create(device, allocator, unfiltered());
  CHECK(made.ok() && made_reference.ok());
  const std::uint32_t sizes[][2] = {{63, 47}, {3, 3}, {64, 48}, {65, 49}};
  for (const auto& size : sizes) {
    const camera::CameraModel cam = pinhole(size[0], size[1]);
    const std::vector<std::uint16_t> raw = slanted(cam);
    const std::vector<std::uint8_t> level =
        scattered_levels(raw.size(), size[0]);
    const std::vector<std::uint16_t> want_raw = zeroed(raw, level, kHigh);
    const std::vector<std::uint32_t> words =
        packed_words(raw.size(), 11 + size[1]);
    sensor::RgbdFrame f = frame_of(raw, cam);
    sensor::RgbdFrame want_f = frame_of(want_raw, cam);
    for (sensor::RgbdFrame* x : {&f, &want_f}) {
      x->color_packed = words.data();
      x->color_camera = lensed(size[0], size[1]);
    }
    const Prepared got = prepared(made.value(), gated(f, level, kHigh));
    const Prepared plain = prepared(made_reference.value(), f);
    const Prepared want = prepared(made_reference.value(), want_f);
    CHECK(!got.color.empty() && !plain.color.empty() && !want.depth.empty());
    std::printf(
        "  slot at %ux%u (depth %zu bytes, plane %zu): colour %zu "
        "off, depth %zu off\n",
        size[0], size[1], raw.size() * 2, raw.size(),
        differing(got.color, plain.color), differing(got.depth, want.depth));
    CHECK(got.color == plain.color);
    CHECK(got.depth == want.depth);
  }
  return 0;
}

// Gated and ungated frames in one set, a camera missing between them, each
// recorded (record_set) or prepared (prepare_batch) as it prepares alone.
int test_sets(vkc::Device& device, vkc::Allocator& allocator) {
  const camera::CameraModel cam_a = lensed(160, 120);
  const camera::CameraModel cam_b = pinhole(96, 80);
  const std::vector<std::uint16_t> raw_a = slanted(cam_a);
  const std::vector<std::uint16_t> raw_b = slanted(cam_b);
  const std::vector<std::uint8_t> level_a = scattered_levels(raw_a.size(), 3);
  const std::vector<std::uint32_t> words = packed_words(raw_a.size(), 13);
  sensor::RgbdFrame a = gated(frame_of(raw_a, cam_a), level_a, kHigh);
  a.color_packed = words.data();
  a.color_camera = cam_a;
  const sensor::RgbdFrame b = frame_of(raw_b, cam_b);
  const std::vector<std::optional<sensor::RgbdFrame>> frames{a, std::nullopt,
                                                             b};
  const auto passes = [&](std::vector<sensor::GpuFramePrep>* out) {
    for (int i = 0; i < 3; ++i) {
      auto p = sensor::GpuFramePrep::create(device, allocator);
      if (!p.ok()) return false;
      out->push_back(std::move(p).value());
    }
    return true;
  };
  std::vector<sensor::GpuFramePrep> alone, batched, recorded;
  CHECK(passes(&alone) && passes(&batched) && passes(&recorded));
  const Prepared want_a = prepared(alone[0], a);
  const Prepared want_b = prepared(alone[2], b);
  CHECK(!want_a.depth.empty() && !want_a.color.empty() &&
        !want_b.depth.empty());

  auto set = sensor::GpuFramePrep::prepare_batch(batched, frames);
  CHECK(set.ok() && set->size() == 3 && (*set)[0] && !(*set)[1] && (*set)[2]);
  CHECK(depth_words(*(*set)[0]) == want_a.depth);
  CHECK(color_words(*(*set)[0]) == want_a.color);
  CHECK(depth_words(*(*set)[2]) == want_b.depth);
  {
    vkc::CommandBatch batch(device, allocator);
    auto got = sensor::GpuFramePrep::record_set(batch, recorded, frames);
    CHECK(got.ok() && got->size() == 3 && !(*got)[1]);
    CHECK(batch.submit().ok());
    CHECK(depth_words(*(*got)[0]) == want_a.depth);
    CHECK(color_words(*(*got)[0]) == want_a.color);
    CHECK(depth_words(*(*got)[2]) == want_b.depth);
  }
  // The other way round: the ungated frame first.
  const std::vector<std::optional<sensor::RgbdFrame>> swapped{b, a};
  auto again = sensor::GpuFramePrep::prepare_batch(batched, swapped);
  CHECK(again.ok() && again->size() == 2);
  CHECK(depth_words(*(*again)[0]) == want_b.depth);
  CHECK(depth_words(*(*again)[1]) == want_a.depth);
  CHECK(color_words(*(*again)[1]) == want_a.color);
  std::puts("  sets: recorded and prepared, each frame as alone");
  return 0;
}

// `buffer` holding `raw`, on the device, as a GPU stage leaves its depth; null
// if it cannot be made.
std::shared_ptr<const vkc::Buffer> on_device(
    const std::vector<std::uint16_t>& raw) {
  std::vector<std::uint8_t> bytes(raw.size() * sizeof(std::uint16_t));
  std::memcpy(bytes.data(), raw.data(), bytes.size());
  auto made = vkc::device_storage_buffer(*g_allocator, bytes.size());
  if (!made.ok() ||
      !vkc::test::write_back(*g_device, *g_allocator, made.value(), bytes)
           .ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

// A plane beside device depth, at any floor, and a floor without a plane:
// refused, alone or in a set, with nothing recorded or timed. A plane at a
// floor of 0 is taken, and read nowhere.
int test_refusals(vkc::Device& device, vkc::Allocator& allocator) {
  const auto invalid = vkc::Status::Code::InvalidArgument;
  std::vector<sensor::GpuFramePrep> preps;
  for (int i = 0; i < 2; ++i) {
    auto p = sensor::GpuFramePrep::create(device, allocator, unfiltered());
    CHECK(p.ok());
    preps.push_back(std::move(p).value());
  }
  sensor::GpuFramePrep& prep = preps[0];
  const camera::CameraModel cam = pinhole(64, 48);
  const std::vector<std::uint16_t> raw = slanted(cam);
  const std::vector<std::uint8_t> level(raw.size(), kHigh);
  const sensor::RgbdFrame host = frame_of(raw, cam);
  sensor::RgbdFrame device_depth = host;
  device_depth.depth = nullptr;
  device_depth.depth_device.buffer = on_device(raw);
  CHECK(device_depth.depth_device.buffer != nullptr);
  CHECK(prep.prepare(device_depth).ok());

  struct Refused {
    const char* what;
    sensor::RgbdFrame frame;
    const char* why;
  };
  sensor::RgbdFrame no_plane = host;
  no_plane.min_confidence = kHigh;
  const Refused refused[] = {
      {"plane beside device depth", gated(device_depth, level, kHigh),
       "confidence plane"},
      {"plane beside device depth, floor 0", gated(device_depth, level, 0),
       "confidence plane"},
      {"floor without a plane", no_plane, "min_confidence"},
  };
  for (const Refused& r : refused) {
    // Recorded: refused, and the batch holds nothing to submit.
    {
      vkc::CommandBatch batch(device, allocator);
      const vkc::DeviceCounters was = device.counters();
      const vkc::Status status = prep.record(batch, r.frame).status();
      CHECK(batch.submit().ok());
      const bool named = status.message().find(r.why) != std::string::npos;
      std::printf("  refused, %s: \"%s\"\n", r.what, status.message().c_str());
      CHECK(status.domain() == invalid && named);
      CHECK(device.counters().submits == was.submits);
    }
    // Prepared, alone and in a set beside a good frame: nothing timed or
    // submitted.
    vkc::StageMetrics metrics;
    CHECK(prep.prepare(r.frame, &metrics).status().domain() == invalid);
    CHECK(metrics.rows().empty());
    const vkc::DeviceCounters was = device.counters();
    const std::vector<std::optional<sensor::RgbdFrame>> set{host, r.frame};
    CHECK(sensor::GpuFramePrep::prepare_batch(preps, set, &metrics)
              .status()
              .domain() == invalid);
    CHECK(metrics.rows().empty());
    CHECK(device.counters().submits == was.submits);
  }
  // A plane at a floor of 0 is taken, as a frame without one.
  CHECK(prep.prepare(gated(host, level, 0)).ok());
  // The pass is as it was.
  CHECK(prep.prepare(gated(host, level, kHigh)).ok());
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  g_device = &gpu.device;
  g_allocator = &gpu.allocator;
  if (test_gate(gpu.device, gpu.allocator) != 0) return 1;
  if (test_before_mix(gpu.device, gpu.allocator) != 0) return 1;
  if (test_unchanged(gpu.device, gpu.allocator) != 0) return 1;
  if (test_slot(gpu.device, gpu.allocator) != 0) return 1;
  if (test_sets(gpu.device, gpu.allocator) != 0) return 1;
  if (test_refusals(gpu.device, gpu.allocator) != 0) return 1;
  std::puts("sensor_gpu_frame_prep_confidence: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
