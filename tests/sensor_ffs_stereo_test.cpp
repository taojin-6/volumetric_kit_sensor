// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// FfsStereo on the test models in tests/data/ffs (tools/ffs/make_test_onnx.py):
// FFS's inputs and output, and a disparity worked out by hand -- a weighted
// difference of the two views' channels, (1/8, 2/8, 5/8), or half of it. So
// each check reaches through the whole path: the rectifier's views read by
// CUDA, the input kernel's normalisation and padding, TensorRT, and the depth
// kernel's units and refusals, into a Vulkan buffer read as Vulkan readers
// read it -- taken over from outside Vulkan -- and through GpuFramePrep. Each
// frame states the noise its options' disparity sigma makes over the rig's fx
// and baseline, or none for none, and GpuFramePrep hands it on. The
// plan cache is checked by what it loads: another model's plan put under a
// model's name is the one that runs. Rgba8 coverage is checked on views made
// here, and pairs kept past the next on the rectifier's new views. The spread
// gate runs on a model whose disparity_spread is a sixteenth of the right
// view, so the pixels it drops are known exactly, on a rig of the network's
// size and on a smaller one.
//
// Where CUDA does not reach the device's GPU, or the device exports no
// memory, the test skips; VS_TEST_HEVC_BACKEND, which CI sets on the NVIDIA
// leg, makes it fail instead, as for the decoders.

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/sensor/ffs/ffs_stereo.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace fs = std::filesystem;

namespace {

using Output = sensor::StereoRectifier::Output;

const fs::path kData = VS_FFS_DATA;
const fs::path kModel = kData / "difference_64x32.onnx";
const fs::path kHalfModel = kData / "half_difference_64x32.onnx";
// The model's disparity and a disparity_spread of a sixteenth of the right
// view's weighted value; the spread at 8 x 16, and named otherwise.
const fs::path kSpreadModel = kData / "spread_64x32.onnx";
const fs::path kQuarterSpreadModel = kData / "spread_quarter_64x32.onnx";
const fs::path kOtherOutputModel = kData / "other_output_64x32.onnx";
constexpr double kFocal = 40.0;
constexpr double kBaseline = 0.1;
// The test model's channel weights.
constexpr double kWeights[3] = {0.125, 0.25, 0.625};

// A pair rectified already, w x h: one pinhole camera, the right one
// `baseline` along the left one's +x.
sensor::StereoInfo pair_info(std::uint32_t w, std::uint32_t h,
                             double baseline = kBaseline) {
  sensor::StereoInfo info;
  info.calibration.left = info.calibration.right = camera::CameraModel{
      {w, h}, {kFocal, kFocal, (w - 1) / 2.0, (h - 1) / 2.0}, {}};
  info.calibration.left_to_right[3] = glm::dvec4(-baseline, 0.0, 0.0, 1.0);
  return info;
}

// A frame's pictures, which it holds.
struct Pictures {
  std::vector<std::uint8_t> left, right;
};

sensor::YuvImage host_image(const std::vector<std::uint8_t>& bytes,
                            sensor::YuvLayout layout, std::uint32_t w,
                            std::uint32_t h) {
  sensor::YuvImage im;
  im.layout = layout;
  im.host = bytes.data();
  im.stride[0] = layout == sensor::YuvLayout::Yuyv ? 2 * std::size_t{w} : w;
  im.width = w;
  im.height = h;
  return im;
}

sensor::StereoFrame frame_of(const std::shared_ptr<Pictures>& p,
                             sensor::YuvLayout layout, std::uint32_t w,
                             std::uint32_t h, std::uint64_t sequence) {
  sensor::StereoFrame f;
  f.left = host_image(p->left, layout, w, h);
  f.right = host_image(p->right, layout, w, h);
  f.timestamp_ns = 1000 + sequence;
  f.sequence = sequence;
  f.arrival_ns = 5000 + sequence;
  f.color_to_world[3] = glm::dvec4(1.0, 2.0, 3.0, 1.0);
  f.pixels = p;
  return f;
}

// Luma pairs whose disparity, left less right, covers every case: row 0 is 0
// and row 1 below 0 (no return); the rest run from 4 to 43, so the near and
// far ends fall outside a range of [0.105, 0.45] m at fx * baseline 4, and a
// disparity past the column points off the right view. `round` moves them.
std::shared_ptr<Pictures> luma_pair(std::uint32_t w, std::uint32_t h,
                                    std::uint32_t round) {
  auto p = std::make_shared<Pictures>();
  p->left.resize(std::size_t{w} * h);
  p->right.assign(std::size_t{w} * h, 10);
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      int v = 4 + static_cast<int>((x * 7 + y * 3 + round * 5) % 40);
      if (y == 0) v = 0;
      if (y == 1) v = -5;
      p->left[std::size_t{y} * w + x] = static_cast<std::uint8_t>(10 + v);
    }
  }
  return p;
}

// What FfsStereo writes for disparity `d` at column `x`; empty where float
// rounding may fall either way (on a boundary), which the checks skip.
std::optional<std::uint16_t> expected_depth(double d, std::uint32_t x,
                                            double focal_baseline,
                                            const sensor::FfsStereo::Options& o,
                                            double metres_per_unit) {
  constexpr double kEps = 1e-3;
  if (std::abs(d) < kEps || std::abs(d - x) < kEps) return std::nullopt;
  if (d < 0.0 || d > x) return std::uint16_t{0};
  const double z = focal_baseline / d;
  if (std::abs(z - o.min_depth) < 1e-5 || std::abs(z - o.max_depth) < 1e-5) {
    return std::nullopt;
  }
  if (z < o.min_depth || z > o.max_depth) return std::uint16_t{0};
  const double units = std::round(z / metres_per_unit);
  return static_cast<std::uint16_t>(std::clamp(units, 1.0, 65535.0));
}

// Every sample against its expected value, within a unit; the samples
// compared and those expected to return.
struct Compared {
  int mismatches = 0;
  int compared = 0;
  int returns = 0;
};

Compared compare(const std::vector<std::uint16_t>& got,
                 const std::vector<std::optional<std::uint16_t>>& want) {
  Compared c;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!want[i]) continue;
    ++c.compared;
    if (*want[i] != 0) ++c.returns;
    const int diff = static_cast<int>(got[i]) - static_cast<int>(*want[i]);
    const bool ok = *want[i] == 0 ? got[i] == 0 : std::abs(diff) <= 1;
    if (!ok && ++c.mismatches <= 5) {
      std::fprintf(stderr, "  sample %zu: %u, expected %u\n", i, got[i],
                   static_cast<unsigned>(*want[i]));
    }
  }
  return c;
}

// The luma pair's expected depth, its disparity scaled by `scale`.
std::vector<std::optional<std::uint16_t>> luma_expected(
    const Pictures& p, std::uint32_t w, std::uint32_t h,
    const sensor::FfsStereo::Options& o, double scale) {
  std::vector<std::optional<std::uint16_t>> want(std::size_t{w} * h);
  const double metres_per_unit = std::max(1e-4, o.max_depth / 65535.0);
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = scale * (static_cast<double>(p.left[i]) -
                              static_cast<double>(p.right[i]));
    want[i] = expected_depth(d, static_cast<std::uint32_t>(i % w),
                             kFocal * kBaseline, o, metres_per_unit);
  }
  return want;
}

// A buffer another API wrote, read as a Vulkan reader reads it: taken over
// from outside Vulkan, read, and handed back.
template <typename T>
vkc::Result<std::vector<T>> read_external(vkc::test::GpuContext& gpu,
                                          const vkc::Buffer& buffer,
                                          std::size_t count) {
  std::vector<T> out(count);
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(batch.acquire(buffer, VK_QUEUE_FAMILY_EXTERNAL));
  VKC_TRY(batch.readback(buffer, 0, count * sizeof(T), out.data()));
  VKC_TRY(batch.release(buffer, VK_QUEUE_FAMILY_EXTERNAL));
  VKC_TRY(batch.submit());
  return out;
}

vkc::Result<std::vector<std::uint16_t>> read_depth(
    vkc::test::GpuContext& gpu, const sensor::RgbdFrame& frame) {
  const camera::ImageSize size = frame.depth_camera.size;
  return read_external<std::uint16_t>(gpu, *frame.depth_device.buffer,
                                      std::size_t{size.width} * size.height);
}

std::vector<fs::path> plans_in(const fs::path& dir) {
  std::vector<fs::path> out;
  std::error_code ec;
  for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
    if (e.path().extension() == ".plan") out.push_back(e.path());
  }
  return out;
}

struct Fixture {
  vkc::test::GpuContext& gpu;
  fs::path scratch;
};

std::unique_ptr<sensor::StereoRectifier> rectifier_for(
    Fixture& t, const sensor::StereoInfo& info, Output output) {
  auto made = sensor::StereoRectifier::create(t.gpu.device, t.gpu.allocator,
                                              info, {{}, 0.0f, output});
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  return made ? std::move(made).value() : nullptr;
}

std::unique_ptr<sensor::FfsStereo> ffs_for(
    Fixture& t, const sensor::StereoRectifier& rectifier,
    const sensor::FfsStereo::Options& options) {
  auto made = sensor::FfsStereo::create(t.gpu.device, t.gpu.allocator,
                                        rectifier, options);
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  return made ? std::move(made).value() : nullptr;
}

sensor::FfsStereo::Options options_for(const fs::path& model,
                                       const fs::path& cache) {
  sensor::FfsStereo::Options o;
  o.model = model;
  o.plan_cache = cache;
  o.min_depth = 0.105f;
  o.max_depth = 0.45f;
  return o;
}

// Rectify and match the luma pair of `round`, and check the frame against
// it, its disparity scaled by `scale`; the frame for more checks.
int run_luma(Fixture& t, sensor::StereoRectifier& rectifier,
             sensor::FfsStereo& ffs, const sensor::FfsStereo::Options& o,
             std::uint32_t round, double scale,
             std::optional<sensor::RgbdFrame>* kept = nullptr,
             vkc::StageMetrics* metrics = nullptr) {
  const camera::ImageSize size = rectifier.rig().camera.size;
  const std::shared_ptr<Pictures> p = luma_pair(size.width, size.height, round);
  const sensor::StereoFrame f =
      frame_of(p, sensor::YuvLayout::Luma, size.width, size.height, round);
  auto pair = rectifier.rectify(f);
  CHECK(pair.ok());
  auto frame = ffs.depth(*pair, metrics);
  if (!frame) std::fprintf(stderr, "%s\n", frame.status().message().c_str());
  CHECK(frame.ok());
  auto got = read_depth(t.gpu, *frame);
  CHECK(got.ok());
  const Compared c =
      compare(*got, luma_expected(*p, size.width, size.height, o, scale));
  std::printf("    round %u: %d samples, %d returns, %d off\n", round,
              c.compared, c.returns, c.mismatches);
  CHECK(c.mismatches == 0);
  CHECK(c.returns > static_cast<int>(size.width * size.height) / 4);
  CHECK(c.compared > static_cast<int>(size.width * size.height) * 9 / 10);
  if (kept != nullptr) *kept = std::move(frame).value();
  return 0;
}

// --- Depth: units, refusals, the frame, the buffers
// ---------------------------

int depth_units(Fixture& t) {
  std::puts("depth units, no returns and the frame");
  const sensor::StereoInfo info = pair_info(64, 32);
  auto rectifier = rectifier_for(t, info, Output::Luma8);
  CHECK(rectifier != nullptr);
  const sensor::FfsStereo::Options o = options_for(kModel, t.scratch / "units");
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);

  // Two frames kept at once write two buffers; let go, one is written again.
  std::optional<sensor::RgbdFrame> first, second, third;
  vkc::StageMetrics metrics;
  CHECK(run_luma(t, *rectifier, *ffs, o, 0, 1.0, &first, &metrics) == 0);
  CHECK(run_luma(t, *rectifier, *ffs, o, 1, 1.0, &second) == 0);
  const vkc::Buffer* buffers[2] = {first->depth_device.buffer.get(),
                                   second->depth_device.buffer.get()};
  CHECK(buffers[0] != buffers[1]);
  // The first is unchanged by the second.
  {
    auto again = read_depth(t.gpu, *first);
    CHECK(again.ok());
    const std::shared_ptr<Pictures> p = luma_pair(64, 32, 0);
    CHECK(compare(*again, luma_expected(*p, 64, 32, o, 1.0)).mismatches == 0);
  }
  first.reset();
  second.reset();
  CHECK(run_luma(t, *rectifier, *ffs, o, 2, 1.0, &third) == 0);
  const vkc::Buffer* reused = third->depth_device.buffer.get();
  CHECK(reused == buffers[0] || reused == buffers[1]);
  std::puts("  two frames kept write two buffers; one let go is reused");

  // The frame: depth on the device in the rig's camera, the rest the pair's.
  const sensor::RgbdFrame& f = *third;
  CHECK(f.depth == nullptr && f.has_depth());
  CHECK(f.depth_device.offset == 0);
  CHECK(f.depth_device.queue_family == sensor::kQueueFamilyExternal);
  CHECK(f.depth_device.buffer->size() >= 64 * 32 * 2);
  CHECK(f.metres_per_unit == 1e-4f);
  CHECK(f.min_depth == o.min_depth && f.max_depth == o.max_depth);
  // No disparity sigma stated: the noise is unknown.
  CHECK(f.depth_noise == sensor::DepthNoise{});
  CHECK(f.depth_camera.size.width == 64 && f.depth_camera.size.height == 32);
  CHECK(f.depth_camera.intrinsics.fx == rectifier->rig().camera.intrinsics.fx);
  const sensor::SensorInfo& d = rectifier->depth_info();
  CHECK(f.color_camera.size.width == d.color->size.width);
  CHECK(f.color_camera.intrinsics.fx == d.color->intrinsics.fx);
  CHECK(f.depth_to_color == d.depth_to_color);
  // No third camera: the colour is the left picture, held by the frame.
  CHECK(f.color.layout == sensor::YuvLayout::Luma && f.color.host != nullptr);
  CHECK(f.pixels != nullptr);
  CHECK(f.color.host ==
        static_cast<const Pictures*>(f.pixels.get())->left.data());
  CHECK(f.timestamp_ns == 1002 && f.sequence == 2 && f.arrival_ns == 5002);
  CHECK(f.color_to_world[3] == glm::dvec4(1.0, 2.0, 3.0, 1.0));
  std::puts("  the frame: device depth, the rig's camera, the pair's colour");

  // The timed rows, on the first frame.
  CHECK(metrics.total_gpu_ms() > 0.0);
  CHECK(metrics.total_cpu_ms() > 0.0);
  std::printf("  timed: %.3f ms of device time, %.3f ms on the host\n",
              metrics.total_gpu_ms(), metrics.total_cpu_ms());

  // A depth range past 6.5535 m takes a unit to reach it. A rectifier of its
  // own: the first matcher to read a view maps it.
  sensor::FfsStereo::Options far = o;
  far.max_depth = 10.0f;
  far.min_depth = 0.105f;
  far.disparity_sigma = 0.04f;
  auto own = rectifier_for(t, info, Output::Luma8);
  CHECK(own != nullptr);
  auto wide = ffs_for(t, *own, far);
  CHECK(wide != nullptr);
  std::optional<sensor::RgbdFrame> w;
  CHECK(run_luma(t, *own, *wide, far, 3, 1.0, &w) == 0);
  CHECK(w->metres_per_unit == 10.0f / 65535.0f);
  std::puts("  a 10 m range: a unit of max_depth / 65535");
  // A disparity sigma of 0.04 px over the rig's fx and baseline (40 px and
  // 0.1 m: the pair is rectified already): c = 0.04 / 4 = 0.01 per metre.
  const sensor::StereoRig& rig = own->rig();
  CHECK(w->depth_noise == sensor::DepthNoise::stereo(
                              0.04f, rig.camera.intrinsics.fx, rig.baseline));
  CHECK(w->depth_noise.a == 0.0f && w->depth_noise.b == 0.0f);
  std::printf("  disparity sigma 0.04 px: c %.6g per metre (fx %.3f, B %.4f)\n",
              w->depth_noise.c, rig.camera.intrinsics.fx, rig.baseline);
  CHECK(std::abs(w->depth_noise.c - 0.01f) < 1e-6f);
  return 0;
}

// --- The plan cache -------------------------------------------------------

int plan_cache(Fixture& t) {
  std::puts("the plan cache");
  const sensor::StereoInfo info = pair_info(64, 32);
  auto rectifier = rectifier_for(t, info, Output::Luma8);
  CHECK(rectifier != nullptr);
  const fs::path cache = t.scratch / "cache";
  const sensor::FfsStereo::Options o = options_for(kModel, cache);

  // A miss builds and keeps one plan; a hit leaves it as it was.
  CHECK(ffs_for(t, *rectifier, o) != nullptr);
  const std::vector<fs::path> plans = plans_in(cache);
  CHECK(plans.size() == 1);
  const fs::path plan = plans[0];
  const auto written = fs::last_write_time(plan);
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);
  CHECK(plans_in(cache).size() == 1 && fs::last_write_time(plan) == written);
  CHECK(run_luma(t, *rectifier, *ffs, o, 0, 1.0) == 0);
  std::printf("  built once: %s\n", plan.filename().c_str());

  // The plan under the model's name is the one that runs: the half model's,
  // put there, halves the disparity.
  const fs::path other = t.scratch / "other";
  CHECK(ffs_for(t, *rectifier, options_for(kHalfModel, other)) != nullptr);
  CHECK(plans_in(other).size() == 1);
  fs::copy_file(plans_in(other)[0], plan, fs::copy_options::overwrite_existing);
  // Each matcher that runs on a rectifier of its own: the first matcher to
  // read a view maps it.
  auto second = rectifier_for(t, info, Output::Luma8);
  CHECK(second != nullptr);
  auto swapped = ffs_for(t, *second, o);
  CHECK(swapped != nullptr);
  CHECK(run_luma(t, *second, *swapped, o, 1, 0.5) == 0);
  std::puts("  a cached plan is loaded, not built: another's runs");

  // A plan TensorRT will not load is built again.
  {
    std::ofstream bad(plan, std::ios::binary | std::ios::trunc);
    bad << "not a plan";
  }
  auto third = rectifier_for(t, info, Output::Luma8);
  CHECK(third != nullptr);
  auto rebuilt = ffs_for(t, *third, o);
  CHECK(rebuilt != nullptr);
  CHECK(fs::file_size(plan) > 10);
  CHECK(run_luma(t, *third, *rebuilt, o, 2, 1.0) == 0);
  std::puts("  a plan cut short is built again");

  // Another model is another plan.
  CHECK(ffs_for(t, *rectifier, options_for(kHalfModel, cache)) != nullptr);
  CHECK(plans_in(cache).size() == 2);
  std::puts("  another model: a plan of its own");

  // A cache that cannot be written runs the plan it built, unkept (a root
  // user writes it anyway).
  const fs::path locked = t.scratch / "locked";
  fs::create_directories(locked);
  fs::permissions(locked, fs::perms::owner_read | fs::perms::owner_exec,
                  fs::perm_options::replace);
  auto fourth = rectifier_for(t, info, Output::Luma8);
  CHECK(fourth != nullptr);
  auto unkept = ffs_for(t, *fourth, options_for(kModel, locked));
  const bool kept_anyway = !plans_in(locked).empty();
  fs::permissions(locked, fs::perms::owner_all, fs::perm_options::replace);
  CHECK(unkept != nullptr);
  CHECK(kept_anyway == (::geteuid() == 0));
  CHECK(run_luma(t, *fourth, *unkept, options_for(kModel, locked), 3, 1.0) ==
        0);
  std::puts("  a cache it cannot write: the plan runs unkept");
  return 0;
}

// --- The network's size: padding, and refusals ----------------------------

int padding(Fixture& t) {
  std::puts("a rig smaller than the network: padded, then cut");
  const sensor::StereoInfo info = pair_info(60, 30);
  auto rectifier = rectifier_for(t, info, Output::Luma8);
  CHECK(rectifier != nullptr);
  const sensor::FfsStereo::Options o = options_for(kModel, t.scratch / "pad");
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);
  CHECK(run_luma(t, *rectifier, *ffs, o, 0, 1.0) == 0);
  CHECK(run_luma(t, *rectifier, *ffs, o, 1, 1.0) == 0);
  std::puts("  60 x 30 on a 64 x 32 network: every sample right");

  for (const auto& [w, h] : {std::pair{64u, 64u}, std::pair{96u, 32u}}) {
    auto other = rectifier_for(t, pair_info(w, h), Output::Luma8);
    CHECK(other != nullptr);
    auto refused =
        sensor::FfsStereo::create(t.gpu.device, t.gpu.allocator, *other, o);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::InvalidArgument);
    const std::string want =
        "1 x 3 x " + std::to_string(h) + " x " + std::to_string(w);
    CHECK(refused.status().message().find(want) != std::string::npos);
  }
  std::puts("  a rig whose network is another size: refused, naming it");
  return 0;
}

// --- Rgba8 views: each channel weighed ------------------------------------

int rgba_views(Fixture& t) {
  std::puts("Rgba8 views from YUYV pictures");
  constexpr std::uint32_t kW = 64, kH = 32;
  const sensor::StereoInfo info = pair_info(kW, kH);
  auto rectifier = rectifier_for(t, info, Output::Rgba8);
  CHECK(rectifier != nullptr);
  const sensor::FfsStereo::Options o = options_for(kModel, t.scratch / "rgba");
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);

  // Random chroma; the left luma 15 to 45 above the right.
  auto p = std::make_shared<Pictures>();
  p->left.resize(2 * std::size_t{kW} * kH);
  p->right.resize(p->left.size());
  std::uint32_t x = 12345;
  const auto next = [&x] {
    x = x * 1664525u + 1013904223u;
    return x >> 24;
  };
  for (std::size_t i = 0; i < p->left.size(); ++i) {
    if (i % 2 == 0) {  // Y
      const std::uint32_t y = 60 + next() % 100;
      p->right[i] = static_cast<std::uint8_t>(y);
      p->left[i] = static_cast<std::uint8_t>(y + 15 + next() % 30);
    } else {  // Cb or Cr
      p->left[i] = static_cast<std::uint8_t>(96 + next() % 64);
      p->right[i] = static_cast<std::uint8_t>(96 + next() % 64);
    }
  }
  const sensor::StereoFrame f = frame_of(p, sensor::YuvLayout::Yuyv, kW, kH, 0);
  auto pair = rectifier->rectify(f);
  CHECK(pair.ok());
  auto frame = ffs->depth(*pair);
  if (!frame) std::fprintf(stderr, "%s\n", frame.status().message().c_str());
  CHECK(frame.ok());

  // The views as the rectifier wrote them, which the expectation reads.
  auto left = read_external<std::uint32_t>(t.gpu, pair->left->buffer, kW * kH);
  auto right =
      read_external<std::uint32_t>(t.gpu, pair->right->buffer, kW * kH);
  auto got = read_depth(t.gpu, *frame);
  CHECK(left.ok() && right.ok() && got.ok());
  std::vector<std::optional<std::uint16_t>> want(std::size_t{kW} * kH);
  for (std::size_t i = 0; i < want.size(); ++i) {
    double d = 0.0;
    for (int c = 0; c < 3; ++c) {
      d += kWeights[c] * (double((*left)[i] >> (8 * c) & 0xFFu) -
                          double((*right)[i] >> (8 * c) & 0xFFu));
    }
    want[i] = expected_depth(d, static_cast<std::uint32_t>(i % kW),
                             kFocal * kBaseline, o, 1e-4);
  }
  const Compared c = compare(*got, want);
  std::printf("  %d samples, %d returns, %d off\n", c.compared, c.returns,
              c.mismatches);
  CHECK(c.mismatches == 0 && c.returns > 200);
  // Grey views would hide a channel swap: these differ.
  int coloured = 0;
  for (const std::uint32_t w : *left) {
    coloured += (w & 0xFFu) != ((w >> 16) & 0xFFu);
  }
  CHECK(coloured > static_cast<int>(kW * kH) / 2);
  return 0;
}

// --- Rgba8 coverage: what a view did not see has no return ----------------

// Words as the rectifier hands its views out: in an exported buffer of their
// own, released to VK_QUEUE_FAMILY_EXTERNAL.
vkc::Result<std::shared_ptr<vkc::ExportedBuffer>> exported_view(
    vkc::test::GpuContext& gpu, const std::vector<std::uint32_t>& words) {
  const std::size_t bytes = words.size() * sizeof(std::uint32_t);
  VKC_ASSIGN(vkc::ExportedBuffer made,
             vkc::create_exported_buffer(gpu.device, gpu.allocator, bytes));
  auto view = std::make_shared<vkc::ExportedBuffer>(std::move(made));
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(batch.upload(view->buffer, 0, words.data(), bytes));
  VKC_TRY(batch.release(view->buffer, VK_QUEUE_FAMILY_EXTERNAL));
  VKC_TRY(batch.submit());
  return view;
}

// Luma bytes as a Luma8 view: four to a word, the first in the low byte.
vkc::Result<std::shared_ptr<vkc::ExportedBuffer>> luma_view(
    vkc::test::GpuContext& gpu, const std::vector<std::uint8_t>& bytes) {
  std::vector<std::uint32_t> words((bytes.size() + 3) / 4);
  std::memcpy(words.data(), bytes.data(), bytes.size());
  return exported_view(gpu, words);
}

// A pair of Luma8 views made here from `p`, on `rig`, as the rectifier hands
// them out.
vkc::Result<sensor::RectifiedPair> made_pair(vkc::test::GpuContext& gpu,
                                             const sensor::StereoRig& rig,
                                             const std::shared_ptr<Pictures>& p,
                                             std::uint32_t sequence) {
  sensor::RectifiedPair pair;
  VKC_ASSIGN(pair.left, luma_view(gpu, p->left));
  VKC_ASSIGN(pair.right, luma_view(gpu, p->right));
  pair.queue_family = sensor::kQueueFamilyExternal;
  pair.rig = rig;
  const camera::ImageSize size = rig.camera.size;
  pair.source =
      frame_of(p, sensor::YuvLayout::Luma, size.width, size.height, sequence);
  return pair;
}

int coverage(Fixture& t) {
  std::puts("Rgba8 coverage: no return where either view saw nothing");
  constexpr std::uint32_t kW = 64, kH = 32;
  const sensor::StereoInfo info = pair_info(kW, kH);
  auto rectifier = rectifier_for(t, info, Output::Rgba8);
  CHECK(rectifier != nullptr);

  // The left view `k` above the right one in every channel, so the
  // disparity is k; coverage 0 on a lattice of left pixels and on a band of
  // right columns, whose colour stays.
  std::vector<std::uint32_t> left(std::size_t{kW} * kH), right(left.size());
  std::vector<int> disparity(left.size());
  const auto seen_left = [](std::uint32_t x, std::uint32_t y) {
    return (x + 2 * y) % 5 != 0;
  };
  const auto seen_right = [](std::uint32_t x) { return x < 20 || x >= 26; };
  const auto grey = [](std::uint32_t v, bool seen) {
    return v | v << 8 | v << 16 | (seen ? 0xFFu << 24 : 0u);
  };
  for (std::uint32_t y = 0; y < kH; ++y) {
    for (std::uint32_t x = 0; x < kW; ++x) {
      const std::size_t i = std::size_t{y} * kW + x;
      disparity[i] = 10 + static_cast<int>((x * 7 + y * 3) % 28);
      left[i] = grey(50 + disparity[i], seen_left(x, y));
      right[i] = grey(50, seen_right(x));
    }
  }
  auto left_view = exported_view(t.gpu, left);
  auto right_view = exported_view(t.gpu, right);
  CHECK(left_view.ok() && right_view.ok());
  const sensor::FfsStereo::Options o =
      options_for(kModel, t.scratch / "coverage");
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);

  sensor::RectifiedPair pair;
  pair.left = *left_view;
  pair.right = *right_view;
  pair.queue_family = sensor::kQueueFamilyExternal;
  pair.rig = rectifier->rig();
  pair.source =
      frame_of(luma_pair(kW, kH, 0), sensor::YuvLayout::Luma, kW, kH, 0);
  auto frame = ffs->depth(pair);
  if (!frame) std::fprintf(stderr, "%s\n", frame.status().message().c_str());
  CHECK(frame.ok());
  auto got = read_depth(t.gpu, *frame);
  CHECK(got.ok());

  std::vector<std::optional<std::uint16_t>> want(left.size());
  int unseen_left = 0, unseen_right = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const auto x = static_cast<std::uint32_t>(i % kW);
    const auto y = static_cast<std::uint32_t>(i / kW);
    want[i] = expected_depth(disparity[i], x, kFocal * kBaseline, o, 1e-4);
    if (!want[i] || *want[i] == 0) continue;
    if (!seen_left(x, y)) {
      want[i] = 0;
      ++unseen_left;
    } else if (!seen_right(x - static_cast<std::uint32_t>(disparity[i]))) {
      want[i] = 0;
      ++unseen_right;
    }
  }
  const Compared c = compare(*got, want);
  std::printf(
      "  %d samples, %d returns, %d off; no return unseen on the left %d, "
      "matched where the right saw nothing %d\n",
      c.compared, c.returns, c.mismatches, unseen_left, unseen_right);
  CHECK(c.mismatches == 0 && c.returns > 400);
  CHECK(unseen_left > 50 && unseen_right > 50);
  return 0;
}

// --- Pairs kept past the next: the rectifier's new views ------------------

int kept_pairs(Fixture& t) {
  std::puts("pairs kept past the next: each new view mapped, eight at most");
  const sensor::StereoInfo info = pair_info(64, 32);
  auto rectifier = rectifier_for(t, info, Output::Luma8);
  CHECK(rectifier != nullptr);
  const sensor::FfsStereo::Options o = options_for(kModel, t.scratch / "kept");
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);

  // Rectify and match round `r`, checking every sample; the pair, kept, and
  // whether its views were new.
  const auto match = [&](sensor::StereoRectifier& rect,
                         sensor::FfsStereo& matcher, std::uint32_t r,
                         std::optional<sensor::RectifiedPair>* out,
                         bool* fresh) -> int {
    const std::shared_ptr<Pictures> p = luma_pair(64, 32, r);
    auto pair = rect.rectify(frame_of(p, sensor::YuvLayout::Luma, 64, 32, r));
    CHECK(pair.ok());
    *fresh = pair->left->fd.valid();
    auto frame = matcher.depth(*pair);
    if (!frame) std::fprintf(stderr, "%s\n", frame.status().message().c_str());
    CHECK(frame.ok());
    auto got = read_depth(t.gpu, *frame);
    CHECK(got.ok());
    CHECK(compare(*got, luma_expected(*p, 64, 32, o, 1.0)).mismatches == 0);
    *out = std::move(pair).value();
    return 0;
  };

  // Five pairs kept at once: ten views, each new and matched right; the
  // first pair's, the least recently read, let go for the last two.
  std::vector<std::optional<sensor::RectifiedPair>> kept(5);
  std::set<const void*> views;
  for (std::uint32_t r = 0; r < 5; ++r) {
    bool fresh = false;
    CHECK(match(*rectifier, *ffs, r, &kept[r], &fresh) == 0);
    CHECK(fresh);
    views.insert(kept[r]->left.get());
    views.insert(kept[r]->right.get());
  }
  CHECK(views.size() == 10);
  const auto refused = ffs->depth(*kept[0]);
  CHECK(refused.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(refused.status().message().find("imported by something else") !=
        std::string::npos);
  CHECK(ffs->depth(*kept[1]).ok());
  std::puts("  five kept: ten views mapped; the first pair's let go");

  // Each kept one frame: every frame's views are new, often where freed ones
  // were, whose mappings go; then, used in turn, the newest are reused.
  kept.clear();
  std::optional<sensor::RectifiedPair> held;
  int fresh_frames = 0, reused_addresses = 0;
  for (std::uint32_t r = 5; r < 15; ++r) {
    std::optional<sensor::RectifiedPair> next;
    bool fresh = false;
    CHECK(match(*rectifier, *ffs, r, &next, &fresh) == 0);
    if (fresh) {
      ++fresh_frames;
      reused_addresses += views.count(next->left.get()) != 0;
    }
    views.insert(next->left.get());
    views.insert(next->right.get());
    held = std::move(next);
  }
  held.reset();
  for (std::uint32_t r = 15; r < 18; ++r) {
    std::optional<sensor::RectifiedPair> next;
    bool fresh = true;
    CHECK(match(*rectifier, *ffs, r, &next, &fresh) == 0);
    CHECK(!fresh);
  }
  std::printf(
      "  kept one frame each: %d frames on new views, %d at a freed view's "
      "address; then reused in turn\n",
      fresh_frames, reused_addresses);
  CHECK(fresh_frames == 9);

  // A new buffer at a mapped view's address -- the descriptor tells, not the
  // address -- takes its old mapping's place, and no other mapping goes for
  // it: four pairs of views made here, mapped (eight), then one view's buffer
  // replaced where it is; the pair read least recently is still mapped. A
  // matcher of its own, for a mapping count of its own.
  auto own = rectifier_for(t, info, Output::Luma8);
  CHECK(own != nullptr);
  auto matcher = ffs_for(t, *own, o);
  CHECK(matcher != nullptr);
  const auto check = [&](const sensor::RectifiedPair& pair,
                         const Pictures& p) -> int {
    auto frame = matcher->depth(pair);
    if (!frame) std::fprintf(stderr, "%s\n", frame.status().message().c_str());
    CHECK(frame.ok());
    auto got = read_depth(t.gpu, *frame);
    CHECK(got.ok());
    CHECK(compare(*got, luma_expected(p, 64, 32, o, 1.0)).mismatches == 0);
    return 0;
  };
  std::vector<sensor::RectifiedPair> made;
  std::vector<std::shared_ptr<Pictures>> pictures;
  for (std::uint32_t r = 30; r < 34; ++r) {
    pictures.push_back(luma_pair(64, 32, r));
    auto pair = made_pair(t.gpu, own->rig(), pictures.back(), r);
    CHECK(pair.ok());
    made.push_back(std::move(pair).value());
    CHECK(check(made.back(), *pictures.back()) == 0);
  }
  auto replacement = luma_view(t.gpu, pictures[3]->left);
  CHECK(replacement.ok());
  const vkc::ExportedBuffer* at = made[3].left.get();
  *made[3].left = std::move(**replacement);
  CHECK(made[3].left.get() == at && made[3].left->fd.valid());
  CHECK(check(made[3], *pictures[3]) == 0);
  CHECK(check(made[0], *pictures[0]) == 0);
  std::puts(
      "  a new buffer at a mapped view's address: its mapping replaced, the "
      "rest kept");
  return 0;
}

// --- Into GpuFramePrep ------------------------------------------------------

int into_prep(Fixture& t) {
  std::puts("the depth prepared in place by GpuFramePrep");
  const sensor::StereoInfo info = pair_info(64, 32);
  auto rectifier = rectifier_for(t, info, Output::Luma8);
  CHECK(rectifier != nullptr);
  sensor::FfsStereo::Options o = options_for(kModel, t.scratch / "prep");
  o.disparity_sigma = 0.1f;
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);
  auto prep = sensor::GpuFramePrep::create(t.gpu.device, t.gpu.allocator);
  CHECK(prep.ok());
  // Three rounds: the depth buffers reused, taken over from CUDA each time.
  for (std::uint32_t round = 0; round < 3; ++round) {
    const std::shared_ptr<Pictures> p = luma_pair(64, 32, round);
    auto pair =
        rectifier->rectify(frame_of(p, sensor::YuvLayout::Luma, 64, 32, round));
    CHECK(pair.ok());
    auto frame = ffs->depth(*pair);
    CHECK(frame.ok());
    auto prepared = prep->prepare(*frame);
    if (!prepared) {
      std::fprintf(stderr, "%s\n", prepared.status().message().c_str());
    }
    CHECK(prepared.ok());
    CHECK(frame->depth_noise.known());
    CHECK(prepared->depth_noise == frame->depth_noise);
    auto metres = vkc::test::read_back<float>(t.gpu.device, t.gpu.allocator,
                                              *prepared->depth, 64 * 32);
    CHECK(metres.ok());
    const auto want = luma_expected(*p, 64, 32, o, 1.0);
    int compared = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (!want[i]) continue;
      ++compared;
      const double m = *want[i] * 1e-4;
      CHECK(std::abs((*metres)[i] - m) <= 1.5e-4);
    }
    CHECK(compared > 64 * 32 * 9 / 10);
  }
  std::puts("  three frames prepared: depth in metres at every sample");
  return 0;
}

// --- The spread gate: what the network had not settled has no return -------

// A luma pair for the spread model: the right view 16 to 79, so its spread, a
// sixteenth of it, runs from 1 to 4.9375 px in steps of 1/16; the left one 4
// to 43 above it, but for row 0 (0) and row 1 (5 below), so luma_pair's no
// returns come too.
std::shared_ptr<Pictures> spread_pair(std::uint32_t w, std::uint32_t h) {
  auto p = std::make_shared<Pictures>();
  p->left.resize(std::size_t{w} * h);
  p->right.resize(p->left.size());
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      const int right = 16 + static_cast<int>((x * 5 + y * 11) % 64);
      int v = 4 + static_cast<int>((x * 7 + y * 3) % 40);
      if (y == 0) v = 0;
      if (y == 1) v = -5;
      const std::size_t i = std::size_t{y} * w + x;
      p->right[i] = static_cast<std::uint8_t>(right);
      p->left[i] = static_cast<std::uint8_t>(right + v);
    }
  }
  return p;
}

// The returns a gate dropped, and those it kept.
struct Gated {
  int dropped = 0;
  int kept = 0;
};

// Match the spread pair on a w x h rig on the spread model with
// `max_spread`, and check every sample: no return where the spread is above
// a gate, else the disparity's depth. On a rig smaller than the network, the
// spread is read where the network has the view's pixel, as the disparity
// is: anywhere else is another pixel's spread.
int run_gated(Fixture& t, std::uint32_t w, std::uint32_t h, float max_spread,
              Gated* gated) {
  // A rectifier of its own: the first matcher to read a view maps it.
  auto rectifier = rectifier_for(t, pair_info(w, h), Output::Luma8);
  CHECK(rectifier != nullptr);
  sensor::FfsStereo::Options o =
      options_for(kSpreadModel, t.scratch / "spread");
  o.max_disparity_spread = max_spread;
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);
  const std::shared_ptr<Pictures> p = spread_pair(w, h);
  auto pair = rectifier->rectify(frame_of(p, sensor::YuvLayout::Luma, w, h, 0));
  CHECK(pair.ok());
  auto frame = ffs->depth(*pair);
  if (!frame) std::fprintf(stderr, "%s\n", frame.status().message().c_str());
  CHECK(frame.ok());
  auto got = read_depth(t.gpu, *frame);
  CHECK(got.ok());

  std::vector<std::optional<std::uint16_t>> want(std::size_t{w} * h);
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d =
        static_cast<double>(p->left[i]) - static_cast<double>(p->right[i]);
    want[i] = expected_depth(d, static_cast<std::uint32_t>(i % w),
                             kFocal * kBaseline, o, 1e-4);
    const bool returns = want[i] && *want[i] != 0;
    if (max_spread > 0.0f && p->right[i] / 16.0 > max_spread) {
      gated->dropped += returns;
      want[i] = 0;
    } else {
      gated->kept += returns;
    }
  }
  const Compared c = compare(*got, want);
  std::printf(
      "    %u x %u, max_disparity_spread %g: %d samples, %d returns, %d off; "
      "%d returns dropped\n",
      w, h, static_cast<double>(max_spread), c.compared, c.returns,
      c.mismatches, gated->dropped);
  CHECK(c.mismatches == 0);
  CHECK(c.compared > static_cast<int>(w * h) * 9 / 10);
  return 0;
}

int spread_gate(Fixture& t) {
  std::puts("the spread gate: exactly the pixels above it, no return");
  // Off: a model with the spread matches as one without, its spread bound
  // and not read.
  Gated off, between, above, padded;
  CHECK(run_gated(t, 64, 32, 0.0f, &off) == 0);
  CHECK(off.dropped == 0 && off.kept > 64 * 32 / 4);
  // Between two spreads, 2.5 and 2.5625 px: those above it go, the rest
  // stay.
  CHECK(run_gated(t, 64, 32, 2.53125f, &between) == 0);
  CHECK(between.dropped > 200 && between.kept > 200);
  CHECK(between.dropped + between.kept == off.kept);
  // Above every spread (4.9375 px at most): none go.
  CHECK(run_gated(t, 64, 32, 5.0f, &above) == 0);
  CHECK(above.dropped == 0 && above.kept == off.kept);
  // A 60 x 30 rig on the 64 x 32 network: the network's rows are wider than
  // the view's, so a spread read at the view's index gates other pixels.
  CHECK(run_gated(t, 60, 30, 2.53125f, &padded) == 0);
  CHECK(padded.dropped > 200 && padded.kept > 200);
  std::puts(
      "  off, between two spreads and above all, and between on a padded "
      "rig: each sample right");

  auto rectifier = rectifier_for(t, pair_info(64, 32), Output::Luma8);
  CHECK(rectifier != nullptr);
  const auto refused = [&](const sensor::FfsStereo::Options& o) {
    return sensor::FfsStereo::create(t.gpu.device, t.gpu.allocator, *rectifier,
                                     o)
        .status();
  };
  const auto refused_naming = [&](const sensor::FfsStereo::Options& o,
                                  const std::string& text) {
    const vkc::Status s = refused(o);
    if (s.message().find(text) == std::string::npos) {
      std::fprintf(stderr, "  refusal without \"%s\": %s\n", text.c_str(),
                   s.message().c_str());
    }
    return s.domain() == vkc::Status::Code::InvalidArgument &&
           s.message().find(text) != std::string::npos;
  };
  // A gate on a model without the spread: refused before the build, and on
  // that model's cached plan, whose own tensors are what runs.
  const fs::path cache = t.scratch / "no_spread";
  sensor::FfsStereo::Options o = options_for(kModel, cache);
  o.max_disparity_spread = 0.2f;
  CHECK(refused_naming(o, "no disparity_spread output"));
  CHECK(plans_in(cache).empty());
  o.max_disparity_spread = 0.0f;
  CHECK(ffs_for(t, *rectifier, o) != nullptr);
  CHECK(plans_in(cache).size() == 1);
  o.max_disparity_spread = 0.2f;
  CHECK(refused_naming(o, "no disparity_spread output"));
  // A spread of another shape, or a fourth tensor not the spread: refused,
  // with or without a gate.
  for (const float gate : {0.0f, 0.2f}) {
    o = options_for(kQuarterSpreadModel, cache);
    o.max_disparity_spread = gate;
    CHECK(refused_naming(o, "disparity_spread is 1 x 1 x 8 x 16"));
    o.model = kOtherOutputModel;
    CHECK(refused_naming(o, "the model has 4 inputs and outputs"));
  }
  std::puts(
      "  a gate without the spread, before the build and from a cached plan; "
      "a spread of another shape; a fourth tensor of another name");
  return 0;
}

// --- Refusals ---------------------------------------------------------------

int refusals(Fixture& t) {
  std::puts("refusals");
  const sensor::StereoInfo info = pair_info(64, 32);
  auto rectifier = rectifier_for(t, info, Output::Luma8);
  CHECK(rectifier != nullptr);
  const fs::path cache = t.scratch / "refusals";
  const auto code = [&](const sensor::FfsStereo::Options& o) {
    return sensor::FfsStereo::create(t.gpu.device, t.gpu.allocator, *rectifier,
                                     o)
        .status()
        .domain();
  };
  sensor::FfsStereo::Options o = options_for(kModel, cache);
  o.min_depth = 0.0f;
  CHECK(code(o) == vkc::Status::Code::InvalidArgument);
  o.min_depth = 0.5f;
  o.max_depth = 0.5f;
  CHECK(code(o) == vkc::Status::Code::InvalidArgument);
  for (const float bad : {-0.01f, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    o = options_for(kModel, cache);
    o.disparity_sigma = bad;
    CHECK(code(o) == vkc::Status::Code::InvalidArgument);
    o = options_for(kSpreadModel, cache);
    o.max_disparity_spread = bad;
    CHECK(code(o) == vkc::Status::Code::InvalidArgument);
  }
  o = options_for(kData / "no_such_model.onnx", cache);
  CHECK(code(o) == vkc::Status::Code::NotFound);
  {
    std::ofstream junk(t.scratch / "junk.onnx", std::ios::binary);
    junk << "not an onnx model";
  }
  o.model = t.scratch / "junk.onnx";
  CHECK(code(o) == vkc::Status::Code::InvalidArgument);
  o = options_for(kModel, fs::path());
  CHECK(code(o) == vkc::Status::Code::InvalidArgument);
  // Under a file: refused before anything is built.
  o.plan_cache = t.scratch / "junk.onnx" / "plans";
  CHECK(code(o) == vkc::Status::Code::IoError);
  std::puts(
      "  a bad depth range, disparity sigma or spread gate, no model, a model "
      "TensorRT cannot parse, no plan cache or one that cannot be made");

  o = options_for(kModel, cache);
  auto ffs = ffs_for(t, *rectifier, o);
  CHECK(ffs != nullptr);
  const std::shared_ptr<Pictures> p = luma_pair(64, 32, 0);
  const sensor::StereoFrame f = frame_of(p, sensor::YuvLayout::Luma, 64, 32, 0);

  // A pair from another rig.
  auto wider = rectifier_for(t, pair_info(64, 32, 0.2), Output::Luma8);
  CHECK(wider != nullptr);
  {
    auto pair = wider->rectify(f);
    CHECK(pair.ok());
    CHECK(ffs->depth(*pair).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }
  // Views not exported.
  {
    auto pair = rectifier->rectify(f);
    CHECK(pair.ok());
    sensor::RectifiedPair kept = *pair;
    kept.queue_family = sensor::kQueueFamilyIgnored;
    CHECK(ffs->depth(kept).status().domain() ==
          vkc::Status::Code::InvalidArgument);
    // The same pair as handed out is matched.
    CHECK(ffs->depth(*pair).ok());
  }
  // Views another importer took: their descriptors closed, never mapped here.
  {
    auto other = rectifier_for(t, info, Output::Luma8);
    CHECK(other != nullptr);
    auto context = sensor::CudaContext::open(t.gpu.device);
    CHECK(context.ok());
    std::unique_ptr<sensor::CudaBufferMapping> taken[2];
    {
      auto pair = other->rectify(f);
      CHECK(pair.ok());
      auto l = sensor::CudaBufferMapping::map(**context, *pair->left);
      auto r = sensor::CudaBufferMapping::map(**context, *pair->right);
      CHECK(l.ok() && r.ok());
      taken[0] = std::move(l).value();
      taken[1] = std::move(r).value();
      const auto refused = ffs->depth(*pair);
      CHECK(refused.status().domain() == vkc::Status::Code::InvalidArgument);
      CHECK(refused.status().message().find("imported by something else") !=
            std::string::npos);
    }
    // Still matching its own pairs.
    CHECK(run_luma(t, *rectifier, *ffs, o, 1, 1.0) == 0);
    for (auto& m : taken) m.reset();
  }
  std::puts(
      "  a pair of another rig, views not exported or imported elsewhere");
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const vkc::Result<int> ordinal = sensor::cuda_ordinal_of(gpu.device);
  if (!ordinal) {
    if (ordinal.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", ordinal.status().message().c_str());
      return 1;
    }
    return vs_test::no_decoder(ordinal.status().message());
  }
  if (!gpu.device.exports_memory()) {
    return vs_test::no_decoder("the device does not export memory");
  }
  const vkc::test::Scratch scratch("sensor_ffs_stereo_");
  Fixture t{gpu, scratch.path};
  CHECK(depth_units(t) == 0);
  CHECK(plan_cache(t) == 0);
  CHECK(padding(t) == 0);
  CHECK(rgba_views(t) == 0);
  CHECK(coverage(t) == 0);
  CHECK(kept_pairs(t) == 0);
  CHECK(into_prep(t) == 0);
  CHECK(spread_gate(t) == 0);
  CHECK(refusals(t) == 0);
  std::puts("sensor_ffs_stereo: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
