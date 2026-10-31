// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "color_fit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <numeric>
#include <utility>

#include "blur_comp.spv.hpp"
#include "normal_equations_comp.spv.hpp"
#include "pushed_comp.spv.hpp"
#include "sample_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace volumetric_kit::sensor::color_fit {
namespace {

// shaders/color_fit.glsl's Camera, under scalar layout.
struct GpuCamera {
  float depth_k[4];
  std::uint32_t depth_width, depth_height;
  float depth_to_world[16];
  float color_k[4];
  std::uint32_t color_width, color_height;
  float color_to_world[16];
  std::uint32_t depth_at, color_at;
};
static_assert(sizeof(GpuCamera) == 184, "GpuCamera layout drift");

// shaders/color_fit.glsl's ResponseParams.
struct GpuResponse {
  float gain[3];
  float a, b, q_max;
};
static_assert(sizeof(GpuResponse) == 24, "GpuResponse layout drift");

struct BlurPush {
  std::uint32_t camera, vertical;
  float sigma;
  std::uint32_t out_at;
};

struct SamplePush {
  std::uint32_t a, b, keep, set, budget, capacity, sigma2_at, sigma8_at, bins;
  float bin_width, cube_m;
};
static_assert(sizeof(SamplePush) == 44, "SamplePush layout drift");

struct NormalPush {
  std::uint32_t reduce, train_held;
  float huber;
};

// shaders/normal_equations.comp's row and where its sums are.
constexpr std::uint32_t kRow = 72;
constexpr std::uint32_t kLoss = 60;
constexpr std::uint32_t kSquares = 61;
constexpr std::uint32_t kCount = 63;
constexpr std::uint32_t kGauge = 65;
constexpr std::uint32_t kNormalLocal = 128;
// Samples a chunk of the normal equations' first pass reads.
constexpr std::uint64_t kChunk = 4096;

constexpr std::uint32_t kLocal = 256;
// The radius histogram: bins of 1/2048 over [0, 2).
constexpr std::uint32_t kBins = 4096;
constexpr float kBinWidth = 2.0f / kBins;
constexpr float kCubeM = 0.25f;  // the hold-out's world cubes
constexpr float kHuberLevels = 3.0f;
constexpr double kRmaxQuantile = 0.99;
constexpr int kMaxIterations = 100;

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("ColorFit: " + what);
}

std::uint32_t pack_size(std::uint32_t w, std::uint32_t h) {
  return w | h << 16;
}

void copy_matrix(const Mat4f& m, float* out) {
  std::memcpy(out, &m[0][0], 16 * sizeof(float));
}

// The free parameters: camera 0's a and b, then each other camera's three
// log-gains, a and b (System's order). -1 is a pinned log-gain.
int gain_index(std::uint32_t k, std::uint32_t c) {
  return k == 0 ? -1 : static_cast<int>(2 + (k - 1) * 5 + c);
}
int a_index(std::uint32_t k) {
  return k == 0 ? 0 : static_cast<int>(2 + (k - 1) * 5 + 3);
}
int b_index(std::uint32_t k) { return a_index(k) + 1; }

GpuResponse gpu_response(const camera::ColorResponse& r) {
  GpuResponse g{};
  for (int c = 0; c < 3; ++c) g.gain[c] = static_cast<float>(r.gain[c]);
  g.a = static_cast<float>(r.a);
  g.b = static_cast<float>(r.b);
  g.q_max = static_cast<float>(r.r_max * r.r_max);
  return g;
}

double rms(double squares, double samples) {
  return samples > 0.0 ? std::sqrt(squares / (3.0 * samples)) : 0.0;
}

double stops(double v) { return std::log2(v); }

}  // namespace

// One evaluation's per-pair rows, in double, pairs in index order.
struct ColorFit::Rows {
  std::vector<double> v;
  const double* pair(std::size_t p) const { return v.data() + p * kRow; }
};

core::Result<std::vector<double>> step(const System& system, double lambda) {
  const std::size_t n = system.n;
  if (system.h.size() != n * n || system.g.size() != n) {
    return bad("step: the system's sizes disagree");
  }
  std::vector<double> l(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = system.h[i * n + j];
      if (i == j) s *= 1.0 + lambda;
      for (std::size_t k = 0; k < j; ++k) s -= l[i * n + k] * l[j * n + k];
      if (i == j) {
        if (!(s > 0.0)) return bad("step: the damped system is not definite");
        l[i * n + i] = std::sqrt(s);
      } else {
        l[i * n + j] = s / l[j * n + j];
      }
    }
  }
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) {
    double s = -system.g[i];
    for (std::size_t k = 0; k < i; ++k) s -= l[i * n + k] * y[k];
    y[i] = s / l[i * n + i];
  }
  std::vector<double> x(n);
  for (std::size_t ii = n; ii-- > 0;) {
    double s = y[ii];
    for (std::size_t k = ii + 1; k < n; ++k) s -= l[k * n + ii] * x[k];
    x[ii] = s / l[ii * n + ii];
  }
  return x;
}

ColorFit::~ColorFit() = default;

core::Result<std::unique_ptr<ColorFit>> ColorFit::create(
    core::Device& device, core::Allocator& allocator,
    std::vector<SensorInfo> sensors, const Options& options) try {
  VKC_TRY(check_device_requirements(device, "ColorFit::create"));
  if (sensors.size() < 2) return bad("a fit needs two sensors or more");
  if (sensors.size() > 255) return bad("more sensors than a sample can name");
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    const SensorInfo& s = sensors[i];
    if (s.id.empty()) return bad("a sensor has no id");
    for (std::size_t j = 0; j < i; ++j) {
      if (sensors[j].id == s.id) return bad("sensor " + s.id + " is repeated");
    }
    if (s.model != sensors[0].model) {
      return bad("sensors " + sensors[0].id + " (" + sensors[0].model +
                 ") and " + s.id + " (" + s.model +
                 ") are of different models: gains alone cannot match them");
    }
    if (!s.color || !s.depth) {
      return bad("sensor " + s.id + " has no colour or no depth");
    }
  }
  if (options.sets == 0 || options.samples < options.sets) {
    return bad("options of no sets, or fewer samples than sets");
  }
  std::unique_ptr<ColorFit> fit(new ColorFit());
  fit->device_ = &device;
  fit->allocator_ = &allocator;
  fit->sensors_ = std::move(sensors);
  fit->options_ = options;
  fit->max_groups_ = device.caps().limits().maxComputeWorkGroupCount[0];
  fit->max_range_ = core::max_storage_buffer_range(device);

  const auto push = [](std::uint32_t bytes) {
    VkPushConstantRange r{};
    r.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    r.size = bytes;
    return r;
  };
  const VkPushConstantRange blur_push = push(sizeof(BlurPush));
  const VkPushConstantRange sample_push = push(sizeof(SamplePush));
  const VkPushConstantRange normal_push = push(sizeof(NormalPush));
  const VkPushConstantRange pushed_push = push(sizeof(std::uint32_t));
  core::KernelSetBuilder kb(device);
  VKC_TRY(kb.add(fit->blur_, "color_fit.blur", vs_color_fit_blur_comp_spv,
                 vs_color_fit_blur_comp_spv_size, 4, &blur_push));
  VKC_TRY(kb.add(fit->sample_, "color_fit.sample", vs_color_fit_sample_comp_spv,
                 vs_color_fit_sample_comp_spv_size, 7, &sample_push));
  VKC_TRY(kb.add(fit->normal_equations_, "color_fit.normal_equations",
                 vs_color_fit_normal_equations_comp_spv,
                 vs_color_fit_normal_equations_comp_spv_size, 6, &normal_push));
  VKC_TRY(kb.add(fit->pushed_, "color_fit.pushed", vs_color_fit_pushed_comp_spv,
                 vs_color_fit_pushed_comp_spv_size, 4, &pushed_push));
  VKC_ASSIGN(fit->pool_, kb.build());
  return fit;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("ColorFit::create: out of host memory");
}

core::Status ColorFit::ensure_set_buffers(const DeviceFrameSet& set) {
  const std::size_t n = sensors_.size();
  if (!depth_at_.empty()) {
    for (std::size_t k = 0; k < n; ++k) {
      const DeviceFrame& f = *set.frames[k];
      if (pack_size(f.depth_camera.width, f.depth_camera.height) !=
              depth_size_[k] ||
          pack_size(f.color_camera.width, f.color_camera.height) !=
              color_size_[k]) {
        return bad("sensor " + sensors_[k].id +
                   "'s frames changed size between sets");
      }
    }
    return {};
  }
  std::uint64_t depth_total = 0, color_total = 0, color_max = 0;
  for (std::size_t k = 0; k < n; ++k) {
    const DeviceFrame& f = *set.frames[k];
    if (f.depth_camera.width >= 65536 || f.depth_camera.height >= 65536 ||
        f.color_camera.width >= 65536 || f.color_camera.height >= 65536) {
      return bad("a frame is too large");
    }
    depth_at_.push_back(static_cast<std::uint32_t>(depth_total));
    color_at_.push_back(static_cast<std::uint32_t>(color_total));
    depth_size_.push_back(
        pack_size(f.depth_camera.width, f.depth_camera.height));
    color_size_.push_back(
        pack_size(f.color_camera.width, f.color_camera.height));
    const std::uint64_t color =
        std::uint64_t{f.color_camera.width} * f.color_camera.height;
    depth_total += std::uint64_t{f.depth_camera.width} * f.depth_camera.height;
    color_total += color;
    color_max = std::max(color_max, color);
  }
  if (2 * color_total >= (std::uint64_t{1} << 32)) {
    return bad("the set's colour is past a 32-bit index");
  }
  depth_total_ = static_cast<std::uint32_t>(depth_total);
  color_total_ = static_cast<std::uint32_t>(color_total);
  color_max_ = static_cast<std::uint32_t>(color_max);

  const std::uint64_t budget =
      (options_.samples + options_.sets - 1) / options_.sets;
  const std::uint64_t slack =
      4 * static_cast<std::uint64_t>(std::ceil(std::sqrt(double(budget)))) + 64;
  const std::uint64_t capacity = options_.sets * (budget + slack);
  if (capacity >= (std::uint64_t{1} << 32)) return bad("too many samples");
  capacity_ = static_cast<std::uint32_t>(capacity);

  struct Want {
    core::Buffer* buffer;
    VkDeviceSize bytes;
    const char* name;
  };
  const Want wants[] = {
      {&cameras_, n * sizeof(GpuCamera), "ColorFit: cameras"},
      {&depths_, depth_total * sizeof(float), "ColorFit: depth"},
      {&words_, color_total * sizeof(std::uint32_t), "ColorFit: colour"},
      {&rows_, color_max * 8, "ColorFit: blur rows"},
      {&blurred_, 2 * color_total * 8, "ColorFit: blurred colour"},
      {&counters_, 4 * sizeof(std::uint32_t), "ColorFit: counters"},
      {&samples_, capacity * sizeof(Sample), "ColorFit: samples"},
      {&histogram_, n * kBins * sizeof(std::uint32_t), "ColorFit: histogram"},
  };
  for (const Want& w : wants) {
    VKC_TRY(core::check_storage_buffer_range(w.name, w.bytes, max_range_));
    VKC_ASSIGN(*w.buffer, core::device_storage_buffer(*allocator_, w.bytes));
  }
  blur_.set.write_storage_buffer(0, cameras_.handle(), 0, VK_WHOLE_SIZE);
  blur_.set.write_storage_buffer(1, words_.handle(), 0, VK_WHOLE_SIZE);
  blur_.set.write_storage_buffer(2, rows_.handle(), 0, VK_WHOLE_SIZE);
  blur_.set.write_storage_buffer(3, blurred_.handle(), 0, VK_WHOLE_SIZE);
  const core::Buffer* sample_bindings[] = {&cameras_,  &depths_,   &words_,
                                           &blurred_,  &counters_, &samples_,
                                           &histogram_};
  for (std::uint32_t b = 0; b < 7; ++b) {
    sample_.set.write_storage_buffer(b, sample_bindings[b]->handle(), 0,
                                     VK_WHOLE_SIZE);
  }
  return {};
}

core::Status ColorFit::add(const DeviceFrameSet& set) try {
  const std::size_t n = sensors_.size();
  if (sets_ >= options_.sets) {
    return bad("add: a set past the " + std::to_string(options_.sets) +
               " the fit takes");
  }
  if (set.frames.size() != n) {
    return bad("add: a set of " + std::to_string(set.frames.size()) +
               " frames for " + std::to_string(n) + " sensors");
  }
  for (std::size_t k = 0; k < n; ++k) {
    const std::optional<DeviceFrame>& f = set.frames[k];
    if (!f || !f->has_color() || !f->depth) {
      return bad("add: sensor " + sensors_[k].id +
                 "'s frame is missing, or has no colour or depth");
    }
    const VkDeviceSize color_bytes = VkDeviceSize{f->color_camera.width} *
                                     f->color_camera.height *
                                     sizeof(std::uint32_t);
    const VkDeviceSize depth_bytes = VkDeviceSize{f->depth_camera.width} *
                                     f->depth_camera.height * sizeof(float);
    if (f->color->size() < color_bytes || f->depth->size() < depth_bytes) {
      return bad("add: sensor " + sensors_[k].id +
                 "'s buffers are smaller than its cameras");
    }
  }
  if (core::Status made = ensure_set_buffers(set); !made.ok()) {
    if (sets_ == 0) {  // the next set lays the buffers out afresh
      depth_at_.clear();
      color_at_.clear();
      depth_size_.clear();
      color_size_.clear();
    }
    return made;
  }

  std::vector<GpuCamera> cams(n);
  for (std::size_t k = 0; k < n; ++k) {
    const DeviceFrame& f = *set.frames[k];
    GpuCamera& g = cams[k];
    g.depth_k[0] = f.depth_camera.fx;
    g.depth_k[1] = f.depth_camera.fy;
    g.depth_k[2] = f.depth_camera.cx;
    g.depth_k[3] = f.depth_camera.cy;
    g.depth_width = f.depth_camera.width;
    g.depth_height = f.depth_camera.height;
    copy_matrix(f.depth_camera.cam_to_world, g.depth_to_world);
    g.color_k[0] = f.color_camera.fx;
    g.color_k[1] = f.color_camera.fy;
    g.color_k[2] = f.color_camera.cx;
    g.color_k[3] = f.color_camera.cy;
    g.color_width = f.color_camera.width;
    g.color_height = f.color_camera.height;
    copy_matrix(f.color_camera.cam_to_world, g.color_to_world);
    g.depth_at = depth_at_[k];
    g.color_at = color_at_[k];
  }

  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(batch.upload(cameras_, 0, cams.data(), n * sizeof(GpuCamera)));
  for (std::size_t k = 0; k < n; ++k) {
    const DeviceFrame& f = *set.frames[k];
    VKC_TRY(batch.copy(*f.depth, 0, depths_, VkDeviceSize{depth_at_[k]} * 4,
                       VkDeviceSize{f.depth_camera.width} *
                           f.depth_camera.height * sizeof(float)));
    VKC_TRY(batch.copy(*f.color, 0, words_, VkDeviceSize{color_at_[k]} * 4,
                       VkDeviceSize{f.color_camera.width} *
                           f.color_camera.height * sizeof(std::uint32_t)));
  }
  if (sets_ == 0) {
    VKC_TRY(batch.zero(counters_, 0, counters_.size()));
    VKC_TRY(batch.zero(histogram_, 0, histogram_.size()));
  } else {
    VKC_TRY(batch.zero(counters_, 0, sizeof(std::uint32_t)));
  }

  // Each camera's colour at sigma 2 and 8.
  for (std::uint32_t k = 0; k < n; ++k) {
    const std::uint32_t pixels = cams[k].color_width * cams[k].color_height;
    const std::uint32_t groups = core::group_count(pixels, kLocal);
    for (const float sigma : {2.0f, 8.0f}) {
      const BlurPush across{k, 0, sigma, 0};
      VKC_TRY(
          batch.dispatch(blur_, &across, sizeof(across), groups, max_groups_));
      const BlurPush down{k, 1, sigma, sigma == 2.0f ? 0u : color_total_};
      VKC_TRY(batch.dispatch(blur_, &down, sizeof(down), groups, max_groups_));
    }
  }

  // Every ordered pair counted, then kept, each pair's samples a run of
  // their own.
  const std::uint32_t budget = static_cast<std::uint32_t>(
      (options_.samples + options_.sets - 1) / options_.sets);
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
  for (std::uint32_t a = 0; a < n; ++a) {
    for (std::uint32_t b = 0; b < n; ++b) {
      if (a != b) pairs.emplace_back(a, b);
    }
  }
  std::vector<std::uint32_t> ends(pairs.size(), 0);
  std::uint32_t overflow = 0;
  for (const std::uint32_t keep : {0u, 1u}) {
    for (std::size_t p = 0; p < pairs.size(); ++p) {
      const auto [a, b] = pairs[p];
      const SamplePush push{a,      b,         keep,  sets_,
                            budget, capacity_, 0u,    color_total_,
                            kBins,  kBinWidth, kCubeM};
      VKC_TRY(batch.dispatch(
          sample_, &push, sizeof(push),
          core::group_count(cams[a].depth_width * cams[a].depth_height, kLocal),
          max_groups_));
      if (keep == 1) {
        VKC_TRY(batch.readback(counters_, 4, 4, &ends[p]));
      }
    }
  }
  VKC_TRY(batch.readback(counters_, 8, 4, &overflow));
  VKC_TRY(batch.submit());
  if (overflow != 0) {
    sets_ = options_.sets;  // nothing more is added to a broken fit
    return bad("add: the kept samples outgrew their buffer");
  }

  std::uint64_t start = kept_;
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const std::uint64_t end = std::min<std::uint64_t>(ends[p], capacity_);
    if (end > start) {
      segments_.push_back({start, end - start, static_cast<std::uint32_t>(p)});
    }
    start = std::max(start, end);
  }
  kept_ = start;
  ++sets_;
  chunks_ready_ = false;
  return {};
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("ColorFit::add: out of host memory");
}

core::Status ColorFit::ensure_chunks() {
  if (chunks_ready_) return {};
  if (sets_ == 0 || kept_ == 0) return bad("no samples: add sets first");
  const std::size_t n = sensors_.size();
  const std::size_t pair_count = n * (n - 1);
  std::vector<std::uint32_t> chunks;  // uvec4 each
  std::vector<std::uint32_t> pairs(2 * pair_count, 0);
  for (std::size_t p = 0; p < pair_count; ++p) {
    pairs[2 * p] = static_cast<std::uint32_t>(chunks.size() / 4);
    for (const Segment& s : segments_) {
      if (s.pair != p) continue;
      for (std::uint64_t at = 0; at < s.count; at += kChunk) {
        chunks.push_back(static_cast<std::uint32_t>(s.first + at));
        chunks.push_back(
            static_cast<std::uint32_t>(std::min(kChunk, s.count - at)));
        chunks.push_back(static_cast<std::uint32_t>(p));
        chunks.push_back(0);
      }
    }
    pairs[2 * p + 1] =
        static_cast<std::uint32_t>(chunks.size() / 4) - pairs[2 * p];
  }
  chunk_count_ = static_cast<std::uint32_t>(chunks.size() / 4);
  if (chunk_count_ > max_groups_) return bad("too many chunks to dispatch");

  struct Want {
    core::Buffer* buffer;
    VkDeviceSize bytes;
    const char* name;
  };
  const Want wants[] = {
      {&params_, n * sizeof(GpuResponse), "ColorFit: responses"},
      {&chunks_, chunks.size() * 4, "ColorFit: chunks"},
      {&pairs_, pairs.size() * 4, "ColorFit: pairs"},
      {&partials_, VkDeviceSize{chunk_count_} * kNormalLocal * kRow * 4,
       "ColorFit: partial rows"},
      {&rows_out_, pair_count * kRow * 4, "ColorFit: rows"},
      {&counts_, n * 2 * 4, "ColorFit: white counts"},
  };
  for (const Want& w : wants) {
    VKC_TRY(core::check_storage_buffer_range(w.name, w.bytes, max_range_));
    VKC_ASSIGN(*w.buffer, core::device_storage_buffer(*allocator_, w.bytes));
  }
  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(batch.upload(chunks_, 0, chunks.data(), chunks.size() * 4));
  VKC_TRY(batch.upload(pairs_, 0, pairs.data(), pairs.size() * 4));
  VKC_TRY(batch.submit());
  const core::Buffer* normal_bindings[] = {&samples_,  &params_, &chunks_,
                                           &partials_, &pairs_,  &rows_out_};
  for (std::uint32_t b = 0; b < 6; ++b) {
    normal_equations_.set.write_storage_buffer(b, normal_bindings[b]->handle(),
                                               0, VK_WHOLE_SIZE);
  }
  const core::Buffer* pushed_bindings[] = {&cameras_, &words_, &params_,
                                           &counts_};
  for (std::uint32_t b = 0; b < 4; ++b) {
    pushed_.set.write_storage_buffer(b, pushed_bindings[b]->handle(), 0,
                                     VK_WHOLE_SIZE);
  }
  chunks_ready_ = true;
  return {};
}

core::Result<ColorFit::Rows> ColorFit::evaluate(
    const std::vector<camera::ColorResponse>& responses) try {
  const std::size_t n = sensors_.size();
  if (responses.size() != n) return bad("one response per camera");
  VKC_TRY(ensure_chunks());
  std::vector<GpuResponse> params(n);
  for (std::size_t k = 0; k < n; ++k) params[k] = gpu_response(responses[k]);
  const std::size_t pair_count = n * (n - 1);
  std::vector<float> rows(pair_count * kRow, 0.0f);
  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(batch.upload(params_, 0, params.data(), n * sizeof(GpuResponse)));
  const NormalPush partial{0, 0, kHuberLevels};
  VKC_TRY(batch.dispatch(normal_equations_, &partial, sizeof(partial),
                         chunk_count_, max_groups_));
  const NormalPush reduce{1, 0, kHuberLevels};
  VKC_TRY(batch.dispatch(normal_equations_, &reduce, sizeof(reduce),
                         static_cast<std::uint32_t>(pair_count), max_groups_));
  VKC_TRY(batch.readback(rows_out_, 0, rows.size() * 4, rows.data()));
  VKC_TRY(batch.submit());
  Rows out;
  out.v.assign(rows.begin(), rows.end());
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("ColorFit: out of host memory");
}

namespace {

// The ordered pairs in index order.
std::vector<std::pair<std::uint32_t, std::uint32_t>> ordered_pairs(
    std::size_t n) {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
  for (std::uint32_t a = 0; a < n; ++a) {
    for (std::uint32_t b = 0; b < n; ++b) {
      if (a != b) pairs.emplace_back(a, b);
    }
  }
  return pairs;
}

// The system the rows of an evaluation sum to.
System system_of(const std::vector<double>& rows, std::size_t cameras) {
  System s;
  s.n = 5 * cameras - 3;
  s.h.assign(s.n * s.n, 0.0);
  s.g.assign(s.n, 0.0);
  const auto pairs = ordered_pairs(cameras);
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const double* row = rows.data() + p * kRow;
    const auto [i, j] = pairs[p];
    for (std::uint32_t c = 0; c < 3; ++c) {
      // The moments' 5x5 matrix from its upper triangle.
      double m[5][5];
      int at = 0;
      for (int r = 0; r < 5; ++r) {
        for (int q = r; q < 5; ++q) m[r][q] = m[q][r] = row[c * 20 + at++];
      }
      const double* rhs = row + c * 20 + 15;
      struct Entry {
        int index;
        int basis;
        double sign;
      };
      const Entry entries[6] = {
          {gain_index(i, c), 0, 1.0}, {gain_index(j, c), 0, -1.0},
          {a_index(i), 1, 1.0},       {b_index(i), 2, 1.0},
          {a_index(j), 3, -1.0},      {b_index(j), 4, -1.0}};
      for (const Entry& e : entries) {
        if (e.index < 0) continue;
        const auto ei = static_cast<std::size_t>(e.index);
        s.g[ei] += e.sign * rhs[e.basis];
        for (const Entry& f : entries) {
          if (f.index < 0) continue;
          s.h[ei * s.n + static_cast<std::size_t>(f.index)] +=
              e.sign * f.sign * m[e.basis][f.basis];
        }
      }
    }
    s.loss += row[kLoss];
  }
  return s;
}

Residuals residuals_of(const std::vector<double>& rows, std::size_t cameras) {
  Residuals out;
  double train = 0.0, held = 0.0;
  const auto pairs = ordered_pairs(cameras);
  for (std::size_t p = 0; p < pairs.size(); ++p) {
    const double* row = rows.data() + p * kRow;
    PairResidual r;
    r.a = pairs[p].first;
    r.b = pairs[p].second;
    r.train_samples = static_cast<std::uint64_t>(row[kCount]);
    r.held_out_samples = static_cast<std::uint64_t>(row[kCount + 1]);
    r.train = rms(row[kSquares], row[kCount]);
    r.held_out = rms(row[kSquares + 1], row[kCount + 1]);
    train += row[kSquares];
    held += row[kSquares + 1];
    out.train_samples += r.train_samples;
    out.held_out_samples += r.held_out_samples;
    out.pairs.push_back(r);
  }
  out.train = rms(train, static_cast<double>(out.train_samples));
  out.held_out = rms(held, static_cast<double>(out.held_out_samples));
  return out;
}

}  // namespace

core::Result<Residuals> ColorFit::residuals(
    const std::vector<camera::ColorResponse>& responses) {
  VKC_ASSIGN(const Rows rows, evaluate(responses));
  return residuals_of(rows.v, sensors_.size());
}

core::Result<System> ColorFit::system(
    const std::vector<camera::ColorResponse>& responses) {
  VKC_ASSIGN(const Rows rows, evaluate(responses));
  return system_of(rows.v, sensors_.size());
}

core::Result<std::vector<Sample>> ColorFit::samples() try {
  std::vector<Sample> out(kept_);
  if (kept_ == 0) return out;
  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(batch.readback(samples_, 0, kept_ * sizeof(Sample), out.data()));
  VKC_TRY(batch.submit());
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("ColorFit: out of host memory");
}

core::Result<std::vector<double>> ColorFit::pushed(
    const std::vector<camera::ColorResponse>& responses) try {
  const std::size_t n = sensors_.size();
  VKC_TRY(ensure_chunks());
  std::vector<GpuResponse> params(n);
  for (std::size_t k = 0; k < n; ++k) params[k] = gpu_response(responses[k]);
  std::vector<std::uint32_t> counts(2 * n, 0);
  core::CommandBatch batch(*device_, *allocator_);
  VKC_TRY(batch.upload(params_, 0, params.data(), n * sizeof(GpuResponse)));
  VKC_TRY(batch.zero(counts_, 0, counts_.size()));
  for (std::uint32_t k = 0; k < n; ++k) {
    const std::uint32_t pixels =
        (color_size_[k] & 0xFFFFu) * (color_size_[k] >> 16);
    VKC_TRY(batch.dispatch(pushed_, &k, sizeof(k),
                           core::group_count(pixels, kLocal), max_groups_));
  }
  VKC_TRY(batch.readback(counts_, 0, counts.size() * 4, counts.data()));
  VKC_TRY(batch.submit());
  std::vector<double> out(n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    if (counts[2 * k] > 0) {
      out[k] = double(counts[2 * k + 1]) / double(counts[2 * k]);
    }
  }
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("ColorFit: out of host memory");
}

core::Result<Report> ColorFit::solve(
    const camera::ArrayCalibration& calibration) try {
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t n = sensors_.size();
  if (!calibration.color_controls) {
    return bad(
        "solve: the calibration names no color_controls, so no response "
        "could hold");
  }
  for (const SensorInfo& s : sensors_) {
    if (camera::find_sensor(calibration, s.id) == nullptr) {
      return bad("solve: the calibration has no sensor " + s.id);
    }
  }
  if (sets_ == 0 || kept_ == 0) return bad("solve: no samples: add sets");

  // Each camera's kept pixels and r_max, from the radius histogram.
  std::vector<std::uint32_t> histogram(n * kBins, 0);
  {
    core::CommandBatch batch(*device_, *allocator_);
    VKC_TRY(
        batch.readback(histogram_, 0, histogram.size() * 4, histogram.data()));
    VKC_TRY(batch.submit());
  }
  std::vector<std::uint64_t> pixels(n, 0);
  std::vector<double> r_max(n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    const std::uint32_t* h = histogram.data() + k * kBins;
    pixels[k] = std::accumulate(h, h + kBins, std::uint64_t{0});
    std::uint64_t seen = 0;
    for (std::uint32_t bin = 0; bin < kBins; ++bin) {
      seen += h[bin];
      if (double(seen) >= kRmaxQuantile * double(pixels[k])) {
        r_max[k] = double(bin + 1) * kBinWidth;
        break;
      }
    }
  }
  for (std::size_t k = 0; k < n; ++k) {
    if (pixels[k] < options_.min_camera_samples) {
      return bad("solve: sensor " + sensors_[k].id + " has " +
                 std::to_string(pixels[k]) + " kept pixels, fewer than " +
                 std::to_string(options_.min_camera_samples) +
                 ": it sees too little of what the others see");
    }
  }
  // The overlap graph joins every camera.
  {
    std::vector<std::uint64_t> between(n * n, 0);
    const auto pairs = ordered_pairs(n);
    for (const Segment& s : segments_) {
      const auto [a, b] = pairs[s.pair];
      between[std::min(a, b) * n + std::max(a, b)] += s.count;
    }
    std::vector<std::size_t> root(n);
    std::iota(root.begin(), root.end(), 0);
    const auto find = [&](std::size_t x) {
      while (root[x] != x) x = root[x] = root[root[x]];
      return x;
    };
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = a + 1; b < n; ++b) {
        if (between[a * n + b] >= options_.min_pair_samples) {
          root[find(a)] = find(b);
        }
      }
    }
    std::vector<std::string> groups(n);
    std::size_t count = 0;
    for (std::size_t k = 0; k < n; ++k) {
      std::string& g = groups[find(k)];
      if (g.empty()) ++count;
      g += (g.empty() ? "" : ", ") + sensors_[k].id;
    }
    if (count > 1) {
      std::string apart;
      for (const std::string& g : groups) {
        if (!g.empty()) apart += (apart.empty() ? "{" : " | {") + g + "}";
      }
      return bad("solve: the overlap graph leaves cameras apart: " + apart +
                 " share fewer than " +
                 std::to_string(options_.min_pair_samples) + " samples");
    }
  }

  Report report;
  std::vector<camera::ColorResponse> theta(n);
  for (std::size_t k = 0; k < n; ++k) theta[k].r_max = r_max[k];
  VKC_ASSIGN(Rows rows, evaluate(theta));
  int evaluations = 1;
  const Residuals identity = residuals_of(rows.v, n);
  System sys = system_of(rows.v, n);

  // Gauss-Newton IRLS with Levenberg-Marquardt damping.
  double lambda = 1e-3;
  int iterations = 0;
  for (int it = 0; it < kMaxIterations && lambda < 1e10; ++it) {
    auto delta = step(sys, lambda);
    if (!delta) {
      lambda *= 10.0;
      continue;
    }
    std::vector<camera::ColorResponse> next = theta;
    for (std::uint32_t k = 0; k < n; ++k) {
      for (std::uint32_t c = 0; c < 3; ++c) {
        const int gi = gain_index(k, c);
        if (gi >= 0) next[k].gain[c] *= std::exp((*delta)[std::size_t(gi)]);
      }
      next[k].a += (*delta)[std::size_t(a_index(k))];
      next[k].b += (*delta)[std::size_t(b_index(k))];
    }
    bool valid = true;
    for (const camera::ColorResponse& r : next) {
      valid = valid && camera::validate(r).ok();
    }
    if (!valid) {
      lambda *= 10.0;
      continue;
    }
    VKC_ASSIGN(Rows trial, evaluate(next));
    ++evaluations;
    const System trial_sys = system_of(trial.v, n);
    if (trial_sys.loss < sys.loss) {
      const double gain = (sys.loss - trial_sys.loss) / sys.loss;
      theta = std::move(next);
      rows = std::move(trial);
      sys = trial_sys;
      lambda = std::max(lambda * 0.1, 1e-12);
      ++iterations;
      double largest = 0.0;
      for (const double d : *delta) largest = std::max(largest, std::fabs(d));
      if (gain < 1e-10 || largest < 1e-9) break;
    } else {
      lambda *= 10.0;
    }
  }

  // The G2 gauge: per channel, the mean of gain F over both pixels of every
  // kept sample is 1, from each camera's n, sum q and sum q^2.
  std::vector<double> moments(3 * n, 0.0);  // n, sum q, sum q^2
  {
    const auto pairs = ordered_pairs(n);
    for (std::size_t p = 0; p < pairs.size(); ++p) {
      const double* row = rows.pair(p);
      const double count = row[kCount] + row[kCount + 1];
      const auto [a, b] = pairs[p];
      moments[3 * a] += count;
      moments[3 * a + 1] += row[kGauge];
      moments[3 * a + 2] += row[kGauge + 1];
      moments[3 * b] += count;
      moments[3 * b + 1] += row[kGauge + 2];
      moments[3 * b + 2] += row[kGauge + 3];
    }
  }
  double all = 0.0;
  for (std::size_t k = 0; k < n; ++k) all += moments[3 * k];
  for (std::uint32_t c = 0; c < 3; ++c) {
    double mean = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      mean +=
          theta[k].gain[c] * (moments[3 * k] + theta[k].a * moments[3 * k + 1] +
                              theta[k].b * moments[3 * k + 2]);
    }
    mean /= all;
    for (std::size_t k = 0; k < n; ++k) theta[k].gain[c] /= mean;
  }

  for (std::size_t k = 0; k < n; ++k) {
    if (!camera::validate(theta[k]).ok()) {
      return bad("solve: sensor " + sensors_[k].id +
                 "'s fitted falloff is not above 0 up to r_max");
    }
  }
  const Residuals fit = residuals_of(rows.v, n);
  if (!(fit.held_out < identity.held_out)) {
    return bad("solve: the fit's held-out RMS, " +
               std::to_string(fit.held_out) +
               " levels, is not below the identity's, " +
               std::to_string(identity.held_out));
  }
  VKC_ASSIGN(const std::vector<double> white, pushed(theta));

  report.calibration = calibration;
  for (std::size_t k = 0; k < n; ++k) {
    Camera cam;
    cam.id = sensors_[k].id;
    cam.response = theta[k];
    cam.samples = pixels[k];
    cam.pushed_to_white = white[k];
    report.cameras.push_back(cam);
    for (camera::SensorCalibration& s : report.calibration.sensors) {
      if (s.id == cam.id) s.color_response = theta[k];
    }
  }
  VKC_TRY(camera::validate_array_calibration(report.calibration));
  report.pairs = fit.pairs;
  for (std::size_t p = 0; p < report.pairs.size(); ++p) {
    report.pairs[p].identity = identity.pairs[p].held_out;
  }
  report.rms_identity = identity.held_out;
  report.rms_fit = fit.train;
  report.rms_held_out = fit.held_out;
  report.train_samples = fit.train_samples;
  report.held_out_samples = fit.held_out_samples;
  report.iterations = iterations;
  report.evaluations = evaluations;
  report.solve_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  return report;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("ColorFit::solve: out of host memory");
}

std::string format_report(const Report& report) {
  std::string out;
  char line[256];
  std::snprintf(line, sizeof(line),
                "RMS at sigma 8, 8-bit levels: identity %.2f (held out), fit "
                "%.2f (train), %.2f (held out); %llu / %llu samples; %d "
                "steps, %.0f ms\n",
                report.rms_identity, report.rms_fit, report.rms_held_out,
                static_cast<unsigned long long>(report.train_samples),
                static_cast<unsigned long long>(report.held_out_samples),
                report.iterations, report.solve_ms);
  out += line;
  for (const PairResidual& p : report.pairs) {
    std::snprintf(line, sizeof(line),
                  "  %s -> %s: identity %.2f, fit %.2f / %.2f (%llu / %llu)\n",
                  report.cameras[p.a].id.c_str(),
                  report.cameras[p.b].id.c_str(), p.identity, p.train,
                  p.held_out, static_cast<unsigned long long>(p.train_samples),
                  static_cast<unsigned long long>(p.held_out_samples));
    out += line;
  }
  for (const Camera& c : report.cameras) {
    const camera::ColorResponse& r = c.response;
    const double mean = (r.gain[0] + r.gain[1] + r.gain[2]) / 3.0;
    std::snprintf(line, sizeof(line),
                  "  %s: %llu samples, gain %.4f %.4f %.4f, correction %+.3f "
                  "stop at r 0 and %+.3f at r_max %.4f, %.2f%% pushed to "
                  "255\n",
                  c.id.c_str(), static_cast<unsigned long long>(c.samples),
                  r.gain[0], r.gain[1], r.gain[2], stops(mean),
                  stops(mean * r.falloff(r.r_max)), r.r_max,
                  100.0 * c.pushed_to_white);
    out += line;
  }
  return out;
}

}  // namespace volumetric_kit::sensor::color_fit
