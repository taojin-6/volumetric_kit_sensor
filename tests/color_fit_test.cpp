// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// tools/color_fit's fit (color_fit::ColorFit) on synthetic rigs: posed
// cameras with a barrel lens over a textured floor and box, ray-cast on the
// host into raw pictures with planted per-camera gains and vignetting and
// 2-level noise, prepared by GpuFramePrep.
//   - The fit, then the sets prepared again through GpuFramePrep with the
//     fitted responses: the held-out residual falls to the noise floor (the
//     same rig planted with nothing), and to the fit's own held-out figure,
//     which ties the fit's r, cap and sRGB curve to the colour pass's.
//   - G2: the mean of gain F over both pixels of every kept sample is 1 per
//     channel, in double on the host; r_max, the samples' 99th-percentile
//     radius; the share pushed to 255, against a host count.
//   - One IRLS step on about 1 k samples against a double host oracle, at
//     responses whose r_max leaves most samples beyond it.
//   - The hold-out half, a function of world position alone.
//   - The refusals: too few sensors, a repeated id, mixed models, a sensor
//     without colour or depth, no hold-out cube; a set of another size, a
//     frame without colour or depth, frames that change size, a set too
//     many; no colour controls, a sensor the calibration lacks, a camera
//     with too few samples and a split overlap graph, each named.
// Skips where no device is present.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "color_fit.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/color_response.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace color_fit = volumetric_kit::sensor::color_fit;

namespace {

using camera::Mat4d;
using camera::Vec3d;

// --- The rig and the scene --------------------------------------------------

camera::CameraModel color_model() {
  camera::CameraModel c;
  c.size = {640, 480};
  c.intrinsics = {470.0, 470.0, 319.3, 240.6};
  c.distortion.k1 = -0.12;
  c.distortion.k2 = 0.03;
  return c;
}

camera::CameraModel depth_model() {
  camera::CameraModel c;
  c.size = {320, 240};
  c.intrinsics = {240.0, 240.0, 159.6, 119.8};
  return c;
}

// A camera at `eye` looking at `target`, the world's z up: +X right, +Y
// down, +Z forward.
Mat4d look_at(const Vec3d& eye, const Vec3d& target) {
  const Vec3d f = glm::normalize(target - eye);
  const Vec3d x = glm::normalize(glm::cross(f, Vec3d(0.0, 0.0, 1.0)));
  const Vec3d y = glm::cross(f, x);
  Mat4d m(1.0);
  m[0] = glm::dvec4(x, 0.0);
  m[1] = glm::dvec4(y, 0.0);
  m[2] = glm::dvec4(f, 0.0);
  m[3] = glm::dvec4(eye, 1.0);
  return m;
}

// Three cameras around the scene's centre.
std::vector<Mat4d> ring_rig() {
  std::vector<Mat4d> poses;
  for (int k = 0; k < 3; ++k) {
    const double phi = 2.0943951 * k + 0.2;
    poses.push_back(
        look_at(Vec3d(1.7 * std::cos(phi), 1.7 * std::sin(phi), 1.5 + 0.05 * k),
                Vec3d(0.1, -0.05, 0.1)));
  }
  return poses;
}

// Two pairs of cameras 8 m apart: two overlaps, none between them.
std::vector<Mat4d> split_rig() {
  std::vector<Mat4d> poses;
  for (const double shift : {0.0, 8.0}) {
    for (int k = 0; k < 2; ++k) {
      const double phi = 2.0943951 * k + 0.2;
      poses.push_back(
          look_at(Vec3d(shift + 1.7 * std::cos(phi), 1.7 * std::sin(phi), 1.5),
                  Vec3d(shift, 0.0, 0.0)));
    }
  }
  return poses;
}

// The scene's linear albedo where the ray from `o` along `d` first meets the
// floor (z = 0) or the box; false if it meets neither.
bool trace(const Vec3d& o, const Vec3d& d, double* t, Vec3d* albedo) {
  const Vec3d lo(-0.35, -0.25, 0.0), hi(0.35, 0.25, 0.45);
  double t0 = 0.0, t1 = 1e9;
  bool box = true;
  for (int a = 0; a < 3; ++a) {
    if (std::fabs(d[a]) < 1e-12) {
      if (o[a] < lo[a] || o[a] > hi[a]) box = false;
      continue;
    }
    double ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a];
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
  }
  box = box && t0 <= t1 && t0 > 0.0;
  if (box) {
    const Vec3d p = o + t0 * d;
    const double s = std::sin(6.2831853 * (p.x + p.y + p.z) / 1.1);
    *t = t0;
    *albedo = Vec3d(0.12, 0.15, 0.22) * (1.0 + 0.25 * s);
    return true;
  }
  if (!(d.z < 0.0)) return false;
  *t = -o.z / d.z;
  const Vec3d p = o + *t * d;
  const double s =
      std::sin(6.2831853 * p.x / 1.7 + 0.3) * std::cos(6.2831853 * p.y / 1.5);
  const double u = std::sin(6.2831853 * (p.x - p.y) / 2.3);
  *albedo =
      Vec3d(0.20, 0.17, 0.13) * (1.0 + 0.3 * s) + Vec3d(0.02, 0.03, 0.01) * u;
  return true;
}

// What a camera does to the scene's light: raw = albedo * gain / F(r), F
// the response family at the pinhole radius, uncapped.
struct Planted {
  std::array<double, 3> gain{1.0, 1.0, 1.0};
  double a = 0.0;
  double b = 0.0;
};

std::vector<Planted> planted_responses() {
  return {{{1.00, 1.00, 1.00}, 0.8, 0.6},
          {{0.85, 0.90, 0.95}, 0.6, 1.2},
          {{1.10, 1.05, 0.92}, 1.0, 0.3}};
}

double srgb_encode(double l) {
  return l <= 0.0031308 ? 12.92 * l : 1.055 * std::pow(l, 1.0 / 2.4) - 0.055;
}

struct Raw {
  std::vector<std::uint16_t> depth;
  std::vector<std::uint32_t> words;
  sensor::RgbdFrame frame;
};

// The pinhole point a raw pixel of `cam` sees: the lens inverted by
// fixed-point iteration.
void undistort(const camera::CameraModel& cam, double u, double v, double* x,
               double* y) {
  const double d[8] = {cam.distortion.k1, cam.distortion.k2, cam.distortion.p1,
                       cam.distortion.p2, cam.distortion.k3, cam.distortion.k4,
                       cam.distortion.k5, cam.distortion.k6};
  const double xd = (u - cam.intrinsics.cx) / cam.intrinsics.fx;
  const double yd = (v - cam.intrinsics.cy) / cam.intrinsics.fy;
  *x = xd;
  *y = yd;
  for (int it = 0; it < 40; ++it) {
    double ex = 0.0, ey = 0.0;
    camera::distort_rational(d, *x, *y, &ex, &ey);
    *x += xd - ex;
    *y += yd - ey;
  }
}

Raw render(const Mat4d& pose, const Planted& planted, double noise,
           std::uint32_t seed) {
  Raw raw;
  const camera::CameraModel cc = color_model();
  const camera::CameraModel dc = depth_model();
  const Vec3d eye(pose[3]);
  const camera::Mat3d rot(pose);
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, noise);
  raw.words.resize(std::size_t{cc.size.width} * cc.size.height);
  for (std::uint32_t v = 0; v < cc.size.height; ++v) {
    for (std::uint32_t u = 0; u < cc.size.width; ++u) {
      double x = 0.0, y = 0.0;
      undistort(cc, u, v, &x, &y);
      double t = 0.0;
      Vec3d albedo(0.1);
      trace(eye, rot * Vec3d(x, y, 1.0), &t, &albedo);
      const double q = x * x + y * y;
      const double f = 1.0 + planted.a * q + planted.b * q * q;
      std::uint32_t word = 0;
      for (int c = 0; c < 3; ++c) {
        const double lin = albedo[c] * planted.gain[std::size_t(c)] / f;
        const double code = std::round(
            std::clamp(255.0 * srgb_encode(lin) + gauss(rng), 0.0, 255.0));
        word |= static_cast<std::uint32_t>(code) << (8 * c);
      }
      raw.words[std::size_t{v} * cc.size.width + u] = word;
    }
  }
  raw.depth.resize(std::size_t{dc.size.width} * dc.size.height);
  for (std::uint32_t v = 0; v < dc.size.height; ++v) {
    for (std::uint32_t u = 0; u < dc.size.width; ++u) {
      const Vec3d ray((u - dc.intrinsics.cx) / dc.intrinsics.fx,
                      (v - dc.intrinsics.cy) / dc.intrinsics.fy, 1.0);
      double t = 0.0;
      Vec3d albedo;
      const bool hit = trace(eye, rot * ray, &t, &albedo);
      raw.depth[std::size_t{v} * dc.size.width + u] =
          hit && t < 60.0 ? static_cast<std::uint16_t>(std::lround(t * 1000.0))
                          : 0;
    }
  }
  sensor::RgbdFrame& f = raw.frame;
  f.depth = raw.depth.data();
  f.metres_per_unit = 0.001f;
  f.depth_camera = dc;
  f.min_depth = 0.2f;
  f.max_depth = 8.0f;
  f.color_packed = raw.words.data();
  f.color_camera = cc;
  f.color_to_world = pose;
  return raw;
}

std::vector<sensor::SensorInfo> infos(std::size_t n) {
  std::vector<sensor::SensorInfo> out(n);
  for (std::size_t k = 0; k < n; ++k) {
    out[k].id = "cam" + std::to_string(k);
    out[k].model = "Synthetic RGB-D";
    out[k].color = color_model();
    out[k].depth = depth_model();
  }
  return out;
}

camera::ArrayCalibration calibration_of(const std::vector<Mat4d>& poses) {
  camera::ArrayCalibration c;
  c.color_controls = camera::ColorControls{167, 60};
  for (std::size_t k = 0; k < poses.size(); ++k) {
    camera::SensorCalibration s;
    s.id = "cam" + std::to_string(k);
    s.color_to_world = poses[k];
    c.sensors.push_back(s);
  }
  return c;
}

// --- Sets prepared ----------------------------------------------------------

struct Rig {
  std::vector<Mat4d> poses;
  std::vector<Planted> planted;
};

// `seeds.size()` sets of `rig`, each camera prepared by its own pass with
// `responses[k]` (identity when empty), the whole rig and scene moved by
// `offset` in the world.
std::vector<sensor::DeviceFrameSet> prepared_sets(
    vkc::test::GpuContext& gpu, const Rig& rig,
    const std::vector<std::uint32_t>& seeds,
    const std::vector<camera::ColorResponse>& responses = {},
    const Vec3d& offset = Vec3d(0.0)) {
  std::vector<sensor::GpuFramePrep> preps;
  for (std::size_t k = 0; k < rig.poses.size(); ++k) {
    sensor::GpuFramePrepConfig config;
    if (!responses.empty()) config.color_response = responses[k];
    auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator, config);
    if (!prep) {
      std::fprintf(stderr, "prep: %s\n", prep.status().message().c_str());
      return {};
    }
    preps.push_back(std::move(prep).value());
  }
  std::vector<sensor::DeviceFrameSet> sets;
  for (const std::uint32_t seed : seeds) {
    std::vector<Raw> raws;
    raws.reserve(rig.poses.size());
    std::vector<std::optional<sensor::RgbdFrame>> frames;
    for (std::size_t k = 0; k < rig.poses.size(); ++k) {
      raws.push_back(render(rig.poses[k], rig.planted[k], 2.0,
                            seed * 7919u + static_cast<std::uint32_t>(k)));
      frames.push_back(raws.back().frame);
      frames.back()->color_to_world[3] += glm::dvec4(offset, 0.0);
    }
    auto prepared = sensor::GpuFramePrep::prepare_batch(preps, frames);
    if (!prepared) {
      std::fprintf(stderr, "prepare: %s\n",
                   prepared.status().message().c_str());
      return {};
    }
    sensor::DeviceFrameSet set;
    set.sequence = seed;
    set.frames = std::move(prepared).value();
    sets.push_back(std::move(set));
  }
  return sets;
}

std::unique_ptr<color_fit::ColorFit> fit_of(
    vkc::test::GpuContext& gpu, const std::vector<sensor::DeviceFrameSet>& sets,
    std::size_t cameras, const color_fit::Options& options) {
  auto fit = color_fit::ColorFit::create(gpu.device, gpu.allocator,
                                         infos(cameras), options);
  if (!fit) {
    std::fprintf(stderr, "create: %s\n", fit.status().message().c_str());
    return nullptr;
  }
  for (const sensor::DeviceFrameSet& s : sets) {
    const vkc::Status added = (*fit)->add(s);
    if (!added.ok()) {
      std::fprintf(stderr, "add: %s\n", added.message().c_str());
      return nullptr;
    }
  }
  return std::move(fit).value();
}

std::vector<camera::ColorResponse> identities(std::size_t n) {
  return std::vector<camera::ColorResponse>(n);
}

bool says(const vkc::Status& s, const std::string& what) {
  const bool ok = !s.ok() && s.message().find(what) != std::string::npos;
  if (!ok) {
    std::fprintf(stderr, "expected a refusal naming \"%s\", got \"%s\"\n",
                 what.c_str(), s.ok() ? "OK" : s.message().c_str());
  }
  return ok;
}

// --- The host oracle --------------------------------------------------------

// The normal equations of design section 4 in double, from the samples:
// System's parameter order, half 0, Huber at 3 levels, F from
// ColorResponse::falloff (the colour pass's oracle).
color_fit::System oracle_system(const std::vector<color_fit::Sample>& samples,
                                const std::vector<camera::ColorResponse>& r) {
  const std::size_t cams = r.size();
  color_fit::System s;
  s.n = 5 * cams - 3;
  s.h.assign(s.n * s.n, 0.0);
  s.g.assign(s.n, 0.0);
  const auto gain_at = [](std::size_t k, std::size_t c) {
    return k == 0 ? -1 : static_cast<int>(2 + (k - 1) * 5 + c);
  };
  const auto a_at = [](std::size_t k) {
    return k == 0 ? 0 : static_cast<int>(2 + (k - 1) * 5 + 3);
  };
  for (const color_fit::Sample& x : samples) {
    if (((x.cams >> 16) & 1u) != 0) continue;
    const std::size_t i = x.cams & 0xFFu, j = (x.cams >> 8) & 0xFFu;
    const double qa = std::min<double>(x.qa, r[i].r_max * r[i].r_max);
    const double qb = std::min<double>(x.qb, r[j].r_max * r[j].r_max);
    const double fa = r[i].falloff(std::sqrt(double(x.qa)));
    const double fb = r[j].falloff(std::sqrt(double(x.qb)));
    for (std::size_t c = 0; c < 3; ++c) {
      const double p = x.a[c] * r[i].gain[c] * fa / (r[j].gain[c] * fb);
      const double res = 255.0 * (srgb_encode(p) - srgb_encode(x.b[c]));
      const double slope =
          255.0 * p *
          (p <= 0.0031308 ? 12.92
                          : (1.055 / 2.4) * std::pow(p, 1.0 / 2.4 - 1.0));
      const double w = std::fabs(res) <= 3.0 ? 1.0 : 3.0 / std::fabs(res);
      // J over (index, value) pairs.
      const std::pair<int, double> jac[6] = {
          {gain_at(i, c), slope},      {gain_at(j, c), -slope},
          {a_at(i), slope * qa / fa},  {a_at(i) + 1, slope * qa * qa / fa},
          {a_at(j), -slope * qb / fb}, {a_at(j) + 1, -slope * qb * qb / fb}};
      for (const auto& [m, vm] : jac) {
        if (m < 0) continue;
        s.g[std::size_t(m)] += w * vm * res;
        for (const auto& [q, vq] : jac) {
          if (q < 0) continue;
          s.h[std::size_t(m) * s.n + std::size_t(q)] += w * vm * vq;
        }
      }
      const double m = std::fabs(res);
      s.loss += m <= 3.0 ? 0.5 * res * res : 3.0 * (m - 1.5);
    }
  }
  return s;
}

// (h + lambda diag h) delta = -g by Gaussian elimination with partial
// pivoting, in double: independent of color_fit::step's Cholesky.
std::vector<double> oracle_step(const color_fit::System& s, double lambda) {
  const std::size_t n = s.n;
  std::vector<double> m(n * (n + 1));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m[i * (n + 1) + j] = s.h[i * n + j] * (i == j ? 1.0 + lambda : 1.0);
    }
    m[i * (n + 1) + n] = -s.g[i];
  }
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t best = c;
    for (std::size_t r = c + 1; r < n; ++r) {
      if (std::fabs(m[r * (n + 1) + c]) > std::fabs(m[best * (n + 1) + c])) {
        best = r;
      }
    }
    for (std::size_t k = 0; k <= n; ++k) {
      std::swap(m[c * (n + 1) + k], m[best * (n + 1) + k]);
    }
    for (std::size_t r = 0; r < n; ++r) {
      if (r == c) continue;
      const double f = m[r * (n + 1) + c] / m[c * (n + 1) + c];
      for (std::size_t k = c; k <= n; ++k) {
        m[r * (n + 1) + k] -= f * m[c * (n + 1) + k];
      }
    }
  }
  std::vector<double> x(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = m[i * (n + 1) + n] / m[i * (n + 1) + i];
  }
  return x;
}

double srgb_decode(double e) {
  return e <= 0.04045 ? e / 12.92 : std::pow((e + 0.055) / 1.055, 2.4);
}

struct Pushed {
  double share = -1.0;  // of the covered pixels
  std::size_t covered = 0;
  std::size_t near = 0;  // within 0.05 code of 254.5, either way
};

// pushed.comp's count in double, from the frame's prepared words read back:
// the covered pixels with no channel at 255 whose linear colour times
// gain F(r), r the pinhole pixel's radius, encodes to 254.5 or above.
Pushed host_pushed(vkc::test::GpuContext& gpu, const sensor::DeviceFrame& f,
                   const camera::ColorResponse& r) {
  Pushed out;
  const auto& cam = f.color_camera;
  std::vector<std::uint32_t> words(std::size_t{cam.width} * cam.height);
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  if (!batch.readback(*f.color, 0, words.size() * 4, words.data()).ok() ||
      !batch.submit().ok()) {
    return out;
  }
  std::size_t pushed = 0;
  for (std::uint32_t v = 0; v < cam.height; ++v) {
    for (std::uint32_t u = 0; u < cam.width; ++u) {
      const std::uint32_t word = words[std::size_t{v} * cam.width + u];
      if ((word >> 24) != 0xFFu) continue;
      ++out.covered;
      double top = 0.0;
      bool white = false;
      const double x = (u - double(cam.cx)) / cam.fx;
      const double y = (v - double(cam.cy)) / cam.fy;
      const double f_r = r.falloff(std::sqrt(x * x + y * y));
      for (int c = 0; c < 3; ++c) {
        const std::uint32_t code = (word >> (8 * c)) & 0xFFu;
        white = white || code == 255u;
        top = std::max(top, 255.0 * srgb_encode(srgb_decode(code / 255.0) *
                                                r.gain[std::size_t(c)] * f_r));
      }
      if (white) continue;
      pushed += top >= 254.5;
      out.near += std::fabs(top - 254.5) < 0.05;
    }
  }
  out.share = out.covered > 0 ? double(pushed) / double(out.covered) : 0.0;
  return out;
}

// --- The tests --------------------------------------------------------------

// Fit, apply through GpuFramePrep, and compare with the noise floor; G2.
int test_round_trip(vkc::test::GpuContext& gpu) {
  const Rig planted{ring_rig(), planted_responses()};
  const Rig plain{ring_rig(), std::vector<Planted>(3)};
  const std::vector<std::uint32_t> seeds = {1, 2};
  color_fit::Options options;
  options.sets = 2;

  const auto sets = prepared_sets(gpu, planted, seeds);
  CHECK(sets.size() == seeds.size());
  auto fit = fit_of(gpu, sets, 3, options);
  CHECK(fit != nullptr);
  const auto t0 = std::chrono::steady_clock::now();
  auto report = fit->solve(calibration_of(planted.poses));
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  if (!report) {
    std::fprintf(stderr, "solve: %s\n", report.status().message().c_str());
  }
  CHECK(report.ok());
  std::printf("%s", color_fit::format_report(*report).c_str());
  std::printf("solve %.1f ms on the host clock, %d evaluations\n", ms,
              report->evaluations);
  CHECK(report->rms_identity > 4.0);
  CHECK(report->rms_held_out < 0.25 * report->rms_identity);
  CHECK(report->held_out_samples > 20'000 && report->train_samples > 20'000);
  for (const color_fit::Camera& c : report->cameras) {
    CHECK(camera::validate(c.response).ok() && !c.response.is_identity());
    const camera::SensorCalibration* s =
        camera::find_sensor(report->calibration, c.id);
    CHECK(s != nullptr && s->color_response.gain == c.response.gain);
  }
  CHECK(camera::format_array_calibration(report->calibration).ok());

  // G2, in double from the kept samples.
  auto samples = fit->samples();
  CHECK(samples.ok() && !samples->empty());
  double worst_gauge = 0.0;
  for (std::size_t c = 0; c < 3; ++c) {
    double sum = 0.0;
    for (const color_fit::Sample& x : *samples) {
      const camera::ColorResponse& ra =
          report->cameras[x.cams & 0xFFu].response;
      const camera::ColorResponse& rb =
          report->cameras[(x.cams >> 8) & 0xFFu].response;
      sum += ra.gain[c] * ra.falloff(std::sqrt(double(x.qa))) +
             rb.gain[c] * rb.falloff(std::sqrt(double(x.qb)));
    }
    worst_gauge = std::max(
        worst_gauge, std::fabs(sum / (2.0 * double(samples->size())) - 1.0));
  }
  std::printf("G2: worst channel mean of gain F off 1 by %.3g\n", worst_gauge);
  CHECK(worst_gauge < 1e-5);

  // r_max, the 99th percentile of each camera's kept pixels' radii, from
  // the samples in double: the upper edge of its 1/2048 bin.
  std::vector<std::vector<double>> radii(3);
  for (const color_fit::Sample& x : *samples) {
    radii[x.cams & 0xFFu].push_back(std::sqrt(double(x.qa)));
    radii[(x.cams >> 8) & 0xFFu].push_back(std::sqrt(double(x.qb)));
  }
  for (std::size_t k = 0; k < 3; ++k) {
    std::vector<double>& v = radii[k];
    CHECK(report->cameras[k].samples == v.size() && !v.empty());
    std::sort(v.begin(), v.end());
    const double p99 =
        v[static_cast<std::size_t>(std::ceil(0.99 * double(v.size()))) - 1];
    const double above = report->cameras[k].response.r_max - p99;
    std::printf("r_max cam%zu: %.6f, the samples' p99 %.6f\n", k,
                report->cameras[k].response.r_max, p99);
    CHECK(above >= -1e-6 && above <= 1.0 / 2048.0 + 1e-6);
  }

  // The share pushed to 255: the report's at the fitted responses, and at
  // strong ones against the host on the last set.
  std::vector<camera::ColorResponse> fitted;
  for (const color_fit::Camera& c : report->cameras)
    fitted.push_back(c.response);
  auto white = fit->pushed_to_white(fitted);
  CHECK(white.ok() && white->size() == 3);
  std::vector<camera::ColorResponse> strong(3);
  for (std::size_t k = 0; k < 3; ++k) {
    CHECK((*white)[k] == report->cameras[k].pushed_to_white);
    strong[k].gain = {7.0 - 0.5 * double(k), 6.0, 6.0 + 0.5 * double(k)};
    strong[k].a = 0.5;
    strong[k].b = -0.2;
    strong[k].r_max = 0.5;
  }
  auto strong_white = fit->pushed_to_white(strong);
  CHECK(strong_white.ok());
  for (std::size_t k = 0; k < 3; ++k) {
    const Pushed host = host_pushed(gpu, *sets.back().frames[k], strong[k]);
    const double off =
        std::fabs((*strong_white)[k] - host.share) * double(host.covered);
    std::printf(
        "pushed to 255 by a strong response, cam%zu: %.2f%% (host %.2f%%, "
        "%.0f pixels apart, %zu near the line)\n",
        k, 100.0 * (*strong_white)[k], 100.0 * host.share, off, host.near);
    CHECK(host.share > 0.02 && host.share < 0.98);
    CHECK(off <= double(host.near) + 2.0);
  }

  // The fitted responses applied by the colour pass, against the floor.
  const auto corrected_sets = prepared_sets(gpu, planted, seeds, fitted);
  auto corrected_fit = fit_of(gpu, corrected_sets, 3, options);
  CHECK(corrected_fit != nullptr);
  auto corrected = corrected_fit->residuals(identities(3));
  const auto floor_sets = prepared_sets(gpu, plain, seeds);
  auto floor_fit = fit_of(gpu, floor_sets, 3, options);
  CHECK(floor_fit != nullptr);
  auto floor = floor_fit->residuals(identities(3));
  CHECK(corrected.ok() && floor.ok());
  std::printf(
      "held out: planted %.3f, fit %.3f, applied by prep %.3f, noise floor "
      "%.3f levels\n",
      report->rms_identity, report->rms_held_out, corrected->held_out,
      floor->held_out);
  CHECK(corrected->held_out <= floor->held_out + 0.15);
  CHECK(std::fabs(corrected->held_out - report->rms_held_out) <= 0.15);
  return 0;
}

// One IRLS step on about 1 k samples against the double oracle.
int test_irls_step(vkc::test::GpuContext& gpu) {
  const Rig planted{ring_rig(), planted_responses()};
  color_fit::Options options;
  options.sets = 1;
  options.samples = 1000;
  const auto sets = prepared_sets(gpu, planted, {3});
  auto fit = fit_of(gpu, sets, 3, options);
  CHECK(fit != nullptr);
  auto samples = fit->samples();
  CHECK(samples.ok());
  std::size_t train = 0, beyond = 0;
  for (const color_fit::Sample& x : *samples) {
    train += ((x.cams >> 16) & 1u) == 0;
    beyond += x.qa > 0.35f * 0.35f;
  }
  std::printf("IRLS step: %zu samples, %zu in half 0, %zu beyond r_max\n",
              samples->size(), train, beyond);
  CHECK(samples->size() > 700 && samples->size() < 1400);
  CHECK(train > 200 && train < samples->size() - 200);
  CHECK(beyond > samples->size() / 4);

  // Responses away from the identity, r_max well inside the samples.
  std::vector<camera::ColorResponse> r(3);
  const double gains[3][3] = {
      {1.0, 1.0, 1.0}, {1.1, 0.95, 1.05}, {0.9, 1.05, 0.97}};
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t c = 0; c < 3; ++c) r[k].gain[c] = gains[k][c];
    r[k].a = 0.3 + 0.2 * double(k);
    r[k].b = 0.5 - 0.1 * double(k);
    r[k].r_max = 0.35;
  }
  auto gpu_system = fit->system(r);
  CHECK(gpu_system.ok());
  const color_fit::System host = oracle_system(*samples, r);
  CHECK(gpu_system->n == host.n);
  double worst_h = 0.0, worst_g = 0.0;
  for (std::size_t i = 0; i < host.n; ++i) {
    const double gscale = std::sqrt(host.h[i * host.n + i] * host.loss) + 1e-12;
    worst_g =
        std::max(worst_g, std::fabs(gpu_system->g[i] - host.g[i]) / gscale);
    for (std::size_t j = 0; j < host.n; ++j) {
      const double scale =
          std::sqrt(host.h[i * host.n + i] * host.h[j * host.n + j]) + 1e-12;
      worst_h = std::max(worst_h, std::fabs(gpu_system->h[i * host.n + j] -
                                            host.h[i * host.n + j]) /
                                      scale);
    }
  }
  const double loss_rel = std::fabs(gpu_system->loss - host.loss) / host.loss;
  const auto gpu_step = color_fit::step(*gpu_system, 1e-3);
  CHECK(gpu_step.ok());
  const std::vector<double> host_step = oracle_step(host, 1e-3);
  double step_scale = 0.0, worst_step = 0.0;
  for (std::size_t i = 0; i < host.n; ++i) {
    step_scale = std::max(step_scale, std::fabs(host_step[i]));
  }
  for (std::size_t i = 0; i < host.n; ++i) {
    worst_step = std::max(worst_step, std::fabs((*gpu_step)[i] - host_step[i]));
  }
  std::printf(
      "IRLS step: h %.3g, g %.3g, loss %.3g relative; step %.3g of %.3g\n",
      worst_h, worst_g, loss_rel, worst_step, step_scale);
  CHECK(worst_h < 1e-4 && worst_g < 1e-4 && loss_rel < 1e-4);
  CHECK(worst_step < 1e-3 * step_scale);
  return 0;
}

// The hold-out half is a function of world position alone: the rig and
// scene moved to (10, 10, 10) m, inside one 100 m cube, hold every sample in
// one half; at 25 cm cubes both halves are filled.
int test_hold_out_cubes(vkc::test::GpuContext& gpu) {
  const Rig planted{ring_rig(), planted_responses()};
  const auto sets =
      prepared_sets(gpu, planted, {6}, {}, Vec3d(10.0, 10.0, 10.0));
  CHECK(sets.size() == 1);
  for (const float cube : {100.0f, 0.25f}) {
    color_fit::Options options;
    options.sets = 1;
    options.samples = 4000;
    options.hold_out_cube_m = cube;
    auto fit = fit_of(gpu, sets, 3, options);
    CHECK(fit != nullptr);
    auto samples = fit->samples();
    CHECK(samples.ok() && samples->size() > 2000);
    std::size_t held = 0;
    for (const color_fit::Sample& x : *samples) held += (x.cams >> 16) & 1u;
    std::printf("hold-out at %.2f m cubes: %zu of %zu samples held out\n",
                double(cube), held, samples->size());
    if (cube > 1.0f) {
      CHECK(held == 0 || held == samples->size());
    } else {
      CHECK(held > samples->size() / 5 && held < samples->size() * 4 / 5);
    }
  }
  return 0;
}

int test_refusals(vkc::test::GpuContext& gpu) {
  const auto create = [&](std::vector<sensor::SensorInfo> s) {
    auto fit =
        color_fit::ColorFit::create(gpu.device, gpu.allocator, std::move(s));
    return fit.ok() ? vkc::Status{} : fit.status();
  };
  CHECK(says(create(infos(1)), "two sensors"));
  auto repeated = infos(3);
  repeated[2].id = "cam0";
  CHECK(says(create(repeated), "cam0 is repeated"));
  auto mixed = infos(3);
  mixed[1].model = "Another RGB-D";
  CHECK(says(create(mixed), "different models"));
  auto colourless = infos(3);
  colourless[2].color.reset();
  CHECK(says(create(colourless), "cam2 has no colour"));
  auto depthless = infos(3);
  depthless[1].depth.reset();
  CHECK(says(create(depthless), "cam1 has no colour or no depth"));
  color_fit::Options no_cube;
  no_cube.hold_out_cube_m = 0.0f;
  {
    auto fit = color_fit::ColorFit::create(gpu.device, gpu.allocator, infos(3),
                                           no_cube);
    CHECK(says(fit.ok() ? vkc::Status{} : fit.status(), "hold-out cube"));
  }

  const Rig planted{ring_rig(), planted_responses()};
  color_fit::Options options;
  options.sets = 1;
  const auto sets = prepared_sets(gpu, planted, {4});
  CHECK(sets.size() == 1);
  {
    auto fit = color_fit::ColorFit::create(gpu.device, gpu.allocator, infos(3),
                                           options);
    CHECK(fit.ok());
    sensor::DeviceFrameSet short_set = sets[0];
    short_set.frames.pop_back();
    CHECK(says((*fit)->add(short_set), "a set of 2 frames for 3"));
    sensor::DeviceFrameSet no_color = sets[0];
    no_color.frames[1]->color.reset();
    CHECK(says((*fit)->add(no_color), "cam1's frame"));
    sensor::DeviceFrameSet no_depth = sets[0];
    no_depth.frames[2]->depth.reset();
    CHECK(says((*fit)->add(no_depth), "cam2's frame"));
    CHECK(says((*fit)->solve(calibration_of(planted.poses)).status(),
               "no samples"));
    CHECK((*fit)->add(sets[0]).ok());
    CHECK(says((*fit)->add(sets[0]), "past the 1"));
    camera::ArrayCalibration uncontrolled = calibration_of(planted.poses);
    uncontrolled.color_controls.reset();
    CHECK(says((*fit)->solve(uncontrolled).status(), "no color_controls"));
    camera::ArrayCalibration missing = calibration_of(planted.poses);
    missing.sensors.pop_back();
    CHECK(says((*fit)->solve(missing).status(), "no sensor cam2"));
  }
  {
    color_fit::Options two = options;
    two.sets = 2;
    auto fit =
        color_fit::ColorFit::create(gpu.device, gpu.allocator, infos(3), two);
    CHECK(fit.ok() && (*fit)->add(sets[0]).ok());
    sensor::DeviceFrameSet resized = sets[0];
    resized.frames[0]->color_camera.width = 320;
    CHECK(says((*fit)->add(resized), "cam0's frames changed size"));
  }
  {
    color_fit::Options few = options;
    few.samples = 300;
    auto fit = fit_of(gpu, sets, 3, few);
    CHECK(fit != nullptr);
    CHECK(says(fit->solve(calibration_of(planted.poses)).status(),
               "kept pixels, fewer than 5000"));
  }
  {
    const Rig split{split_rig(), std::vector<Planted>(4)};
    const auto split_sets = prepared_sets(gpu, split, {5});
    CHECK(split_sets.size() == 1);
    color_fit::Options apart = options;
    apart.min_camera_samples = 100;
    auto fit = fit_of(gpu, split_sets, 4, apart);
    CHECK(fit != nullptr);
    CHECK(says(fit->solve(calibration_of(split.poses)).status(),
               "leaves cameras apart: {cam0, cam1} | {cam2, cam3}"));
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  if (test_round_trip(gpu) != 0) return 1;
  if (test_irls_step(gpu) != 0) return 1;
  if (test_hold_out_cubes(gpu) != 0) return 1;
  if (test_refusals(gpu) != 0) return 1;
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
