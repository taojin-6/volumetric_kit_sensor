// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrepConfig::color_response on one picture in every layout the colour
// pass reads (prep_layouts.hpp), through a barrel lens:
//   - every colour layout against the host oracle within a code: the pass's
//     R'G'B' clamped to [0, 1], decoded by color_space.hpp's curve, times
//     gain * F(r) at the pinhole pixel's radius, held beyond r_max, encoded;
//     the picture leaves [0, 1] at both ends and most pixels lie beyond
//     r_max, so neither the clamp nor the hold can go unseen;
//   - an identity response, whatever its r_max, gives the default pass's
//     words bit for bit;
//   - luma alone is refused on a pass with a response, and prepared on one
//     without;
//   - create refuses a response camera::validate refuses, or one past
//     float's range.
// Skips where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "prep_layouts.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/color_response.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
using prep_layouts::Layout;
using prep_layouts::Picture;

namespace {

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 240;

camera::CameraModel lensed() {
  camera::CameraModel c;
  c.size = {kWidth, kHeight};
  c.intrinsics = {260.0f, 258.0f, 161.3f, 118.7f};
  c.distortion = {-0.28f, 0.09f, 0.0012f, -0.0009f, -0.01f, 0.05f, 0.0f, 0.0f};
  return c;
}

// Gains below and above 1, and a falloff that dims by 8% at r_max, 0.3, past
// which most of the image lies.
camera::ColorResponse response() {
  camera::ColorResponse r;
  r.gain = {0.8, 1.15, 0.93};
  r.a = -0.9;
  r.b = 0.6;
  r.r_max = 0.3;
  return r;
}

sensor::GpuFramePrepConfig config_of(const camera::ColorResponse& r) {
  sensor::GpuFramePrepConfig config;
  config.depth_outliers.enabled = false;
  config.color_response = r;
  return config;
}

struct Frame {
  std::vector<std::uint16_t> raw;
  sensor::RgbdFrame f;
};

Frame frame_of(const camera::CameraModel& cam) {
  Frame out;
  out.raw.assign(std::size_t{cam.size.width} * cam.size.height, 1500);
  out.f.depth = out.raw.data();
  out.f.metres_per_unit = 0.001f;
  out.f.depth_camera = cam;
  out.f.color_camera = cam;
  out.f.min_depth = 0.1f;
  out.f.max_depth = 10.0f;
  return out;
}

std::vector<std::uint32_t> words_of(vkc::test::GpuContext& gpu,
                                    sensor::GpuFramePrep& prep,
                                    sensor::RgbdFrame& f,
                                    const Layout& layout) {
  f.color = layout.image;
  f.color_packed = layout.packed;
  auto out = prep.prepare(f);
  if (!out) {
    std::fprintf(stderr, "%s: %s\n", layout.name.c_str(),
                 out.status().message().c_str());
    return {};
  }
  auto words = vkc::test::read_back<std::uint32_t>(
      gpu.device, gpu.allocator, *out->color,
      std::size_t{f.color_camera.size.width} * f.color_camera.size.height);
  return words.ok() ? std::move(words).value() : std::vector<std::uint32_t>{};
}

// The oracle's word at pinhole pixel (u, v), in float as the pass computes,
// F from ColorResponse::falloff at the pixel's radius; 0 outside the picture.
std::uint32_t oracle(const Picture& p, const Layout& layout,
                     const camera::CameraModel& cam,
                     const camera::ColorResponse& r, std::uint32_t u,
                     std::uint32_t v, bool* overshoot, bool* beyond) {
  sensor::Vec3f rgb;
  if (!prep_layouts::rgb_at(p, layout, cam, u, v, &rgb)) return 0;
  const float nx =
      (static_cast<float>(u) - prep_layouts::f32(cam.intrinsics.cx)) /
      prep_layouts::f32(cam.intrinsics.fx);
  const float ny =
      (static_cast<float>(v) - prep_layouts::f32(cam.intrinsics.cy)) /
      prep_layouts::f32(cam.intrinsics.fy);
  const double radius = std::sqrt(double{nx} * nx + double{ny} * ny);
  *beyond = radius > r.r_max;
  const auto f = static_cast<float>(r.falloff(radius));
  std::uint32_t word = 0xFF000000u;
  for (int k = 0; k < 3; ++k) {
    *overshoot = *overshoot || rgb[k] > 1.0f;
    const float linear =
        sensor::srgb_to_linear(std::clamp(rgb[k], 0.0f, 1.0f)) *
        (static_cast<float>(r.gain[static_cast<std::size_t>(k)]) * f);
    word |= prep_layouts::code(sensor::linear_to_srgb(linear)) << (8 * k);
  }
  return word;
}

int test_layouts(vkc::test::GpuContext& gpu, const Picture& p) {
  const camera::CameraModel cam = lensed();
  const camera::ColorResponse r = response();
  auto plain =
      sensor::GpuFramePrep::create(gpu.device, gpu.allocator, config_of({}));
  auto responded =
      sensor::GpuFramePrep::create(gpu.device, gpu.allocator, config_of(r));
  CHECK(plain.ok() && responded.ok());
  Frame frame = frame_of(cam);
  int layouts = 0;
  for (const Layout& layout : p.layouts) {
    if (layout.image.layout == sensor::YuvLayout::Luma &&
        layout.packed == nullptr) {
      continue;  // test_luma's
    }
    ++layouts;
    const std::vector<std::uint32_t> got =
        words_of(gpu, responded.value(), frame.f, layout);
    const std::vector<std::uint32_t> today =
        words_of(gpu, plain.value(), frame.f, layout);
    CHECK(got.size() == frame.raw.size() && today.size() == got.size());
    int worst = 0, off = 0, coverage_off = 0, overshoots = 0, beyond = 0,
        changed = 0;
    for (std::uint32_t v = 0; v < kHeight; ++v) {
      for (std::uint32_t u = 0; u < kWidth; ++u) {
        const std::size_t i = std::size_t{v} * kWidth + u;
        bool over = false, past = false;
        const std::uint32_t want =
            oracle(p, layout, cam, r, u, v, &over, &past);
        if ((got[i] >> 24) != (want >> 24)) {
          ++coverage_off;
          continue;
        }
        if (want == 0) continue;
        overshoots += over ? 1 : 0;
        beyond += past ? 1 : 0;
        const int diff = prep_layouts::channel_diff(got[i], want);
        worst = std::max(worst, diff);
        off += diff > 1 ? 1 : 0;
        changed += prep_layouts::channel_diff(got[i], today[i]) > 2 ? 1 : 0;
      }
    }
    std::printf(
        "  %-12s worst %d, %d past a code, %d coverage off; %d overshoot, %d "
        "beyond r_max, %d changed\n",
        layout.name.c_str(), worst, off, coverage_off, overshoots, beyond,
        changed);
    const int pixels = static_cast<int>(frame.raw.size());
    CHECK(off == 0 && worst <= 1);
    CHECK(coverage_off <= 20);  // within round-off of the picture's edge
    // Not vacuous: the matrix overshoots (packed words cannot), most pixels
    // hold F, and the response moves most words.
    CHECK(layout.packed != nullptr || overshoots > pixels / 50);
    CHECK(beyond > pixels / 2 && changed > pixels / 2);
  }
  CHECK(layouts == 6);
  return 0;
}

// An identity response is the default pass, bit for bit, whatever its r_max.
int test_identity(vkc::test::GpuContext& gpu, const Picture& p) {
  auto plain =
      sensor::GpuFramePrep::create(gpu.device, gpu.allocator, config_of({}));
  camera::ColorResponse identity;
  identity.r_max = 0.4;
  CHECK(identity.is_identity());
  auto same = sensor::GpuFramePrep::create(gpu.device, gpu.allocator,
                                           config_of(identity));
  CHECK(plain.ok() && same.ok());
  Frame frame = frame_of(lensed());
  for (const Layout& layout : p.layouts) {
    const std::vector<std::uint32_t> a =
        words_of(gpu, plain.value(), frame.f, layout);
    CHECK(!a.empty() && a == words_of(gpu, same.value(), frame.f, layout));
  }
  return 0;
}

// Luma alone is grey: a pass with a response refuses it, before any work,
// and prepares the next frame; one without prepares it.
int test_luma(vkc::test::GpuContext& gpu, const Picture& p) {
  auto responded = sensor::GpuFramePrep::create(gpu.device, gpu.allocator,
                                                config_of(response()));
  auto plain =
      sensor::GpuFramePrep::create(gpu.device, gpu.allocator, config_of({}));
  CHECK(responded.ok() && plain.ok());
  Frame frame = frame_of(lensed());
  const Layout* mono = nullptr;
  for (const Layout& layout : p.layouts) {
    if (layout.packed == nullptr &&
        layout.image.layout == sensor::YuvLayout::Luma) {
      mono = &layout;
    }
  }
  CHECK(mono != nullptr);
  frame.f.color = mono->image;
  const vkc::Status refused = responded->prepare(frame.f).status();
  std::printf("  refused as expected: %s\n", refused.message().c_str());
  CHECK(refused.domain() == vkc::Status::Code::InvalidArgument &&
        refused.message().find("luma alone") != std::string::npos);
  CHECK(plain->prepare(frame.f).ok());
  // Depth alone, and colour, still prepare on it.
  frame.f.color = {};
  CHECK(responded->prepare(frame.f).ok());
  CHECK(!words_of(gpu, responded.value(), frame.f, p.layouts.front()).empty());
  return 0;
}

int test_create_refusals(vkc::test::GpuContext& gpu) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<camera::ColorResponse> bad(6, response());
  bad[0].gain[0] = 0.0;
  bad[1].a = nan;
  bad[2].r_max = 0.0;
  bad[3].a = -20.0;        // F(r_max) = 1 - 1.8 + ...: below 0
  bad[4].gain[2] = 1e300;  // finite, but not as a float
  bad[5].b = -1e300;
  for (std::size_t i = 0; i < bad.size(); ++i) {
    const vkc::Status s = sensor::GpuFramePrep::create(
                              gpu.device, gpu.allocator, config_of(bad[i]))
                              .status();
    std::printf("  refused as expected: %s\n", s.message().c_str());
    CHECK(s.domain() == vkc::Status::Code::InvalidArgument &&
          s.message().find("GpuFramePrep::create: ") == 0);
  }
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  auto picture =
      prep_layouts::make_picture(gpu.device, gpu.allocator, kWidth, kHeight);
  CHECK(picture->layouts.size() == 7);
  if (test_layouts(gpu, *picture) != 0) return 1;
  if (test_identity(gpu, *picture) != 0) return 1;
  if (test_luma(gpu, *picture) != 0) return 1;
  if (test_create_refusals(gpu) != 0) return 1;
  std::puts("sensor_gpu_frame_prep_response: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
