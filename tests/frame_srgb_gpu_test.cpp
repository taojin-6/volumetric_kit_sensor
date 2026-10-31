// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// sensor/frame/shaders/srgb.glsl on the device (shaders/srgb_cases.comp)
// against frame/color_space.hpp's curve, which it mirrors, on all 256 codes:
// each code decoded to the host's linear value within float round-off of the
// device's pow (2e-6 relative), and the host's linear value of each code
// encoded back to that code exactly. Skips where no device is present.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "srgb_cases_comp.spv.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

namespace {

constexpr std::uint32_t kCodes = 256;

int gpu_main(vkc::test::GpuContext& gpu) {
  std::vector<float> probe(kCodes);
  for (std::uint32_t c = 0; c < kCodes; ++c) {
    probe[c] = sensor::srgb_to_linear(static_cast<float>(c) / 255.0f);
  }
  vkc::ComputeKernel kernel;
  vkc::KernelSetBuilder builder(gpu.device);
  CHECK(builder
            .add(kernel, "srgb_cases", vs_srgb_cases_comp_spv,
                 vs_srgb_cases_comp_spv_size, 3)
            .ok());
  auto pool = builder.build();
  auto in = vkc::test::upload_device_buffer(
      gpu.device, gpu.allocator, probe.data(), kCodes * sizeof(float));
  auto linear =
      vkc::device_storage_buffer(gpu.allocator, kCodes * sizeof(float));
  auto encoded =
      vkc::device_storage_buffer(gpu.allocator, kCodes * sizeof(float));
  CHECK(pool.ok() && in.ok() && linear.ok() && encoded.ok());
  kernel.set.write_storage_buffer(0, in->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(1, linear->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(2, encoded->handle(), 0, VK_WHOLE_SIZE);
  CHECK(vkc::dispatch(gpu.device, kernel, nullptr, 0,
                      vkc::group_count(kCodes, 64),
                      gpu.device.caps().limits().maxComputeWorkGroupCount[0])
            .ok());
  auto got_linear =
      vkc::test::read_back<float>(gpu.device, gpu.allocator, *linear, kCodes);
  auto got_encoded =
      vkc::test::read_back<float>(gpu.device, gpu.allocator, *encoded, kCodes);
  CHECK(got_linear.ok() && got_encoded.ok());

  int failures = 0;
  double worst_linear = 0.0, worst_encoded = 0.0;
  for (std::uint32_t c = 0; c < kCodes; ++c) {
    const double want = probe[c];
    const double gl = got_linear.value()[c];
    const double ge = got_encoded.value()[c];
    const double rel = std::fabs(gl - want) / std::max(want, 1e-12);
    const double e = std::fabs(ge - static_cast<double>(c) / 255.0);
    worst_linear = std::max(worst_linear, c == 0 ? std::fabs(gl) : rel);
    worst_encoded = std::max(worst_encoded, e);
    const auto code = static_cast<std::uint32_t>(
        std::fmin(std::fmax(ge, 0.0), 1.0) * 255.0 + 0.5);
    const bool ok = (c == 0 ? gl == 0.0 : rel <= 2e-6) && code == c;
    if (!ok && ++failures <= 10) {
      std::fprintf(stderr,
                   "code %u: linear %.9g (host %.9g), encoded %.9g -> %u\n", c,
                   gl, want, ge, code);
    }
  }
  std::printf(
      "256 codes: worst decode %.3g relative, worst encode %.3g, %d "
      "failures\n",
      worst_linear, worst_encoded, failures);
  CHECK(failures == 0);
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
