// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A host allocation failure while staging one frame or a batch returns
// OutOfMemory, keeps the previous frame intact, and permits a retry.
// Skips without a device.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "gpu_test.hpp"
#include "test_allocation_failure.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

namespace vkc = volumetric_kit::core;
namespace vr = volumetric_kit::recon;
namespace sensor = vr::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int staging_failure_case(vr_test::Gpu ctx, bool single) {
  auto prep = sensor::GpuFramePrep::create(ctx.device, ctx.allocator);
  CHECK(prep.ok());
  std::vector<sensor::GpuFramePrep> preps;
  preps.push_back(std::move(*prep));
  std::vector<std::uint16_t> depth(64, 1000);
  sensor::RgbdFrame frame;
  frame.depth = depth.data();
  frame.depth_camera = {{8, 8}, {8.0, 8.0, 3.5, 3.5}, {}};
  frame.min_depth = 0.1f;
  frame.max_depth = 5.0f;
  std::vector<std::optional<sensor::RgbdFrame>> frames{frame};
  const auto prepare = [&]() -> vkc::Result<sensor::DeviceFrame> {
    if (single) return preps[0].prepare(frame);
    VKC_ASSIGN(auto prepared,
               sensor::GpuFramePrep::prepare_batch(preps, frames));
    if (prepared.size() != 1 || !prepared[0]) {
      return vkc::Status::invalid_argument("test expected one prepared frame");
    }
    return std::move(*prepared[0]);
  };
  auto held = prepare();
  CHECK(held.ok());

  // Measure the allocation for a shared Buffer instead of assuming a standard
  // library's control-block size. This reaches output staging after
  // prepare_batch has allocated its bookkeeping vectors.
  auto& failure = vr_test::allocation_failure;
  failure.measure = true;
  auto owner = std::make_shared<vkc::Buffer>();
  failure.measure = false;
  CHECK(failure.bytes != 0 && !owner->valid());
  owner.reset();

  // Keeping the previous output forces a new one for this frame.
  for (auto& sample : depth) sample = 2000;
  failure.injected = false;
  failure.armed = true;
  try {
    auto refused = prepare();
    failure.armed = false;
    CHECK(failure.injected);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::OutOfMemory);
  } catch (const std::bad_alloc&) {
    failure.armed = false;
    std::fprintf(stderr, "FAIL: bad_alloc escaped %s\n",
                 single ? "prepare" : "prepare_batch");
    return 1;
  }

  auto retry = prepare();
  CHECK(retry.ok());
  CHECK(held->depth != retry->depth);
  auto before = vr_test::read_back<float>(ctx.device, ctx.allocator,
                                          *held->depth, depth.size());
  auto after = vr_test::read_back<float>(ctx.device, ctx.allocator,
                                         *retry->depth, depth.size());
  CHECK(before.ok() && after.ok());
  for (std::size_t i = 0; i < depth.size(); ++i) {
    CHECK((*before)[i] == 1000.0f * frame.metres_per_unit);
    CHECK((*after)[i] == 2000.0f * frame.metres_per_unit);
  }
  std::printf("sensor_gpu_frame_prep_oom: %s OK\n",
              single ? "single" : "batch");
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  CHECK(staging_failure_case(gpu, false) == 0);
  CHECK(staging_failure_case(gpu, true) == 0);
  return 0;
}

int main() { return vr_test::run_on_gpu(gpu_main); }
