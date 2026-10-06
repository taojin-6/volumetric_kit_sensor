// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A host allocation failure while staging a batch returns OutOfMemory, keeps
// the previous frame intact, and permits a retry. Skips without a device.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "no_device.hpp"
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

int main() {
  auto instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  auto gpu = instance->select_physical_device(vr::device_requirements());
  if (!gpu) {
    return vr_test::no_device("no compute-capable device",
                              gpu.status().message());
  }
  auto device = vkc::Device::create(*instance, *gpu, vr::device_requirements());
  CHECK(device.ok());
  auto allocator = vkc::Allocator::create(instance->handle(), *device);
  CHECK(allocator.ok());
  auto prep = sensor::GpuFramePrep::create(*device, *allocator);
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
  auto held = sensor::GpuFramePrep::prepare_batch(preps, frames);
  CHECK(held.ok());
  CHECK(held->size() == 1 && (*held)[0].has_value());

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
  failure.armed = true;
  try {
    auto refused = sensor::GpuFramePrep::prepare_batch(preps, frames);
    failure.armed = false;
    CHECK(failure.injected);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::OutOfMemory);
  } catch (const std::bad_alloc&) {
    failure.armed = false;
    std::fprintf(stderr, "FAIL: bad_alloc escaped prepare_batch\n");
    return 1;
  }

  auto retry = sensor::GpuFramePrep::prepare_batch(preps, frames);
  CHECK(retry.ok());
  CHECK(retry->size() == 1 && (*retry)[0].has_value());
  CHECK((*held)[0]->depth != (*retry)[0]->depth);
  auto before = vr_test::read_back<float>(*device, *allocator,
                                          *(*held)[0]->depth, depth.size());
  auto after = vr_test::read_back<float>(*device, *allocator,
                                         *(*retry)[0]->depth, depth.size());
  CHECK(before.ok() && after.ok());
  for (std::size_t i = 0; i < depth.size(); ++i) {
    CHECK((*before)[i] == 1000.0f * frame.metres_per_unit);
    CHECK((*after)[i] == 2000.0f * frame.metres_per_unit);
  }
  std::puts("sensor_gpu_frame_prep_oom: OK");
  return 0;
}
