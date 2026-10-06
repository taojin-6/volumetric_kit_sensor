// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A host allocation failure while staging a batch returns OutOfMemory, keeps
// the previous frame intact, and permits a retry. Skips without a device.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "no_device.hpp"
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

namespace {

// Limit injection to the calling thread, so a driver's worker cannot consume
// it. Measure the allocation for a shared Buffer instead of assuming a
// standard library's control-block size. This reaches output staging after
// prepare_batch has allocated its bookkeeping vectors.
thread_local bool measure_allocation = false;
thread_local std::size_t owner_bytes = 0;
thread_local bool fail_owner = false;
thread_local bool injected = false;

}  // namespace

void* operator new(std::size_t bytes) {
  if (measure_allocation) owner_bytes = bytes;
  if (fail_owner && bytes == owner_bytes) {
    fail_owner = false;  // The error Status can allocate its message.
    injected = true;
    throw std::bad_alloc();
  }
  if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) return p;
  throw std::bad_alloc();
}

void operator delete(void* p) noexcept { std::free(p); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
#if defined(__cpp_sized_deallocation)
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept {
  ::operator delete[](p);
}
#endif

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

  measure_allocation = true;
  auto owner = std::make_shared<vkc::Buffer>();
  measure_allocation = false;
  CHECK(owner_bytes != 0 && !owner->valid());
  owner.reset();

  // Keeping the previous output forces a new one for this frame.
  for (auto& sample : depth) sample = 2000;
  fail_owner = true;
  try {
    auto refused = sensor::GpuFramePrep::prepare_batch(preps, frames);
    fail_owner = false;
    CHECK(injected);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::OutOfMemory);
  } catch (const std::bad_alloc&) {
    fail_owner = false;
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
