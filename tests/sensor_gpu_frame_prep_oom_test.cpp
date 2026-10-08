// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A host with no memory to hold a new output: prepare and prepare_batch return
// OutOfMemory, without an exception, keep the frame handed out before intact,
// and prepare the frame on a retry. Skips without a device.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
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

namespace {

// `prepare(depth)` prepares an 8 x 8 frame of `depth`'s samples. A frame of
// 1000s is held, so a frame of 2000s needs a new output, whose holder the
// host then has no memory for.
template <class Prepare>
int check_out_of_memory(const char* who, vkc::Device& device,
                        vkc::Allocator& allocator, Prepare prepare) {
  std::vector<std::uint16_t> depth(64, 1000);
  auto held = prepare(depth);
  CHECK(held.ok());

  // The holder's size, measured rather than assumed of a standard library:
  // the pass's holder is make_shared's, its allocator holding no state.
  auto& failure = vr_test::allocation_failure;
  failure.measure = true;
  auto owner = std::make_shared<vkc::Buffer>();
  failure.measure = false;
  CHECK(failure.bytes != 0 && !owner->valid());
  owner.reset();

  for (auto& sample : depth) sample = 2000;
  failure.injected = false;
  failure.armed = true;
  auto refused = prepare(depth);
  failure.armed = false;
  CHECK(failure.injected);
  CHECK(!refused.ok());
  CHECK(refused.status().domain() == vkc::Status::Code::OutOfMemory);

  auto retry = prepare(depth);
  CHECK(retry.ok());
  CHECK(held->depth != retry->depth);
  auto before =
      vr_test::read_back<float>(device, allocator, *held->depth, depth.size());
  auto after =
      vr_test::read_back<float>(device, allocator, *retry->depth, depth.size());
  CHECK(before.ok() && after.ok());
  const float metres_per_unit = sensor::RgbdFrame{}.metres_per_unit;
  for (std::size_t i = 0; i < depth.size(); ++i) {
    CHECK((*before)[i] == 1000.0f * metres_per_unit);
    CHECK((*after)[i] == 2000.0f * metres_per_unit);
  }
  std::printf("  %s: OutOfMemory, the held frame intact, then prepared\n", who);
  return 0;
}

sensor::RgbdFrame frame_of(const std::vector<std::uint16_t>& depth) {
  sensor::RgbdFrame frame;
  frame.depth = depth.data();
  frame.depth_camera = {{8, 8}, {8.0, 8.0, 3.5, 3.5}, {}};
  frame.min_depth = 0.1f;
  frame.max_depth = 5.0f;
  return frame;
}

}  // namespace

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
  const int single =
      check_out_of_memory("prepare", *device, *allocator,
                          [&](const std::vector<std::uint16_t>& depth) {
                            return prep->prepare(frame_of(depth));
                          });
  if (single != 0) return single;

  auto pass = sensor::GpuFramePrep::create(*device, *allocator);
  CHECK(pass.ok());
  std::vector<sensor::GpuFramePrep> preps;
  preps.push_back(std::move(*pass));
  const int batch = check_out_of_memory(
      "prepare_batch", *device, *allocator,
      [&](const std::vector<std::uint16_t>& depth)
          -> vkc::Result<sensor::DeviceFrame> {
        const std::vector<std::optional<sensor::RgbdFrame>> frames{
            frame_of(depth)};
        VKC_ASSIGN(auto set,
                   sensor::GpuFramePrep::prepare_batch(preps, frames));
        return std::move(*set[0]);
      });
  if (batch != 0) return batch;
  std::puts("sensor_gpu_frame_prep_oom: OK");
  return 0;
}
