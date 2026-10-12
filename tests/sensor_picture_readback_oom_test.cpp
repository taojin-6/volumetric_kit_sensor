// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host allocation failures in creation and reading return OutOfMemory
// rather than throwing, and retries succeed. Skips without a device.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "gpu_test.hpp"
#include "test_allocation_failure.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/picture_readback.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::PictureReadback;

namespace {

int test_create(vr_test::GpuContext& gpu) {
  // Capture a timer allocation's size and call site, so a validation-layer
  // allocation of the same size cannot consume the injected failure.
  auto& failure = vr_test::allocation_failure;
  failure.measure = true;
  auto timer = vkc::GpuTimer::create(gpu.device);
  failure.measure = false;
  CHECK(timer.ok() && failure.bytes != 0 && failure.measured_site != nullptr);
  failure.site = failure.measured_site;
  failure.injected = false;
  failure.armed = true;
  try {
    auto refused = PictureReadback::create(gpu.device, gpu.allocator);
    failure.armed = false;
    CHECK(failure.injected);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::OutOfMemory);
  } catch (const std::bad_alloc&) {
    failure.armed = false;
    std::fprintf(stderr, "FAIL: bad_alloc escaped PictureReadback::create\n");
    return 1;
  }
  failure = {};
  CHECK(PictureReadback::create(gpu.device, gpu.allocator).ok());
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  CHECK(test_create(gpu) == 0);
  auto readback = PictureReadback::create(gpu.device, gpu.allocator);
  CHECK(readback.ok());
  // An 8x8 NV12 picture in a buffer, its luma 0..63.
  std::vector<std::uint8_t> bytes(8 * 8 + 8 * 4);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(i);
  }
  auto buffer = vr_test::upload_device_buffer(gpu.device, gpu.allocator,
                                              bytes.data(), bytes.size());
  CHECK(buffer.ok());
  sensor::YuvImage picture;
  picture.layout = sensor::YuvLayout::Nv12;
  picture.device = std::make_shared<const vkc::Buffer>(std::move(*buffer));
  picture.width = picture.height = 8;
  picture.offset[1] = 64;
  picture.stride[0] = picture.stride[1] = 8;
  std::vector<std::optional<sensor::RgbdFrame>> frames(1);
  frames[0].emplace().color = picture;
  const auto read = [&] {
    return (*readback)->read(frames, PictureReadback::Planes::Luma);
  };
  CHECK(read().ok());

  // The views the read hands out are an allocation of their own size.
  auto& failure = vr_test::allocation_failure;
  failure.measure = true;
  std::vector<std::optional<sensor::PictureView>> views(frames.size());
  failure.measure = false;
  CHECK(failure.bytes != 0 && views.size() == 1);

  failure.injected = false;
  failure.armed = true;
  try {
    auto refused = read();
    failure.armed = false;
    CHECK(failure.injected);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::OutOfMemory);
  } catch (const std::bad_alloc&) {
    failure.armed = false;
    std::fprintf(stderr, "FAIL: bad_alloc escaped PictureReadback::read\n");
    return 1;
  }

  auto retry = read();
  CHECK(retry.ok() && (*retry)[0]);
  CHECK(std::memcmp((*retry)[0]->luma, bytes.data(), 64) == 0);
  std::puts("sensor_picture_readback_oom: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
