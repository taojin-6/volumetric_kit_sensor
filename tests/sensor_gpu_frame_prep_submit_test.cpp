// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A submission that loses the device may still read external pictures. The
// process exits without teardown after that injected failure: the core keeps
// an unsignaled fence, and the prep deliberately abandons the pictures.
// Other submit failures must release pictures normally and allow a retry.

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gpu_test.hpp"
#include "test_check.hpp"
#include "test_image.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

namespace sensor = volumetric_kit::recon::sensor;
namespace vkc = volumetric_kit::core;

namespace {

thread_local VkResult submit_failure = VK_SUCCESS;
thread_local unsigned failures = 0;

}  // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkQueueSubmit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
              VkFence fence) {
  if (submit_failure != VK_SUCCESS) {
    ++failures;
    return std::exchange(submit_failure, VK_SUCCESS);
  }
  static const auto real =
      reinterpret_cast<PFN_vkQueueSubmit>(dlsym(RTLD_NEXT, "vkQueueSubmit"));
  if (real == nullptr) std::abort();
  return real(queue, count, submits, fence);
}

namespace {

// Buffer-backed NV12, image-backed NV12, and an absent camera in one set.
// Weak references observe whether dropping the set returns either picture
// to its producer, which could immediately reuse it for the next frame.
struct Set {
  std::vector<std::uint16_t> depth = std::vector<std::uint16_t>(16 * 16, 1000);
  std::vector<sensor::GpuFramePrep> preps;
  std::vector<std::optional<sensor::RgbdFrame>> frames{3};
  std::weak_ptr<const vkc::Buffer> buffer;
  std::weak_ptr<const vkc::Image> luma, chroma;

  bool pictures_released() const {
    return buffer.expired() && luma.expired() && chroma.expired();
  }
  bool pictures_retained() const {
    return !buffer.expired() && !luma.expired() && !chroma.expired();
  }

  int make(vr_test::GpuContext& gpu) {
    for (std::size_t i = 0; i < frames.size(); ++i) {
      auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
      CHECK(prep.ok());
      preps.push_back(std::move(*prep));
      if (i == 1) continue;
      sensor::RgbdFrame& frame = frames[i].emplace();
      frame.depth = depth.data();
      frame.min_depth = 0.1f;
      frame.max_depth = 4.0f;
      frame.depth_camera.size = {16, 16};
      frame.depth_camera.intrinsics = {16.0, 16.0, 8.0, 8.0};
      frame.color_camera = frame.depth_camera;
      frame.color.layout = sensor::YuvLayout::Nv12;
      frame.color.width = 16;
      frame.color.height = 16;
    }
    auto made_buffer = vkc::device_storage_buffer(gpu.allocator, 384);
    CHECK(made_buffer.ok());
    frames[0]->color.device =
        std::make_shared<vkc::Buffer>(std::move(*made_buffer));
    frames[0]->color.offset[1] = 256;
    frames[0]->color.stride[0] = frames[0]->color.stride[1] = 16;
    buffer = frames[0]->color.device;

    auto made_luma = test_image::make(
        gpu.device, gpu.allocator, VK_FORMAT_R8_UNORM, 16, 16,
        std::vector<std::uint8_t>(256, 128), VK_IMAGE_USAGE_SAMPLED_BIT);
    CHECK(made_luma.ok());
    auto made_chroma = test_image::make(
        gpu.device, gpu.allocator, VK_FORMAT_R8G8_UNORM, 8, 8,
        std::vector<std::uint8_t>(128, 128), VK_IMAGE_USAGE_SAMPLED_BIT);
    CHECK(made_chroma.ok());
    frames[2]->color.image[0] =
        std::make_shared<vkc::Image>(std::move(*made_luma));
    frames[2]->color.image[1] =
        std::make_shared<vkc::Image>(std::move(*made_chroma));
    luma = frames[2]->color.image[0];
    chroma = frames[2]->color.image[1];
    return 0;
  }
};

int refused_submit(vr_test::GpuContext& gpu, VkResult error) {
  Set set;
  CHECK(set.make(gpu) == 0);
  submit_failure = error;
  failures = 0;
  auto refused = sensor::GpuFramePrep::prepare_batch(set.preps, set.frames);
  submit_failure = VK_SUCCESS;
  CHECK(failures == 1);
  CHECK(!refused.ok());
  CHECK(vkc::vk_result(refused.status()) == error);

  if (error != VK_ERROR_DEVICE_LOST) {
    // A definitely unsubmitted set leaves the passes reusable.
    CHECK(sensor::GpuFramePrep::prepare_batch(set.preps, set.frames).ok());
  }
  set.frames.clear();
  if (error == VK_ERROR_DEVICE_LOST) {
    CHECK(set.pictures_retained());
  } else {
    CHECK(set.pictures_released());
  }
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  CHECK(refused_submit(gpu, VK_ERROR_OUT_OF_HOST_MEMORY) == 0);
  CHECK(refused_submit(gpu, VK_ERROR_OUT_OF_DEVICE_MEMORY) == 0);
  const int result = refused_submit(gpu, VK_ERROR_DEVICE_LOST);
  // Include validation errors already reported, then avoid teardown of the
  // deliberately pending submission and abandoned pictures, even on failure.
  const int exit_code = gpu.log.exit_code(result);
  if (exit_code == 0) std::puts("sensor_gpu_frame_prep_submit: OK");
  std::fflush(stdout);
  std::fflush(stderr);
  std::_Exit(exit_code);
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
