// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "depth_outliers.hpp"

#include <new>

#include "remove_depth_outliers_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"

namespace volumetric_kit::sensor::detail {
namespace {

// The kernel's local_size_x.
constexpr std::uint32_t kLocalSize = 256;

}  // namespace

core::Result<std::unique_ptr<DepthOutlierPass>> DepthOutlierPass::create(
    const core::Device& device) try {
  std::unique_ptr<DepthOutlierPass> pass(new DepthOutlierPass());
  // Its input, then its output, and the camera and rule as push constants.
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.size = sizeof(OutlierParams);
  core::KernelSetBuilder kb(device);
  VKC_TRY(kb.add(pass->kernel_, "remove_depth_outliers",
                 vs_remove_depth_outliers_comp_spv,
                 vs_remove_depth_outliers_comp_spv_size, 2, &push));
  VKC_ASSIGN(pass->pool_, kb.build());
  pass->max_workgroup_count_x_ =
      device.caps().limits().maxComputeWorkGroupCount[0];
  return pass;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory(
      "DepthOutlierPass::create: out of host memory");
}

core::Status DepthOutlierPass::record(core::CommandBatch& batch, VkBuffer in,
                                      VkBuffer out, const OutlierParams& params,
                                      core::GpuStageScope* stage) const {
  const std::uint64_t pixels = std::uint64_t{params.width} * params.height;
  if (pixels > std::uint64_t{max_workgroup_count_x_} * kLocalSize) {
    return core::Status::invalid_argument(
        "DepthOutlierPass: the depth image is past a single dispatch");
  }
  const VkDeviceSize bytes = pixels * sizeof(float);
  kernel_.set.write_storage_buffer(0, in, 0, bytes);
  kernel_.set.write_storage_buffer(1, out, 0, bytes);
  return batch.dispatch(
      kernel_, &params, sizeof(params),
      core::group_count(static_cast<std::uint32_t>(pixels), kLocalSize),
      max_workgroup_count_x_, stage);
}

}  // namespace volumetric_kit::sensor::detail
