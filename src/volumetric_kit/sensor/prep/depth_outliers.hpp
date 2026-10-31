// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// GpuFramePrep's radius outlier pass (shaders/remove_depth_outliers.comp): a
// return kept only where enough others lie near it in 3D, recorded after the
// depth undistortion, from one buffer of metres into another. Internal to
// sensor_prep.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"

namespace volumetric_kit::sensor::detail {

// shaders/remove_depth_outliers.comp's Params, under scalar layout.
struct OutlierParams {
  float fx, fy, cx, cy;
  float inv_fx, inv_fy;  // 1 / fx and 1 / fy, each rounded once
  std::uint32_t width, height;
  float radius_per_m;  // k / fx, the radius one metre deep
  std::uint32_t min_points;
  std::uint32_t zero;  // always 0: the shader's opaque() xors with it
};
static_assert(sizeof(OutlierParams) == 44, "OutlierParams layout drift");

// The pass's parameters for the pinhole camera `k` of `size`: a radius of
// `footprints` pixels at each return's own depth, kept with at least
// `min_points` returns within it. Narrowed to float here, the reciprocals and
// k / fx from the doubles.
inline OutlierParams outlier_params(const camera::PinholeIntrinsics& k,
                                    const camera::ImageSize& size,
                                    double footprints,
                                    std::uint32_t min_points) noexcept {
  const auto f = [](double v) { return static_cast<float>(v); };
  return OutlierParams{f(k.fx),
                       f(k.fy),
                       f(k.cx),
                       f(k.cy),
                       f(1.0 / k.fx),
                       f(1.0 / k.fy),
                       size.width,
                       size.height,
                       f(footprints / k.fx),
                       min_points,
                       0u};
}

// The pass: its kernel and the set it rewrites each record.
class DepthOutlierPass {
 public:
  // Built on `device`, which must outlive it.
  static core::Result<std::unique_ptr<DepthOutlierPass>> create(
      const core::Device& device);

  ~DepthOutlierPass() = default;
  DepthOutlierPass(const DepthOutlierPass&) = delete;
  DepthOutlierPass& operator=(const DepthOutlierPass&) = delete;
  DepthOutlierPass(DepthOutlierPass&&) = delete;
  DepthOutlierPass& operator=(DepthOutlierPass&&) = delete;

  // `params.width * height` metres of `in` filtered into `out`, recorded
  // into `batch` under `stage`; `in` and `out` are distinct storage buffers
  // that hold them from byte 0. Refuses an image past one dispatch, with
  // nothing recorded.
  // @pre The batch of this pass's last record has resolved: it binds the
  //      pass's set, which this rewrites.
  core::Status record(core::CommandBatch& batch, VkBuffer in, VkBuffer out,
                      const OutlierParams& params,
                      core::GpuStageScope* stage = nullptr) const;

 private:
  DepthOutlierPass() = default;

  core::ComputeKernel kernel_;
  core::DescriptorPool pool_;
  std::uint32_t max_workgroup_count_x_ = 0;
};

}  // namespace volumetric_kit::sensor::detail
