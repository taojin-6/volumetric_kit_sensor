// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A Vulkan timeline that CUDA signals on one stream (VS_WITH_CUDA, Linux):
// how NVIDIA's decoders hand a picture over before its copy has run. Each
// picture takes the next value, which CUDA sets once the work queued on the
// stream before it is done, and carries it (YuvImage::ready); a reader's batch
// waits for it on the device, so neither host waits. Internal until a second
// crossing needs it.

#include <cuda.h>

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::video {

class CudaTimeline {
 public:
  // A timeline on @p device, exported, imported into @p context and signalled
  // on @p stream, both of which must outlive this. Unsupported where the
  // device does not export semaphores or cannot export a timeline one; a
  // Backend status if CUDA cannot import it. @p who names the decoder in
  // errors.
  static core::Result<std::unique_ptr<CudaTimeline>> create(
      const core::Device& device, const CudaContext& context, CUstream stream,
      const char* who);

  // settle(), then CUDA's import goes; the semaphore lives on with the
  // pictures that hold it.
  ~CudaTimeline();
  CudaTimeline(const CudaTimeline&) = delete;
  CudaTimeline& operator=(const CudaTimeline&) = delete;

  // Queue the next value's signal on the stream, after everything queued on
  // it so far, and give the value to @p out (YuvImage::ready and
  // ready_value). On failure @p out is left as it was and nothing is queued.
  core::Status mark(YuvImage& out);

  // Whether the counter has reached @p value; a failed read is not reached.
  bool reached(std::uint64_t value) const;

  // Wait on the host until the counter reaches @p value, or the stream fails
  // (its error), after which it sets no more values: settle() then.
  core::Status wait(std::uint64_t value) const;

  // After a CUDA failure, and before the import goes: wait for the stream,
  // which returns once its queued work has run or failed, so CUDA sets
  // nothing more; then, if the counter is below the last value handed out,
  // set that value from the host, so no batch waits for good on a signal a
  // failed stream dropped (core's ExportedTimeline orders it so). Where the
  // context cannot be made current the stream is not waited for, and the
  // host sets nothing.
  void settle() noexcept;

 private:
  CudaTimeline(CUcontext context, CUstream stream, const char* who) noexcept
      : context_(context), stream_(stream), who_(who) {}

  std::shared_ptr<core::TimelineSemaphore> semaphore_;
  CUexternalSemaphore imported_ = nullptr;
  CUcontext context_;
  CUstream stream_;
  const char* who_;
  std::uint64_t last_ = 0;  // the highest value handed out
};

}  // namespace volumetric_kit::sensor::video
