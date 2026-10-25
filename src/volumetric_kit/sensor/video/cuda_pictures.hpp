// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// NVIDIA's decoders' pictures, handed over on the device (VR_WITH_CUDA,
// Linux): NVDEC's copied device to device, nvJPEG's decoded in place, into
// Vulkan buffers CUDA has imported (sensor/video/cuda_interop.hpp), so the
// kernels read them where they are and they never cross the bus.

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cuda_driver.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/video/cuda_interop.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace volumetric_kit::recon::sensor::video {

// @p v rounded up to a multiple of @p to.
constexpr std::uint64_t round_up(std::uint64_t v, std::uint64_t to) noexcept {
  return (v + to - 1) / to * to;
}

// @p buffer, which CUDA wrote, as the buffer of @p out's planes: a reader
// takes it over from outside Vulkan.
inline void place_cuda_buffer(std::shared_ptr<const core::Buffer> buffer,
                              YuvImage& out) noexcept {
  out.device = std::move(buffer);
  out.queue_family = kQueueFamilyExternal;
}

// A decoder's CUDA: the primary context of its device's GPU, and a stream of
// its own on it. Released together, the stream first.
struct CudaSession {
  // @p who names the decoder in errors.
  static core::Result<std::unique_ptr<CudaSession>> open(
      const core::Device& device, const char* who);
  CudaSession() = default;
  CudaSession(const CudaSession&) = delete;
  CudaSession& operator=(const CudaSession&) = delete;

  std::unique_ptr<CudaContext> context;
  std::unique_ptr<CudaStream> stream;
};

// A ring of exported Vulkan buffers on one device, each mapped into CUDA
// once, in the decoder's CUDA context. A picture takes one that no earlier
// picture still holds, or a new one; a free one too small for it is let go.
class CudaPictures {
 public:
  // The buffers are made through @p allocator, which, like @p device and
  // @p context, must outlive this and every picture on it. @p who names the
  // decoder in errors.
  static core::Result<std::unique_ptr<CudaPictures>> create(
      const core::Device& device, core::Allocator& allocator,
      const CudaContext& context, CUstream stream, const char* who);
  CudaPictures(const CudaPictures&) = delete;
  CudaPictures& operator=(const CudaPictures&) = delete;

  // Copy an NV12 picture on the device, Y at `luma` and the chroma at
  // `chroma` with their pitches, into a buffer, rows packed; the copy has
  // finished when this returns. Fills @p out's size, layout and planes.
  core::Status copy(CUdeviceptr luma, std::size_t luma_pitch,
                    CUdeviceptr chroma, std::size_t chroma_pitch,
                    std::uint32_t width, std::uint32_t height, YuvImage& out);

  // A buffer of at least @p bytes that no picture holds, and where CUDA sees
  // it, for a decoder that writes the picture itself.
  struct Target {
    std::shared_ptr<const core::Buffer> buffer;
    CUdeviceptr pointer = 0;
  };
  core::Result<Target> take(std::uint64_t bytes);

 private:
  // One exported buffer and CUDA's view of it, which goes first. A picture
  // can keep the buffer alive after the view is gone.
  struct Slot {
    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    std::shared_ptr<core::Buffer> buffer;
    std::unique_ptr<CudaBufferMapping> mapping;
    std::uint64_t bytes = 0;
  };

  CudaPictures(const core::Device& device, core::Allocator& allocator,
               const CudaContext& context, CUstream stream, const char* who)
      : device_(&device),
        allocator_(&allocator),
        context_(&context),
        stream_(stream),
        who_(who) {}
  core::Result<Slot*> slot(std::uint64_t bytes);

  const core::Device* device_;
  core::Allocator* allocator_;
  const CudaContext* context_;
  CUstream stream_;
  const char* who_;
  std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace volumetric_kit::recon::sensor::video
