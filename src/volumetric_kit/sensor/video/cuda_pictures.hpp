// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// NVIDIA's decoders' pictures, handed over on the device (VR_WITH_CUDA,
// Linux), in Vulkan memory CUDA has imported (sensor/video/cuda_interop.hpp),
// so the kernels read them where they are and they never cross the bus:
// NVDEC's copied device to device into two plane images, which shaders
// sample in place, and nvJPEG's decoded in place into a buffer, the only
// memory it writes.

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cuda_driver.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
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

// nvJPEG's: a ring of exported Vulkan buffers on one device, each mapped into
// CUDA once, in the decoder's CUDA context. A picture takes one that no
// earlier picture still holds, or a new one; a free one too small for it is
// let go.
class CudaPictures {
 public:
  // The buffers are made through @p allocator, which, like @p device and
  // @p context, must outlive this and every picture on it. @p who names the
  // decoder in errors.
  static core::Result<std::unique_ptr<CudaPictures>> create(
      const core::Device& device, core::Allocator& allocator,
      const CudaContext& context, const char* who);
  CudaPictures(const CudaPictures&) = delete;
  CudaPictures& operator=(const CudaPictures&) = delete;

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
               const CudaContext& context, const char* who)
      : device_(&device),
        allocator_(&allocator),
        context_(&context),
        who_(who) {}

  const core::Device* device_;
  core::Allocator* allocator_;
  const CudaContext* context_;
  const char* who_;
  std::vector<std::unique_ptr<Slot>> slots_;
};

// NVDEC's: a ring of pictures on one device, each an exported R8 luma image
// and an R8G8 chroma image of one picture's size, mapped into CUDA once, in
// the decoder's CUDA context, and handed over to CUDA once, in GENERAL. A
// picture takes a set that no earlier picture still holds, or a new one; a
// free one of another size is let go.
class CudaPlanePictures {
 public:
  // The images are made through @p allocator, which, like @p device and
  // @p context, must outlive this and every picture on it, and are shared
  // with @p queue_families as PixelBufferImporter::create shares its images.
  // @p who names the decoder in errors. InvalidArgument for families
  // detail::sharing_families refuses; Unsupported where @p device does not
  // export memory.
  static core::Result<std::unique_ptr<CudaPlanePictures>> create(
      const core::Device& device, core::Allocator& allocator,
      const CudaContext& context, CUstream stream,
      const std::uint32_t* queue_families, std::uint32_t queue_family_count,
      const char* who);
  CudaPlanePictures(const CudaPlanePictures&) = delete;
  CudaPlanePictures& operator=(const CudaPlanePictures&) = delete;

  // Copy an NV12 picture on the device, Y at `luma` and the chroma at
  // `chroma` with their row strides in bytes, into a set's two images; the
  // copy has finished when this returns. Fills @p out's size, layout and
  // images, which a reader takes over from outside Vulkan
  // (kQueueFamilyExternal).
  core::Status copy(CUdeviceptr luma, std::size_t luma_stride,
                    CUdeviceptr chroma, std::size_t chroma_stride,
                    std::uint32_t width, std::uint32_t height, YuvImage& out);

 private:
  // A picture's two images. Pictures hold them through aliases of one owner,
  // so one count says whether any picture holds either.
  struct Planes {
    core::Image image[2];  // R8 luma, R8G8 chroma
  };
  // One set and CUDA's views of its images, which go first: a picture can
  // keep the images alive after the views are gone.
  struct Slot {
    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    std::shared_ptr<Planes> planes;
    std::unique_ptr<CudaImageMapping> mapping[2];
    std::uint32_t width = 0;
    std::uint32_t height = 0;
  };

  CudaPlanePictures(const core::Device& device, core::Allocator& allocator,
                    const CudaContext& context, CUstream stream,
                    std::vector<std::uint32_t> families, const char* who)
      : device_(&device),
        allocator_(&allocator),
        context_(&context),
        stream_(stream),
        families_(std::move(families)),
        who_(who) {}
  core::Result<Slot*> slot(std::uint32_t width, std::uint32_t height);
  // A new set of @p width x @p height, mapped into CUDA and handed over to it.
  core::Result<std::unique_ptr<Slot>> make_slot(std::uint32_t width,
                                                std::uint32_t height);

  const core::Device* device_;
  core::Allocator* allocator_;
  const CudaContext* context_;
  CUstream stream_;
  std::vector<std::uint32_t> families_;  // CONCURRENT across; empty: EXCLUSIVE
  const char* who_;
  std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace volumetric_kit::recon::sensor::video
