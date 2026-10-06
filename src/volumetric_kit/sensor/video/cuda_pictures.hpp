// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// NVIDIA's decoders' pictures, handed over on the device (VR_WITH_CUDA,
// Linux): NVDEC's copied device to device, nvJPEG's decoded in place, into
// Vulkan buffers CUDA has imported, so the kernels read them where they are
// and they never cross the bus. libcuda is loaded at run time, as FFmpeg
// loads it, so a build with CUDA starts without it.

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

// A macro's expansion as a string: a cuda.h name's versioned symbol, or
// nvjpeg.h's major version.
#define VR_CUDA_STRING(name) VR_CUDA_STRING_(name)
#define VR_CUDA_STRING_(name) #name

namespace volumetric_kit::recon::sensor::video {

// @p v rounded up to a multiple of @p to.
constexpr std::uint64_t round_up(std::uint64_t v, std::uint64_t to) noexcept {
  return (v + to - 1) / to * to;
}

// The driver entry points used here. Each name goes through cuda.h's macros,
// so it is the versioned symbol its prototype declares (cuMemFree is
// cuMemFree_v2).
#define VR_CUDA_FUNCTIONS(X)         \
  X(cuInit)                          \
  X(cuGetErrorName)                  \
  X(cuDeviceGetCount)                \
  X(cuDeviceGet)                     \
  X(cuDeviceGetUuid)                 \
  X(cuDevicePrimaryCtxRetain)        \
  X(cuDevicePrimaryCtxRelease)       \
  X(cuCtxPushCurrent)                \
  X(cuCtxPopCurrent)                 \
  X(cuStreamCreate)                  \
  X(cuStreamDestroy)                 \
  X(cuStreamSynchronize)             \
  X(cuEventCreate)                   \
  X(cuEventDestroy)                  \
  X(cuEventRecord)                   \
  X(cuEventSynchronize)              \
  X(cuImportExternalMemory)          \
  X(cuExternalMemoryGetMappedBuffer) \
  X(cuDestroyExternalMemory)         \
  X(cuMemFree)                       \
  X(cuMemcpy2DAsync)

struct CudaDriver {
#define VR_CUDA_DECLARE(name) decltype(&::name) name = nullptr;
  VR_CUDA_FUNCTIONS(VR_CUDA_DECLARE)
#undef VR_CUDA_DECLARE
};

// libcuda, loaded once and kept; null where it does not load or lacks an
// entry point.
const CudaDriver* cuda_driver();

// A Backend error naming @p who, @p what it was doing and CUDA's name for
// @p result, which is not CUDA_SUCCESS.
core::Status cuda_error(const char* who, CUresult result, const char* what);

// A CUDA context made current for a scope, and the one before put back.
// libcuda must have loaded.
class CudaContextScope {
 public:
  explicit CudaContextScope(CUcontext context)
      : result_(cuda_driver()->cuCtxPushCurrent(context)) {}
  ~CudaContextScope() {
    CUcontext popped = nullptr;
    if (ok()) cuda_driver()->cuCtxPopCurrent(&popped);
  }
  CudaContextScope(const CudaContextScope&) = delete;
  CudaContextScope& operator=(const CudaContextScope&) = delete;
  bool ok() const noexcept { return result_ == CUDA_SUCCESS; }
  CUresult result() const noexcept { return result_; }

 private:
  CUresult result_;
};

// The CUDA device that is @p device's GPU: Vulkan's deviceUUID matched
// against each CUDA device's. Unsupported where libcuda does not load or
// start, or no CUDA device is. @p who names the decoder in errors.
core::Result<int> cuda_ordinal_of(const core::Device& device, const char* who);

// A ring of exported Vulkan buffers on one device, each imported into CUDA
// once, in the decoder's CUDA context. A picture takes one that no earlier
// picture still holds, or a new one; a free one too small for it is let go.
class CudaPictures {
 public:
  // The buffers are made through @p allocator, which, like @p device, must
  // outlive this and every picture on it. @p who names the decoder in errors.
  static core::Result<std::unique_ptr<CudaPictures>> create(
      const core::Device& device, core::Allocator& allocator, CUcontext context,
      CUstream stream, const char* who);
  ~CudaPictures();
  CudaPictures(const CudaPictures&) = delete;
  CudaPictures& operator=(const CudaPictures&) = delete;

  // Copy an NV12 picture on the device, Y at `luma` and the chroma at
  // `chroma` with their pitches, into a buffer, rows packed; the copy has
  // finished when this returns. Fills @p out's size, layout and planes.
  core::Status copy(CUdeviceptr luma, std::size_t luma_pitch,
                    CUdeviceptr chroma, std::size_t chroma_pitch,
                    std::uint32_t width, std::uint32_t height,
                    DecodedPicture& out);

  // A buffer of at least @p bytes that no picture holds, and where CUDA sees
  // it, for a decoder that writes the picture itself.
  struct Target {
    std::shared_ptr<const core::Buffer> buffer;
    CUdeviceptr pointer = 0;
  };
  core::Result<Target> take(std::uint64_t bytes);

 private:
  struct Slot {
    std::shared_ptr<core::Buffer> buffer;
    CUexternalMemory memory = nullptr;
    CUdeviceptr pointer = 0;
    std::uint64_t bytes = 0;
  };
  CudaPictures(const core::Device& device, core::Allocator& allocator,
               CUcontext context, CUstream stream, const char* who)
      : device_(&device),
        allocator_(&allocator),
        context_(context),
        stream_(stream),
        who_(who) {}
  core::Result<Slot*> slot(std::uint64_t bytes);
  static void release(Slot& s);

  const core::Device* device_;
  core::Allocator* allocator_;
  CUcontext context_;
  CUstream stream_;
  const char* who_;
  std::vector<Slot> slots_;
};

}  // namespace volumetric_kit::recon::sensor::video
