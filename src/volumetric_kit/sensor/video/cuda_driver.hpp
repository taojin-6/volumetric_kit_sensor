// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// libcuda's driver API as this target calls it (VR_WITH_CUDA, Linux): loaded
// at run time, as FFmpeg loads it, so a build with CUDA starts without it.
// Internal.

#include <cuda.h>

#include "volumetric_kit/core/base/result.hpp"

// A macro's expansion as a string: a cuda.h name's versioned symbol, or
// nvjpeg.h's major version.
#define VR_CUDA_STRING(name) VR_CUDA_STRING_(name)
#define VR_CUDA_STRING_(name) #name

namespace volumetric_kit::recon::sensor::video {

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

// A Cuda Backend error naming @p who, @p what it was doing and CUDA's name for
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

}  // namespace volumetric_kit::recon::sensor::video
