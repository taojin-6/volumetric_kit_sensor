// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "cuda_driver.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <optional>
#include <string>

namespace volumetric_kit::sensor::video {

const CudaDriver* cuda_driver() {
  static const std::optional<CudaDriver> loaded =
      []() -> std::optional<CudaDriver> {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return std::nullopt;
    CudaDriver d;
    bool ok = true;
#define VS_CUDA_LOAD(name)                                                  \
  d.name =                                                                  \
      reinterpret_cast<decltype(d.name)>(dlsym(lib, VS_CUDA_STRING(name))); \
  ok = ok && d.name != nullptr;
    VS_CUDA_FUNCTIONS(VS_CUDA_LOAD)
#undef VS_CUDA_LOAD
    if (!ok) {
      dlclose(lib);
      return std::nullopt;
    }
    return d;
  }();
  return loaded ? &*loaded : nullptr;
}

core::Status cuda_error(const char* who, CUresult result, const char* what) {
  const char* name = nullptr;
  cuda_driver()->cuGetErrorName(result, &name);
  return core::Status::backend_error(
      core::Status::Backend::Cuda, static_cast<std::int64_t>(result),
      std::string(who) + ": " + what + ": " +
          (name != nullptr ? name : "CUDA error"));
}

}  // namespace volumetric_kit::sensor::video
