// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/cuda_interop.hpp"

#include <cstring>
#include <string>

#include "cuda_driver.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::recon::sensor {

using video::cuda_driver;
using video::cuda_error;
using video::CudaContextScope;
using video::CudaDriver;

core::Result<int> cuda_ordinal_of(const core::Device& device) {
  constexpr const char* kWho = "cuda_ordinal_of";
  VkPhysicalDeviceIDProperties id{};
  id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 props{};
  props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props.pNext = &id;
  vkGetPhysicalDeviceProperties2(device.physical_device(), &props);

  const CudaDriver* cu = cuda_driver();
  if (cu == nullptr) {
    return core::Status::unsupported(std::string(kWho) +
                                     ": libcuda does not load");
  }
  CUresult r = cu->cuInit(0);
  if (r != CUDA_SUCCESS) {
    return core::Status::unsupported(
        cuda_error(kWho, r, "starting CUDA").message());
  }
  int count = 0;
  r = cu->cuDeviceGetCount(&count);
  if (r != CUDA_SUCCESS) return cuda_error(kWho, r, "counting CUDA devices");
  for (int i = 0; i < count; ++i) {
    CUdevice ordinal = 0;
    CUuuid uuid{};
    if (cu->cuDeviceGet(&ordinal, i) != CUDA_SUCCESS ||
        cu->cuDeviceGetUuid(&uuid, ordinal) != CUDA_SUCCESS) {
      continue;
    }
    if (std::memcmp(uuid.bytes, id.deviceUUID, VK_UUID_SIZE) == 0) return i;
  }
  return core::Status::unsupported(
      std::string(kWho) + ": no CUDA device is the Vulkan device's GPU");
}

core::Result<std::unique_ptr<CudaContext>> CudaContext::open(
    const core::Device& device) {
  constexpr const char* kWho = "CudaContext";
  VKC_ASSIGN(const int ordinal, cuda_ordinal_of(device));
  // Owned before CUDA retains anything, so no host allocation can strand it.
  std::unique_ptr<CudaContext> c(new CudaContext());
  const CudaDriver* cu = cuda_driver();
  CUresult r = cu->cuDeviceGet(&c->device_, ordinal);
  if (r == CUDA_SUCCESS) {
    r = cu->cuDevicePrimaryCtxRetain(&c->context_, c->device_);
  }
  if (r != CUDA_SUCCESS) {
    c->context_ = nullptr;
    return cuda_error(kWho, r, "retaining the GPU's primary CUDA context");
  }
  return c;
}

CudaContext::~CudaContext() {
  if (context_ != nullptr) cuda_driver()->cuDevicePrimaryCtxRelease(device_);
}

core::Result<std::unique_ptr<CudaStream>> CudaStream::open(
    const CudaContext& context) {
  constexpr const char* kWho = "CudaStream";
  std::unique_ptr<CudaStream> s(new CudaStream(context.handle()));
  const CudaContextScope scope(context.handle());
  if (!scope.ok()) {
    return cuda_error(kWho, scope.result(), "making the CUDA context current");
  }
  const CUresult r =
      cuda_driver()->cuStreamCreate(&s->stream_, CU_STREAM_NON_BLOCKING);
  if (r != CUDA_SUCCESS) {
    s->stream_ = nullptr;
    return cuda_error(kWho, r, "creating a CUDA stream");
  }
  return s;
}

CudaStream::~CudaStream() {
  if (stream_ == nullptr) return;
  const CudaContextScope scope(context_);
  cuda_driver()->cuStreamDestroy(stream_);
}

core::Result<std::unique_ptr<CudaBufferMapping>> CudaBufferMapping::map(
    const CudaContext& context, core::ExportedBuffer& exported) {
  constexpr const char* kWho = "CudaBufferMapping";
  if (!exported.fd.valid()) {
    return core::Status::invalid_argument(
        std::string(kWho) + ": the buffer's descriptor was taken");
  }
  if (!exported.buffer.valid()) {
    return core::Status::invalid_argument(std::string(kWho) + ": no buffer");
  }
  // Owned before the import, so a failed mapping or host allocation releases
  // CUDA's view in its context.
  std::unique_ptr<CudaBufferMapping> m(new CudaBufferMapping(context.handle()));
  const CudaContextScope scope(context.handle());
  if (!scope.ok()) {
    return cuda_error(kWho, scope.result(), "making the CUDA context current");
  }
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
  handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  handle.handle.fd = exported.fd.get();
  handle.size = exported.memory_size;
  handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
  const CudaDriver* cu = cuda_driver();
  CUresult r = cu->cuImportExternalMemory(&m->memory_, &handle);
  if (r != CUDA_SUCCESS) {
    // An import that fails leaves the descriptor the caller's.
    m->memory_ = nullptr;
    return cuda_error(kWho, r, "importing a Vulkan buffer");
  }
  // One that succeeds hands it to CUDA, which closes it.
  static_cast<void>(exported.fd.release());
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapped{};
  mapped.offset = 0;
  mapped.size = exported.buffer.size();
  r = cu->cuExternalMemoryGetMappedBuffer(&m->pointer_, m->memory_, &mapped);
  if (r != CUDA_SUCCESS) {
    m->pointer_ = 0;
    return cuda_error(kWho, r, "mapping a Vulkan buffer");
  }
  return m;
}

CudaBufferMapping::~CudaBufferMapping() {
  if (memory_ == nullptr) return;
  const CudaDriver* cu = cuda_driver();
  const CudaContextScope scope(context_);
  if (pointer_ != 0) cu->cuMemFree(pointer_);
  cu->cuDestroyExternalMemory(memory_);
}

}  // namespace volumetric_kit::recon::sensor
