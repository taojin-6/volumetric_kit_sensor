// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "cuda_pictures.hpp"

#include <dlfcn.h>

#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"

namespace volumetric_kit::recon::sensor::video {

const CudaDriver* cuda_driver() {
  static const std::optional<CudaDriver> loaded =
      []() -> std::optional<CudaDriver> {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return std::nullopt;
    CudaDriver d;
    bool ok = true;
#define VR_CUDA_LOAD(name)                                                  \
  d.name =                                                                  \
      reinterpret_cast<decltype(d.name)>(dlsym(lib, VR_CUDA_STRING(name))); \
  ok = ok && d.name != nullptr;
    VR_CUDA_FUNCTIONS(VR_CUDA_LOAD)
#undef VR_CUDA_LOAD
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
      static_cast<std::int64_t>(result),
      std::string(who) + ": " + what + ": " +
          (name != nullptr ? name : "CUDA error"));
}

core::Result<int> cuda_ordinal_of(const core::Device& device, const char* who) {
  VkPhysicalDeviceIDProperties id{};
  id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 props{};
  props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props.pNext = &id;
  vkGetPhysicalDeviceProperties2(device.physical_device(), &props);

  const CudaDriver* cu = cuda_driver();
  if (cu == nullptr) {
    return core::Status::unsupported(std::string(who) +
                                     ": libcuda does not load");
  }
  CUresult r = cu->cuInit(0);
  if (r != CUDA_SUCCESS) {
    return core::Status::unsupported(
        cuda_error(who, r, "starting CUDA").message());
  }
  int count = 0;
  r = cu->cuDeviceGetCount(&count);
  if (r != CUDA_SUCCESS) return cuda_error(who, r, "counting CUDA devices");
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
      std::string(who) + ": no CUDA device is the Vulkan device's GPU");
}

core::Result<std::unique_ptr<CudaPictures>> CudaPictures::create(
    const core::Device& device, core::Allocator& allocator, CUcontext context,
    CUstream stream, const char* who) {
  if (!device.exports_memory()) {
    return core::Status::unsupported(std::string(who) +
                                     ": the device does not export memory "
                                     "(VK_KHR_external_memory_fd)");
  }
  if (cuda_driver() == nullptr) {
    return core::Status::unsupported(std::string(who) +
                                     ": libcuda does not load");
  }
  return std::unique_ptr<CudaPictures>(
      new CudaPictures(device, allocator, context, stream, who));
}

CudaPictures::~CudaPictures() {
  const CudaContextScope scope(context_);
  for (Slot& s : slots_) release(s);
}

// CUDA's view goes; the buffer stays with any picture still holding it.
void CudaPictures::release(Slot& s) {
  cuda_driver()->cuMemFree(s.pointer);
  cuda_driver()->cuDestroyExternalMemory(s.memory);
}

core::Result<CudaPictures::Slot*> CudaPictures::slot(std::uint64_t bytes) {
  // Reused only once no picture holds it. A free one left over is too small,
  // so it is let go rather than kept past a change of picture size. Vulkan
  // does not release a reused buffer back to CUDA: CUDA overwrites the
  // picture, a family may take a buffer without a transfer when its contents
  // need not survive, and its last reader finished before letting it go.
  for (Slot& s : slots_) {
    if (s.buffer.use_count() == 1 && s.bytes >= bytes) return &s;
  }
  for (auto it = slots_.begin(); it != slots_.end();) {
    if (it->buffer.use_count() == 1) {
      release(*it);
      it = slots_.erase(it);
    } else {
      ++it;
    }
  }
  VKC_ASSIGN(core::ExportedBuffer exported,
             core::create_exported_buffer(*device_, *allocator_, bytes));
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           core::debug_object_handle(exported.buffer.handle()),
                           "sensor.decoded_picture");
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC handle{};
  handle.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  handle.handle.fd = exported.fd.get();
  handle.size = exported.memory_size;
  handle.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
  Slot s;
  s.bytes = bytes;
  const CudaDriver* cu = cuda_driver();
  CUresult r = cu->cuImportExternalMemory(&s.memory, &handle);
  if (r != CUDA_SUCCESS) {
    // An import that fails leaves the descriptor ours: the UniqueFd closes it.
    return cuda_error(who_, r, "importing a Vulkan buffer");
  }
  // One that succeeds hands it to CUDA, which closes it.
  static_cast<void>(exported.fd.release());
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapped{};
  mapped.offset = 0;
  mapped.size = bytes;
  r = cu->cuExternalMemoryGetMappedBuffer(&s.pointer, s.memory, &mapped);
  if (r != CUDA_SUCCESS) {
    cu->cuDestroyExternalMemory(s.memory);
    return cuda_error(who_, r, "mapping a Vulkan buffer");
  }
  s.buffer = std::make_shared<core::Buffer>(std::move(exported.buffer));
  slots_.push_back(std::move(s));
  return &slots_.back();
}

core::Result<CudaPictures::Target> CudaPictures::take(std::uint64_t bytes) {
  const CudaContextScope scope(context_);
  if (!scope.ok()) {
    return cuda_error(who_, scope.result(), "making the CUDA context current");
  }
  VKC_ASSIGN(Slot * s, slot(bytes));
  return Target{s->buffer, s->pointer};
}

core::Status CudaPictures::copy(CUdeviceptr luma, std::size_t luma_pitch,
                                CUdeviceptr chroma, std::size_t chroma_pitch,
                                std::uint32_t width, std::uint32_t height,
                                DecodedPicture& out) {
  const CudaContextScope scope(context_);
  if (!scope.ok()) {
    return cuda_error(who_, scope.result(), "making the CUDA context current");
  }
  // Rows packed: Y, then the interleaved chroma at half height, each pair of
  // samples covering two luma columns, so a chroma row is the width rounded
  // up to even.
  const std::uint64_t chroma_row = (std::uint64_t{width} + 1) / 2 * 2;
  const std::uint64_t chroma_rows = (std::uint64_t{height} + 1) / 2;
  const std::uint64_t chroma_at = round_up(std::uint64_t{width} * height, 256);
  const std::uint64_t bytes = round_up(chroma_at + chroma_row * chroma_rows, 4);
  VKC_ASSIGN(Slot * s, slot(bytes));

  CUDA_MEMCPY2D plane{};
  plane.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  plane.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  plane.srcDevice = luma;
  plane.srcPitch = luma_pitch;
  plane.dstDevice = s->pointer;
  plane.dstPitch = width;
  plane.WidthInBytes = width;
  plane.Height = height;
  const CudaDriver* cu = cuda_driver();
  CUresult r = cu->cuMemcpy2DAsync(&plane, stream_);
  if (r == CUDA_SUCCESS) {
    plane.srcDevice = chroma;
    plane.srcPitch = chroma_pitch;
    plane.dstDevice = s->pointer + chroma_at;
    plane.dstPitch = chroma_row;
    plane.WidthInBytes = chroma_row;
    plane.Height = chroma_rows;
    r = cu->cuMemcpy2DAsync(&plane, stream_);
  }
  if (r == CUDA_SUCCESS) r = cu->cuStreamSynchronize(stream_);
  if (r != CUDA_SUCCESS) return cuda_error(who_, r, "copying a picture");

  out.width = width;
  out.height = height;
  out.layout = VideoPixelLayout::Nv12;
  out.stride[0] = width;
  out.stride[1] = chroma_row;
  out.device = s->buffer;
  out.offset[0] = 0;
  out.offset[1] = chroma_at;
  return {};
}

}  // namespace volumetric_kit::recon::sensor::video
