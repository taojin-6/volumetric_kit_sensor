// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The public CUDA interop on this machine's GPU: cuda_ordinal_of names the
// CUDA device whose UUID is the Vulkan device's, and CudaContext is that
// device's primary context; a buffer, and R8 and R8G8 images, that CUDA
// writes on a stream read back through Vulkan with the bytes written, round
// after round; and an exported buffer or image whose descriptor was already
// taken, or an image of a format CUDA is not given, is refused. The checks
// call libcuda through a loader of their own, so they do not lean on recon's.
//
// Where CUDA does not reach the device's GPU the test skips;
// VR_TEST_HEVC_BACKEND, which CI sets on the NVIDIA leg, makes it fail
// instead, as for the decoders.

#include <cuda.h>
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "gpu_test.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/sensor/video/cuda_interop.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;

// A macro's expansion as a string: the versioned symbol a cuda.h name stands
// for (cuMemcpyHtoDAsync is cuMemcpyHtoDAsync_v2).
#define VR_TEST_STRING(name) VR_TEST_STRING_(name)
#define VR_TEST_STRING_(name) #name

namespace {

using Gpu = vr_test::GpuContext;

// The libcuda calls the checks make themselves.
#define VR_TEST_CUDA_FUNCTIONS(X) \
  X(cuDeviceGet)                  \
  X(cuDeviceGetUuid)              \
  X(cuCtxPushCurrent)             \
  X(cuCtxPopCurrent)              \
  X(cuCtxGetDevice)               \
  X(cuMemcpyHtoDAsync)            \
  X(cuMemcpy2DAsync)              \
  X(cuStreamSynchronize)

struct Libcuda {
#define VR_TEST_DECLARE(name) decltype(&::name) name = nullptr;
  VR_TEST_CUDA_FUNCTIONS(VR_TEST_DECLARE)
#undef VR_TEST_DECLARE
};

// libcuda, loaded once and kept; null where it does not load or lacks a
// call.
const Libcuda* libcuda() {
  static const std::unique_ptr<Libcuda> loaded = [] {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return std::unique_ptr<Libcuda>();
    auto l = std::make_unique<Libcuda>();
    bool ok = true;
#define VR_TEST_LOAD(name)                                                   \
  l->name =                                                                  \
      reinterpret_cast<decltype(l->name)>(dlsym(lib, VR_TEST_STRING(name))); \
  ok = ok && l->name != nullptr;
    VR_TEST_CUDA_FUNCTIONS(VR_TEST_LOAD)
#undef VR_TEST_LOAD
    return ok ? std::move(l) : std::unique_ptr<Libcuda>();
  }();
  return loaded.get();
}

// The CUDA device at @p ordinal names the Vulkan device's GPU by its UUID,
// and the context CudaContext opens is on it, the one primary context every
// opener shares.
int test_ordinal_and_context(const Gpu& gpu, const Libcuda& cu, int ordinal) {
  VkPhysicalDeviceIDProperties id{};
  id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 props{};
  props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props.pNext = &id;
  vkGetPhysicalDeviceProperties2(gpu.device.physical_device(), &props);
  CUdevice device = 0;
  CUuuid uuid{};
  CHECK(cu.cuDeviceGet(&device, ordinal) == CUDA_SUCCESS);
  CHECK(cu.cuDeviceGetUuid(&uuid, device) == CUDA_SUCCESS);
  CHECK(std::memcmp(uuid.bytes, id.deviceUUID, VK_UUID_SIZE) == 0);

  auto a = sensor::CudaContext::open(gpu.device);
  auto b = sensor::CudaContext::open(gpu.device);
  CHECK(a.ok() && b.ok());
  CHECK(a.value()->handle() != nullptr);
  CHECK(a.value()->handle() == b.value()->handle());
  CHECK(cu.cuCtxPushCurrent(a.value()->handle()) == CUDA_SUCCESS);
  CUdevice current = -1;
  const CUresult r = cu.cuCtxGetDevice(&current);
  CUcontext popped = nullptr;
  CHECK(cu.cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
  CHECK(r == CUDA_SUCCESS && current == device);
  CHECK(popped == a.value()->handle());
  return 0;
}

// Bytes that change along the buffer and from round to round, so a stale
// round or a shifted pointer reads back wrong.
std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint32_t round) {
  std::vector<std::uint8_t> out(bytes);
  std::uint32_t x = 0x9e3779b9u * (round + 1);
  for (std::uint8_t& b : out) {
    x = x * 1664525u + 1013904223u;
    b = static_cast<std::uint8_t>(x >> 24);
  }
  return out;
}

// A buffer CUDA writes on a stream of its own, synchronized, reads back
// through Vulkan, taken over from VK_QUEUE_FAMILY_EXTERNAL, with the bytes
// written; handed back after each read, it takes CUDA's next write.
int test_cuda_writes_vulkan_reads(const Gpu& gpu, const Libcuda& cu) {
  constexpr std::size_t kBytes = std::size_t{1} << 16;
  auto exported =
      vkc::create_exported_buffer(gpu.device, gpu.allocator, kBytes);
  CHECK(exported.ok());
  const vkc::Buffer& buffer = exported.value().buffer;
  auto context = sensor::CudaContext::open(gpu.device);
  CHECK(context.ok());
  auto stream = sensor::CudaStream::open(*context.value());
  CHECK(stream.ok() && stream.value()->handle() != nullptr);
  auto mapping = sensor::CudaBufferMapping::map(*context.value(), *exported);
  CHECK(mapping.ok() && mapping.value()->pointer() != 0);
  CHECK(!exported.value().fd.valid());  // CUDA's now
  CHECK(buffer.valid());                // the caller's still

  for (std::uint32_t round = 0; round < 3; ++round) {
    const std::vector<std::uint8_t> written = pattern(kBytes, round);
    CHECK(cu.cuCtxPushCurrent(context.value()->handle()) == CUDA_SUCCESS);
    CUresult r =
        cu.cuMemcpyHtoDAsync(mapping.value()->pointer(), written.data(), kBytes,
                             stream.value()->handle());
    if (r == CUDA_SUCCESS) r = cu.cuStreamSynchronize(stream.value()->handle());
    CUcontext popped = nullptr;
    CHECK(cu.cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
    CHECK(r == CUDA_SUCCESS);

    std::vector<std::uint8_t> read(kBytes);
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    CHECK(batch.acquire(buffer, VK_QUEUE_FAMILY_EXTERNAL).ok());
    CHECK(batch.readback(buffer, 0, kBytes, read.data()).ok());
    CHECK(batch.release(buffer, VK_QUEUE_FAMILY_EXTERNAL).ok());
    CHECK(batch.submit().ok());
    CHECK(read == written);
  }
  return 0;
}

// An exported buffer whose descriptor was taken -- by a mapping before, or
// let go by its holder -- or that has no buffer, is refused, and the buffer
// stays the caller's.
int test_taken_descriptor(const Gpu& gpu) {
  auto context = sensor::CudaContext::open(gpu.device);
  CHECK(context.ok());
  auto exported = vkc::create_exported_buffer(gpu.device, gpu.allocator, 256);
  CHECK(exported.ok());
  auto first = sensor::CudaBufferMapping::map(*context.value(), *exported);
  CHECK(first.ok());
  auto again = sensor::CudaBufferMapping::map(*context.value(), *exported);
  CHECK(again.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(exported.value().buffer.valid());

  auto closed = vkc::create_exported_buffer(gpu.device, gpu.allocator, 256);
  CHECK(closed.ok());
  closed.value().fd.reset();
  CHECK(sensor::CudaBufferMapping::map(*context.value(), *closed)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  CHECK(closed.value().buffer.valid());

  auto emptied = vkc::create_exported_buffer(gpu.device, gpu.allocator, 256);
  CHECK(emptied.ok());
  const vkc::Buffer kept = std::move(emptied.value().buffer);
  CHECK(sensor::CudaBufferMapping::map(*context.value(), *emptied)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  CHECK(emptied.value().fd.valid());  // refused before CUDA took it
  return 0;
}

// An R8 and an R8G8 image, of odd size, that CUDA writes on a stream of its
// own, synchronized, read back through Vulkan with the bytes written, round
// after round: handed over to CUDA once, in GENERAL, before its first write,
// then taken back from VK_QUEUE_FAMILY_EXTERNAL after each write and never
// handed back, as NVDEC's pictures are.
int test_cuda_writes_vulkan_reads_images(const Gpu& gpu, const Libcuda& cu) {
  constexpr std::uint32_t kWidth = 129;
  constexpr std::uint32_t kHeight = 65;
  auto context = sensor::CudaContext::open(gpu.device);
  CHECK(context.ok());
  auto stream = sensor::CudaStream::open(*context.value());
  CHECK(stream.ok());
  for (const VkFormat format : {VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM}) {
    const std::uint32_t bytes_per = format == VK_FORMAT_R8_UNORM ? 1 : 2;
    vkc::ImageDesc desc;
    desc.extent = {kWidth, kHeight};
    desc.format = format;
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    auto exported = vkc::create_exported_image(gpu.device, gpu.allocator, desc);
    CHECK(exported.ok());
    vkc::Image& image = exported.value().image;
    auto mapping = sensor::CudaImageMapping::map(*context.value(), *exported);
    CHECK(mapping.ok() && mapping.value()->array() != nullptr);
    CHECK(!exported.value().fd.valid());  // CUDA's now
    CHECK(image.valid());                 // the caller's still
    {
      vkc::CommandBatch first(gpu.device, gpu.allocator);
      CHECK(
          first
              .release(image, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL)
              .ok());
      CHECK(first.submit().ok());
      image.set_layout(VK_IMAGE_LAYOUT_GENERAL);
    }
    const std::size_t row = std::size_t{kWidth} * bytes_per;
    const std::size_t bytes = row * kHeight;
    auto landing = vkc::device_storage_buffer(gpu.allocator,
                                              (bytes + 3) & ~std::size_t{3});
    CHECK(landing.ok());
    for (std::uint32_t round = 0; round < 3; ++round) {
      const std::vector<std::uint8_t> written =
          pattern(bytes, round + 4 * bytes_per);
      CHECK(cu.cuCtxPushCurrent(context.value()->handle()) == CUDA_SUCCESS);
      CUDA_MEMCPY2D copy{};
      copy.srcMemoryType = CU_MEMORYTYPE_HOST;
      copy.srcHost = written.data();
      copy.srcPitch = row;
      copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      copy.dstArray = mapping.value()->array();
      copy.WidthInBytes = row;
      copy.Height = kHeight;
      CUresult r = cu.cuMemcpy2DAsync(&copy, stream.value()->handle());
      if (r == CUDA_SUCCESS) {
        r = cu.cuStreamSynchronize(stream.value()->handle());
      }
      CUcontext popped = nullptr;
      CHECK(cu.cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
      CHECK(r == CUDA_SUCCESS);

      std::vector<std::uint8_t> read(bytes);
      vkc::CommandBatch batch(gpu.device, gpu.allocator);
      CHECK(
          batch
              .acquire(image, VK_QUEUE_FAMILY_EXTERNAL, VK_IMAGE_LAYOUT_GENERAL)
              .ok());
      CHECK(batch.copy(image, kWidth, kHeight, *landing, 0).ok());
      CHECK(batch.readback(*landing, 0, bytes, read.data()).ok());
      CHECK(batch.submit().ok());
      CHECK(read == written);
    }
  }
  return 0;
}

// An exported image whose descriptor was taken, or that has no image, or
// whose format CUDA is not given, is refused, and the image stays the
// caller's, its descriptor too where CUDA never took it.
int test_taken_image_descriptor(const Gpu& gpu) {
  auto context = sensor::CudaContext::open(gpu.device);
  CHECK(context.ok());
  vkc::ImageDesc desc;
  desc.extent = {16, 8};
  desc.format = VK_FORMAT_R8_UNORM;
  desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  auto exported = vkc::create_exported_image(gpu.device, gpu.allocator, desc);
  CHECK(exported.ok());
  auto first = sensor::CudaImageMapping::map(*context.value(), *exported);
  CHECK(first.ok());
  auto again = sensor::CudaImageMapping::map(*context.value(), *exported);
  CHECK(again.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(exported.value().image.valid());

  auto emptied = vkc::create_exported_image(gpu.device, gpu.allocator, desc);
  CHECK(emptied.ok());
  const vkc::Image kept = std::move(emptied.value().image);
  CHECK(sensor::CudaImageMapping::map(*context.value(), *emptied)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  CHECK(emptied.value().fd.valid());

  desc.format = VK_FORMAT_R16_UNORM;
  auto wide = vkc::create_exported_image(gpu.device, gpu.allocator, desc);
  if (wide.ok()) {
    CHECK(sensor::CudaImageMapping::map(*context.value(), *wide)
              .status()
              .domain() == vkc::Status::Code::InvalidArgument);
    CHECK(wide.value().fd.valid());
  }
  return 0;
}

int gpu_main(Gpu& gpu) {
  const vkc::Result<int> ordinal = sensor::cuda_ordinal_of(gpu.device);
  if (!ordinal) {
    if (ordinal.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", ordinal.status().message().c_str());
      return 1;
    }
    return vr_test::no_decoder(ordinal.status().message());
  }
  if (!gpu.device.exports_memory()) {
    return vr_test::no_decoder("the device does not export memory");
  }
  const Libcuda* cu = libcuda();
  CHECK(cu != nullptr);
  if (test_ordinal_and_context(gpu, *cu, ordinal.value()) != 0) return 1;
  if (test_cuda_writes_vulkan_reads(gpu, *cu) != 0) return 1;
  if (test_taken_descriptor(gpu) != 0) return 1;
  if (test_cuda_writes_vulkan_reads_images(gpu, *cu) != 0) return 1;
  if (test_taken_image_descriptor(gpu) != 0) return 1;
  std::puts("sensor_video_cuda_interop: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
