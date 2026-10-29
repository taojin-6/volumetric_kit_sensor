// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The rectifier's views read through CUDA, as a stereo matcher reads them:
// each exported view mapped once (CudaBufferMapping), copied to the host by
// CUDA once rectify has returned, holds the bytes the pass wrote; rectified
// again, the same mappings read the next pair, the views taken back from
// VK_QUEUE_FAMILY_EXTERNAL and handed out there again. The checks call
// libcuda through a loader of their own.
//
// Where CUDA does not reach the device's GPU, or the device exports no
// memory, the test skips; VS_TEST_HEVC_BACKEND, which CI sets on the NVIDIA
// leg, makes it fail instead, as for the decoders.

#include <cuda.h>
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

// A macro's expansion as a string: the versioned symbol a cuda.h name stands
// for (cuMemcpyDtoH is cuMemcpyDtoH_v2).
#define VS_TEST_STRING(name) VS_TEST_STRING_(name)
#define VS_TEST_STRING_(name) #name

namespace {

#define VS_TEST_CUDA_FUNCTIONS(X) \
  X(cuCtxPushCurrent)             \
  X(cuCtxPopCurrent)              \
  X(cuMemcpyDtoH)

struct Libcuda {
#define VS_TEST_DECLARE(name) decltype(&::name) name = nullptr;
  VS_TEST_CUDA_FUNCTIONS(VS_TEST_DECLARE)
#undef VS_TEST_DECLARE
};

// libcuda, loaded once and kept; null where it does not load or lacks a
// call.
const Libcuda* libcuda() {
  static const std::unique_ptr<Libcuda> loaded = [] {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return std::unique_ptr<Libcuda>();
    auto l = std::make_unique<Libcuda>();
    bool ok = true;
#define VS_TEST_LOAD(name)                                                   \
  l->name =                                                                  \
      reinterpret_cast<decltype(l->name)>(dlsym(lib, VS_TEST_STRING(name))); \
  ok = ok && l->name != nullptr;
    VS_TEST_CUDA_FUNCTIONS(VS_TEST_LOAD)
#undef VS_TEST_LOAD
    return ok ? std::move(l) : std::unique_ptr<Libcuda>();
  }();
  return loaded.get();
}

// Random luma, a byte a pixel, rows packed: a round's picture.
std::vector<std::uint8_t> noise(std::size_t bytes, std::uint32_t seed) {
  std::vector<std::uint8_t> out(bytes);
  std::uint32_t x = 0x9e3779b9u * (seed + 1);
  for (std::uint8_t& b : out) {
    x = x * 1664525u + 1013904223u;
    b = static_cast<std::uint8_t>(x >> 24);
  }
  return out;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const vkc::Result<int> ordinal = sensor::cuda_ordinal_of(gpu.device);
  if (!ordinal) {
    if (ordinal.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", ordinal.status().message().c_str());
      return 1;
    }
    return vs_test::no_decoder(ordinal.status().message());
  }
  if (!gpu.device.exports_memory()) {
    return vs_test::no_decoder("the device does not export memory");
  }
  const Libcuda* cu = libcuda();
  CHECK(cu != nullptr);
  auto context = sensor::CudaContext::open(gpu.device);
  CHECK(context.ok());

  // A pair rectified already, at its own size: each view is its picture,
  // byte for byte.
  constexpr std::uint32_t kW = 640, kH = 360;
  sensor::StereoInfo info;
  info.calibration.left = info.calibration.right =
      camera::CameraModel{{kW, kH}, {330.0, 330.0, 319.5, 179.5}, {}};
  info.calibration.left_to_right[3] = glm::dvec4(-0.05, 0.0, 0.0, 1.0);
  auto rectifier = sensor::StereoRectifier::create(
      gpu.device, gpu.allocator, info,
      {{}, 0.0f, sensor::StereoRectifier::Output::Luma8});
  CHECK(rectifier.ok());
  // Made after the rectifier, so gone before the buffers they map.
  std::unique_ptr<sensor::CudaBufferMapping> maps[2];
  const vkc::ExportedBuffer* mapped[2] = {nullptr, nullptr};
  const std::size_t bytes = std::size_t{kW} * kH;
  for (std::uint32_t round = 0; round < 3; ++round) {
    std::vector<std::uint8_t> pictures[2] = {noise(bytes, 2 * round),
                                             noise(bytes, 2 * round + 1)};
    sensor::StereoFrame frame;
    sensor::YuvImage* images[2] = {&frame.left, &frame.right};
    for (int v = 0; v < 2; ++v) {
      images[v]->layout = sensor::YuvLayout::Luma;
      images[v]->host = pictures[v].data();
      images[v]->stride[0] = kW;
      images[v]->width = kW;
      images[v]->height = kH;
    }
    auto pair = rectifier.value()->rectify(frame);
    if (!pair) std::fprintf(stderr, "%s\n", pair.status().message().c_str());
    CHECK(pair.ok());
    CHECK(pair->queue_family == sensor::kQueueFamilyExternal);
    const std::shared_ptr<vkc::ExportedBuffer> views[2] = {pair->left,
                                                           pair->right};
    for (int v = 0; v < 2; ++v) {
      if (maps[v] == nullptr) {
        auto map = sensor::CudaBufferMapping::map(*context.value(), *views[v]);
        if (!map) std::fprintf(stderr, "%s\n", map.status().message().c_str());
        CHECK(map.ok());
        maps[v] = std::move(map).value();
        mapped[v] = views[v].get();
      }
      // The view no pair held is the one written again, mapped already.
      CHECK(views[v].get() == mapped[v] && !views[v]->fd.valid());
      std::vector<std::uint8_t> read(bytes);
      CHECK(cu->cuCtxPushCurrent(context.value()->handle()) == CUDA_SUCCESS);
      const CUresult copied =
          cu->cuMemcpyDtoH(read.data(), maps[v]->pointer(), bytes);
      CUcontext popped = nullptr;
      CHECK(cu->cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
      CHECK(copied == CUDA_SUCCESS);
      CHECK(read == pictures[v]);
    }
  }
  std::puts("  three pairs read through CUDA, each view mapped once");
  std::puts("sensor_stereo_cuda: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
