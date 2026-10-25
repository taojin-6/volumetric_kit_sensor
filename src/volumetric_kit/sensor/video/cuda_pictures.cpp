// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "cuda_pictures.hpp"

#include <atomic>
#include <string>
#include <utility>

#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"

namespace volumetric_kit::recon::sensor::video {

core::Result<std::unique_ptr<CudaSession>> CudaSession::open(
    const core::Device& device, const char* who) {
  auto s = std::make_unique<CudaSession>();
  core::Result<std::unique_ptr<CudaContext>> opened = CudaContext::open(device);
  if (!opened) return opened.status().with_context(who);
  s->context = std::move(opened).value();
  core::Result<std::unique_ptr<CudaStream>> made =
      CudaStream::open(*s->context);
  if (!made) return made.status().with_context(who);
  s->stream = std::move(made).value();
  return s;
}

core::Result<std::unique_ptr<CudaPictures>> CudaPictures::create(
    const core::Device& device, core::Allocator& allocator,
    const CudaContext& context, CUstream stream, const char* who) {
  if (!device.exports_memory()) {
    return core::Status::unsupported(std::string(who) +
                                     ": the device does not export memory "
                                     "(VK_KHR_external_memory_fd)");
  }
  return std::unique_ptr<CudaPictures>(
      new CudaPictures(device, allocator, context, stream, who));
}

core::Result<CudaPictures::Slot*> CudaPictures::slot(std::uint64_t bytes) {
  // Reused only once no picture holds it. A free one left over is too small,
  // so it is let go rather than kept past a change of picture size. Vulkan
  // does not release a reused buffer back to CUDA: CUDA overwrites the
  // picture, a family may take a buffer without a transfer when its contents
  // need not survive, and its last reader finished before letting it go.
  // use_count() is a relaxed load. Each fence orders CUDA's next write, or
  // the release, after whatever the thread that dropped the last other
  // reference did first, such as its fence wait on its last read.
  for (const std::unique_ptr<Slot>& s : slots_) {
    if (s->buffer.use_count() == 1 && s->bytes >= bytes) {
      std::atomic_thread_fence(std::memory_order_acquire);
      return s.get();
    }
  }
  for (auto it = slots_.begin(); it != slots_.end();) {
    if ((*it)->buffer.use_count() == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
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
  // Each owner is held from the moment it exists, the mapping after the
  // buffer, so a failed host allocation releases CUDA's view first.
  core::Result<std::unique_ptr<CudaBufferMapping>> mapping =
      CudaBufferMapping::map(*context_, exported);
  if (!mapping) return mapping.status().with_context(who_);
  auto s = std::make_unique<Slot>();
  s->buffer = std::make_shared<core::Buffer>(std::move(exported.buffer));
  s->mapping = std::move(mapping).value();
  s->bytes = bytes;
  slots_.push_back(std::move(s));
  return slots_.back().get();
}

core::Result<CudaPictures::Target> CudaPictures::take(std::uint64_t bytes) {
  VKC_ASSIGN(Slot * s, slot(bytes));
  return Target{s->buffer, s->mapping->pointer()};
}

core::Status CudaPictures::copy(CUdeviceptr luma, std::size_t luma_pitch,
                                CUdeviceptr chroma, std::size_t chroma_pitch,
                                std::uint32_t width, std::uint32_t height,
                                YuvImage& out) {
  // Rows packed: Y, then the interleaved chroma at half height, each pair of
  // samples covering two luma columns, so a chroma row is the width rounded
  // up to even.
  const std::uint64_t chroma_row = (std::uint64_t{width} + 1) / 2 * 2;
  const std::uint64_t chroma_rows = (std::uint64_t{height} + 1) / 2;
  const std::uint64_t chroma_at = round_up(std::uint64_t{width} * height, 256);
  const std::uint64_t bytes = round_up(chroma_at + chroma_row * chroma_rows, 4);
  VKC_ASSIGN(Slot * s, slot(bytes));

  const CudaContextScope scope(context_->handle());
  if (!scope.ok()) {
    return cuda_error(who_, scope.result(), "making the CUDA context current");
  }
  const CUdeviceptr pointer = s->mapping->pointer();
  CUDA_MEMCPY2D plane{};
  plane.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  plane.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  plane.srcDevice = luma;
  plane.srcPitch = luma_pitch;
  plane.dstDevice = pointer;
  plane.dstPitch = width;
  plane.WidthInBytes = width;
  plane.Height = height;
  const CudaDriver* cu = cuda_driver();
  CUresult r = cu->cuMemcpy2DAsync(&plane, stream_);
  if (r == CUDA_SUCCESS) {
    plane.srcDevice = chroma;
    plane.srcPitch = chroma_pitch;
    plane.dstDevice = pointer + chroma_at;
    plane.dstPitch = chroma_row;
    plane.WidthInBytes = chroma_row;
    plane.Height = chroma_rows;
    r = cu->cuMemcpy2DAsync(&plane, stream_);
  }
  if (r == CUDA_SUCCESS) r = cu->cuStreamSynchronize(stream_);
  if (r != CUDA_SUCCESS) return cuda_error(who_, r, "copying a picture");

  out.width = width;
  out.height = height;
  out.layout = YuvLayout::Nv12;
  out.stride[0] = width;
  out.stride[1] = chroma_row;
  place_cuda_buffer(s->buffer, out);
  out.offset[0] = 0;
  out.offset[1] = chroma_at;
  return {};
}

}  // namespace volumetric_kit::recon::sensor::video
