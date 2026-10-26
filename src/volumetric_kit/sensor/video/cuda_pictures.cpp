// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "cuda_pictures.hpp"

#include <atomic>
#include <string>
#include <utility>

#include "../sharing_families.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"

namespace volumetric_kit::recon::sensor::video {
namespace {

core::Status no_export(const char* who) {
  return core::Status::unsupported(std::string(who) +
                                   ": the device does not export memory "
                                   "(VK_KHR_external_memory_fd)");
}

// The first of @p slots that no picture holds and @p fits, or null, after
// letting go of every other one no picture holds. A slot is held while its
// `owner(slot)` has another reference than the slot's own.
//
// Reused only once no picture holds it, and a free one left over does not
// fit, so it is let go rather than kept past a change of picture size.
// Vulkan does not hand a reused one back to CUDA: CUDA overwrites the whole
// picture, a family may take memory without a transfer when its contents need
// not survive, and its last reader finished before letting it go. An image
// stays in GENERAL, the layout CUDA was handed it in and its readers find it
// in. use_count() is a relaxed load. Each fence orders CUDA's next write, or
// the release, after whatever the thread that dropped the last other
// reference did first, such as its fence wait on its last read.
template <typename Slot, typename Owner, typename Fits>
Slot* free_slot(std::vector<std::unique_ptr<Slot>>& slots, Owner owner,
                Fits fits) {
  for (const std::unique_ptr<Slot>& s : slots) {
    if (owner(*s).use_count() == 1 && fits(*s)) {
      std::atomic_thread_fence(std::memory_order_acquire);
      return s.get();
    }
  }
  for (auto it = slots.begin(); it != slots.end();) {
    if (owner(**it).use_count() == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      it = slots.erase(it);
    } else {
      ++it;
    }
  }
  return nullptr;
}

}  // namespace

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
    const CudaContext& context, const char* who) {
  if (!device.exports_memory()) return no_export(who);
  return std::unique_ptr<CudaPictures>(
      new CudaPictures(device, allocator, context, who));
}

core::Result<CudaPictures::Target> CudaPictures::take(std::uint64_t bytes) {
  Slot* s = free_slot(
      slots_, [](const Slot& slot) -> const auto& { return slot.buffer; },
      [bytes](const Slot& slot) { return slot.bytes >= bytes; });
  if (s == nullptr) {
    VKC_ASSIGN(core::ExportedBuffer exported,
               core::create_exported_buffer(*device_, *allocator_, bytes));
    device_->set_object_name(
        VK_OBJECT_TYPE_BUFFER,
        core::debug_object_handle(exported.buffer.handle()),
        "sensor.decoded_picture");
    // Each owner is held from the moment it exists, the mapping after the
    // buffer, so a failed host allocation releases CUDA's view first.
    core::Result<std::unique_ptr<CudaBufferMapping>> mapping =
        CudaBufferMapping::map(*context_, exported);
    if (!mapping) return mapping.status().with_context(who_);
    auto made = std::make_unique<Slot>();
    made->buffer = std::make_shared<core::Buffer>(std::move(exported.buffer));
    made->mapping = std::move(mapping).value();
    made->bytes = bytes;
    slots_.push_back(std::move(made));
    s = slots_.back().get();
  }
  return Target{s->buffer, s->mapping->pointer()};
}

core::Result<std::unique_ptr<CudaPlanePictures>> CudaPlanePictures::create(
    const core::Device& device, core::Allocator& allocator,
    const CudaContext& context, CUstream stream,
    const std::uint32_t* queue_families, std::uint32_t queue_family_count,
    const char* who) {
  VKC_ASSIGN(std::vector<std::uint32_t> families,
             detail::sharing_families(device, queue_families,
                                      queue_family_count, who));
  if (!device.exports_memory()) return no_export(who);
  return std::unique_ptr<CudaPlanePictures>(new CudaPlanePictures(
      device, allocator, context, stream, std::move(families), who));
}

core::Result<std::unique_ptr<CudaPlanePictures::Slot>>
CudaPlanePictures::make_slot(std::uint32_t width, std::uint32_t height) {
  // Each owner is held from the moment it exists, CUDA's views after the
  // images, so a failure releases the views first.
  auto s = std::make_unique<Slot>();
  s->planes = std::make_shared<Planes>();
  s->width = width;
  s->height = height;
  static constexpr const char* kNames[2] = {"sensor.decoded_luma",
                                            "sensor.decoded_chroma"};
  for (int p = 0; p < 2; ++p) {
    core::ImageDesc desc;
    desc.extent = p == 0 ? VkExtent2D{width, height}
                         : VkExtent2D{(width + 1) / 2, (height + 1) / 2};
    desc.format = p == 0 ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
    // Sampled where it is (GpuFramePrep, a renderer), copied by readback.
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    desc.queue_families = families_.empty() ? nullptr : families_.data();
    desc.queue_family_count = static_cast<std::uint32_t>(families_.size());
    VKC_ASSIGN(core::ExportedImage exported,
               core::create_exported_image(*device_, *allocator_, desc));
    device_->set_object_name(VK_OBJECT_TYPE_IMAGE,
                             core::debug_object_handle(exported.image.handle()),
                             kNames[p]);
    core::Result<std::unique_ptr<CudaImageMapping>> mapping =
        CudaImageMapping::map(*context_, exported);
    if (!mapping) return mapping.status().with_context(who_);
    s->planes->image[p] = std::move(exported.image);
    s->mapping[p] = std::move(mapping).value();
  }
  // Handed over to CUDA once, before its first write, in GENERAL; the batch
  // waits, so CUDA writes only after it. Readers take each picture back from
  // outside Vulkan.
  core::CommandBatch batch(*device_, *allocator_);
  for (const core::Image& image : s->planes->image) {
    VKC_TRY(batch.release(image, VK_QUEUE_FAMILY_EXTERNAL,
                          VK_IMAGE_LAYOUT_GENERAL));
  }
  const core::Status handed = batch.submit();
  if (!handed.ok()) {
    // A failed wait may leave the barriers on the device, so the images are
    // let go rather than freed, as GpuFramePrep lets go of a picture.
    static_cast<void>(new std::shared_ptr<Planes>(s->planes));
    return handed.with_context(who_);
  }
  for (core::Image& image : s->planes->image) {
    image.set_layout(VK_IMAGE_LAYOUT_GENERAL);
  }
  return s;
}

core::Result<CudaPlanePictures::Slot*> CudaPlanePictures::slot(
    std::uint32_t width, std::uint32_t height) {
  Slot* s = free_slot(
      slots_, [](const Slot& slot) -> const auto& { return slot.planes; },
      [width, height](const Slot& slot) {
        return slot.width == width && slot.height == height;
      });
  if (s != nullptr) return s;
  VKC_ASSIGN(std::unique_ptr<Slot> made, make_slot(width, height));
  slots_.push_back(std::move(made));
  return slots_.back().get();
}

core::Status CudaPlanePictures::copy(CUdeviceptr luma, std::size_t luma_stride,
                                     CUdeviceptr chroma,
                                     std::size_t chroma_stride,
                                     std::uint32_t width, std::uint32_t height,
                                     YuvImage& out) {
  VKC_ASSIGN(Slot * s, slot(width, height));

  const CudaContextScope scope(context_->handle());
  if (!scope.ok()) {
    return cuda_error(who_, scope.result(), "making the CUDA context current");
  }
  // Y, then the interleaved chroma at half height: Cb and Cr for each two
  // luma columns, so a chroma row is the width rounded up to even.
  CUDA_MEMCPY2D plane{};
  plane.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  plane.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  plane.srcDevice = luma;
  plane.srcPitch = luma_stride;
  plane.dstArray = s->mapping[0]->array();
  plane.WidthInBytes = width;
  plane.Height = height;
  const CudaDriver* cu = cuda_driver();
  CUresult r = cu->cuMemcpy2DAsync(&plane, stream_);
  if (r == CUDA_SUCCESS) {
    plane.srcDevice = chroma;
    plane.srcPitch = chroma_stride;
    plane.dstArray = s->mapping[1]->array();
    plane.WidthInBytes = (std::size_t{width} + 1) / 2 * 2;
    plane.Height = (std::size_t{height} + 1) / 2;
    r = cu->cuMemcpy2DAsync(&plane, stream_);
  }
  if (r == CUDA_SUCCESS) r = cu->cuStreamSynchronize(stream_);
  if (r != CUDA_SUCCESS) return cuda_error(who_, r, "copying a picture");

  out.width = width;
  out.height = height;
  out.layout = YuvLayout::Nv12;
  out.device = nullptr;
  for (int p = 0; p < 2; ++p) {
    out.image[p] =
        std::shared_ptr<const core::Image>(s->planes, &s->planes->image[p]);
  }
  out.queue_family = kQueueFamilyExternal;
  return {};
}

}  // namespace volumetric_kit::recon::sensor::video
