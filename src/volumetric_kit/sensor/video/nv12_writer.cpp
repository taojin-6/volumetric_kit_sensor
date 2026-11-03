// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// testing::Nv12Writer: one compute kernel (shaders/write_nv12.comp) that
// stores a packed NV12 picture into two plane images, built by hand, since
// the core's KernelSetBuilder binds storage buffers only, with a pool of
// kPending sets taken in turn.

#include "volumetric_kit/sensor/testing/nv12_writer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_pipeline.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/shader.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "write_nv12_comp.spv.hpp"

namespace volumetric_kit::sensor::testing {
namespace {

constexpr const char* kWho = "Nv12Writer";
constexpr std::uint32_t kLocalSize = 256;  // write_nv12.comp's local_size_x

// write_nv12.comp's Params, under scalar layout.
struct Params {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t chroma_at = 0;
};
static_assert(sizeof(Params) == 12, "Params must match write_nv12.comp");

core::Status invalid(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

std::size_t luma_bytes(std::uint32_t width, std::uint32_t height) {
  return std::size_t{width} * height;
}

std::size_t chroma_bytes(std::uint32_t width, std::uint32_t height) {
  return std::size_t{(width + 1) / 2} * ((height + 1) / 2) * 2;
}

// Empty when @p image is a plane the kernel stores into, of @p format, at
// least @p columns x @p rows; else what it is not.
std::string plane_breach(const std::shared_ptr<const core::Image>& image,
                         VkFormat format, std::uint32_t columns,
                         std::uint32_t rows) {
  if (image == nullptr || !image->valid() || image->view() == VK_NULL_HANDLE) {
    return "no image with a view";
  }
  if (image->format() != format) return "another format";
  if (image->layout() != VK_IMAGE_LAYOUT_GENERAL) return "not in GENERAL";
  if ((image->usage() & VK_IMAGE_USAGE_STORAGE_BIT) == 0) {
    return "no STORAGE usage";
  }
  if (image->extent().width < columns || image->extent().height < rows) {
    return "smaller than the picture";
  }
  return {};
}

}  // namespace

struct Nv12Writer::Impl {
  const core::Device* device = nullptr;
  core::ComputeKernel kernel;
  core::DescriptorPool pool;
  std::array<core::DescriptorSet, kPending> sets;
  std::uint32_t next = 0;  // the set the next record writes
  std::uint32_t max_groups = 0;
};

Nv12Writer::Nv12Writer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
Nv12Writer::Nv12Writer(Nv12Writer&& other) noexcept = default;
Nv12Writer& Nv12Writer::operator=(Nv12Writer&& other) noexcept = default;
Nv12Writer::~Nv12Writer() = default;

core::Result<Nv12Writer> Nv12Writer::create(const core::Device& device) try {
  VKC_TRY(device.check_enabled(hevc_encoder_device_requirements())
              .with_context("Nv12Writer::create"));
  auto impl = std::make_unique<Impl>();
  impl->device = &device;
  // The packed picture, then the luma and chroma planes as storage images.
  VkDescriptorSetLayoutBinding bindings[3]{};
  for (std::uint32_t i = 0; i < 3; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  core::ComputeKernel& kernel = impl->kernel;
  kernel.name = "write_nv12";
  kernel.push_bytes = sizeof(Params);
  kernel.bindings = 1;  // the storage buffers among them
  VKC_ASSIGN(kernel.layout,
             core::DescriptorSetLayout::create(device.handle(), bindings, 3));
  // The module is transient: the pipeline does not retain it. SPIR-V is a
  // stream of words, and vkc_embed_shaders aligns its arrays to 4.
  VKC_ASSIGN(const core::ShaderModule module,
             core::ShaderModule::create(
                 device.handle(),
                 // NOLINTNEXTLINE(*-reinterpret-cast)
                 reinterpret_cast<const std::uint32_t*>(vs_write_nv12_comp_spv),
                 vs_write_nv12_comp_spv_size));
  const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(Params)};
  const VkDescriptorSetLayout layout = kernel.layout.handle();
  core::ComputePipelineDesc desc;
  desc.shader = &module;
  desc.set_layouts = &layout;
  desc.set_layout_count = 1;
  desc.push_ranges = &push;
  desc.push_range_count = 1;
  desc.cache = device.pipeline_cache();
  VKC_ASSIGN(kernel.pipeline,
             core::ComputePipeline::create(device.handle(), desc));
  // The kernel's own set, which a dispatch must have, and the ones taken in
  // turn.
  constexpr std::uint32_t kSets = kPending + 1;
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSets},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kSets}};
  VKC_ASSIGN(impl->pool,
             core::DescriptorPool::create(device.handle(), sizes, 2, kSets));
  VKC_ASSIGN(kernel.set, impl->pool.allocate(layout));
  for (core::DescriptorSet& set : impl->sets) {
    VKC_ASSIGN(set, impl->pool.allocate(layout));
  }
  device.set_object_name(VK_OBJECT_TYPE_PIPELINE,
                         core::debug_object_handle(kernel.pipeline.handle()),
                         kernel.name);
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  impl->max_groups = props.limits.maxComputeWorkGroupCount[0];
  return Nv12Writer(std::move(impl));
} catch (...) {
  return core::exception_status("Nv12Writer::create",
                                core::Status::Code::OutOfMemory);
}

core::Result<core::Buffer> Nv12Writer::upload(core::CommandBatch& batch,
                                              core::Allocator& allocator,
                                              const HostNv12& picture) try {
  const std::size_t luma = luma_bytes(picture.width, picture.height);
  const std::size_t chroma = chroma_bytes(picture.width, picture.height);
  if (picture.width == 0 || picture.height == 0 ||
      picture.luma.size() != luma || picture.chroma.size() != chroma) {
    return invalid(
        "upload: an NV12 picture of no size or with planes of other lengths");
  }
  // Whole words, which the kernel reads.
  const std::size_t bytes = (luma + chroma + 3) & ~std::size_t{3};
  VKC_ASSIGN(core::Buffer planes,
             core::device_storage_buffer(allocator, bytes));
  VKC_TRY(batch.upload(planes, 0, picture.luma.data(), luma));
  VKC_TRY(batch.upload(planes, luma, picture.chroma.data(), chroma));
  return planes;
} catch (...) {
  return core::exception_status("Nv12Writer::upload",
                                core::Status::Code::OutOfMemory);
}

core::Status Nv12Writer::record(core::CommandBatch& batch,
                                const core::Buffer& planes,
                                const YuvImage& target) try {
  if (impl_ == nullptr) return invalid("moved from");
  Impl& m = *impl_;
  const std::uint32_t w = target.width;
  const std::uint32_t h = target.height;
  if (target.layout != YuvLayout::Nv12 || w == 0 || h == 0) {
    return invalid("record: the target is not an NV12 picture of a size");
  }
  if (const std::string b =
          plane_breach(target.image[0], VK_FORMAT_R8_UNORM, w, h);
      !b.empty()) {
    return invalid("record: the target's luma image: " + b);
  }
  if (const std::string b = plane_breach(target.image[1], VK_FORMAT_R8G8_UNORM,
                                         (w + 1) / 2, (h + 1) / 2);
      !b.empty()) {
    return invalid("record: the target's chroma image: " + b);
  }
  const std::size_t luma = luma_bytes(w, h);
  const std::size_t bytes = luma + chroma_bytes(w, h);
  if (!planes.valid() || planes.size() < bytes ||
      (planes.usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0) {
    return invalid("record: the planes are no storage buffer of the picture");
  }
  const core::DescriptorSet& set = m.sets[m.next];
  m.next = (m.next + 1) % kPending;
  set.write_storage_buffer(0, planes.handle(), 0, VK_WHOLE_SIZE);
  set.write_storage_image(1, target.image[0]->view(), VK_IMAGE_LAYOUT_GENERAL);
  set.write_storage_image(2, target.image[1]->view(), VK_IMAGE_LAYOUT_GENERAL);
  Params params;
  params.width = w;
  params.height = h;
  params.chroma_at = static_cast<std::uint32_t>(luma);
  const std::uint32_t groups =
      core::group_count(static_cast<std::uint32_t>((luma + 3) / 4), kLocalSize);
  return batch.dispatch(m.kernel, set, &params, sizeof(params), groups,
                        m.max_groups);
} catch (...) {
  return core::exception_status("Nv12Writer::record",
                                core::Status::Code::OutOfMemory);
}

core::Status Nv12Writer::record_upload(core::CommandBatch& batch,
                                       core::Allocator& allocator,
                                       const HostNv12& picture,
                                       const YuvImage& target) {
  if (picture.width != target.width || picture.height != target.height) {
    return invalid("record_upload: a picture of another size than the target");
  }
  VKC_ASSIGN(core::Buffer planes, upload(batch, allocator, picture));
  const core::Status recorded = record(batch, planes, target);
  // Kept even when the kernel was refused: the upload into it is recorded.
  batch.retain(std::move(planes));
  return recorded;
}

}  // namespace volumetric_kit::sensor::testing
