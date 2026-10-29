// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "color_pass.hpp"
#include "undistort_color_comp.spv.hpp"
#include "undistort_color_images_comp.spv.hpp"
#include "undistort_depth_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_pipeline.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/shader.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace volumetric_kit::sensor {
namespace {

using detail::lens_params;
using detail::LensParams;

// Both kernels' local_size_x.
constexpr std::uint32_t kLocalSize = 256;

// The most commands record_inputs and record_passes time for one frame: the
// slot's copy and the two passes.
constexpr std::uint32_t kSpansPerFrame = 3;

struct DepthParams {
  LensParams cam;
  float metres_per_unit;
  std::uint32_t within_color;  // mask by OverlapParams
  float depth_offset_m;        // subtracted from every return
  std::uint32_t raw_first;     // the image's first sample in the binding
};
static_assert(sizeof(DepthParams) == 72, "DepthParams layout drift");
static_assert(offsetof(DepthParams, depth_offset_m) == 64,
              "DepthParams layout drift");
static_assert(offsetof(DepthParams, raw_first) == 68,
              "DepthParams layout drift");

// undistort_depth.comp's Overlap block: the colour camera, and the rigid
// transform from the depth camera's frame into the colour camera's
// (column-major, as the shader's mat4).
struct OverlapParams {
  LensParams color;
  Mat4f depth_to_color;
};
static_assert(sizeof(OverlapParams) == 56 + 64, "OverlapParams layout drift");
static_assert(offsetof(OverlapParams, depth_to_color) == 56,
              "OverlapParams layout drift");

// rgbd_frame.hpp spells Vulkan's special queue families without Vulkan.
static_assert(kQueueFamilyIgnored == VK_QUEUE_FAMILY_IGNORED,
              "kQueueFamilyIgnored must be VK_QUEUE_FAMILY_IGNORED");
static_assert(kQueueFamilyExternal == VK_QUEUE_FAMILY_EXTERNAL,
              "kQueueFamilyExternal must be VK_QUEUE_FAMILY_EXTERNAL");

bool finite(float v) noexcept { return std::isfinite(v); }

// A camera the pass can undistort (detail::check_camera).
core::Status check_camera(const char* what, const camera::CameraModel& c) {
  return detail::check_camera("GpuFramePrep", what, c);
}

// A pose the pass can apply; `what` names it in the error.
core::Status check_pose(const char* what, const camera::Mat4d& pose) {
  const core::Status rigid = camera::check_rigid(pose);
  if (!rigid.ok()) {
    return core::Status::invalid_argument(std::string("GpuFramePrep: ") + what +
                                          ": " + rigid.message());
  }
  return {};
}

VkDeviceSize round_up4(VkDeviceSize bytes) noexcept {
  return (bytes + 3) & ~VkDeviceSize{3};
}

// `bytes` rounded up to a multiple of `alignment`, a power of two.
VkDeviceSize round_up(VkDeviceSize bytes, VkDeviceSize alignment) noexcept {
  return (bytes + alignment - 1) & ~(alignment - 1);
}

// A frame's depth half, checked: the range the kernel binds, where the image
// starts in it, and the bytes it moves. Host samples are copied into the slot
// from byte 0 and bound from there; samples in a buffer are read where they
// are, bound from the first one rounded down to the alignment a binding's
// offset needs.
struct DepthLayout {
  std::uint32_t pixels = 0;
  bool on_device = false;
  VkDeviceSize staged_bytes = 0;  // host samples the slot takes, to a word
  VkDeviceSize bind_offset = 0, bind_bytes = 0;
  std::uint32_t first = 0;     // the image's first sample in the binding
  VkDeviceSize out_bytes = 0;  // the metres
};

// A frame's colour half, checked: its planes as the kernel binds them
// (detail::PlaneBinding), packed words staged whole and bound from byte 0,
// and the bytes it writes.
struct ColorLayout {
  std::uint32_t pixels = 0;
  detail::PlaneBinding planes;
  VkDeviceSize out_bytes = 0;
};

core::Result<DepthLayout> check_depth(const RgbdFrame& frame,
                                      std::uint64_t max_pixels,
                                      VkDeviceSize max_range,
                                      VkDeviceSize offset_alignment) {
  const camera::CameraModel& cam = frame.depth_camera;
  const DeviceDepth& device = frame.depth_device;
  if (!frame.has_depth()) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the frame has no depth");
  }
  if (frame.depth != nullptr && device.buffer != nullptr) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth must be on the host or on the device, not "
        "both");
  }
  VKC_TRY(check_camera("depth", cam));
  if (!(finite(frame.metres_per_unit) && frame.metres_per_unit > 0.0f)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: metres_per_unit is not finite and positive");
  }
  // 0 is the pass's "no return", where the sensor had none and where the lens
  // maps outside the image, so a range reaching it would fuse those pixels
  // as a surface at the camera; and a range left at RgbdFrame's zeros would
  // quietly fuse nothing.
  if (!(finite(frame.min_depth) && finite(frame.max_depth) &&
        frame.min_depth > 0.0f && frame.min_depth < frame.max_depth)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth range [" + std::to_string(frame.min_depth) +
        ", " + std::to_string(frame.max_depth) +
        "] m must be finite, with 0 < min_depth < max_depth");
  }
  VKC_TRY(check_pose("color_to_world", frame.color_to_world));
  VKC_TRY(check_pose("depth_to_color", frame.depth_to_color));
  const std::uint64_t pixels = std::uint64_t{cam.size.width} * cam.size.height;
  if (pixels > max_pixels) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth image is past a single dispatch");
  }
  DepthLayout out;
  out.pixels = static_cast<std::uint32_t>(pixels);
  out.out_bytes = pixels * sizeof(float);
  VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the depth buffer",
                                           out.out_bytes, max_range));
  const VkDeviceSize bytes = pixels * sizeof(std::uint16_t);
  if (device.buffer == nullptr) {
    // Copied into the slot from byte 0, which the kernel reads in words.
    out.staged_bytes = out.bind_bytes = round_up4(bytes);
    return out;
  }
  out.on_device = true;
  // Empty or without storage usage, before its size is weighed.
  const core::StorageInput input(*device.buffer);
  VKC_TRY(input.check("GpuFramePrep: the depth's buffer", 0));
  if (device.offset % sizeof(std::uint16_t) != 0) {
    return core::Status::invalid_argument("GpuFramePrep: the depth's offset, " +
                                          std::to_string(device.offset) +
                                          ", is odd; its samples are 16-bit");
  }
  const VkDeviceSize size = device.buffer->size();
  if (device.offset > size || bytes > size - device.offset) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth runs past its buffer");
  }
  // The kernel reads whole words, so the one holding the last sample must be
  // in the buffer.
  const VkDeviceSize end = round_up4(device.offset + bytes);
  VKC_TRY(input.check("GpuFramePrep: the depth's buffer", end));
  // Bound from the first sample, rounded down to the alignment a binding's
  // offset needs (and to a word), and the kernel told where the image starts
  // in it, so depth at any even byte is read where it is, with no copy, and
  // the limit below weighs the image rather than where it sits in its buffer.
  const VkDeviceSize align = std::max<VkDeviceSize>(offset_alignment, 4);
  out.bind_offset = device.offset / align * align;
  out.bind_bytes = end - out.bind_offset;
  out.first = static_cast<std::uint32_t>((device.offset - out.bind_offset) /
                                         sizeof(std::uint16_t));
  VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the depth samples",
                                           out.bind_bytes, max_range));
  return out;
}

core::Result<ColorLayout> check_color(const RgbdFrame& frame,
                                      std::uint64_t max_pixels,
                                      VkDeviceSize max_range,
                                      VkDeviceSize offset_alignment) {
  const camera::ImageSize& cam = frame.color_camera.size;
  const YuvImage& image = frame.color;
  if (!is_canonical(frame.color_encoding)) {
    // TODO(sensor): the other transfers and primaries, through the curve
    // and matrix sensor::to_canonical uses on the host.
    return core::Status::unsupported(
        "GpuFramePrep: converts colour from the canonical encoding only (sRGB "
        "or BT.709 transfer, BT.709 primaries)");
  }
  VKC_TRY(check_camera("colour", frame.color_camera));
  const std::uint64_t pixels = std::uint64_t{cam.width} * cam.height;
  if (pixels > max_pixels) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour image is past a single dispatch");
  }
  ColorLayout out;
  out.pixels = static_cast<std::uint32_t>(pixels);
  out.out_bytes = pixels * sizeof(std::uint32_t);
  if (frame.color_packed != nullptr) {
    if (image.device != nullptr || image.host != nullptr ||
        image.image[0] != nullptr || image.image[1] != nullptr) {
      return core::Status::invalid_argument(
          "GpuFramePrep: the colour must be packed words or Y'CbCr planes, "
          "not both");
    }
    // Staged whole, a word a pixel, and bound from byte 0 of the input.
    out.planes.bind_bytes = out.planes.copy_bytes = out.out_bytes;
    VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                             out.out_bytes, max_range));
    return out;
  }
  VKC_ASSIGN(out.planes,
             detail::bind_planes(image, cam, "GpuFramePrep", "colour",
                                 max_range, offset_alignment));
  VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                           out.out_bytes, max_range));
  return out;
}

// An output's queue families: no more than the config's array holds, each one
// the device has.
core::Status check_queue_families(const core::Device& device,
                                  const std::uint32_t* families,
                                  std::uint32_t count, const char* who) {
  VKC_TRY(core::check_queue_family_count(count, who));
  const std::size_t device_families = device.caps().queue_families().size();
  for (std::uint32_t i = 0; i < count; ++i) {
    if (families[i] >= device_families) {
      return core::Status::invalid_argument(
          std::string(who) + ": queue family " + std::to_string(families[i]) +
          ", which the device does not have (" +
          std::to_string(device_families) + " families)");
    }
  }
  return {};
}

// The image entry's sampler: nearest and clamped, though texelFetch, its one
// read, filters nothing; a combined image sampler needs one.
core::Result<core::UniqueHandle<VkSampler, vkDestroySampler>> make_sampler(
    const core::Device& device) {
  VkSamplerCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = VK_FILTER_NEAREST;
  info.minFilter = VK_FILTER_NEAREST;
  info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VkSampler sampler = VK_NULL_HANDLE;
  VKC_VK_TRY(vkCreateSampler(device.handle(), &info, nullptr, &sampler));
  return core::UniqueHandle<VkSampler, vkDestroySampler>(device.handle(),
                                                         sampler);
}

// The colour pass's image entry, built by hand, since KernelSetBuilder binds
// storage buffers only: two combined image samplers, NV12's planes, then the
// output, with a pool of its own for the kernel's one set. `out` is replaced
// only once everything is built.
core::Status build_images_kernel(const core::Device& device,
                                 const VkPushConstantRange& push,
                                 core::ComputeKernel& out,
                                 core::DescriptorPool& pool) {
  VkDescriptorSetLayoutBinding bindings[3]{};
  for (std::uint32_t i = 0; i < 3; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = i < 2
                                     ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  core::ComputeKernel kernel;
  kernel.name = "undistort_color_images";
  kernel.push_bytes = push.size;
  kernel.bindings = 1;  // the storage buffers among them
  VKC_ASSIGN(kernel.layout,
             core::DescriptorSetLayout::create(device.handle(), bindings, 3));
  // The module is transient: the pipeline does not retain it. SPIR-V is a
  // stream of words, and vkc_embed_shaders aligns its arrays to 4.
  VKC_ASSIGN(
      const core::ShaderModule module,
      core::ShaderModule::create(device.handle(),
                                 // NOLINTNEXTLINE(*-reinterpret-cast)
                                 reinterpret_cast<const std::uint32_t*>(
                                     vs_undistort_color_images_comp_spv),
                                 vs_undistort_color_images_comp_spv_size));
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
  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
  VKC_ASSIGN(core::DescriptorPool made,
             core::DescriptorPool::create(device.handle(), sizes, 2, 1));
  VKC_ASSIGN(kernel.set, made.allocate(layout));
  device.set_object_name(VK_OBJECT_TYPE_PIPELINE,
                         core::debug_object_handle(kernel.pipeline.handle()),
                         kernel.name);
  out = std::move(kernel);
  pool = std::move(made);
  return {};
}

}  // namespace

core::Result<GpuFramePrep> GpuFramePrep::create(
    core::Device& device, core::Allocator& allocator,
    const GpuFramePrepConfig& config) {
  VKC_TRY(check_device_requirements(device, "GpuFramePrep::create"));
  // Here rather than at the first output, where it would surface as a buffer
  // failure on the first frame, naming neither the pass nor the output -- or,
  // for colour, not until a frame carries some.
  VKC_TRY(check_queue_families(device, config.color_queue_families,
                               config.color_queue_family_count,
                               "GpuFramePrep::create, colour"));
  VKC_TRY(check_queue_families(device, config.depth_queue_families,
                               config.depth_queue_family_count,
                               "GpuFramePrep::create, depth"));
  // A NaN would reach every return, and the shader's max of one is undefined.
  if (!finite(config.depth_offset_m)) {
    return core::Status::invalid_argument(
        "GpuFramePrep::create: depth_offset_m is not finite");
  }
  GpuFramePrep prep;
  prep.device_ = &device;
  prep.allocator_ = &allocator;
  prep.config_ = config;

  // Storage bindings, input then output, and the kernel's params as push
  // constants.
  VkPushConstantRange depth_push{};
  depth_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  depth_push.size = sizeof(DepthParams);
  VkPushConstantRange color_push = depth_push;
  color_push.size = sizeof(detail::ColorParams);
  core::KernelSetBuilder kb(device);
  // Depth: raw in, depth out, and the overlap it masks by.
  VKC_TRY(kb.add(prep.depth_kernel_, "undistort_depth",
                 vs_undistort_depth_comp_spv, vs_undistort_depth_comp_spv_size,
                 3, &depth_push));
  VKC_TRY(kb.add(prep.color_kernel_, "undistort_color",
                 vs_undistort_color_comp_spv, vs_undistort_color_comp_spv_size,
                 2, &color_push));
  VKC_ASSIGN(prep.pool_, kb.build());
  // Colour from NV12's plane images, sampled where they are.
  VKC_ASSIGN(prep.sampler_, make_sampler(device));
  const core::Status images = build_images_kernel(
      device, color_push, prep.color_images_kernel_, prep.images_pool_);
  if (!images.ok()) return images.with_context("undistort_color_images");

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  prep.max_workgroup_count_x_ = props.limits.maxComputeWorkGroupCount[0];
  prep.max_storage_buffer_range_ = props.limits.maxStorageBufferRange;
  prep.min_storage_buffer_offset_alignment_ =
      props.limits.minStorageBufferOffsetAlignment;
  VKC_ASSIGN(prep.gpu_timer_, core::GpuTimer::create(device));
  // Always bound, so the set is complete whether or not the mask runs; the
  // kernel reads it only when it does.
  VKC_ASSIGN(prep.overlap_,
             core::device_storage_buffer(allocator, sizeof(OverlapParams)));
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(prep.overlap_.handle()),
                         "sensor.depth_overlap");
  prep.depth_kernel_.set.write_storage_buffer(2, prep.overlap_.handle(), 0,
                                              VK_WHOLE_SIZE);
  return prep;
}

// A checked frame: its two halves' layouts, where its colour comes from, and
// its bytes in the slot.
struct GpuFramePrep::Layout {
  DepthLayout depth;
  ColorLayout color;
  // NV12 plane images, sampled in place; host bytes, packed words or planes,
  // copied into the slot after any host depth; else planes in a buffer, read
  // in place.
  bool from_images = false;
  bool from_host = false;
  // The slot holds host depth from byte 0, and host colour from this offset,
  // at minStorageBufferOffsetAlignment; its bytes used, one word at least.
  VkDeviceSize color_at = 0;
  VkDeviceSize slot_bytes = 0;
};

// An output the fusion tiers read, and its lease: held while a DeviceFrame
// shares the buffer. The frame's deleter clears it with release order and
// ensure_output reads it with acquire order, so whatever a holder did with
// the buffer on any thread happens before the pass writes it again.
struct GpuFramePrep::Output {
  core::Buffer buffer;
  std::atomic<bool> held{false};
};

core::Result<GpuFramePrep::Layout> GpuFramePrep::check(
    const RgbdFrame& frame) const {
  if (!valid()) {
    return core::Status::invalid_argument("GpuFramePrep: moved-from pass");
  }
  const std::uint64_t max_pixels =
      std::uint64_t{max_workgroup_count_x_} * kLocalSize;
  Layout layout;
  VKC_ASSIGN(layout.depth,
             check_depth(frame, max_pixels, max_storage_buffer_range_,
                         min_storage_buffer_offset_alignment_));
  if (frame.has_color()) {
    VKC_ASSIGN(layout.color,
               check_color(frame, max_pixels, max_storage_buffer_range_,
                           min_storage_buffer_offset_alignment_));
  }
  layout.from_images =
      frame.color.image[0] != nullptr || frame.color.image[1] != nullptr;
  layout.from_host =
      frame.color_packed != nullptr || frame.color.host != nullptr;
  // Device depth and planes come from a family the batch can take them over
  // from, so the acquire refuses nothing once recording starts. An EXCLUSIVE
  // image crosses only from outside Vulkan (CommandBatch::acquire); one
  // another family reads is made CONCURRENT.
  const auto has_family = [this](std::uint32_t family) {
    return family == kQueueFamilyIgnored || family == kQueueFamilyExternal ||
           family < device_->caps().queue_families().size();
  };
  if (layout.depth.on_device && !has_family(frame.depth_device.queue_family)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth comes from queue family " +
        std::to_string(frame.depth_device.queue_family) +
        ", which the device does not have");
  }
  if (frame.color.device != nullptr || layout.from_images) {
    const std::uint32_t family = frame.color.queue_family;
    const bool special =
        family == kQueueFamilyIgnored || family == kQueueFamilyExternal;
    if (!has_family(family)) {
      return core::Status::invalid_argument(
          "GpuFramePrep: the colour planes come from queue family " +
          std::to_string(family) + ", which the device does not have");
    }
    if (layout.from_images && !special && family != device_->queue_family()) {
      for (const std::shared_ptr<const core::Image>& plane :
           frame.color.image) {
        if (plane->sharing_mode() != VK_SHARING_MODE_CONCURRENT) {
          return core::Status::invalid_argument(
              "GpuFramePrep: the colour images come from queue family " +
              std::to_string(family) +
              ", and EXCLUSIVE images cross only from outside Vulkan");
        }
      }
    }
  }
  layout.slot_bytes = layout.depth.staged_bytes;
  if (layout.from_host) {
    layout.color_at = round_up(
        layout.depth.staged_bytes,
        std::max<VkDeviceSize>(min_storage_buffer_offset_alignment_, 4));
    layout.slot_bytes = layout.color_at + layout.color.planes.bind_bytes;
  }
  // A frame with nothing on the host still claims a word of the slot and its
  // batch binds it, so its batch holds the slot as any frame's does, and a
  // record before that batch resolves is refused (stage) rather than
  // rewriting the sets and outputs it uses.
  layout.slot_bytes = std::max<VkDeviceSize>(layout.slot_bytes, 4);
  return layout;
}

core::Result<core::HostSlot> GpuFramePrep::stage(const RgbdFrame& frame,
                                                 const Layout& layout) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  // One slot: the frame is copied in as it is recorded, so the batch of the
  // pass's last record, which binds the slot whatever the frame staged, has
  // resolved by then (its kernels' sets are rewritten too). A record before
  // it has refuses here, before ensure_buffers can free a buffer that batch
  // still uses.
  if (inputs_ == nullptr) {
    core::HostRingDesc desc;
    desc.slot_bytes = layout.slot_bytes;
    desc.slots = 1;
    desc.name = "sensor.raw_frame";
    desc.host_access = core::HostAccess::SequentialWrite;
    VKC_ASSIGN(inputs_, core::HostRing::create(*device_, *allocator_, desc));
  }
  VKC_ASSIGN(core::HostSlot slot, inputs_->claim(layout.slot_bytes));
  // Host depth, then any host colour at its aligned offset.
  if (!depth.on_device) {
    std::memcpy(slot.data(), frame.depth,
                std::size_t{depth.pixels} * sizeof(std::uint16_t));
  }
  if (layout.from_host) {
    const void* from = frame.color_packed != nullptr
                           ? static_cast<const void*>(frame.color_packed)
                           : static_cast<const void*>(frame.color.host +
                                                      color.planes.bind_offset);
    std::memcpy(slot.data() + layout.color_at, from,
                static_cast<std::size_t>(color.planes.copy_bytes));
  }
  return slot;
}

core::Status GpuFramePrep::ensure_buffers(const RgbdFrame& frame,
                                          const Layout& layout) {
  VKC_TRY(ensure_output(depth_out_, layout.depth.out_bytes,
                        "sensor.depth_frame", config_.depth_queue_families,
                        config_.depth_queue_family_count));
  if (!frame.has_color()) return {};
  return ensure_output(color_out_, layout.color.out_bytes, "sensor.color_frame",
                       config_.color_queue_families,
                       config_.color_queue_family_count);
}

core::Status GpuFramePrep::record_inputs(core::CommandBatch& batch,
                                         const RgbdFrame& frame,
                                         const Layout& layout,
                                         const core::HostSlot& slot,
                                         VkBuffer* input) {
  const YuvImage& image = frame.color;
  // Plane images taken over from the family that wrote them, in the layout
  // they are in. Buffers, depth's and planes', record_frames has taken over.
  if (layout.from_images) {
    for (const std::shared_ptr<const core::Image>& plane : image.image) {
      VKC_TRY(batch.acquire(*plane, image.queue_family, plane->layout()));
    }
  }
  // The slot's used range: bound in place, or one copy into its mirror, which
  // joins the open tile, the frame prep's. Bound even when it holds only the
  // word a frame with nothing on the host claims, so the batch holds it.
  VKC_ASSIGN(*input, batch.input(slot, 0, layout.slot_bytes));
  // The overlap mask, for a frame with colour when the config asks: the
  // colour camera and the depth-to-colour transform, written inline.
  if (config_.depth_within_color && frame.has_color()) {
    const OverlapParams overlap{lens_params(frame.color_camera),
                                Mat4f(frame.depth_to_color)};
    VKC_TRY(batch.upload(overlap_, 0, &overlap, sizeof(overlap)));
  }
  return {};
}

core::Status GpuFramePrep::record_passes(core::CommandBatch& batch,
                                         const RgbdFrame& frame,
                                         const Layout& layout, VkBuffer input,
                                         core::GpuStageScope* stage) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  const YuvImage& image = frame.color;
  // Host depth where the slot holds it; depth on the device where its writer
  // left it.
  const VkBuffer raw =
      depth.on_device ? frame.depth_device.buffer->handle() : input;
  depth_kernel_.set.write_storage_buffer(0, raw, depth.bind_offset,
                                         depth.bind_bytes);
  depth_kernel_.set.write_storage_buffer(1, depth_out_->buffer.handle(), 0,
                                         depth.out_bytes);
  const bool within_color = config_.depth_within_color && frame.has_color();
  const DepthParams depth_params{lens_params(frame.depth_camera),
                                 frame.metres_per_unit, within_color ? 1u : 0u,
                                 config_.depth_offset_m, depth.first};
  VKC_TRY(batch.dispatch(depth_kernel_, &depth_params, sizeof(depth_params),
                         core::group_count(depth.pixels, kLocalSize),
                         max_workgroup_count_x_, stage));
  if (frame.has_color()) {
    // Plane images sampled where they are, in the layout they are in; host
    // words or planes where the slot holds them; planes in a buffer where the
    // decoder left them.
    const core::ComputeKernel& kernel =
        layout.from_images ? color_images_kernel_ : color_kernel_;
    if (layout.from_images) {
      for (std::uint32_t p = 0; p < 2; ++p) {
        kernel.set.write_combined_image_sampler(p, image.image[p]->view(),
                                                sampler_.get(),
                                                image.image[p]->layout());
      }
      kernel.set.write_storage_buffer(2, color_out_->buffer.handle(), 0,
                                      color.out_bytes);
    } else {
      const VkBuffer planes = layout.from_host ? input : image.device->handle();
      const VkDeviceSize at =
          layout.from_host ? layout.color_at : color.planes.bind_offset;
      kernel.set.write_storage_buffer(0, planes, at, color.planes.bind_bytes);
      kernel.set.write_storage_buffer(1, color_out_->buffer.handle(), 0,
                                      color.out_bytes);
    }
    const detail::ColorParams color_params =
        detail::color_params(lens_params(frame.color_camera), image,
                             color.planes, frame.color_packed != nullptr);
    VKC_TRY(batch.dispatch(kernel, &color_params, sizeof(color_params),
                           core::group_count(color.pixels, kLocalSize),
                           max_workgroup_count_x_, stage));
  }
  return {};
}

void GpuFramePrep::abandon(const RgbdFrame& frame) {
  // The device may still run the kernels, so the device depth and planes,
  // buffers or images, which the caller may drop, and a writer reuse, as soon
  // as the call returns, are let go rather than freed. The slot stays with
  // the batch, which the device keeps.
  if (frame.depth_device.buffer != nullptr) {
    static_cast<void>(
        new std::shared_ptr<const core::Buffer>(frame.depth_device.buffer));
  }
  if (frame.color.device != nullptr) {
    static_cast<void>(
        new std::shared_ptr<const core::Buffer>(frame.color.device));
  }
  for (const std::shared_ptr<const core::Image>& plane : frame.color.image) {
    if (plane != nullptr) {
      static_cast<void>(new std::shared_ptr<const core::Image>(plane));
    }
  }
}

DeviceFrame GpuFramePrep::finish(const RgbdFrame& frame) const {
  // An output's buffer for the frame, its lease held until every copy goes.
  const auto lease = [](const std::shared_ptr<Output>& output) {
    output->held.store(true, std::memory_order_relaxed);
    return std::shared_ptr<const core::Buffer>(
        &output->buffer, [keep = output](const core::Buffer*) {
          keep->held.store(false, std::memory_order_release);
        });
  };
  // The pinhole cameras of the undistorted images, narrowed as the passes
  // read them, each posed: the depth camera through the sensor's extrinsic.
  const LensParams d = lens_params(frame.depth_camera);
  DeviceFrame out;
  out.depth = lease(depth_out_);
  out.depth_camera.fx = d.fx;
  out.depth_camera.fy = d.fy;
  out.depth_camera.cx = d.cx;
  out.depth_camera.cy = d.cy;
  out.depth_camera.min_depth = frame.min_depth;
  out.depth_camera.max_depth = frame.max_depth;
  out.depth_camera.width = d.width;
  out.depth_camera.height = d.height;
  out.depth_camera.cam_to_world =
      Mat4f(frame.color_to_world * frame.depth_to_color);
  if (frame.has_color()) {
    const LensParams c = lens_params(frame.color_camera);
    out.color = lease(color_out_);
    out.color_camera = ColorCameraParams{
        c.fx, c.fy, c.cx, c.cy, c.width, c.height, Mat4f(frame.color_to_world)};
    out.color_encoding = frame.color_encoding;
  }
  out.timestamp_ns = frame.timestamp_ns;
  return out;
}

namespace {

// A set's frames as pointers, a missing one null; refused with fewer passes
// than frames, naming `op`.
template <typename Pass>
core::Result<std::vector<const RgbdFrame*>> listed_frames(
    const char* op, const std::vector<Pass>& preps,
    const std::vector<std::optional<RgbdFrame>>& frames) {
  if (preps.size() < frames.size()) {
    return core::Status::invalid_argument(
        std::string("GpuFramePrep::") + op + ": " +
        std::to_string(frames.size()) + " frames for " +
        std::to_string(preps.size()) + " passes");
  }
  std::vector<const RgbdFrame*> listed(frames.size(), nullptr);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames[i]) listed[i] = &*frames[i];
  }
  return listed;
}

// The family `buffer`, named as written on `family`, is taken over from, as
// CommandBatch::acquire decides: kQueueFamilyIgnored where nothing is
// transferred -- written on `own`, the pass's family, or a CONCURRENT buffer
// another Vulkan family wrote.
std::uint32_t transfer_family(const core::Buffer& buffer, std::uint32_t family,
                              std::uint32_t own) noexcept {
  if (family == own) return kQueueFamilyIgnored;
  if (family != kQueueFamilyExternal &&
      buffer.sharing_mode() == VK_SHARING_MODE_CONCURRENT) {
    return kQueueFamilyIgnored;
  }
  return family;
}

// A transfer_family for a message.
std::string family_name(std::uint32_t family) {
  if (family == kQueueFamilyIgnored) return "the pass's own";
  if (family == kQueueFamilyExternal) return "kQueueFamilyExternal";
  return std::to_string(family);
}

}  // namespace

core::Result<DeviceFrame> GpuFramePrep::record(
    core::CommandBatch& batch, const RgbdFrame& frame,
    core::StageMetrics* metrics) try {
  // A set of one, through record_set's path.
  const RgbdFrame* const frames[] = {&frame};
  Layout layout;
  std::vector<Acquire> acquires;
  VKC_ASSIGN(GpuFramePrep* const first,
             check_frames(this, frames, &layout, 1, &acquires));
  VKC_TRY(gpu_timer_.reserve(*device_, kSpansPerFrame));
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  std::optional<DeviceFrame> out;
  VKC_TRY(
      record_frames(batch, this, frames, &layout, acquires, &out, 1, &stage));
  return std::move(*out);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("GpuFramePrep: out of host memory");
}

core::Result<std::vector<std::optional<DeviceFrame>>> GpuFramePrep::record_set(
    core::CommandBatch& batch, std::vector<GpuFramePrep>& preps,
    const std::vector<std::optional<RgbdFrame>>& frames,
    core::StageMetrics* metrics) try {
  VKC_ASSIGN(const std::vector<const RgbdFrame*> listed,
             listed_frames("record_set", preps, frames));
  std::vector<Layout> layouts(frames.size());
  std::vector<std::optional<DeviceFrame>> out(frames.size());
  std::vector<Acquire> acquires;
  VKC_ASSIGN(GpuFramePrep* const first,
             check_frames(preps.data(), listed.data(), layouts.data(),
                          frames.size(), &acquires));
  if (first == nullptr) return out;
  // The set's timed commands all in one window, which grows with the array.
  VKC_TRY(first->gpu_timer_.reserve(
      *first->device_,
      kSpansPerFrame * static_cast<std::uint32_t>(frames.size())));
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  VKC_TRY(record_frames(batch, preps.data(), listed.data(), layouts.data(),
                        acquires, out.data(), frames.size(), &stage));
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("GpuFramePrep: out of host memory");
}

core::Result<DeviceFrame> GpuFramePrep::prepare(const RgbdFrame& frame,
                                                core::StageMetrics* metrics) {
  // A set of one, through prepare_batch's path.
  const RgbdFrame* const frames[] = {&frame};
  Layout layout;
  std::optional<DeviceFrame> out;
  VKC_TRY(prepare_frames(this, frames, &layout, &out, 1, metrics));
  return std::move(*out);
}

core::Result<std::vector<std::optional<DeviceFrame>>>
GpuFramePrep::prepare_batch(std::vector<GpuFramePrep>& preps,
                            const std::vector<std::optional<RgbdFrame>>& frames,
                            core::StageMetrics* metrics) {
  VKC_ASSIGN(const std::vector<const RgbdFrame*> listed,
             listed_frames("prepare_batch", preps, frames));
  std::vector<Layout> layouts(frames.size());
  std::vector<std::optional<DeviceFrame>> out(frames.size());
  VKC_TRY(prepare_frames(preps.data(), listed.data(), layouts.data(),
                         out.data(), frames.size(), metrics));
  return out;
}

core::Result<GpuFramePrep*> GpuFramePrep::check_frames(
    GpuFramePrep* preps, const RgbdFrame* const* frames, Layout* layouts,
    std::size_t count, std::vector<Acquire>* acquires) {
  // Every frame checked before any is recorded, so a refused set costs no
  // work and times no row; and every pass on the first one's device, which
  // runs the batch.
  GpuFramePrep* first = nullptr;
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_ASSIGN(layouts[i], preps[i].check(*frames[i]));
    if (first == nullptr) {
      first = &preps[i];
    } else if (preps[i].device_ != first->device_) {
      return core::Status::invalid_argument(
          "GpuFramePrep::prepare_batch: the passes are on different devices");
    }
  }
  // Each buffer the frames read on the device is taken over once, however
  // many depths and pictures name it -- a frame's depth and planes in one
  // buffer, or every camera's depth -- since its writer released it once; so
  // every reference to it must name the same writer.
  acquires->clear();
  if (first == nullptr) return first;
  const std::uint32_t own = first->device_->queue_family();
  std::vector<Acquire> named;  // every distinct buffer, transferred or not
  const auto name = [&](const core::Buffer& buffer,
                        std::uint32_t family) -> core::Status {
    const std::uint32_t from = transfer_family(buffer, family, own);
    for (const Acquire& seen : named) {
      if (seen.first->handle() != buffer.handle()) continue;
      if (seen.second == from) return {};
      return core::Status::invalid_argument(
          "GpuFramePrep: one buffer is named with two writers' queue "
          "families, " +
          family_name(seen.second) + " and " + family_name(from) +
          "; every depth and picture in it must name the one that wrote it");
    }
    named.emplace_back(&buffer, from);
    return {};
  };
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    const RgbdFrame& frame = *frames[i];
    if (frame.depth_device.buffer != nullptr) {
      VKC_TRY(
          name(*frame.depth_device.buffer, frame.depth_device.queue_family));
    }
    if (frame.color.device != nullptr) {
      VKC_TRY(name(*frame.color.device, frame.color.queue_family));
    }
  }
  for (const Acquire& buffer : named) {
    if (buffer.second != kQueueFamilyIgnored) acquires->push_back(buffer);
  }
  return first;
}

core::Status GpuFramePrep::record_frames(
    core::CommandBatch& batch, GpuFramePrep* preps,
    const RgbdFrame* const* frames, const Layout* layouts,
    const std::vector<Acquire>& acquires, std::optional<DeviceFrame>* out,
    std::size_t count, core::GpuStageScope* stage) {
  // Every frame's slot claimed and filled before anything is recorded or
  // resized, so a slot held refuses with the batch and every pass as they
  // were; a slot claimed before a refusal goes back free. A claimed slot
  // says its pass's last batch has resolved, so then its buffers may grow.
  std::vector<core::HostSlot> slots(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_ASSIGN(slots[i], preps[i].stage(*frames[i], layouts[i]));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_TRY(preps[i].ensure_buffers(*frames[i], layouts[i]));
  }
  // The device depth and planes' buffers taken over from their writers, each
  // once; then every camera's inputs, then every camera's passes: the slots'
  // copies write buffers of their own, so they run with no barrier between
  // them.
  for (const Acquire& buffer : acquires) {
    VKC_TRY(batch.acquire(*buffer.first, buffer.second));
  }
  std::vector<VkBuffer> inputs(count, VK_NULL_HANDLE);
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_TRY(preps[i].record_inputs(batch, *frames[i], layouts[i], slots[i],
                                   &inputs[i]));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_TRY(preps[i].record_passes(batch, *frames[i], layouts[i], inputs[i],
                                   stage));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) out[i] = preps[i].finish(*frames[i]);
  }
  return {};
}

core::Status GpuFramePrep::prepare_frames(GpuFramePrep* preps,
                                          const RgbdFrame* const* frames,
                                          Layout* layouts,
                                          std::optional<DeviceFrame>* out,
                                          std::size_t count,
                                          core::StageMetrics* metrics) try {
  std::vector<Acquire> acquires;
  VKC_ASSIGN(GpuFramePrep* const first,
             check_frames(preps, frames, layouts, count, &acquires));
  if (first == nullptr) return {};
  VKC_TRY(first->gpu_timer_.reserve(
      *first->device_, kSpansPerFrame * static_cast<std::uint32_t>(count)));
  // The row spans the wait too, and takes the batch's device half after it.
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  core::CommandBatch batch(*first->device_, *first->allocator_);
  VKC_TRY(record_frames(batch, preps, frames, layouts, acquires, out, count,
                        &stage));
  core::Result<core::PendingBatch> pending = batch.submit_async({}, {});
  const core::Status ran = pending.ok() ? pending->wait() : pending.status();
  if (!ran.ok()) {
    // A submit that loses the device may still be pending, as may a failed
    // wait. The core keeps the command resources in both cases, but the
    // external pictures read by those commands belong to the frame.
    if ((pending.ok() && pending->in_flight()) ||
        (!pending.ok() && core::vk_result(ran) == VK_ERROR_DEVICE_LOST)) {
      for (std::size_t i = 0; i < count; ++i) {
        if (frames[i] != nullptr) abandon(*frames[i]);
      }
    }
    for (std::size_t i = 0; i < count; ++i) out[i].reset();
    return ran;
  }
  if (metrics != nullptr) metrics->merge_device(pending->device_rows());
  return {};
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("GpuFramePrep: out of host memory");
}

core::Status GpuFramePrep::ensure_output(std::shared_ptr<Output>& output,
                                         VkDeviceSize bytes, const char* name,
                                         const std::uint32_t* families,
                                         std::uint32_t family_count) {
  // Reused only once no DeviceFrame holds its lease: a frame kept past this
  // call keeps its contents, and this frame goes to a new buffer, which
  // measured no slower than reusing one (the residency decision's step 5b),
  // so there is no ring.
  if (output != nullptr && !output->held.load(std::memory_order_acquire) &&
      output->buffer.size() >= bytes) {
    return {};
  }
  // Device-local: only the kernels touch it, and on a discrete GPU the
  // fusion kernels' reads would otherwise cross the bus. Shared with the
  // families the config names for it, for a consumer on another queue, with
  // no usage beyond the pass's storage and transfer usage. The host's half
  // first, so running out of host memory leaves no buffer made.
  auto made = std::make_shared<Output>();
  VKC_ASSIGN(made->buffer, core::device_storage_buffer(*allocator_, bytes, 0,
                                                       families, family_count));
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           core::debug_object_handle(made->buffer.handle()),
                           name);
  output = std::move(made);
  return {};
}

}  // namespace volumetric_kit::sensor
