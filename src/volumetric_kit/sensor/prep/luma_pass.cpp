// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/prep/luma_pass.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include "../sharing_families.hpp"
#include "color_pass.hpp"
#include "luma_comp.spv.hpp"
#include "luma_half_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/frame/ready_points.hpp"
#include "yuv_planes.hpp"

namespace volumetric_kit::sensor {
namespace {

// Both kernels' local_size_x.
constexpr std::uint32_t kLocalSize = 256;

// The commands a picture times: its full luma (a dispatch, or its luma
// image's copy) and its half.
constexpr std::uint32_t kSpansPerPicture = 2;

// luma_half.comp's Params, under scalar layout: the full luma's size.
struct HalfParams {
  std::uint32_t width;
  std::uint32_t height;
};
static_assert(sizeof(HalfParams) == 8, "HalfParams must match luma_half.comp");

VkDeviceSize round_up(VkDeviceSize bytes, VkDeviceSize alignment) noexcept {
  return (bytes + alignment - 1) / alignment * alignment;
}

// A pooled output, and its lease: held while a DeviceLuma shares the buffer.
// The lease's deleter clears it with release order and the pass reads it with
// acquire order, so whatever a holder did with the buffer on any thread
// happens before the pass writes it again; the holder's device reads have
// completed by then (DeviceLuma::owner's contract).
struct Output {
  core::Buffer buffer;
  std::atomic<bool> held{false};
};

// The picture as the luma kernel binds it: its luma plane alone, so a 4:2:0
// picture's chroma is neither bound nor staged, and no matrix or chroma
// siting, which luma does not read, to be checked. YUYV's and V210's one
// plane, and luma alone, keep their layout.
YuvImage luma_view(const YuvImage& picture) {
  YuvImage view = picture;
  if (picture.layout == YuvLayout::Nv12 || picture.layout == YuvLayout::I420) {
    view.layout = YuvLayout::Luma;
    for (int p = 1; p < 3; ++p) {
      view.offset[p] = 0;
      view.stride[p] = 0;
    }
  }
  const YcbcrWeights bt709 = ycbcr_weights(VideoColorMatrix::Bt709);
  view.kr = bt709.kr;
  view.kb = bt709.kb;
  view.chroma_location = ChromaLocation::Left;
  return view;
}

// One picture, checked: where its luma comes from, how it is bound, and
// where its outputs lie in their buffer.
struct Plan {
  bool from_images = false;  // NV12 plane images: the luma image copied
  bool from_host = false;    // host planes, staged into the slot
  YuvImage view;             // the luma kernel's view (luma_view)
  detail::PlaneBinding binding;
  VkDeviceSize slot_at = 0;  // host planes' first byte in the slot
  std::uint32_t pixels = 0;
  std::uint32_t half_pixels = 0;
  VkDeviceSize full_bytes = 0;  // to a word
  VkDeviceSize half_at = 0;     // at a binding's alignment
  VkDeviceSize half_bytes = 0;  // to a word
  VkDeviceSize total = 0;
  std::shared_ptr<Output> out;
};

// The picture as one of the pass's outputs: luma alone in `lease`'s buffer
// from `at`, rows packed, its range `full_range`.
YuvImage luma_picture(const std::shared_ptr<const core::Buffer>& lease,
                      VkDeviceSize at, std::uint32_t width,
                      std::uint32_t height, bool full_range) {
  YuvImage out;
  out.layout = YuvLayout::Luma;
  out.device = lease;
  out.offset[0] = at;
  out.stride[0] = width;
  out.width = width;
  out.height = height;
  out.full_range = full_range;
  out.queue_family = kQueueFamilyIgnored;
  return out;
}

bool concurrent(VkSharingMode mode) noexcept {
  return mode == VK_SHARING_MODE_CONCURRENT;
}

}  // namespace

struct LumaPass::Impl {
  const core::Device* device = nullptr;
  core::Allocator* allocator = nullptr;
  // The outputs' families: distinct, two or more, or none for EXCLUSIVE.
  std::vector<std::uint32_t> families;

  std::uint32_t max_groups = 0;
  VkDeviceSize max_range = 0;
  VkDeviceSize align = 4;  // a binding's offset alignment, and a word

  core::ComputeKernel luma_kernel;
  core::ComputeKernel half_kernel;
  core::DescriptorPool pool;
  // A set per dispatch of a run: a batch binds its sets at the submit.
  core::KernelSets luma_sets;
  core::KernelSets half_sets;
  core::GpuTimer timer;
  // Host pictures copied in, one after another; made at the first that has
  // one, and free again once each run's batch has resolved.
  std::unique_ptr<core::HostRing> inputs;
  // The outputs, leased to the DeviceLumas handed out.
  std::vector<std::shared_ptr<Output>> outputs;
  // A submit or wait that failed: the device may still run that batch.
  core::Status failed;

  core::Result<Plan> check(const YuvImage* picture, std::size_t index) const;
  // `plans[i]` given an output that holds it: the smallest free one that
  // does, else a free one replaced, else a new one.
  core::Status take_outputs(std::vector<Plan>& plans);
  core::Result<std::vector<DeviceLuma>> run(
      const std::vector<YuvImage*>& pictures, core::StageMetrics* metrics);
};

LumaPass::LumaPass(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

LumaPass::~LumaPass() = default;

core::Result<std::unique_ptr<LumaPass>> LumaPass::create(
    const core::Device& device, core::Allocator& allocator,
    const std::uint32_t* queue_families, std::uint32_t queue_family_count) try {
  if (device.handle() == VK_NULL_HANDLE || !allocator.valid()) {
    return core::Status::invalid_argument(
        "LumaPass::create: the device or allocator is empty");
  }
  VKC_TRY(check_device_requirements(device, "LumaPass::create"));
  auto impl = std::make_unique<Impl>();
  impl->device = &device;
  impl->allocator = &allocator;
  VKC_ASSIGN(impl->families,
             detail::sharing_families(device, queue_families,
                                      queue_family_count, "LumaPass::create"));

  // The luma kernel: the planes, then the full luma, and the picture's
  // coding as the colour pass's push constants. The half kernel: the full
  // luma, then the half, and the full luma's size.
  VkPushConstantRange luma_push{};
  luma_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  luma_push.size = sizeof(detail::ColorParams);
  VkPushConstantRange half_push = luma_push;
  half_push.size = sizeof(HalfParams);
  core::KernelSetBuilder builder(device);
  VKC_TRY(builder.add(impl->luma_kernel, "luma", vs_luma_comp_spv,
                      vs_luma_comp_spv_size, 2, &luma_push));
  VKC_TRY(builder.add(impl->half_kernel, "luma_half", vs_luma_half_comp_spv,
                      vs_luma_half_comp_spv_size, 2, &half_push));
  VKC_ASSIGN(impl->pool, builder.build());

  const VkPhysicalDeviceLimits& limits = device.caps().limits();
  impl->max_groups = limits.maxComputeWorkGroupCount[0];
  impl->max_range = core::max_storage_buffer_range(device);
  impl->align =
      std::max<VkDeviceSize>(limits.minStorageBufferOffsetAlignment, 4);
  VKC_ASSIGN(impl->timer, core::GpuTimer::create(device));
  return std::unique_ptr<LumaPass>(new LumaPass(std::move(impl)));
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("LumaPass::create: out of host memory");
}

core::Result<Plan> LumaPass::Impl::check(const YuvImage* picture,
                                         std::size_t index) const {
  const std::string who = "LumaPass, picture " + std::to_string(index);
  const auto refuse = [&who](const std::string& why) {
    return core::Status::invalid_argument(who + ": " + why);
  };
  if (picture == nullptr) return refuse("is null");
  const YuvImage& p = *picture;
  if (p.width < 2 || p.height < 2) {
    return refuse("is " + std::to_string(p.width) + "x" +
                  std::to_string(p.height) +
                  ", under 2x2, so its half would be empty");
  }
  Plan plan;
  plan.from_images = p.image[0] != nullptr || p.image[1] != nullptr;
  plan.from_host = p.host != nullptr;
  if (!plan.from_images && !plan.from_host && p.device == nullptr) {
    return refuse("has no planes");
  }
  if (plan.from_images) {
    // NV12's two images, copied from: the luma image is the full luma.
    VKC_TRY(detail::check_yuv_planes(p, who.c_str(), detail::ImageRead::Copy)
                .status());
  } else {
    // The luma plane as the kernel binds it (its buffer's usage first), then
    // the whole picture as it was handed out.
    plan.view = luma_view(p);
    VKC_ASSIGN(plan.binding,
               detail::bind_planes(plan.view, {p.width, p.height}, who.c_str(),
                                   "input", max_range, align));
    VKC_TRY(detail::check_yuv_planes(p, who.c_str(), detail::ImageRead::Sample)
                .status());
  }
  // Device planes come from a family the batch can take them over from, so
  // nothing is refused once recording starts. An EXCLUSIVE image crosses only
  // from outside Vulkan (CommandBatch::acquire).
  if (!plan.from_host) {
    const std::uint32_t family = p.queue_family;
    const bool special =
        family == kQueueFamilyIgnored || family == kQueueFamilyExternal;
    if (!special && family >= device->caps().queue_families().size()) {
      return refuse("comes from queue family " + std::to_string(family) +
                    ", which the device does not have");
    }
    if (plan.from_images && !special && family != device->queue_family() &&
        !(concurrent(p.image[0]->sharing_mode()) &&
          concurrent(p.image[1]->sharing_mode()))) {
      return refuse("is EXCLUSIVE images from queue family " +
                    std::to_string(family) +
                    ", and EXCLUSIVE images cross only from outside Vulkan");
    }
  }
  // The kernels count pixels, and each word's first of four, in 32 bits.
  const std::uint64_t pixels = std::uint64_t{p.width} * p.height;
  if (pixels > std::numeric_limits<std::uint32_t>::max() - 3 ||
      (pixels + 3) / 4 > std::uint64_t{max_groups} * kLocalSize) {
    return refuse("is past a single dispatch");
  }
  plan.pixels = static_cast<std::uint32_t>(pixels);
  plan.half_pixels = (p.width / 2) * (p.height / 2);
  plan.full_bytes = round_up(pixels, 4);
  VKC_TRY(core::check_storage_buffer_range((who + ": the full luma").c_str(),
                                           plan.full_bytes, max_range));
  plan.half_at = round_up(plan.full_bytes, align);
  plan.half_bytes = round_up(plan.half_pixels, 4);
  plan.total = plan.half_at + plan.half_bytes;
  return plan;
}

core::Status LumaPass::Impl::take_outputs(std::vector<Plan>& plans) {
  std::vector<bool> taken(outputs.size(), false);
  const auto is_free = [&](std::size_t j) {
    return !taken[j] && (outputs[j] == nullptr ||
                         !outputs[j]->held.load(std::memory_order_acquire));
  };
  const auto size_of = [&](std::size_t j) -> VkDeviceSize {
    return outputs[j] == nullptr ? 0 : outputs[j]->buffer.size();
  };
  for (Plan& plan : plans) {
    constexpr std::size_t kNone = ~std::size_t{0};
    std::size_t fit = kNone;
    std::size_t smallest = kNone;
    for (std::size_t j = 0; j < outputs.size(); ++j) {
      if (!is_free(j)) continue;
      if (smallest == kNone || size_of(j) < size_of(smallest)) smallest = j;
      if (size_of(j) >= plan.total &&
          (fit == kNone || size_of(j) < size_of(fit))) {
        fit = j;
      }
    }
    if (fit == kNone) {
      // None free holds it: the smallest free one goes first, so a grow
      // never holds both, else the pool gains one.
      if (smallest != kNone) {
        fit = smallest;
        outputs[fit].reset();
      } else {
        fit = outputs.size();
        outputs.emplace_back();
        taken.push_back(false);
      }
      // Device-local, shared with the families the pass was made for, with
      // the transfer usage a readback copies it by.
      auto made = std::make_shared<Output>();
      VKC_ASSIGN(made->buffer,
                 core::device_storage_buffer(
                     *allocator, plan.total, 0,
                     families.empty() ? nullptr : families.data(),
                     static_cast<std::uint32_t>(families.size())));
      device->set_object_name(VK_OBJECT_TYPE_BUFFER,
                              core::debug_object_handle(made->buffer.handle()),
                              "sensor.luma");
      outputs[fit] = std::move(made);
    }
    taken[fit] = true;
    plan.out = outputs[fit];
  }
  return {};
}

core::Result<std::vector<DeviceLuma>> LumaPass::run(
    const std::vector<YuvImage*>& pictures, core::StageMetrics* metrics) try {
  return impl_->run(pictures, metrics);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("LumaPass: out of host memory");
}

core::Result<std::vector<DeviceLuma>> LumaPass::Impl::run(
    const std::vector<YuvImage*>& pictures, core::StageMetrics* metrics) {
  if (!failed.ok()) {
    return failed.with_context(
        "LumaPass: an earlier run failed and may still run");
  }
  const std::size_t count = pictures.size();
  std::vector<DeviceLuma> out;
  if (count == 0) return out;

  // Every picture checked before anything is allocated or recorded.
  // What the batch waits for on the device: the values of the pictures'
  // writers that did not wait for their work (YuvImage::ready).
  std::vector<Plan> plans(count);
  std::vector<core::TimelinePoint> ready;
  for (std::size_t i = 0; i < count; ++i) {
    VKC_ASSIGN(plans[i], check(pictures[i], i));
    VKC_TRY(detail::check_ready(*pictures[i], *device, "LumaPass"));
    add_ready_point(*pictures[i], ready);
  }
  // Each buffer the pictures are in is taken over once, however many name
  // it -- a side-by-side pair in one, say -- since its writer released it
  // once; so every picture in it must name the same writer. Images likewise,
  // each once.
  const std::uint32_t own = device->queue_family();
  std::vector<std::pair<const core::Buffer*, std::uint32_t>> buffers;
  std::vector<std::pair<const core::Image*, std::uint32_t>> images;
  for (std::size_t i = 0; i < count; ++i) {
    const YuvImage& p = *pictures[i];
    if (p.device != nullptr) {
      const std::uint32_t from =
          detail::transfer_family(*p.device, p.queue_family, own);
      const auto seen =
          std::find_if(buffers.begin(), buffers.end(), [&p](const auto& b) {
            return b.first->handle() == p.device->handle();
          });
      if (seen == buffers.end()) {
        buffers.emplace_back(p.device.get(), from);
      } else if (seen->second != from) {
        return core::Status::invalid_argument(
            "LumaPass, picture " + std::to_string(i) +
            ": its buffer is taken over once, but the pictures in it name "
            "two writers' queue families, " +
            detail::family_name(seen->second) + " and " +
            detail::family_name(from));
      }
    } else if (plans[i].from_images) {
      for (const std::shared_ptr<const core::Image>& plane : p.image) {
        const std::uint32_t from =
            detail::transfer_family(plane->sharing_mode(), p.queue_family, own);
        const auto seen = std::find_if(
            images.begin(), images.end(), [&plane](const auto& image) {
              return image.first->handle() == plane->handle();
            });
        if (seen == images.end()) {
          images.emplace_back(plane.get(), from);
        } else if (seen->second != from) {
          return core::Status::invalid_argument(
              "LumaPass, picture " + std::to_string(i) +
              ": an image of it is taken over once, but the pictures name "
              "two writers' queue families for it, " +
              detail::family_name(seen->second) + " and " +
              detail::family_name(from));
        }
      }
    }
  }

  // Host pictures copied into the one slot, each from a binding's
  // alignment; the last run's batch, which bound it, has resolved.
  VkDeviceSize slot_bytes = 0;
  std::uint32_t kernel_pictures = 0;
  for (Plan& plan : plans) {
    if (!plan.from_images) ++kernel_pictures;
    if (!plan.from_host) continue;
    plan.slot_at = round_up(slot_bytes, align);
    slot_bytes = plan.slot_at + plan.binding.bind_bytes;
  }
  core::HostSlot slot;
  if (slot_bytes != 0) {
    if (inputs == nullptr) {
      core::HostRingDesc desc;
      desc.slot_bytes = slot_bytes;
      desc.slots = 1;
      desc.name = "sensor.luma_pictures";
      desc.host_access = core::HostAccess::SequentialWrite;
      VKC_ASSIGN(inputs, core::HostRing::create(*device, *allocator, desc));
    }
    VKC_ASSIGN(slot, inputs->claim(slot_bytes));
    for (std::size_t i = 0; i < count; ++i) {
      const Plan& plan = plans[i];
      if (!plan.from_host) continue;
      std::memcpy(slot.data() + plan.slot_at,
                  pictures[i]->host + plan.binding.bind_offset,
                  static_cast<std::size_t>(plan.binding.copy_bytes));
    }
  }
  VKC_TRY(take_outputs(plans));
  if (kernel_pictures != 0) {
    VKC_TRY(luma_sets.reserve(*device, luma_kernel, kernel_pictures));
  }
  VKC_TRY(half_sets.reserve(*device, half_kernel,
                            static_cast<std::uint32_t>(count)));

  if (metrics != nullptr) {
    VKC_TRY(timer.reserve(
        *device, kSpansPerPicture * static_cast<std::uint32_t>(count)));
  }
  core::GpuStageScope stage(metrics, timer, "luma");
  core::CommandBatch batch(*device, *allocator);
  // The pictures' buffers and images taken over from their writers, each
  // once; the outputs, which the kernels write whole, need none.
  for (const auto& buffer : buffers) {
    if (buffer.second != kQueueFamilyIgnored) {
      VKC_TRY(batch.acquire(*buffer.first, buffer.second));
    }
  }
  for (const auto& image : images) {
    VKC_TRY(batch.acquire(*image.first, image.second, image.first->layout()));
  }
  VkBuffer input = VK_NULL_HANDLE;
  if (slot_bytes != 0) {
    VKC_ASSIGN(input, batch.input(slot, 0, slot_bytes));
  }
  // Every picture's full luma, then every half, which reads its full luma:
  // the barrier the batch puts before each dispatch orders each half after
  // the full luma it reads.
  std::uint32_t next_set = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const Plan& plan = plans[i];
    const YuvImage& p = *pictures[i];
    if (plan.from_images) {
      VKC_TRY(batch.copy(*p.image[0], p.width, p.height, plan.out->buffer, 0,
                         &stage));
      continue;
    }
    const core::DescriptorSet& set = luma_sets[next_set++];
    if (plan.from_host) {
      set.write_storage_buffer(0, input, plan.slot_at, plan.binding.bind_bytes);
    } else {
      set.write_storage_buffer(0, p.device->handle(), plan.binding.bind_offset,
                               plan.binding.bind_bytes);
    }
    set.write_storage_buffer(1, plan.out->buffer.handle(), 0, plan.full_bytes);
    // The colour pass's push constants: the planes' places, and the
    // picture's size as its camera's.
    detail::LensParams size{};
    size.width = p.width;
    size.height = p.height;
    const detail::ColorParams params =
        detail::color_params(size, plan.view, plan.binding, false);
    VKC_TRY(batch.dispatch(luma_kernel, set, &params, sizeof(params),
                           core::group_count((plan.pixels + 3) / 4, kLocalSize),
                           max_groups, &stage));
  }
  for (std::size_t i = 0; i < count; ++i) {
    const Plan& plan = plans[i];
    const core::DescriptorSet& set = half_sets[i];
    set.write_storage_buffer(0, plan.out->buffer.handle(), 0, plan.full_bytes);
    set.write_storage_buffer(1, plan.out->buffer.handle(), plan.half_at,
                             plan.half_bytes);
    const HalfParams params{pictures[i]->width, pictures[i]->height};
    VKC_TRY(batch.dispatch(
        half_kernel, set, &params, sizeof(params),
        core::group_count((plan.half_pixels + 3) / 4, kLocalSize), max_groups,
        &stage));
  }
  core::Result<core::PendingBatch> pending = batch.submit_async(ready, {});
  const core::Status submitted =
      pending.ok() ? pending->wait() : pending.status();
  if (!submitted.ok()) {
    // The batch may still run: its outputs and the pictures' buffers and
    // images, which the caller may drop as soon as this returns, are let go
    // rather than freed, and no later run rewrites what it binds.
    for (std::size_t i = 0; i < count; ++i) {
      static_cast<void>(new std::shared_ptr<Output>(plans[i].out));
      static_cast<void>(new YuvImage(*pictures[i]));
    }
    failed = submitted;
    return submitted;
  }
  if (metrics != nullptr) metrics->merge_device(pending->device_rows());

  // The pictures' ownership as the batch left it, published only now: a
  // later stage must not repeat the writer's release and this acquire.
  // Images that needed no transfer keep their kQueueFamilyIgnored.
  for (YuvImage* picture : pictures) {
    YuvImage& p = *picture;
    if (p.device != nullptr) {
      p.queue_family =
          concurrent(p.device->sharing_mode()) ? kQueueFamilyIgnored : own;
    } else if (p.host == nullptr && p.queue_family != kQueueFamilyIgnored) {
      p.queue_family = concurrent(p.image[0]->sharing_mode()) &&
                               concurrent(p.image[1]->sharing_mode())
                           ? kQueueFamilyIgnored
                           : own;
    }
  }
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const Plan& plan = plans[i];
    const YuvImage& p = *pictures[i];
    plan.out->held.store(true, std::memory_order_relaxed);
    // The deleter runs if the lease cannot be made, so a throw frees it.
    const std::shared_ptr<const core::Buffer> lease(
        &plan.out->buffer, [keep = plan.out](const core::Buffer*) {
          keep->held.store(false, std::memory_order_release);
        });
    DeviceLuma luma;
    luma.full = luma_picture(lease, 0, p.width, p.height, p.full_range);
    luma.half = luma_picture(lease, plan.half_at, p.width / 2, p.height / 2,
                             p.full_range);
    luma.owner = lease;
    out.push_back(std::move(luma));
  }
  return out;
}

}  // namespace volumetric_kit::sensor
