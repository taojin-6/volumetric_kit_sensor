// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/testing/pattern_reader.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "../sharing_families.hpp"
#include "pattern_index_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/frame/ready_points.hpp"
#include "volumetric_kit/sensor/testing/index_band.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"
#include "yuv_planes.hpp"

namespace volumetric_kit::sensor::testing {
namespace {

constexpr const char* kWho = "PatternReader";

// The band as pattern_index.comp holds it: 16 squares of 16 x 16, so 256
// samples a row and 16 rows; the picture's luma bytes at 0, the depth's
// 16-bit samples after them, and a result word for each.
static_assert(kIndexSquare == 16 && kIndexBits == 14,
              "pattern_index.comp's band is index_band.hpp's");
constexpr std::uint32_t kBandWidth = kIndexSquare * (2 + kIndexBits);
constexpr std::uint32_t kBandRows = kIndexSquare;
constexpr VkDeviceSize kLumaBytes = VkDeviceSize{kBandWidth} * kBandRows;
constexpr VkDeviceSize kDepthAt = kLumaBytes;
constexpr VkDeviceSize kDepthRowBytes = 2 * VkDeviceSize{kBandWidth};
constexpr VkDeviceSize kDepthBytes = kDepthRowBytes * kBandRows;
constexpr std::uint32_t kNone = 0xFFFFFFFFu;

// pattern_index.comp's Params, under scalar layout.
struct PatternParams {
  std::uint32_t depth;    // 0 reads the picture's band, 1 the depth's
  std::uint32_t near_mm;  // kPatternNearMm
  std::uint32_t far_mm;   // kPatternFarMm
};
static_assert(sizeof(PatternParams) == 12,
              "PatternParams must match pattern_index.comp");

core::Status refuse(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

bool holds_band(std::uint32_t width, std::uint32_t height) noexcept {
  return width >= kBandWidth && height >= kBandRows;
}

bool concurrent(VkSharingMode mode) noexcept {
  return mode == VK_SHARING_MODE_CONCURRENT;
}

// `picture` checked as a read takes it: luma bytes in a buffer, on the host
// or as NV12's images, with what each needs.
core::Status check_picture(const YuvImage& picture,
                           const core::Device& device) {
  const bool in_buffer = picture.device != nullptr;
  const bool on_host = picture.host != nullptr;
  const bool as_images =
      picture.image[0] != nullptr || picture.image[1] != nullptr;
  if (!in_buffer && !on_host && !as_images) {
    return refuse("the picture has no planes");
  }
  if (picture.width == 0 || picture.height == 0) {
    return refuse("the picture is empty");
  }
  if (picture.layout == YuvLayout::Yuyv || picture.layout == YuvLayout::V210) {
    return refuse(
        "reads 8-bit luma planes (NV12, I420, luma alone), and this picture "
        "is YUYV or V210");
  }
  if (in_buffer && !on_host && !as_images) {
    if (!picture.device->valid()) {
      return refuse("the picture's buffer is empty");
    }
    if ((picture.device->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
      return refuse("the picture's buffer needs TRANSFER_SRC usage");
    }
  }
  VKC_TRY(detail::check_yuv_planes(picture, kWho, detail::ImageRead::Copy)
              .status());
  return detail::check_ready(picture, device, kWho);
}

// `frame`'s depth checked as a read takes it: on the host, or 16-bit samples
// in a device buffer at an even byte, all inside it.
core::Status check_depth(const RgbdFrame& frame) {
  const DeviceDepth& device = frame.depth_device;
  if (frame.depth != nullptr && device.buffer != nullptr) {
    return refuse("the depth must be on the host or on the device, not both");
  }
  if (device.buffer == nullptr) return {};
  if (!device.buffer->valid()) return refuse("the depth's buffer is empty");
  if ((device.buffer->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
    return refuse("the depth's buffer needs TRANSFER_SRC usage");
  }
  if (device.offset % 2 != 0) {
    return refuse("the depth's samples start at an odd byte");
  }
  const camera::ImageSize& size = frame.depth_camera.size;
  const std::uint64_t bytes = 2 * std::uint64_t{size.width} * size.height;
  if (device.offset > device.buffer->size() ||
      bytes > device.buffer->size() - device.offset) {
    return refuse("the depth runs past its buffer");
  }
  return {};
}

// A band's rows, `row_bytes` each and `stride` apart from `first`, packed
// into `out`: what a host read uploads.
void pack_rows(const std::uint8_t* first, std::size_t stride,
               std::size_t row_bytes, std::uint8_t* out) {
  for (std::uint32_t y = 0; y < kBandRows; ++y) {
    std::memcpy(out + y * row_bytes, first + y * stride, row_bytes);
  }
}

}  // namespace

struct PatternReader::Impl {
  const core::Device* device = nullptr;
  core::Allocator* allocator = nullptr;
  std::uint32_t max_groups = 0;
  core::ComputeKernel kernel;
  core::DescriptorPool pool;
  // The bands, copied in each read, and the kernel's result words: its own
  // set binds both, written once at create.
  core::Buffer band;
  core::Buffer found;
  // A submit or wait that failed: the device may still run that batch.
  core::Status failed;

  // Read `picture`'s band (when not null) and `frame`'s depth (when not
  // null) in one batch, into `picture_index` and `depth_index`.
  core::Status read(YuvImage* picture, RgbdFrame* frame,
                    std::optional<std::uint32_t>& picture_index,
                    std::optional<std::uint32_t>& depth_index);
};

PatternReader::PatternReader(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

PatternReader::~PatternReader() = default;

core::Result<std::unique_ptr<PatternReader>> PatternReader::create(
    const core::Device& device, core::Allocator& allocator) try {
  if (device.handle() == VK_NULL_HANDLE || !allocator.valid()) {
    return core::Status::invalid_argument(
        "PatternReader::create: the device or allocator is empty");
  }
  VKC_TRY(check_device_requirements(device, "PatternReader::create"));
  auto impl = std::make_unique<Impl>();
  impl->device = &device;
  impl->allocator = &allocator;
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.size = sizeof(PatternParams);
  core::KernelSetBuilder builder(device);
  VKC_TRY(builder.add(impl->kernel, "pattern_index", vs_pattern_index_comp_spv,
                      vs_pattern_index_comp_spv_size, 2, &push));
  VKC_ASSIGN(impl->pool, builder.build());
  VKC_ASSIGN(impl->band,
             core::device_storage_buffer(allocator, kDepthAt + kDepthBytes));
  VKC_ASSIGN(impl->found, core::device_storage_buffer(allocator, 8));
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(impl->band.handle()),
                         "sensor.pattern_band");
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(impl->found.handle()),
                         "sensor.pattern_found");
  impl->kernel.set.write_storage_buffer(0, impl->band.handle(), 0,
                                        VK_WHOLE_SIZE);
  impl->kernel.set.write_storage_buffer(1, impl->found.handle(), 0,
                                        VK_WHOLE_SIZE);
  impl->max_groups = device.caps().limits().maxComputeWorkGroupCount[0];
  return std::unique_ptr<PatternReader>(new PatternReader(std::move(impl)));
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory(
      "PatternReader::create: out of host memory");
}

core::Result<std::optional<std::uint32_t>> PatternReader::read(
    YuvImage& picture) try {
  std::optional<std::uint32_t> index;
  std::optional<std::uint32_t> unused;
  VKC_TRY(impl_->read(&picture, nullptr, index, unused));
  return index;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("PatternReader: out of host memory");
}

core::Result<FrameIndex> PatternReader::read(RgbdFrame& frame) try {
  const YuvImage& color = frame.color;
  const bool has_picture = color.device != nullptr || color.host != nullptr ||
                           color.image[0] != nullptr ||
                           color.image[1] != nullptr;
  FrameIndex out;
  VKC_TRY(impl_->read(has_picture ? &frame.color : nullptr, &frame, out.picture,
                      out.depth));
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("PatternReader: out of host memory");
}

core::Status PatternReader::Impl::read(
    YuvImage* picture, RgbdFrame* frame,
    std::optional<std::uint32_t>& picture_index,
    std::optional<std::uint32_t>& depth_index) {
  if (!failed.ok()) {
    return failed.with_context(
        "PatternReader: an earlier read failed and may still run");
  }
  picture_index.reset();
  depth_index.reset();
  // Everything checked before anything is recorded; a band that does not fit
  // is read by nothing.
  if (picture != nullptr) VKC_TRY(check_picture(*picture, *device));
  if (frame != nullptr) VKC_TRY(check_depth(*frame));
  const bool read_picture =
      picture != nullptr && holds_band(picture->width, picture->height);
  const camera::ImageSize depth_size =
      frame != nullptr ? frame->depth_camera.size : camera::ImageSize{};
  const bool read_depth = frame != nullptr && frame->has_depth() &&
                          holds_band(depth_size.width, depth_size.height);
  if (!read_picture && !read_depth) return {};
  const DeviceDepth* device_depth =
      read_depth && frame->depth_device.buffer != nullptr ? &frame->depth_device
                                                          : nullptr;
  const bool picture_in_buffer = read_picture && picture->device != nullptr;
  // One buffer holding the depth and the picture's planes is taken over
  // once, so both name the one writer.
  const std::uint32_t own = device->queue_family();
  const bool shared =
      device_depth != nullptr && picture_in_buffer &&
      device_depth->buffer->handle() == picture->device->handle();
  if (shared &&
      detail::transfer_family(*picture->device, picture->queue_family, own) !=
          detail::transfer_family(*device_depth->buffer,
                                  device_depth->queue_family, own)) {
    return refuse(
        "the depth and the picture share a buffer, which is taken over once, "
        "but name two writers' queue families");
  }

  core::CommandBatch batch(*device, *allocator);
  std::vector<core::TimelinePoint> waits;
  std::array<std::uint32_t, 2> words{kNone, kNone};
  // Every buffer and image taken over first, so a family the device lacks,
  // or one an EXCLUSIVE image cannot cross from, is refused before anything
  // else is recorded.
  if (read_picture) {
    if (picture->device != nullptr) {
      VKC_TRY(batch.acquire(*picture->device, picture->queue_family));
    } else if (picture->host == nullptr) {
      for (const std::shared_ptr<const core::Image>& plane : picture->image) {
        VKC_TRY(batch.acquire(*plane, picture->queue_family, plane->layout()));
      }
    }
    add_ready_point(*picture, waits);
  }
  if (device_depth != nullptr && !shared) {
    VKC_TRY(batch.acquire(*device_depth->buffer, device_depth->queue_family));
  }
  // Each band into its place, then the kernel over it, then its word back.
  if (read_picture) {
    const YuvImage& p = *picture;
    if (p.image[0] != nullptr) {
      VKC_TRY(batch.copy(*p.image[0], kBandWidth, kBandRows, band, 0));
    } else if (p.device != nullptr) {
      for (std::uint32_t y = 0; y < kBandRows; ++y) {
        VKC_TRY(batch.copy(*p.device, p.offset[0] + y * p.stride[0], band,
                           y * VkDeviceSize{kBandWidth}, kBandWidth));
      }
    } else {
      std::array<std::uint8_t, kLumaBytes> rows;
      pack_rows(p.host + p.offset[0], p.stride[0], kBandWidth, rows.data());
      VKC_TRY(batch.upload(band, 0, rows.data(), rows.size()));
    }
    const PatternParams params{0, kPatternNearMm, kPatternFarMm};
    VKC_TRY(batch.dispatch(kernel, &params, sizeof(params), 1, max_groups));
    VKC_TRY(batch.readback(found, 0, 4, &words[0]));
  }
  if (read_depth) {
    const std::size_t stride = 2 * std::size_t{depth_size.width};
    if (device_depth != nullptr) {
      for (std::uint32_t y = 0; y < kBandRows; ++y) {
        VKC_TRY(batch.copy(*device_depth->buffer,
                           device_depth->offset + y * VkDeviceSize{stride},
                           band, kDepthAt + y * kDepthRowBytes,
                           kDepthRowBytes));
      }
    } else {
      std::array<std::uint8_t, kDepthBytes> rows;
      pack_rows(reinterpret_cast<const std::uint8_t*>(frame->depth), stride,
                kDepthRowBytes, rows.data());
      VKC_TRY(batch.upload(band, kDepthAt, rows.data(), rows.size()));
    }
    const PatternParams params{1, kPatternNearMm, kPatternFarMm};
    VKC_TRY(batch.dispatch(kernel, &params, sizeof(params), 1, max_groups));
    VKC_TRY(batch.readback(found, 4, 4, &words[1]));
  }
  core::Result<core::PendingBatch> pending = batch.submit_async(waits, {});
  const core::Status submitted =
      pending.ok() ? pending->wait() : pending.status();
  if (!submitted.ok()) {
    // The batch may still run: the bands, the result words and the inputs,
    // which the caller may drop as soon as this returns, are let go rather
    // than freed, and every later read is refused.
    static_cast<void>(new core::Buffer(std::move(band)));
    static_cast<void>(new core::Buffer(std::move(found)));
    if (read_picture) static_cast<void>(new YuvImage(*picture));
    if (device_depth != nullptr) {
      static_cast<void>(
          new std::shared_ptr<const core::Buffer>(device_depth->buffer));
    }
    failed = submitted;
    return submitted;
  }

  // The inputs' ownership as the batch left it, published only now: a later
  // stage must not repeat the writer's release and this acquire. Images that
  // needed no transfer keep their kQueueFamilyIgnored.
  if (read_picture) {
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
    if (words[0] != kNone) picture_index = words[0];
  }
  if (device_depth != nullptr) {
    frame->depth_device.queue_family =
        concurrent(device_depth->buffer->sharing_mode()) ? kQueueFamilyIgnored
                                                         : own;
  }
  if (read_depth && words[1] != kNone) depth_index = words[1];
  return {};
}

}  // namespace volumetric_kit::sensor::testing
