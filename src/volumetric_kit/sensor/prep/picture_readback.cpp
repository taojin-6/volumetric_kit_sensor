// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/utils/picture_readback.hpp"

#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "yuv_planes.hpp"

namespace volumetric_kit::recon::sensor {
namespace {

constexpr const char* kWho = "PictureReadback";

core::Status refuse(const char* why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

VkDeviceSize round_up4(VkDeviceSize bytes) noexcept {
  return (bytes + 3) & ~VkDeviceSize{3};
}

// One plane's copy into the staging: from an image's corner, rows packed, or
// from its buffer where it sits, rows as they were there.
struct PlaneCopy {
  const core::Image* image = nullptr;  // else the picture's buffer
  std::uint32_t width = 0;             // an image's texels a row
  std::uint32_t rows = 0;
  std::uint64_t from = 0;  // a buffer plane's first byte
  VkDeviceSize bytes = 0;  // what it moves
  std::size_t stride = 0;  // between its rows in the staging
  VkDeviceSize at = 0;     // its first byte in the staging
};

// A checked picture's copies: Y, then any chroma.
struct Plan {
  PlaneCopy plane[3];
  int count = 0;
};

// `picture` checked, and its copies placed in the staging from `staged`,
// which moves past them; each starts on a word, as an image copy needs.
core::Result<Plan> plan(const YuvImage& picture, PictureReadback::Planes planes,
                        VkDeviceSize& staged) {
  const bool in_buffer = picture.device != nullptr;
  const bool as_images =
      picture.image[0] != nullptr || picture.image[1] != nullptr;
  if (!in_buffer && !as_images) {
    return refuse("the picture is not on the device");
  }
  if (picture.width == 0 || picture.height == 0) {
    return refuse("the picture is empty");
  }
  if (in_buffer && !as_images) {
    if (!picture.device->valid()) {
      return refuse("the picture's buffer is empty");
    }
    if ((picture.device->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
      return refuse("the picture's buffer needs TRANSFER_SRC usage");
    }
  }
  VKC_ASSIGN(const detail::YuvPlanes shape,
             detail::check_yuv_planes(picture, kWho, detail::ImageRead::Copy));
  Plan out;
  out.count = planes == PictureReadback::Planes::Luma ? 1 : shape.count;
  for (int p = 0; p < out.count; ++p) {
    PlaneCopy& copy = out.plane[p];
    copy.rows = static_cast<std::uint32_t>(shape.rows[p]);
    copy.at = round_up4(staged);
    if (in_buffer) {
      copy.from = shape.begin[p];
      copy.bytes = shape.end[p] - shape.begin[p];
      copy.stride = picture.stride[p];
    } else {
      copy.image = picture.image[p].get();
      copy.width = p == 0 ? picture.width : (picture.width + 1) / 2;
      copy.stride = static_cast<std::size_t>(shape.row_bytes[p]);
      copy.bytes = shape.row_bytes[p] * shape.rows[p];
    }
    staged = copy.at + copy.bytes;
  }
  return out;
}

bool on_device(const YuvImage& picture) noexcept {
  return picture.device != nullptr || picture.image[0] != nullptr ||
         picture.image[1] != nullptr;
}

}  // namespace

PictureReadback::PictureReadback(const core::Device& device,
                                 core::Allocator& allocator,
                                 core::GpuTimer timer)
    : device_(device), allocator_(allocator), timer_(std::move(timer)) {}

core::Result<std::unique_ptr<PictureReadback>> PictureReadback::create(
    const core::Device& device, core::Allocator& allocator) {
  if (device.handle() == VK_NULL_HANDLE || !allocator.valid()) {
    return core::Status::invalid_argument(
        "PictureReadback::create: the device or allocator is empty");
  }
  try {
    VKC_ASSIGN(core::GpuTimer timer, core::GpuTimer::create(device));
    return std::unique_ptr<PictureReadback>(
        new PictureReadback(device, allocator, std::move(timer)));
  } catch (const std::bad_alloc&) {
    return core::Status::out_of_memory(
        "PictureReadback::create: out of host memory");
  }
}

core::Result<std::vector<std::optional<PictureView>>> PictureReadback::read(
    std::vector<std::optional<RgbdFrame>>& frames, Planes planes,
    core::StageMetrics* metrics) {
  try {
    std::vector<YuvImage*> pictures(frames.size(), nullptr);
    for (std::size_t i = 0; i < frames.size(); ++i) {
      if (frames[i] && on_device(frames[i]->color)) {
        pictures[i] = &frames[i]->color;
      }
    }
    std::vector<std::optional<PictureView>> views(frames.size());
    VKC_TRY(read_pictures(pictures.data(), pictures.size(), planes,
                          views.data(), metrics));
    return views;
  } catch (const std::bad_alloc&) {
    return core::Status::out_of_memory("PictureReadback: out of host memory");
  }
}

core::Result<PictureView> PictureReadback::read(YuvImage& picture,
                                                Planes planes,
                                                core::StageMetrics* metrics) {
  try {
    YuvImage* const pictures[] = {&picture};
    std::optional<PictureView> view;
    VKC_TRY(read_pictures(pictures, 1, planes, &view, metrics));
    return *view;
  } catch (const std::bad_alloc&) {
    return core::Status::out_of_memory("PictureReadback: out of host memory");
  }
}

core::Status PictureReadback::read_pictures(YuvImage* const* pictures,
                                            std::size_t count, Planes planes,
                                            std::optional<PictureView>* views,
                                            core::StageMetrics* metrics) {
  // Every picture checked and placed before anything is recorded.
  std::vector<Plan> plans(count);
  VkDeviceSize staged = 0;
  std::uint32_t copies = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (pictures[i] == nullptr) continue;
    VKC_ASSIGN(plans[i], plan(*pictures[i], planes, staged));
    copies += static_cast<std::uint32_t>(plans[i].count);
  }
  if (copies == 0) return {};

  if (metrics != nullptr) VKC_TRY(timer_.reserve(device_, copies));
  core::GpuStageScope stage(metrics, timer_, "readback");
  core::CommandBatch batch(device_, allocator_);
  // Planes taken over from the family that wrote them, images in the layout
  // they are in, first, so a family the device lacks, or one an EXCLUSIVE
  // image cannot cross from, is refused before the staging grows.
  for (std::size_t i = 0; i < count; ++i) {
    if (pictures[i] == nullptr) continue;
    const YuvImage& picture = *pictures[i];
    if (picture.device != nullptr) {
      VKC_TRY(batch.acquire(*picture.device, picture.queue_family));
    } else {
      for (const std::shared_ptr<const core::Image>& plane : picture.image) {
        VKC_TRY(batch.acquire(*plane, picture.queue_family, plane->layout()));
      }
    }
  }
  if (!staging_.valid() || staging_.size() < staged) {
    staging_ = core::Buffer();  // freed first: the last read's views end here
    core::BufferDesc desc;
    desc.size = staged;
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memory = core::MemoryUsage::Staging;
    desc.host_access = core::HostAccess::Random;
    VKC_ASSIGN(staging_, allocator_.create_buffer(desc));
    device_.set_object_name(VK_OBJECT_TYPE_BUFFER,
                            core::debug_object_handle(staging_.handle()),
                            "sensor.picture_readback");
  }
  for (std::size_t i = 0; i < count; ++i) {
    for (int p = 0; p < plans[i].count; ++p) {
      const PlaneCopy& c = plans[i].plane[p];
      if (c.image != nullptr) {
        VKC_TRY(batch.copy(*c.image, c.width, c.rows, staging_, c.at, &stage));
      } else {
        VKC_TRY(batch.copy(*pictures[i]->device, c.from, staging_, c.at,
                           c.bytes, &stage));
      }
    }
  }
  const core::Status submitted = batch.submit();
  if (!submitted.ok()) {
    // A failed wait may leave the copies running, so the staging is let go
    // rather than freed or rewritten, as GpuFramePrep lets go of its own; and
    // so are the pictures, which the caller may drop, and a decoder reuse, as
    // soon as this returns.
    static_cast<void>(new core::Buffer(std::move(staging_)));
    for (std::size_t i = 0; i < count; ++i) {
      if (pictures[i] == nullptr) continue;
      static_cast<void>(new YuvImage(*pictures[i]));
    }
    return submitted;
  }

  const auto* base = static_cast<const std::uint8_t*>(staging_.mapped());
  for (std::size_t i = 0; i < count; ++i) {
    if (pictures[i] == nullptr) continue;
    YuvImage& picture = *pictures[i];
    // Publish ownership only after the submit and wait succeed. A later
    // read or GpuFramePrep must not repeat the writer's release/acquire.
    // Images that needed no transfer keep their kQueueFamilyIgnored.
    const auto concurrent = [](VkSharingMode mode) {
      return mode == VK_SHARING_MODE_CONCURRENT;
    };
    if (picture.device != nullptr) {
      picture.queue_family = concurrent(picture.device->sharing_mode())
                                 ? kQueueFamilyIgnored
                                 : device_.queue_family();
    } else if (picture.queue_family != kQueueFamilyIgnored) {
      picture.queue_family =
          concurrent(picture.image[0]->sharing_mode()) &&
                  concurrent(picture.image[1]->sharing_mode())
              ? kQueueFamilyIgnored
              : device_.queue_family();
    }
    const Plan& placed = plans[i];
    PictureView view;
    view.width = picture.width;
    view.height = picture.height;
    view.full_range = picture.full_range;
    view.layout = picture.layout;
    view.luma = base + placed.plane[0].at;
    view.stride = placed.plane[0].stride;
    for (int p = 1; p < placed.count; ++p) {
      view.chroma[p - 1] = base + placed.plane[p].at;
      view.chroma_stride[p - 1] = placed.plane[p].stride;
    }
    views[i] = view;
  }
  return {};
}

}  // namespace volumetric_kit::recon::sensor
