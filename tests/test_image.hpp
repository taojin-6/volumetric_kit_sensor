// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// An image made and filled by hand and left in GENERAL, as an import hands
// one to recon, on any GPU: what the tests of the image copy and of colour
// arriving as images read.

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/external_memory.hpp"
#include "volumetric_kit/recon/core/image.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/vk_result.hpp"
#include "volumetric_kit/recon/core/vulkan.hpp"

namespace test_image {

// A `width` x `height` image of `format` holding `texels`, rows packed, in
// device-local memory of its own; `usage` beside TRANSFER_DST, which filling
// it needs.
inline volumetric_kit::recon::Result<volumetric_kit::recon::Image> make(
    const volumetric_kit::recon::Device& device,
    volumetric_kit::recon::Allocator& allocator, VkFormat format,
    std::uint32_t width, std::uint32_t height,
    const std::vector<std::uint8_t>& texels,
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
  namespace vr = volumetric_kit::recon;
  const VkDevice dev = device.handle();
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImage image = VK_NULL_HANDLE;
  VR_VK_TRY(vkCreateImage(dev, &info, nullptr, &image));
  VkMemoryRequirements needs{};
  vkGetImageMemoryRequirements(dev, image, &needs);
  const std::optional<std::uint32_t> type =
      vr::find_memory_type(device, needs.memoryTypeBits, 0);
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.allocationSize = needs.size;
  alloc.memoryTypeIndex = type.value_or(0);
  VkDeviceMemory backing = VK_NULL_HANDLE;
  if (!type || vkAllocateMemory(dev, &alloc, nullptr, &backing) != VK_SUCCESS ||
      vkBindImageMemory(dev, image, backing, 0) != VK_SUCCESS) {
    vkFreeMemory(dev, backing, nullptr);
    vkDestroyImage(dev, image, nullptr);
    return vr::Status::out_of_memory("test image: no memory for it");
  }
  vr::ImageInfo adopted;
  adopted.image = image;
  adopted.format = format;
  adopted.extent = info.extent;
  adopted.usage = info.usage;
  adopted.layout = VK_IMAGE_LAYOUT_GENERAL;
  vr::Image out(adopted, [dev, image, backing] {
    vkDestroyImage(dev, image, nullptr);
    vkFreeMemory(dev, backing, nullptr);
  });

  VR_ASSIGN(
      vr::Buffer source,
      allocator.create_buffer({texels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               vr::MemoryUsage::Staging}));
  std::memcpy(source.mapped(), texels.data(), texels.size());
  VR_TRY(device.submit_single_time([&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, source.handle(), image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
  }));
  return out;
}

}  // namespace test_image
