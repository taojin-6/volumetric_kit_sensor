// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// An image made and filled by hand and left in GENERAL, as an import hands
// one to recon, on any GPU: what the tests of the image copy and of colour
// arriving as images read.

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/write_image.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace test_image {

// A `width` x `height` image of `format` holding `texels`, rows packed, in
// memory of its own, filled by the core's test::write_image; `usage` beside
// TRANSFER_DST, which filling it needs.
inline volumetric_kit::core::Result<volumetric_kit::core::Image> make(
    const volumetric_kit::core::Device& device,
    volumetric_kit::core::Allocator& allocator, VkFormat format,
    std::uint32_t width, std::uint32_t height,
    const std::vector<std::uint8_t>& texels,
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
  namespace vkc = volumetric_kit::core;
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
  VKC_VK_TRY(vkCreateImage(dev, &info, nullptr, &image));
  VkMemoryRequirements needs{};
  vkGetImageMemoryRequirements(dev, image, &needs);
  const std::optional<std::uint32_t> type =
      vkc::find_memory_type(device, needs.memoryTypeBits, 0);
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.allocationSize = needs.size;
  alloc.memoryTypeIndex = type.value_or(0);
  VkDeviceMemory backing = VK_NULL_HANDLE;
  if (!type || vkAllocateMemory(dev, &alloc, nullptr, &backing) != VK_SUCCESS ||
      vkBindImageMemory(dev, image, backing, 0) != VK_SUCCESS) {
    vkFreeMemory(dev, backing, nullptr);
    vkDestroyImage(dev, image, nullptr);
    return vkc::Status::out_of_memory("test image: no memory for it");
  }
  vkc::ImageInfo adopted;
  adopted.image = image;
  adopted.format = format;
  adopted.extent = info.extent;
  adopted.usage = info.usage;
  vkc::Image out(adopted, [dev, image, backing] {
    vkDestroyImage(dev, image, nullptr);
    vkFreeMemory(dev, backing, nullptr);
  });

  VKC_TRY(vkc::test::write_image(device, allocator, out, texels.data(),
                                 texels.size(), VK_IMAGE_LAYOUT_GENERAL));
  return out;
}

// A `width` x `height` image of `format`, `texel` bytes a texel, made by
// make(): `count` rows of `row_bytes` from `rows` in its corner, junk
// elsewhere; null if it cannot be made.
inline std::shared_ptr<const volumetric_kit::core::Image> with_corner(
    const volumetric_kit::core::Device& device,
    volumetric_kit::core::Allocator& allocator, VkFormat format,
    std::uint32_t texel, std::uint32_t width, std::uint32_t height,
    const std::uint8_t* rows, std::uint32_t row_bytes, std::uint32_t count,
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
  std::vector<std::uint8_t> texels(std::size_t{width} * height * texel, 0xEE);
  for (std::uint32_t r = 0; r < count; ++r) {
    std::memcpy(&texels[std::size_t{r} * width * texel],
                rows + std::size_t{r} * row_bytes, row_bytes);
  }
  auto made = make(device, allocator, format, width, height, texels, usage);
  return made.ok() ? std::make_shared<const volumetric_kit::core::Image>(
                         std::move(made).value())
                   : nullptr;
}

}  // namespace test_image
