// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Before anything includes Vulkan: VK_EXT_metal_objects' structures.
#define VK_USE_PLATFORM_METAL_EXT

#include "metal_device.hpp"

#include <cstdint>
#include <optional>

#include "volumetric_kit/core/vulkan/external_memory.hpp"

namespace volumetric_kit::sensor::detail {

VkDeviceMemory bind_memory(const core::Device& device, VkImage image) {
  const VkDevice dev = device.handle();
  VkMemoryRequirements needs{};
  vkGetImageMemoryRequirements(dev, image, &needs);
  const std::optional<std::uint32_t> type = core::find_memory_type(
      device, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (!type) return VK_NULL_HANDLE;
  VkMemoryDedicatedAllocateInfo dedicated{};
  dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated.image = image;
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.pNext = &dedicated;
  alloc.allocationSize = needs.size;
  alloc.memoryTypeIndex = *type;
  VkDeviceMemory out = VK_NULL_HANDLE;
  if (vkAllocateMemory(dev, &alloc, nullptr, &out) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  if (vkBindImageMemory(dev, image, out, 0) != VK_SUCCESS) {
    vkFreeMemory(dev, out, nullptr);
    return VK_NULL_HANDLE;
  }
  return out;
}

id<MTLDevice> metal_device(const core::Device& device) {
  const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
      vkGetDeviceProcAddr(device.handle(), "vkExportMetalObjectsEXT"));
  VkExportMetalObjectCreateInfoEXT exported{};
  exported.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT;
  exported.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT;
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = &exported;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_R8_UNORM;
  info.extent = {1, 1, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  const VkDevice dev = device.handle();
  VkImage probe = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  id<MTLDevice> metal = nil;
  if (export_objects != nullptr &&
      vkCreateImage(dev, &info, nullptr, &probe) == VK_SUCCESS &&
      (memory = bind_memory(device, probe)) != VK_NULL_HANDLE) {
    VkExportMetalTextureInfoEXT texture{};
    texture.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT;
    texture.image = probe;
    texture.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT objects{};
    objects.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT;
    objects.pNext = &texture;
    export_objects(dev, &objects);
    metal = texture.mtlTexture.device;
  }
  vkDestroyImage(dev, probe, nullptr);
  vkFreeMemory(dev, memory, nullptr);
  return metal;
}

}  // namespace volumetric_kit::sensor::detail
