// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Before anything includes Vulkan: VK_EXT_metal_objects' structures.
#define VK_USE_PLATFORM_METAL_EXT

#include "metal_allocation.hpp"

#include <IOSurface/IOSurfaceRef.h>
#include <Metal/Metal.h>

#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace vr_test {
namespace {

namespace vkc = volumetric_kit::core;

// The Metal device under `device`, or nil. The probe's memory is dedicated to
// it, so MoltenVK makes no heap for it.
id<MTLDevice> metal_device(const vkc::Device& device) {
  const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
      vkGetDeviceProcAddr(device.handle(), "vkExportMetalObjectsEXT"));
  if (export_objects == nullptr) return nil;
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
  if (vkCreateImage(dev, &info, nullptr, &probe) != VK_SUCCESS) return nil;
  VkMemoryRequirements needs{};
  vkGetImageMemoryRequirements(dev, probe, &needs);
  const std::optional<std::uint32_t> type = vkc::find_memory_type(
      device, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  VkMemoryDedicatedAllocateInfo dedicated{};
  dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicated.image = probe;
  VkMemoryAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc.pNext = &dedicated;
  alloc.allocationSize = needs.size;
  alloc.memoryTypeIndex = type.value_or(0);
  VkDeviceMemory memory = VK_NULL_HANDLE;
  id<MTLDevice> metal = nil;
  if (type && vkAllocateMemory(dev, &alloc, nullptr, &memory) == VK_SUCCESS &&
      vkBindImageMemory(dev, probe, memory, 0) == VK_SUCCESS) {
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

}  // namespace

std::optional<std::int64_t> metal_allocated_size(const vkc::Device& device) {
  @autoreleasepool {
    const id<MTLDevice> metal = metal_device(device);
    if (metal == nil) return std::nullopt;
    return static_cast<std::int64_t>(metal.currentAllocatedSize);
  }
}

std::optional<std::int64_t> bare_plane_textures_size(
    const vkc::Device& device, const std::vector<CVPixelBufferRef>& pixels) {
  @autoreleasepool {
    const id<MTLDevice> metal = metal_device(device);
    if (metal == nil) return std::nullopt;
    const auto before = static_cast<std::int64_t>(metal.currentAllocatedSize);
    std::vector<id<MTLTexture>> textures;
    for (CVPixelBufferRef p : pixels) {
      for (std::size_t plane = 0; plane < 2; ++plane) {
        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:plane == 0
                                                   ? MTLPixelFormatR8Unorm
                                                   : MTLPixelFormatRG8Unorm
                                         width:CVPixelBufferGetWidthOfPlane(
                                                   p, plane)
                                        height:CVPixelBufferGetHeightOfPlane(
                                                   p, plane)
                                     mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        id<MTLTexture> texture =
            [metal newTextureWithDescriptor:desc
                                  iosurface:CVPixelBufferGetIOSurface(p)
                                      plane:plane];
        if (texture == nil) return std::nullopt;
        textures.push_back(texture);
      }
    }
    return static_cast<std::int64_t>(metal.currentAllocatedSize) - before;
  }
}

}  // namespace vr_test
