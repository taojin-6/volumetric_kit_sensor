// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Before anything includes Vulkan: VK_EXT_metal_objects' structures.
#define VK_USE_PLATFORM_METAL_EXT

#include "metal_allocation.hpp"

#include <IOSurface/IOSurfaceRef.h>
#include <Metal/Metal.h>

#include "metal_device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace vs_test {

namespace vkc = volumetric_kit::core;
using volumetric_kit::sensor::detail::metal_device;

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

}  // namespace vs_test
