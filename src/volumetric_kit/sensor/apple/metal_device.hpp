// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The Metal device MoltenVK runs a Vulkan device on, and image memory as
// MoltenVK backs it: what PixelBufferImporter wraps a pixel buffer's planes
// with, and what tests read Metal's allocations off. Objective-C++ only, and
// internal to sensor_apple; a test compiles metal_device.mm itself.

#if !defined(__OBJC__)
#error "metal_device.hpp is Objective-C++ only"
#endif

#include <Metal/Metal.h>

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::sensor::detail {

// @p image bound to memory dedicated to it, or null. MoltenVK binds the
// texture's own storage, and makes nothing for memory dedicated to an image;
// other memory it backs with a heap of the allocation's size, which the
// texture never uses. Device-local and not host-visible, which it would back
// with a buffer.
VkDeviceMemory bind_memory(const core::Device& device, VkImage image);

// The MTLDevice under @p device, off the texture of a 1x1 probe image
// (VK_EXT_metal_objects), or nil where it names none: exporting the device
// itself needs the instance to have asked for it when it was made, which an
// embedder's may not have, and an image asks for its own. The probe's memory
// is dedicated to it (bind_memory), so MoltenVK makes no heap for it.
id<MTLDevice> metal_device(const core::Device& device);

}  // namespace volumetric_kit::sensor::detail
