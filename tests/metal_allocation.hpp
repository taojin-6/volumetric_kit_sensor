// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What the Metal device under a MoltenVK device has allocated
// (`MTLDevice.currentAllocatedSize`), for tests of what Vulkan objects cost
// there (Apple only). The device is read off the exported texture of a 1x1
// probe image, as PixelBufferImporter::create reads it.

#include <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "volumetric_kit/core/vulkan/device.hpp"

namespace vr_test {

// The bytes the Metal device under `device` has allocated; empty where
// `device` names none.
std::optional<std::int64_t> metal_allocated_size(
    const volumetric_kit::core::Device& device);

// What wrapping both planes of each of `pixels` (8-bit NV12 on IOSurfaces) in
// bare Metal textures, as PixelBufferImporter wraps them, adds to that while
// the textures are alive; empty where a plane does not wrap or `device` names
// no Metal device.
std::optional<std::int64_t> bare_plane_textures_size(
    const volumetric_kit::core::Device& device,
    const std::vector<CVPixelBufferRef>& pixels);

}  // namespace vr_test
