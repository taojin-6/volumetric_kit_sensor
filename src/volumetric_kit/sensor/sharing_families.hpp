// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The queue families a picture's plane images are shared with, checked as
// both of their makers check them: Apple's PixelBufferImporter and NVDEC's
// plane images (sensor/video/cuda_pictures.cpp). Header-only, and internal to
// the targets whose sources include it.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/physical_device_info.hpp"

namespace volumetric_kit::recon::sensor::detail {

// The families the images are CONCURRENT across: the distinct entries of
// `families` when two or more, as the core's allocator picks a buffer's
// sharing; none, for EXCLUSIVE, otherwise. `who` starts every message.
// TODO(sensor): everything but the own-family check repeats the core
// allocator's internal sharing_for (allocator.cpp), behind its buffers and
// images. Call it instead once the core makes it public and recon's core pin
// reaches that.
inline core::Result<std::vector<std::uint32_t>> sharing_families(
    const core::Device& device, const std::uint32_t* families,
    std::uint32_t count, const std::string& who) {
  if (families == nullptr && count != 0) {
    return core::Status::invalid_argument(
        who + ": queue_family_count is non-zero but queue_families is null");
  }
  std::vector<std::uint32_t> distinct;
  if (count != 0) distinct.assign(families, families + count);
  std::sort(distinct.begin(), distinct.end());
  distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
  if (distinct.size() > core::BufferDesc::kMaxQueueFamilies) {
    return core::Status::invalid_argument(
        who + ": " + std::to_string(distinct.size()) +
        " distinct queue families, more than BufferDesc::kMaxQueueFamilies (" +
        std::to_string(core::BufferDesc::kMaxQueueFamilies) + ")");
  }
  // CONCURRENT naming a family the device lacks is undefined
  // (VUID-VkImageCreateInfo-sharingMode-01420), and vkCreateImage still
  // succeeds.
  const std::size_t device_families = device.caps().queue_families().size();
  if (!distinct.empty() && distinct.back() >= device_families) {
    return core::Status::invalid_argument(
        who + ": queue_families names family " +
        std::to_string(distinct.back()) + ", which the device does not have (" +
        std::to_string(device_families) + " families)");
  }
  // The images are readied on the device's own family, which must be one
  // they are shared with.
  if (count != 0 && !std::binary_search(distinct.begin(), distinct.end(),
                                        device.queue_family())) {
    return core::Status::invalid_argument(
        who + ": queue_families leaves out the device's own family, " +
        std::to_string(device.queue_family()));
  }
  if (distinct.size() < 2) distinct.clear();
  return distinct;
}

}  // namespace volumetric_kit::recon::sensor::detail
