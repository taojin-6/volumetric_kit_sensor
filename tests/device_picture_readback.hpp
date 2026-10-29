// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A picture on the device, read back for a test through PictureReadback, as
// its Y, U and V planes, rows packed.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/prep/picture_readback.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace vs_test {

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

// Fills `planes` and updates `p`'s ownership, or leaves them empty on failure.
inline void read_device_picture(sensor::YuvImage& p, vkc::Device& device,
                                vkc::Allocator& allocator,
                                std::vector<std::uint8_t> (&planes)[3]) {
  using sensor::PictureReadback;
  auto readback = PictureReadback::create(device, allocator);
  if (!readback) return;
  auto read = (*readback)->read(p, PictureReadback::Planes::All);
  if (!read) return;
  const sensor::PictureView& v = *read;
  const std::uint32_t cw = (v.width + 1) / 2, ch = (v.height + 1) / 2;
  for (std::uint32_t r = 0; r < v.height; ++r) {
    const std::uint8_t* line = v.luma + r * v.stride;
    planes[0].insert(planes[0].end(), line, line + v.width);
  }
  for (std::uint32_t r = 0; r < ch; ++r) {
    if (v.layout == sensor::YuvLayout::Nv12) {
      const std::uint8_t* line = v.chroma[0] + r * v.chroma_stride[0];
      for (std::uint32_t x = 0; x < cw; ++x) {
        planes[1].push_back(line[2 * x]);
        planes[2].push_back(line[2 * x + 1]);
      }
      continue;
    }
    for (int i = 0; i < 2; ++i) {
      const std::uint8_t* line = v.chroma[i] + r * v.chroma_stride[i];
      planes[i + 1].insert(planes[i + 1].end(), line, line + cw);
    }
  }
}

}  // namespace vs_test
