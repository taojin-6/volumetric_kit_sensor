// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A decoder's picture on the device, read back for a test as its Y, U and V
// planes, rows packed: a buffer, NV12 or I420 at the picture's offsets and
// strides, taken over from CUDA, which wrote it; or NV12 plane images, copied
// into a buffer first.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/image.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace vr_test {

namespace vr = volumetric_kit::recon;

// Fills `planes`, or leaves them empty if a step fails.
inline void read_device_picture(const vr::sensor::DecodedPicture& p,
                                vr::Device& device, vr::Allocator& allocator,
                                std::vector<std::uint8_t> (&planes)[3]) {
  const std::uint32_t cw = (p.width + 1) / 2, ch = (p.height + 1) / 2;
  const bool images = p.image[0] != nullptr;
  // The images' rows packed, the chroma at the 4-byte offset a copy needs.
  const VkDeviceSize chroma_at =
      (VkDeviceSize{p.width} * p.height + 3) & ~VkDeviceSize{3};
  std::shared_ptr<const vr::Buffer> source = p.device;
  if (images) {
    auto made = vr::device_storage_buffer(
        allocator, chroma_at + VkDeviceSize{cw} * ch * 2);
    if (!made) return;
    source = std::make_shared<const vr::Buffer>(std::move(made).value());
  }
  std::vector<std::uint8_t> b(static_cast<std::size_t>(source->size()));
  vr::CommandBatch batch(device, allocator);
  const bool recorded =
      images ? batch.copy(*p.image[0], p.width, p.height, *source, 0).ok() &&
                   batch.copy(*p.image[1], cw, ch, *source, chroma_at).ok()
             : batch.acquire(*source, VK_QUEUE_FAMILY_EXTERNAL).ok();
  if (!recorded || !batch.readback(*source, 0, b.size(), b.data()).ok() ||
      !batch.submit().ok()) {
    return;
  }
  const std::uint64_t at[3] = {images ? 0 : p.offset[0],
                               images ? chroma_at : p.offset[1], p.offset[2]};
  const std::size_t row[3] = {images ? p.width : p.stride[0],
                              images ? 2 * std::size_t{cw} : p.stride[1],
                              p.stride[2]};
  const bool nv12 = images || p.layout == vr::sensor::VideoPixelLayout::Nv12;
  for (std::uint32_t r = 0; r < p.height; ++r) {
    const std::uint8_t* line = b.data() + at[0] + r * row[0];
    planes[0].insert(planes[0].end(), line, line + p.width);
  }
  for (std::uint32_t r = 0; r < ch; ++r) {
    if (nv12) {
      const std::uint8_t* line = b.data() + at[1] + r * row[1];
      for (std::uint32_t x = 0; x < cw; ++x) {
        planes[1].push_back(line[2 * x]);
        planes[2].push_back(line[2 * x + 1]);
      }
      continue;
    }
    for (int i = 1; i < 3; ++i) {
      const std::uint8_t* line = b.data() + at[i] + r * row[i];
      planes[i].insert(planes[i].end(), line, line + cw);
    }
  }
}

}  // namespace vr_test
