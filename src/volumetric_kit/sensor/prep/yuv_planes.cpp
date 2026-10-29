// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "yuv_planes.hpp"

#include <string>

#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"

namespace volumetric_kit::sensor::detail {
namespace {

// Whether `rows` rows of `row_bytes`, `stride` apart from `offset`, fit in
// `size` bytes, worked out without overflowing 64 bits. `stride` is at least
// `row_bytes`, which is at least 1.
bool plane_fits(std::uint64_t offset, std::uint64_t stride, std::uint64_t rows,
                std::uint64_t row_bytes, std::uint64_t size) noexcept {
  if (offset > size || row_bytes > size - offset) return false;
  return rows <= 1 || rows - 1 <= (size - offset - row_bytes) / stride;
}

core::Status refuse(const char* who, const char* why) {
  return core::Status::invalid_argument(std::string(who) + ": " + why);
}

// NV12's two planes as images `read` takes: R8 luma and R8G8 chroma, each at
// least the picture's size, sampled through their views or copied, as
// CommandBatch::copy checks them too.
core::Status check_images(const YuvImage& image, const char* who,
                          ImageRead read) {
  if (image.layout != YuvLayout::Nv12 || image.image[0] == nullptr ||
      image.image[1] == nullptr) {
    return refuse(who, "colour planes as images are NV12's two");
  }
  const core::Image& y = *image.image[0];
  const core::Image& c = *image.image[1];
  const std::uint32_t cw = image.width / 2 + image.width % 2;
  const std::uint32_t ch = image.height / 2 + image.height % 2;
  if (y.format() != VK_FORMAT_R8_UNORM || c.format() != VK_FORMAT_R8G8_UNORM ||
      y.width() < image.width || y.height() < image.height || c.width() < cw ||
      c.height() < ch) {
    return refuse(who,
                  "the colour images must be R8 luma and R8G8 chroma, at "
                  "least the picture's size");
  }
  if (read == ImageRead::Sample) {
    if ((y.usage() & c.usage() & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 ||
        y.view() == VK_NULL_HANDLE || c.view() == VK_NULL_HANDLE) {
      return refuse(who, "the colour images need SAMPLED usage and a view");
    }
    // The pass reads each through its view as a sampler2D, which a 3D, an
    // arrayed or a multisampled image's view does not match.
    const auto flat = [](const core::Image& i) {
      return i.type() == VK_IMAGE_TYPE_2D && i.array_layers() == 1 &&
             i.samples() == VK_SAMPLE_COUNT_1_BIT;
    };
    if (!flat(y) || !flat(c)) {
      return refuse(who,
                    "the colour images must be 2D, of one layer and one "
                    "sample");
    }
    const auto sampled = [](const core::Image& i) {
      return i.layout() == VK_IMAGE_LAYOUT_GENERAL ||
             i.layout() == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    };
    if (!sampled(y) || !sampled(c)) {
      return refuse(who,
                    "the colour images must be in GENERAL or "
                    "SHADER_READ_ONLY_OPTIMAL");
    }
    return {};
  }
  if ((y.usage() & c.usage() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
    return refuse(who, "the colour images need TRANSFER_SRC usage");
  }
  const auto copyable = [](const core::Image& i) {
    return i.layout() == VK_IMAGE_LAYOUT_GENERAL ||
           i.layout() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  };
  if (!copyable(y) || !copyable(c)) {
    return refuse(who,
                  "the colour images must be in GENERAL or "
                  "TRANSFER_SRC_OPTIMAL");
  }
  return {};
}

}  // namespace

core::Result<YuvPlanes> check_yuv_planes(const YuvImage& image, const char* who,
                                         ImageRead read) {
  const bool on_device = image.device != nullptr;
  const bool as_images = image.image[0] != nullptr || image.image[1] != nullptr;
  if (on_device && as_images) {
    return refuse(who,
                  "the colour planes must be in a buffer or images, not both");
  }
  if (as_images) VKC_TRY(check_images(image, who, read));
  const bool nv12 = image.layout == YuvLayout::Nv12;
  const std::uint64_t cw = (std::uint64_t{image.width} + 1) / 2;
  const std::uint64_t ch = (std::uint64_t{image.height} + 1) / 2;
  const std::uint64_t c_row = nv12 ? 2 * cw : cw;
  YuvPlanes out;
  out.count = nv12 ? 2 : 3;
  out.rows[0] = image.height;
  out.rows[1] = out.rows[2] = ch;
  out.row_bytes[0] = image.width;
  out.row_bytes[1] = out.row_bytes[2] = c_row;
  if (!on_device) return out;

  for (int p = 0; p < out.count; ++p) {
    if (image.stride[p] < out.row_bytes[p]) {
      return refuse(who, "a colour plane's stride is shorter than its rows");
    }
  }
  for (int p = 0; p < out.count; ++p) {
    if (!plane_fits(image.offset[p], image.stride[p], out.rows[p],
                    out.row_bytes[p], image.device->size())) {
      return refuse(who, "a colour plane runs past its buffer");
    }
    out.begin[p] = image.offset[p];
    out.end[p] = image.offset[p] + (out.rows[p] - 1) * image.stride[p] +
                 out.row_bytes[p];
  }
  // Planes sharing bytes would read one as another, as I420 does with its
  // offsets left at zero.
  for (int p = 0; p < out.count; ++p) {
    for (int q = p + 1; q < out.count; ++q) {
      if (out.begin[p] < out.end[q] && out.begin[q] < out.end[p]) {
        return refuse(who, "the colour planes overlap");
      }
    }
  }
  return out;
}

}  // namespace volumetric_kit::sensor::detail
