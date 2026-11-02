// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/nv12_picture.hpp
/// @brief Test support for 8-bit NV12 pictures, with no test framework: a
///        picture on the host, the family's test pattern, whose index a
///        reader recovers after lossy coding, and a picture on the device as
///        a hardware decoder hands one out.
///
/// Header-only. Its user links `volumetric_kit::sensor_frame` (or a tier
/// above it) and, for @ref device_picture, the core's
/// `volumetric_kit::core_test_policy`, whose `core::test::write_image`
/// writes the planes.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// const testing::HostNv12 picture = testing::test_pattern(256, 144, frame);
/// VKC_ASSIGN(YuvImage yuv,
///            testing::device_picture(device, allocator, picture));
/// yuv.full_range = false;  // the coding the test means it to have
/// const YcbcrWeights w = ycbcr_weights(VideoColorMatrix::Bt709);
/// yuv.kr = w.kr;
/// yuv.kb = w.kb;
/// @endcode

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/write_image.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief An 8-bit NV12 picture on the host: its luma rows, then its chroma
///        rows, each plane's rows packed.
struct HostNv12 {
  std::uint32_t width = 0;   ///< Luma width (pixels).
  std::uint32_t height = 0;  ///< Luma height (pixels).
  /// @ref width x @ref height bytes.
  std::vector<std::uint8_t> luma;
  /// (@ref height + 1) / 2 rows of (@ref width + 1) / 2 Cb, Cr pairs, Cb
  /// first: 2 x (@ref width + 1) / 2 bytes a row.
  std::vector<std::uint8_t> chroma;
};

/// @brief The side of one cell of @ref test_pattern's index band, in luma
///        pixels: 16, a coding block's, so each cell is whole blocks.
inline constexpr std::uint32_t kIndexCell = 16;
/// @brief The bits of the index @ref test_pattern's band holds: 14, so it
///        names `index % 16384`.
inline constexpr std::uint32_t kIndexBits = 14;

/// @return Picture @p index of the family's test pattern, @p width x
///         @p height: 8-bit codes, with 0 and 255 in them where the picture
///         is at least 32 wide, and, in a picture at least 256 x 16, a
///         different picture at each index below 16384.
///
/// - **The index band**, luma rows 0 to 15 and columns 0 to 255, is 16 cells
///   of @ref kIndexCell x @ref kIndexCell, left to right. Cell 0 is 255 and
///   cell 1 is 0, the references. Cell 2 + b is 255 where bit b of @p index
///   is set and 0 where it is not, for b from 0 to @ref kIndexBits - 1, so
///   the band names `index % 16384`. A reader takes each cell's mean away
///   from its edges and compares it with the references' midpoint, which
///   holds after lossy coding and a change of range. The band's chroma
///   (chroma rows 0 to 7, pairs 0 to 127) is 128, neutral. A picture under
///   256 x 16 holds as much of the band as fits.
/// - **Elsewhere** each plane is a triangle wave, smooth, so it codes well,
///   and moving with the index, as a P-picture predicts it: luma
///   `tri(x + 2y + 3 index, 255)`, Cb `64 + tri(2x + y + 2 index, 127)` and
///   Cr `64 + tri(x + 2y + 5 index, 127)` at chroma pair x of chroma row y,
///   where `tri(v, top)` rises from 0 to `top` and falls back over
///   `2 top`.
inline HostNv12 test_pattern(std::uint32_t width, std::uint32_t height,
                             std::uint32_t index) {
  const auto tri = [](std::uint64_t v, std::uint64_t top) {
    const std::uint64_t r = v % (2 * top);
    return static_cast<std::uint8_t>(r <= top ? r : 2 * top - r);
  };
  constexpr std::uint32_t kBandWidth = kIndexCell * (2 + kIndexBits);
  const std::uint64_t i = index;
  HostNv12 p;
  p.width = width;
  p.height = height;
  p.luma.resize(std::size_t{width} * height);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      std::uint8_t v = tri(x + 2 * std::uint64_t{y} + 3 * i, 255);
      if (y < kIndexCell && x < kBandWidth) {
        const std::uint32_t cell = x / kIndexCell;
        const bool set = cell == 0 || (cell >= 2 && ((i >> (cell - 2)) & 1));
        v = set ? 255 : 0;
      }
      p.luma[std::size_t{y} * width + x] = v;
    }
  }
  const std::uint32_t pairs = (width + 1) / 2;
  const std::uint32_t rows = (height + 1) / 2;
  p.chroma.resize(std::size_t{pairs} * rows * 2);
  for (std::uint32_t y = 0; y < rows; ++y) {
    for (std::uint32_t x = 0; x < pairs; ++x) {
      std::uint8_t* pair = &p.chroma[(std::size_t{y} * pairs + x) * 2];
      if (y < kIndexCell / 2 && x < kBandWidth / 2) {
        pair[0] = pair[1] = 128;
        continue;
      }
      pair[0] = static_cast<std::uint8_t>(
          64 + tri(2 * std::uint64_t{x} + y + 2 * i, 127));
      pair[1] = static_cast<std::uint8_t>(
          64 + tri(x + 2 * std::uint64_t{y} + 5 * i, 127));
    }
  }
  return p;
}

/// @return @p picture on @p device as a hardware decoder hands one out:
///         NV12's two plane images (`YuvImage::image`), R8 and R8G8, in
///         `GENERAL`, each with a view and `SAMPLED` and `TRANSFER_SRC`
///         usage, and `TRANSFER_DST`, which writing them takes; exclusive
///         to the device's queue family, and written and waited on. Each
///         image is @p pad texels wider and @p pad rows taller than its
///         plane, as VideoToolbox's can be: the plane at its corner, 0xEE
///         past it. The colour fields are `YuvImage`'s defaults: set the
///         matrix's weights, the range and the siting the test means.
///         InvalidArgument for a picture of no size or with planes of
///         other lengths; else an image's or a write's failure.
inline core::Result<YuvImage> device_picture(const core::Device& device,
                                             core::Allocator& allocator,
                                             const HostNv12& picture,
                                             std::uint32_t pad = 0) {
  const std::uint32_t pairs = (picture.width + 1) / 2;
  const std::uint32_t rows = (picture.height + 1) / 2;
  if (picture.width == 0 || picture.height == 0 ||
      picture.luma.size() != std::size_t{picture.width} * picture.height ||
      picture.chroma.size() != std::size_t{pairs} * rows * 2) {
    return core::Status::invalid_argument(
        "device_picture: an NV12 picture of no size or with planes of other "
        "lengths");
  }
  struct Plane {
    const std::vector<std::uint8_t>* bytes;
    std::uint32_t columns, rows, texel;
    VkFormat format;
  };
  const Plane planes[2] = {
      {&picture.luma, picture.width, picture.height, 1, VK_FORMAT_R8_UNORM},
      {&picture.chroma, pairs, rows, 2, VK_FORMAT_R8G8_UNORM}};
  YuvImage yuv;
  yuv.layout = YuvLayout::Nv12;
  yuv.width = picture.width;
  yuv.height = picture.height;
  for (std::size_t k = 0; k < 2; ++k) {
    const Plane& plane = planes[k];
    const std::uint32_t columns = plane.columns + pad;
    const std::size_t row_bytes = std::size_t{plane.columns} * plane.texel;
    std::vector<std::uint8_t> texels(
        std::size_t{columns} * (plane.rows + pad) * plane.texel, 0xEE);
    for (std::uint32_t r = 0; r < plane.rows; ++r) {
      std::copy_n(plane.bytes->data() + r * row_bytes, row_bytes,
                  texels.data() + std::size_t{r} * columns * plane.texel);
    }
    core::ImageDesc desc;
    desc.extent = {columns, plane.rows + pad};
    desc.format = plane.format;
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKC_ASSIGN(core::Image image, allocator.create_image(desc));
    VKC_TRY(core::test::write_image(device, allocator, image, texels.data(),
                                    texels.size(), VK_IMAGE_LAYOUT_GENERAL));
    yuv.image[k] = std::make_shared<const core::Image>(std::move(image));
  }
  return yuv;
}

}  // namespace volumetric_kit::sensor::testing
