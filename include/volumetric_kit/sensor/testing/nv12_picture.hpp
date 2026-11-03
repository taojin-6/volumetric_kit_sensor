// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/nv12_picture.hpp
/// @brief Test support for 8-bit NV12 pictures, with no test framework: a
///        picture on the device as a hardware decoder hands one out, and,
///        through testing/nv12_pattern.hpp, a picture on the host and the
///        family's test pattern, whose index a reader recovers after lossy
///        coding.
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
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::testing {

/// @return @p picture on @p device as a hardware decoder hands one out:
///         NV12's two plane images (`YuvImage::image`), R8 and R8G8, in
///         `GENERAL`, each with a view and `SAMPLED` and `TRANSFER_SRC`
///         usage, and `TRANSFER_DST`, which writing them takes; exclusive
///         to the device's queue family, and written and waited on. Each
///         image is @p pad columns wider and @p pad rows taller than its
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
    std::uint32_t columns, rows, bytes_per_column;
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
    const std::size_t row_bytes =
        std::size_t{plane.columns} * plane.bytes_per_column;
    std::vector<std::uint8_t> filled(
        std::size_t{columns} * (plane.rows + pad) * plane.bytes_per_column,
        0xEE);
    for (std::uint32_t r = 0; r < plane.rows; ++r) {
      std::copy_n(
          plane.bytes->data() + r * row_bytes, row_bytes,
          filled.data() + std::size_t{r} * columns * plane.bytes_per_column);
    }
    core::ImageDesc desc;
    desc.extent = {columns, plane.rows + pad};
    desc.format = plane.format;
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKC_ASSIGN(core::Image image, allocator.create_image(desc));
    VKC_TRY(core::test::write_image(device, allocator, image, filled.data(),
                                    filled.size(), VK_IMAGE_LAYOUT_GENERAL));
    yuv.image[k] = std::make_shared<const core::Image>(std::move(image));
  }
  return yuv;
}

}  // namespace volumetric_kit::sensor::testing
