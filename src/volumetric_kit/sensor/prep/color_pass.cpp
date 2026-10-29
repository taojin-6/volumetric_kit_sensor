// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "color_pass.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "yuv_planes.hpp"

namespace volumetric_kit::sensor::detail {
namespace {

bool finite(float v) noexcept { return std::isfinite(v); }

VkDeviceSize round_up4(VkDeviceSize bytes) noexcept {
  return (bytes + 3) & ~VkDeviceSize{3};
}

}  // namespace

core::Status check_camera(const char* who, const char* what,
                          const camera::CameraModel& c) {
  const std::string the = std::string(who) + ": the " + what;
  const core::Status valid = camera::check_camera_model(c);
  if (!valid.ok()) {
    return core::Status::invalid_argument(the + " " + valid.message());
  }
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  for (const double v : {k.fx, k.fy, k.cx, k.cy, d.k1, d.k2, d.p1, d.p2, d.k3,
                         d.k4, d.k5, d.k6}) {
    if (std::fabs(v) > std::numeric_limits<float>::max()) {
      return core::Status::invalid_argument(the +
                                            " camera is past float's range");
    }
  }
  const LensParams p = lens_params(c);
  if (!(p.fx > 0.0f && p.fy > 0.0f)) {
    return core::Status::invalid_argument(
        the + " camera's focal lengths narrow to 0 in float");
  }
  return {};
}

core::Result<PlaneBinding> bind_planes(const YuvImage& image,
                                       const camera::ImageSize& size,
                                       const char* who, const char* what,
                                       VkDeviceSize max_range,
                                       VkDeviceSize offset_alignment) {
  const std::string the = std::string(who) + ": the " + what;
  // Luma alone is grey: no matrix or chroma siting applies, so none is
  // checked.
  const bool grey = image.layout == YuvLayout::Luma;
  if (!grey && static_cast<unsigned>(image.chroma_location) >
                   static_cast<unsigned>(ChromaLocation::Bottom)) {
    return core::Status::invalid_argument(std::string(who) +
                                          ": unknown chroma location");
  }
  if (image.width != size.width || image.height != size.height) {
    return core::Status::invalid_argument(
        the + " picture is " + std::to_string(image.width) + "x" +
        std::to_string(image.height) + ", its camera " +
        std::to_string(size.width) + "x" + std::to_string(size.height));
  }
  if (!grey && !(finite(image.kr) && finite(image.kb) && image.kr > 0.0f &&
                 image.kb > 0.0f && image.kr + image.kb < 1.0f)) {
    return core::Status::invalid_argument(
        the + " matrix's kr and kb must be positive and sum below 1");
  }
  const bool on_device = image.device != nullptr;
  const bool on_host = image.host != nullptr;
  const bool as_images = image.image[0] != nullptr || image.image[1] != nullptr;
  if (on_device && !as_images) {
    // Empty or without storage usage, before its size is weighed.
    VKC_TRY(core::StorageInput(*image.device)
                .check((the + " planes' buffer").c_str(), 0));
  }
  VKC_ASSIGN(const YuvPlanes planes,
             check_yuv_planes(image, who, ImageRead::Sample));
  PlaneBinding out;
  // Read at integer coordinates through their views: no byte is bound.
  if (as_images) return out;

  const std::uint64_t first =
      *std::min_element(planes.begin, planes.begin + planes.count);
  const std::uint64_t last =
      *std::max_element(planes.end, planes.end + planes.count);
  if (on_host) {
    // Copied from the first plane's first byte to the last one's last, into
    // a slot the kernel reads in whole words.
    out.bind_offset = first;
    out.copy_bytes = last - first;
    // Weighed before it is rounded up to a word, which near 2^64 would wrap.
    if (out.copy_bytes > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::invalid_argument(the +
                                            " planes span more than 4 GiB");
    }
    out.bind_bytes = round_up4(out.copy_bytes);
  } else {
    // The kernel reads whole words, so the last one must be in the buffer.
    VKC_TRY(core::StorageInput(*image.device)
                .check((the + " planes' buffer").c_str(), round_up4(last)));
    // Bound from the first plane, rounded down to the alignment a binding's
    // offset needs (and to a word, which the kernel reads), so the limits
    // below weigh the picture rather than where it sits in its buffer: a
    // decoder's ring slot far into one buffer, say.
    const VkDeviceSize align = std::max<VkDeviceSize>(offset_alignment, 4);
    out.bind_offset = first / align * align;
    out.bind_bytes = round_up4(last) - out.bind_offset;
  }
  out.y_offset = image.offset[0] - out.bind_offset;
  out.y_stride = image.stride[0];
  switch (image.layout) {
    case YuvLayout::Yuyv:
      // Y0 Cb Y1 Cr: luma every second byte, each chroma every fourth.
      out.cb_offset = out.y_offset + 1;
      out.cr_offset = out.y_offset + 3;
      out.cb_stride = out.cr_stride = image.stride[0];
      out.y_step = 2;
      out.c_step = 4;
      out.chroma_full_rows = true;
      break;
    case YuvLayout::Nv12:
      out.cb_offset = image.offset[1] - out.bind_offset;
      out.cr_offset = out.cb_offset + 1;
      out.cb_stride = out.cr_stride = image.stride[1];
      out.c_step = 2;
      break;
    case YuvLayout::I420:
      out.cb_offset = image.offset[1] - out.bind_offset;
      out.cr_offset = image.offset[2] - out.bind_offset;
      out.cb_stride = image.stride[1];
      out.cr_stride = image.stride[2];
      break;
    case YuvLayout::Luma:
      // No chroma: the pass reads none.
      out.luma_only = true;
      break;
  }
  // The kernel addresses the planes in 32-bit bytes from the binding.
  if (out.bind_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return core::Status::invalid_argument(the + " planes span more than 4 GiB");
  }
  VKC_TRY(core::check_storage_buffer_range((the + " planes").c_str(),
                                           out.bind_bytes, max_range));
  return out;
}

}  // namespace volumetric_kit::sensor::detail
