// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The colour pass's interface and the checks that feed it: what
// GpuFramePrep's colour pass (shaders/undistort_color.comp) and the stereo
// rectifier's kernel (stereo/shaders/rectify.comp), which share its body
// (shaders/undistort_color.glsl, shaders/color_planes.glsl), push and bind.
// Internal to the package: sensor_prep builds it, and sensor_stereo calls it.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/chroma_location.hpp"
#include "volumetric_kit/sensor/prep/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::detail {

// shaders/lens.glsl's LensCamera, under scalar layout.
struct LensParams {
  float fx, fy, cx, cy;
  std::uint32_t width, height;
  float k1, k2, p1, p2, k3, k4, k5, k6;
};
static_assert(sizeof(LensParams) == 56, "LensParams must match lens.glsl");

// shaders/undistort_color.glsl's Params: the camera the picture was taken
// through, and where and how its planes are coded.
struct ColorParams {
  LensParams cam;
  std::uint32_t y_offset;
  std::uint32_t cb_offset;
  std::uint32_t cr_offset;
  std::uint32_t y_stride;
  std::uint32_t cb_stride;
  std::uint32_t cr_stride;
  std::uint32_t c_step;
  float kr;
  float kb;
  std::uint32_t full_range;
  float chroma_x;
  float chroma_y;
  std::uint32_t packed_rgb;        // packed R'G'B' words, not Y'CbCr planes
  std::uint32_t y_step;            // bytes between luma samples
  std::uint32_t chroma_full_rows;  // 4:2:2: chroma on every row
  std::uint32_t luma_only;         // Luma: no chroma, so grey
};
static_assert(sizeof(ColorParams) == 120, "ColorParams layout drift");
static_assert(offsetof(ColorParams, chroma_x) == 96,
              "ColorParams layout drift");
static_assert(offsetof(ColorParams, packed_rgb) == 104,
              "ColorParams layout drift");
static_assert(offsetof(ColorParams, chroma_full_rows) == 112,
              "ColorParams layout drift");
static_assert(offsetof(ColorParams, luma_only) == 116,
              "ColorParams layout drift");

// The model as the shaders read it: narrowed to float once, here.
inline LensParams lens_params(const camera::CameraModel& c) noexcept {
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  const auto f = [](double v) { return static_cast<float>(v); };
  return LensParams{f(k.fx),       f(k.fy), f(k.cx), f(k.cy), c.size.width,
                    c.size.height, f(d.k1), f(d.k2), f(d.p1), f(d.p2),
                    f(d.k3),       f(d.k4), f(d.k5), f(d.k6)};
}

// A camera the passes can sample through: camera::check_camera_model's
// checks, and every value within float's range, as the shaders read it --
// compared before narrowing, which past that range is undefined -- with
// focal lengths that stay positive once narrowed. Refusals read "<who>: the
// <what> camera ...".
VS_PREP_API core::Status check_camera(const char* who, const char* what,
                                      const camera::CameraModel& c);

// A picture's planes as the colour pass binds them: where each plane starts
// from the binding, its rows' stride and its samples' step, and the range of
// bytes bound. Planes on the host are copied into a slot from bind_offset,
// copy_bytes of them, and bound from their first byte there; planes in a
// buffer are bound where they are, from bind_offset rounded down to the
// binding alignment; plane images bind no bytes.
struct PlaneBinding {
  VkDeviceSize y_offset = 0, cb_offset = 0, cr_offset = 0;
  VkDeviceSize y_stride = 0, cb_stride = 0, cr_stride = 0;
  VkDeviceSize y_step = 1, c_step = 1;
  bool chroma_full_rows = false;  // 4:2:2 chroma on every row
  bool luma_only = false;         // Luma: no chroma
  VkDeviceSize bind_offset = 0, bind_bytes = 0;
  VkDeviceSize copy_bytes = 0;  // host bytes the slot takes
};

// `image`, the picture a camera of `size` took, checked as the colour pass
// reads it -- its size the camera's, a known chroma location and a matrix of
// positive weights summing below 1 (luma alone has neither), a buffer it can
// bind, its planes as check_yuv_planes takes them for sampling, every byte
// within 4 GiB and a storage binding's range -- and bound. Refusals read
// "<who>: ..." and name the picture as "the <what> ...".
VS_PREP_API core::Result<PlaneBinding> bind_planes(
    const YuvImage& image, const camera::ImageSize& size, const char* who,
    const char* what, VkDeviceSize max_range, VkDeviceSize offset_alignment);

// The colour pass's push constants for `image`, bound as `binding` and seen
// through `cam`; `packed_rgb` for a dataset's packed words.
inline ColorParams color_params(const LensParams& cam, const YuvImage& image,
                                const PlaneBinding& binding,
                                bool packed_rgb) noexcept {
  const auto u32 = [](VkDeviceSize v) { return static_cast<std::uint32_t>(v); };
  const auto chroma = chroma_offset(image.chroma_location);
  return ColorParams{cam,
                     u32(binding.y_offset),
                     u32(binding.cb_offset),
                     u32(binding.cr_offset),
                     u32(binding.y_stride),
                     u32(binding.cb_stride),
                     u32(binding.cr_stride),
                     u32(binding.c_step),
                     image.kr,
                     image.kb,
                     image.full_range ? 1u : 0u,
                     chroma[0],
                     chroma[1],
                     packed_rgb ? 1u : 0u,
                     u32(binding.y_step),
                     binding.chroma_full_rows ? 1u : 0u,
                     binding.luma_only ? 1u : 0u};
}

}  // namespace volumetric_kit::sensor::detail
