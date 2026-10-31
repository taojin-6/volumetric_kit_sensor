// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// One picture in every layout GpuFramePrep's colour pass reads -- I420 and
// NV12 planes in a device buffer, NV12 plane images, I420 planes, YUYV and
// luma on the host, and packed R'G'B' words on the host -- and the pass's
// sampling of it on the host, before the 8-bit code: the oracle every layout
// is held to (sensor_gpu_frame_prep_identity, sensor_gpu_frame_prep_response).
// Its luma and chroma swing far enough that limited-range Y'CbCr leaves
// [0, 1] at both ends, as a camera's can.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "test_image.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"
#include "volumetric_kit/sensor/chroma_location.hpp"
#include "volumetric_kit/sensor/frame/vector_types.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace prep_layouts {

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

// A layout of the picture, as it reaches a frame: `image`, or packed words.
struct Layout {
  std::string name;
  sensor::YuvImage image;
  const std::uint32_t* packed = nullptr;  // RgbdFrame::color_packed
};

// The picture, its host bytes, and its layouts, which point into them.
struct Picture {
  std::uint32_t w = 0, h = 0;
  std::uint32_t cw = 0, ch = 0;              // 4:2:0 chroma
  std::vector<std::uint8_t> y, cb, cr;       // 4:2:0 planes
  std::vector<std::uint8_t> cb422, cr422;    // YUYV's chroma, cw x h
  std::vector<std::uint8_t> i420, yuyv, mono;  // host bytes
  std::vector<std::uint32_t> packed;           // R'G'B' words, junk high byte
  std::vector<Layout> layouts;
};

// `bytes`, padded to whole words, in a device storage buffer; null if it
// cannot be made.
inline std::shared_ptr<const vkc::Buffer> on_device(
    vkc::Device& device, vkc::Allocator& allocator,
    std::vector<std::uint8_t> bytes) {
  bytes.resize((bytes.size() + 3) & ~std::size_t{3}, 0xAB);
  auto made = vkc::device_storage_buffer(allocator, bytes.size());
  if (!made.ok() ||
      !vkc::test::write_back(device, allocator, made.value(), bytes).ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

inline std::uint8_t byte(double v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
}

// A w x h picture (w even, for YUYV) in every layout; a layout whose buffer
// or image cannot be made is left out, which a caller's count shows.
inline std::unique_ptr<Picture> make_picture(vkc::Device& device,
                                             vkc::Allocator& allocator,
                                             std::uint32_t w,
                                             std::uint32_t h) {
  auto p = std::make_unique<Picture>();
  p->w = w;
  p->h = h;
  p->cw = (w + 1) / 2;
  p->ch = (h + 1) / 2;
  p->y.resize(std::size_t{w} * h);
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t x = 0; x < w; ++x) {
      p->y[std::size_t{r} * w + x] =
          byte(128.0 + 127.0 * std::sin(0.031 * x + 0.017 * r));
    }
  }
  const auto cb_at = [](double x, double r) {
    return byte(128.0 + 120.0 * std::cos(0.09 * x + 0.04 * r));
  };
  const auto cr_at = [](double x, double r) {
    return byte(128.0 + 120.0 * std::sin(0.1 * r - 0.06 * x));
  };
  p->cb.resize(std::size_t{p->cw} * p->ch);
  p->cr.resize(p->cb.size());
  for (std::uint32_t r = 0; r < p->ch; ++r) {
    for (std::uint32_t x = 0; x < p->cw; ++x) {
      p->cb[std::size_t{r} * p->cw + x] = cb_at(x, 2.0 * r);
      p->cr[std::size_t{r} * p->cw + x] = cr_at(x, 2.0 * r);
    }
  }
  p->cb422.resize(std::size_t{p->cw} * h);
  p->cr422.resize(p->cb422.size());
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t x = 0; x < p->cw; ++x) {
      p->cb422[std::size_t{r} * p->cw + x] = cb_at(x, r);
      p->cr422[std::size_t{r} * p->cw + x] = cr_at(x, r);
    }
  }

  // I420, planes one after another, rows packed; on the device and the host.
  p->i420 = p->y;
  p->i420.insert(p->i420.end(), p->cb.begin(), p->cb.end());
  p->i420.insert(p->i420.end(), p->cr.begin(), p->cr.end());
  sensor::YuvImage i420;
  i420.layout = sensor::YuvLayout::I420;
  i420.width = w;
  i420.height = h;
  i420.kr = 0.2126f;
  i420.kb = 0.0722f;
  i420.full_range = false;
  i420.offset[1] = p->y.size();
  i420.offset[2] = p->y.size() + p->cb.size();
  i420.stride[0] = w;
  i420.stride[1] = i420.stride[2] = p->cw;
  sensor::YuvImage host = i420;
  host.host = p->i420.data();
  i420.device = on_device(device, allocator, p->i420);
  if (i420.device != nullptr) p->layouts.push_back({"I420 buffer", i420});
  p->layouts.push_back({"I420 host", host});

  // NV12: Cb and Cr interleaved, in a buffer and as two plane images.
  std::vector<std::uint8_t> cbcr(2 * p->cb.size());
  for (std::size_t i = 0; i < p->cb.size(); ++i) {
    cbcr[2 * i] = p->cb[i];
    cbcr[2 * i + 1] = p->cr[i];
  }
  std::vector<std::uint8_t> nv12_bytes = p->y;
  nv12_bytes.insert(nv12_bytes.end(), cbcr.begin(), cbcr.end());
  sensor::YuvImage nv12 = i420;
  nv12.layout = sensor::YuvLayout::Nv12;
  nv12.offset[1] = p->y.size();
  nv12.offset[2] = 0;
  nv12.stride[1] = 2 * std::size_t{p->cw};
  nv12.stride[2] = 0;
  nv12.device = on_device(device, allocator, nv12_bytes);
  if (nv12.device != nullptr) p->layouts.push_back({"NV12 buffer", nv12});
  sensor::YuvImage images = nv12;
  images.device = nullptr;
  images.offset[1] = 0;
  images.stride[0] = images.stride[1] = 0;
  images.image[0] = test_image::with_corner(
      device, allocator, VK_FORMAT_R8_UNORM, 1, w, h, p->y.data(), w, h,
      VK_IMAGE_USAGE_SAMPLED_BIT);
  images.image[1] = test_image::with_corner(
      device, allocator, VK_FORMAT_R8G8_UNORM, 2, p->cw, p->ch, cbcr.data(),
      2 * p->cw, p->ch, VK_IMAGE_USAGE_SAMPLED_BIT);
  if (images.image[0] != nullptr && images.image[1] != nullptr) {
    p->layouts.push_back({"NV12 images", images});
  }

  // YUYV on the host, BT.601 limited range, as a RealSense colour camera's.
  p->yuyv.resize(2 * std::size_t{w} * h);
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t i = 0; i < w / 2; ++i) {
      std::uint8_t* at = &p->yuyv[(std::size_t{r} * w + 2 * i) * 2];
      at[0] = p->y[std::size_t{r} * w + 2 * i];
      at[1] = p->cb422[std::size_t{r} * p->cw + i];
      at[2] = p->y[std::size_t{r} * w + 2 * i + 1];
      at[3] = p->cr422[std::size_t{r} * p->cw + i];
    }
  }
  sensor::YuvImage yuyv;
  yuyv.layout = sensor::YuvLayout::Yuyv;
  yuyv.host = p->yuyv.data();
  yuyv.stride[0] = 2 * std::size_t{w};
  yuyv.width = w;
  yuyv.height = h;
  yuyv.kr = 0.299f;
  yuyv.kb = 0.114f;
  yuyv.full_range = false;
  p->layouts.push_back({"YUYV host", yuyv});

  // Luma alone on the host, full range, as an infrared camera's.
  p->mono = p->y;
  sensor::YuvImage mono;
  mono.layout = sensor::YuvLayout::Luma;
  mono.host = p->mono.data();
  mono.stride[0] = w;
  mono.width = w;
  mono.height = h;
  p->layouts.push_back({"luma host", mono});

  // Packed R'G'B' words, as a dataset decodes them, the high byte junk.
  p->packed.resize(std::size_t{w} * h);
  for (std::size_t i = 0; i < p->packed.size(); ++i) {
    const std::uint32_t x = static_cast<std::uint32_t>(i % w);
    const std::uint32_t r = static_cast<std::uint32_t>(i / w);
    p->packed[i] = p->y[i] | (std::uint32_t{cb_at(x, r)} << 8) |
                   (std::uint32_t{cr_at(x, r)} << 16) | 0x5A000000u;
  }
  Layout packed;
  packed.name = "packed host";
  packed.packed = p->packed.data();
  p->layouts.push_back(packed);
  return p;
}

// --- the pass on the host -----------------------------------------------------

inline float f32(double v) { return static_cast<float>(v); }

// Where the lens images pinhole pixel (u, v), in float, as lens.glsl does.
inline sensor::Vec2f source_pixel(const camera::CameraModel& c, float u,
                                  float v) {
  const camera::RationalDistortion& d = c.distortion;
  const float k[8] = {f32(d.k1), f32(d.k2), f32(d.p1), f32(d.p2),
                      f32(d.k3), f32(d.k4), f32(d.k5), f32(d.k6)};
  const float fx = f32(c.intrinsics.fx), fy = f32(c.intrinsics.fy);
  const float cx = f32(c.intrinsics.cx), cy = f32(c.intrinsics.cy);
  sensor::Vec2f out;
  camera::distort_rational(k, (u - cx) / fx, (v - cy) / fy, &out.x, &out.y);
  return sensor::Vec2f(out.x * fx + cx, out.y * fy + cy);
}

// Bilinear over a w x h byte plane, centres at integer coordinates, clamped
// to the edge.
inline float bilinear(const std::vector<std::uint8_t>& plane, std::uint32_t w,
                      std::uint32_t h, float x, float y) {
  x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
  y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
  const auto x0 = static_cast<std::uint32_t>(x);
  const auto y0 = static_cast<std::uint32_t>(y);
  const std::uint32_t x1 = std::min(x0 + 1, w - 1);
  const std::uint32_t y1 = std::min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const auto t = [&](std::uint32_t a, std::uint32_t b) {
    return static_cast<float>(plane[std::size_t{b} * w + a]);
  };
  const float top = t(x0, y0) + (t(x1, y0) - t(x0, y0)) * fx;
  const float bottom = t(x0, y1) + (t(x1, y1) - t(x0, y1)) * fx;
  return top + (bottom - top) * fy;
}

// The R'G'B' the pass forms for `layout` at pinhole pixel (u, v) through
// `cam`, before it is clamped and coded: the matrix's output may leave
// [0, 1]. False where the lens maps the pixel outside the picture.
inline bool rgb_at(const Picture& p, const Layout& layout,
                   const camera::CameraModel& cam, std::uint32_t u,
                   std::uint32_t v, sensor::Vec3f* rgb) {
  const sensor::Vec2f s =
      source_pixel(cam, static_cast<float>(u), static_cast<float>(v));
  if (!(s.x >= -0.5f && s.y >= -0.5f && s.x <= p.w - 0.5f &&
        s.y <= p.h - 0.5f)) {
    return false;
  }
  if (layout.packed != nullptr) {
    for (int k = 0; k < 3; ++k) {
      // The channel's bilinear sample, from the words around `s`.
      const float x = std::clamp(s.x, 0.0f, static_cast<float>(p.w - 1));
      const float y = std::clamp(s.y, 0.0f, static_cast<float>(p.h - 1));
      const auto x0 = static_cast<std::uint32_t>(x);
      const auto y0 = static_cast<std::uint32_t>(y);
      const std::uint32_t x1 = std::min(x0 + 1, p.w - 1);
      const std::uint32_t y1 = std::min(y0 + 1, p.h - 1);
      const float fx = x - static_cast<float>(x0);
      const float fy = y - static_cast<float>(y0);
      const auto t = [&](std::uint32_t a, std::uint32_t b) {
        return static_cast<float>(
            (p.packed[std::size_t{b} * p.w + a] >> (8 * k)) & 0xFFu);
      };
      const float top = t(x0, y0) + (t(x1, y0) - t(x0, y0)) * fx;
      const float bottom = t(x0, y1) + (t(x1, y1) - t(x0, y1)) * fx;
      (*rgb)[k] = (top + (bottom - top) * fy) / 255.0f;
    }
    return true;
  }
  const sensor::YuvImage& im = layout.image;
  const float yv = bilinear(p.y, p.w, p.h, s.x, s.y);
  const float luma = im.full_range ? yv / 255.0f : (yv - 16.0f) / 219.0f;
  if (im.layout == sensor::YuvLayout::Luma) {
    *rgb = sensor::Vec3f(luma);
    return true;
  }
  const auto siting = sensor::chroma_offset(im.chroma_location);
  float cbv = 0.0f, crv = 0.0f;
  if (im.layout == sensor::YuvLayout::Yuyv) {
    const float cx = (s.x - siting[0]) * 0.5f;
    cbv = bilinear(p.cb422, p.cw, p.h, cx, s.y);
    crv = bilinear(p.cr422, p.cw, p.h, cx, s.y);
  } else {
    const float cx = (s.x - siting[0]) * 0.5f;
    const float cy = (s.y - siting[1]) * 0.5f;
    cbv = bilinear(p.cb, p.cw, p.ch, cx, cy);
    crv = bilinear(p.cr, p.cw, p.ch, cx, cy);
  }
  const float range = im.full_range ? 255.0f : 224.0f;
  const float pb = (cbv - 128.0f) / range;
  const float pr = (crv - 128.0f) / range;
  const float r = luma + 2.0f * (1.0f - im.kr) * pr;
  const float b = luma + 2.0f * (1.0f - im.kb) * pb;
  const float g = (luma - im.kr * r - im.kb * b) / (1.0f - im.kr - im.kb);
  *rgb = sensor::Vec3f(r, g, b);
  return true;
}

// A value as the pass codes it: clamped to [0, 1], rounded to 0..255.
inline std::uint32_t code(float c) {
  return static_cast<std::uint32_t>(std::clamp(c, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// The widest gap between two words' R, G and B codes.
inline int channel_diff(std::uint32_t a, std::uint32_t b) {
  int worst = 0;
  for (int k = 0; k < 3; ++k) {
    const int ca = static_cast<int>((a >> (8 * k)) & 0xFFu);
    const int cb = static_cast<int>((b >> (8 * k)) & 0xFFu);
    worst = std::max(worst, std::abs(ca - cb));
  }
  return worst;
}

}  // namespace prep_layouts
