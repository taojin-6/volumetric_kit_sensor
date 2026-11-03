// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A picture's planes in a buffer, as undistort_color.glsl reads them: the
// colour pass's buffer entry (undistort_color.comp) and the stereo
// rectifier's kernel (stereo/shaders/rectify.comp) bind them at binding 0 and
// read them through these. Include after undistort_color.glsl, whose push
// constants place the planes.

#ifndef VS_PREP_COLOR_PLANES_GLSL
#define VS_PREP_COLOR_PLANES_GLSL

// The planes, four bytes to a word: luma rows y_stride bytes apart from
// y_offset, samples y_step bytes apart (2 in YUYV, else 1); Cb and Cr rows
// cb_stride and cr_stride bytes apart from cb_offset and cr_offset, samples
// c_step bytes apart: 1 in I420's planes of their own, 2 in NV12's
// interleaved one, where Cr sits a byte after Cb, 4 in YUYV's Y0 Cb Y1 Cr;
// luma alone has none. V210's one plane is words, rows starting on one, in
// YuvLayout::V210's order (yuv_image.hpp). With packed_rgb, the picture's
// own words instead, R | G << 8 | B << 16, from byte 0.
layout(set = 0, binding = 0, scalar) readonly buffer Planes { uint planes[]; };

float byte_at(uint base, uint row_bytes, uint step, uint x, uint y) {
  uint i = base + y * row_bytes + x * step;
  return float((planes[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu);
}

// V210's sample k (0 to 11) of six-pixel group g of row y, read at 10 bits
// and returned on the 8-bit scale, code / 4: so the levels (16..235 is
// 64..940) and the conversion are the 8-bit layouts', and a picture of
// 8-bit codes times 4 converts exactly as they do.
float v210_at(uint base, uint row_bytes, uint g, uint k, uint y) {
  uint word = planes[(base + y * row_bytes) / 4u + g * 4u + k / 3u];
  return float((word >> (10u * (k % 3u))) & 0x3FFu) * 0.25;
}

float luma_at(uint x, uint y) {
  if (pc.v210 != 0u) {
    return v210_at(pc.y_offset, pc.y_stride, x / 6u, 2u * (x % 6u) + 1u, y);
  }
  return byte_at(pc.y_offset, pc.y_stride, pc.y_step, x, y);
}

// Chroma sample x is pixel 2x's: in V210, of group x / 3, its Cb at 4 (x % 3)
// and its Cr two samples on.
vec2 chroma_at(uint x, uint y) {
  if (pc.v210 != 0u) {
    uint k = 4u * (x % 3u);
    return vec2(v210_at(pc.cb_offset, pc.cb_stride, x / 3u, k, y),
                v210_at(pc.cr_offset, pc.cr_stride, x / 3u, k + 2u, y));
  }
  return vec2(byte_at(pc.cb_offset, pc.cb_stride, pc.c_step, x, y),
              byte_at(pc.cr_offset, pc.cr_stride, pc.c_step, x, y));
}

// A packed word's R, G and B; the high byte is never read.
vec3 rgb_at(uint x, uint y) {
  uint word = planes[y * pc.cam.width + x];
  return vec3(float(word & 0xFFu), float((word >> 8) & 0xFFu),
              float((word >> 16) & 0xFFu));
}

#endif  // VS_PREP_COLOR_PLANES_GLSL
