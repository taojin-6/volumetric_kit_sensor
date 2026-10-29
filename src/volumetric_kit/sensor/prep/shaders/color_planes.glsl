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
// luma alone has none. With packed_rgb, the picture's own words instead,
// R | G << 8 | B << 16, from byte 0.
layout(set = 0, binding = 0, scalar) readonly buffer Planes { uint planes[]; };

float byte_at(uint base, uint row_bytes, uint step, uint x, uint y) {
  uint i = base + y * row_bytes + x * step;
  return float((planes[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu);
}

float luma_at(uint x, uint y) {
  return byte_at(pc.y_offset, pc.y_stride, pc.y_step, x, y);
}

vec2 chroma_at(uint x, uint y) {
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
