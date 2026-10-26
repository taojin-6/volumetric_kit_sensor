// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The colour pass's body, which its two entries share: undistort_color.comp
// reads the picture from a buffer, undistort_color_images.comp from NV12's
// two plane images where they are. One thread per pixel of the pinhole image:
// find where the lens puts it in the captured picture, sample luma and chroma
// there bilinearly, each at its own siting, and convert by the picture's
// matrix and range. One pass, so no intermediate R'G'B' image is written at
// the captured size. The high byte is the pixel's coverage: 0xFF where the
// lens maps inside the captured picture, 0 where it maps outside, which is
// black and which fusion then skips (ColorFrame::coverage_in_alpha) rather
// than fusing as colour.

#ifndef VR_SENSOR_UNDISTORT_COLOR_GLSL
#define VR_SENSOR_UNDISTORT_COLOR_GLSL

#extension GL_GOOGLE_include_directive : require
#include "lens.glsl"

layout(local_size_x = 256) in;

// gpu_frame_prep.cpp's ColorParams. The offsets, strides and c_step place the
// planes in a buffer, and only that entry reads them.
layout(push_constant, scalar) uniform Params {
  LensCamera cam;
  uint y_offset;   // bytes
  uint cb_offset;  // bytes
  uint cr_offset;  // bytes
  uint y_stride;   // bytes a luma row
  uint cb_stride;  // bytes a Cb row
  uint cr_stride;  // bytes a Cr row
  uint c_step;     // bytes between chroma samples
  float kr;
  float kb;
  uint full_range;
  float chroma_x;
  float chroma_y;
  uint packed_rgb;  // packed R'G'B' words, not Y'CbCr planes
} pc;

// Each entry's: the codes, 0 to 255, of the picture's luma and of its chroma
// (Cb, Cr) at sample (x, y), counted from the picture's corner.
float luma_at(uint x, uint y);
vec2 chroma_at(uint x, uint y);

// Bilinear over a plane of w x h samples, centres at integer coordinates,
// clamped to the edge: the corner samples lo and hi, and the weights f.
void bilinear_taps(uint w, uint h, vec2 p, out uvec2 lo, out uvec2 hi,
                   out vec2 f) {
  p = clamp(p, vec2(0.0), vec2(float(w - 1u), float(h - 1u)));
  lo = uvec2(floor(p));
  hi = min(lo + 1u, uvec2(w - 1u, h - 1u));
  f = p - vec2(lo);
}

float bilinear(float aa, float ba, float ab, float bb, vec2 f) {
  return mix(mix(aa, ba, f.x), mix(ab, bb, f.x), f.y);
}
vec2 bilinear(vec2 aa, vec2 ba, vec2 ab, vec2 bb, vec2 f) {
  return mix(mix(aa, ba, f.x), mix(ab, bb, f.x), f.y);
}
vec3 bilinear(vec3 aa, vec3 ba, vec3 ab, vec3 bb, vec2 f) {
  return mix(mix(aa, ba, f.x), mix(ab, bb, f.x), f.y);
}

uint code(float c) { return uint(clamp(c, 0.0, 1.0) * 255.0 + 0.5); }

// Where the lens puts pinhole pixel `gid` in the captured picture, `s`; false
// where that is outside it, which the entry writes as 0.
bool source_of(uint gid, out vec2 s) {
  uint w = pc.cam.width;
  uint h = pc.cam.height;
  s = source_pixel(pc.cam, vec2(float(gid % w), float(gid / w)));
  return s.x >= -0.5 && s.y >= -0.5 && s.x <= float(w) - 0.5 &&
         s.y <= float(h) - 0.5;
}

// The picture's Y'CbCr at `s`, converted, as an R'G'B' word with full
// coverage.
uint ycbcr_word(vec2 s) {
  uint w = pc.cam.width;
  uint h = pc.cam.height;
  // The decoder's sample positions: JPEG is centred in both axes; HEVC's
  // default is left-aligned horizontally and centred vertically.
  uint cw = (w + 1u) / 2u;
  uint ch = (h + 1u) / 2u;
  vec2 c = (s - vec2(pc.chroma_x, pc.chroma_y)) * 0.5;
  uvec2 lo, hi;
  vec2 f;
  bilinear_taps(w, h, s, lo, hi, f);
  float y = bilinear(luma_at(lo.x, lo.y), luma_at(hi.x, lo.y),
                     luma_at(lo.x, hi.y), luma_at(hi.x, hi.y), f);
  bilinear_taps(cw, ch, c, lo, hi, f);
  vec2 cbcr = bilinear(chroma_at(lo.x, lo.y), chroma_at(hi.x, lo.y),
                       chroma_at(lo.x, hi.y), chroma_at(hi.x, hi.y), f);

  float luma;
  float pb;
  float pr;
  if (pc.full_range != 0u) {
    luma = y / 255.0;
    pb = (cbcr.x - 128.0) / 255.0;
    pr = (cbcr.y - 128.0) / 255.0;
  } else {
    luma = (y - 16.0) / 219.0;
    pb = (cbcr.x - 128.0) / 224.0;
    pr = (cbcr.y - 128.0) / 224.0;
  }
  float kg = 1.0 - pc.kr - pc.kb;
  float r = luma + 2.0 * (1.0 - pc.kr) * pr;
  float b = luma + 2.0 * (1.0 - pc.kb) * pb;
  float g = (luma - pc.kr * r - pc.kb * b) / kg;
  return code(r) | (code(g) << 8) | (code(b) << 16) | 0xFF000000u;
}

#endif  // VR_SENSOR_UNDISTORT_COLOR_GLSL
