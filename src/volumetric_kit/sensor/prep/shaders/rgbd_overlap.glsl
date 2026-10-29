// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A frame's two cameras, for the passes that keep only what both see: the
// depth pass zeroes depth outside the colour camera's view
// (undistort_depth.comp), and the colour pass zeroes colour outside the depth
// camera's (undistort_color.comp, undistort_color_images.comp). Both bind one
// buffer, gpu_frame_prep.cpp's OverlapParams, written for every frame with
// colour.

#ifndef VS_PREP_RGBD_OVERLAP_GLSL
#define VS_PREP_RGBD_OVERLAP_GLSL

#extension GL_GOOGLE_include_directive : require
#include "lens.glsl"
#include "volumetric_kit/sensor/frame/shaders/pinhole.glsl"

// OverlapParams under scalar layout: the two cameras, the rigid transforms
// between their frames, and the frame's depth range in metres.
struct RgbdOverlap {
  LensCamera color;
  mat4 depth_to_color;
  LensCamera depth;
  mat4 color_to_depth;
  float near_m;  // RgbdFrame::min_depth
  float far_m;   // RgbdFrame::max_depth
};

// Whether coordinate `p` rounds, half up, to a pixel of a w x h image; a NaN
// does not.
bool rounds_into(vec2 p, uint w, uint h) {
  vec2 r = floor(p + 0.5);
  return r.x >= 0.0 && r.y >= 0.0 && r.x < float(w) && r.y < float(h);
}

// Whether the depth pass covers point `q` of the depth camera's frame: in
// front of the camera, on a pinhole pixel that rounds into the picture, whose
// source in the captured image does too -- the pass's own rounding and lens
// test.
bool depth_covers(LensCamera d, vec3 q) {
  vec2 px;
  if (!vs_pinhole_pixel(q, vec4(d.fx, d.fy, d.cx, d.cy), px) ||
      !rounds_into(px, d.width, d.height)) {
    return false;
  }
  return rounds_into(source_pixel(d, floor(px + 0.5)), d.width, d.height);
}

// Whether the depth camera sees colour pinhole pixel `pixel`'s ray over the
// whole depth range: its points at near_m and far_m along the colour camera's
// axis are both covered. The undistorted depth picture is a pinhole image, so
// what it sees is a convex cone, and a segment lies inside it when both its
// ends do; a lens that misses part of the picture is tested at the ends alone.
bool depth_sees(RgbdOverlap o, vec2 pixel) {
  vec4 k = vec4(o.color.fx, o.color.fy, o.color.cx, o.color.cy);
  vec3 near_end = vs_unproject_pinhole(pixel, o.near_m, k);
  vec3 far_end = vs_unproject_pinhole(pixel, o.far_m, k);
  return depth_covers(o.depth,
                      (o.color_to_depth * vec4(near_end, 1.0)).xyz) &&
         depth_covers(o.depth, (o.color_to_depth * vec4(far_end, 1.0)).xyz);
}

#endif  // VS_PREP_RGBD_OVERLAP_GLSL
