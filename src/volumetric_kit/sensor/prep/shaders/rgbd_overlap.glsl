// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A frame's two cameras, for the passes that keep only what both see: the
// colour pass zeroes colour outside the depth camera's view
// (undistort_color.comp, undistort_color_images.comp), and the depth pass
// zeroes depth on any colour pixel the colour pass does not keep
// (undistort_depth.comp). Both bind one buffer, gpu_frame_prep.cpp's
// OverlapParams, written for every frame with colour.

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

// The point `q`, in the depth camera's frame, where colour pinhole pixel
// `pixel`'s ray lies `z` metres deep for the depth camera, a depth that
// changes linearly along the ray. False where the ray reaches that depth
// only behind the colour camera, or never.
bool ray_at_depth(RgbdOverlap o, vec2 pixel, float z, out vec3 q) {
  vec2 n =
      (pixel - vec2(o.color.cx, o.color.cy)) / vec2(o.color.fx, o.color.fy);
  vec3 dir = mat3(o.color_to_depth) * vec3(n, 1.0);
  vec3 origin = o.color_to_depth[3].xyz;
  float t = (z - origin.z) / dir.z;
  if (!(t > 0.0)) {
    return false;  // behind the colour camera, or not a number
  }
  q = origin + t * dir;
  return true;
}

// Whether the depth camera sees colour pinhole pixel `pixel`'s ray over the
// frame's whole depth range: its points near_m and far_m deep for the depth
// camera, the ends of the part of the ray a depth sample can lie on, are both
// covered. The undistorted depth picture is a pinhole image, so what it sees
// is a convex cone, and a segment lies inside it when both its ends do; a
// lens that misses part of the picture is tested at the ends alone.
bool depth_sees(RgbdOverlap o, vec2 pixel) {
  vec3 near_end;
  vec3 far_end;
  return ray_at_depth(o, pixel, o.near_m, near_end) &&
         depth_covers(o.depth, near_end) &&
         ray_at_depth(o, pixel, o.far_m, far_end) &&
         depth_covers(o.depth, far_end);
}

#endif  // VS_PREP_RGBD_OVERLAP_GLSL
