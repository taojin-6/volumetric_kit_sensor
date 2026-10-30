// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A frame's two cameras, for the passes that keep only what both see: the
// colour pass zeroes colour outside the depth camera's view
// (undistort_color.comp, undistort_color_images.comp), and the depth pass
// zeroes depth outside the colour camera's view and on any colour pixel the
// colour pass does not keep (undistort_depth.comp). Both bind one buffer,
// gpu_frame_prep.cpp's OverlapParams, written for every frame with colour.

#ifndef VS_PREP_RGBD_OVERLAP_GLSL
#define VS_PREP_RGBD_OVERLAP_GLSL

#extension GL_GOOGLE_include_directive : require
#include "lens.glsl"
#include "volumetric_kit/sensor/frame/shaders/pinhole.glsl"

// OverlapParams under scalar layout: the two cameras, and the rigid transform
// from the depth camera's frame into the colour camera's.
struct RgbdOverlap {
  LensCamera color;
  mat4 depth_to_color;
  LensCamera depth;
};

// Whether coordinate `p` rounds, half up, to a pixel of a w x h image; a NaN
// does not.
bool rounds_into(vec2 p, uint w, uint h) {
  vec2 r = floor(p + 0.5);
  return r.x >= 0.0 && r.y >= 0.0 && r.x < float(w) && r.y < float(h);
}

// Whether the depth pass covers point or direction `q` of the depth camera's
// frame: in front of the camera, on a pinhole pixel that rounds into the
// picture, whose source in the captured image does too -- the pass's own
// rounding and lens test.
bool depth_covers(LensCamera d, vec3 q) {
  vec2 px;
  if (!vs_pinhole_pixel(q, vec4(d.fx, d.fy, d.cx, d.cy), px) ||
      !rounds_into(px, d.width, d.height)) {
    return false;
  }
  return rounds_into(source_pixel(d, floor(px + 0.5)), d.width, d.height);
}

// Whether the depth camera's field of view holds colour pinhole pixel
// `pixel`'s ray: its direction, turned into the depth camera's frame by R^T,
// the rotation of rigid_inverse(depth_to_color) with no translation, is
// covered by the depth pass.
bool depth_sees(RgbdOverlap o, vec2 pixel) {
  vec2 n =
      (pixel - vec2(o.color.cx, o.color.cy)) / vec2(o.color.fx, o.color.fy);
  // v * M is transpose(M) * v.
  return depth_covers(o.depth, vec3(n, 1.0) * mat3(o.depth_to_color));
}

#endif  // VS_PREP_RGBD_OVERLAP_GLSL
