// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The pinhole camera on the device: recon's one GLSL projection, which its
// kernels and its consumers' include (spelled
// "volumetric_kit/recon/core/shaders/pinhole.glsl", under the include root
// recon_core's VR_SHADER_INCLUDE_DIR names). The camera is camera_model.hpp's:
// +X right, +Y down, +Z forward, pixel centres at integer coordinates.
//
// The functions take values, never a struct, so a kernel whose camera is a
// scalar-layout block (recon's) and one whose camera is std430 (calib's) both
// pass theirs field by field. `k` is (fx, fy, cx, cy) in pixels of `size`.
// tests/core_pinhole_gpu_test.cpp holds them to the host's arithmetic.

#ifndef VR_CORE_PINHOLE_GLSL
#define VR_CORE_PINHOLE_GLSL

// The camera-space point at pixel `px` and depth `z` along +Z:
// x = (u - cx) z / fx, y = (v - cy) z / fy.
vec3 vr_unproject_pinhole(vec2 px, float z, vec4 k) {
  return vec3((px - k.zw) * z / k.xy, z);
}

// Projects camera-space `p_cam`: u = fx x / z + cx, v = fy y / z + cy. Returns
// true and sets `px` when the point is in front of the camera (z > 0) and its
// pixel lies in [0, width) x [0, height); otherwise false, with `px` unset.
//
// Both tests are negated, so a NaN fails them: every comparison with NaN is
// false, so the direct forms would accept a NaN depth or pixel and hand the
// caller an index nothing else bounds.
bool vr_project_pinhole(vec3 p_cam, vec4 k, uvec2 size, out vec2 px) {
  if (!(p_cam.z > 0.0)) {
    return false;  // behind the camera, or not a number
  }
  vec2 q = k.xy * (p_cam.xy / p_cam.z) + k.zw;
  if (!(q.x >= 0.0 && q.x < float(size.x) && q.y >= 0.0 &&
        q.y < float(size.y))) {
    return false;  // outside the image, or not a number
  }
  px = q;
  return true;
}

// The pixel whose centre is nearest `px`, a coordinate vr_project_pinhole
// accepted: round half up, clamped into the image, since the last half pixel
// of [0, width) rounds to `width`.
uvec2 vr_nearest_pixel(vec2 px, uvec2 size) {
  return min(uvec2(floor(px + 0.5)), size - 1u);
}

#endif  // VR_CORE_PINHOLE_GLSL
