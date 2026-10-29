// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The pinhole camera on the device: recon's GLSL projection, which its kernels
// and its consumers' include (spelled
// "volumetric_kit/recon/core/shaders/pinhole.glsl", through the core's
// INCLUDE_TARGETS volumetric_kit::recon_core). The camera is camera_model.hpp's:
// +X right, +Y down, +Z forward, pixel centres at integer coordinates, and a
// rigid cam_to_world pose.
//
// The functions take values, never a struct, so a kernel whose camera is a
// scalar-layout block (recon's) and one whose camera is std430 (calib's) both
// pass theirs field by field. `k` is (fx, fy, cx, cy) in pixels of `size`.
// tests/core_pinhole_gpu_test.cpp holds them to the host's arithmetic.

#ifndef VR_CORE_PINHOLE_GLSL
#define VR_CORE_PINHOLE_GLSL

// A world point in the space of the camera posed at `cam_to_world`:
// R^T (world - t), the rigid inverse, with no general matrix inverse.
vec3 vr_world_to_camera(mat4 cam_to_world, vec3 world) {
  return transpose(mat3(cam_to_world)) * (world - cam_to_world[3].xyz);
}

// The camera-space point at pixel `px` and depth `z` along +Z:
// x = (u - cx) z / fx, y = (v - cy) z / fy.
vec3 vr_unproject_pinhole(vec2 px, float z, vec4 k) {
  return vec3((px - k.zw) * z / k.xy, z);
}

// The pixel camera-space `p_cam` lands on, u = fx x / z + cx and
// v = fy y / z + cy, wherever that is. Returns true and sets `px` when the
// point is in front of the camera (z > 0), even if the pixel lies outside the
// image or is not finite (an infinite or NaN x, y or intrinsic); otherwise
// false, with `px` unset. vr_project_pinhole adds the image's bounds.
//
// The depth test is negated, so a NaN depth fails it: every comparison with
// NaN is false, so the direct form would divide by it.
bool vr_pinhole_pixel(vec3 p_cam, vec4 k, out vec2 px) {
  if (!(p_cam.z > 0.0)) {
    return false;  // behind the camera, or not a number
  }
  px = k.xy * (p_cam.xy / p_cam.z) + k.zw;
  return true;
}

// vr_pinhole_pixel within the image: returns true and sets `px` when the point
// is in front of the camera and its pixel lies in [0, width) x [0, height);
// otherwise false, with `px` unset.
//
// The bounds test is negated too, so a NaN pixel fails it: the direct form
// would hand the caller an index nothing else bounds.
bool vr_project_pinhole(vec3 p_cam, vec4 k, uvec2 size, out vec2 px) {
  vec2 q;
  if (!vr_pinhole_pixel(p_cam, k, q)) {
    return false;
  }
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
