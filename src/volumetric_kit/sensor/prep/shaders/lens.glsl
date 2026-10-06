// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The device mirror of camera/projection.hpp: a camera before undistortion,
// narrowed to float, and the forward lens model both undistortion passes
// sample through. Keep in lockstep with camera::distort_rational;
// tests/sensor_gpu_frame_prep_test holds the passes to a host reference built
// on it.

#ifndef VR_SENSOR_LENS_GLSL
#define VR_SENSOR_LENS_GLSL

#extension GL_EXT_scalar_block_layout : require

// A camera::CameraModel narrowed to float (gpu_frame_prep.cpp's LensParams),
// under scalar layout: 14 four-byte fields, 56 bytes.
struct LensCamera {
  float fx;
  float fy;
  float cx;
  float cy;
  uint width;
  uint height;
  float k1;
  float k2;
  float p1;
  float p2;
  float k3;
  float k4;
  float k5;
  float k6;
};

// Where the lens images a normalized pinhole point: OpenCV's rational model.
vec2 distort_normalized(LensCamera c, vec2 p) {
  float r2 = p.x * p.x + p.y * p.y;
  float r4 = r2 * r2;
  float r6 = r4 * r2;
  float radial = (1.0 + c.k1 * r2 + c.k2 * r4 + c.k3 * r6) /
                 (1.0 + c.k4 * r2 + c.k5 * r4 + c.k6 * r6);
  float xy = p.x * p.y;
  return vec2(p.x * radial + 2.0 * c.p1 * xy + c.p2 * (r2 + 2.0 * p.x * p.x),
              p.y * radial + c.p1 * (r2 + 2.0 * p.y * p.y) + 2.0 * c.p2 * xy);
}

// The captured-image pixel that pinhole pixel `pixel` of the undistorted image
// shows, the two sharing intrinsics; pixel centres at integer coordinates.
vec2 source_pixel(LensCamera c, vec2 pixel) {
  vec2 p = vec2((pixel.x - c.cx) / c.fx, (pixel.y - c.cy) / c.fy);
  vec2 d = distort_normalized(c, p);
  return vec2(d.x * c.fx + c.cx, d.y * c.fy + c.cy);
}

#endif  // VR_SENSOR_LENS_GLSL
