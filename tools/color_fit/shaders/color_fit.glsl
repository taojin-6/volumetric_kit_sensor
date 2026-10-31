// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// What the colour fit's kernels share: a set's cameras as color_fit.cpp packs
// them (GpuCamera, scalar layout), a kept sample (color_fit.hpp's Sample), a
// camera's response in the solve (ResponseParams), and the hash that keeps
// samples and picks their hold-out half. The projection and the sRGB curve
// are the package's exported ones, through INCLUDE_TARGETS
// volumetric_kit::sensor_frame as a consumer's kernels reach them.

#ifndef VS_COLOR_FIT_GLSL
#define VS_COLOR_FIT_GLSL

#include "volumetric_kit/sensor/frame/shaders/pinhole.glsl"
#include "volumetric_kit/sensor/frame/shaders/srgb.glsl"

// One camera of the set: its prepared depth and colour pinholes and poses,
// and where its pixels start in the set's depth and colour buffers.
struct Camera {
  vec4 depth_k;  // fx, fy, cx, cy
  uint depth_width;
  uint depth_height;
  mat4 depth_to_world;
  vec4 color_k;
  uint color_width;
  uint color_height;
  mat4 color_to_world;
  uint depth_at;  // the camera's first depth sample in the depth buffer
  uint color_at;  // its first pixel in the word and blurred buffers
};

// A kept sample: A's and B's colour at sigma 8 in linear light, each pixel's
// squared pinhole radius, uncapped, and `a | b << 8 | held << 16`, held 1
// for the held-out half.
struct Sample {
  vec3 a;
  vec3 b;
  float qa;
  float qb;
  uint cams;
};

// A camera's response as the solve evaluates it: gain * F(q), F = 1 + a q +
// b q^2, q capped at q_max = r_max^2.
struct ResponseParams {
  vec3 gain;
  float a;
  float b;
  float q_max;
};

// lowbias32 (Chris Wellons): a 32-bit integer hash with no visible bias.
uint vs_cf_hash(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// F at a squared radius, held beyond q_max.
float vs_cf_falloff(ResponseParams p, float q) {
  const float c = min(q, p.q_max);
  return 1.0 + p.a * c + p.b * c * c;
}

#endif  // VS_COLOR_FIT_GLSL
