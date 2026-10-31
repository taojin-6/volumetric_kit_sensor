// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The exact piecewise sRGB transfer on the device: frame/color_space.hpp's
// srgb_to_linear and linear_to_srgb, which the sensor's kernels and its
// consumers' include (spelled "volumetric_kit/sensor/frame/shaders/srgb.glsl",
// through the core's INCLUDE_TARGETS volumetric_kit::sensor_frame). Values
// outside [0, 1] take the same branches, unclamped, as on the host; a caller
// clamps. tests/frame_srgb_gpu_test.cpp holds them to the host's curve on all
// 256 codes.

#ifndef VS_FRAME_SRGB_GLSL
#define VS_FRAME_SRGB_GLSL

// An encoded channel as linear light.
float vs_srgb_to_linear(float e) {
  return e <= 0.04045 ? e * (1.0 / 12.92)
                      : pow((e + 0.055) * (1.0 / 1.055), 2.4);
}

// A linear channel encoded: the inverse of vs_srgb_to_linear.
float vs_linear_to_srgb(float l) {
  return l <= 0.0031308 ? l * 12.92 : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}

vec3 vs_srgb_to_linear(vec3 e) {
  return vec3(vs_srgb_to_linear(e.r), vs_srgb_to_linear(e.g),
              vs_srgb_to_linear(e.b));
}

vec3 vs_linear_to_srgb(vec3 l) {
  return vec3(vs_linear_to_srgb(l.r), vs_linear_to_srgb(l.g),
              vs_linear_to_srgb(l.b));
}

#endif  // VS_FRAME_SRGB_GLSL
