// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// FfsStereo's two CUDA kernels (ffs_kernels.cu), on the stream the network
// runs on: the network's input from the rectifier's views, and depth from its
// disparity. Internal.

#include <cuda_runtime_api.h>

#include <cstdint>

namespace volumetric_kit::sensor::ffs {

// FFS's input normalisation (ImageNet's, on 0-255 R'G'B'), which the exported
// model leaves to its caller: (value - mean[c]) / deviation[c].
inline constexpr float kMean[3] = {123.675f, 116.28f, 103.53f};
inline constexpr float kDeviation[3] = {58.395f, 57.12f, 57.375f};

struct InputParams {
  std::uint32_t width = 0, height = 0;          // the views'
  std::uint32_t net_width = 0, net_height = 0;  // the network's, >= the views'
  // Luma8: a byte a pixel, repeated in all three channels; else Rgba8, a word
  // a pixel, R in the low byte.
  bool luma = false;
};

// Both views into the network's two inputs, each float 3 x net_height x
// net_width, normalised; the network's pixels past a view's last row or
// column repeat it.
cudaError_t make_input(const void* left, const void* right, float* left_input,
                       float* right_input, const InputParams& params,
                       cudaStream_t stream);

struct DepthParams {
  std::uint32_t width = 0, height = 0;  // the views'
  std::uint32_t net_width = 0;          // the disparity's row length
  float focal_baseline = 0.0f;          // fx * baseline: depth = this / d
  float min_depth = 0.0f, max_depth = 0.0f;
  float units_per_metre = 0.0f;  // 1 / metres_per_unit
  // Rgba8 views: their coverage bytes (the top byte of each word) are read.
  bool coverage = false;
  // With a spread: a pixel whose spread is above this, or NaN, is no return.
  float max_spread = 0.0f;
};

// The view's width x height pixels of `disparity` (net_width a row) as 16-bit
// depth, rows packed; 0 where the disparity is not above 0, points off the
// right view, or gives a depth outside [min_depth, max_depth]; with `spread`
// (the network's disparity_spread, laid out as `disparity`), where it is
// above `max_spread` or NaN; and, with `coverage`, where the left view's
// pixel or the right view's pixel it matched (the nearest) has coverage 0.
// `spread` null gates nothing; `left` and `right` are the views, read only
// with `coverage`.
cudaError_t make_depth(const float* disparity, const float* spread,
                       const void* left, const void* right,
                       std::uint16_t* depth, const DepthParams& params,
                       cudaStream_t stream);

}  // namespace volumetric_kit::sensor::ffs
