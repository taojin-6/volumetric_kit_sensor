// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "ffs_kernels.hpp"

namespace volumetric_kit::sensor::ffs {
namespace {

constexpr unsigned kBlock = 256;

struct Normalise {
  float mean[3];
  float deviation[3];
};

// One thread a network pixel, blockIdx.y the view: its three channels.
__global__ void input_kernel(const void* left, const void* right,
                             float* left_input, float* right_input,
                             InputParams p, Normalise n) {
  const unsigned plane = p.net_width * p.net_height;
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= plane) return;
  const unsigned x = i % p.net_width, y = i / p.net_width;
  const unsigned sx = min(x, p.width - 1), sy = min(y, p.height - 1);
  const unsigned at = sy * p.width + sx;
  const void* view = blockIdx.y == 0 ? left : right;
  float* out = blockIdx.y == 0 ? left_input : right_input;
  float value[3];
  if (p.luma) {
    const float v = static_cast<const unsigned char*>(view)[at];
    value[0] = value[1] = value[2] = v;
  } else {
    const unsigned word = static_cast<const unsigned*>(view)[at];
    value[0] = static_cast<float>(word & 0xffu);
    value[1] = static_cast<float>((word >> 8) & 0xffu);
    value[2] = static_cast<float>((word >> 16) & 0xffu);
  }
  for (int c = 0; c < 3; ++c) {
    out[c * plane + i] = (value[c] - n.mean[c]) / n.deviation[c];
  }
}

// One thread a view pixel.
__global__ void depth_kernel(const float* disparity, unsigned short* depth,
                             DepthParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.width * p.height) return;
  const unsigned x = i % p.width, y = i / p.width;
  const float d = disparity[y * p.net_width + x];
  unsigned short out = 0;
  // NaN fails every comparison, so it is no return too.
  if (d > 0.0f && d <= static_cast<float>(x)) {
    const float z = p.focal_baseline / d;
    if (z >= p.min_depth && z <= p.max_depth) {
      const float units = rintf(z * p.units_per_metre);
      out = static_cast<unsigned short>(fminf(fmaxf(units, 1.0f), 65535.0f));
    }
  }
  depth[i] = out;
}

}  // namespace

cudaError_t make_input(const void* left, const void* right, float* left_input,
                       float* right_input, const InputParams& params,
                       cudaStream_t stream) {
  const unsigned plane = params.net_width * params.net_height;
  Normalise n{};
  for (int c = 0; c < 3; ++c) {
    n.mean[c] = kMean[c];
    n.deviation[c] = kDeviation[c];
  }
  const dim3 grid((plane + kBlock - 1) / kBlock, 2);
  input_kernel<<<grid, kBlock, 0, stream>>>(left, right, left_input,
                                            right_input, params, n);
  return cudaGetLastError();
}

cudaError_t make_depth(const float* disparity, std::uint16_t* depth,
                       const DepthParams& params, cudaStream_t stream) {
  const unsigned pixels = params.width * params.height;
  depth_kernel<<<(pixels + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      disparity, reinterpret_cast<unsigned short*>(depth), params);
  return cudaGetLastError();
}

}  // namespace volumetric_kit::sensor::ffs
