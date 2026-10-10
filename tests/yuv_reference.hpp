// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The YCbCr-to-RGB conversion written out from the standards' constants, the
// reference the picture tests hold the GPU pass's conversion, and the weights a
// picture is described with, to.

#include <algorithm>
#include <array>
#include <cmath>

#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace yuv_reference {

// @p m's red and blue luma weights.
inline std::array<double, 2> weights(
    volumetric_kit::recon::sensor::VideoColorMatrix m) {
  using volumetric_kit::recon::sensor::VideoColorMatrix;
  switch (m) {
    case VideoColorMatrix::Bt601:
      return {0.299, 0.114};
    case VideoColorMatrix::Bt709:
      return {0.2126, 0.0722};
    case VideoColorMatrix::Smpte240m:
      return {0.212, 0.087};
    case VideoColorMatrix::Fcc:
      return {0.30, 0.11};
    case VideoColorMatrix::Bt2020:
      break;
  }
  return {0.2627, 0.0593};
}

// Whether @p image says it is coded in @p m at @p full_range.
inline bool coded_in(const volumetric_kit::recon::sensor::YuvImage& image,
                     volumetric_kit::recon::sensor::VideoColorMatrix m,
                     bool full_range) {
  const std::array<double, 2> k = weights(m);
  return std::abs(image.kr - k[0]) < 1e-6 && std::abs(image.kb - k[1]) < 1e-6 &&
         image.full_range == full_range;
}

inline std::array<int, 3> rgb(double y, double u, double v,
                              volumetric_kit::recon::sensor::VideoColorMatrix m,
                              bool full_range) {
  const std::array<double, 2> k = weights(m);
  const double kr = k[0];
  const double kb = k[1];
  const double kg = 1.0 - kr - kb;
  const double luma = full_range ? y / 255.0 : (y - 16) / 219.0;
  const double cb = (u - 128) / (full_range ? 255.0 : 224.0);
  const double cr = (v - 128) / (full_range ? 255.0 : 224.0);
  const double r = luma + 2.0 * (1.0 - kr) * cr;
  const double g =
      luma - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr;
  const double b = luma + 2.0 * (1.0 - kb) * cb;
  const auto code = [](double c) {
    return static_cast<int>(std::lround(std::clamp(c, 0.0, 1.0) * 255.0));
  };
  return {code(r), code(g), code(b)};
}

}  // namespace yuv_reference
