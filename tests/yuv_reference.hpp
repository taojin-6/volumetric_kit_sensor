// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The YCbCr-to-RGB conversion written out from the standards' constants, the
// reference the video tests hold swscale's output to.

#include <algorithm>
#include <array>
#include <cmath>

#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace yuv_reference {

inline std::array<int, 3> rgb(int y, int u, int v,
                              volumetric_kit::recon::sensor::VideoColorMatrix m,
                              bool full_range) {
  using volumetric_kit::recon::sensor::VideoColorMatrix;
  const double kr = m == VideoColorMatrix::Bt601   ? 0.299
                    : m == VideoColorMatrix::Bt709 ? 0.2126
                                                   : 0.2627;
  const double kb = m == VideoColorMatrix::Bt601   ? 0.114
                    : m == VideoColorMatrix::Bt709 ? 0.0722
                                                   : 0.0593;
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
