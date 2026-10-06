// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The YCbCr-to-RGB conversion written out from the standards' constants, the
// reference the video tests hold the GPU pass's conversion to.

#include <algorithm>
#include <array>
#include <cmath>

#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace yuv_reference {

inline std::array<int, 3> rgb(double y, double u, double v,
                              volumetric_kit::recon::sensor::VideoColorMatrix m,
                              bool full_range) {
  using volumetric_kit::recon::sensor::VideoColorMatrix;
  double kr = 0.2627;  // BT.2020
  double kb = 0.0593;
  switch (m) {
    case VideoColorMatrix::Bt601:
      kr = 0.299, kb = 0.114;
      break;
    case VideoColorMatrix::Bt709:
      kr = 0.2126, kb = 0.0722;
      break;
    case VideoColorMatrix::Smpte240m:
      kr = 0.212, kb = 0.087;
      break;
    case VideoColorMatrix::Fcc:
      kr = 0.30, kb = 0.11;
      break;
    case VideoColorMatrix::Bt2020:
      break;
  }
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
