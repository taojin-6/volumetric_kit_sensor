// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/color_response.hpp
/// @brief A colour camera's response: what its prepared colour is multiplied
///        by, in linear light, so a rig's cameras agree on a surface they
///        share.

#include <algorithm>
#include <array>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/export.hpp"

namespace volumetric_kit::sensor::camera {

/// @brief What a colour camera's prepared colour is multiplied by, in linear
///        BT.709, so the rig's cameras agree on a surface they share:
///        `gain[c] * F(r)`, `F = 1 + a q + b q^2`, `q = min(r, r_max)^2`.
///
/// Per-unit gain and white balance in @ref gain, lens falloff (vignetting) in
/// F. `r = |((x - cx) / fx, (y - cy) / fy)|` at the prepared (pinhole) pixel,
/// whose intrinsics are the captured camera's, so r is about the tangent of
/// the ray's angle off the axis. F is held at its value at @ref r_max beyond
/// it, where a fit has no samples. A response holds at the exposure it was
/// fitted at, with auto white balance on (@ref ColorControls), which the
/// calibration file checks.
///
/// @code
/// camera::ColorResponse r;  // the identity: gain 1, F = 1
/// r.gain = {0.9794, 0.9705, 0.9940};
/// r.a = 0.7822;
/// r.b = 1.7020;
/// r.r_max = 0.6713;
/// VKC_TRY(camera::validate(r));
/// @endcode
struct ColorResponse {
  std::array<double, 3> gain{1.0, 1.0, 1.0};  ///< R, G and B.
  double a = 0.0;                             ///< F's q coefficient.
  double b = 0.0;                             ///< F's q^2 coefficient.
  double r_max = 0.0;                         ///< F is held beyond it.

  /// @return Whether the response changes nothing: gains 1, a = b = 0.
  bool is_identity() const noexcept {
    return gain[0] == 1.0 && gain[1] == 1.0 && gain[2] == 1.0 && a == 0.0 &&
           b == 0.0;
  }

  /// @return F at radius @p r: `1 + a q + b q^2`, `q = min(r, r_max)^2`.
  double falloff(double r) const noexcept {
    const double c = std::min(r, r_max);
    const double q = c * c;
    return 1.0 + a * q + b * q * q;
  }
};

/// @brief The checks a response passes to be read, written or applied.
/// @param response  The response.
/// @return OK for the identity (@ref ColorResponse::is_identity), or for gains
///         finite and above 0, a and b finite, @ref ColorResponse::r_max
///         finite and above 0, and F above 0 over `[0, r_max]`; otherwise
///         `Status::Code::InvalidArgument` naming what is wrong.
VS_CAMERA_API core::Status validate(const ColorResponse& response);

}  // namespace volumetric_kit::sensor::camera
