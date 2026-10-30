// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame/depth_noise.hpp
/// @brief How noisy a depth camera's samples are: the model its producer
///        states with every frame (`DeviceFrame::depth_noise`).

#include <cmath>

namespace volumetric_kit::sensor {

/// @brief A depth camera's noise: the frame-to-frame standard deviation of one
///        sample, `sigma(z) = a + b z + c z^2` metres at a depth of z metres.
///
/// Temporal: how much one pixel's depth varies from frame to frame on a still
/// surface. An error that repeats every frame, such as calibration or a learned
/// matcher's fixed pattern, is not in it. Each producer states it as it was
/// measured, and says where: the Femto Mega's face on, a stereo camera's over
/// the slants of the room it was measured in. A consumer that knows a
/// surface's slant applies its own term.
///
/// The usual terms:
/// - time of flight: b (the Femto Mega: 0.95 mm per metre);
/// - rectified stereo: c = sigma_d / (f B), from the disparity's sigma_d (px),
///   the focal length f (px) and the baseline B (m); see @ref stereo;
/// - a constant: a.
///
/// All zero, the default, means the producer does not know its noise. A
/// consumer that needs a model refuses such a frame rather than guess one.
///
/// @code
/// RgbdFrame frame;  // in a driver's poll, as it states its depth range
/// frame.depth_noise = {0.0f, 0.95e-3f, 0.0f};  // time of flight
/// frame.depth_noise = DepthNoise::stereo(0.13, fx, baseline);  // stereo
/// @endcode
struct DepthNoise {
  float a = 0.0f;  ///< Constant term (m).
  float b = 0.0f;  ///< Linear term (m per m).
  float c = 0.0f;  ///< Quadratic term (per m).

  /// @return The standard deviation at a depth of @p z metres, in metres.
  constexpr float sigma(float z) const noexcept { return a + z * (b + z * c); }

  /// @return Whether this states a model: every term finite and at least 0,
  ///         and at least one above 0.
  bool known() const noexcept {
    const auto term = [](float x) { return std::isfinite(x) && x >= 0.0f; };
    return term(a) && term(b) && term(c) && (a > 0.0f || b > 0.0f || c > 0.0f);
  }

  /// @brief Rectified stereo's model: `c = disparity_sigma / (focal *
  ///        baseline)`, a and b 0.
  ///
  /// Depth is `focal * baseline / d` at a disparity of d pixels, so a
  /// disparity error of sigma_d moves it by `z^2 sigma_d / (focal *
  /// baseline)`.
  /// @param disparity_sigma  The disparity's frame-to-frame standard
  ///                         deviation, in pixels at the size @p focal is for.
  /// @param focal            The rectified focal length (px).
  /// @param baseline         The baseline (m).
  /// @return The model; unknown (all zero) unless all three are finite and
  ///         above 0.
  static DepthNoise stereo(double disparity_sigma, double focal,
                           double baseline) noexcept {
    const auto positive = [](double x) { return std::isfinite(x) && x > 0.0; };
    DepthNoise noise;
    if (positive(disparity_sigma) && positive(focal) && positive(baseline)) {
      noise.c = static_cast<float>(disparity_sigma / (focal * baseline));
    }
    return noise;
  }

  /// @return Whether @p other is the same model, term by term.
  constexpr bool operator==(const DepthNoise& other) const noexcept {
    return a == other.a && b == other.b && c == other.c;
  }
  /// @return Whether @p other is another model.
  constexpr bool operator!=(const DepthNoise& other) const noexcept {
    return !(*this == other);
  }
};

}  // namespace volumetric_kit::sensor
