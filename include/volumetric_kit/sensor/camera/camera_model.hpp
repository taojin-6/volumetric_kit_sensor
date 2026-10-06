// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/camera_model.hpp
/// @brief A camera as it captures: its image size, pinhole intrinsics, and
///        OpenCV's rational lens.
///
/// Conventions, the same everywhere in this repo:
/// - The camera looks down **+Z** with **+X right** and **+Y down**.
/// - Pixel centres sit at integer coordinates, so a W-pixel-wide image spans
///   `[-0.5, W - 0.5]`. Normalized coordinates are the pinhole's:
///   `x = (u - cx) / fx`.
/// - Intrinsics are in pixels of @ref CameraModel::size and mean nothing
///   without it: another stream mode of the same sensor has other intrinsics.
///
/// The lens model itself is @ref distort_rational.

#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/export.hpp"

namespace volumetric_kit::recon::camera {

/// @brief An image's size in pixels.
struct ImageSize {
  std::uint32_t width = 0;   ///< Width (pixels).
  std::uint32_t height = 0;  ///< Height (pixels).
};

/// @brief Pinhole intrinsics, in pixels.
struct PinholeIntrinsics {
  double fx = 0.0;  ///< Focal length x.
  double fy = 0.0;  ///< Focal length y.
  double cx = 0.0;  ///< Principal point x.
  double cy = 0.0;  ///< Principal point y.
};

/// @brief OpenCV's rational lens distortion, its eight coefficients in its
///        order; all zero is a pinhole lens, and `k4 = k5 = k6 = 0` the
///        five-coefficient Brown-Conrady model.
struct RationalDistortion {
  double k1 = 0.0;  ///< Radial, numerator, r^2.
  double k2 = 0.0;  ///< Radial, numerator, r^4.
  double p1 = 0.0;  ///< Tangential.
  double p2 = 0.0;  ///< Tangential.
  double k3 = 0.0;  ///< Radial, numerator, r^6.
  double k4 = 0.0;  ///< Radial, denominator, r^2.
  double k5 = 0.0;  ///< Radial, denominator, r^4.
  double k6 = 0.0;  ///< Radial, denominator, r^6.
};

/// @brief A camera as it captures: the image size, the intrinsics in its
///        pixels, and the lens between them.
struct CameraModel {
  ImageSize size;                 ///< The image the intrinsics describe.
  PinholeIntrinsics intrinsics;   ///< In pixels of @ref size.
  RationalDistortion distortion;  ///< All zero for a pinhole camera.
};

/// @brief Whether @p camera is usable: both sides of its size nonzero,
///        intrinsics finite with positive focal lengths, and every distortion
///        coefficient finite.
/// @param camera  The model.
/// @return OK; or `Status::Code::InvalidArgument` naming the field.
VR_CAMERA_API core::Status check_camera_model(const CameraModel& camera);

}  // namespace volumetric_kit::recon::camera
