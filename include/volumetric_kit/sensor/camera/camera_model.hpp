// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/camera_model.hpp
/// @brief A camera as it captures: its image size, pinhole intrinsics, and
///        OpenCV's rational lens.
///
/// The family's one camera model: a driver reports its factory calibration in
/// it, the calibration file stores it, and calib solves for it.
///
/// Conventions, the same everywhere in this repo:
/// - The camera looks down **+Z** with **+X right** and **+Y down**.
/// - Pixel centres sit at integer coordinates, so a W-pixel-wide image spans
///   `[-0.5, W - 0.5]`. Normalized coordinates are the pinhole's:
///   `x = (u - cx) / fx`.
/// - Intrinsics are in pixels of @ref CameraModel::size and mean nothing
///   without it: another stream mode of the same sensor has other intrinsics.
///
/// @code
/// const CameraModel camera{{3840, 2160},
///                          {2239.5, 2239.0, 1913.2, 1039.2},
///                          {0.0754, -0.1053, -0.0003, 0.0003, 0.0436}};
/// VKC_TRY(check_camera_model(camera));
/// VKC_ASSIGN(const Vec2d pixel, project(camera, {0.2, -0.1, 2.0}));
/// VKC_ASSIGN(const Vec2d ray, unproject(camera, pixel));  // (0.1, -0.05)
/// @endcode

#include <array>
#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/export.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/camera/projection.hpp"

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

/// @return The eight coefficients of @p d in OpenCV's order, as
///         @ref distort_rational and a solver's parameter block take them.
inline std::array<double, 8> coefficients(const RationalDistortion& d) {
  return {d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6};
}

/// @brief Whether @p camera can project: both sides of its size nonzero,
///        intrinsics finite with positive focal lengths, and every distortion
///        coefficient finite.
/// @param camera  The model.
/// @return OK; or `Status::Code::InvalidArgument` naming the field.
VR_CAMERA_API core::Status check_camera_model(const CameraModel& camera);

/// @brief The same camera at another image size of the same aspect ratio.
///
/// The focal length scales with the size, and the principal point about the
/// pixel centres: `c' = (c + 0.5) s - 0.5`. A centred principal point stays
/// centred; dropping the half-pixel term would bias every ray by
/// `0.5 (1 - s)` px. Normalized coordinates do not change, so neither does the
/// lens. A mode of another aspect ratio crops the sensor rather than scaling
/// it (the Femto Mega's 4:3 colour modes against its 16:9 ones), which no
/// rescale describes, so it is refused.
///
/// @code
/// // A 3840x2160 calibration, streamed at 1280x720.
/// VKC_ASSIGN(const CameraModel at_720p,
///            scale_camera_model(camera, {1280, 720}));
/// @endcode
/// @param camera  A valid model (@ref check_camera_model).
/// @param size    The size to express it at.
/// @return The model at @p size; or `Status::Code::InvalidArgument` for an
///         invalid @p camera, a zero side of @p size, or another aspect ratio.
VR_CAMERA_API core::Result<CameraModel> scale_camera_model(
    const CameraModel& camera, ImageSize size);

/// @brief Where the lens images a normalized point; @ref distort_rational in
///        double.
/// @param d  The lens.
/// @param p  A point in normalized pinhole coordinates.
/// @return The normalized coordinates @p p is imaged at.
inline Vec2d distort_normalized(const RationalDistortion& d,
                                const Vec2d& p) noexcept {
  const std::array<double, 8> c = coefficients(d);
  Vec2d out;
  distort_rational(c.data(), p.x, p.y, &out.x, &out.y);
  return out;
}

/// @brief How far from the axis the lens stays invertible: the squared
///        normalized radius at which the distorted radius stops growing, or
///        the radial factor's denominator reaches zero, whichever is nearer.
///
/// Inside it every pixel has exactly one ray; past it, strong distortion
/// folds several rays onto one pixel, and @ref unproject cannot say which is
/// right. Found by a scan of 2% steps in r^2 out to r = 1000 (89.94 degrees
/// off-axis), each crossing refined by bisection; a fold and an unfold within
/// one step would be missed, which no physical lens produces.
/// @param d  The lens.
/// @return The squared normalized radius; +infinity when the lens never folds
///         within r = 1000.
VR_CAMERA_API double invertible_radius2(const RationalDistortion& d) noexcept;

/// @brief Project a point in the camera's frame to a pixel.
/// @param camera        The model.
/// @param point_camera  The point, in the camera's frame.
/// @return The pixel, which may lie outside the image; or
///         `Status::Code::InvalidArgument` for a model
///         @ref check_camera_model refuses, or a point that is not finite or
///         not in front of the camera (`z <= 0`); or `Status::Code::Numerical`
///         if the projection overflows.
VR_CAMERA_API core::Result<Vec2d> project(const CameraModel& camera,
                                          const Vec3d& point_camera);

/// @brief The ray through a pixel: the undistorted normalized point `(x, y)`,
///        whose ray is `(x, y, 1)`.
///
/// Newton's method on the distortion, kept inside @ref invertible_radius2 so
/// it cannot settle on a folded-back ray that also lands on the pixel.
/// @param camera  The model.
/// @param pixel   The pixel.
/// @return The normalized point that @ref project maps to within 1e-9 px of
///         @p pixel; `Status::Code::InvalidArgument` for a model
///         @ref check_camera_model refuses or a pixel that is not finite; or
///         `Status::Code::Numerical` if no ray inside the invertible radius
///         lands on it.
VR_CAMERA_API core::Result<Vec2d> unproject(const CameraModel& camera,
                                            const Vec2d& pixel);

/// @brief @ref unproject with the lens's @ref invertible_radius2 computed
///        once by the caller, for a loop over many pixels.
/// @param camera          The model.
/// @param pixel           The pixel.
/// @param invertible_r2   `invertible_radius2(camera.distortion)`.
/// @return As @ref unproject.
VR_CAMERA_API core::Result<Vec2d> unproject(const CameraModel& camera,
                                            const Vec2d& pixel,
                                            double invertible_r2);

}  // namespace volumetric_kit::recon::camera
