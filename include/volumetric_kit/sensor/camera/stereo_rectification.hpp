// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/stereo_rectification.hpp
/// @brief A calibrated stereo pair, the pair from two entries of a
///        calibration file, its rectification and a lens's fold: on the host,
///        with no GPU.
///
/// The stereo tier's `StereoRectifier` rectifies through @ref rectify_stereo
/// and cuts its views at @ref fold_radius2, so a tool that checks a pair's
/// rows, or a lens's fold, without a GPU uses the same rectification and the
/// same fold the runtime does. These names keep the `sensor` namespace they
/// had in the stereo tier.

#include <string_view>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/export.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace volumetric_kit::sensor {

/// @brief Two calibrated cameras: each with its own lens, and where the right
///        one is.
///
/// "Left" and "right" are the matcher's: the right camera sits along the left
/// one's +x, give or take the small rotations and offsets rectification
/// removes.
struct StereoCalibration {
  camera::CameraModel left;   ///< The left camera, lens included.
  camera::CameraModel right;  ///< The right camera, lens included.
  /// A left-camera point in the right camera (OpenCV's `R`, `T`), metres.
  camera::Mat4d left_to_right = camera::Mat4d(1.0);
};

/// @brief The rectified pair: both views are @ref camera, the right one
///        @ref baseline metres along the left one's +x.
///
/// A point at depth `z` in the rectified left camera shows in the right view
/// `camera.intrinsics.fx * baseline / z` pixels to the left, on the same row:
/// what a matcher's disparity turns back into depth.
struct StereoRig {
  /// Both views' camera: pinhole (no lens), at the rectified size.
  camera::CameraModel camera;
  double baseline = 0.0;  ///< Metres; above 0.
  /// The rectified left camera's frame to the source left camera's: a
  /// rotation, so depth from the pair is posed through it.
  camera::Mat4d rectified_to_left = camera::Mat4d(1.0);
};

/// @brief A pair's rectification: the rectified pair, and the right
///        camera's turn into its view.
///
/// A point `p` of the left camera is `R1 * p` in the rectified left camera,
/// where `R1` (OpenCV's) is the rotation of `rig.rectified_to_left`,
/// transposed. The same point as the right camera sees it, `q`, is
/// `right_to_rectified * q` in the rectified right camera: the rectified
/// left one moved `rig.baseline` along +x, so the two land on one row.
struct StereoRectification {
  StereoRig rig;  ///< The views' camera, baseline and rotation.
  /// A right-camera point in the rectified right camera: OpenCV's `R2`, the
  /// left view's `R1` times the inverse of `left_to_right`'s rotation.
  camera::Mat3d right_to_rectified = camera::Mat3d(1.0);
};

/// @brief Rectify @p pair: Bouguet's method, as OpenCV's `stereoRectify`
///        computes it with `CALIB_ZERO_DISPARITY`, step for step in double.
///        The rectification `StereoRectifier::create` uses.
///
/// The relative rotation is split in half between the cameras, then the
/// baseline turned onto +x. Both views are one pinhole camera: its focal
/// length the two cameras' vertical ones scaled to the rectified height, its
/// principal point where the image corners, undistorted and turned, centre,
/// and @p alpha's scale between the rectangle every view pixel is valid in
/// (0) and the one holding every source pixel (1), from the border of a
/// 9 x 9 grid. Corners and borders are undistorted as OpenCV's
/// `undistortPoints` does by default: five fixed-point iterations of the
/// lens's inverse, which near a fisheye's image corners stop short of it.
///
/// A pair that is rectified already -- no rotation between the cameras, no
/// lens, one camera model, the right camera along +x alone, each within
/// 1e-9 -- is only resampled, both rotations the identity: its camera scaled
/// to @p size by one factor `s`, so its pixels keep their shape, about the
/// centres (`fx * s`, `(w' - 1) / 2 + (cx - (w - 1) / 2) * s`). Alpha picks
/// `s` between the larger of `w' / w` and `h' / h` (0: the view filled, the
/// picture cropped) and the smaller (1: all of the picture, the view
/// bordered).
///
/// @code
/// VKC_ASSIGN(const StereoCalibration pair,
///            stereo_calibration(file, "bm-1", "bm-2"));
/// VKC_ASSIGN(const StereoRectification rect,
///            rectify_stereo(pair, {960, 544}, 0.0));
/// const camera::Mat3d r1 =
///     glm::transpose(camera::Mat3d(rect.rig.rectified_to_left));
/// const camera::Mat3d& r2 = rect.right_to_rectified;
/// @endcode
/// @param pair   The calibrated pair.
/// @param size   The rectified size; {0, 0}: the left camera's.
/// @param alpha  0: only pixels both views see; 1: every source pixel
///               (OpenCV's meaning); between, between. In [0, 1].
/// @return The rectification; `Status::Code::InvalidArgument` for a camera
///         that `camera::check_camera_model` refuses (a pair with no
///         calibration among them: both cameras 0 x 0), cameras of
///         different sizes, a `left_to_right` that is not rigid, an alpha
///         outside [0, 1], a size with one side 0, cameras at one place, a
///         pair whose right camera is not along the left one's +x once the
///         rotation is split (a vertical pair, or the two swapped), or a
///         rectification that comes out degenerate, each named; or
///         `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Result<StereoRectification> rectify_stereo(
    const StereoCalibration& pair, camera::ImageSize size, double alpha);

/// @brief Where a lens's radial model folds: the undistorted normalized
///        radius, squared, past which it stops carrying rays outward.
///
/// Past the fold `r * radial(r^2)` shrinks again, or its denominator has
/// reached 0, so rays the lens cannot image would land inside the picture
/// again, mirrored. With `s = r^2`, numerator `N` and denominator `D`,
/// `d/dr [r N / D]` has the sign of `N D + 2 s (N' D - N D')`, so the fold
/// is the first `s > 0` where that or `D` stops being positive: marched out
/// by the angle off the axis (`s = tan^2`), 0.01 degrees a step to 89.99,
/// then bisected. The tangential terms play no part. The fold's angle off
/// the axis is `atan(sqrt(fold_radius2(d)))`.
/// @param distortion  The lens.
/// @return The fold's squared radius, at most float's largest; float's
///         largest for a lens that does not fold within 89.99 degrees of the
///         axis.
VS_CAMERA_API float fold_radius2(const camera::RationalDistortion& distortion);

/// @brief The pair two entries of a calibration file make, as
///        @ref rectify_stereo and `StereoRectifier::create` take it.
///
/// Each camera of the pair is a sensor of the file, keyed by the camera's
/// name, posed and with its colour camera's model: a pair of cine cameras,
/// say. The pair's cameras are those models, and
/// `left_to_right = inverse(right.color_to_world) * left.color_to_world`.
///
/// @code
/// VKC_ASSIGN(const camera::ArrayCalibration file,
///            camera::read_array_calibration("array_calibration.json"));
/// VKC_ASSIGN(const StereoCalibration pair,
///            stereo_calibration(file, "bm-1", "bm-2"));
/// @endcode
/// @param file      The calibration.
/// @param left_id   The left camera's id.
/// @param right_id  The right camera's id.
/// @return The pair; `Status::Code::InvalidArgument` for a file that
///         `camera::validate_array_calibration` refuses, the same id twice,
///         an id not in @p file, unposed or without a colour model (each
///         named), cameras at one place, or colour models of different
///         sizes; or `Status::Code::OutOfMemory`.
VS_CAMERA_API core::Result<StereoCalibration> stereo_calibration(
    const camera::ArrayCalibration& file, std::string_view left_id,
    std::string_view right_id);

}  // namespace volumetric_kit::sensor
