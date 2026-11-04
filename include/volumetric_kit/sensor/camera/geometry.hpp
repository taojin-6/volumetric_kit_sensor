// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/geometry.hpp
/// @brief The camera tier's geometry: double-precision vectors and matrices,
///        rigid transforms, and the Rodrigues form OpenCV writes them in.
///
/// Double, not float, so a calibration's poses keep every digit until a GPU
/// consumer narrows them once, where it uploads (`Mat4f(m)`). The types are
/// GLM's, column-major like the rest of this package (`m[column][row]`).
///
/// A transform is named by what it maps: `color_to_world` takes a point in
/// the colour camera's frame to the world's.

#include <glm/glm.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/export.hpp"

namespace volumetric_kit::sensor::camera {

using Vec2d = glm::dvec2;  ///< A 2D point or vector.
using Vec3d = glm::dvec3;  ///< A 3D point or vector.
using Mat3d = glm::dmat3;  ///< A 3x3 matrix, column-major.
using Mat4d = glm::dmat4;  ///< A 4x4 matrix, column-major.

/// @brief A rigid transform as OpenCV writes one: `x' = R(rvec) x + tvec`.
///
/// `rvec` is a Rodrigues vector, its direction the axis and its length the
/// angle in radians; `tvec` is in metres. OpenCV's extrinsic of a camera is
/// this form of the world-to-camera transform.
struct RodriguesTransform {
  Vec3d rvec{0.0};  ///< The rotation, as a Rodrigues vector (radians).
  Vec3d tvec{0.0};  ///< The translation, applied after it (metres).
};

/// @brief Whether @p transform is rigid: finite, a proper rotation, and a
///        bottom row of `[0, 0, 0, 1]`.
///
/// The rotation's columns must be orthonormal to within 1e-6, which refuses a
/// matrix that is not a rotation without refusing one that went through float
/// on its way here, and its determinant positive, which refuses a reflection.
/// @param transform  The transform to check.
/// @return OK; or `Status::Code::InvalidArgument` naming what is wrong.
VS_CAMERA_API core::Status check_rigid(const Mat4d& transform);

/// @brief The inverse of a rigid transform: `R^T` and `-R^T t`.
/// @pre @p transform is rigid (@ref check_rigid); the transpose inverts
///      nothing else.
/// @param transform  The transform.
/// @return The transform that undoes @p transform.
VS_CAMERA_API Mat4d rigid_inverse(const Mat4d& transform) noexcept;

/// @brief How far a bottom-row entry of a transform made in float may be
///        from `[0, 0, 0, 1]`'s for @ref exact_rigid to take it as rounding:
///        float leaves the row a few ulp off (3 ulp of 1 is 3.6e-7).
inline constexpr double kFloatRowTolerance = 1e-6;

/// @brief How far an entry of `R^T R` of a transform made in float may be
///        from the identity's for @ref exact_rigid to take it as rounding: a
///        rotation rounded to float is about 1e-7 off orthonormal, and one
///        composed from such rotations drifts further (1e-5 after a few
///        corrections composed on it); one further off is not a rotation.
inline constexpr double kFloatRotationTolerance = 1e-4;

/// @brief The rigid transform a transform made in float stands for: its
///        bottom row exactly `[0, 0, 0, 1]`, its rotation the nearest one
///        (@ref nearest_rotation), its translation as given.
///
/// Float leaves a rigid transform a little off: ARKit's own read up to 3 ulp
/// off 1 in the corner, which @ref check_rigid refuses, and its rotation is
/// orthonormal only to float's rounding, which a product of such rotations
/// carries on. A transform within @ref kFloatRowTolerance of the row and
/// @ref kFloatRotationTolerance of orthonormal is made exact: the row set,
/// and the rotation orthonormal to double's rounding. One whose rotation is
/// orthonormal to 1e-12 already, as a calibration's is, keeps it bit for
/// bit. Deterministic: the same transform gives the same bits. A driver
/// whose poses come through float hands them out through this
/// (`RgbdFrame::color_to_world`).
/// @param transform  A transform that went through float.
/// @return The rigid transform, which @ref check_rigid takes; or
///         `Status::Code::InvalidArgument` naming what is wrong: a transform
///         that is not finite, a row or rotation further off than the
///         tolerances, or a reflection.
VS_CAMERA_API core::Result<Mat4d> exact_rigid(const Mat4d& transform);

/// @brief The rotation nearest @p m: the orthogonal factor of its polar
///        decomposition.
///
/// For a matrix that is a rotation with its rows scaled -- as an SDK's
/// factory extrinsic can arrive (the 2026-10-06 sensor-frame decision) -- it
/// is that rotation exactly. Newton's iteration `R <- (R + R^-T) / 2`, which
/// converges quadratically from a matrix near a rotation.
/// @pre @p m is invertible with a positive determinant; far from a rotation
///      the result is the nearest one still, but says nothing about where
///      @p m came from.
/// @param m  A matrix near a rotation.
/// @return The rotation.
VS_CAMERA_API Mat3d nearest_rotation(const Mat3d& m) noexcept;

/// @brief The rotation a Rodrigues vector describes: about its direction, by
///        its length in radians.
///
/// Below 1e-12 rad it is the first-order `I + [r]x`, where the axis is
/// undefined.
/// @param rvec  A rotation vector, as `cv::Rodrigues` takes.
/// @return The rotation matrix.
VS_CAMERA_API Mat3d rotation_from_rodrigues(const Vec3d& rvec) noexcept;

/// @brief The Rodrigues vector of a rotation, its angle in [0, pi]; the
///        inverse of @ref rotation_from_rodrigues.
///
/// Accurate to round-off at every angle, where `cv::Rodrigues` returns
/// exactly zero below about 1e-5 rad. The angle comes from `atan2`, not
/// `acos`, which turns round-off in the matrix into angle error near 0 and
/// pi. Past pi/2 the axis comes from the matrix's symmetric part, since
/// `sin(angle)` vanishes toward pi. At exactly pi the axis's sign is
/// arbitrary, as it is for OpenCV.
/// @pre @p rotation is a rotation (@ref check_rigid).
/// @param rotation  The rotation.
/// @return The rotation vector.
VS_CAMERA_API Vec3d rodrigues_from_rotation(const Mat3d& rotation) noexcept;

/// @brief The matrix of a transform OpenCV wrote: `[R(rvec) | tvec]`.
/// @param transform  The transform.
/// @return The same transform as a matrix.
VS_CAMERA_API Mat4d
matrix_from_rodrigues(const RodriguesTransform& transform) noexcept;

/// @brief A rigid transform in OpenCV's form; the inverse of
///        @ref matrix_from_rodrigues.
/// @pre @p transform is rigid (@ref check_rigid).
/// @param transform  The transform.
/// @return Its rotation as a Rodrigues vector (@ref rodrigues_from_rotation)
///         and its translation.
VS_CAMERA_API RodriguesTransform
rodrigues_from_matrix(const Mat4d& transform) noexcept;

}  // namespace volumetric_kit::sensor::camera
