// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera/geometry.hpp
/// @brief The camera tier's geometry: double-precision vectors and matrices,
///        rigid transforms, and the Rodrigues form OpenCV writes them in.
///
/// Double, not float: calibration solves in double and writes the shortest
/// text that reads back to the same value, so the tier that reads and writes
/// its results keeps every digit. A GPU consumer narrows once, where it
/// uploads (`Mat4f(m)`). The types are GLM's, column-major like the rest of
/// this repo (`m[column][row]`), so an Eigen consumer maps them in place.
///
/// A transform is named by what it maps: `color_to_world` takes a point in
/// the colour camera's frame to the world's.

#include <glm/glm.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/export.hpp"

namespace volumetric_kit::recon::camera {

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
VR_CAMERA_API core::Status check_rigid(const Mat4d& transform);

/// @brief The inverse of a rigid transform: `R^T` and `-R^T t`.
/// @pre @p transform is rigid (@ref check_rigid); the transpose inverts
///      nothing else.
/// @param transform  The transform.
/// @return The transform that undoes @p transform.
VR_CAMERA_API Mat4d rigid_inverse(const Mat4d& transform) noexcept;

/// @brief The rotation a Rodrigues vector describes: about its direction, by
///        its length in radians.
///
/// Below 1e-12 rad it is the first-order `I + [r]x`, where the axis is
/// undefined.
/// @param rvec  A rotation vector, as `cv::Rodrigues` takes.
/// @return The rotation matrix.
VR_CAMERA_API Mat3d rotation_from_rodrigues(const Vec3d& rvec) noexcept;

/// @brief The Rodrigues vector of a rotation, its angle in [0, pi].
///
/// The angle comes from `atan2` rather than `acos`, which turns round-off in
/// the matrix into angle error as the angle nears 0 or pi. Past pi/2 the axis
/// comes from the matrix's symmetric part, since `sin(angle)` vanishes toward
/// pi. At exactly pi the axis's sign is arbitrary, as it is for OpenCV.
/// @pre @p rotation is a rotation.
/// @param rotation  The rotation.
/// @return The rotation vector, which @ref rotation_from_rodrigues maps back to
///         @p rotation.
VR_CAMERA_API Vec3d rodrigues_from_rotation(const Mat3d& rotation) noexcept;

/// @brief The matrix of a transform OpenCV wrote: `[R(rvec) | tvec]`.
/// @param transform  The transform.
/// @return The same transform as a matrix.
VR_CAMERA_API Mat4d
matrix_from_rodrigues(const RodriguesTransform& transform) noexcept;

/// @brief A rigid transform in OpenCV's form; the inverse of
///        @ref matrix_from_rodrigues.
/// @pre @p transform is rigid (@ref check_rigid).
/// @param transform  The transform.
/// @return The transform as a Rodrigues vector and a translation.
VR_CAMERA_API RodriguesTransform
rodrigues_from_matrix(const Mat4d& transform) noexcept;

}  // namespace volumetric_kit::recon::camera
