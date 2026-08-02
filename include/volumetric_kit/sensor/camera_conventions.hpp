// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file camera_conventions.hpp
/// @brief Conversions between a capture API's camera conventions and the ones
///        this repo projects with.
///
/// These live here, in platform-neutral C++, rather than in whichever
/// platform-bound driver produced the numbers (the 2026-08-02 sensor-tier
/// decision). Two reasons: they are the part of a capture integration most
/// likely to be **silently** wrong -- a flipped axis or a half-pixel shift
/// produces a smeared or subtly misaligned reconstruction, never an error --
/// and they are pure arithmetic, so they can be pinned by host tests on any
/// platform instead of only on the hardware that generated them.

#include <cstdint>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/export.hpp"
#include "volumetric_kit/recon/tsdf/camera_params.hpp"
#include "volumetric_kit/recon/volume/camera_params.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief Reinterpret a camera-to-world transform from the OpenGL-style camera
///        convention into the computer-vision one this repo projects with.
///
/// ARKit, ARCore and OpenGL all place the camera looking down **−Z** with **+Y
/// up**; recon projects with `u = fx·x/z + cx`, i.e. looking down **+Z** with
/// **+Y down**. The two differ by a 180° roll about the view axis, so a pose
/// taken straight from ARKit describes a camera pointing the opposite way and
/// mirrored vertically.
///
/// Writing the camera-space change of basis as `B = diag(1, −1, −1, 1)`, a
/// point satisfies `p_gl = B · p_cv`, so `p_world = T_world_gl · p_gl =
/// (T_world_gl · B) · p_cv` — that is, the conversion is a
/// **right**-multiplication by `B`, which negates the second and third basis
/// columns and leaves the translation untouched. (Left-multiplying instead
/// would flip the camera's *position* in the world, which is the mistake this
/// function exists to prevent.)
///
/// The operation is an involution: applying it twice returns the input.
///
/// @param cam_to_world  Camera-to-world transform in the OpenGL/ARKit
///                      convention.
/// @return The same pose expressed in the computer-vision convention, ready for
///         @ref volume::DepthCameraParams::cam_to_world.
VR_SENSOR_API Mat4f cv_from_gl_camera(const Mat4f& cam_to_world);

/// @brief Derive depth-camera parameters from a colour camera the depth is
///        **registered** to.
///
/// The motivating case is ARKit, which reports one set of intrinsics — for
/// `capturedImage` (e.g. 1920×1440) — while `sceneDepth` arrives at a much
/// smaller size (256×192) from the *same physical camera*. Because the depth is
/// registered to the colour frame, the two share a pose and the same field of
/// view, and differ only by an image-size scale. That is the *registered* case
/// the tsdf integrator documents as free of the occlusion and
/// partial-colouring caveats an independently-posed colour camera carries.
///
/// Focal lengths scale with the size ratio. The principal point does **not**
/// scale naively: with pixel centres at integer coordinates (this repo's
/// convention — @ref volume::DepthCameraParams feeds
/// `x = (u − cx)·d/fx` on integer pixels), an image of width `W` spans
/// `[−0.5, W−0.5]`, so the correct map is
/// `c' = (c + 0.5)·s − 0.5`, not `c·s`. The difference is `0.5·(1 − s)`, which
/// at ARKit's 256/1920 is about 0.43 px — small, but a fixed bias in every
/// unprojected ray rather than noise. The half-pixel term is exactly what keeps
/// a centred principal point centred: `c = (W−1)/2` maps to `(W′−1)/2`.
///
/// @param color         The colour camera's intrinsics, size and pose.
/// @param depth_width   Depth image width in pixels (non-zero).
/// @param depth_height  Depth image height in pixels (non-zero).
/// @param min_depth     Reject depth samples nearer than this (metres).
/// @param max_depth     Reject depth samples farther than this (metres).
/// @return The depth camera, sharing @p color's pose; or
///         @ref Status::Code::InvalidArgument if either requested size is zero,
///         if @p color carries a zero size or a focal length that is not finite
///         and positive (the unprojection divides by it, so a bad one yields
///         inf/NaN rays rather than an error), or if @p min_depth is negative
///         or not below @p max_depth.
VR_SENSOR_API Result<volume::DepthCameraParams> depth_from_registered_color(
    const tsdf::ColorCameraParams& color, std::uint32_t depth_width,
    std::uint32_t depth_height, float min_depth, float max_depth);

}  // namespace volumetric_kit::recon::sensor
