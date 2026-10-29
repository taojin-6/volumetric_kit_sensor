// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame/camera_params.hpp
/// @brief Posed pinhole camera parameter blocks -- the depth and color cameras
///        every frame-consuming tier is handed.
///
/// These live in `core`, next to the GLM vocabulary they are built from, rather
/// than in the tiers that consume them, because **they are vocabulary, not
/// capability.** A posed pinhole camera is pure math over @ref Mat4f: no
/// Vulkan, no voxels, no fusion. Three tiers take one (`tsdf` fuses depth and
/// color, `texture` projects a mesh into a camera, `sensor`'s GPU pass hands
/// out a prepared frame's), and keeping the types where each was first needed
/// split a matched pair across tiers (`DepthCameraParams` first lived in
/// `volume`, its color counterpart in `tsdf`). Consumers see them as one
/// concept; `sensor::DeviceFrame` carries one of each side by side. A driver
/// describes its cameras as captured, lens included, in `camera::CameraModel`
/// instead; these are the pinhole cameras fusion reads. `volume`, which
/// allocates blocks from a depth frame, takes a camera of its own,
/// `volume::DepthPinhole`, which `tsdf::to_depth_pinhole` fills from a
/// `DepthCameraParams`.
///
/// Both structs are uploaded verbatim to compute kernels that read them through
/// **scalar block layout** (the 2026-07-05 ABI), so the `static_assert`s below
/// pin the host packing byte-for-byte against the GLSL mirrors in
/// `tsdf/shaders/` and `texture/shaders/`. A drift is a compile error here,
/// not a silent misprojection there.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "volumetric_kit/sensor/frame/vector_types.hpp"

namespace volumetric_kit::sensor {

/// @brief Pinhole depth-camera intrinsics, image size, valid range and pose.
///
/// Consumed by `tsdf::TsdfIntegrator::integrate` and `tsdf::Fuser`, whose
/// allocation takes it through `tsdf::to_depth_pinhole`, and by
/// `texture::ProjectiveTexturer::texture`. Packs to the shaders'
/// `DepthCameraParams`: the scalars at their natural 4-byte offsets, the
/// `mat4` at offset 32.
struct DepthCameraParams {
  float fx;              ///< Focal length x (pixels).
  float fy;              ///< Focal length y (pixels).
  float cx;              ///< Principal point x (pixels).
  float cy;              ///< Principal point y (pixels).
  float min_depth;       ///< Reject samples nearer than this (metres).
  float max_depth;       ///< Reject samples farther than this (metres).
  std::uint32_t width;   ///< Depth image width (pixels).
  std::uint32_t height;  ///< Depth image height (pixels).
  Mat4f cam_to_world;    ///< Camera -> world rigid transform (column-major).
};
static_assert(sizeof(DepthCameraParams) == 96,
              "DepthCameraParams must be 96 bytes");
static_assert(offsetof(DepthCameraParams, min_depth) == 16, "layout drift");
static_assert(offsetof(DepthCameraParams, width) == 24, "layout drift");
static_assert(offsetof(DepthCameraParams, cam_to_world) == 32, "layout drift");
static_assert(std::is_trivially_copyable_v<DepthCameraParams>,
              "DepthCameraParams must be trivially copyable");
static_assert(std::is_standard_layout_v<DepthCameraParams>,
              "DepthCameraParams must be standard-layout");

/// @brief Pinhole intrinsics, image size and pose of the (separate) color
///        camera a `tsdf::ColorFrame` was captured with.
///
/// The color analogue of @ref DepthCameraParams, without the depth-range fields
/// a color image has no use for -- which is why the two are separate types
/// rather than one: they pin different layouts (88 vs 96 bytes). May differ
/// from the depth camera (unregistered RGB-D); where depth is *registered* to
/// color, derive the depth camera from this one with
/// `sensor::depth_from_registered_color` so the two poses cannot drift apart.
/// Packs to the shader's `ColorCameraParams`: the scalars at their natural
/// 4-byte offsets, the `mat4` at offset 24.
struct ColorCameraParams {
  float fx;              ///< Focal length x (pixels).
  float fy;              ///< Focal length y (pixels).
  float cx;              ///< Principal point x (pixels).
  float cy;              ///< Principal point y (pixels).
  std::uint32_t width;   ///< Color image width (pixels).
  std::uint32_t height;  ///< Color image height (pixels).
  Mat4f cam_to_world;    ///< Camera -> world rigid transform (column-major).
};
static_assert(sizeof(ColorCameraParams) == 88,
              "ColorCameraParams must be 88 bytes");
static_assert(offsetof(ColorCameraParams, width) == 16, "layout drift");
static_assert(offsetof(ColorCameraParams, cam_to_world) == 24, "layout drift");
static_assert(std::is_trivially_copyable_v<ColorCameraParams>,
              "ColorCameraParams must be trivially copyable");
static_assert(std::is_standard_layout_v<ColorCameraParams>,
              "ColorCameraParams must be standard-layout");

}  // namespace volumetric_kit::sensor
