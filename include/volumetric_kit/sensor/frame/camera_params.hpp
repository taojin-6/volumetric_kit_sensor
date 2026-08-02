// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/camera_params.hpp
/// @brief Posed pinhole camera parameter blocks -- the depth and color cameras
///        every frame-consuming tier is handed.
///
/// These live in `core`, next to the GLM vocabulary they are built from, rather
/// than in the tiers that consume them. Two reasons.
///
/// **They are vocabulary, not capability.** A posed pinhole camera is pure math
/// over @ref Mat4f: no Vulkan, no voxels, no fusion. Four tiers take one
/// (`volume` allocates blocks from a depth frame, `tsdf` fuses depth and color,
/// `texture` projects a mesh into a camera, `sensor` reports what a device
/// captured), so keeping the types where they are *used* meant splitting a
/// matched pair across two tiers -- `DepthCameraParams` in `volume`, its color
/// counterpart in `tsdf`, because that is where each was first needed.
/// Consumers see them as one concept; `sensor::CapturedFrame` carries one of
/// each side by side.
///
/// **It keeps the `sensor` contract standalone.** That tier's implementers are
/// out of tree by design (an ARKit driver in `volumetric_kit_ios` -- the
/// 2026-08-02 decision), and with the camera types in `core` its public headers
/// and its link line reach `core` alone: a platform driver compiles against the
/// math vocabulary and nothing else.
///
/// Both structs are uploaded verbatim to compute kernels that read them through
/// **scalar block layout** (the 2026-07-05 ABI), so the `static_assert`s below
/// pin the host packing byte-for-byte against the GLSL mirrors in
/// `volume/shaders/`, `tsdf/shaders/` and `texture/shaders/`. A drift is a
/// compile error here, not a silent misprojection there.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon {

/// @brief Pinhole depth-camera intrinsics, image size, valid range and pose.
///
/// Consumed by `volume::VoxelHashMap::allocate_from_depth` (which unprojects a
/// posed depth frame to decide which blocks to allocate) and
/// `compact_active_blocks_in_frustum`, by `tsdf::TsdfIntegrator::integrate`,
/// and by `texture::ProjectiveTexturer::texture`. Packs to the shader's
/// `CameraParams`: the scalars at their natural 4-byte offsets, the `mat4` at
/// offset 32.
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

}  // namespace volumetric_kit::recon
