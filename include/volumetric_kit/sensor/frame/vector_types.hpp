// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame/vector_types.hpp
/// @brief The sensor package's float vector and matrix types: GLM's, under
///        the package's names.
///
/// Thin aliases over GLM, the family's math library, so the frame contract's
/// structs read the same as the kernels' buffers: GLM's float types are
/// trivially copyable, standard-layout and packed (`vec3` 12 B, `vec4` 16 B,
/// `mat4` 64 B, column-major), byte for byte a GLSL **scalar block layout**
/// block's. Each package keeps its own aliases; the camera tier, which links
/// no Vulkan, names its double types in `camera/geometry.hpp`. A
/// default-constructed `glm::mat4` is indeterminate: spell the identity
/// `Mat4f(1.0f)`.

#include <glm/glm.hpp>

namespace volumetric_kit::sensor {

/// @brief 2-component float vector (packed 8 B).
using Vec2f = glm::vec2;
/// @brief 3-component float vector (packed 12 B).
using Vec3f = glm::vec3;
/// @brief 4-component float vector (packed 16 B).
using Vec4f = glm::vec4;
/// @brief 3x3 column-major matrix (36 B; element (row, col) is `m[col][row]`):
///        the colour primaries conversion of `frame/color_space.hpp`.
using Mat3f = glm::mat3;
/// @brief 4x4 column-major matrix (64 B; element (row, col) is `m[col][row]`).
using Mat4f = glm::mat4;

// The buffer ABI depends on these packing exactly as documented above; a GLM
// configuration change (forced SIMD alignment) becomes a compile error.
static_assert(sizeof(Vec2f) == 8, "Vec2f must pack to 8 bytes");
static_assert(sizeof(Vec3f) == 12, "Vec3f must pack to 12 bytes");
static_assert(sizeof(Vec4f) == 16, "Vec4f must pack to 16 bytes");
static_assert(sizeof(Mat3f) == 36, "Mat3f must pack to 36 bytes");
static_assert(sizeof(Mat4f) == 64, "Mat4f must pack to 64 bytes");

}  // namespace volumetric_kit::sensor
