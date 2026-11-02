// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The JSON the camera tier's files share, each block with one reader and one
// writer: the array file's (array_calibration.cpp) and the tracked sensor's
// registration (tracked_registration.cpp). A refusal names the field, after
// the `where` its caller passes ("world: ", "fit: "), and not the file: each
// file's public functions add their own name. Internal to sensor_camera.

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace volumetric_kit::sensor::camera::detail {

using Json = nlohmann::json;

// `Status::Code::InvalidArgument` and `Status::Code::Unsupported` with `what`.
core::Status bad(const std::string& what);
core::Status unsupported(const std::string& what);

// Whether `s` is UTF-8, as JSON text must be.
bool is_utf8(const std::string& s);

// --- reading -----------------------------------------------------------------

// The finite number at `key`.
core::Result<double> number(const Json& object, const char* key,
                            const std::string& where);

// The whole number at `key`, at most 2^32 - 1.
core::Result<std::uint32_t> whole_number(const Json& object, const char* key,
                                         const std::string& where);

// The object at `key`; null when there is none.
core::Result<const Json*> object_at(const Json& parent, const char* key,
                                    const std::string& where);

// A transform written as `{"rvec": [3], "tvec": [3]}`:
// `x' = R(rvec) x + tvec` (`matrix_from_rodrigues`).
core::Result<Mat4d> transform(const Json& block, const std::string& where);

// A world block: `{"sensor": id}` or `{"apriltag": {...}}`, refusals named
// "world: ". Both kinds are read where a block names both, for the file's
// checks to refuse; a block of neither is Unsupported.
core::Result<WorldFrame> world_frame(const Json& block);

// --- checking ----------------------------------------------------------------

// A world is one tag or one sensor, never both, and neither only where
// `allow_none`. A tag's family is non-empty UTF-8, its id 0 or more, its size
// finite and above 0; a tag is not in the array, so nothing else of it is
// checked. What a named sensor must be is each file's own check.
core::Status check_world_kind(const WorldFrame& world, bool allow_none);

// --- writing -----------------------------------------------------------------

// A value as JSON text: a number in a form that reads back to the same double
// (at most 17 significant digits), a string quoted and escaped.
std::string text(double v);
std::string text(std::uint64_t v);
std::string text(const std::string& s);

std::string vector_text(const Vec3d& v);

// `transform` as `{"rvec": [...], "tvec": [...]}`, which `transform` reads.
// @pre `transform` is rigid.
std::string transform_text(const Mat4d& transform);

// The world block `world_frame` reads. @pre `world` is one sensor or one tag.
std::string world_text(const WorldFrame& world);

}  // namespace volumetric_kit::sensor::camera::detail
