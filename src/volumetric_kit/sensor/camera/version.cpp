// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/version.hpp"

namespace volumetric_kit::sensor {

const char* version_string() noexcept { return VS_VERSION_STRING; }

int version_major() noexcept { return VS_VERSION_MAJOR; }

int version_minor() noexcept { return VS_VERSION_MINOR; }

int version_patch() noexcept { return VS_VERSION_PATCH; }

}  // namespace volumetric_kit::sensor
