// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The package's own pieces: the version CMake wires through, its GLM aliases
// (including vector_types.hpp proves they compile at their sizes), and what
// it asks of a device -- scalarBlockLayout, and the decoders' two extensions
// as optional, so a device without them still opens. Host only.

#include <algorithm>
#include <string>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/frame/vector_types.hpp"
#include "volumetric_kit/sensor/version.hpp"

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

namespace {

bool lists(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

}  // namespace

int main() {
  // The integer accessors compose to the string, which comes from the
  // independent PROJECT_VERSION macro: a swapped MAJOR/MINOR/PATCH is caught
  // without a literal that a version bump would invalidate.
  const std::string composed = std::to_string(sensor::version_major()) + "." +
                               std::to_string(sensor::version_minor()) + "." +
                               std::to_string(sensor::version_patch());
  CHECK(composed == sensor::version_string());
  CHECK(composed == VS_VERSION_STRING);

  const vkc::DeviceRequirements reqs = sensor::device_requirements();
  CHECK(reqs.scalar_block_layout);
  CHECK(lists(reqs.optional_extensions,
              VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME));
  CHECK(lists(reqs.optional_extensions, "VK_EXT_metal_objects"));
  CHECK(reqs.extensions.empty());

  const sensor::Mat4f identity(1.0f);
  CHECK(identity[3][3] == 1.0f && identity[3][0] == 0.0f);
  return 0;
}
