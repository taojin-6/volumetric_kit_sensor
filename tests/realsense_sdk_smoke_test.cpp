// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// librealsense smoke: the prerequisite VS_WITH_REALSENSE finds is the one that
// loads, and its context opens. The SDK is installed once, outside the repo,
// and linked in place, so the failure this guards against is a different copy
// answering at run time. It looks for cameras but needs none: tests use no
// hardware, and a camera that happens to be attached is only listed.

#include <cstdio>

#include <librealsense2/rs.hpp>

#include "volumetric_kit/core/testing/test_check.hpp"

int main() {
  rs2_error* error = nullptr;
  const int version = rs2_get_api_version(&error);
  CHECK(error == nullptr);
  const int major = version / 10000, minor = version / 100 % 100,
            patch = version % 100;
  std::printf("librealsense %d.%d.%d (configured %d.%d.%d)\n", major, minor,
              patch, VS_REALSENSE_SDK_VERSION_MAJOR,
              VS_REALSENSE_SDK_VERSION_MINOR, VS_REALSENSE_SDK_VERSION_PATCH);
  CHECK(major == VS_REALSENSE_SDK_VERSION_MAJOR);
  CHECK(minor == VS_REALSENSE_SDK_VERSION_MINOR);
  CHECK(patch == VS_REALSENSE_SDK_VERSION_PATCH);
  try {
    rs2::context context;
    std::printf("cameras attached: %u\n", context.query_devices().size());
  } catch (const rs2::error& e) {
    std::fprintf(stderr, "librealsense context: %s\n", e.what());
    return 1;
  }
  return 0;
}
