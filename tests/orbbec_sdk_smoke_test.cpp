// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Orbbec SDK smoke: the prerequisite VR_WITH_ORBBEC finds is the one that
// loads.
//
// The SDK is installed once, outside the repo, and linked in place (the
// 2026-09-24 decision), so the failure this guards against is a *different*
// copy answering at runtime -- a stale install found first on the loader path
// -- rather than a build error. It asserts the runtime version is exactly the
// one CMake configured against. It opens no SDK context, so it looks for no
// camera: tests use no hardware.

#include <cstdio>

#include <libobsensor/ObSensor.hpp>

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  const int major = ob::Version::getMajor();
  const int minor = ob::Version::getMinor();
  const int patch = ob::Version::getPatch();
  std::printf("Orbbec SDK %d.%d.%d (configured %d.%d.%d)\n", major, minor,
              patch, VR_ORBBEC_SDK_VERSION_MAJOR, VR_ORBBEC_SDK_VERSION_MINOR,
              VR_ORBBEC_SDK_VERSION_PATCH);
  CHECK(major == VR_ORBBEC_SDK_VERSION_MAJOR);
  CHECK(minor == VR_ORBBEC_SDK_VERSION_MINOR);
  CHECK(patch == VR_ORBBEC_SDK_VERSION_PATCH);
  return 0;
}
