// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Orbbec SDK smoke: the prerequisite VR_WITH_ORBBEC finds is the one that
// loads, and it initialises.
//
// The SDK is installed once, outside the repo, and linked in place (the
// 2026-09-24 decision), so the failure this guards against is a *different*
// copy answering at runtime -- a stale install found first on the loader path
// -- rather than a build error. It asserts the runtime version is exactly the
// one CMake configured against, then opens an SDK context and enumerates
// devices. No camera is needed: zero devices passes, and whatever is attached
// (USB or Ethernet) is listed for the log. Enumeration probes the network for
// Ethernet devices, so this takes a couple of seconds even with none present.

#include <cstdint>
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

  // The SDK logs to ./Log at DEBUG by default; keep the test's working
  // directory clean and its console quiet unless something is wrong.
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);

  try {
    ob::Context context;
    const auto devices = context.queryDeviceList();
    const std::uint32_t count = devices->getCount();
    std::printf("%u device(s)\n", count);
    for (std::uint32_t i = 0; i < count; ++i) {
      std::printf("  %s  serial %s  via %s\n", devices->getName(i),
                  devices->getSerialNumber(i), devices->getConnectionType(i));
    }
  } catch (const ob::Error& e) {
    std::fprintf(stderr, "FAIL: Orbbec SDK error: %s\n", e.what());
    return 1;
  }
  return 0;
}
