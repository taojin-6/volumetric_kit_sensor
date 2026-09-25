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
// devices. No camera is needed: zero devices passes, and whatever answers the
// first query (USB or Ethernet) is listed for the log. That query blocks about
// a second while it probes the network, and even so an Ethernet camera can
// miss it from cold: on the rig, three Femto Megas answered the first query in
// 12 of 14 runs, and in the other two none answered for seconds (see the
// 2026-09-24 decision). So the listing is a log line, not a reachability
// check; "0 device(s)" does not mean none is attached.

#include <cstdint>
#include <cstdio>

#include <libobsensor/ObSensor.hpp>

namespace {

// The SDK's device-list strings are plain `const char*`, and printf("%s") of a
// null one is undefined.
const char* or_unknown(const char* s) { return s != nullptr ? s : "?"; }

}  // namespace

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

  try {
    // The SDK logs to ./Log at DEBUG by default; keep the test's working
    // directory clean and its console quiet unless something is wrong. One
    // call per sink: setLoggerSeverity sets *every* sink, the file one
    // included, so calling it after the file sink is switched off switches it
    // back on.
    ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);

    ob::Context context;
    const auto devices = context.queryDeviceList();
    const std::uint32_t count = devices->getCount();
    std::printf("%u device(s) answered the first query\n", count);
    for (std::uint32_t i = 0; i < count; ++i) {
      std::printf("  %s  serial %s  via %s\n", or_unknown(devices->getName(i)),
                  or_unknown(devices->getSerialNumber(i)),
                  or_unknown(devices->getConnectionType(i)));
    }
  } catch (const ob::Error& e) {
    std::fprintf(stderr, "FAIL: Orbbec SDK error: %s\n", e.what());
    return 1;
  }
  return 0;
}
