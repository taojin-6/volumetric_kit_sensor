// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A stand-in for the Orbbec driver's internal camera (CameraStream and
// open_camera), so a test that compiles orbbec_sensor.cpp with
// orbbec_camera_stub.cpp opens OrbbecSensors with no camera or SDK context.
// A camera answers if it is in `cameras`; open_camera refuses any other,
// NotFound, as discovery does. The stub's cameras open and take sync writes;
// they never start or poll.

#include <cstdint>
#include <map>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"

namespace orbbec_stub {

// One camera that answers.
struct FakeCamera {
  // The sync settings it reads back, as SDK 2.10.6 leaves them after a write.
  volumetric_kit::recon::sensor::OrbbecSyncSettings stored;
  volumetric_kit::core::Status write_status;  // what a sync write returns
  int writes = 0;                             // sync writes asked of it
  bool host_clock = false;                    // global timestamps asked of it
  std::uint32_t discovery_timeout_ms = 0;     // what its last open waited
};

// The cameras that answer, by serial.
extern std::map<std::string, FakeCamera> cameras;
// open_camera calls: the cameras looked for.
extern int lookups;
// Cameras opened and not yet closed.
extern int open_streams;

// No camera answers, none has been looked for, and none is open.
void reset();

}  // namespace orbbec_stub
