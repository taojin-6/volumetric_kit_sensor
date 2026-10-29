// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A stand-in for the Orbbec driver's internal camera (CameraStream and
// open_camera), so a test that compiles orbbec_sensor.cpp with
// orbbec_camera_stub.cpp opens OrbbecSensors with no camera or SDK context.
// A camera answers if it is in `cameras`; open_camera refuses any other,
// NotFound, as discovery does. The stub's cameras open and take sync writes
// and colour-control writes, the latter through the driver's own sequence
// (write_color_controls); they never start or poll.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/orbbec/orbbec_sync_config.hpp"

namespace orbbec_stub {

// One of a camera's colour properties: a flag is 0 or 1.
struct FakeProperty {
  // Whether it can be read and written; any use of one that cannot throws,
  // as touching a control the camera lacks would fail.
  bool writable = true;
  std::int32_t value = 0;
  std::int32_t min = 0;  // its range
  std::int32_t max = 1;
  std::int32_t step = 1;
};

// One camera that answers.
struct FakeCamera {
  // The sync settings it reads back, as SDK 2.10.6 leaves them after a write.
  volumetric_kit::sensor::OrbbecSyncSettings stored;
  volumetric_kit::core::Status write_status;  // what a sync write returns
  int writes = 0;                             // sync writes asked of it
  bool host_clock = false;                    // global timestamps asked of it
  std::uint32_t discovery_timeout_ms = 0;     // what its last open waited

  // The colour properties, as a Femto Mega ships: auto exposure over a stored
  // exposure of 200 and gain of 60, auto white balance, 6500 K stored.
  FakeProperty auto_exposure{true, 1, 0, 1, 1};
  FakeProperty exposure{true, 200, 1, 2000, 1};
  FakeProperty gain{true, 60, 1, 240, 1};
  FakeProperty auto_white_balance{true, 1, 0, 1, 1};
  FakeProperty white_balance{true, 6500, 2000, 11000, 100};
  // The colour writes asked of it, in order: "auto exposure 0", "exposure
  // 167", ...
  std::vector<std::string> color_writes;
  // A property (as color_writes names it) whose write throws, as an SDK error
  // would, and one whose write the camera takes and ignores.
  std::string failing_write;
  std::string ignored_write;
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
