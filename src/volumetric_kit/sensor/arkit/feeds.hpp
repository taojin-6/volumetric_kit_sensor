// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed), Objective-C++ with ARC: the two feeds an
// ArkitSensor opens, each calling Driver::deliver on the sensor's serial
// queue.

#include <dispatch/dispatch.h>

#include <memory>

#include "driver.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"

namespace volumetric_kit::sensor::arkit {

// The test pattern (pattern_feed.mm): its first frame delivered now, on the
// caller's thread, whose cameras become the sensor's; then, from each start,
// a frame each tick of a timer on `queue` at options.fps, as
// ArkitSensor's documentation describes. OutOfMemory where Core Video makes
// no buffers.
core::Result<std::unique_ptr<Feed>> open_pattern_feed(
    Driver& driver, const ArkitSensor::Options& options,
    dispatch_queue_t queue);

// ARKit's world tracking (arkit_feed.mm): the session configured, the camera
// asked for, the session run on `queue`, and its first frame with depth
// waited for (Driver::wait_opened), as ArkitSensor::open says.
// Unsupported off iOS and iPadOS.
core::Result<std::unique_ptr<Feed>> open_camera_feed(
    Driver& driver, const ArkitSensor::Options& options,
    dispatch_queue_t queue);

}  // namespace volumetric_kit::sensor::arkit
