// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A stand-in for the RealSense drivers' internal camera (camera.hpp), so a
// test that compiles realsense_sensor.cpp with realsense_camera_stub.cpp in
// place of camera.cpp opens a RealSenseSensor with no camera attached. The
// camera is librealsense's software device: a depth module (Z16, depth units
// 1 mm) and a colour module (YUYV), each with one mode, the depth's
// extrinsic to the colour 5 cm across. The test sends it captures; each
// becomes an SDK frameset of both modes, which lands in the driver's frame
// queue as the SDK's thread lands one, counted received here in place of
// camera.cpp's callback, so the sensor's own conversion, poll, drain, its
// delivered and failed counts and the queue's drops run on real framesets.
// Only one camera is open at a time.

#include <cstdint>

#include <librealsense2/h/rs_frame.h>

namespace realsense_stub {

// The modes the camera offers: both 64 x 48 at 30 fps.
inline constexpr std::uint32_t kWidth = 64;
inline constexpr std::uint32_t kHeight = 48;
inline constexpr std::uint32_t kFps = 30;
// Every depth sample the camera sends, in millimetres.
inline constexpr std::uint16_t kDepthMm = 1500;

// Sends capture `number`, its depth and colour frames stamped `ms` on
// `domain`, as one frameset. Once the camera runs it lands in the driver's
// queue, counted received; before, it is let go. False if the SDK made no
// frameset of both.
bool send(std::uint64_t number, double ms,
          rs2_timestamp_domain domain = RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME);

}  // namespace realsense_stub
