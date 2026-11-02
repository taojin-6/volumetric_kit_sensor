// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// RealSenseSensor keeps the frame contract (testing/rgbd_contract.hpp) under
// injected faults, its own conversion, poll, drain and stats running on real
// framesets from a camera stub (realsense_camera_stub.hpp, a software device,
// in place of camera.cpp and its received count): framesets pushed out of a
// full queue and passed over by a poll, a capture never sent (a gap in the
// sequences), a frame stamped on the camera's clock refused and counted
// failed mid-poll, and the same refusal mid-drain, the frames before it
// handed out and the rest counted dropped. Every frame handed out keeps the
// contract, and so do the counters, exactly as the faults say. No camera
// attached.

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "realsense_camera_stub.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"
#include "volumetric_kit/sensor/testing/rgbd_contract.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;

namespace {

// Capture `number` at 30 fps, on the host's clock or the camera's.
bool send(std::uint64_t number, bool camera_clock = false) {
  return realsense_stub::send(
      number, 1000.0 + 1000.0 / 30.0 * static_cast<double>(number),
      camera_clock ? RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK
                   : RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME);
}

bool keeps(const sensor::IRgbdSensor& s,
           const std::vector<sensor::RgbdFrame>& frames) {
  const std::string breach = testing::rgbd_breach(s.info(), frames, s.stats());
  if (!breach.empty()) std::fprintf(stderr, "breach: %s\n", breach.c_str());
  return breach.empty();
}

int test_contract() {
  sensor::RealSenseSensor::Options options;
  options.depth_width = options.color_width = realsense_stub::kWidth;
  options.depth_height = options.color_height = realsense_stub::kHeight;
  options.fps = realsense_stub::kFps;
  auto opened = sensor::RealSenseSensor::open(options);
  if (!opened) std::fprintf(stderr, "%s\n", opened.status().message().c_str());
  CHECK(opened.ok());
  sensor::RealSenseSensor& s = *opened;
  CHECK(s.info().color && s.info().depth);
  CHECK(s.info().depth->size.width == realsense_stub::kWidth);

  // Polled, the newest each time: 2 pushed out of the one-frame queue; 4
  // never sent, a gap; 6 on the camera's clock refused, the poll failed.
  CHECK(s.start().ok());
  std::vector<sensor::RgbdFrame> polled;
  int failures = 0;
  const auto poll = [&]() -> bool {
    auto frame = s.poll();
    if (!frame.ok()) {
      ++failures;
      return true;
    }
    if (*frame) polled.push_back(std::move(**frame));
    return true;
  };
  CHECK(send(1) && poll());
  CHECK(send(2) && send(3) && poll());
  CHECK(send(5) && poll());
  CHECK(send(6, true) && poll());
  CHECK(send(7) && poll());
  CHECK(failures == 1);
  CHECK(polled.size() == 4);
  CHECK(polled[1].sequence == 3 && polled[2].sequence == 5 &&
        polled[3].sequence == 7);
  // The conversion ran: the depth as the camera sent it, the picture YUYV.
  CHECK(polled[0].depth != nullptr &&
        polled[0].depth[0] == realsense_stub::kDepthMm);
  CHECK(polled[0].color.layout == sensor::YuvLayout::Yuyv &&
        polled[0].color.width == realsense_stub::kWidth);
  sensor::SensorStats st = s.stats();
  std::printf(
      "  polled: received %llu, delivered %llu, dropped %llu, "
      "failed %llu\n",
      static_cast<unsigned long long>(st.received),
      static_cast<unsigned long long>(st.delivered),
      static_cast<unsigned long long>(st.dropped),
      static_cast<unsigned long long>(st.failed));
  CHECK(st.received == 6 && st.delivered == 4 && st.dropped == 1 &&
        st.failed == 1);
  CHECK(keeps(s, polled));
  std::puts("poll under faults: the frames and counters keep the contract");

  // Drained from a deeper queue after a restart: the refusal mid-drain
  // fails it, the two frames before it handed out, the two after dropped;
  // the next drain goes on.
  s.stop();
  CHECK(s.set_queue_depth(8).ok());
  CHECK(s.start().ok());
  CHECK(send(1) && send(2) && send(3, true) && send(4) && send(5));
  std::vector<sensor::RgbdFrame> drained;
  const vkc::Status mid = s.drain(&drained);
  CHECK(mid.domain() == vkc::Status::Code::IoError);
  CHECK(drained.size() == 2 && drained[1].sequence == 2);
  CHECK(send(6));
  CHECK(s.drain(&drained).ok());
  CHECK(drained.size() == 3 && drained[2].sequence == 6);
  st = s.stats();
  CHECK(st.received == 6 && st.delivered == 3 && st.dropped == 2 &&
        st.failed == 1);
  CHECK(keeps(s, drained));
  std::puts("drain under faults: IoError mid-drain, the frames before kept");
  s.stop();
  return 0;
}

}  // namespace

int main() {
  if (test_contract() != 0) return 1;
  std::puts("sensor_realsense_contract: OK");
  return 0;
}
