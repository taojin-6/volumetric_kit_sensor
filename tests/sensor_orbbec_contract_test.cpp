// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// OrbbecSensor keeps the frame contract (testing/rgbd_contract.hpp) under
// injected faults, its own poll, drain and stats running on a camera stub
// (orbbec_camera_stub.hpp) that stands in for CameraStream's take and read,
// with counts of its own, and plays a script of captures into the driver's
// own mailbox and queue: captures pushed out of a full queue and passed over
// by a poll (at depths 1 and 2), a colour frame lost
// (a gap in the sequences, counted failed), a capture the SDK failed on
// skipped and counted, one refused mid-poll, and IoError mid-drain with the
// frames before it handed out and the rest counted dropped. Every frame
// handed out keeps the contract, and so do the counters, exactly as the
// faults say. CameraStream's own counting is checked only on a camera. No
// camera or SDK context; the executable compiles orbbec_sensor.cpp, as the
// sync-apply test does.

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "orbbec_camera_stub.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/orbbec/orbbec_sensor.hpp"
#include "volumetric_kit/sensor/testing/rgbd_contract.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;
using orbbec_stub::FakeCapture;

namespace {

constexpr const char* kSerial = "TEST-CONTRACT";
const std::uint16_t kDepth[1] = {1000};
const std::uint8_t kPlanes[1] = {0};

// Capture `sequence` as the driver reads one: host depth and the decoded
// picture at the cameras' sizes, 40 ms apart at 25 fps.
FakeCapture capture(const orbbec_stub::FakeCamera& fake,
                    std::uint64_t sequence) {
  FakeCapture p;
  sensor::RgbdFrame& f = p.frame;
  f.depth = kDepth;
  f.depth_camera = fake.depth_camera;
  f.color.layout = sensor::YuvLayout::Nv12;
  f.color.host = kPlanes;
  f.color.width = fake.color_camera.size.width;
  f.color.height = fake.color_camera.size.height;
  f.color_camera = fake.color_camera;
  f.sequence = sequence;
  f.timestamp_ns = 40'000'000 * sequence;
  return p;
}

FakeCapture fault(FakeCapture::Kind kind) {
  FakeCapture p;
  p.kind = kind;
  return p;
}

bool keeps(const sensor::IRgbdSensor& s,
           const std::vector<sensor::RgbdFrame>& frames) {
  const std::string breach = testing::rgbd_breach(s.info(), frames, s.stats());
  if (!breach.empty()) std::fprintf(stderr, "breach: %s\n", breach.c_str());
  return breach.empty();
}

int test_contract() {
  orbbec_stub::reset();
  orbbec_stub::FakeCamera& fake = orbbec_stub::cameras[kSerial];
  sensor::OrbbecSensor::Options options;
  options.serial = kSerial;
  auto opened = sensor::OrbbecSensor::open(options);
  if (!opened) std::fprintf(stderr, "%s\n", opened.status().message().c_str());
  CHECK(opened.ok());
  sensor::OrbbecSensor& s = *opened;
  CHECK(s.info().color && s.info().depth);

  // Polled, the newest of each batch: 2 pushed out of the one-frame queue;
  // 4's colour lost, a gap; a capture skipped; one refused, the poll failed.
  fake.batches = {{capture(fake, 1)},
                  {capture(fake, 2), capture(fake, 3)},
                  {fault(FakeCapture::Kind::Lost), capture(fake, 5)},
                  {fault(FakeCapture::Kind::Skipped)},
                  {fault(FakeCapture::Kind::Refused)},
                  {capture(fake, 8)}};
  CHECK(s.start().ok());
  std::vector<sensor::RgbdFrame> polled;
  int failures = 0;
  for (int i = 0; i < 6; ++i) {
    auto frame = s.poll();
    if (!frame.ok()) {
      ++failures;
      continue;
    }
    if (*frame) polled.push_back(std::move(**frame));
  }
  CHECK(failures == 1);
  CHECK(polled.size() == 4);
  CHECK(polled[2].sequence == 5 && polled[3].sequence == 8);
  sensor::SensorStats st = s.stats();
  std::printf(
      "  polled: received %llu, delivered %llu, dropped %llu, "
      "failed %llu\n",
      static_cast<unsigned long long>(st.received),
      static_cast<unsigned long long>(st.delivered),
      static_cast<unsigned long long>(st.dropped),
      static_cast<unsigned long long>(st.failed));
  CHECK(st.received == 8 && st.delivered == 4 && st.dropped == 1 &&
        st.failed == 3);  // skipped, refused, and the lost colour
  CHECK(keeps(s, polled));
  std::puts("poll under faults: the frames and counters keep the contract");

  // Drained from a deeper queue after a restart: a refusal mid-drain fails
  // it, the two frames before it handed out, the two after it dropped; the
  // next drain goes on.
  s.stop();
  CHECK(s.set_queue_depth(8).ok());
  fake.batches = {
      {capture(fake, 1), capture(fake, 2), fault(FakeCapture::Kind::Refused),
       capture(fake, 4), capture(fake, 5)},
      {capture(fake, 6)}};
  CHECK(s.start().ok());
  std::vector<sensor::RgbdFrame> drained;
  const vkc::Status mid = s.drain(&drained);
  CHECK(mid.domain() == vkc::Status::Code::IoError);
  CHECK(drained.size() == 2 && drained[1].sequence == 2);
  CHECK(s.drain(&drained).ok());
  CHECK(drained.size() == 3 && drained[2].sequence == 6);
  st = s.stats();
  CHECK(st.received == 6 && st.delivered == 3 && st.dropped == 2 &&
        st.failed == 1);
  CHECK(keeps(s, drained));
  std::puts("drain under faults: IoError mid-drain, the frames before kept");

  // The checker sees what a driver could get wrong: a frame handed out twice
  // breaks the order and the count.
  std::vector<sensor::RgbdFrame> twice = drained;
  twice.push_back(drained.back());
  CHECK(!testing::rgbd_breach(s.info(), twice, s.stats()).empty());

  // Polled from a two-frame queue: each poll passes over the older frame
  // held, counted dropped, as is the one pushed out of the full queue.
  s.stop();
  CHECK(s.set_queue_depth(2).ok());
  fake.batches = {{capture(fake, 1), capture(fake, 2)},
                  {capture(fake, 3), capture(fake, 4), capture(fake, 5)}};
  CHECK(s.start().ok());
  std::vector<sensor::RgbdFrame> newest;
  for (int i = 0; i < 2; ++i) {
    auto frame = s.poll();
    CHECK(frame.ok() && *frame);
    newest.push_back(std::move(**frame));
  }
  CHECK(newest[0].sequence == 2 && newest[1].sequence == 5);
  st = s.stats();
  CHECK(st.received == 5 && st.delivered == 2 && st.dropped == 3 &&
        st.failed == 0);  // 1 and 4 passed over, 3 pushed out
  CHECK(keeps(s, newest));
  std::puts("poll at depth 2: the frames passed over counted dropped");
  s.stop();
  return 0;
}

}  // namespace

int main() {
  if (test_contract() != 0) return 1;
  std::puts("sensor_orbbec_contract: OK");
  return 0;
}
