// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The stereo tier's own pieces, no camera:
//   - the contract every stereo source keeps (stereo_contract.hpp): a source
//     opened without a calibration reports both cameras 0 x 0 and one opened
//     with one neither camera with a side 0, and frames that keep the sync
//     rules pass; each rule broken is the one named; a time of 0 is none,
//     the sequence still checked, and a time after a 0 still increases on
//     the last one reported;
//   - StereoRectifier::create takes a pair from two entries of a calibration
//     file (stereo_calibration), its rig rectify_stereo's bit for bit, the
//     baseline the cameras' distance; and refuses a source with no
//     calibration, named.
// The rectifier's part runs on the real driver and skips where no device is
// present; the rest runs on the host first.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "stereo_contract.hpp"
#include "stereo_pair_file.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/camera/stereo_rectification.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

using camera::Vec3d;
using stereo_contract::Opened;

// Frames that keep the contract pass; each rule broken is the one named.
int test_contract() {
  sensor::StereoInfo info;  // opened without a calibration
  info.max_skew_ns = 16'666'000;
  std::vector<sensor::StereoFrame> frames(3);
  const std::int64_t skews[3] = {8'200'000, -16'666'000, 16'666'000};
  for (std::size_t i = 0; i < frames.size(); ++i) {
    frames[i].sequence = 10 + 2 * i;  // a gap: a lost frame is no breach
    frames[i].timestamp_ns = 1'000'000'000 + 33'333'333 * i;
    frames[i].skew_ns = skews[i];
  }
  sensor::SensorStats stats;
  stats.received = 5;
  stats.delivered = 3;
  stats.dropped = 1;
  stats.failed = 1;
  const auto says = [](const std::string& got, const char* why) -> bool {
    if (got.find(why) == std::string::npos) {
      std::fprintf(stderr, "expected \"%s\": \"%s\"\n", why, got.c_str());
      return false;
    }
    return true;
  };
  const auto names = [&](const std::vector<sensor::StereoFrame>& f,
                         const sensor::SensorStats& s,
                         const char* why) -> bool {
    return says(stereo_contract::breach(info, Opened::Uncalibrated, f, s), why);
  };
  CHECK(stereo_contract::breach(info, Opened::Uncalibrated, frames, stats)
            .empty());

  std::vector<sensor::StereoFrame> bad = frames;
  bad[1].skew_ns = -16'666'001;
  CHECK(names(bad, stats, "frame 1: skew -16666001 ns past the bound"));
  bad = frames;
  bad[2].skew_ns = 16'666'001;
  CHECK(names(bad, stats, "frame 2: skew 16666001 ns past the bound"));
  bad = frames;
  bad[2].sequence = bad[1].sequence;
  CHECK(names(bad, stats, "frame 2: sequence 12 after 12"));
  bad = frames;
  bad[1].timestamp_ns = bad[0].timestamp_ns;
  CHECK(names(bad, stats, "frame 1: time"));
  bad = frames;  // ordered on the frame before, not on the first
  bad[2].timestamp_ns = bad[1].timestamp_ns;
  CHECK(names(bad, stats, "frame 2: time 1033333333 ns after 1033333333"));

  // A time of 0 is none (StereoFrame::timestamp_ns, as RgbdFrame's): pairs
  // with none keep the order, the sequence still counted; a 0 between rising
  // times keeps it too; a time after a 0 still increases on the last one
  // reported.
  bad = frames;
  for (sensor::StereoFrame& f : bad) f.timestamp_ns = 0;
  CHECK(
      stereo_contract::breach(info, Opened::Uncalibrated, bad, stats).empty());
  bad[2].sequence = bad[1].sequence;
  CHECK(names(bad, stats, "frame 2: sequence 12 after 12"));
  bad = frames;
  bad[1].timestamp_ns = 0;
  CHECK(
      stereo_contract::breach(info, Opened::Uncalibrated, bad, stats).empty());
  bad[2].timestamp_ns = bad[0].timestamp_ns - 1;
  CHECK(names(bad, stats, "frame 2: time 999999999 ns after 1000000000"));
  sensor::SensorStats over = stats;
  over.failed = 2;
  CHECK(names(frames, over, "past received 5"));
  over = stats;
  over.delivered = 4;
  CHECK(names(frames, over, "delivered 4, taken 3"));
  sensor::StereoInfo exact;  // one instant: any skew breaks it
  CHECK(!stereo_contract::breach(exact, Opened::Uncalibrated, frames, stats)
             .empty());

  // The calibration says how the source was opened: both cameras 0 x 0
  // without one, neither camera with a side 0 with one.
  sensor::StereoInfo calibrated = info;
  auto pair =
      sensor::stereo_calibration(stereo_pair_file::pair_file(), "bm-1", "bm-2");
  CHECK(pair.ok());
  calibrated.calibration = *pair;
  CHECK(stereo_contract::breach(calibrated, Opened::Calibrated, frames, stats)
            .empty());
  CHECK(stereo_contract::calibration_breach(calibrated, Opened::Calibrated)
            .empty());
  CHECK(says(stereo_contract::breach(info, Opened::Calibrated, frames, stats),
             "the left camera is 0 x 0 from a source opened with a "
             "calibration"));
  CHECK(says(
      stereo_contract::breach(calibrated, Opened::Uncalibrated, frames, stats),
      "the left camera is 3840 x 2160 from a source opened without a "
      "calibration, not 0 x 0"));
  sensor::StereoInfo half = calibrated;
  half.calibration.right.size = {};
  CHECK(says(stereo_contract::calibration_breach(half, Opened::Calibrated),
             "the right camera is 0 x 0"));
  CHECK(says(stereo_contract::calibration_breach(half, Opened::Uncalibrated),
             "the left camera is 3840 x 2160"));
  half = info;
  half.calibration.left.size = {0, 2160};
  CHECK(says(stereo_contract::calibration_breach(half, Opened::Uncalibrated),
             "the left camera is 0 x 2160"));
  half = calibrated;
  half.calibration.left.size = {0, 2160};
  CHECK(says(stereo_contract::calibration_breach(half, Opened::Calibrated),
             "the left camera is 0 x 2160 from a source opened with a "
             "calibration"));
  std::puts("the contract: kept, and each rule broken named");
  return 0;
}

bool same_rig(const sensor::StereoRig& a, const sensor::StereoRig& b) {
  const camera::CameraModel& x = a.camera;
  const camera::CameraModel& y = b.camera;
  return x.size.width == y.size.width && x.size.height == y.size.height &&
         x.intrinsics.fx == y.intrinsics.fx &&
         x.intrinsics.fy == y.intrinsics.fy &&
         x.intrinsics.cx == y.intrinsics.cx &&
         x.intrinsics.cy == y.intrinsics.cy && a.baseline == b.baseline &&
         a.rectified_to_left == b.rectified_to_left;
}

// The rectifier takes the pair, its rig rectify_stereo's, its baseline the
// cameras' distance apart.
int gpu_main(vkc::test::GpuContext& gpu) {
  auto made =
      sensor::stereo_calibration(stereo_pair_file::pair_file(), "bm-1", "bm-2");
  CHECK(made.ok());
  sensor::StereoInfo info;
  info.calibration = *made;
  auto rectifier = sensor::StereoRectifier::create(
      gpu.device, gpu.allocator, info,
      {{960, 544}, 0.0f, sensor::StereoRectifier::Output::Rgba8});
  if (!rectifier) {
    std::fprintf(stderr, "%s\n", rectifier.status().message().c_str());
  }
  CHECK(rectifier.ok());
  auto host = sensor::rectify_stereo(*made, {960, 544}, 0.0);
  CHECK(host.ok());
  CHECK(same_rig((*rectifier)->rig(), host->rig));
  const double distance =
      glm::length(Vec3d(stereo_pair_file::left_to_right()[3]));
  const double baseline = (*rectifier)->rig().baseline;
  std::printf("  rectified: baseline %.9f m (cameras %.9f m apart), fx %.2f\n",
              baseline, distance, (*rectifier)->rig().camera.intrinsics.fx);
  CHECK(std::fabs(baseline - distance) < 1e-9);

  // A source with no calibration (a DeckLink pair opened to calibrate its
  // lenses) is refused, named, before its cameras are checked.
  auto none = sensor::StereoRectifier::create(
      gpu.device, gpu.allocator, sensor::StereoInfo{},
      {{960, 544}, 0.0f, sensor::StereoRectifier::Output::Rgba8});
  CHECK(!none.ok());
  CHECK(none.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(none.status().message().find("no calibration") != std::string::npos);
  std::puts("sensor_stereo_sensor: OK");
  return 0;
}

}  // namespace

int main() {
  if (test_contract() != 0) return 1;
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
