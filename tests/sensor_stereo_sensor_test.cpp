// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The stereo tier's own pieces, no camera:
//   - stereo_calibration: a pair from two entries of a calibration file, the
//     left camera's point in the right one, to round-off, named either way
//     round; which StereoRectifier::create accepts, the baseline the
//     cameras' distance; and each refusal, named;
//   - the sync contract every stereo source keeps (stereo_contract.hpp):
//     frames that keep it pass, and each rule broken is the one named.
// The rectifier's part runs on the real driver and skips where no device is
// present; the rest runs on the host first.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "stereo_contract.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/stereo/stereo_calibration.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

using camera::Mat4d;
using camera::Vec3d;

// A 2160p lens with the 09-29 Blackmagic model's intrinsics, scaled from
// 1080p, and a mild rational distortion; `dx` moves its principal point.
camera::CameraModel lens(double dx) {
  camera::CameraModel c;
  c.size = {3840, 2160};
  c.intrinsics = {1702.41, 1702.12, 1917.38 + dx, 1108.27};
  c.distortion.k1 = 0.05;
  c.distortion.k2 = -0.01;
  c.distortion.p1 = 2e-4;
  c.distortion.k4 = 0.02;
  return c;
}

// The left camera's point in the right one: 0.4 m along +x of the left
// camera, a little toed in and tilted.
Mat4d left_to_right() {
  return camera::matrix_from_rodrigues(
      {Vec3d(0.004, -0.03, 0.002), Vec3d(-0.398, 0.006, -0.011)});
}

// A file of a Femto and two cine cameras, "bm-1" (left) and "bm-2"
// (right), posed in a world neither of them is at.
camera::ArrayCalibration pair_file() {
  camera::ArrayCalibration file;
  camera::SensorCalibration femto;
  femto.id = "femto";
  femto.color_to_world = Mat4d(1.0);
  camera::SensorCalibration left;
  left.id = "bm-1";
  left.color_to_world = camera::matrix_from_rodrigues(
      {Vec3d(0.3, -1.1, 0.2), Vec3d(1.5, -0.4, 2.25)});
  left.color = camera::CameraCalibration{lens(0.0),
                                         camera::IntrinsicsSource::Calibrated};
  camera::SensorCalibration right = left;
  right.id = "bm-2";
  right.color_to_world =
      *left.color_to_world * camera::rigid_inverse(left_to_right());
  right.color->model = lens(3.5);
  file.sensors = {std::move(left), std::move(right), std::move(femto)};
  return file;
}

double worst_difference(const Mat4d& a, const Mat4d& b) {
  double worst = 0.0;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      worst = std::fmax(worst, std::fabs(a[c][r] - b[c][r]));
    }
  }
  return worst;
}

bool same_model(const camera::CameraModel& a, const camera::CameraModel& b) {
  return a.size.width == b.size.width && a.size.height == b.size.height &&
         a.intrinsics.fx == b.intrinsics.fx &&
         a.intrinsics.fy == b.intrinsics.fy &&
         a.intrinsics.cx == b.intrinsics.cx &&
         a.intrinsics.cy == b.intrinsics.cy &&
         a.distortion.k1 == b.distortion.k1 &&
         a.distortion.k2 == b.distortion.k2 &&
         a.distortion.p1 == b.distortion.p1 &&
         a.distortion.k4 == b.distortion.k4;
}

// Whether `made` is refused as InvalidArgument, naming `why`.
bool refused(const vkc::Result<sensor::StereoCalibration>& made,
             const char* why) {
  if (made.ok() ||
      made.status().domain() != vkc::Status::Code::InvalidArgument ||
      made.status().message().find(why) == std::string::npos) {
    std::fprintf(stderr, "expected \"%s\": %s\n", why,
                 made.ok() ? "accepted" : made.status().message().c_str());
    return false;
  }
  return true;
}

// The pair is the two entries' colour models and the left camera's point in
// the right one, named either way round; and each refusal.
int test_calibration() {
  const camera::ArrayCalibration file = pair_file();
  CHECK(camera::validate_array_calibration(file).ok());
  auto made = sensor::stereo_calibration(file, "bm-1", "bm-2");
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  CHECK(same_model(made->left, lens(0.0)) &&
        same_model(made->right, lens(3.5)));
  const double forward = worst_difference(made->left_to_right, left_to_right());
  auto swapped = sensor::stereo_calibration(file, "bm-2", "bm-1");
  CHECK(swapped.ok());
  CHECK(same_model(swapped->left, lens(3.5)));
  const double back = worst_difference(swapped->left_to_right,
                                       camera::rigid_inverse(left_to_right()));
  std::printf("  left_to_right %.1e from the pair's, %.1e named back\n",
              forward, back);
  CHECK(forward < 1e-12 && back < 1e-12);

  CHECK(refused(
      sensor::stereo_calibration(camera::ArrayCalibration{}, "bm-1", "bm-2"),
      "stereo_calibration: array calibration: no sensors"));
  CHECK(refused(sensor::stereo_calibration(file, "bm-1", "bm-1"),
                "are both \"bm-1\""));
  CHECK(refused(sensor::stereo_calibration(file, "bm-1", "bm-3"),
                "the right camera \"bm-3\" is not in the file"));
  camera::ArrayCalibration bad = file;
  bad.sensors[0].color_to_world.reset();
  CHECK(refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"),
                "the left camera \"bm-1\" is unposed"));
  bad = file;
  bad.sensors[1].color.reset();
  CHECK(refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"),
                "the right camera \"bm-2\" has no colour model"));
  CHECK(refused(sensor::stereo_calibration(file, "bm-1", "femto"),
                "the right camera \"femto\" has no colour model"));
  bad = file;
  bad.sensors[1].color->model.size = {1920, 1080};
  CHECK(refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"),
                "sizes differ: 3840x2160 and 1920x1080"));
  // Turned, but at the left camera's place.
  bad = file;
  Mat4d turned = *bad.sensors[0].color_to_world *
                 camera::matrix_from_rodrigues({Vec3d(0, 0.1, 0), Vec3d(0.0)});
  turned[3] = (*bad.sensors[0].color_to_world)[3];
  bad.sensors[1].color_to_world = turned;
  CHECK(
      refused(sensor::stereo_calibration(bad, "bm-1", "bm-2"), "at one place"));
  std::puts("stereo_calibration: the pair either way round, each refusal");
  return 0;
}

// Frames that keep the contract pass; each rule broken is the one named.
int test_contract() {
  sensor::StereoInfo info;
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
  const auto names = [&](const std::vector<sensor::StereoFrame>& f,
                         const sensor::SensorStats& s,
                         const char* why) -> bool {
    const std::string got = stereo_contract::breach(info, f, s);
    if (got.find(why) == std::string::npos) {
      std::fprintf(stderr, "expected \"%s\": \"%s\"\n", why, got.c_str());
      return false;
    }
    return true;
  };
  CHECK(stereo_contract::breach(info, frames, stats).empty());

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
  sensor::SensorStats over = stats;
  over.failed = 2;
  CHECK(names(frames, over, "past received 5"));
  over = stats;
  over.delivered = 4;
  CHECK(names(frames, over, "delivered 4, taken 3"));
  sensor::StereoInfo exact;  // one instant: any skew breaks it
  CHECK(!stereo_contract::breach(exact, frames, stats).empty());
  std::puts("the sync contract: kept, and each rule broken named");
  return 0;
}

// The rectifier takes the pair, its baseline the cameras' distance apart.
int gpu_main(vkc::test::GpuContext& gpu) {
  const camera::ArrayCalibration file = pair_file();
  auto made = sensor::stereo_calibration(file, "bm-1", "bm-2");
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
  const double distance = glm::length(Vec3d(left_to_right()[3]));
  const double baseline = (*rectifier)->rig().baseline;
  std::printf("  rectified: baseline %.9f m (cameras %.9f m apart), fx %.2f\n",
              baseline, distance, (*rectifier)->rig().camera.intrinsics.fx);
  CHECK(std::fabs(baseline - distance) < 1e-9);
  std::puts("sensor_stereo_sensor: OK");
  return 0;
}

}  // namespace

int main() {
  if (test_calibration() != 0 || test_contract() != 0) return 1;
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
