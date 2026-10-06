// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's arithmetic, with no camera: the SDK's camera models and
// extrinsic to recon's, the sync-mode mapping, and the option checks open()
// makes before it touches the SDK -- each a place the driver can be silently
// wrong.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "frame_conversion.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_stream.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
namespace orbbec = volumetric_kit::recon::sensor::orbbec;
namespace camera = volumetric_kit::recon::camera;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

// The rig's colour intrinsics at 1280x720, as the SDK reports them.
OBCameraIntrinsic femto_color_720p() {
  OBCameraIntrinsic k{};
  k.fx = 746.494f;
  k.fy = 746.319f;
  k.cx = 630.403f;
  k.cy = 345.615f;
  k.width = 1280;
  k.height = 720;
  return k;
}

bool invalid(const vkc::Status& s) {
  return s.domain() == vkc::Status::Code::InvalidArgument;
}

int test_sync_mode() {
  CHECK(orbbec::sync_mode_from(OB_MULTI_DEVICE_SYNC_MODE_PRIMARY) ==
        sensor::OrbbecSyncMode::Primary);
  // The rig's secondaries report SECONDARY_SYNCED (16).
  CHECK(orbbec::sync_mode_from(OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED) ==
        sensor::OrbbecSyncMode::SecondarySynced);
  CHECK(orbbec::sync_mode_from(OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN) ==
        sensor::OrbbecSyncMode::FreeRun);
  CHECK(orbbec::sync_mode_from(OB_MULTI_DEVICE_SYNC_MODE_IR_IMU_SYNC) ==
        sensor::OrbbecSyncMode::Other);

  CHECK(!sensor::waits_for_primary(sensor::OrbbecSyncMode::Primary));
  CHECK(!sensor::waits_for_primary(sensor::OrbbecSyncMode::Standalone));
  CHECK(!sensor::waits_for_primary(sensor::OrbbecSyncMode::FreeRun));
  CHECK(sensor::waits_for_primary(sensor::OrbbecSyncMode::Secondary));
  CHECK(sensor::waits_for_primary(sensor::OrbbecSyncMode::SecondarySynced));
  CHECK(sensor::waits_for_primary(sensor::OrbbecSyncMode::HardwareTriggering));
  // Waits for the host, not another camera -- and open() refuses it.
  CHECK(!sensor::waits_for_primary(sensor::OrbbecSyncMode::SoftwareTriggering));

  CHECK(std::string(sensor::to_string(sensor::OrbbecSyncMode::Primary)) ==
        "primary");
  CHECK(std::string(sensor::to_string(
            sensor::OrbbecSyncMode::SecondarySynced)) == "secondary-synced");
  return 0;
}

// The SDK's lens models to the camera tier's one: Brown-Conrady's
// coefficients in OpenCV's order, the others refused.
int test_camera_model() {
  OBCameraDistortion d{};
  d.k1 = 0.1f;
  d.k2 = -0.2f;
  d.k3 = 0.03f;
  d.k4 = 0.4f;
  d.k5 = -0.05f;
  d.k6 = 0.06f;
  d.p1 = 0.001f;
  d.p2 = -0.002f;
  d.model = OB_DISTORTION_BROWN_CONRADY_K6;
  auto lens = orbbec::camera_model_from(femto_color_720p(), d, "colour");
  CHECK(lens.ok());
  CHECK(lens->intrinsics.fx == 746.494f && lens->intrinsics.cy == 345.615f);
  CHECK(lens->size.width == 1280 && lens->size.height == 720);
  CHECK(lens->distortion.k1 == 0.1f && lens->distortion.k2 == -0.2f &&
        lens->distortion.k3 == 0.03f && lens->distortion.k4 == 0.4f &&
        lens->distortion.k5 == -0.05f && lens->distortion.k6 == 0.06f &&
        lens->distortion.p1 == 0.001f && lens->distortion.p2 == -0.002f);
  d.model = OB_DISTORTION_NONE;  // coefficients ignored
  lens = orbbec::camera_model_from(femto_color_720p(), d, "colour");
  CHECK(lens.ok() && lens->distortion.k1 == 0.0f &&
        lens->distortion.p2 == 0.0f);
  d.model = OB_DISTORTION_KANNALA_BRANDT4;
  CHECK(orbbec::camera_model_from(femto_color_720p(), d, "colour")
            .status()
            .domain() == vkc::Status::Code::Unsupported);
  // The plain model is the polynomial k1..k3: whatever the SDK leaves in
  // k4..k6 is not a term of it, and would divide the radial term if read.
  d.model = OB_DISTORTION_BROWN_CONRADY;
  lens = orbbec::camera_model_from(femto_color_720p(), d, "colour");
  CHECK(lens.ok());
  CHECK(lens->distortion.k1 == 0.1f && lens->distortion.k2 == -0.2f &&
        lens->distortion.k3 == 0.03f && lens->distortion.p1 == 0.001f &&
        lens->distortion.p2 == -0.002f);
  CHECK(lens->distortion.k4 == 0.0f && lens->distortion.k5 == 0.0f &&
        lens->distortion.k6 == 0.0f);
  d.k2 = std::numeric_limits<float>::quiet_NaN();
  CHECK(invalid(
      orbbec::camera_model_from(femto_color_720p(), d, "colour").status()));
  // The stream is named in the error, the depth one included.
  OBCameraIntrinsic k = femto_color_720p();
  k.fx = 0.0f;
  d.k2 = 0.0f;
  const vkc::Status bad = orbbec::camera_model_from(k, d, "depth").status();
  CHECK(invalid(bad));
  CHECK(bad.message().find("depth") != std::string::npos);
  CHECK(bad.message().find("colour") == std::string::npos);

  // The SDK sizes images in int16_t. A negative one must be refused, not
  // wrapped into a four-billion-pixel uint32_t.
  const auto refused = [&d](const OBCameraIntrinsic& bad_k) {
    return invalid(orbbec::camera_model_from(bad_k, d, "colour").status());
  };
  k = femto_color_720p();
  k.width = -1280;
  CHECK(refused(k));
  k = femto_color_720p();
  k.height = 0;
  CHECK(refused(k));
  // What getCameraParam() returns before a stream starts: all zeros.
  CHECK(refused(OBCameraIntrinsic{}));
  k = femto_color_720p();
  k.fx = std::numeric_limits<float>::quiet_NaN();
  CHECK(refused(k));
  k = femto_color_720p();
  k.fy = -746.0f;
  CHECK(refused(k));
  k = femto_color_720p();
  k.cx = std::numeric_limits<float>::infinity();
  CHECK(refused(k));
  return 0;
}

// The SDK's extrinsic, rotation row-major and translation in millimetres, as
// a transform in metres: a quarter turn about z and a 32 mm baseline.
int test_transform_from() {
  OBExtrinsic e{};
  const float rot[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  for (int i = 0; i < 9; ++i) e.rot[i] = rot[i];
  e.trans[0] = 32.0f;
  e.trans[1] = -1.5f;
  e.trans[2] = 4.0f;
  const auto quarter = orbbec::transform_from(e);
  CHECK(quarter.ok());
  const vr::Mat4f m(quarter.value());
  const vr::Vec4f p = m * vr::Vec4f(1.0f, 2.0f, 3.0f, 1.0f);
  // R (1, 2, 3) = (-2, 1, 3), plus t.
  CHECK(std::fabs(p.x - (-2.0f + 0.032f)) < 1e-6f);
  CHECK(std::fabs(p.y - (1.0f - 0.0015f)) < 1e-6f);
  CHECK(std::fabs(p.z - (3.0f + 0.004f)) < 1e-6f);
  CHECK(p.w == 1.0f);

  // CL2A141000N's factory depth-to-colour rotation, as the SDK reported it
  // (2026-10-06): the depth camera's 6.35 degree tilt, but with its first row
  // 0.6% short of unit length. What comes out is a rotation, and the two
  // unit rows stay where they were.
  const float femto[9] = {0.993856311f,  0.006035595f,  0.003297412f,
                          -0.006363322f, 0.993843257f,  0.110612832f,
                          -0.002609496f, -0.110631190f, 0.993858099f};
  for (int i = 0; i < 9; ++i) e.rot[i] = femto[i];
  const auto made = orbbec::transform_from(e);
  CHECK(made.ok());
  const camera::Mat4d& rigid = made.value();
  CHECK(camera::check_rigid(rigid).ok());
  for (int row = 1; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      CHECK(std::fabs(rigid[col][row] - femto[3 * row + col]) < 1e-4);
    }
  }

  // A zeroed rotation, as from an uncalibrated unit, and a reflected one are
  // not a tilt: refused, rather than handed to every frame.
  for (int i = 0; i < 9; ++i) e.rot[i] = 0.0f;
  CHECK(invalid(orbbec::transform_from(e).status()));
  for (int i = 0; i < 9; ++i) e.rot[i] = femto[i];
  for (int col = 0; col < 3; ++col) e.rot[col] = -femto[col];
  CHECK(invalid(orbbec::transform_from(e).status()));
  return 0;
}

int test_validate_streams() {
  const auto check = [](const sensor::OrbbecStreamOptions& o) {
    return orbbec::validate_streams(o, "test");
  };
  using Options = sensor::OrbbecStreamOptions;
  CHECK(check(Options{}).ok());

  Options o;
  o.depth_width = 0;
  CHECK(invalid(check(o)));
  o = Options{};
  o.color_height = 0;
  CHECK(invalid(check(o)));
  o = Options{};
  o.fps = 0;
  CHECK(invalid(check(o)));
  o = Options{};
  o.min_depth = -0.1f;
  CHECK(invalid(check(o)));
  o = Options{};
  o.min_depth = 3.0f;
  o.max_depth = 3.0f;
  CHECK(invalid(check(o)));
  o = Options{};
  o.max_depth = std::numeric_limits<float>::quiet_NaN();
  CHECK(invalid(check(o)));
  o = Options{};
  o.max_depth = std::numeric_limits<float>::infinity();
  CHECK(invalid(check(o)));
  // A depth gate at 0: the GPU pass refuses it.
  o = Options{};
  o.min_depth = 0.0f;
  CHECK(invalid(check(o)));
  return 0;
}

// OrbbecSensor::open refuses the same before the SDK -- which is what makes
// this runnable with no camera and no network -- in its own name, and its
// pose's reason is kept.
int test_sensor_open() {
  const auto refused = [](const sensor::OrbbecSensor::Options& o,
                          const std::string& says) {
    const vkc::Status s = sensor::OrbbecSensor::open(o).status();
    return invalid(s) && s.message().rfind("OrbbecSensor: ", 0) == 0 &&
           s.message().find(says) != std::string::npos;
  };
  sensor::OrbbecSensor::Options o;
  o.depth_width = 0;
  CHECK(refused(o, "sizes"));
  o = {};
  o.min_depth = 0.0f;
  CHECK(refused(o, "min_depth > 0"));
  o = {};
  o.color_to_world[0][0] = -1.0;
  CHECK(refused(o, "color_to_world: transform's rotation is a reflection"));
  // Not rigid: refused at open, not at every frame the GPU pass prepares.
  o = {};
  o.color_to_world[3][1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(refused(o, "color_to_world"));
  o = {};
  o.color_to_world = camera::Mat4d(2.0);
  CHECK(refused(o, "color_to_world"));
  return 0;
}

int test_validate_rig() {
  sensor::OrbbecRig::Options r;
  r.sync.devices = {{"A", {}}, {"B", {}}};
  CHECK(orbbec::validate(r).ok());  // no calibration: every camera at origin

  auto o = r;
  o.sync.devices.pop_back();
  CHECK(invalid(orbbec::validate(o)));
  o = r;
  o.sync.devices[1].serial = "A";
  CHECK(invalid(orbbec::validate(o)));
  // A calibration must pose every camera, and pass its own checks.
  const auto posed = [](const char* id) {
    camera::SensorCalibration s;
    s.id = id;
    s.color_to_world = camera::Mat4d(1.0);
    return s;
  };
  o = r;
  o.calibration.sensors = {posed("A")};
  CHECK(invalid(orbbec::validate(o)));
  o.calibration.sensors.push_back(posed("B"));
  o.calibration.sensors.push_back(posed("C"));  // extra: fine
  CHECK(orbbec::validate(o).ok());
  o.calibration.sensors[1].color_to_world = camera::Mat4d(2.0);
  CHECK(invalid(orbbec::validate(o)));

  // The tolerance must be under half a frame period (16 666 us at 30 fps),
  // or one secondary frame can match two neighbouring triggers.
  o = r;
  o.sync_tolerance_us = 0;
  CHECK(invalid(orbbec::validate(o)));
  o.sync_tolerance_us = 16666;
  CHECK(invalid(orbbec::validate(o)));
  o.sync_tolerance_us = 16665;
  CHECK(orbbec::validate(o).ok());
  o.fps = 15;
  o.sync_tolerance_us = 20000;
  CHECK(orbbec::validate(o).ok());
  return 0;
}

// A rig whose cameras do not answer is refused, not aborted: open builds its
// state before it looks for the cameras. It queries the SDK once and opens
// no camera, so it passes whatever is attached.
int test_rig_open() {
  sensor::OrbbecRig::Options r;
  r.sync.devices = {{"VR-TEST-ABSENT-A", {}}, {"VR-TEST-ABSENT-B", {}}};
  r.discovery_timeout_ms = 1;
  CHECK(sensor::OrbbecRig::open(r).status().domain() ==
        vkc::Status::Code::NotFound);
  return 0;
}

}  // namespace

int main() {
  if (test_camera_model() != 0) return 1;
  if (test_transform_from() != 0) return 1;
  if (test_sync_mode() != 0) return 1;
  if (test_validate_streams() != 0) return 1;
  if (test_validate_rig() != 0) return 1;
  if (test_sensor_open() != 0) return 1;
  if (test_rig_open() != 0) return 1;
  std::printf("orbbec conversion tests passed\n");
  return 0;
}
