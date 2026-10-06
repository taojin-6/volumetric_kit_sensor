// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's arithmetic, with no camera: the SDK-to-contract unit and
// layout conversions, the sync-mode mapping, and the option checks open() makes
// before it touches the SDK, OrbbecSensor's too -- each a place the driver can
// be silently wrong.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

#include "frame_conversion.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"

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

int test_color_camera() {
  vr::Mat4f pose(1.0f);
  pose[3] = vr::Vec4f(0.5f, -1.0f, 2.0f, 1.0f);  // a translation to carry
  const auto cam = orbbec::color_camera_from(femto_color_720p(), pose);
  CHECK(cam.ok());
  CHECK(cam.value().fx == 746.494f && cam.value().fy == 746.319f);
  CHECK(cam.value().cx == 630.403f && cam.value().cy == 345.615f);
  CHECK(cam.value().width == 1280u && cam.value().height == 720u);
  CHECK(cam.value().cam_to_world == pose);

  // The SDK sizes images in int16_t. A negative one must be refused, not
  // wrapped into a four-billion-pixel uint32_t.
  OBCameraIntrinsic k = femto_color_720p();
  k.width = -1280;
  CHECK(invalid(orbbec::color_camera_from(k, pose).status()));
  k = femto_color_720p();
  k.height = 0;
  CHECK(invalid(orbbec::color_camera_from(k, pose).status()));
  // What getCameraParam() returns before a stream starts: all zeros.
  CHECK(invalid(orbbec::color_camera_from(OBCameraIntrinsic{}, pose).status()));
  k = femto_color_720p();
  k.fx = std::numeric_limits<float>::quiet_NaN();
  CHECK(invalid(orbbec::color_camera_from(k, pose).status()));
  k = femto_color_720p();
  k.fy = -746.0f;
  CHECK(invalid(orbbec::color_camera_from(k, pose).status()));
  k = femto_color_720p();
  k.cx = std::numeric_limits<float>::infinity();
  CHECK(invalid(orbbec::color_camera_from(k, pose).status()));
  return 0;
}

int test_same_pinhole() {
  // The first-pair check: the SDK's intrinsics of a processed frame against
  // the camera the frame is stamped with.
  const auto cam = orbbec::color_camera_from(femto_color_720p(), vr::Mat4f(1));
  CHECK(cam.ok());
  const float tol = 1e-3f;
  CHECK(orbbec::same_pinhole(femto_color_720p(), cam.value(), tol));
  OBCameraIntrinsic k = femto_color_720p();
  k.cx += 0.5f * tol;  // within tolerance
  CHECK(orbbec::same_pinhole(k, cam.value(), tol));
  k = femto_color_720p();
  k.fy += 1.0f;  // a re-projected camera matrix
  CHECK(!orbbec::same_pinhole(k, cam.value(), tol));
  // NaN compares false against everything; a `> tol` test would pass it.
  k = femto_color_720p();
  k.fx = std::numeric_limits<float>::quiet_NaN();
  CHECK(!orbbec::same_pinhole(k, cam.value(), tol));
  k = femto_color_720p();
  k.cy = std::numeric_limits<float>::quiet_NaN();
  CHECK(!orbbec::same_pinhole(k, cam.value(), tol));
  return 0;
}

int test_depth_to_metres() {
  // The Femto Mega reports a value scale of 1.0 mm per unit; other Orbbec
  // cameras report fractions. 0 is "no return" and must stay exactly 0.
  const std::uint16_t raw[] = {0, 1, 1000, 2500, 65535};
  float m[5] = {};
  orbbec::depth_to_metres(raw, 5, 1.0f, m);
  CHECK(m[0] == 0.0f);
  CHECK(std::fabs(m[1] - 0.001f) < 1e-9f);
  CHECK(std::fabs(m[2] - 1.0f) < 1e-6f);
  CHECK(std::fabs(m[3] - 2.5f) < 1e-6f);
  CHECK(std::fabs(m[4] - 65.535f) < 1e-4f);

  orbbec::depth_to_metres(raw, 5, 0.1f, m);
  CHECK(m[0] == 0.0f);
  CHECK(std::fabs(m[2] - 0.1f) < 1e-7f);
  CHECK(std::fabs(m[3] - 0.25f) < 1e-7f);
  return 0;
}

int test_pack_rgb() {
  // R first in memory, R in the low byte of the word, high byte 0.
  const std::uint8_t rgb[] = {0x11, 0x22, 0x33, 0xFF, 0x00, 0x80};
  std::uint32_t packed[2] = {0xDEADBEEFu, 0xDEADBEEFu};
  orbbec::pack_rgb(rgb, 2, packed);
  CHECK(packed[0] == 0x00332211u);
  CHECK(packed[1] == 0x008000FFu);
  return 0;
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

int test_validate() {
  using Options = sensor::OrbbecCapture::Options;
  CHECK(orbbec::validate(Options{}).ok());

  Options o;
  o.depth_width = 0;
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.color_height = 0;
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.fps = 0;
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.min_depth = -0.1f;
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.min_depth = 3.0f;
  o.max_depth = 3.0f;
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.max_depth = std::numeric_limits<float>::quiet_NaN();
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.max_depth = std::numeric_limits<float>::infinity();
  CHECK(invalid(orbbec::validate(o)));
  o = Options{};
  o.cam_to_world[3][1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(invalid(orbbec::validate(o)));
  // Not rigid: refused at open, not at every frame the GPU pass prepares.
  o = Options{};
  o.cam_to_world = camera::Mat4d(2.0);
  CHECK(invalid(orbbec::validate(o)));

  // A depth gate at 0 passes the host path, but the GPU pass refuses it.
  o = Options{};
  o.min_depth = 0.0f;
  CHECK(orbbec::validate(o).ok());
  o.raw = true;
  CHECK(invalid(orbbec::validate(o)));

  // Raw frames take either codec, each decoded for the GPU pass.
  o = Options{};
  o.raw = true;
  CHECK(orbbec::validate(o).ok());
  o.color_codec = sensor::OrbbecColorCodec::Hevc;
  CHECK(orbbec::validate(o).ok());

  // open() makes the same checks first, before it touches the SDK -- which is
  // what makes this runnable with no camera and no network.
  o = Options{};
  o.fps = 0;
  const auto opened = sensor::OrbbecCapture::open(o);
  CHECK(!opened.ok());
  CHECK(invalid(opened.status()));
#if !VR_TEST_HEVC
  // Without the decoders, H.265 colour is refused before discovery, which
  // would otherwise wait for a camera that is not there.
  o = Options{};
  o.serial = "not-a-camera";
  o.color_codec = sensor::OrbbecColorCodec::Hevc;
  CHECK(sensor::OrbbecCapture::open(o).status().domain() ==
        vkc::Status::Code::Unsupported);
#endif
  return 0;
}

// OrbbecSensor::open refuses the same before the SDK, in its own name: its
// streams are raw whatever `raw` says, and its pose's reason is kept.
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
  o.raw = false;
  o.min_depth = 0.0f;
  CHECK(refused(o, "min_depth > 0"));
  o = {};
  o.color_to_world[0][0] = -1.0;
  CHECK(refused(o, "color_to_world: transform's rotation is a reflection"));
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

  // A rig may be raw, over either codec.
  o = r;
  o.raw = true;
  o.color_codec = sensor::OrbbecColorCodec::Hevc;
  CHECK(orbbec::validate(o).ok());
  o.color_codec = sensor::OrbbecColorCodec::Mjpeg;
  CHECK(orbbec::validate(o).ok());
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

}  // namespace

int main() {
  if (test_color_camera() != 0) return 1;
  if (test_camera_model() != 0) return 1;
  if (test_transform_from() != 0) return 1;
  if (test_same_pinhole() != 0) return 1;
  if (test_depth_to_metres() != 0) return 1;
  if (test_pack_rgb() != 0) return 1;
  if (test_sync_mode() != 0) return 1;
  if (test_validate() != 0) return 1;
  if (test_validate_rig() != 0) return 1;
  if (test_sensor_open() != 0) return 1;
  std::printf("orbbec conversion tests passed\n");
  return 0;
}
