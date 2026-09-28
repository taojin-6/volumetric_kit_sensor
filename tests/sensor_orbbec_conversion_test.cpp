// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's arithmetic, with no camera: the SDK-to-contract unit and
// layout conversions, the sync-mode mapping, and the option checks open() makes
// before it touches the SDK -- each a place the driver can be silently wrong.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>

#include "frame_conversion.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"

namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;
namespace orbbec = volumetric_kit::recon::sensor::orbbec;

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

bool invalid(const vr::Status& s) {
  return s.domain() == vr::Status::Code::InvalidArgument;
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
  o.cam_to_world[3][1] = std::numeric_limits<float>::quiet_NaN();
  CHECK(invalid(orbbec::validate(o)));

  // open() makes the same checks first, before it touches the SDK -- which is
  // what makes this runnable with no camera and no network.
  o = Options{};
  o.fps = 0;
  const auto opened = sensor::OrbbecCapture::open(o);
  CHECK(!opened.ok());
  CHECK(invalid(opened.status()));
  return 0;
}

int test_validate_rig() {
  sensor::OrbbecRig::Options r;
  r.cameras = {{"A", vr::Mat4f(1.0f)}, {"B", vr::Mat4f(1.0f)}};
  CHECK(orbbec::validate(r).ok());

  auto o = r;
  o.cameras.pop_back();
  CHECK(invalid(orbbec::validate(o)));
  o = r;
  o.cameras[1].serial = "A";
  CHECK(invalid(orbbec::validate(o)));
  // The pose file's own checks, rigidity included.
  o = r;
  o.cameras[1].cam_to_world = vr::Mat4f(2.0f);
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

}  // namespace

int main() {
  if (test_color_camera() != 0) return 1;
  if (test_same_pinhole() != 0) return 1;
  if (test_depth_to_metres() != 0) return 1;
  if (test_pack_rgb() != 0) return 1;
  if (test_sync_mode() != 0) return 1;
  if (test_validate() != 0) return 1;
  if (test_validate_rig() != 0) return 1;
  std::printf("orbbec conversion tests passed\n");
  return 0;
}
