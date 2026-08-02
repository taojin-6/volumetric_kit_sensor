// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Camera-convention tests for the sensor tier: the OpenGL/ARKit -> CV pose
// conversion and the registered-depth intrinsics rescale.
//
// These pin the two things a capture integration gets silently wrong. A flipped
// axis or a half-pixel bias never raises an error -- it produces a smeared or
// subtly misaligned reconstruction that looks like a tracking problem. Pure
// host math (no device), so they always run, which is the whole reason this
// arithmetic lives in recon while the ARKit driver that feeds it does not (the
// 2026-08-02 sensor-tier decision).

#include <cmath>
#include <cstdio>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/camera_capture.hpp"
#include "volumetric_kit/recon/sensor/camera_conventions.hpp"

namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;
namespace tsdf = volumetric_kit::recon::tsdf;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

bool close(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps;
}

bool close(const vr::Vec4f& a, float x, float y, float z, float w) {
  return close(a.x, x) && close(a.y, y) && close(a.z, z) && close(a.w, w);
}

// A colour camera shaped like ARKit's: 1920x1440 with a centred principal
// point, which is what makes the centred-stays-centred property below testable.
tsdf::ColorCameraParams arkit_like_color() {
  tsdf::ColorCameraParams c{};
  c.fx = 1440.0f;
  c.fy = 1440.0f;
  c.width = 1920;
  c.height = 1440;
  c.cx = (static_cast<float>(c.width) - 1.0f) * 0.5f;   // 959.5
  c.cy = (static_cast<float>(c.height) - 1.0f) * 0.5f;  // 719.5
  c.cam_to_world = vr::Mat4f(1.0f);
  return c;
}

}  // namespace

int main() {
  // --- Pose: OpenGL/ARKit -> computer vision ------------------------------
  // Identity in, the basis change out: X kept, Y and Z negated, translation
  // untouched. Columns are basis vectors (GLM is column-major).
  {
    const vr::Mat4f out = sensor::cv_from_gl_camera(vr::Mat4f(1.0f));
    CHECK(close(out[0], 1.0f, 0.0f, 0.0f, 0.0f));
    CHECK(close(out[1], 0.0f, -1.0f, 0.0f, 0.0f));
    CHECK(close(out[2], 0.0f, 0.0f, -1.0f, 0.0f));
    CHECK(close(out[3], 0.0f, 0.0f, 0.0f, 1.0f));
  }

  // The translation column must survive untouched. This is the test that fails
  // if the multiplication is written the other way round: left-multiplying by
  // diag(1,-1,-1,1) would mirror the camera's *position* through the world
  // origin instead of turning the camera in place.
  {
    vr::Mat4f pose(1.0f);
    pose[3] = vr::Vec4f(1.5f, -2.25f, 3.0f, 1.0f);
    const vr::Mat4f out = sensor::cv_from_gl_camera(pose);
    CHECK(close(out[3], 1.5f, -2.25f, 3.0f, 1.0f));
  }

  // A camera looking down world -Z in the GL convention (its own -Z axis) is
  // looking down world -Z in CV too, but its +Z axis now points there: the
  // third column flips from (0,0,1) to (0,0,-1), and +Y down flips likewise.
  // Hand-checked rather than derived, so a sign error cannot cancel out.
  {
    vr::Mat4f pose(1.0f);
    pose[0] = vr::Vec4f(0.0f, 0.0f, -1.0f, 0.0f);
    pose[1] = vr::Vec4f(0.0f, 1.0f, 0.0f, 0.0f);
    pose[2] = vr::Vec4f(1.0f, 0.0f, 0.0f, 0.0f);
    pose[3] = vr::Vec4f(0.5f, 0.0f, 0.0f, 1.0f);
    const vr::Mat4f out = sensor::cv_from_gl_camera(pose);
    CHECK(close(out[0], 0.0f, 0.0f, -1.0f, 0.0f));  // X untouched
    CHECK(close(out[1], 0.0f, -1.0f, 0.0f, 0.0f));  // Y negated
    CHECK(close(out[2], -1.0f, 0.0f, 0.0f, 0.0f));  // Z negated
    CHECK(close(out[3], 0.5f, 0.0f, 0.0f, 1.0f));   // position untouched
  }

  // Involution: applying the change of basis twice is the identity, so a
  // double-converted pose must equal the original bit for bit.
  {
    vr::Mat4f pose(1.0f);
    pose[0] = vr::Vec4f(0.36f, 0.48f, -0.8f, 0.0f);
    pose[1] = vr::Vec4f(-0.8f, 0.6f, 0.0f, 0.0f);
    pose[2] = vr::Vec4f(0.48f, 0.64f, 0.6f, 0.0f);
    pose[3] = vr::Vec4f(-1.0f, 2.0f, 0.75f, 1.0f);
    const vr::Mat4f round_trip =
        sensor::cv_from_gl_camera(sensor::cv_from_gl_camera(pose));
    for (int c = 0; c < 4; ++c) {
      CHECK(close(round_trip[c], pose[c].x, pose[c].y, pose[c].z, pose[c].w));
    }
  }

  // --- Intrinsics: registered depth derived from colour --------------------
  const tsdf::ColorCameraParams color = arkit_like_color();

  // ARKit's real shape: 1920x1440 colour -> 256x192 depth, both 1/7.5 scale.
  {
    vr::Result<vr::volume::DepthCameraParams> r =
        sensor::depth_from_registered_color(color, 256, 192, 0.1f, 5.0f);
    CHECK(r.ok());
    const vr::volume::DepthCameraParams& d = r.value();

    // Focal lengths scale with the size ratio.
    const float s = 256.0f / 1920.0f;
    CHECK(close(d.fx, 1440.0f * s));
    CHECK(close(d.fy, 1440.0f * s));

    // THE point of the half-pixel term: a principal point at the colour
    // image's centre must land on the depth image's centre. The naive c*s
    // gives 959.5 * (256/1920) = 127.9333, which is NOT 127.5 -- so this
    // assertion fails against the naive rescale and passes against the
    // integer-centred one.
    CHECK(close(d.cx, (256.0f - 1.0f) * 0.5f));  // 127.5
    CHECK(close(d.cy, (192.0f - 1.0f) * 0.5f));  // 95.5
    CHECK(!close(d.cx, 959.5f * s, 1e-3f));      // and is not the naive value

    CHECK(d.width == 256 && d.height == 192);
    CHECK(close(d.min_depth, 0.1f) && close(d.max_depth, 5.0f));
  }

  // Registered means one physical camera: the depth pose is the colour pose,
  // never separately assigned.
  {
    tsdf::ColorCameraParams posed = color;
    posed.cam_to_world[3] = vr::Vec4f(0.25f, -1.5f, 4.0f, 1.0f);
    posed.cam_to_world[1] = vr::Vec4f(0.0f, 0.0f, 1.0f, 0.0f);
    vr::Result<vr::volume::DepthCameraParams> r =
        sensor::depth_from_registered_color(posed, 256, 192, 0.1f, 5.0f);
    CHECK(r.ok());
    for (int c = 0; c < 4; ++c) {
      const vr::Vec4f& want = posed.cam_to_world[c];
      CHECK(close(r.value().cam_to_world[c], want.x, want.y, want.z, want.w));
    }
  }

  // Rescaling to the same size is the identity -- the half-pixel term cancels
  // at s == 1, so an unscaled camera must come back untouched.
  {
    vr::Result<vr::volume::DepthCameraParams> r =
        sensor::depth_from_registered_color(color, color.width, color.height,
                                            0.1f, 5.0f);
    CHECK(r.ok());
    CHECK(close(r.value().fx, color.fx) && close(r.value().fy, color.fy));
    CHECK(close(r.value().cx, color.cx) && close(r.value().cy, color.cy));
  }

  // Non-square scaling: each axis rescales independently, so a size change in
  // one axis must not disturb the other.
  {
    vr::Result<vr::volume::DepthCameraParams> r =
        sensor::depth_from_registered_color(color, 960, 1440, 0.1f, 5.0f);
    CHECK(r.ok());
    CHECK(close(r.value().fx, 720.0f));    // halved
    CHECK(close(r.value().fy, 1440.0f));   // untouched
    CHECK(close(r.value().cx, 479.5f));    // (959.5 + 0.5)*0.5 - 0.5
    CHECK(close(r.value().cy, color.cy));  // untouched
  }

  // --- Rejected arguments ---------------------------------------------------
  {
    CHECK(!sensor::depth_from_registered_color(color, 0, 192, 0.1f, 5.0f).ok());
    CHECK(!sensor::depth_from_registered_color(color, 256, 0, 0.1f, 5.0f).ok());
    // Inverted range would reject every sample -- indistinguishable from a dead
    // sensor at the call site, so it is refused here instead.
    CHECK(
        !sensor::depth_from_registered_color(color, 256, 192, 5.0f, 0.1f).ok());
    CHECK(!sensor::depth_from_registered_color(color, 256, 192, -0.1f, 5.0f)
               .ok());
    tsdf::ColorCameraParams empty = color;
    empty.width = 0;
    CHECK(
        !sensor::depth_from_registered_color(empty, 256, 192, 0.1f, 5.0f).ok());
  }

  // --- CapturedFrame: the view the fusion tiers consume --------------------
  // Depth-only frames are ordinary (a colour camera may drop a frame), so
  // has_color() must key on the pointer rather than on the params being set.
  {
    sensor::CapturedFrame frame{};
    CHECK(!frame.has_color());
    const std::uint32_t pixel = 0xFFFFFFFFu;
    frame.color = &pixel;
    CHECK(frame.has_color());
  }

  std::printf("sensor camera-convention tests passed\n");
  return 0;
}
