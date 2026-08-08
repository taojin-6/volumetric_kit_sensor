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
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>

#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/camera_capture.hpp"
#include "volumetric_kit/recon/sensor/camera_conventions.hpp"
#include "volumetric_kit/recon/sensor/color_conventions.hpp"

// Only two namespaces: the sensor contract, and core for the camera types and
// Status/Result. That this test never names the volume or tsdf tier is the
// point -- it is the same surface an out-of-tree driver compiles against.
namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;

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
vr::ColorCameraParams arkit_like_color() {
  vr::ColorCameraParams c{};
  c.fx = 1440.0f;
  c.fy = 1440.0f;
  c.width = 1920;
  c.height = 1440;
  c.cx = (static_cast<float>(c.width) - 1.0f) * 0.5f;   // 959.5
  c.cy = (static_cast<float>(c.height) - 1.0f) * 0.5f;  // 719.5
  c.cam_to_world = vr::Mat4f(1.0f);
  return c;
}

// A minimal in-tree implementation of the capture contract.
//
// This tier ships an interface and deliberately no driver (the 2026-08-02
// decision), so without something like this nothing here ever compiles
// ICameraCapture as a base class -- and the first real implementer is in
// another repo, where a defect in the contract shows up as *their* compile
// error. Written the way a driver is: buffer the newest frame, hand it over
// once, and distinguish "nothing yet" from "the device broke".
class FakeCapture final : public sensor::ICameraCapture {
 public:
  vr::Status start() override {
    if (failed_) return vr::Status::io_error("FakeCapture: device faulted");
    running_ = true;
    pending_ = make_frame();  // one frame waiting, as a live sensor would have
    return {};
  }

  void stop() noexcept override {
    running_ = false;
    pending_.reset();
  }

  vr::Result<std::optional<sensor::CapturedFrame>> poll() override {
    if (failed_) return vr::Status::io_error("FakeCapture: device faulted");
    if (!running_ || !pending_) return no_frame();
    sensor::CapturedFrame frame = *pending_;
    pending_.reset();
    return some_frame(frame);
  }

  // Make the next call report a device failure rather than an empty poll.
  void fail() noexcept { failed_ = true; }

  const float* depth_pixels() const noexcept { return depth_; }

 private:
  sensor::CapturedFrame make_frame() const {
    sensor::CapturedFrame frame{};
    frame.depth = depth_;
    vr::Result<vr::DepthCameraParams> cam = sensor::depth_from_registered_color(
        arkit_like_color(), 256, 192, 0.1f, 5.0f);
    if (cam.ok()) frame.depth_camera = cam.value();
    frame.timestamp_ns = 1;
    return frame;
  }

  // Borrowed by CapturedFrame, which is a non-owning view -- so this outlives
  // every frame it hands out, exactly as a driver's own buffer must.
  float depth_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  std::optional<sensor::CapturedFrame> pending_;
  bool running_ = false;
  bool failed_ = false;
};

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
  const vr::ColorCameraParams color = arkit_like_color();

  // ARKit's real shape: 1920x1440 colour -> 256x192 depth, both 1/7.5 scale.
  {
    vr::Result<vr::DepthCameraParams> r =
        sensor::depth_from_registered_color(color, 256, 192, 0.1f, 5.0f);
    CHECK(r.ok());
    const vr::DepthCameraParams& d = r.value();

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
    vr::ColorCameraParams posed = color;
    posed.cam_to_world[3] = vr::Vec4f(0.25f, -1.5f, 4.0f, 1.0f);
    posed.cam_to_world[1] = vr::Vec4f(0.0f, 0.0f, 1.0f, 0.0f);
    vr::Result<vr::DepthCameraParams> r =
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
    vr::Result<vr::DepthCameraParams> r = sensor::depth_from_registered_color(
        color, color.width, color.height, 0.1f, 5.0f);
    CHECK(r.ok());
    CHECK(close(r.value().fx, color.fx) && close(r.value().fy, color.fy));
    CHECK(close(r.value().cx, color.cx) && close(r.value().cy, color.cy));
  }

  // Non-square scaling: each axis rescales independently, so a size change in
  // one axis must not disturb the other.
  {
    vr::Result<vr::DepthCameraParams> r =
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
    vr::ColorCameraParams empty = color;
    empty.width = 0;
    CHECK(
        !sensor::depth_from_registered_color(empty, 256, 192, 0.1f, 5.0f).ok());

    // A focal length the unprojection divides by: zero, negative, and
    // non-finite all produce inf/NaN rays instead of an error downstream, so
    // they are refused where the argument still has a name.
    for (const float bad :
         {0.0f, -1440.0f, std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
      vr::ColorCameraParams bad_fx = color;
      bad_fx.fx = bad;
      CHECK(!sensor::depth_from_registered_color(bad_fx, 256, 192, 0.1f, 5.0f)
                 .ok());
      vr::ColorCameraParams bad_fy = color;
      bad_fy.fy = bad;
      CHECK(!sensor::depth_from_registered_color(bad_fy, 256, 192, 0.1f, 5.0f)
                 .ok());
    }
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

  // --- ICameraCapture: the contract itself ---------------------------------
  // FakeCapture above is the point of this block: this tier ships an interface
  // and no driver, so unless something in-tree implements it, nothing proves
  // the contract is implementable at all -- and the first implementer is in
  // another repo, where a mistake here surfaces as their compile error. Drive
  // it through the base-class pointer a consumer would hold.
  {
    FakeCapture capture;
    sensor::ICameraCapture& device = capture;

    CHECK(device.poll().ok());  // not running: no frame, not an error
    CHECK(!device.poll().value().has_value());

    CHECK(device.start().ok());
    CHECK(device.start().ok());  // idempotent

    vr::Result<std::optional<sensor::CapturedFrame>> first = device.poll();
    CHECK(first.ok() && first.value().has_value());
    CHECK(first.value()->depth == capture.depth_pixels());
    CHECK(first.value()->depth_camera.width == 256);
    CHECK(!first.value()->has_color());

    // Each frame is handed over once: polling faster than the sensor runs is
    // the ordinary case, and must read as "nothing yet", never as an error.
    vr::Result<std::optional<sensor::CapturedFrame>> second = device.poll();
    CHECK(second.ok() && !second.value().has_value());

    // A device failure is distinguishable from an empty poll.
    capture.fail();
    CHECK(!device.poll().ok());

    device.stop();
    device.stop();  // idempotent, and safe after a failure
  }

  // --- to_canonical: the colour half of the capture boundary ----------------
  {
    // A declaration that is already canonical costs an identity. This is the
    // common path by construction -- the canonical form was chosen to match
    // what the sensors in hand produce -- and ARKit's `{Bt709, Bt709}` takes
    // it, so the frame is not walked at all.
    const std::uint32_t src[3] = {0xFF0000FFu, 0xFF00FF00u, 0xFF123456u};
    std::uint32_t dst[3] = {0u, 0u, 0u};
    CHECK(sensor::to_canonical(src, 3, vr::ColorEncoding{}, dst).ok());
    CHECK(dst[0] == src[0] && dst[1] == src[1] && dst[2] == src[2]);
    // Zeroed first, and all three checked. Left holding the previous call's
    // output, `dst` already satisfied this, so the assertion could not tell a
    // call that carried all three words across from one that wrote nothing.
    // It still cannot tell the canonical path from the conversion path -- for
    // 8-bit input the decode/encode round trip is the identity, which is
    // exactly why Bt709 is accepted as sRGB rather than converted -- so the
    // predicate itself is pinned where it lives, in core_color_space_test.
    dst[0] = dst[1] = dst[2] = 0u;
    CHECK(sensor::to_canonical(src, 3,
                               {vr::ColorEncoding::Transfer::Bt709,
                                vr::ColorEncoding::Primaries::Bt709},
                               dst)
              .ok());
    CHECK(dst[0] == src[0] && dst[1] == src[1] && dst[2] == src[2]);

    // In-place is allowed and is the same identity.
    std::uint32_t inplace[2] = {0xFF804020u, 0xFF010203u};
    CHECK(sensor::to_canonical(inplace, 2, vr::ColorEncoding{}, inplace).ok());
    CHECK(inplace[0] == 0xFF804020u && inplace[1] == 0xFF010203u);

    // Alpha is forced to 0xFF on the identity path too, not only where the
    // curve runs -- a guarantee that held on one branch would be worth nothing
    // to the driver relying on it. The colour bytes still cross verbatim. This
    // is what lets a converted frame double as the projective-texturing atlas
    // (alpha 0 would sample fully transparent), and it is what keeps the tsdf
    // tier's "colour unobserved" sentinel exact, since a written colour is then
    // never 0 -- a source that packs its high byte as 0 is not exotic.
    const std::uint32_t no_alpha[2] = {0x00123456u, 0x00000000u};
    std::uint32_t alpha_dst[2] = {0u, 0u};
    CHECK(
        sensor::to_canonical(no_alpha, 2, vr::ColorEncoding{}, alpha_dst).ok());
    CHECK(alpha_dst[0] == 0xFF123456u);  // colour verbatim, alpha forced
    CHECK(alpha_dst[1] == 0xFF000000u);  // pure black is non-zero: the sentinel
    std::uint32_t alpha_inplace[1] = {0x00654321u};
    CHECK(sensor::to_canonical(alpha_inplace, 1, vr::ColorEncoding{},
                               alpha_inplace)
              .ok());
    CHECK(alpha_inplace[0] == 0xFF654321u);

    // A LINEAR 8-bit source is encoded on the way in: mid-grey 128/255 linear
    // becomes ~0.7356 encoded, i.e. code ~188. Getting this backwards (or
    // skipping it) is the washed-out/darkened frame that has no error message.
    const std::uint32_t linear_src[1] = {0xFF808080u};
    std::uint32_t linear_dst[1] = {0u};
    CHECK(sensor::to_canonical(linear_src, 1,
                               {vr::ColorEncoding::Transfer::Linear,
                                vr::ColorEncoding::Primaries::Bt709},
                               linear_dst)
              .ok());
    CHECK((linear_dst[0] & 0xFFu) >= 186u && (linear_dst[0] & 0xFFu) <= 190u);
    CHECK((linear_dst[0] >> 24) == 0xFFu);  // alpha always forced

    // A wide-gamut source is ROTATED, not merely decoded -- the step that makes
    // ColorEncoding::Primaries a value something converts. A saturated P3 green
    // lies outside BT.709, so it clips to full green with the off-channels
    // driven to zero; a decode-only implementation would leave it unchanged at
    // (0, 255, 0)'s exact encoding with a non-zero red, so this discriminates.
    const std::uint32_t p3_green[1] = {0xFF00FF00u};
    std::uint32_t rotated[1] = {0u};
    CHECK(sensor::to_canonical(p3_green, 1,
                               {vr::ColorEncoding::Transfer::Srgb,
                                vr::ColorEncoding::Primaries::DisplayP3},
                               rotated)
              .ok());
    CHECK((rotated[0] & 0xFFu) == 0u);           // R clipped from negative
    CHECK(((rotated[0] >> 8) & 0xFFu) == 255u);  // G saturated past 1.0
    // White survives the rotation exactly, since every declarable primary set
    // is D65 -- the invariant that catches a transposed matrix.
    const std::uint32_t p3_white[1] = {0xFFFFFFFFu};
    CHECK(sensor::to_canonical(p3_white, 1,
                               {vr::ColorEncoding::Transfer::Srgb,
                                vr::ColorEncoding::Primaries::DisplayP3},
                               rotated)
              .ok());
    CHECK((rotated[0] & 0xFFFFFFu) == 0xFFFFFFu);

    // PQ is refused rather than approximated: mapping an absolute-luminance HDR
    // curve into an 8-bit SDR form is tone mapping, which this repo does not
    // do. A driver that declares it gets an inspectable error instead of a
    // quietly wrong reconstruction -- the point of declaring at all.
    CHECK(!sensor::to_canonical(src, 3,
                                {vr::ColorEncoding::Transfer::Bt2020Pq,
                                 vr::ColorEncoding::Primaries::Bt2020},
                                dst)
               .ok());
    // ...and refused for an EMPTY frame too, ahead of the zero-count shortcut:
    // the declaration is unsupported however many pixels carry it, and a
    // driver whose first poll returns nothing should not have the label it will
    // be refused for on the next frame validated here.
    CHECK(!sensor::to_canonical(nullptr, 0,
                                {vr::ColorEncoding::Transfer::Bt2020Pq,
                                 vr::ColorEncoding::Primaries::Bt2020},
                                nullptr)
               .ok());

    // Degenerate arguments.
    CHECK(sensor::to_canonical(nullptr, 0, vr::ColorEncoding{}, nullptr).ok());
    CHECK(!sensor::to_canonical(nullptr, 3, vr::ColorEncoding{}, dst).ok());
  }

  std::printf(
      "sensor camera- and colour-convention tests passed (canonical encodings "
      "convert by identity, a linear source is encoded, a P3 source is rotated "
      "into BT.709 and clips, and PQ is refused)\n");
  return 0;
}
