// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ArkitSensor's host arithmetic (arkit/conversion.hpp, sensor_arkit_units),
// with no session and no device: arkit_depth_mm's rounding and its zeros; the
// depth copy out of buffers with padded rows, and its refusals, each named;
// the picture's checks; its colour from its tags (601, 709, untagged as 601,
// the range, the siting, HLG refused); ARKit's pinhole read column-major and
// the depth camera registered to it; ARKit's pose in this package's axes,
// and its float poses (the corner a few ulp off 1, the rotation 4e-7 off
// orthonormal) made exactly rigid, one further off refused; the clock's
// offset, kept across an interruption's re-measure; and the missed periods.

#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "conversion.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"
#include "volumetric_kit/sensor/camera_conventions.hpp"
#include "volumetric_kit/sensor/testing/nv12_pixel_buffer.hpp"
#include "volumetric_kit/sensor/video/frame_color.hpp"

namespace sensor = volumetric_kit::sensor;
namespace arkit = volumetric_kit::sensor::arkit;
namespace testing = volumetric_kit::sensor::testing;
namespace vkc = volumetric_kit::core;

namespace {

constexpr std::uint32_t kW = 256;
constexpr std::uint32_t kH = 192;
// ARKit's own row strides are padded; these are wider still, so a copy that
// steps by the packed row reads the padding.
constexpr std::size_t kDepthStride = 1088;  // 272 floats, 16 past the row
constexpr std::size_t kLevelStride = 320;

// A buffer released with this.
struct Pixels {
  CVPixelBufferRef ref = nullptr;
  Pixels() = default;
  explicit Pixels(CVPixelBufferRef r) : ref(r) {}
  ~Pixels() {
    if (ref != nullptr) CVPixelBufferRelease(ref);
  }
  Pixels(const Pixels&) = delete;
  Pixels& operator=(const Pixels&) = delete;
};

// Bytes of a buffer made over them, with the rows `stride` apart.
CVPixelBufferRef wrap(std::vector<std::uint8_t>& bytes, std::uint32_t width,
                      std::uint32_t height, OSType format, std::size_t stride) {
  CVPixelBufferRef out = nullptr;
  if (CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width, height, format,
                                   bytes.data(), stride, nullptr, nullptr,
                                   nullptr, &out) != kCVReturnSuccess) {
    return nullptr;
  }
  return out;
}

// Depth in metres at (x, y), every sample distinct in millimetres; the
// padding past each row 9.999 m, which no sample is.
float metres(std::uint32_t x, std::uint32_t y) {
  return 0.25f + 0.001f * static_cast<float>(x) +
         0.017f * static_cast<float>(y);
}

std::vector<std::uint8_t> depth_bytes() {
  std::vector<std::uint8_t> bytes(kDepthStride * kH);
  for (std::uint32_t y = 0; y < kH; ++y) {
    for (std::size_t x = 0; x < kDepthStride / sizeof(float); ++x) {
      const float v =
          x < kW ? metres(static_cast<std::uint32_t>(x), y) : 9.999f;
      std::memcpy(&bytes[y * kDepthStride + x * sizeof(float)], &v, sizeof v);
    }
  }
  return bytes;
}

std::vector<std::uint8_t> level_bytes() {
  std::vector<std::uint8_t> bytes(kLevelStride * kH, 7);  // padding: no level
  for (std::uint32_t y = 0; y < kH; ++y) {
    for (std::uint32_t x = 0; x < kW; ++x) {
      bytes[y * kLevelStride + x] = static_cast<std::uint8_t>((x + 2 * y) % 3);
    }
  }
  return bytes;
}

int test_depth_mm() {
  static_assert(sensor::arkit_depth_mm(1.0f) == 1000, "constexpr");
  CHECK(sensor::arkit_depth_mm(1.0f) == 1000);
  CHECK(sensor::arkit_depth_mm(2.0f) == 2000);
  CHECK(sensor::arkit_depth_mm(1.0004f) == 1000);
  CHECK(sensor::arkit_depth_mm(1.0006f) == 1001);
  CHECK(sensor::arkit_depth_mm(0.0004f) == 0);
  CHECK(sensor::arkit_depth_mm(0.0006f) == 1);
  CHECK(sensor::arkit_depth_mm(65.534f) == 65534);
  CHECK(sensor::arkit_depth_mm(65.6f) == 0);
  for (const float none :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), -1.0f, 0.0f, -0.0f}) {
    CHECK(sensor::arkit_depth_mm(none) == 0);
  }
  return 0;
}

int test_copy() {
  std::vector<std::uint8_t> depth = depth_bytes();
  std::vector<std::uint8_t> levels = level_bytes();
  const Pixels d(
      wrap(depth, kW, kH, kCVPixelFormatType_DepthFloat32, kDepthStride));
  const Pixels c(
      wrap(levels, kW, kH, kCVPixelFormatType_OneComponent8, kLevelStride));
  CHECK(d.ref != nullptr && c.ref != nullptr);
  CHECK(CVPixelBufferGetBytesPerRow(d.ref) == kDepthStride);
  std::vector<std::uint16_t> mm(std::size_t{kW} * kH, 0xFFFF);
  std::vector<std::uint8_t> out(std::size_t{kW} * kH, 0xFF);
  CHECK(arkit::copy_depth(d.ref, c.ref, kW, kH, mm.data(), out.data()).ok());
  for (std::uint32_t y = 0; y < kH; ++y) {
    for (std::uint32_t x = 0; x < kW; ++x) {
      const std::size_t at = std::size_t{y} * kW + x;
      CHECK(mm[at] == sensor::arkit_depth_mm(metres(x, y)));
      // As given: the GPU gates them.
      CHECK(out[at] == (x + 2 * y) % 3);
    }
  }
  // Without somewhere for the levels, the depth alone.
  std::vector<std::uint16_t> alone(std::size_t{kW} * kH, 0);
  CHECK(arkit::copy_depth(d.ref, c.ref, kW, kH, alone.data(), nullptr).ok());
  CHECK(alone == mm);
  return 0;
}

int test_copy_refusals() {
  std::vector<std::uint8_t> depth = depth_bytes();
  std::vector<std::uint8_t> levels = level_bytes();
  const Pixels d(
      wrap(depth, kW, kH, kCVPixelFormatType_DepthFloat32, kDepthStride));
  const Pixels c(
      wrap(levels, kW, kH, kCVPixelFormatType_OneComponent8, kLevelStride));
  const Pixels disparity(
      wrap(depth, kW, kH, kCVPixelFormatType_DisparityFloat32, kDepthStride));
  const Pixels wide_levels(
      wrap(depth, kW, kH, kCVPixelFormatType_OneComponent16Half, kDepthStride));
  const Pixels small_levels(
      wrap(levels, kW - 1, kH, kCVPixelFormatType_OneComponent8, kLevelStride));
  std::vector<std::uint16_t> mm(std::size_t{kW} * kH);
  std::vector<std::uint8_t> out(std::size_t{kW} * kH);
  const auto refusal = [&](CVPixelBufferRef depth_map,
                           CVPixelBufferRef confidence, std::uint32_t w,
                           std::uint32_t h) {
    const vkc::Status s =
        arkit::copy_depth(depth_map, confidence, w, h, mm.data(), out.data());
    return s.domain() == vkc::Status::Code::InvalidArgument ? s.message()
                                                            : std::string();
  };
  CHECK(refusal(disparity.ref, c.ref, kW, kH) ==
        "a depth map that is not DepthFloat32");
  CHECK(refusal(d.ref, wide_levels.ref, kW, kH) ==
        "a confidence map that is not OneComponent8");
  CHECK(refusal(d.ref, small_levels.ref, kW, kH) ==
        "a confidence map of 255 x 192, not the depth's 256 x 192");
  CHECK(refusal(d.ref, nullptr, kW, kH) == "no confidence map");
  CHECK(refusal(nullptr, c.ref, kW, kH) == "no depth map");
  CHECK(refusal(d.ref, c.ref, 320, 240) ==
        "a depth map of 256 x 192, not the sensor's 320 x 240");
  return 0;
}

CVPixelBufferRef picture(const sensor::ColorCodes& codes,
                         std::uint32_t width = 64, std::uint32_t height = 32) {
  vkc::Result<CVPixelBufferRef> made =
      testing::pixel_buffer(testing::test_pattern(width, height, 3), codes);
  return made.ok() ? made.value() : nullptr;
}

int test_picture_checks() {
  const Pixels nv12(picture({6, 1, 1, true, 0}));
  CHECK(nv12.ref != nullptr);
  CHECK(arkit::check_picture(nv12.ref, 64, 32).ok());
  CHECK(arkit::check_picture(nv12.ref, 64, 30).message() ==
        "a picture of 64 x 32, not the sensor's 64 x 30");
  CVPixelBufferRef bgra = nullptr;
  CHECK(CVPixelBufferCreate(kCFAllocatorDefault, 64, 32,
                            kCVPixelFormatType_32BGRA, nullptr,
                            &bgra) == kCVReturnSuccess);
  const Pixels bgra_held(bgra);
  CHECK(arkit::check_picture(bgra, 64, 32).message() ==
        "a picture that is not 8-bit NV12");
  // NV12, but in memory only: no IOSurface.
  CVPixelBufferRef plain = nullptr;
  CHECK(CVPixelBufferCreate(kCFAllocatorDefault, 64, 32,
                            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                            nullptr, &plain) == kCVReturnSuccess);
  const Pixels plain_held(plain);
  CHECK(arkit::check_picture(plain, 64, 32).message() ==
        "a picture not on an IOSurface");
  CHECK(arkit::check_picture(nullptr, 64, 32).message() == "no picture");
  return 0;
}

int test_picture_color() {
  const auto weights = [](sensor::VideoColorMatrix m) {
    return sensor::ycbcr_weights(m);
  };
  {
    const Pixels p(picture({6, 1, 1, true, 0}));  // ARKit's: 601, 709, full
    const vkc::Result<arkit::PictureColor> c = arkit::picture_color(p.ref);
    CHECK(c.ok());
    CHECK(c->kr == weights(sensor::VideoColorMatrix::Bt601).kr);
    CHECK(c->kb == weights(sensor::VideoColorMatrix::Bt601).kb);
    CHECK(c->full_range);
    CHECK(c->chroma_location == sensor::ChromaLocation::Left);
    CHECK(c->encoding == *sensor::resolve_encoding(1, 1));
  }
  {
    const Pixels p(picture({1, 1, 1, false, 0}));  // 709, video range
    const vkc::Result<arkit::PictureColor> c = arkit::picture_color(p.ref);
    CHECK(c.ok());
    CHECK(c->kr == weights(sensor::VideoColorMatrix::Bt709).kr);
    CHECK(c->kb == weights(sensor::VideoColorMatrix::Bt709).kb);
    CHECK(!c->full_range);
  }
  {
    // Untagged: BT.601, ARKit's, at any height (a guess would say 709 above
    // 576 rows).
    const Pixels p(picture({2, 2, 2, true, 0}, 64, 720));
    const vkc::Result<arkit::PictureColor> c = arkit::picture_color(p.ref);
    CHECK(c.ok());
    CHECK(c->kr == weights(sensor::VideoColorMatrix::Bt601).kr);
    CHECK(c->encoding == *sensor::resolve_encoding(2, 2));
  }
  {
    // P3 primaries are named, and so declared; the GPU pass decides on them.
    const Pixels p(picture({6, 1, 12, true, 0}));
    const vkc::Result<arkit::PictureColor> c = arkit::picture_color(p.ref);
    CHECK(c.ok());
    CHECK(c->encoding == *sensor::resolve_encoding(1, 12));
    CHECK(c->encoding.primaries == sensor::ColorEncoding::Primaries::DisplayP3);
  }
  {
    // The siting from its tag.
    const Pixels p(picture({6, 1, 1, true, 0}));
    CVBufferSetAttachment(p.ref, kCVImageBufferChromaLocationTopFieldKey,
                          kCVImageBufferChromaLocation_Center,
                          kCVAttachmentMode_ShouldPropagate);
    const vkc::Result<arkit::PictureColor> c = arkit::picture_color(p.ref);
    CHECK(c.ok() && c->chroma_location == sensor::ChromaLocation::Center);
  }
  {
    // HLG, which ColorEncoding cannot name: refused, named.
    const Pixels p(picture({9, 18, 9, true, 0}));
    const vkc::Result<arkit::PictureColor> c = arkit::picture_color(p.ref);
    CHECK(!c.ok());
    CHECK(c.status().message() ==
          "colour tags ColorEncoding cannot name: transfer 18, primaries 9");
  }
  return 0;
}

int test_cameras() {
  // Off centre, fx != fy, as ARKit reports them: column-major, so fx at
  // columns[0][0], fy at columns[1][1], cx at columns[2][0], cy at
  // columns[2][1].
  const std::array<float, 9> k = {1450.5f, 0.0f,    0.0f,   0.0f, 1452.25f,
                                  0.0f,    955.75f, 721.5f, 1.0f};
  const vkc::Result<sensor::camera::CameraModel> color =
      arkit::color_camera_of(k, 1920, 1440);
  CHECK(color.ok());
  CHECK(color->size.width == 1920 && color->size.height == 1440);
  CHECK(color->intrinsics.fx == 1450.5 && color->intrinsics.fy == 1452.25);
  CHECK(color->intrinsics.cx == 955.75 && color->intrinsics.cy == 721.5);
  const sensor::camera::RationalDistortion none{};
  CHECK(std::memcmp(&color->distortion, &none, sizeof none) == 0);

  const vkc::Result<sensor::camera::CameraModel> depth =
      arkit::depth_camera_of(*color, 256, 192, 0.15f, 5.0f);
  CHECK(depth.ok());
  sensor::ColorCameraParams params{};
  params.fx = 1450.5f;
  params.fy = 1452.25f;
  params.cx = 955.75f;
  params.cy = 721.5f;
  params.width = 1920;
  params.height = 1440;
  params.cam_to_world = sensor::Mat4f(1.0f);
  const vkc::Result<sensor::DepthCameraParams> want =
      sensor::depth_from_registered_color(params, 256, 192, 0.15f, 5.0f);
  CHECK(want.ok());
  CHECK(depth->size.width == 256 && depth->size.height == 192);
  CHECK(depth->intrinsics.fx == double{want->fx});
  CHECK(depth->intrinsics.fy == double{want->fy});
  CHECK(depth->intrinsics.cx == double{want->cx});
  CHECK(depth->intrinsics.cy == double{want->cy});

  // A pinhole the camera tier refuses is refused.
  std::array<float, 9> bad = k;
  bad[0] = 0.0f;
  CHECK(!arkit::color_camera_of(bad, 1920, 1440).ok());
  CHECK(!arkit::color_camera_of(k, 0, 1440).ok());
  return 0;
}

// ARCamera.transform in this package's axes, as float left it: the columns
// negated, nothing made exact.
sensor::camera::Mat4d raw_pose(const std::array<float, 16>& t) {
  sensor::Mat4f gl(1.0f);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) gl[c][r] = t[std::size_t(c * 4 + r)];
  }
  return sensor::camera::Mat4d(sensor::cv_from_gl_camera(gl));
}

int test_pose() {
  // ARKit's camera at (1, 2, 3), yawed 90 degrees about +Y (up): its -Z,
  // ARKit's forward, points along world -X. Column-major.
  const std::array<float, 16> t = {0.0f, 0.0f, -1.0f, 0.0f,  //
                                   0.0f, 1.0f, 0.0f,  0.0f,  //
                                   1.0f, 0.0f, 0.0f,  0.0f,  //
                                   1.0f, 2.0f, 3.0f,  1.0f};
  const vkc::Result<sensor::camera::Mat4d> converted =
      arkit::color_to_world_of(t);
  CHECK(converted.ok());
  const sensor::camera::Mat4d& pose = *converted;
  const sensor::camera::Mat4d to_camera = sensor::camera::rigid_inverse(pose);
  const auto in_camera = [&](double x, double y, double z) {
    return sensor::camera::Vec3d(to_camera * glm::dvec4(x, y, z, 1.0));
  };
  const auto near = [](double a, double b) { return std::fabs(a - b) < 1e-6; };
  // A point 1 m in front: +Z in this package's camera axes.
  const sensor::camera::Vec3d ahead = in_camera(0.0, 2.0, 3.0);
  CHECK(near(ahead.x, 0.0) && near(ahead.y, 0.0) && near(ahead.z, 1.0));
  // A point 1 m in front and 1 m up: the image's top, camera -Y.
  const sensor::camera::Vec3d up = in_camera(0.0, 3.0, 3.0);
  CHECK(near(up.x, 0.0) && near(up.y, -1.0) && near(up.z, 1.0));
  // And 1 m to the camera's right (ARKit's +X, world -Z once yawed): camera
  // +X.
  const sensor::camera::Vec3d right = in_camera(0.0, 2.0, 2.0);
  CHECK(near(right.x, 1.0) && near(right.y, 0.0) && near(right.z, 1.0));
  // The translation is kept.
  CHECK(pose[3][0] == 1.0 && pose[3][1] == 2.0 && pose[3][2] == 3.0);
  CHECK(sensor::camera::check_rigid(pose).ok());
  // Exact already: as given, the two columns negated.
  CHECK(pose == raw_pose(t));
  return 0;
}

// The largest entry's distance of R^T R from the identity's.
double orthonormal_offset(const sensor::camera::Mat4d& m) {
  const sensor::camera::Mat3d r(m);
  double offset = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      offset = std::fmax(
          offset, std::fabs(glm::dot(r[a], r[b]) - (a == b ? 1.0 : 0.0)));
    }
  }
  return offset;
}

// ARCamera.transform as ARKit hands one over, in float: a turn and a
// translation rounded, its second column stretched 2e-7 (R^T R 4e-7 off the
// identity), the corner `corner` and the row's first entry `row`.
std::array<float, 16> arkit_like(float corner, float row) {
  sensor::camera::Mat3d r =
      sensor::camera::rotation_from_rodrigues({0.21, 1.13, -0.17});
  r[1] *= 1.0 + 2.0e-7;
  std::array<float, 16> t{};
  for (int c = 0; c < 3; ++c) {
    for (int k = 0; k < 3; ++k) t[std::size_t(c * 4 + k)] = float(r[c][k]);
  }
  t[3] = row;
  t[12] = 0.375f;
  t[13] = 1.3f;
  t[14] = -2.0625f;
  t[15] = corner;
  return t;
}

// ARKit's float poses come out exactly rigid: the bottom row [0, 0, 0, 1],
// the rotation the nearest one, orthonormal to double's rounding; one further
// from rigid than float leaves one is refused, named.
int test_pose_exact() {
  const float above = std::nextafter(
      std::nextafter(std::nextafter(1.0f, 2.0f), 2.0f), 2.0f);  // 3 ulp
  const float below = std::nextafter(1.0f, 0.0f);
  for (const std::array<float, 16>& t :
       {arkit_like(above, 0.0f), arkit_like(below, 1.0e-7f),
        arkit_like(1.0f, 0.0f)}) {
    const sensor::camera::Mat4d raw = raw_pose(t);
    CHECK(orthonormal_offset(raw) > 2.5e-7);
    CHECK(!sensor::camera::check_rigid(raw).ok() || t[15] == 1.0f);
    const vkc::Result<sensor::camera::Mat4d> pose = arkit::color_to_world_of(t);
    CHECK(pose.ok());
    const sensor::camera::Mat4d& m = *pose;
    CHECK(m[0][3] == 0.0 && m[1][3] == 0.0 && m[2][3] == 0.0 && m[3][3] == 1.0);
    CHECK(orthonormal_offset(m) < 1e-12);
    CHECK(sensor::camera::check_rigid(m).ok());
    // The translation as ARKit gave it; the rotation within its rounding.
    CHECK(m[3][0] == 0.375 && m[3][1] == double(1.3f) && m[3][2] == -2.0625);
    for (int c = 0; c < 3; ++c) {
      for (int r = 0; r < 3; ++r) CHECK(std::fabs(m[c][r] - raw[c][r]) < 1e-6);
    }
    // Deterministic: the same transform, the same bits.
    const vkc::Result<sensor::camera::Mat4d> again =
        arkit::color_to_world_of(t);
    CHECK(again.ok() && *again == m);
  }

  // Further off than float leaves a pose: refused, named.
  std::array<float, 16> stretched = arkit_like(1.0f, 0.0f);
  for (int k = 0; k < 3; ++k) stretched[std::size_t(k)] *= 1.001f;
  const vkc::Result<sensor::camera::Mat4d> far =
      arkit::color_to_world_of(stretched);
  CHECK(!far.ok() &&
        far.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(far.status().message() ==
        "ARKit's pose: transform's rotation is more than 1e-4 off orthonormal");
  std::array<float, 16> lifted = arkit_like(1.0f, 0.0f);
  lifted[7] = 1.0e-3f;
  CHECK(arkit::color_to_world_of(lifted).status().message() ==
        "ARKit's pose: transform's bottom row is more than 1e-6 off "
        "[0, 0, 0, 1]");
  std::array<float, 16> lost = arkit_like(1.0f, 0.0f);
  lost[13] = std::numeric_limits<float>::quiet_NaN();
  CHECK(arkit::color_to_world_of(lost).status().message() ==
        "ARKit's pose: transform is not finite");
  return 0;
}

int test_clock() {
  constexpr std::int64_t kSystem = 1'700'000'000'000'000'123;
  constexpr std::int64_t kUptime = 5'000'000'000;
  arkit::ClockMap clock;
  clock.start(kSystem, kUptime);
  CHECK(clock.offset_ns() == kSystem - kUptime);
  // Uptime seconds to system_clock ns, exactly: the uptime then the offset.
  CHECK(clock.to_system_ns(5.0) == std::uint64_t{kSystem});
  CHECK(clock.to_system_ns(5.016666666) == std::uint64_t{kSystem} + 16'666'666);
  std::uint64_t last = 0;
  double t = 5.0;
  for (int i = 0; i < 10; ++i, t += 1.0 / 60) {
    const std::uint64_t now = clock.to_system_ns(t);
    CHECK(now > last);
    last = now;
  }
  // The system clock set back 50 ms during an interruption: the offset is
  // kept, so the times go on increasing.
  clock.remeasure(kSystem + 1'000'000'000 - 50'000'000,
                  kUptime + 1'000'000'000);
  CHECK(clock.offset_ns() == kSystem - kUptime);
  for (int i = 0; i < 10; ++i, t += 1.0 / 60) {
    const std::uint64_t now = clock.to_system_ns(t);
    CHECK(now > last);
    last = now;
  }
  // The device slept 30 s, which the uptime clock does not count: the
  // offset grows, so the times jump ahead, still increasing.
  clock.remeasure(kSystem + 31'000'000'000, kUptime + 1'000'000'000);
  CHECK(clock.offset_ns() == kSystem - kUptime + 30'000'000'000);
  const std::uint64_t after = clock.to_system_ns(t);
  CHECK(after > last + 29'000'000'000);
  // A new start takes the offset as it is.
  clock.start(kSystem, kUptime);
  CHECK(clock.offset_ns() == kSystem - kUptime);
  return 0;
}

int test_missed() {
  CHECK(arkit::missed_periods(10.0, 10.0 + 1.0 / 60, 60) == 0);
  CHECK(arkit::missed_periods(10.0, 10.0 + 3.0 / 60, 60) == 2);
  CHECK(arkit::missed_periods(10.0, 10.0 + 2.0 / 60, 60) == 1);
  // ARKit's jitter: under half a period either way is on time.
  CHECK(arkit::missed_periods(10.0, 10.0 + 1.4 / 60, 60) == 0);
  CHECK(arkit::missed_periods(10.0, 10.0 + 2.6 / 60, 60) == 2);
  CHECK(arkit::missed_periods(10.0, 10.0, 60) == 0);
  CHECK(arkit::missed_periods(10.0, 9.0, 60) == 0);
  CHECK(arkit::missed_periods(10.0, 11.0, 30) == 29);
  return 0;
}

int test_tracking_names() {
  CHECK(std::string(sensor::to_string(sensor::ArkitTracking::Normal)) ==
        "normal");
  CHECK(std::string(sensor::to_string(sensor::ArkitTracking::Relocalizing)) ==
        "relocalizing");
  CHECK(std::string(sensor::to_string(sensor::ArkitTracking::NotAvailable)) ==
        "not available");
  return 0;
}

int test_noise_constant() {
  // ARKit's measured curve, as is, and a model known() takes.
  CHECK(sensor::kArkitDepthNoise.a == 8.58e-3f);
  CHECK(sensor::kArkitDepthNoise.b == -7.67e-3f);
  CHECK(sensor::kArkitDepthNoise.c == 3.79e-3f);
  CHECK(sensor::kArkitDepthNoise.known());
  return 0;
}

}  // namespace

int main() {
  if (test_depth_mm() != 0) return 1;
  if (test_copy() != 0) return 1;
  if (test_copy_refusals() != 0) return 1;
  if (test_picture_checks() != 0) return 1;
  if (test_picture_color() != 0) return 1;
  if (test_cameras() != 0) return 1;
  if (test_pose() != 0) return 1;
  if (test_pose_exact() != 0) return 1;
  if (test_clock() != 0) return 1;
  if (test_missed() != 0) return 1;
  if (test_tracking_names() != 0) return 1;
  if (test_noise_constant() != 0) return 1;
  std::puts("sensor_arkit_conversion: OK");
  return 0;
}
