// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The RealSense drivers' arithmetic (realsense/conversion.hpp), with no
// camera, against librealsense's own rsutil.h:
//   - the D455's colour lens (inverse Brown-Conrady) projects as
//     rs2_project_point_to_pixel does within 0.05 px over its picture, and
//     plain Brown-Conrady as it does to float rounding;
//   - the depth stream is a pinhole, and refused if it reports a lens; other
//     lens models with coefficients are refused, named; bad intrinsics too;
//   - the extrinsic moves points as rs2_transform_point_to_point does, and a
//     non-rigid one is refused;
//   - the infrared cameras' StereoInfo: a point projects where rsutil.h puts
//     it in the left, right and colour cameras, on one row in both infrared
//     ones, fx * baseline / z apart; a lens on either infrared camera and a
//     non-rigid extrinsic are refused, each named;
//   - a frameset is one moment when it holds a frame of each mode opened,
//     each within half a frame of the first;
//   - a frame points into the view, YUYV on the host coded as librealsense
//     converts it, stamped in nanoseconds; a view that contradicts its
//     streams is refused;
//   - a stereo frame points into the view: each infrared picture luma on the
//     host, full range, its rows as the camera's; the colour as the frame's;
//     a view that contradicts its streams is refused;
//   - the options each driver's open refuses before it looks for a camera.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <librealsense2/rsutil.h>

#include "conversion.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/projection.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace realsense = volumetric_kit::sensor::realsense;

namespace {

const std::string kWho = "RealSenseSensor: camera 123";

// The lab D455's colour camera at 1280x800, as librealsense 2.59.1 reports it.
rs2_intrinsics d455_color() {
  rs2_intrinsics in{};
  in.width = 1280;
  in.height = 800;
  in.ppx = 647.659f;
  in.ppy = 406.994f;
  in.fx = 643.013f;
  in.fy = 642.299f;
  in.model = RS2_DISTORTION_INVERSE_BROWN_CONRADY;
  const float c[5] = {-0.054965f, 0.0648387f, -0.000696181f, -0.000130702f,
                      -0.0209472f};
  for (int i = 0; i < 5; ++i) in.coeffs[i] = c[i];
  return in;
}

// The lab D455's infrared cameras (and depth, in the left one) at 1280x720:
// rectified on the camera, so a pinhole, the same for both.
rs2_intrinsics d455_infrared() {
  rs2_intrinsics in{};
  in.width = 1280;
  in.height = 720;
  in.ppx = 651.384f;
  in.ppy = 359.031f;
  in.fx = in.fy = 657.271f;
  in.model = RS2_DISTORTION_BROWN_CONRADY;
  return in;
}

// The lab D455's depth (left infrared) camera to its colour camera.
rs2_extrinsics d455_left_to_color() {
  rs2_extrinsics e{};
  const float r[9] = {0.999996f,    0.00231108f, -0.00170157f,
                      -0.00231856f, 0.999988f,   -0.00441104f,
                      0.00169135f,  0.00441496f, 0.999989f};
  for (int i = 0; i < 9; ++i) e.rotation[i] = r[i];
  e.translation[0] = -0.0590472f;
  e.translation[1] = -7.2951e-05f;
  e.translation[2] = 0.000360092f;
  return e;
}

// The lab D455's left infrared camera to its right one: no rotation, the
// baseline (95.03 mm) along x.
rs2_extrinsics d455_left_to_right() {
  rs2_extrinsics e{};
  e.rotation[0] = e.rotation[4] = e.rotation[8] = 1.0f;
  e.translation[0] = -0.0950313f;
  return e;
}

// Where `model` images camera point `p` (its lens included).
glm::dvec2 project(const camera::CameraModel& model, const glm::dvec3& p) {
  const double d[8] = {model.distortion.k1, model.distortion.k2,
                       model.distortion.p1, model.distortion.p2,
                       model.distortion.k3, model.distortion.k4,
                       model.distortion.k5, model.distortion.k6};
  double xd = 0.0, yd = 0.0;
  camera::distort_rational(d, p.x / p.z, p.y / p.z, &xd, &yd);
  return {xd * model.intrinsics.fx + model.intrinsics.cx,
          yd * model.intrinsics.fy + model.intrinsics.cy};
}

bool inside(const float pixel[2], const rs2_intrinsics& in) {
  return pixel[0] >= 0.0f && pixel[1] >= 0.0f && pixel[0] < in.width &&
         pixel[1] < in.height;
}

// The worst distance, in pixels, between where rsutil.h projects a grid of
// rays and where the camera model does, over those landing in the picture.
double worst_projection(const rs2_intrinsics& in,
                        const camera::CameraModel& model) {
  double worst = 0.0;
  int seen = 0;
  for (int j = 0; j <= 40; ++j) {
    for (int i = 0; i <= 60; ++i) {
      const float point[3] = {-1.1f + 2.2f * static_cast<float>(i) / 60.0f,
                              -0.7f + 1.4f * static_cast<float>(j) / 40.0f,
                              1.0f};
      float pixel[2];
      rs2_project_point_to_pixel(pixel, &in, point);
      if (!inside(pixel, in)) continue;
      ++seen;
      const glm::dvec2 ours =
          project(model, glm::dvec3(point[0], point[1], point[2]));
      worst = std::fmax(worst, std::hypot(ours.x - pixel[0], ours.y - pixel[1]));
    }
  }
  return seen > 1000 ? worst : std::numeric_limits<double>::infinity();
}

bool unsupported(const vkc::Status& s, const char* why) {
  const bool ok = s.domain() == vkc::Status::Code::Unsupported &&
                  s.message().find(why) != std::string::npos;
  if (!ok) std::fprintf(stderr, "unexpected: %s\n", s.message().c_str());
  return ok;
}

int test_lenses() {
  const rs2_intrinsics color = d455_color();
  auto model = realsense::camera_model_from(color, "colour", true, kWho);
  CHECK(model.ok());
  const camera::CameraModel& m = model.value();
  CHECK(m.size.width == 1280 && m.size.height == 800);
  CHECK(m.intrinsics.fx == double{color.fx} &&
        m.intrinsics.cy == double{color.ppy});
  CHECK(m.distortion.k1 == double{color.coeffs[0]} &&
        m.distortion.p1 == double{color.coeffs[2]} &&
        m.distortion.p2 == double{color.coeffs[3]} &&
        m.distortion.k3 == double{color.coeffs[4]});
  CHECK(m.distortion.k4 == 0.0 && m.distortion.k5 == 0.0 &&
        m.distortion.k6 == 0.0);
  const double inverse = worst_projection(color, m);
  // Plain Brown-Conrady is OpenCV's model itself: only float rounding apart.
  rs2_intrinsics plain = color;
  plain.model = RS2_DISTORTION_BROWN_CONRADY;
  auto plain_model = realsense::camera_model_from(plain, "colour", true, kWho);
  CHECK(plain_model.ok());
  const double exact = worst_projection(plain, plain_model.value());
  std::printf(
      "  lenses: inverse Brown-Conrady %.4f px from rsutil.h at worst, "
      "Brown-Conrady %.5f px\n",
      inverse, exact);
  CHECK(inverse < 0.05);
  CHECK(exact < 0.002);

  // The depth stream: rectified, a pinhole, whatever model it names.
  rs2_intrinsics depth = d455_infrared();
  auto pinhole = realsense::camera_model_from(depth, "depth", false, kWho);
  CHECK(pinhole.ok());
  CHECK(pinhole->distortion.k1 == 0.0 && pinhole->distortion.p1 == 0.0);
  depth.coeffs[3] = 1e-4f;
  CHECK(unsupported(
      realsense::camera_model_from(depth, "depth", false, kWho).status(),
      "the depth stream reports a lens"));

  // Lens models the pass does not undistort, with coefficients; without
  // any, a pinhole.
  for (const rs2_distortion other :
       {RS2_DISTORTION_MODIFIED_BROWN_CONRADY, RS2_DISTORTION_FTHETA,
        RS2_DISTORTION_KANNALA_BRANDT4}) {
    rs2_intrinsics in = color;
    in.model = other;
    CHECK(unsupported(
        realsense::camera_model_from(in, "colour", true, kWho).status(),
        rs2_distortion_to_string(other)));
    for (float& c : in.coeffs) c = 0.0f;
    CHECK(realsense::camera_model_from(in, "colour", true, kWho).ok());
  }

  // Intrinsics that are no camera.
  rs2_intrinsics bad = color;
  bad.width = 0;
  CHECK(unsupported(
      realsense::camera_model_from(bad, "colour", true, kWho).status(),
      "not finite and positive"));
  bad = color;
  bad.fx = std::numeric_limits<float>::quiet_NaN();
  CHECK(unsupported(
      realsense::camera_model_from(bad, "colour", true, kWho).status(),
      "not finite and positive"));
  bad = color;
  bad.fy = -1.0f;
  CHECK(unsupported(
      realsense::camera_model_from(bad, "colour", true, kWho).status(),
      "not finite and positive"));
  return 0;
}

int test_extrinsic() {
  const rs2_extrinsics e = d455_left_to_color();
  auto m = realsense::transform_from(e, "depth-to-colour", kWho);
  CHECK(m.ok());
  double worst = 0.0;
  for (const auto& p : {std::vector<float>{0.3f, -0.2f, 1.5f},
                        std::vector<float>{-1.0f, 0.5f, 4.0f},
                        std::vector<float>{0.0f, 0.0f, 0.4f}}) {
    float to[3];
    rs2_transform_point_to_point(to, &e, p.data());
    const glm::dvec4 moved = m.value() * glm::dvec4(p[0], p[1], p[2], 1.0);
    for (int k = 0; k < 3; ++k)
      worst = std::fmax(worst, std::fabs(moved[k] - to[k]));
  }
  std::printf("  extrinsic: %.2e m from rsutil.h at worst\n", worst);
  CHECK(worst < 1e-6);
  rs2_extrinsics scaled = e;
  for (float& v : scaled.rotation) v *= 1.5f;
  CHECK(unsupported(
      realsense::transform_from(scaled, "depth-to-colour", kWho).status(),
      "the factory depth-to-colour extrinsic is not rigid"));
  return 0;
}

// The infrared cameras as the stereo tier takes them: points imaged through
// them land where librealsense's own rsutil.h puts them, in every camera.
int test_stereo_info() {
  const rs2_intrinsics ir = d455_infrared();
  const rs2_intrinsics color = d455_color();
  const rs2_extrinsics l2r = d455_left_to_right();
  const rs2_extrinsics l2c = d455_left_to_color();
  auto made = realsense::stereo_info(ir, ir, l2r, color, l2c, kWho);
  CHECK(made.ok());
  const sensor::StereoInfo& info = made.value();
  const sensor::StereoCalibration& cal = info.calibration;
  for (const camera::CameraModel* c : {&cal.left, &cal.right}) {
    CHECK(c->size.width == 1280 && c->size.height == 720);
    CHECK(c->intrinsics.fx == double{ir.fx} && c->intrinsics.cx == double{ir.ppx});
    CHECK(c->distortion.k1 == 0.0 && c->distortion.p2 == 0.0);
  }
  CHECK(info.color.has_value());
  CHECK(info.color->size.width == 1280 && info.color->size.height == 800);
  CHECK(info.color->distortion.k1 == double{color.coeffs[0]});
  const double baseline = -cal.left_to_right[3][0];
  CHECK(std::fabs(baseline - 0.0950313) < 1e-7);

  double worst_right = 0.0, worst_row = 0.0, worst_disparity = 0.0,
         worst_color = 0.0;
  int seen_color = 0;
  for (const float z : {0.5f, 2.0f, 5.0f}) {
    for (int j = 0; j <= 8; ++j) {
      for (int i = 0; i <= 12; ++i) {
        const float p[3] = {z * (-0.9f + 1.8f * static_cast<float>(i) / 12.0f),
                            z * (-0.5f + 1.0f * static_cast<float>(j) / 8.0f),
                            z};
        float in_right[3], right_px[2], in_color[3], color_px[2];
        rs2_transform_point_to_point(in_right, &l2r, p);
        rs2_project_point_to_pixel(right_px, &ir, in_right);
        rs2_transform_point_to_point(in_color, &l2c, p);
        rs2_project_point_to_pixel(color_px, &color, in_color);

        const glm::dvec3 point(p[0], p[1], p[2]);
        const glm::dvec2 left = project(cal.left, point);
        const glm::dvec2 right = project(
            cal.right, glm::dvec3(cal.left_to_right * glm::dvec4(point, 1.0)));
        worst_right = std::fmax(worst_right, std::hypot(right.x - right_px[0],
                                                        right.y - right_px[1]));
        worst_row = std::fmax(worst_row, std::fabs(left.y - right.y));
        worst_disparity = std::fmax(
            worst_disparity, std::fabs((left.x - right.x) -
                                       cal.left.intrinsics.fx * baseline / z));
        if (!inside(color_px, color)) continue;
        ++seen_color;
        const glm::dvec2 c = project(
            *info.color, glm::dvec3(info.left_to_color * glm::dvec4(point, 1.0)));
        worst_color = std::fmax(
            worst_color, std::hypot(c.x - color_px[0], c.y - color_px[1]));
      }
    }
  }
  std::printf(
      "  infrared cameras: right %.5f px from rsutil.h at worst, rows "
      "%.1e px apart, disparity %.1e px from fx*b/z; colour camera %.4f px "
      "(%d points)\n",
      worst_right, worst_row, worst_disparity, worst_color, seen_color);
  CHECK(worst_right < 0.002);
  CHECK(worst_row < 1e-9);
  CHECK(worst_disparity < 1e-9);
  CHECK(seen_color > 200 && worst_color < 0.05);

  // A lens on either infrared camera, the left-to-right or the colour
  // extrinsic not rigid, a colour lens the pass does not undistort: each
  // named.
  rs2_intrinsics lensed = ir;
  lensed.coeffs[0] = 0.01f;
  CHECK(unsupported(
      realsense::stereo_info(lensed, ir, l2r, color, l2c, kWho).status(),
      "the left infrared stream reports a lens"));
  CHECK(unsupported(
      realsense::stereo_info(ir, lensed, l2r, color, l2c, kWho).status(),
      "the right infrared stream reports a lens"));
  rs2_extrinsics scaled = l2r;
  scaled.rotation[0] = 1.5f;
  CHECK(unsupported(
      realsense::stereo_info(ir, ir, scaled, color, l2c, kWho).status(),
      "left-to-right infrared extrinsic is not rigid"));
  scaled = l2c;
  scaled.rotation[4] = 0.5f;
  CHECK(unsupported(
      realsense::stereo_info(ir, ir, l2r, color, scaled, kWho).status(),
      "infrared-to-colour extrinsic is not rigid"));
  rs2_intrinsics fisheye = color;
  fisheye.model = RS2_DISTORTION_FTHETA;
  CHECK(unsupported(
      realsense::stereo_info(ir, ir, l2r, fisheye, l2c, kWho).status(),
      rs2_distortion_to_string(RS2_DISTORTION_FTHETA)));
  return 0;
}

int test_moment() {
  CHECK(realsense::same_moment(1000.0, 1000.04, 30));
  CHECK(realsense::same_moment(1000.0, 1016.6, 30));
  CHECK(!realsense::same_moment(1000.0, 1016.7, 30));  // half of 33.3 ms
  CHECK(
      !realsense::same_moment(2287.7, 1000.0, 30));  // the syncer's stale pair
  CHECK(realsense::same_moment(1000.0, 1008.0, 60));
  CHECK(!realsense::same_moment(1000.0, 1009.0, 60));
  CHECK(!realsense::same_moment(1000.0, 1000.0, 0));

  // A frameset: a frame of each mode opened, each within half a frame of
  // the first.
  const double stamps[] = {2287.7, 2287.7, 2290.1};
  CHECK(realsense::together(stamps, 3, 3, 30));
  CHECK(realsense::together(stamps, 2, 2, 30));
  CHECK(!realsense::together(stamps, 2, 3, 30));  // a mode missing
  CHECK(!realsense::together(stamps, 3, 2, 30));  // one more than opened
  CHECK(!realsense::together(stamps, 0, 0, 30));
  const double stale[] = {2287.7, 2287.7, 1000.0};  // stale colour
  CHECK(!realsense::together(stale, 3, 3, 30));
  // Each against the first, not the one before: 16 ms steps add up.
  const double drifting[] = {1000.0, 1016.0, 1032.0};
  CHECK(!realsense::together(drifting, 3, 3, 30));
  CHECK(!realsense::together(stamps, 3, 3, 0));
  return 0;
}

int test_frame() {
  realsense::FrameSetup setup;
  setup.depth_camera.size = {4, 2};
  setup.depth_camera.intrinsics = {4.0, 4.0, 2.0, 1.0};
  setup.color_camera.size = {6, 2};
  setup.color_camera.intrinsics = {6.0, 6.0, 3.0, 1.0};
  setup.depth_to_color[3][0] = -0.059;
  setup.color_to_world[3][2] = 2.0;
  setup.metres_per_unit = 0.001f;
  setup.min_depth = 0.3f;
  setup.max_depth = 6.0f;
  std::vector<std::uint16_t> depth(8, 1000);
  std::vector<std::uint8_t> color(2 * 16, 128);  // rows padded to 16 bytes
  realsense::FrameView view;
  view.depth = {depth.data(), 4, 2, 8};
  view.color = {color.data(), 6, 2, 16};
  view.timestamp_ms = 1793308892747.3442;
  view.sequence = 42;
  view.arrival_ns = 1793308892762000000ull;
  auto held = std::make_shared<int>(7);
  auto frame = realsense::rgbd_frame(view, setup, held, kWho);
  CHECK(frame.ok());
  const sensor::RgbdFrame& f = frame.value();
  CHECK(f.depth == depth.data() && f.metres_per_unit == 0.001f);
  CHECK(f.min_depth == 0.3f && f.max_depth == 6.0f);
  CHECK(f.depth_camera.size.width == 4 && f.color_camera.size.width == 6);
  CHECK(f.color.layout == sensor::YuvLayout::Yuyv);
  CHECK(f.color.host == color.data() && f.color.offset[0] == 0 &&
        f.color.stride[0] == 16);
  CHECK(f.color.device == nullptr && f.color.image[0] == nullptr);
  CHECK(f.color.width == 6 && f.color.height == 2);
  CHECK(f.color.kr == 0.299f && f.color.kb == 0.114f && !f.color.full_range);
  CHECK(f.color.chroma_location == sensor::ChromaLocation::Left);
  CHECK(f.has_color());
  CHECK(f.depth_to_color == setup.depth_to_color &&
        f.color_to_world == setup.color_to_world);
  // Within a microsecond: a double holds this time to 256 ns.
  CHECK(f.timestamp_ns > 1793308892747343200ull &&
        f.timestamp_ns < 1793308892747345200ull);
  CHECK(f.sequence == 42 && f.arrival_ns == view.arrival_ns);
  CHECK(f.pixels.get() == held.get());

  const auto contradicts = [&](realsense::FrameView v) {
    const vkc::Status s = realsense::rgbd_frame(v, setup, held, kWho).status();
    return s.domain() == vkc::Status::Code::IoError &&
           s.message().find("contradicts its streams") != std::string::npos;
  };
  realsense::FrameView bad = view;
  bad.depth.width = 5;
  CHECK(contradicts(bad));
  bad = view;
  bad.depth.stride = 10;  // padded depth rows: the frame takes them packed
  CHECK(contradicts(bad));
  bad = view;
  bad.color.stride = 11;
  CHECK(contradicts(bad));
  bad = view;
  bad.color.height = 3;
  CHECK(contradicts(bad));
  bad = view;
  bad.color.data = nullptr;
  CHECK(contradicts(bad));
  return 0;
}

int test_stereo_frame() {
  sensor::StereoInfo info;
  info.calibration.left.size = {4, 2};
  info.calibration.left.intrinsics = {4.0, 4.0, 2.0, 1.0};
  info.calibration.right = info.calibration.left;
  info.calibration.left_to_right[3][0] = -0.095;
  camera::CameraModel color_camera;
  color_camera.size = {6, 2};
  color_camera.intrinsics = {6.0, 6.0, 3.0, 1.0};
  info.color = color_camera;
  std::vector<std::uint8_t> left(2 * 8, 10);   // rows padded to 8 bytes
  std::vector<std::uint8_t> right(2 * 4, 20);  // rows packed
  std::vector<std::uint8_t> color(2 * 16, 128);
  realsense::StereoView view;
  view.left = {left.data(), 4, 2, 8};
  view.right = {right.data(), 4, 2, 4};
  view.color = {color.data(), 6, 2, 16};
  view.timestamp_ms = 1793308892747.3442;
  view.sequence = 42;
  view.arrival_ns = 1793308892762000000ull;
  camera::Mat4d to_world(1.0);
  to_world[3][2] = 2.0;
  auto held = std::make_shared<int>(7);
  auto made = realsense::stereo_frame(view, info, to_world, held, kWho);
  CHECK(made.ok());
  const sensor::StereoFrame& f = made.value();
  const struct {
    const sensor::YuvImage* image;
    const std::uint8_t* data;
    std::size_t stride;
  } views[] = {{&f.left, left.data(), 8}, {&f.right, right.data(), 4}};
  for (const auto& v : views) {
    const sensor::YuvImage& p = *v.image;
    CHECK(p.layout == sensor::YuvLayout::Luma);
    CHECK(p.host == v.data && p.offset[0] == 0 && p.stride[0] == v.stride);
    CHECK(p.device == nullptr && p.image[0] == nullptr);
    CHECK(p.width == 4 && p.height == 2);
    CHECK(p.full_range);  // the camera's 8-bit infrared, as it is
  }
  CHECK(f.color.layout == sensor::YuvLayout::Yuyv);
  CHECK(f.color.host == color.data() && f.color.stride[0] == 16);
  CHECK(f.color.width == 6 && f.color.height == 2);
  CHECK(f.color.kr == 0.299f && f.color.kb == 0.114f && !f.color.full_range);
  CHECK(f.color.chroma_location == sensor::ChromaLocation::Left);
  CHECK(f.color_to_world == to_world);
  CHECK(f.timestamp_ns > 1793308892747343200ull &&
        f.timestamp_ns < 1793308892747345200ull);
  CHECK(f.sequence == 42 && f.arrival_ns == view.arrival_ns);
  CHECK(f.pixels.get() == held.get());

  const auto contradicts = [&](realsense::StereoView v,
                               const sensor::StereoInfo& i) {
    const vkc::Status s =
        realsense::stereo_frame(v, i, to_world, held, kWho).status();
    return s.domain() == vkc::Status::Code::IoError &&
           s.message().find("contradicts its streams") != std::string::npos;
  };
  realsense::StereoView bad = view;
  bad.left.width = 5;
  CHECK(contradicts(bad, info));
  bad = view;
  bad.right.height = 3;
  CHECK(contradicts(bad, info));
  bad = view;
  bad.left.stride = 3;  // shorter than a row
  CHECK(contradicts(bad, info));
  bad = view;
  bad.right.data = nullptr;
  CHECK(contradicts(bad, info));
  bad = view;
  bad.color.stride = 11;
  CHECK(contradicts(bad, info));
  bad = view;
  bad.color.width = 4;
  CHECK(contradicts(bad, info));
  sensor::StereoInfo no_color = info;
  no_color.color.reset();
  CHECK(contradicts(view, no_color));
  return 0;
}

int test_options() {
  using Options = sensor::RealSenseSensor::Options;
  CHECK(realsense::check_options(Options{}).ok());
  Options o;
  o.depth_height = 0;
  CHECK(realsense::check_options(o).domain() ==
        vkc::Status::Code::InvalidArgument);
  o = Options{};
  o.max_depth = std::numeric_limits<float>::quiet_NaN();
  CHECK(realsense::check_options(o).domain() ==
        vkc::Status::Code::InvalidArgument);
  o = Options{};
  o.color_controls = camera::ColorControls{0, 64};
  CHECK(realsense::check_options(o).domain() ==
        vkc::Status::Code::InvalidArgument);
  o.color_controls = camera::ColorControls{166, 64};
  CHECK(realsense::check_options(o).ok());

  // The stereo camera's: no depth range; the projector off unless asked.
  using StereoOptions = sensor::RealSenseStereo::Options;
  CHECK(realsense::check_options(StereoOptions{}).ok());
  CHECK(!StereoOptions{}.emitter);
  StereoOptions s;
  s.infrared_width = 0;
  CHECK(realsense::check_options(s).domain() ==
        vkc::Status::Code::InvalidArgument);
  s = StereoOptions{};
  s.color_height = 0;
  CHECK(realsense::check_options(s).domain() ==
        vkc::Status::Code::InvalidArgument);
  s = StereoOptions{};
  s.fps = 0;
  CHECK(realsense::check_options(s).domain() ==
        vkc::Status::Code::InvalidArgument);
  s = StereoOptions{};
  s.color_controls = camera::ColorControls{166, 0};
  CHECK(realsense::check_options(s).domain() ==
        vkc::Status::Code::InvalidArgument);
  s = StereoOptions{};
  s.color_to_world[1][1] = 2.0;
  CHECK(realsense::check_options(s).domain() ==
        vkc::Status::Code::InvalidArgument);
  s.color_to_world = camera::Mat4d(1.0);
  s.color_to_world[3][0] = 1.5;  // a translation is rigid
  CHECK(realsense::check_options(s).ok());
  return 0;
}

}  // namespace

int main() {
  if (test_lenses() != 0) return 1;
  if (test_extrinsic() != 0) return 1;
  if (test_stereo_info() != 0) return 1;
  if (test_moment() != 0) return 1;
  if (test_frame() != 0) return 1;
  if (test_stereo_frame() != 0) return 1;
  if (test_options() != 0) return 1;
  std::puts("sensor_realsense_conversion: OK");
  return 0;
}
