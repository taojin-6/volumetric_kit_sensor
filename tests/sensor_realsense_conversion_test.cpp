// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The RealSense driver's arithmetic (realsense/conversion.hpp), with no camera,
// against librealsense's own rsutil.h:
//   - the D455's colour lens (inverse Brown-Conrady) projects as
//     rs2_project_point_to_pixel does within 0.05 px over its picture, and
//     plain Brown-Conrady as it does to float rounding;
//   - the depth stream is a pinhole, and refused if it reports a lens; other
//     lens models with coefficients are refused, named; bad intrinsics too;
//   - the extrinsic moves points as rs2_transform_point_to_point does, and a
//     non-rigid one is refused;
//   - depth and colour are one moment within half a frame;
//   - a frame points into the view, YUYV on the host coded as librealsense
//     converts it, stamped in nanoseconds; a view that contradicts its
//     streams is refused;
//   - the options open refuses before it looks for a camera.

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

// The worst distance, in pixels, between where rsutil.h projects a grid of
// rays and where the camera model does, over those landing in the picture.
double worst_projection(const rs2_intrinsics& in,
                        const camera::CameraModel& model) {
  const double d[8] = {model.distortion.k1, model.distortion.k2,
                       model.distortion.p1, model.distortion.p2,
                       model.distortion.k3, model.distortion.k4,
                       model.distortion.k5, model.distortion.k6};
  double worst = 0.0;
  int inside = 0;
  for (int j = 0; j <= 40; ++j) {
    for (int i = 0; i <= 60; ++i) {
      const float point[3] = {-1.1f + 2.2f * static_cast<float>(i) / 60.0f,
                              -0.7f + 1.4f * static_cast<float>(j) / 40.0f,
                              1.0f};
      float pixel[2];
      rs2_project_point_to_pixel(pixel, &in, point);
      if (!(pixel[0] >= 0.0f && pixel[1] >= 0.0f && pixel[0] < in.width &&
            pixel[1] < in.height)) {
        continue;
      }
      ++inside;
      double xd = 0.0, yd = 0.0;
      camera::distort_rational(d, double{point[0]}, double{point[1]}, &xd, &yd);
      const double u = xd * model.intrinsics.fx + model.intrinsics.cx;
      const double v = yd * model.intrinsics.fy + model.intrinsics.cy;
      worst = std::fmax(worst, std::hypot(u - pixel[0], v - pixel[1]));
    }
  }
  return inside > 1000 ? worst : std::numeric_limits<double>::infinity();
}

bool unsupported(const vkc::Status& s, const char* why) {
  const bool ok = s.domain() == vkc::Status::Code::Unsupported &&
                  s.message().find(why) != std::string::npos;
  if (!ok) std::fprintf(stderr, "unexpected: %s\n", s.message().c_str());
  return ok;
}

int test_lenses() {
  const rs2_intrinsics color = d455_color();
  auto model = realsense::camera_model_from(color, true, kWho);
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
  auto plain_model = realsense::camera_model_from(plain, true, kWho);
  CHECK(plain_model.ok());
  const double exact = worst_projection(plain, plain_model.value());
  std::printf(
      "  lenses: inverse Brown-Conrady %.4f px from rsutil.h at worst, "
      "Brown-Conrady %.5f px\n",
      inverse, exact);
  CHECK(inverse < 0.05);
  CHECK(exact < 0.002);

  // The depth stream: rectified, a pinhole, whatever model it names.
  rs2_intrinsics depth{};
  depth.width = 1280;
  depth.height = 720;
  depth.ppx = 651.384f;
  depth.ppy = 359.031f;
  depth.fx = depth.fy = 657.271f;
  depth.model = RS2_DISTORTION_BROWN_CONRADY;
  auto pinhole = realsense::camera_model_from(depth, false, kWho);
  CHECK(pinhole.ok());
  CHECK(pinhole->distortion.k1 == 0.0 && pinhole->distortion.p1 == 0.0);
  depth.coeffs[3] = 1e-4f;
  CHECK(unsupported(realsense::camera_model_from(depth, false, kWho).status(),
                    "the depth stream reports a lens"));

  // Lens models the pass does not undistort, with coefficients; without
  // any, a pinhole.
  for (const rs2_distortion other :
       {RS2_DISTORTION_MODIFIED_BROWN_CONRADY, RS2_DISTORTION_FTHETA,
        RS2_DISTORTION_KANNALA_BRANDT4}) {
    rs2_intrinsics in = color;
    in.model = other;
    CHECK(unsupported(realsense::camera_model_from(in, true, kWho).status(),
                      rs2_distortion_to_string(other)));
    for (float& c : in.coeffs) c = 0.0f;
    CHECK(realsense::camera_model_from(in, true, kWho).ok());
  }

  // Intrinsics that are no camera.
  rs2_intrinsics bad = color;
  bad.width = 0;
  CHECK(unsupported(realsense::camera_model_from(bad, true, kWho).status(),
                    "not finite and positive"));
  bad = color;
  bad.fx = std::numeric_limits<float>::quiet_NaN();
  CHECK(unsupported(realsense::camera_model_from(bad, true, kWho).status(),
                    "not finite and positive"));
  bad = color;
  bad.fy = -1.0f;
  CHECK(unsupported(realsense::camera_model_from(bad, true, kWho).status(),
                    "not finite and positive"));
  return 0;
}

int test_extrinsic() {
  // The lab D455's depth-to-colour extrinsic.
  rs2_extrinsics e{};
  const float r[9] = {0.999996f,    0.00231108f, -0.00170157f,
                      -0.00231856f, 0.999988f,   -0.00441104f,
                      0.00169135f,  0.00441496f, 0.999989f};
  for (int i = 0; i < 9; ++i) e.rotation[i] = r[i];
  e.translation[0] = -0.0590472f;
  e.translation[1] = -7.2951e-05f;
  e.translation[2] = 0.000360092f;
  auto m = realsense::transform_from(e, kWho);
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
  CHECK(unsupported(realsense::transform_from(scaled, kWho).status(),
                    "not rigid"));
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
  view.depth = depth.data();
  view.depth_width = 4;
  view.depth_height = 2;
  view.depth_stride = 8;
  view.color = color.data();
  view.color_width = 6;
  view.color_height = 2;
  view.color_stride = 16;
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
  bad.depth_width = 5;
  CHECK(contradicts(bad));
  bad = view;
  bad.depth_stride = 10;  // padded depth rows: the frame takes them packed
  CHECK(contradicts(bad));
  bad = view;
  bad.color_stride = 11;
  CHECK(contradicts(bad));
  bad = view;
  bad.color_height = 3;
  CHECK(contradicts(bad));
  bad = view;
  bad.color = nullptr;
  CHECK(contradicts(bad));
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
  return 0;
}

}  // namespace

int main() {
  if (test_lenses() != 0) return 1;
  if (test_extrinsic() != 0) return 1;
  if (test_moment() != 0) return 1;
  if (test_frame() != 0) return 1;
  if (test_options() != 0) return 1;
  std::puts("sensor_realsense_conversion: OK");
  return 0;
}
