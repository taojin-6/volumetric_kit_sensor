// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "conversion.hpp"

#include <cmath>
#include <optional>
#include <utility>

namespace volumetric_kit::sensor::realsense {

namespace {

bool positive(float v) noexcept { return std::isfinite(v) && v > 0.0f; }

// What both drivers refuse of the colour camera's options.
core::Status check_color(const char* kind,
                         const std::optional<camera::ColorControls>& controls,
                         const camera::Mat4d& color_to_world) {
  const auto refuse = [kind](const std::string& why) {
    return core::Status::invalid_argument(std::string(kind) + ": " + why);
  };
  if (controls && (controls->exposure == 0 || controls->gain == 0)) {
    return refuse("color_controls needs an exposure and a gain above 0");
  }
  const core::Status rigid = camera::check_rigid(color_to_world);
  if (!rigid.ok()) return refuse("color_to_world: " + rigid.message());
  return {};
}

// A YUYV picture on the host as the camera sends it: BT.601, limited range,
// chroma sited left, what librealsense's own YUY2 conversion assumes.
YuvImage yuyv_on_host(const PictureView& view) {
  YuvImage image;
  image.layout = YuvLayout::Yuyv;
  image.host = static_cast<const std::uint8_t*>(view.data);
  image.stride[0] = view.stride;
  image.width = view.width;
  image.height = view.height;
  const YcbcrWeights bt601 = ycbcr_weights(VideoColorMatrix::Bt601);
  image.kr = bt601.kr;
  image.kb = bt601.kb;
  image.full_range = false;
  image.chroma_location = ChromaLocation::Left;
  return image;
}

// An 8-bit infrared picture on the host, as luma: grey, full range.
YuvImage luma_on_host(const PictureView& view) {
  YuvImage image;
  image.layout = YuvLayout::Luma;
  image.host = static_cast<const std::uint8_t*>(view.data);
  image.stride[0] = view.stride;
  image.width = view.width;
  image.height = view.height;
  image.full_range = true;
  return image;
}

// librealsense's global time in milliseconds, in nanoseconds; 0 when unset.
std::uint64_t nanoseconds(double ms) noexcept {
  return ms > 0.0 ? static_cast<std::uint64_t>(std::llround(ms * 1e6)) : 0;
}

// Whether a picture is there at `size`, with rows of at least `bytes` a
// pixel.
bool fits(const PictureView& view, const camera::ImageSize& size,
          std::size_t bytes) noexcept {
  return view.data != nullptr && view.width == size.width &&
         view.height == size.height &&
         view.stride >= bytes * std::size_t{size.width};
}

std::string describe(const char* name, const PictureView& view) {
  return std::string(name) + " " + std::to_string(view.width) + "x" +
         std::to_string(view.height) + " (" + std::to_string(view.stride) +
         " bytes a row)";
}

}  // namespace

core::Status check_options(const RealSenseSensor::Options& options) {
  const auto refuse = [](const std::string& why) {
    return core::Status::invalid_argument("RealSenseSensor: " + why);
  };
  if (options.depth_width == 0 || options.depth_height == 0 ||
      options.color_width == 0 || options.color_height == 0) {
    return refuse("a stream's width or height is 0");
  }
  if (options.fps == 0) return refuse("the frame rate is 0");
  if (!(std::isfinite(options.min_depth) && std::isfinite(options.max_depth) &&
        options.min_depth > 0.0f && options.min_depth < options.max_depth)) {
    return refuse("the depth range [" + std::to_string(options.min_depth) +
                  ", " + std::to_string(options.max_depth) +
                  "] m must be finite, with 0 < min_depth < max_depth");
  }
  return check_color("RealSenseSensor", options.color_controls,
                     options.color_to_world);
}

core::Status check_options(const RealSenseStereo::Options& options) {
  const auto refuse = [](const std::string& why) {
    return core::Status::invalid_argument("RealSenseStereo: " + why);
  };
  if (options.infrared_width == 0 || options.infrared_height == 0 ||
      options.color_width == 0 || options.color_height == 0) {
    return refuse("a stream's width or height is 0");
  }
  if (options.fps == 0) return refuse("the frame rate is 0");
  return check_color("RealSenseStereo", options.color_controls,
                     options.color_to_world);
}

core::Result<camera::CameraModel> camera_model_from(const rs2_intrinsics& in,
                                                    const char* stream,
                                                    bool lens,
                                                    const std::string& who) {
  if (in.width <= 0 || in.height <= 0 || !positive(in.fx) || !positive(in.fy) ||
      !std::isfinite(in.ppx) || !std::isfinite(in.ppy)) {
    return core::Status::unsupported(
        who + ": the " + stream +
        " stream's factory intrinsics are not a camera the GPU pass can use "
        "(a size, focal length or principal point not finite and positive)");
  }
  bool any = false;
  for (const float c : in.coeffs) any = any || c != 0.0f;
  if (any && !lens) {
    return core::Status::unsupported(
        who + ": the " + stream + " stream reports a lens (" +
        rs2_distortion_to_string(in.model) +
        "), and it is taken as rectified on the camera");
  }
  if (any && in.model != RS2_DISTORTION_BROWN_CONRADY &&
      in.model != RS2_DISTORTION_INVERSE_BROWN_CONRADY) {
    return core::Status::unsupported(
        who + ": the " + stream + " stream's lens model " +
        rs2_distortion_to_string(in.model) +
        " is not one the GPU pass undistorts (Brown-Conrady)");
  }
  camera::CameraModel out;
  out.size = {static_cast<std::uint32_t>(in.width),
              static_cast<std::uint32_t>(in.height)};
  out.intrinsics = {in.fx, in.fy, in.ppx, in.ppy};
  if (any) {
    out.distortion.k1 = in.coeffs[0];
    out.distortion.k2 = in.coeffs[1];
    out.distortion.p1 = in.coeffs[2];
    out.distortion.p2 = in.coeffs[3];
    out.distortion.k3 = in.coeffs[4];
  }
  const core::Status valid = camera::check_camera_model(out);
  if (!valid.ok()) {
    return core::Status::unsupported(who + ": the " + stream + " camera " +
                                     valid.message());
  }
  return out;
}

core::Result<camera::Mat4d> transform_from(const rs2_extrinsics& extrinsics,
                                           const char* what,
                                           const std::string& who) {
  camera::Mat4d m(1.0);
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) m[c][r] = extrinsics.rotation[3 * c + r];
    m[3][c] = extrinsics.translation[c];
  }
  const core::Status rigid = camera::check_rigid(m);
  if (!rigid.ok()) {
    return core::Status::unsupported(
        who + ": the factory " + what +
        " extrinsic is not rigid: " + rigid.message());
  }
  return m;
}

core::Result<StereoInfo> stereo_info(const rs2_intrinsics& left,
                                     const rs2_intrinsics& right,
                                     const rs2_extrinsics& left_to_right,
                                     const rs2_intrinsics& color,
                                     const rs2_extrinsics& left_to_color,
                                     const std::string& who) {
  StereoInfo info;
  StereoCalibration& cameras = info.calibration;
  VKC_ASSIGN(cameras.left,
             camera_model_from(left, "left infrared", false, who));
  VKC_ASSIGN(cameras.right,
             camera_model_from(right, "right infrared", false, who));
  VKC_ASSIGN(cameras.left_to_right,
             transform_from(left_to_right, "left-to-right infrared", who));
  VKC_ASSIGN(info.color, camera_model_from(color, "colour", true, who));
  VKC_ASSIGN(info.left_to_color,
             transform_from(left_to_color, "infrared-to-colour", who));
  return info;
}

DepthNoise chip_depth_noise(const camera::CameraModel& depth_camera,
                            const rs2_extrinsics& depth_to_right) noexcept {
  const float* t = depth_to_right.translation;
  const double baseline = std::sqrt(double{t[0]} * t[0] + double{t[1]} * t[1] +
                                    double{t[2]} * t[2]);
  return DepthNoise::stereo(kChipDisparitySigma, depth_camera.intrinsics.fx,
                            baseline);
}

bool same_moment(double a_ms, double b_ms, std::uint32_t fps) noexcept {
  return fps > 0 && std::fabs(a_ms - b_ms) <= 500.0 / fps;
}

bool together(const double* stamps_ms, std::size_t count, std::size_t opened,
              std::uint32_t fps) noexcept {
  if (count != opened || count == 0) return false;
  for (std::size_t i = 1; i < count; ++i) {
    if (!same_moment(stamps_ms[0], stamps_ms[i], fps)) return false;
  }
  return true;
}

core::Result<RgbdFrame> rgbd_frame(const FrameView& view,
                                   const FrameSetup& setup,
                                   std::shared_ptr<const void> pixels,
                                   const std::string& who) {
  const camera::ImageSize& d = setup.depth_camera.size;
  if (!fits(view.depth, d, 2) ||
      view.depth.stride != 2 * std::size_t{d.width} ||
      !fits(view.color, setup.color_camera.size, 2)) {
    return core::Status::io_error(
        who + ": a frameset contradicts its streams: " +
        describe("depth", view.depth) + ", " + describe("colour", view.color));
  }
  RgbdFrame frame;
  frame.depth = static_cast<const std::uint16_t*>(view.depth.data);
  frame.metres_per_unit = setup.metres_per_unit;
  frame.depth_camera = setup.depth_camera;
  frame.min_depth = setup.min_depth;
  frame.max_depth = setup.max_depth;
  frame.depth_noise = setup.depth_noise;
  frame.color = yuyv_on_host(view.color);
  frame.color_camera = setup.color_camera;
  frame.color_to_world = setup.color_to_world;
  frame.depth_to_color = setup.depth_to_color;
  frame.timestamp_ns = nanoseconds(view.timestamp_ms);
  frame.sequence = view.sequence;
  frame.arrival_ns = view.arrival_ns;
  frame.pixels = std::move(pixels);
  return frame;
}

core::Result<StereoFrame> stereo_frame(const StereoView& view,
                                       const StereoInfo& info,
                                       const camera::Mat4d& color_to_world,
                                       std::shared_ptr<const void> pixels,
                                       const std::string& who) {
  if (!info.color || !fits(view.left, info.calibration.left.size, 1) ||
      !fits(view.right, info.calibration.right.size, 1) ||
      !fits(view.color, info.color->size, 2)) {
    return core::Status::io_error(who +
                                  ": a frameset contradicts its streams: " +
                                  describe("left infrared", view.left) + ", " +
                                  describe("right infrared", view.right) +
                                  ", " + describe("colour", view.color));
  }
  StereoFrame frame;
  frame.left = luma_on_host(view.left);
  frame.right = luma_on_host(view.right);
  frame.color = yuyv_on_host(view.color);
  frame.color_to_world = color_to_world;
  frame.timestamp_ns = nanoseconds(view.timestamp_ms);
  frame.sequence = view.sequence;
  frame.arrival_ns = view.arrival_ns;
  frame.pixels = std::move(pixels);
  return frame;
}

}  // namespace volumetric_kit::sensor::realsense
