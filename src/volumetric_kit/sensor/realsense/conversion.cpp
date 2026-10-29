// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "conversion.hpp"

#include <cmath>
#include <utility>

namespace volumetric_kit::sensor::realsense {

namespace {

bool positive(float v) noexcept { return std::isfinite(v) && v > 0.0f; }

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
  if (options.color_controls && (options.color_controls->exposure == 0 ||
                                 options.color_controls->gain == 0)) {
    return refuse("color_controls needs an exposure and a gain above 0");
  }
  const core::Status rigid = camera::check_rigid(options.color_to_world);
  if (!rigid.ok()) return refuse("color_to_world: " + rigid.message());
  return {};
}

core::Result<camera::CameraModel> camera_model_from(const rs2_intrinsics& in,
                                                    bool lens,
                                                    const std::string& who) {
  const std::string stream = lens ? "colour" : "depth";
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
    return core::Status::unsupported(who +
                                     ": the depth stream reports a lens (" +
                                     rs2_distortion_to_string(in.model) +
                                     "), and its depth is taken as rectified");
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
                                           const std::string& who) {
  camera::Mat4d m(1.0);
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) m[c][r] = extrinsics.rotation[3 * c + r];
    m[3][c] = extrinsics.translation[c];
  }
  const core::Status rigid = camera::check_rigid(m);
  if (!rigid.ok()) {
    return core::Status::unsupported(
        who + ": the factory depth-to-colour extrinsic is not rigid: " +
        rigid.message());
  }
  return m;
}

bool same_moment(double depth_ms, double color_ms, std::uint32_t fps) noexcept {
  return fps > 0 && std::fabs(depth_ms - color_ms) <= 500.0 / fps;
}

core::Result<RgbdFrame> rgbd_frame(const FrameView& view,
                                   const FrameSetup& setup,
                                   std::shared_ptr<const void> pixels,
                                   const std::string& who) {
  const camera::ImageSize& d = setup.depth_camera.size;
  const camera::ImageSize& c = setup.color_camera.size;
  if (view.depth == nullptr || view.color == nullptr ||
      view.depth_width != d.width || view.depth_height != d.height ||
      view.color_width != c.width || view.color_height != c.height ||
      view.depth_stride != 2 * std::size_t{d.width} ||
      view.color_stride < 2 * std::size_t{c.width}) {
    return core::Status::io_error(
        who + ": a frameset contradicts its streams: depth " +
        std::to_string(view.depth_width) + "x" +
        std::to_string(view.depth_height) + " (" +
        std::to_string(view.depth_stride) + " bytes a row), colour " +
        std::to_string(view.color_width) + "x" +
        std::to_string(view.color_height) + " (" +
        std::to_string(view.color_stride) + " bytes a row)");
  }
  RgbdFrame frame;
  frame.depth = static_cast<const std::uint16_t*>(view.depth);
  frame.metres_per_unit = setup.metres_per_unit;
  frame.depth_camera = setup.depth_camera;
  frame.min_depth = setup.min_depth;
  frame.max_depth = setup.max_depth;
  YuvImage& color = frame.color;
  color.layout = YuvLayout::Yuyv;
  color.host = static_cast<const std::uint8_t*>(view.color);
  color.stride[0] = view.color_stride;
  color.width = c.width;
  color.height = c.height;
  // What librealsense's own YUY2 conversion assumes.
  const YcbcrWeights bt601 = ycbcr_weights(VideoColorMatrix::Bt601);
  color.kr = bt601.kr;
  color.kb = bt601.kb;
  color.full_range = false;
  color.chroma_location = ChromaLocation::Left;
  frame.color_camera = setup.color_camera;
  frame.color_to_world = setup.color_to_world;
  frame.depth_to_color = setup.depth_to_color;
  frame.timestamp_ns =
      view.timestamp_ms > 0.0
          ? static_cast<std::uint64_t>(std::llround(view.timestamp_ms * 1e6))
          : 0;
  frame.sequence = view.sequence;
  frame.arrival_ns = view.arrival_ns;
  frame.pixels = std::move(pixels);
  return frame;
}

}  // namespace volumetric_kit::sensor::realsense
