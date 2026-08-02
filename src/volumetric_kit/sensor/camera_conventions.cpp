// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/camera_conventions.hpp"

#include <cmath>

namespace volumetric_kit::recon::sensor {
namespace {

// Rescale one axis of a pinhole camera for a resized image.
//
// Focal length scales with the ratio. The principal point carries a half-pixel
// term because pixel centres sit at integer coordinates here, so a W-wide image
// spans [-0.5, W-0.5] rather than [0, W]: matching those spans gives
// (c + 0.5)/W == (c' + 0.5)/W', hence c' = (c + 0.5)*s - 0.5. Dropping it
// biases every unprojected ray by 0.5*(1 - s) px -- and would stop a centred
// principal point mapping to the centre.
void rescale_axis(float scale, float& focal, float& principal) {
  focal *= scale;
  principal = (principal + 0.5f) * scale - 0.5f;
}

}  // namespace

Mat4f cv_from_gl_camera(const Mat4f& cam_to_world) {
  // Right-multiply by diag(1, -1, -1, 1). Written as a column negation because
  // that is all the product does: the second and third basis vectors flip, the
  // first basis vector and the translation are untouched. Doing it as a matrix
  // multiply would be identical but invites writing the operands the other way
  // round, which silently moves the camera instead of turning it.
  Mat4f out = cam_to_world;
  out[1] = -out[1];
  out[2] = -out[2];
  return out;
}

Result<volume::DepthCameraParams> depth_from_registered_color(
    const tsdf::ColorCameraParams& color, std::uint32_t depth_width,
    std::uint32_t depth_height, float min_depth, float max_depth) {
  if (color.width == 0 || color.height == 0) {
    return Status::invalid_argument(
        "depth_from_registered_color: color camera has a zero image size");
  }
  if (depth_width == 0 || depth_height == 0) {
    return Status::invalid_argument(
        "depth_from_registered_color: depth image size must be non-zero");
  }
  // A negative near plane would unproject points behind the camera; an
  // unordered range rejects every sample, which would look like a dead sensor
  // rather than a bad argument.
  if (!(min_depth >= 0.0f) || !(min_depth < max_depth)) {
    return Status::invalid_argument(
        "depth_from_registered_color: require 0 <= min_depth < max_depth");
  }
  // The unprojection this camera feeds divides by the focal length
  // (x = (u - cx)*d/fx), so a zero or non-finite focal makes every ray inf/NaN
  // -- fusion then reads garbage block coordinates rather than reporting
  // anything. Reject it here, where the argument still has a name. The
  // comparisons are written to reject NaN (every NaN compare is false).
  if (!(color.fx > 0.0f) || !(color.fy > 0.0f) || !std::isfinite(color.fx) ||
      !std::isfinite(color.fy)) {
    return Status::invalid_argument(
        "depth_from_registered_color: color focal lengths must be finite and "
        "positive");
  }

  volume::DepthCameraParams depth{};
  depth.fx = color.fx;
  depth.fy = color.fy;
  depth.cx = color.cx;
  depth.cy = color.cy;
  rescale_axis(
      static_cast<float>(depth_width) / static_cast<float>(color.width),
      depth.fx, depth.cx);
  rescale_axis(
      static_cast<float>(depth_height) / static_cast<float>(color.height),
      depth.fy, depth.cy);
  depth.min_depth = min_depth;
  depth.max_depth = max_depth;
  depth.width = depth_width;
  depth.height = depth_height;
  // Registered: one physical camera, so the depth frame shares the colour
  // pose exactly. Deriving it here rather than letting a caller set both is the
  // point -- two independently-assigned poses are free to drift apart.
  depth.cam_to_world = color.cam_to_world;
  return depth;
}

}  // namespace volumetric_kit::recon::sensor
