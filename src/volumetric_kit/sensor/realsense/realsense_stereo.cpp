// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/realsense/realsense_stereo.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <librealsense2/rs.hpp>

#include "camera.hpp"
#include "conversion.hpp"

namespace volumetric_kit::sensor {

namespace {

constexpr const char* kWho = "RealSenseStereo";

// librealsense's indices of the left and right infrared cameras.
constexpr int kLeft = 1;
constexpr int kRight = 2;

}  // namespace

struct RealSenseStereo::Impl {
  std::unique_ptr<realsense::Camera> camera;
  StereoInfo info;
  camera::Mat4d color_to_world = camera::Mat4d(1.0);

  // A frameset taken from the queue as the stereo frame it is; one it
  // refuses is counted failed.
  core::Result<std::optional<StereoFrame>> read(realsense::Arrived arrived) {
    core::Result<std::optional<StereoFrame>> frame =
        convert(std::move(arrived));
    if (frame.ok()) {
      camera->count_delivered();
    } else {
      camera->count_failed();
    }
    return frame;
  }

  core::Result<std::optional<StereoFrame>> convert(
      realsense::Arrived arrived) try {
    const rs2::video_frame left = arrived.frames.get_infrared_frame(kLeft);
    const rs2::video_frame right = arrived.frames.get_infrared_frame(kRight);
    const rs2::video_frame color = arrived.frames.get_color_frame();
    const std::string& who = camera->who();
    if (!left || !right || !color) {
      return core::Status::io_error(
          who + ": a frameset lacks an infrared or the colour picture");
    }
    if (left.get_frame_timestamp_domain() ==
        RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK) {
      return core::Status::io_error(
          who + ": a frame is stamped on the camera's clock, not the host's");
    }
    realsense::StereoView view;
    view.left = realsense::picture_of(left);
    view.right = realsense::picture_of(right);
    view.color = realsense::picture_of(color);
    view.timestamp_ms = left.get_timestamp();
    view.sequence = left.get_frame_number();
    view.arrival_ns = arrived.arrival_ns;
    auto pixels =
        std::make_shared<const rs2::frameset>(std::move(arrived.frames));
    VKC_ASSIGN(StereoFrame frame,
               realsense::stereo_frame(view, info, color_to_world,
                                       std::move(pixels), who));
    return std::optional<StereoFrame>(std::move(frame));
  } catch (const std::exception& e) {
    return realsense::sdk_failure(kWho, "reading a stereo frame", e);
  }
};

RealSenseStereo::RealSenseStereo(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
RealSenseStereo::RealSenseStereo(RealSenseStereo&& other) noexcept = default;
RealSenseStereo& RealSenseStereo::operator=(RealSenseStereo&& other) noexcept =
    default;
RealSenseStereo::~RealSenseStereo() = default;

core::Result<RealSenseStereo> RealSenseStereo::open(
    const Options& options) try {
  VKC_TRY(realsense::check_options(options));
  auto impl = std::make_unique<Impl>();
  VKC_ASSIGN(impl->camera, realsense::Camera::open(options.serial, kWho));
  realsense::Camera& camera = *impl->camera;

  // The requested modes: both infrared cameras' Y8, as the camera rectifies
  // them, and the colour camera's YUYV.
  VKC_ASSIGN(
      const rs2::video_stream_profile left,
      camera.mode(camera.stereo_module(), RS2_STREAM_INFRARED, kLeft,
                  RS2_FORMAT_Y8, options.infrared_width,
                  options.infrared_height, options.fps, "Y8 left infrared"));
  VKC_ASSIGN(
      const rs2::video_stream_profile right,
      camera.mode(camera.stereo_module(), RS2_STREAM_INFRARED, kRight,
                  RS2_FORMAT_Y8, options.infrared_width,
                  options.infrared_height, options.fps, "Y8 right infrared"));
  VKC_ASSIGN(const rs2::video_stream_profile color,
             camera.mode(camera.color_module(), RS2_STREAM_COLOR, -1,
                         RS2_FORMAT_YUYV, options.color_width,
                         options.color_height, options.fps, "YUYV colour"));

  // The factory calibration, as the stereo tier reads it.
  VKC_ASSIGN(impl->info,
             realsense::stereo_info(
                 left.get_intrinsics(), right.get_intrinsics(),
                 left.get_extrinsics_to(right), color.get_intrinsics(),
                 left.get_extrinsics_to(color), camera.who()));
  impl->color_to_world = options.color_to_world;

  // The projector and the colour exposure: last, so a camera refused above
  // gets no write; then the modes start opens.
  VKC_TRY(camera.configure(options.emitter, options.color_controls).status());
  VKC_TRY(camera.set_streams({left, right, color}, options.fps));
  return RealSenseStereo(std::move(impl));
} catch (const std::exception& e) {
  return realsense::sdk_failure(kWho, "opening the camera", e);
}

const RealSenseDeviceInfo& RealSenseStereo::device_info() const noexcept {
  static const RealSenseDeviceInfo kEmpty{};
  return impl_ != nullptr ? impl_->camera->device_info() : kEmpty;
}

const StereoInfo& RealSenseStereo::info() const noexcept {
  static const StereoInfo kEmpty{};
  return impl_ != nullptr ? impl_->info : kEmpty;
}

core::Status RealSenseStereo::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseStereo: start on a moved-from camera");
  }
  return impl_->camera->start();
}

void RealSenseStereo::stop() noexcept {
  if (impl_ != nullptr) impl_->camera->stop();
}

core::Result<std::optional<StereoFrame>> RealSenseStereo::poll() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseStereo: poll on a moved-from camera");
  }
  VKC_ASSIGN(std::optional<realsense::Arrived> arrived,
             impl_->camera->take_newest());
  if (!arrived) return std::optional<StereoFrame>{};
  return impl_->read(std::move(*arrived));
}

SensorStats RealSenseStereo::stats() const noexcept {
  return impl_ != nullptr ? impl_->camera->stats() : SensorStats{};
}

core::Status RealSenseStereo::set_frame_notify(std::function<void()> notify) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseStereo: set_frame_notify on a moved-from camera");
  }
  return impl_->camera->set_frame_notify(std::move(notify));
}

}  // namespace volumetric_kit::sensor
