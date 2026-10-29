// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <librealsense2/rs.hpp>

#include "camera.hpp"
#include "conversion.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor {

namespace {

constexpr const char* kWho = "RealSenseSensor";

}  // namespace

struct RealSenseSensor::Impl {
  std::unique_ptr<realsense::Camera> camera;
  SensorInfo info;
  realsense::FrameSetup setup;

  // A frameset taken from the queue as the frame it is; one it refuses is
  // counted failed.
  core::Result<std::optional<RgbdFrame>> read(realsense::Arrived arrived) {
    core::Result<std::optional<RgbdFrame>> frame = convert(std::move(arrived));
    if (frame.ok()) {
      camera->count_delivered();
    } else {
      camera->count_failed();
    }
    return frame;
  }

  core::Result<std::optional<RgbdFrame>> convert(
      realsense::Arrived arrived) try {
    const rs2::depth_frame depth = arrived.frames.get_depth_frame();
    const rs2::video_frame color = arrived.frames.get_color_frame();
    const std::string& who = camera->who();
    if (depth.get_frame_timestamp_domain() ==
        RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK) {
      return core::Status::io_error(
          who + ": a frame is stamped on the camera's clock, not the host's");
    }
    realsense::FrameView view;
    view.depth = realsense::picture_of(depth);
    view.color = realsense::picture_of(color);
    view.timestamp_ms = depth.get_timestamp();
    view.sequence = depth.get_frame_number();
    view.arrival_ns = arrived.arrival_ns;
    auto pixels =
        std::make_shared<const rs2::frameset>(std::move(arrived.frames));
    VKC_ASSIGN(RgbdFrame frame,
               realsense::rgbd_frame(view, setup, std::move(pixels), who));
    return IRgbdSensor::some_frame(std::move(frame));
  } catch (const std::exception& e) {
    return realsense::sdk_failure(kWho, "reading a frame", e);
  }
};

RealSenseSensor::RealSenseSensor(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
RealSenseSensor::RealSenseSensor(RealSenseSensor&& other) noexcept = default;
RealSenseSensor& RealSenseSensor::operator=(RealSenseSensor&& other) noexcept =
    default;
RealSenseSensor::~RealSenseSensor() = default;

core::Result<RealSenseSensor> RealSenseSensor::open(
    const Options& options) try {
  VKC_TRY(realsense::check_options(options));
  auto impl = std::make_unique<Impl>();
  VKC_ASSIGN(impl->camera, realsense::Camera::open(options.serial, kWho));
  realsense::Camera& camera = *impl->camera;
  const RealSenseDeviceInfo& about = camera.device_info();

  // The requested modes.
  VKC_ASSIGN(const rs2::video_stream_profile depth_profile,
             camera.mode(camera.stereo_module(), RS2_STREAM_DEPTH, -1,
                         RS2_FORMAT_Z16, options.depth_width,
                         options.depth_height, options.fps, "Z16 depth"));
  VKC_ASSIGN(const rs2::video_stream_profile color_profile,
             camera.mode(camera.color_module(), RS2_STREAM_COLOR, -1,
                         RS2_FORMAT_YUYV, options.color_width,
                         options.color_height, options.fps, "YUYV colour"));

  // The factory calibration, as the GPU pass reads it.
  const std::string& who = camera.who();
  realsense::FrameSetup& setup = impl->setup;
  VKC_ASSIGN(setup.depth_camera,
             realsense::camera_model_from(depth_profile.get_intrinsics(),
                                          "depth", false, who));
  VKC_ASSIGN(setup.color_camera,
             realsense::camera_model_from(color_profile.get_intrinsics(),
                                          "colour", true, who));
  VKC_ASSIGN(
      setup.depth_to_color,
      realsense::transform_from(depth_profile.get_extrinsics_to(color_profile),
                                "depth-to-colour", who));
  setup.color_to_world = options.color_to_world;
  setup.metres_per_unit = camera.stereo_module().get_depth_scale();
  setup.min_depth = options.min_depth;
  setup.max_depth = options.max_depth;

  // The projector and the colour exposure: last, so a camera refused above
  // gets no write; then the modes start opens.
  SensorInfo& info = impl->info;
  VKC_ASSIGN(info.color_controls,
             camera.configure(options.emitter, options.color_controls));
  VKC_TRY(camera.set_streams({depth_profile, color_profile}, options.fps));

  info.id = about.serial;
  info.model = about.name;
  info.color = setup.color_camera;
  info.depth = setup.depth_camera;
  info.depth_to_color = setup.depth_to_color;
  info.role = SyncRole::FreeRun;
  info.clock = ClockDomain::Host;
  info.pose = PoseSource::Fixed;
  info.fps = options.fps;
  return RealSenseSensor(std::move(impl));
} catch (const std::exception& e) {
  return realsense::sdk_failure(kWho, "opening the camera", e);
}

const RealSenseDeviceInfo& RealSenseSensor::device_info() const noexcept {
  static const RealSenseDeviceInfo kEmpty{};
  return impl_ != nullptr ? impl_->camera->device_info() : kEmpty;
}

const SensorInfo& RealSenseSensor::info() const noexcept {
  static const SensorInfo kEmpty{};
  return impl_ != nullptr ? impl_->info : kEmpty;
}

core::Status RealSenseSensor::set_queue_depth(std::size_t frames) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: set_queue_depth on a moved-from sensor");
  }
  return impl_->camera->set_queue_depth(frames);
}

core::Status RealSenseSensor::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: start on a moved-from sensor");
  }
  return impl_->camera->start();
}

void RealSenseSensor::stop() noexcept {
  if (impl_ != nullptr) impl_->camera->stop();
}

core::Result<std::optional<RgbdFrame>> RealSenseSensor::poll() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: poll on a moved-from sensor");
  }
  VKC_ASSIGN(std::optional<realsense::Arrived> arrived,
             impl_->camera->take_newest());
  if (!arrived) return no_frame();
  return impl_->read(std::move(*arrived));
}

core::Status RealSenseSensor::drain(std::vector<RgbdFrame>* out) {
  VKC_CHECK(out != nullptr,
            "RealSenseSensor::drain needs somewhere to put them");
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: drain on a moved-from sensor");
  }
  std::vector<realsense::Arrived> taken;
  VKC_TRY(impl_->camera->take_all(&taken));
  for (std::size_t i = 0; i < taken.size(); ++i) {
    core::Result<std::optional<RgbdFrame>> frame =
        impl_->read(std::move(taken[i]));
    if (!frame.ok()) {
      // The frames before it stay handed out; the rest were taken and will
      // not be.
      for (std::size_t j = i + 1; j < taken.size(); ++j) {
        impl_->camera->discard();
      }
      return frame.status();
    }
    if (frame.value()) out->push_back(std::move(*frame.value()));
  }
  return {};
}

bool RealSenseSensor::exhausted() const noexcept {
  return impl_ == nullptr || impl_->camera->disconnected();
}

SensorStats RealSenseSensor::stats() const noexcept {
  return impl_ != nullptr ? impl_->camera->stats() : SensorStats{};
}

core::Status RealSenseSensor::set_frame_notify(std::function<void()> notify) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: set_frame_notify on a moved-from sensor");
  }
  return impl_->camera->set_frame_notify(std::move(notify));
}

}  // namespace volumetric_kit::sensor
