// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "realsense_camera_stub.hpp"

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <librealsense2/hpp/rs_internal.hpp>
#include <librealsense2/rs.hpp>

#include "camera.hpp"

namespace realsense_stub {
namespace {

std::vector<std::uint16_t> depth_pixels(std::size_t{kWidth} * kHeight,
                                        kDepthMm);
std::vector<std::uint8_t> color_pixels(std::size_t{kWidth} * kHeight * 2, 128);

void keep(void*) {}  // the pixels outlive every frame

// The software device and its two modules, made at the camera's open and
// kept until the next one, so a frame outlives its camera as a driver's may.
struct Rig {
  rs2::software_device device;
  rs2::software_sensor depth;
  rs2::software_sensor color;
  rs2::stream_profile depth_profile;
  rs2::stream_profile color_profile;
  rs2::frame_queue depth_frames{1, true};
  rs2::frame_queue color_frames{1, true};
  bool streaming = false;
};
std::unique_ptr<Rig> rig;

// Where a frameset lands while the camera runs: its frame queue.
std::function<void(rs2::frameset)> land;

rs2_intrinsics pinhole() {
  rs2_intrinsics k{};
  k.width = static_cast<int>(kWidth);
  k.height = static_cast<int>(kHeight);
  k.ppx = 31.5f;
  k.ppy = 23.5f;
  k.fx = 50.0f;
  k.fy = 50.0f;
  k.model = RS2_DISTORTION_BROWN_CONRADY;  // every coefficient 0: a pinhole
  return k;
}

rs2_video_stream stream_of(rs2_stream type, int uid, rs2_format format) {
  return {type,
          0,
          uid,
          static_cast<int>(kWidth),
          static_cast<int>(kHeight),
          static_cast<int>(kFps),
          2,
          format,
          pinhole()};
}

std::unique_ptr<Rig> make_rig() {
  rs2::software_device device;
  device.register_info(RS2_CAMERA_INFO_NAME, "Software D455");
  device.register_info(RS2_CAMERA_INFO_SERIAL_NUMBER, "STUB-455");
  rs2::software_sensor depth = device.add_sensor("Stereo Module");
  rs2::software_sensor color = device.add_sensor("RGB Camera");
  depth.add_read_only_option(RS2_OPTION_DEPTH_UNITS, 0.001f);
  rs2::stream_profile depth_profile = depth.add_video_stream(
      stream_of(RS2_STREAM_DEPTH, 0, RS2_FORMAT_Z16), true);
  rs2::stream_profile color_profile = color.add_video_stream(
      stream_of(RS2_STREAM_COLOR, 1, RS2_FORMAT_YUYV), true);
  depth_profile.register_extrinsics_to(
      color_profile, {{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0.05f, 0, 0}});
  return std::unique_ptr<Rig>(
      new Rig{device, depth, color, depth_profile, color_profile});
}

// `sensor`'s frame of capture `number`.
void on_frame(rs2::software_sensor& sensor, const rs2::stream_profile& profile,
              void* pixels, std::uint64_t number, double ms,
              rs2_timestamp_domain domain) {
  sensor.on_video_frame({pixels, keep, static_cast<int>(kWidth) * 2, 2, ms,
                         domain, static_cast<int>(number), profile.get(),
                         0.001f});
}

}  // namespace

bool send(std::uint64_t number, double ms, rs2_timestamp_domain domain) {
  if (rig == nullptr) return false;
  if (!rig->streaming) {
    rig->depth.open(rig->depth_profile);
    rig->color.open(rig->color_profile);
    rig->depth.start(rig->depth_frames);
    rig->color.start(rig->color_frames);
    rig->streaming = true;
  }
  on_frame(rig->depth, rig->depth_profile, depth_pixels.data(), number, ms,
           domain);
  on_frame(rig->color, rig->color_profile, color_pixels.data(), number, ms,
           domain);
  rs2::frame depth;
  rs2::frame color;
  if (!rig->depth_frames.try_wait_for_frame(&depth, 1000) ||
      !rig->color_frames.try_wait_for_frame(&color, 1000)) {
    return false;
  }
  // One frameset of the two, as the pipeline hands one over.
  rs2::frame_queue made(1, true);
  rs2::processing_block combine(
      [color](rs2::frame frame, rs2::frame_source& source) {
        source.frame_ready(source.allocate_composite_frame({frame, color}));
      });
  combine.start(made);
  combine.invoke(depth);
  rs2::frame set;
  if (!made.try_wait_for_frame(&set, 1000)) return false;
  const auto frames = set.as<rs2::frameset>();
  if (!frames || frames.size() != 2) return false;
  if (land) land(frames);
  return true;
}

}  // namespace realsense_stub

namespace volumetric_kit::sensor::realsense {

// camera.cpp's own helpers, which the driver calls.
core::Status sdk_failure(const std::string& who, const std::string& what,
                         const std::exception& e) {
  return core::Status::io_error(who + ": " + what + ": " + e.what());
}

std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

PictureView picture_of(const rs2::video_frame& frame) {
  return {frame.get_data(), static_cast<std::uint32_t>(frame.get_width()),
          static_cast<std::uint32_t>(frame.get_height()),
          static_cast<std::size_t>(frame.get_stride_in_bytes())};
}

core::Result<std::unique_ptr<Camera>> Camera::open(const std::string& serial,
                                                   const char* kind) try {
  std::unique_ptr<Camera> camera(new Camera());
  camera->kind_ = kind;
  realsense_stub::rig = realsense_stub::make_rig();
  realsense_stub::Rig& rig = *realsense_stub::rig;
  camera->device_ = rig.device;
  camera->about_ = {"Software D455", "STUB-455", "0", "3.2"};
  if (!serial.empty() && serial != camera->about_.serial) {
    return core::Status::not_found(std::string(kind) + ": camera " + serial +
                                   " is not attached (stub)");
  }
  camera->who_ = std::string(kind) + ": camera " + camera->about_.serial;
  camera->stereo_ = rig.depth.as<rs2::depth_sensor>();
  camera->color_ = rs2::color_sensor(rig.color);
  return camera;
} catch (const std::exception& e) {
  return sdk_failure(kind, "opening the camera", e);
}

Camera::~Camera() { stop(); }

core::Result<rs2::video_stream_profile> Camera::mode(
    const rs2::sensor&, rs2_stream type, int, rs2_format format,
    std::uint32_t width, std::uint32_t height, std::uint32_t fps,
    const char* what) const {
  const realsense_stub::Rig& rig = *realsense_stub::rig;
  for (const rs2::stream_profile& p : {rig.depth_profile, rig.color_profile}) {
    const auto v = p.as<rs2::video_stream_profile>();
    if (p.stream_type() == type && p.format() == format &&
        v.width() == static_cast<int>(width) &&
        v.height() == static_cast<int>(height) &&
        p.fps() == static_cast<int>(fps)) {
      return v;
    }
  }
  return core::Status::unsupported(who_ + " has no " + what + " mode (stub)");
}

// Written and read back as asked: the stub's camera keeps any exposure.
core::Result<std::optional<camera::ColorControls>> Camera::configure(
    bool, const std::optional<camera::ColorControls>& controls) {
  return controls;
}

core::Status Camera::set_streams(
    const std::vector<rs2::video_stream_profile>& profiles, std::uint32_t fps) {
  fps_ = fps;
  modes_ = profiles.size();
  return {};
}

core::Status Camera::set_queue_depth(std::size_t frames) {
  if (frames == 0 || running_) {
    return core::Status::invalid_argument(std::string(kind_) +
                                          ": at least one frame, before start");
  }
  shared_->queue.set_depth(frames);
  return {};
}

core::Status Camera::set_frame_notify(std::function<void()> notify) {
  if (running_) {
    return core::Status::invalid_argument(
        std::string(kind_) + ": the frame notification is set before start");
  }
  shared_->queue.set_notify(std::move(notify));
  return {};
}

// The counters restart, and each frameset sent lands in the queue, counted
// received: the stub's camera sends whole moments only.
core::Status Camera::start() {
  if (running_) return {};
  shared_->received.store(0);
  shared_->failed.store(0);
  delivered_.store(0);
  shared_->queue.clear();
  shared_->queue.reset_dropped();
  realsense_stub::land = [shared = shared_](rs2::frameset frames) {
    shared->received.fetch_add(1);
    shared->queue.push(Arrived{std::move(frames), now_ns()});
  };
  running_ = true;
  return {};
}

void Camera::stop() noexcept {
  running_ = false;
  realsense_stub::land = nullptr;
  shared_->queue.clear();
}

SensorStats Camera::stats() const noexcept {
  SensorStats s;
  s.delivered = delivered_.load();
  s.dropped = shared_->queue.dropped();
  s.failed = shared_->failed.load();
  s.received = shared_->received.load();
  return s;
}

}  // namespace volumetric_kit::sensor::realsense
