// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): one RealSense camera as both drivers run it.
// RealSenseSensor (the camera's own depth) and RealSenseStereo (its two
// infrared cameras) are this camera opened at other modes: it finds the
// camera, its stereo and colour modules and their modes, writes the projector
// and the colour exposure, starts and stops the pipeline with global time on,
// watches for a disconnect, and keeps the framesets its SDK thread delivers
// in a frame queue, with the counters every driver reports. Each driver turns
// a frameset into its own frame.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <librealsense2/rs.hpp>

#include "conversion.hpp"
#include "frame_queue.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor::realsense {

// A frameset the SDK delivered, and when.
struct Arrived {
  rs2::frameset frames;
  std::uint64_t arrival_ns = 0;
};

// librealsense reports every failure by throwing; each becomes an IoError
// naming `who`, what was being done, and the SDK's failed function.
core::Status sdk_failure(const std::string& who, const std::string& what,
                         const std::exception& e);

// The host's system clock, in nanoseconds: a frame's arrival.
std::uint64_t now_ns() noexcept;

// A frame's picture as plain values.
PictureView picture_of(const rs2::video_frame& frame);

class Camera {
 public:
  // The attached camera with `serial`, or the only one attached when it is
  // empty, with its stereo and colour modules; `kind` ("RealSenseSensor")
  // names the driver in every message. NotFound naming the cameras attached;
  // InvalidArgument for an empty serial with more than one attached;
  // Unsupported on a USB 2 link or without a stereo or colour module; IoError
  // for any SDK failure.
  static core::Result<std::unique_ptr<Camera>> open(const std::string& serial,
                                                    const char* kind);

  // Stops the pipeline, then lets go of the device before the context goes.
  ~Camera();
  Camera(const Camera&) = delete;
  Camera& operator=(const Camera&) = delete;
  Camera(Camera&&) = delete;
  Camera& operator=(Camera&&) = delete;

  const RealSenseDeviceInfo& device_info() const noexcept { return about_; }
  // "<kind>: camera <serial>", for messages.
  const std::string& who() const noexcept { return who_; }
  rs2::depth_sensor& stereo_module() noexcept { return *stereo_; }
  rs2::color_sensor& color_module() noexcept { return *color_; }

  // `module`'s mode of `stream` (`index`: librealsense's index, 1 or 2 for
  // the infrared cameras; -1 any) in `format` at the size and rate;
  // Unsupported naming `what` and the modes it offers otherwise.
  core::Result<rs2::video_stream_profile> mode(
      const rs2::sensor& module, rs2_stream stream, int index,
      rs2_format format, std::uint32_t width, std::uint32_t height,
      std::uint32_t fps, const char* what) const;

  // The projector and the colour camera's exposure (written and read back:
  // what it returns), last in an open so a camera refused before gets no
  // write. Unsupported for a camera without global time (which start turns
  // on), with no projector to switch on, or without a colour control or
  // whose range excludes the value (nothing written then).
  core::Result<std::optional<camera::ColorControls>> configure(
      bool emitter, const std::optional<camera::ColorControls>& controls);

  // The modes start opens, checked to run together at `fps`, and the
  // disconnect watch, which fails every later take.
  core::Status set_streams(
      const std::vector<rs2::video_stream_profile>& profiles,
      std::uint32_t fps);

  // As IRgbdSensor's: InvalidArgument once started, or for no frames.
  core::Status set_queue_depth(std::size_t frames);
  core::Status set_frame_notify(std::function<void()> notify);

  // Starts the pipeline, the counters restarted, and turns global time on
  // where it streams. Each frameset that is not a frame of every mode, of
  // one moment (`together`), is counted failed and never queued. OK if
  // already running; IoError once disconnected or if the SDK refuses.
  core::Status start();
  // Stops the pipeline; the framesets held let go, the counters kept.
  void stop() noexcept;

  core::Result<std::optional<Arrived>> take_newest() {
    return shared_->queue.take_newest();
  }
  core::Status take_all(std::vector<Arrived>* out) {
    return shared_->queue.take_all(out);
  }
  // A frameset taken and never handed out, counted dropped.
  void discard() noexcept { shared_->queue.discard(); }
  // A frame handed out, and one a driver refused.
  void count_delivered() noexcept {
    delivered_.fetch_add(1, std::memory_order_release);
  }
  void count_failed() noexcept {
    shared_->failed.fetch_add(1, std::memory_order_release);
  }

  bool disconnected() const noexcept {
    return shared_->disconnected.load(std::memory_order_acquire);
  }
  // The counters since the last start, kept through a stop.
  SensorStats stats() const noexcept;

 private:
  // What the SDK's threads write and the polling thread reads, held by the
  // callbacks too, since neither is guaranteed gone when the camera is.
  struct Shared {
    FrameQueue<Arrived> queue;
    std::atomic<std::uint64_t> received{0};
    std::atomic<std::uint64_t> failed{0};
    std::atomic<bool> disconnected{false};
  };

  Camera() = default;

  rs2::context context_;
  rs2::device device_;
  std::optional<rs2::depth_sensor> stereo_;
  std::optional<rs2::color_sensor> color_;
  rs2::config config_;
  std::optional<rs2::pipeline> pipeline_;
  std::shared_ptr<Shared> shared_ = std::make_shared<Shared>();
  RealSenseDeviceInfo about_;
  const char* kind_ = "";
  std::string who_;
  std::uint32_t fps_ = 0;
  std::size_t modes_ = 0;
  bool running_ = false;
  std::atomic<std::uint64_t> delivered_{0};
};

}  // namespace volumetric_kit::sensor::realsense
