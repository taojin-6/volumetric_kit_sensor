// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): ArkitSensor's behaviour over any feed of ARFrame
// parts -- the option checks, the cameras taken from the first frame, the
// queue, the depth copies, the counters, the status, and the import at poll --
// with no ARKit type. ARKit's feed (arkit_feed.mm, iOS and iPadOS), the
// pattern's (pattern_feed.mm) and the tests' scripts all call deliver(), as
// DeckLink's driver runs over any capture.

#include <CoreVideo/CoreVideo.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "../realsense/frame_queue.hpp"
#include "conversion.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/apple/pixel_buffer_importer.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor::arkit {

// One ARFrame's parts as a feed hands them over, borrowed for the call.
struct Capture {
  // The requested map's DepthFloat32 metres; null for a frame without it,
  // which takes no number.
  CVPixelBufferRef depth = nullptr;
  CVPixelBufferRef confidence = nullptr;  // OneComponent8: 0, 1, 2
  // '420f' on an IOSurface; retained when held.
  CVPixelBufferRef picture = nullptr;
  std::array<float, 9> intrinsics{};  // ARCamera.intrinsics, column-major
  // ARCamera.transform, column-major, in ARKit's axes.
  std::array<float, 16> transform{};
  double timestamp_s = 0.0;  // ARFrame.timestamp, on the uptime clock
  ArkitTracking tracking = ArkitTracking::Normal;
  ColorExposure exposure{};
};

// ArkitSensor::Options' refusals before ARKit is asked (InvalidArgument), as
// ArkitSensor::open's documentation lists them, naming the field.
core::Status check_options(const ArkitSensor::Options& options);

// What a source opens and closes: ARKit's session, or the pattern's timer.
class Feed {
 public:
  virtual ~Feed() = default;
  // At a start, after Driver::begin: deliver from now (the camera's session
  // runs again after a stop, its world kept).
  virtual core::Status start() = 0;
  // At a stop, before Driver::end: no more captures (the session paused, the
  // timer stopped); one in flight may still land, which the driver lets go.
  virtual void stop() noexcept = 0;
};

class Driver {
 public:
  // The picture importer, on options.device for options' queue families
  // (PixelBufferImporter::create's refusals). `model` is SensorInfo::model.
  static core::Result<std::unique_ptr<Driver>> create(
      const ArkitSensor::Options& options, std::string model);

  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;
  Driver(Driver&&) = delete;
  Driver& operator=(Driver&&) = delete;
  ~Driver();

  // --- The feed's side: from one thread at a time.

  // One ARFrame, as ArkitSensor's documentation says. Before the first with
  // the map, its cameras become the sensor's, which wait_opened waits for.
  // Before a start and after a stop, let go uncounted. While started: its
  // period after the one before counted missed; without the map, let go
  // uncounted; else numbered and received, then withheld, refused or held,
  // the consumer notified once it is.
  void deliver(const Capture& capture, std::uint64_t arrival_ns);
  // ARKit stopped or resumed the camera; at the end, while started, the
  // clock is measured again (ClockMap::remeasure).
  void interrupted(bool now);
  // ARKit's session failed: every later poll, drain and start says so, as
  // does a wait_opened still waiting; the consumer is notified.
  void fail(const std::string& why);
  // The sequence the next capture with the map takes, for the pattern.
  std::uint64_t next_sequence() const;

  // --- The sensor's side: from the consumer's thread, but status and stats.

  // To `timeout` for the first capture with the map. NotFound where none
  // came, saying so where ARKit has the session interrupted then; IoError
  // where the session failed first (its message), or the first one is not a
  // picture of the options' size or a usable camera.
  core::Status wait_opened(std::chrono::milliseconds timeout);
  const SensorInfo& info() const noexcept { return info_; }
  core::Status set_queue_depth(std::size_t frames);
  core::Status set_frame_notify(std::function<void()> notify);
  // Frames from now: the counters and sequences from 0, the clock measured
  // (ClockMap::start). IoError once the session failed.
  core::Status begin();
  // Frames from now let go, and those held dropped uncounted.
  void end() noexcept;
  bool started() const;
  core::Result<std::optional<RgbdFrame>> poll();
  core::Status drain(std::vector<RgbdFrame>* out);
  bool failed() const noexcept { return queue_.failed(); }
  SensorStats stats() const noexcept;
  ArkitStatus status() const;

 private:
  class CopyPool;
  // A frame's depth and levels, packed: the one copy out of ARKit's buffers,
  // pooled.
  struct DepthCopy {
    std::vector<std::uint16_t> depth;
    std::vector<std::uint8_t> confidence;
  };
  // A CVPixelBuffer retained while this holds it.
  class Retained {
   public:
    Retained() = default;
    explicit Retained(CVPixelBufferRef p) : p_(CVPixelBufferRetain(p)) {}
    Retained(Retained&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    Retained& operator=(Retained&& o) noexcept;
    ~Retained() {
      if (p_ != nullptr) CVPixelBufferRelease(p_);
    }
    Retained(const Retained&) = delete;
    Retained& operator=(const Retained&) = delete;
    CVPixelBufferRef get() const { return p_; }

   private:
    CVPixelBufferRef p_ = nullptr;
  };
  // A frame received and converted, held until a poll or drain imports its
  // picture.
  struct Held {
    std::shared_ptr<const DepthCopy> depth;
    Retained picture;
    PictureColor color;
    camera::CameraModel color_camera;
    camera::CameraModel depth_camera;
    camera::Mat4d color_to_world{1.0};
    ColorExposure exposure;
    std::uint64_t timestamp_ns = 0;
    std::uint64_t sequence = 0;
    std::uint64_t arrival_ns = 0;
  };

  Driver(const ArkitSensor::Options& options, std::string model,
         PixelBufferImporter importer);
  // The sensor's cameras from the first capture with the map (under mutex_).
  void open_from(const Capture& capture);
  // A received capture converted to what a frame holds; or why it is refused.
  core::Result<Held> convert(const Capture& capture);
  // A taken frame's picture imported and the frame made; its import's
  // failure counted failed.
  core::Result<RgbdFrame> hand_out(Held held);

  ArkitSensor::Options options_;
  SensorInfo info_;               // written once, before wait_opened returns
  PixelBufferImporter importer_;  // the consumer's thread alone
  std::shared_ptr<CopyPool> copies_;
  realsense::FrameQueue<Held> queue_;

  mutable std::mutex mutex_;
  std::condition_variable opened_cv_;
  // Guarded by mutex_:
  bool opened_ = false;
  std::string open_error_;
  bool started_ = false;
  ClockMap clock_;
  std::optional<double> last_time_s_;  // the last capture's, since the start
  std::uint64_t next_sequence_ = 0;
  std::uint64_t received_ = 0;
  std::uint64_t delivered_ = 0;
  std::uint64_t import_failed_ = 0;
  ArkitStatus status_;
  std::function<void()> notify_;
};

}  // namespace volumetric_kit::sensor::arkit
