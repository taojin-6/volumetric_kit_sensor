// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): one Orbbec camera's streams and frame path, shared
// by OrbbecCapture (one camera) and OrbbecRig (several). Taking a pair and
// processing it are separate calls so the rig can group pairs by timestamp
// before it pays for processing any.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/camera_capture.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

// The SDK reports every failure as a thrown ob::Error; this repo returns
// Status across its API. `who` names the caller ("OrbbecCapture", ...).
Status sdk_error(const std::string& who, const std::string& what,
                 const std::exception& e);

// The SDK's logger is process-wide: file sink off, console at WARN. One call
// per sink -- setLoggerSeverity sets every sink, the file one included.
void configure_sdk_logging();

// Find cameras on the network, re-querying until they answer or the window
// closes; one query is not proof of absence for an Ethernet camera. Named
// cameras (in `serials`' order) are returned the moment all have answered.
// An empty `serials` asks for the only camera: it is opened once the whole
// window has passed with no other answering, and refused as soon as a second
// does.
Result<std::vector<std::shared_ptr<ob::Device>>> discover(
    ob::Context& context, const std::vector<std::string>& serials,
    std::uint32_t timeout_ms, const std::string& who);

// What the SDK's threads write and the polling thread reads. Captured by
// shared_ptr in both SDK callbacks, since neither is guaranteed gone when the
// stream is: the SDK calls a *copy* of the device-changed callback after
// releasing its lock, so unregistering does not wait for a call in flight, and
// a pipeline stop that throws may leave the SDK holding the frame callback.
struct Mailbox {
  std::mutex mutex;
  // The newest pairs not yet taken, oldest first, at most `depth` of them: one
  // for a single camera (newest wins), a few for a rig, whose cameras must all
  // still hold a trigger's frame when a slow poll comes to group them.
  std::deque<std::shared_ptr<ob::FrameSet>> pending;  // guarded by mutex
  std::size_t depth = 1;                              // guarded by mutex
  std::string fault;                                  // guarded by mutex
  // Set, after `fault`, once the camera is gone; read without the lock.
  std::atomic<bool> disconnected{false};
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> dropped{0};

  void on_frameset(std::shared_ptr<ob::FrameSet> frameset);
  void on_devices_changed(const std::string& serial,
                          const ob::DeviceList& removed);
};

class CameraStream {
 public:
  // Read the camera's identity and role, check its orientation, find the
  // modes, derive the frame's cameras, and build the filters. Does not start.
  // Refuses a software-triggered camera. Messages name `who` and the serial.
  static Result<std::unique_ptr<CameraStream>> create(
      std::shared_ptr<ob::Context> context, std::shared_ptr<ob::Device> device,
      const OrbbecStreamOptions& streams, const Mat4f& cam_to_world,
      const std::string& who);

  CameraStream(const CameraStream&) = delete;
  CameraStream& operator=(const CameraStream&) = delete;
  ~CameraStream();

  const OrbbecDeviceInfo& info() const noexcept { return info_; }
  const ColorCameraParams& color_camera() const noexcept {
    return color_camera_;
  }
  bool running() const noexcept { return running_; }
  bool disconnected() const noexcept {
    return mailbox_->disconnected.load(std::memory_order_acquire);
  }
  OrbbecCaptureStats stats() const noexcept;

  // Start both streams, with fresh counters. OK if already running; IoError
  // once the camera has disconnected, or if the SDK refuses.
  Status start();
  // Stop both streams; drop the pending pair and the processed frame's
  // storage. Idempotent. The camera stays open, and held.
  void stop() noexcept;

  // How many untaken pairs the mailbox keeps (default 1). Set before start.
  void set_queue_depth(std::size_t depth);
  // The newest pair not yet taken, or null, the older ones counted dropped;
  // IoError once disconnected. A taken pair is this stream's to account for:
  // process() it, or discard() it.
  Result<std::shared_ptr<ob::FrameSet>> take();
  // Every pair not yet taken, oldest first, appended to `out`; as take()
  // otherwise.
  Status take_all(std::vector<std::shared_ptr<ob::FrameSet>>* out);
  // A taken pair that will never be processed, counted as dropped.
  void discard() noexcept;
  // The device timestamp of a pair's depth frame (us), or 0 when it has none.
  static std::uint64_t timestamp_us(const ob::FrameSet& pair) noexcept;

  // Undistort colour, register depth to it, convert -- into this stream's
  // storage, which the returned frame borrows until the next process() or
  // stop(). An empty optional is a pair the SDK failed on, skipped and
  // counted; IoError is a pair contradicting the negotiated stream, or a run
  // of ~a second's skips.
  Result<std::optional<CapturedFrame>> process(
      const std::shared_ptr<ob::FrameSet>& pair);

 private:
  CameraStream() = default;

  std::string who_;
  // Declared first so it is destroyed last: every SDK object below belongs to
  // this context.
  std::shared_ptr<ob::Context> context_;
  // Next, so a pending frame is released while the context lives.
  std::shared_ptr<Mailbox> mailbox_ = std::make_shared<Mailbox>();
  std::shared_ptr<ob::Device> device_;
  std::shared_ptr<ob::Pipeline> pipeline_;
  std::shared_ptr<ob::StreamProfile> depth_profile_;
  std::shared_ptr<ob::StreamProfile> color_profile_;
  std::shared_ptr<ob::UnDistortionFilter> undistort_color_;
  std::shared_ptr<ob::Align> align_to_color_;
  bool device_callback_registered_ = false;
  OBCallbackId device_callback_id_ = 0;

  OrbbecDeviceInfo info_;
  ColorCameraParams color_camera_{};
  DepthCameraParams depth_camera_{};

  // The polling thread's counters; the SDK's thread counts in the mailbox.
  std::uint64_t delivered_ = 0;
  std::uint64_t failed_ = 0;
  std::uint64_t discarded_ = 0;
  std::uint32_t failed_in_a_row_ = 0;
  bool running_ = false;
  // Whether this start's first processed pair has been held to the camera
  // the frames are stamped with.
  bool first_pair_checked_ = false;

  std::vector<float> depth_metres_;
  std::vector<std::uint32_t> color_packed_;
};

}  // namespace volumetric_kit::recon::sensor::orbbec
