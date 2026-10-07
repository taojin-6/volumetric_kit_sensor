// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): one Orbbec camera's streams and frame path, behind
// OrbbecSensor. Taking pairs and reading them into frames are separate calls,
// so a drain takes every pending pair at once.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_stream.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

class HevcColorDecoder;
class JpegColorDecoder;

// What the SDK's threads write and the polling thread reads. Captured by
// shared_ptr in both SDK callbacks, since neither is guaranteed gone when the
// stream is: the SDK calls a *copy* of the device-changed callback after
// releasing its lock, so unregistering does not wait for a call in flight, and
// a pipeline stop that throws may leave the SDK holding the frame callback.
struct Mailbox {
  std::mutex mutex;
  // The newest pairs not yet taken, oldest first, at most `depth` of them: one
  // for a camera on its own (newest wins), a few in a sensor array, whose
  // cameras must all still hold a trigger's frame when a slow poll comes to
  // group them.
  std::deque<std::shared_ptr<ob::FrameSet>> pending;  // guarded by mutex
  std::size_t depth = 1;                              // guarded by mutex
  std::string fault;                                  // guarded by mutex
  // Set, after `fault`, once the camera is gone; read without the lock.
  std::atomic<bool> disconnected{false};
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> dropped{0};

  // A pair into `pending`, uncounted: one the SDK delivered already and a
  // colour decoder has decoded.
  void post(std::shared_ptr<ob::FrameSet> frameset);
  void on_devices_changed(const std::string& serial,
                          const ob::DeviceList& removed);
};

class CameraStream {
 public:
  // Read the camera's identity and role, check its orientation, find the
  // modes, and read each camera's factory model at them. Does not start.
  // Messages name `who` and the serial. `configure_logging` sets FFmpeg's log
  // level at the first start.
  static core::Result<std::unique_ptr<CameraStream>> create(
      std::shared_ptr<ob::Context> context, std::shared_ptr<ob::Device> device,
      const OrbbecStreamOptions& streams, const camera::Mat4d& color_to_world,
      bool configure_logging, const std::string& who);

  CameraStream(const CameraStream&) = delete;
  CameraStream& operator=(const CameraStream&) = delete;
  ~CameraStream();

  const OrbbecDeviceInfo& info() const noexcept { return info_; }
  bool running() const noexcept { return running_; }
  // The camera's stored sync settings, as the SDK reads them back.
  const OrbbecSyncSettings& sync_settings() const noexcept {
    return sync_settings_;
  }
  // Write sync settings to the camera, where they persist (its flash). Not
  // while running. What the SDK reads back after it is its own cache of the
  // effective settings, including its normalization, not the camera: the next
  // open is what reads the camera again.
  core::Status apply_sync(const OrbbecSyncSettings& settings);
  // Stamp frames with the SDK's global timestamps -- the camera's clock mapped
  // onto the host's, re-fitted as the two drift -- rather than the camera's
  // own. Before sync_clock_to_host, so that sync re-fits the mapping at once.
  // Unsupported where the camera has none.
  core::Status use_host_clock();
  // Set this camera's clock to the host's, once, before start. Never the
  // context's enableDeviceClockSync, which re-syncs every camera in the
  // process on a thread of its own. IoError if the camera cannot.
  core::Status sync_clock_to_host();
  bool disconnected() const noexcept {
    return mailbox_->disconnected.load(std::memory_order_acquire);
  }
  OrbbecStreamStats stats() const noexcept;

  // Start both streams, with fresh counters. OK if already running; IoError
  // once the camera has disconnected, or if the SDK refuses; the colour
  // decoder's error (Unsupported where it has no device path) if it does not
  // open or start.
  core::Status start();
  // Stop both streams and drop the pending pairs. Idempotent. The camera stays
  // open, and held.
  void stop() noexcept;

  // How many untaken pairs the mailbox keeps (default 1). Set before start.
  void set_queue_depth(std::size_t depth);
  // The newest pair not yet taken, or null, the older ones counted dropped;
  // IoError once disconnected, and the colour decoder's failure once it has
  // stopped (Unsupported, Backend, OutOfMemory). A taken pair is this
  // stream's to account for: read() it, or discard() it.
  core::Result<std::shared_ptr<ob::FrameSet>> take();
  // Every pair not yet taken, oldest first, appended to `out`; as take()
  // otherwise.
  core::Status take_all(std::vector<std::shared_ptr<ob::FrameSet>>* out);
  // A taken pair that will never be processed, counted as dropped.
  void discard() noexcept;

  // A pair as the cameras captured it: raw depth and the decoded colour, each
  // camera's model and the colour camera's pose. The colour is the picture
  // the hardware left on the device. Depth points into the pair, which the
  // frame holds with the SDK context, so it may outlive the stream. An empty
  // optional is a pair the SDK failed on, skipped and counted; IoError is a
  // pair contradicting the negotiated stream, or a run of ~a second's skips.
  core::Result<std::optional<RgbdFrame>> read(
      const std::shared_ptr<ob::FrameSet>& pair);
  // Each camera as it captures, and the depth camera's extrinsic to the colour
  // one.
  const camera::CameraModel& depth_camera() const noexcept {
    return depth_camera_;
  }
  const camera::CameraModel& color_camera() const noexcept {
    return color_camera_;
  }
  const camera::Mat4d& depth_to_color() const noexcept {
    return depth_to_color_;
  }
  std::uint32_t fps() const noexcept { return fps_; }

 private:
  CameraStream() = default;
  // The colour decoder's failure, OK while it runs.
  core::Status decoder_failure() const;

  std::string who_;
  // A handle on the SDK's process-wide runtime, which every ob::Context
  // shares. Declared first so it is destroyed last: every SDK object below
  // belongs to it.
  std::shared_ptr<ob::Context> context_;
  // Next, so a pending frame is released while the context lives.
  std::shared_ptr<Mailbox> mailbox_ = std::make_shared<Mailbox>();
  std::shared_ptr<ob::Device> device_;
  std::shared_ptr<ob::Pipeline> pipeline_;
  std::shared_ptr<ob::StreamProfile> depth_profile_;
  // The H.265 or MJPG mode streamed, whose calibration is the colour
  // camera's.
  std::shared_ptr<ob::StreamProfile> color_profile_;
  std::uint32_t fps_ = 0;
  bool configure_ffmpeg_logging_ = true;  // cleared by the first start
  // The device the colour is decoded onto (streams.device), and the allocator
  // its pictures are made through (streams.allocator).
  const core::Device* vulkan_device_ = nullptr;
  core::Allocator* vulkan_allocator_ = nullptr;
  // The colour decoder between the SDK and the mailbox: hevc_ for H.265,
  // jpeg_ for MJPEG, the other null. Replaced at each start, so its counters
  // start fresh with the rest.
  std::shared_ptr<HevcColorDecoder> hevc_;
  std::shared_ptr<JpegColorDecoder> jpeg_;
  OrbbecColorCodec color_codec_ = OrbbecColorCodec::Mjpeg;
  bool device_callback_registered_ = false;
  OBCallbackId device_callback_id_ = 0;

  OrbbecDeviceInfo info_;
  OrbbecSyncSettings sync_settings_;
  bool host_clock_ = false;  // global timestamps, set by use_host_clock()

  // The polling thread's counters; the SDK's thread counts in the mailbox.
  std::uint64_t delivered_ = 0;
  std::uint64_t failed_ = 0;
  std::uint64_t discarded_ = 0;
  std::uint32_t failed_in_a_row_ = 0;
  bool running_ = false;

  camera::CameraModel depth_camera_;
  camera::CameraModel color_camera_;
  camera::Mat4d color_to_world_ = camera::Mat4d(1.0);
  camera::Mat4d depth_to_color_ = camera::Mat4d(1.0);
  float min_depth_ = 0.0f;
  float max_depth_ = 0.0f;
};

// One camera -- `serial`, or the only one that answers -- found and created
// through a context the stream holds.
core::Result<std::unique_ptr<CameraStream>> open_camera(
    const std::string& serial, std::uint32_t discovery_timeout_ms,
    bool configure_logging, const OrbbecStreamOptions& streams,
    const camera::Mat4d& color_to_world, const std::string& who);

}  // namespace volumetric_kit::recon::sensor::orbbec
