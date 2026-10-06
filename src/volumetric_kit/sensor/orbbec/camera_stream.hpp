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

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/camera_capture.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

class HevcColorDecoder;
class JpegColorDecoder;

// The SDK reports every failure as a thrown ob::Error; this repo returns
// Status across its API. `who` names the caller ("OrbbecCapture", ...).
core::Status sdk_error(const std::string& who, const std::string& what,
                       const std::exception& e);

// The SDK's logger is process-wide: file sink off, console at WARN. One call
// per sink -- setLoggerSeverity sets every sink, the file one included.
void configure_sdk_logging();

// Unsupported for H.265 colour or raw frames in a build without the video
// decoders; OK otherwise. Asked by open before the SDK is touched.
core::Status check_color_codec(const OrbbecStreamOptions& streams,
                               const std::string& who);

// Find cameras on the network, re-querying until they answer or the window
// closes; one query is not proof of absence for an Ethernet camera. Named
// cameras (in `serials`' order) are returned the moment all have answered.
// An empty `serials` asks for the only camera: it is opened once the whole
// window has passed with no other answering, and refused as soon as a second
// does.
core::Result<std::vector<std::shared_ptr<ob::Device>>> discover(
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

  // A pair from the SDK: counted as received, and posted.
  void on_frameset(std::shared_ptr<ob::FrameSet> frameset);
  // A pair into `pending`, uncounted: one the SDK delivered already and a
  // colour decoder has decoded.
  void post(std::shared_ptr<ob::FrameSet> frameset);
  void on_devices_changed(const std::string& serial,
                          const ob::DeviceList& removed);
};

class CameraStream {
 public:
  // Read the camera's identity and role, check its orientation, find the
  // modes, derive the frame's cameras, and build the filters. Does not start.
  // Refuses a software-triggered camera, and a colour mode on the wire (H.265,
  // or a raw stream's MJPG) whose calibration is not the RGB mode's. Messages
  // name `who` and the serial. `configure_logging` sets FFmpeg's log level at
  // the first start, for a stream decoded here (H.265, or raw).
  static core::Result<std::unique_ptr<CameraStream>> create(
      std::shared_ptr<ob::Context> context, std::shared_ptr<ob::Device> device,
      const OrbbecStreamOptions& streams, const camera::Mat4d& cam_to_world,
      bool configure_logging, const std::string& who);

  CameraStream(const CameraStream&) = delete;
  CameraStream& operator=(const CameraStream&) = delete;
  ~CameraStream();

  const OrbbecDeviceInfo& info() const noexcept { return info_; }
  const ColorCameraParams& color_camera() const noexcept {
    return color_camera_;
  }
  bool running() const noexcept { return running_; }
  // The camera's stored sync settings, as the SDK reads them back.
  const OrbbecSyncSettings& sync_settings() const noexcept {
    return sync_settings_;
  }
  // Write sync settings to the camera, where they persist. Not while running.
  core::Status apply_sync(const OrbbecSyncSettings& settings);
  bool disconnected() const noexcept {
    return mailbox_->disconnected.load(std::memory_order_acquire);
  }
  OrbbecCaptureStats stats() const noexcept;

  // Start both streams, with fresh counters. OK if already running; IoError
  // once the camera has disconnected, or if the SDK refuses. For a stream
  // decoded here (H.265, or raw MJPEG), Unsupported or IoError if the
  // decoder does not open or start.
  core::Status start();
  // Stop both streams; drop the pending pair and the processed frame's
  // storage. Idempotent. The camera stays open, and held.
  void stop() noexcept;

  // How many untaken pairs the mailbox keeps (default 1). Set before start.
  void set_queue_depth(std::size_t depth);
  // The newest pair not yet taken, or null, the older ones counted dropped;
  // IoError once disconnected. A taken pair is this stream's to account for:
  // process() it, or discard() it.
  core::Result<std::shared_ptr<ob::FrameSet>> take();
  // Every pair not yet taken, oldest first, appended to `out`; as take()
  // otherwise.
  core::Status take_all(std::vector<std::shared_ptr<ob::FrameSet>>* out);
  // A taken pair that will never be processed, counted as dropped.
  void discard() noexcept;
  // A pair process() delivered that the caller will not hand out after all,
  // recounted as dropped.
  void withdraw() noexcept;
  // The device timestamp of a pair's depth frame (us), or 0 when it has none.
  static std::uint64_t timestamp_us(const ob::FrameSet& pair) noexcept;

  // Undistort colour, register depth to it, convert -- into this stream's
  // storage, which the returned frame borrows until the next process() or
  // stop(). An empty optional is a pair the SDK failed on, skipped and
  // counted; IoError is a pair contradicting the negotiated stream, or a run
  // of ~a second's skips.
  core::Result<std::optional<CapturedFrame>> process(
      const std::shared_ptr<ob::FrameSet>& pair);
  // A pair as the cameras captured it, for a stream opened raw: raw depth and
  // the decoded colour, each camera's model and the colour camera's pose. The
  // colour is the picture the hardware left on the device, or I420 host
  // planes. Depth and host planes point into the pair, which the frame holds
  // with the SDK context, so it may outlive the stream.
  core::Result<std::optional<RgbdFrame>> process_raw(
      const std::shared_ptr<ob::FrameSet>& pair);
  bool raw() const noexcept { return raw_; }

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
  // The RGB mode: the colour camera's calibration, and the profile of the
  // frames process() is handed. It is also what the wire carries, unless
  // `wire_color_profile_` is set: the H.265 mode of the same size, or a raw
  // stream's MJPG one, whose calibration create() holds to be the same, byte
  // for byte.
  std::shared_ptr<ob::StreamProfile> color_profile_;
  std::shared_ptr<ob::StreamProfile> wire_color_profile_;
  std::uint32_t fps_ = 0;
  bool configure_ffmpeg_logging_ = true;  // cleared by the first start
  // The device a raw stream's colour is decoded onto (streams.device), and the
  // allocator its pictures are made through (streams.allocator).
  const core::Device* vulkan_device_ = nullptr;
  core::Allocator* vulkan_allocator_ = nullptr;
  // Decodes the H.265 colour, between the SDK and the mailbox; null for
  // MJPEG. Replaced at each start, so its counters start fresh with the rest.
  std::shared_ptr<HevcColorDecoder> hevc_;
  // Decodes a raw MJPEG stream's colour onto the GPU, between the SDK and the
  // mailbox; null otherwise. Replaced at each start, as hevc_ is.
  std::shared_ptr<JpegColorDecoder> jpeg_;
  OrbbecColorCodec color_codec_ = OrbbecColorCodec::Mjpeg;
  std::shared_ptr<ob::UnDistortionFilter> undistort_color_;
  std::shared_ptr<ob::Align> align_to_color_;
  bool device_callback_registered_ = false;
  OBCallbackId device_callback_id_ = 0;

  OrbbecDeviceInfo info_;
  OrbbecSyncSettings sync_settings_;
  ColorCameraParams color_camera_{};
  DepthCameraParams depth_camera_{};

  // The polling thread's counters; the SDK's thread counts in the mailbox.
  std::uint64_t delivered_ = 0;
  std::uint64_t failed_ = 0;
  std::uint64_t discarded_ = 0;
  std::uint64_t host_pictures_ = 0;  // OrbbecCaptureStats::host_pictures
  // Whether the last frame process_raw() delivered counted in host_pictures_,
  // so withdraw() takes it back out.
  bool host_picture_delivered_ = false;
  std::uint32_t failed_in_a_row_ = 0;
  bool running_ = false;
  // Whether this start's first processed pair has been held to the camera
  // the frames are stamped with.
  bool first_pair_checked_ = false;

  std::vector<float> depth_metres_;
  std::vector<std::uint32_t> color_packed_;

  // A raw stream's cameras (OrbbecStreamOptions::raw) and poses.
  bool raw_ = false;
  camera::CameraModel raw_depth_camera_;
  camera::CameraModel raw_color_camera_;
  camera::Mat4d raw_color_to_world_ = camera::Mat4d(1.0);
  camera::Mat4d raw_depth_to_color_ = camera::Mat4d(1.0);
  float min_depth_ = 0.0f;
  float max_depth_ = 0.0f;
};

}  // namespace volumetric_kit::recon::sensor::orbbec
