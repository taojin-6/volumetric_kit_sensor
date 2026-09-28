// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "camera_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

#include "frame_conversion.hpp"
#include "volumetric_kit/recon/sensor/camera_conventions.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

namespace {

const char* or_empty(const char* s) { return s != nullptr ? s : ""; }

// How often discovery re-asks the network. The query itself blocks about a
// second probing for Ethernet devices, so this only spaces out the retries.
constexpr std::chrono::milliseconds kDiscoveryRetry{250};

// Pairs in a row process() may skip before it calls the camera broken: about
// a second at the 30 fps default.
constexpr std::uint32_t kMaxFailedPairsInARow = 30;

// How far the SDK's intrinsics may sit from the camera a frame is stamped with
// before the first-pair check refuses them, in pixels.
constexpr float kIntrinsicsTolerance = 1e-3f;

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (const std::string& item : items) {
    if (!out.empty()) out += ", ";
    out += item;
  }
  return out;
}

// "640x576@30 Y16, 320x288@30 Y16, ..." -- what a camera offers, for the error
// that says the requested mode is not among them. It runs while that error is
// being built, so it must not throw one of its own over it.
std::string list_modes(const ob::StreamProfileList& profiles) {
  std::string out;
  try {
    for (std::uint32_t i = 0; i < profiles.getCount(); ++i) {
      const auto profile = profiles.getProfile(i);
      if (!profile->is<ob::VideoStreamProfile>()) continue;
      const auto video = profile->as<ob::VideoStreamProfile>();
      if (!out.empty()) out += ", ";
      out += std::to_string(video->getWidth()) + "x" +
             std::to_string(video->getHeight()) + "@" +
             std::to_string(video->getFps()) + " " +
             ob::TypeHelper::convertOBFormatTypeToString(video->getFormat());
    }
  } catch (const std::exception&) {
    out += out.empty() ? "(none it could list)" : ", ... (the rest unreadable)";
  }
  return out;
}

// A camera that reports its image mirrored, flipped or rotated delivers pixels
// its intrinsics do not describe, and nothing downstream can tell. Refused
// rather than undone, since undoing it would mean writing the camera's
// settings.
Status check_orientation(ob::Device& device, const std::string& who) {
  struct BoolProp {
    OBPropertyID id;
    const char* name;
  };
  const BoolProp bool_props[] = {
      {OB_PROP_DEPTH_MIRROR_BOOL, "depth mirror"},
      {OB_PROP_DEPTH_FLIP_BOOL, "depth flip"},
      {OB_PROP_COLOR_MIRROR_BOOL, "colour mirror"},
      {OB_PROP_COLOR_FLIP_BOOL, "colour flip"},
  };
  for (const BoolProp& prop : bool_props) {
    if (device.isPropertySupported(prop.id, OB_PERMISSION_READ) &&
        device.getBoolProperty(prop.id)) {
      return Status::unsupported(who + " has " + prop.name +
                                 " switched on; its intrinsics describe the "
                                 "unmirrored image. Switch it off.");
    }
  }
  for (const OBPropertyID id :
       {OB_PROP_DEPTH_ROTATE_INT, OB_PROP_COLOR_ROTATE_INT}) {
    if (device.isPropertySupported(id, OB_PERMISSION_READ) &&
        device.getIntProperty(id) != 0) {
      return Status::unsupported(
          who +
          " rotates its image; its intrinsics describe the unrotated one. Set "
          "the rotation to 0.");
    }
  }
  return {};
}

}  // namespace

Status sdk_error(const std::string& who, const std::string& what,
                 const std::exception& e) {
  return Status::io_error(who + ": " + what + ": " + e.what());
}

void configure_sdk_logging() {
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
}

Result<std::vector<std::shared_ptr<ob::Device>>> discover(
    ob::Context& context, const std::vector<std::string>& serials,
    std::uint32_t timeout_ms, const std::string& who) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::vector<std::string> answered;  // every serial any query listed
  std::vector<std::shared_ptr<ob::Device>> found(
      std::max<std::size_t>(serials.size(), 1));
  // Unnamed: the latest list that held the one camera, opened at the end.
  std::shared_ptr<ob::DeviceList> sole;
  for (;;) {
    const auto list = context.queryDeviceList();
    const std::uint32_t count = list->getCount();
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::string serial = or_empty(list->getSerialNumber(i));
      if (std::find(answered.begin(), answered.end(), serial) ==
          answered.end()) {
        answered.push_back(serial);
      }
      for (std::size_t k = 0; k < serials.size(); ++k) {
        if (found[k] == nullptr && serial == serials[k]) {
          found[k] = list->getDevice(i);
        }
      }
    }
    if (serials.empty()) {
      if (answered.size() > 1) {
        return Status::invalid_argument(
            who + ": " + std::to_string(answered.size()) +
            " cameras answered (" + join(answered) + "); name one");
      }
      if (count == 1) sole = list;
    } else if (std::all_of(found.begin(), found.end(),
                           [](const auto& d) { return d != nullptr; })) {
      return found;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      if (sole != nullptr) {
        found[0] = sole->getDevice(0);
        return found;
      }
      std::vector<std::string> missing;
      for (std::size_t k = 0; k < serials.size(); ++k) {
        if (found[k] == nullptr) missing.push_back(serials[k]);
      }
      const std::string wanted =
          serials.empty()
              ? std::string("no camera")
              : "camera" + std::string(missing.size() > 1 ? "s " : " ") +
                    join(missing) + " not";
      return Status::not_found(who + ": " + wanted + " found within " +
                               std::to_string(timeout_ms) + " ms" +
                               (answered.empty()
                                    ? std::string(" (none answered)")
                                    : " (answered: " + join(answered) + ")"));
    }
    std::this_thread::sleep_for(kDiscoveryRetry);
  }
}

void Mailbox::on_frameset(std::shared_ptr<ob::FrameSet> frameset) {
  // Runs on the SDK's thread; it must never throw back into the SDK.
  if (frameset == nullptr) return;
  std::lock_guard<std::mutex> lock(mutex);
  pending.push_back(std::move(frameset));
  while (pending.size() > depth) {
    pending.pop_front();
    dropped.fetch_add(1, std::memory_order_relaxed);
  }
  received.fetch_add(1, std::memory_order_relaxed);
}

void Mailbox::on_devices_changed(const std::string& serial,
                                 const ob::DeviceList& removed) {
  try {
    for (std::uint32_t i = 0; i < removed.getCount(); ++i) {
      if (serial == or_empty(removed.getSerialNumber(i))) {
        std::lock_guard<std::mutex> lock(mutex);
        fault = "camera " + serial + " disconnected; reopen it to use it again";
        disconnected.store(true, std::memory_order_release);
        return;
      }
    }
  } catch (...) {
    // A list the SDK cannot read out is not evidence of a disconnect.
  }
}

Result<std::unique_ptr<CameraStream>> CameraStream::create(
    std::shared_ptr<ob::Context> context, std::shared_ptr<ob::Device> device,
    const OrbbecStreamOptions& streams, const Mat4f& cam_to_world,
    const std::string& who) {
  std::unique_ptr<CameraStream> s(new CameraStream());
  s->context_ = std::move(context);
  s->device_ = std::move(device);
  try {
    const auto device_info = s->device_->getDeviceInfo();
    s->info_.name = or_empty(device_info->getName());
    s->info_.serial = or_empty(device_info->getSerialNumber());
    s->info_.firmware_version = or_empty(device_info->getFirmwareVersion());
    s->info_.connection_type = or_empty(device_info->getConnectionType());
    s->info_.ip_address = or_empty(device_info->getIpAddress());
    s->who_ = who + ": camera " + s->info_.serial;
    if (s->device_->getSupportedMultiDeviceSyncModeBitmap() != 0) {
      s->sync_settings_ =
          sync_settings_from(s->device_->getMultiDeviceSyncConfig());
    }
    s->info_.sync_mode = s->sync_settings_.mode;
    if (s->info_.sync_mode == OrbbecSyncMode::SoftwareTriggering) {
      return Status::unsupported(
          s->who_ +
          " is in software-triggering mode, which captures only when the host "
          "sends a trigger, and this driver sends none. Set another sync mode "
          "on the camera.");
    }

    VR_TRY(check_orientation(*s->device_, s->who_));

    s->pipeline_ = std::make_shared<ob::Pipeline>(s->device_);
    const auto depth_modes =
        s->pipeline_->getStreamProfileList(OB_SENSOR_DEPTH);
    const auto color_modes =
        s->pipeline_->getStreamProfileList(OB_SENSOR_COLOR);
    // getVideoStreamProfile throws when nothing matches; the SDK's message
    // does not say what would have, so each lookup is caught on its own.
    try {
      s->depth_profile_ = depth_modes->getVideoStreamProfile(
          static_cast<int>(streams.depth_width),
          static_cast<int>(streams.depth_height), OB_FORMAT_Y16,
          static_cast<int>(streams.fps));
    } catch (const ob::Error&) {
      return Status::unsupported(s->who_ + " has no depth mode " +
                                 std::to_string(streams.depth_width) + "x" +
                                 std::to_string(streams.depth_height) + "@" +
                                 std::to_string(streams.fps) +
                                 " Y16; it offers " + list_modes(*depth_modes));
    }
    try {
      // RGB: the wire carries MJPG either way, decoded on the SDK's thread.
      // TODO(sensor): stream H.265, which the SDK passes through undecoded, so
      // 4K on three cameras needs a decoder of ours (the 2026-09-26 decision).
      s->color_profile_ = color_modes->getVideoStreamProfile(
          static_cast<int>(streams.color_width),
          static_cast<int>(streams.color_height), OB_FORMAT_RGB,
          static_cast<int>(streams.fps));
    } catch (const ob::Error&) {
      return Status::unsupported(s->who_ + " has no colour mode " +
                                 std::to_string(streams.color_width) + "x" +
                                 std::to_string(streams.color_height) + "@" +
                                 std::to_string(streams.fps) +
                                 " RGB; it offers " + list_modes(*color_modes));
    }

    // The pinhole camera of the undistorted colour image: undistortion keeps
    // the stream's intrinsics and drops only the distortion, and registration
    // re-projects depth with exactly these. process() checks both claims on
    // the first pair of every start.
    const OBCameraIntrinsic intrinsic =
        s->color_profile_->as<ob::VideoStreamProfile>()->getIntrinsic();
    VR_ASSIGN(s->color_camera_, color_camera_from(intrinsic, cam_to_world));
    if (s->color_camera_.width != streams.color_width ||
        s->color_camera_.height != streams.color_height) {
      return Status::io_error(
          s->who_ + " reports colour intrinsics for a " +
          std::to_string(s->color_camera_.width) + "x" +
          std::to_string(s->color_camera_.height) + " image, not the " +
          std::to_string(streams.color_width) + "x" +
          std::to_string(streams.color_height) + " mode it opened");
    }
    VR_ASSIGN(s->depth_camera_,
              depth_from_registered_color(
                  s->color_camera_, streams.color_width, streams.color_height,
                  streams.min_depth, streams.max_depth));

    s->undistort_color_ =
        std::make_shared<ob::UnDistortionFilter>(OB_STREAM_COLOR);
    s->align_to_color_ = std::make_shared<ob::Align>(OB_STREAM_COLOR);
    // Registered depth at the colour image's full size -- the frame's one
    // resolution. Set rather than left to the SDK's default.
    s->align_to_color_->setMatchTargetResolution(true);

    s->device_callback_id_ = s->context_->registerDeviceChangedCallback(
        [mailbox = s->mailbox_, serial = s->info_.serial](
            std::shared_ptr<ob::DeviceList> removed,
            std::shared_ptr<ob::DeviceList> /*added*/) {
          if (removed != nullptr) mailbox->on_devices_changed(serial, *removed);
        });
    s->device_callback_registered_ = true;
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(s->who_.empty() ? who : s->who_, "opening the camera", e);
  }
  return s;
}

CameraStream::~CameraStream() {
  stop();
  if (device_callback_registered_) {
    try {
      context_->unregisterDeviceChangedCallback(device_callback_id_);
    } catch (...) {
      // Nothing to do at teardown; the context goes next regardless.
    }
  }
}

OrbbecCaptureStats CameraStream::stats() const noexcept {
  OrbbecCaptureStats s;
  s.received = mailbox_->received.load(std::memory_order_relaxed);
  s.delivered = delivered_;
  s.dropped = mailbox_->dropped.load(std::memory_order_relaxed) + discarded_;
  s.failed = failed_;
  return s;
}

Status CameraStream::start() {
  Mailbox& box = *mailbox_;
  {
    std::lock_guard<std::mutex> lock(box.mutex);
    // Checked before `running`: a camera that went away while streaming is
    // not "already started".
    if (!box.fault.empty()) return Status::io_error(box.fault);
    if (running_) return {};
    box.pending.clear();
  }
  box.received = 0;
  box.dropped = 0;
  delivered_ = 0;
  failed_ = 0;
  discarded_ = 0;
  failed_in_a_row_ = 0;
  first_pair_checked_ = false;
  try {
    auto config = std::make_shared<ob::Config>();
    config->enableStream(depth_profile_);
    config->enableStream(color_profile_);
    // Only pairs: a frame set missing either half is never handed over.
    config->setFrameAggregateOutputMode(
        OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    // Pair depth with the colour frame nearest it in time.
    pipeline_->enableFrameSync();
    pipeline_->start(config,
                     [mailbox = mailbox_](std::shared_ptr<ob::FrameSet> fs) {
                       try {
                         mailbox->on_frameset(std::move(fs));
                       } catch (...) {
                         // Never throw into the SDK's thread.
                       }
                     });
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(who_, "starting", e);
  }
  running_ = true;
  return {};
}

void CameraStream::stop() noexcept {
  if (running_) {
    try {
      pipeline_->stop();
    } catch (...) {
      // A camera that already went away cannot be stopped, and there is no
      // one to tell; a late frame lands in the mailbox, not in this object.
    }
    running_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(mailbox_->mutex);
    mailbox_->pending.clear();
  }
  depth_metres_.clear();
  color_packed_.clear();
}

void CameraStream::set_queue_depth(std::size_t depth) {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  mailbox_->depth = depth > 0 ? depth : 1;
}

Result<std::shared_ptr<ob::FrameSet>> CameraStream::take() {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  if (!mailbox_->fault.empty()) return Status::io_error(mailbox_->fault);
  if (!running_ || mailbox_->pending.empty()) {
    return std::shared_ptr<ob::FrameSet>{};
  }
  std::shared_ptr<ob::FrameSet> pair = std::move(mailbox_->pending.back());
  mailbox_->dropped.fetch_add(mailbox_->pending.size() - 1,
                              std::memory_order_relaxed);
  mailbox_->pending.clear();
  return pair;
}

Status CameraStream::take_all(std::vector<std::shared_ptr<ob::FrameSet>>* out) {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  if (!mailbox_->fault.empty()) return Status::io_error(mailbox_->fault);
  if (!running_) return {};
  for (auto& pair : mailbox_->pending) out->push_back(std::move(pair));
  mailbox_->pending.clear();
  return {};
}

void CameraStream::discard() noexcept { ++discarded_; }

Status CameraStream::apply_sync(const OrbbecSyncSettings& settings) {
  if (running_) {
    return Status::invalid_argument(who_ +
                                    ": sync settings are written before start");
  }
  try {
    device_->setMultiDeviceSyncConfig(sdk_sync_config(settings));
    sync_settings_ = sync_settings_from(device_->getMultiDeviceSyncConfig());
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(who_, "writing its sync settings", e);
  }
  info_.sync_mode = sync_settings_.mode;
  return {};
}

void CameraStream::withdraw() noexcept {
  --delivered_;
  ++discarded_;
}

std::uint64_t CameraStream::timestamp_us(const ob::FrameSet& pair) noexcept {
  try {
    const auto depth = pair.getDepthFrame();
    return depth != nullptr ? depth->getTimeStampUs() : 0;
  } catch (...) {
    return 0;
  }
}

Result<std::optional<CapturedFrame>> CameraStream::process(
    const std::shared_ptr<ob::FrameSet>& frameset) {
  // The pair is delivered, or counted as failed by one of these two. `refuse`
  // is for a pair that contradicts the stream create() negotiated -- every
  // pair after it would too; `skip` for one the SDK failed on, which the next
  // may not be.
  const auto refuse = [this](Status why) {
    ++failed_;
    return why;
  };
  const auto skip =
      [this](const std::string& why) -> Result<std::optional<CapturedFrame>> {
    ++failed_;
    if (++failed_in_a_row_ < kMaxFailedPairsInARow) {
      return ICameraCapture::no_frame();
    }
    return Status::io_error(who_ + ": " + std::to_string(failed_in_a_row_) +
                            " pairs in a row could not be processed; the "
                            "last: " +
                            why);
  };

  const std::uint32_t width = color_camera_.width;
  const std::uint32_t height = color_camera_.height;
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  std::uint64_t timestamp_us = 0;
  try {
    // Undistort colour, then register depth to it: the SDK's registration
    // ignores the colour lens, so the order is what puts both on one pinhole
    // camera (the 2026-09-26 decision).
    // TODO(sensor): the same steps as GPU kernels, keeping the frame on the
    // device through fusion -- which needs device-resident fusion entry
    // points, or the upload is merely moved. And measure the camera's own
    // registration (ALIGN_D2C_HW_MODE; the 2026-09-26 decision).
    const auto undistorted = undistort_color_->process(frameset);
    if (undistorted == nullptr) {
      return skip("colour undistortion produced no frame");
    }
    const auto registered = align_to_color_->process(undistorted);
    if (registered == nullptr) {
      return skip("depth registration produced no frame");
    }
    const auto pair = registered->as<ob::FrameSet>();
    const auto depth = pair->getDepthFrame();
    const auto color = pair->getColorFrame();
    if (depth == nullptr || color == nullptr) {
      return skip("a processed pair is missing its depth or colour frame");
    }
    if (depth->getWidth() != width || depth->getHeight() != height ||
        color->getWidth() != width || color->getHeight() != height) {
      return refuse(Status::io_error(
          who_ + ": processed pair is depth " +
          std::to_string(depth->getWidth()) + "x" +
          std::to_string(depth->getHeight()) + ", colour " +
          std::to_string(color->getWidth()) + "x" +
          std::to_string(color->getHeight()) + "; expected both " +
          std::to_string(width) + "x" + std::to_string(height)));
    }
    if (depth->getFormat() != OB_FORMAT_Y16 ||
        depth->getDataSize() < pixels * sizeof(std::uint16_t)) {
      return refuse(Status::io_error(
          who_ + ": registered depth is not a full Y16 image"));
    }
    if (color->getFormat() != OB_FORMAT_RGB ||
        color->getDataSize() < pixels * 3) {
      return refuse(Status::io_error(
          who_ + ": undistorted colour is not a full RGB image"));
    }
    if (!first_pair_checked_) {
      // The frames are stamped with the camera computed at create; hold the
      // SDK to it once per start. Registration must have re-projected into
      // these intrinsics, and undistortion must have kept them and dropped
      // only the distortion.
      const auto depth_video =
          depth->getStreamProfile()->as<ob::VideoStreamProfile>();
      const auto color_video =
          color->getStreamProfile()->as<ob::VideoStreamProfile>();
      if (!same_pinhole(depth_video->getIntrinsic(), color_camera_,
                        kIntrinsicsTolerance)) {
        return refuse(Status::io_error(
            who_ +
            ": registered depth reports intrinsics other than the colour "
            "camera's; the frame would be unprojected wrongly"));
      }
      if (!same_pinhole(color_video->getIntrinsic(), color_camera_,
                        kIntrinsicsTolerance)) {
        return refuse(Status::io_error(
            who_ +
            ": undistorted colour reports intrinsics other than the ones "
            "read at open; the frame would be projected wrongly"));
      }
      // `!= 0` refuses a NaN coefficient too.
      const OBCameraDistortion d = color_video->getDistortion();
      if (d.k1 != 0.0f || d.k2 != 0.0f || d.k3 != 0.0f || d.k4 != 0.0f ||
          d.k5 != 0.0f || d.k6 != 0.0f || d.p1 != 0.0f || d.p2 != 0.0f) {
        return refuse(Status::io_error(
            who_ + ": undistorted colour still reports lens distortion"));
      }
      first_pair_checked_ = true;
    }
    const float value_scale = depth->getValueScale();
    if (!std::isfinite(value_scale) || !(value_scale > 0.0f)) {
      return refuse(Status::io_error(who_ +
                                     ": depth frame reports value scale " +
                                     std::to_string(value_scale)));
    }
    depth_metres_.resize(pixels);
    color_packed_.resize(pixels);
    depth_to_metres(reinterpret_cast<const std::uint16_t*>(depth->getData()),
                    pixels, value_scale, depth_metres_.data());
    pack_rgb(color->getData(), pixels, color_packed_.data());
    // The camera's hardware clock -- on one clock with the rest of a rig's
    // cameras once OrbbecRig has the SDK sync them.
    timestamp_us = depth->getTimeStampUs();
  } catch (const std::exception& e) {  // ob::Error is one
    return skip(std::string("the SDK failed on it: ") + e.what());
  }

  CapturedFrame frame{};
  frame.depth = depth_metres_.data();
  frame.color = color_packed_.data();
  frame.depth_camera = depth_camera_;
  frame.color_camera = color_camera_;
  // The camera's colour is ordinary 8-bit sRGB: the canonical form, declared
  // by leaving the default.
  frame.timestamp_ns = timestamp_us * 1000;
  failed_in_a_row_ = 0;
  ++delivered_;
  return ICameraCapture::some_frame(frame);
}

}  // namespace volumetric_kit::recon::sensor::orbbec
