// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "camera.hpp"

#include <chrono>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor::realsense {

namespace {

std::string info_of(const rs2::device& device, rs2_camera_info key) {
  return device.supports(key) ? device.get_info(key) : std::string();
}

// The colour camera's manual exposure written and read back, or auto
// exposure. Every control is checked before any is written.
core::Result<std::optional<camera::ColorControls>> apply_color_controls(
    const rs2::sensor& color,
    const std::optional<camera::ColorControls>& controls,
    const std::string& who) {
  if (!controls) {
    // Auto exposure, and auto white balance on, as with a manual exposure.
    for (const rs2_option o : {RS2_OPTION_ENABLE_AUTO_EXPOSURE,
                               RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE}) {
      if (color.supports(o)) color.set_option(o, 1.0f);
    }
    return std::optional<camera::ColorControls>{};
  }
  const rs2_option needed[] = {RS2_OPTION_ENABLE_AUTO_EXPOSURE,
                               RS2_OPTION_EXPOSURE, RS2_OPTION_GAIN,
                               RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE};
  for (const rs2_option o : needed) {
    if (!color.supports(o)) {
      return core::Status::unsupported(who + "'s colour camera has no " +
                                       rs2_option_to_string(o) + " control");
    }
  }
  const std::pair<rs2_option, std::uint32_t> values[] = {
      {RS2_OPTION_EXPOSURE, controls->exposure},
      {RS2_OPTION_GAIN, controls->gain}};
  for (const auto& [option, value] : values) {
    const rs2::option_range range = color.get_option_range(option);
    const auto v = static_cast<float>(value);
    if (v < range.min || v > range.max) {
      return core::Status::unsupported(
          who + "'s colour " + rs2_option_to_string(option) + " " +
          std::to_string(value) + " is outside its range [" +
          std::to_string(range.min) + ", " + std::to_string(range.max) + "]");
    }
  }
  color.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0f);
  color.set_option(RS2_OPTION_EXPOSURE, static_cast<float>(controls->exposure));
  color.set_option(RS2_OPTION_GAIN, static_cast<float>(controls->gain));
  color.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1.0f);
  camera::ColorControls read;
  read.exposure =
      static_cast<std::uint32_t>(color.get_option(RS2_OPTION_EXPOSURE));
  read.gain = static_cast<std::uint32_t>(color.get_option(RS2_OPTION_GAIN));
  if (read.exposure != controls->exposure || read.gain != controls->gain ||
      color.get_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE) != 0.0f ||
      color.get_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE) != 1.0f) {
    return core::Status::unsupported(
        who + "'s colour controls read back as exposure " +
        std::to_string(read.exposure) + ", gain " + std::to_string(read.gain) +
        ", auto exposure " +
        std::to_string(color.get_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) +
        ", auto white balance " +
        std::to_string(color.get_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) +
        ", not as written");
  }
  return std::optional<camera::ColorControls>(read);
}

// The most modes a driver opens: two infrared cameras and the colour camera.
constexpr std::size_t kMaxModes = 3;

}  // namespace

core::Status sdk_failure(const std::string& who, const std::string& what,
                         const std::exception& e) {
  std::string message = who + ": " + what + ": " + e.what();
  if (const auto* rs = dynamic_cast<const rs2::error*>(&e)) {
    message += std::string(" (") + rs->get_failed_function() + ")";
  }
  return core::Status::io_error(message);
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
  // The camera: the named one, or the only one attached.
  const rs2::device_list list = camera->context_.query_devices();
  std::vector<rs2::device> found;
  std::string attached;
  for (std::uint32_t i = 0; i < list.size(); ++i) {
    const rs2::device d = list[i];
    const std::string s = info_of(d, RS2_CAMERA_INFO_SERIAL_NUMBER);
    attached += (attached.empty() ? "" : ", ") +
                info_of(d, RS2_CAMERA_INFO_NAME) + " " + s;
    if (serial.empty() || s == serial) found.push_back(d);
  }
  if (found.empty()) {
    return core::Status::not_found(
        std::string(kind) + ": " +
        (serial.empty() ? std::string("no camera is attached")
                        : "camera " + serial + " is not attached") +
        "; attached: " + (attached.empty() ? "none" : attached));
  }
  if (found.size() > 1) {
    return core::Status::invalid_argument(
        std::string(kind) + ": more than one camera is attached (" + attached +
        "); name one by its serial");
  }
  camera->device_ = found.front();
  RealSenseDeviceInfo& about = camera->about_;
  about.name = info_of(camera->device_, RS2_CAMERA_INFO_NAME);
  about.serial = info_of(camera->device_, RS2_CAMERA_INFO_SERIAL_NUMBER);
  about.firmware_version =
      info_of(camera->device_, RS2_CAMERA_INFO_FIRMWARE_VERSION);
  about.usb_type =
      info_of(camera->device_, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);
  camera->who_ = std::string(kind) + ": camera " + about.serial;
  const std::string& who = camera->who_;
  if (about.usb_type.empty() || about.usb_type[0] == '2') {
    return core::Status::unsupported(
        who + " is on a USB " +
        (about.usb_type.empty() ? std::string("(unknown)") : about.usb_type) +
        " link, which cannot carry these streams; plug it into a USB 3 port");
  }

  // Its stereo module (depth and the infrared cameras) and colour camera.
  for (rs2::sensor& s : camera->device_.query_sensors()) {
    if (!camera->stereo_ && s.is<rs2::depth_sensor>()) {
      camera->stereo_ = s.as<rs2::depth_sensor>();
    } else if (!camera->color_ && s.is<rs2::color_sensor>()) {
      camera->color_ = s.as<rs2::color_sensor>();
    }
  }
  if (!camera->stereo_ || !camera->color_) {
    return core::Status::unsupported(who + " (" + about.name +
                                     ") has no depth or no colour camera");
  }
  return camera;
} catch (const std::exception& e) {
  return sdk_failure(kind, "opening the camera", e);
}

Camera::~Camera() {
  stop();
  // The callback holds the device, and librealsense's context waits
  // forever in its destructor for a callback that does (measured on the
  // lab D455 with 2.59.1): replaced first, which lets the device go.
  try {
    context_.set_devices_changed_callback([](rs2::event_information&) {});
  } catch (...) {  // NOLINT(bugprone-empty-catch): nothing left to stop
  }
}

core::Result<rs2::video_stream_profile> Camera::mode(
    const rs2::sensor& module, rs2_stream stream, int index, rs2_format format,
    std::uint32_t width, std::uint32_t height, std::uint32_t fps,
    const char* what) const {
  std::string modes;
  for (const rs2::stream_profile& p : module.get_stream_profiles()) {
    if (p.stream_type() != stream || p.format() != format ||
        (index >= 0 && p.stream_index() != index) ||
        !p.is<rs2::video_stream_profile>()) {
      continue;
    }
    const auto v = p.as<rs2::video_stream_profile>();
    if (v.width() == static_cast<int>(width) &&
        v.height() == static_cast<int>(height) &&
        p.fps() == static_cast<int>(fps)) {
      return v;
    }
    modes += (modes.empty() ? "" : ", ") + std::to_string(v.width()) + "x" +
             std::to_string(v.height()) + "@" + std::to_string(p.fps());
  }
  return core::Status::unsupported(
      who_ + " has no " + what + " mode " + std::to_string(width) + "x" +
      std::to_string(height) + "@" + std::to_string(fps) + "; it offers " +
      (modes.empty() ? std::string("none") : modes));
}

core::Result<std::optional<camera::ColorControls>> Camera::configure(
    bool emitter, const std::optional<camera::ColorControls>& controls) {
  for (const rs2::sensor* s : {static_cast<const rs2::sensor*>(&*stereo_),
                               static_cast<const rs2::sensor*>(&*color_)}) {
    if (!s->supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
      return core::Status::unsupported(
          who_ +
          " has no global timestamps, which put its frames on the "
          "host's clock");
    }
  }
  if (stereo_->supports(RS2_OPTION_EMITTER_ENABLED)) {
    stereo_->set_option(RS2_OPTION_EMITTER_ENABLED, emitter ? 1.0f : 0.0f);
  } else if (emitter) {
    return core::Status::unsupported(who_ + " has no projector to switch on");
  }
  return apply_color_controls(*color_, controls, who_);
}

core::Status Camera::set_streams(
    const std::vector<rs2::video_stream_profile>& profiles, std::uint32_t fps) {
  VKC_CHECK(!profiles.empty() && profiles.size() <= kMaxModes,
            "a RealSense driver opens one to three modes");
  config_.enable_device(about_.serial);
  for (const rs2::video_stream_profile& p : profiles) {
    config_.enable_stream(p.stream_type(), p.stream_index(), p.width(),
                          p.height(), p.format(), p.fps());
  }
  pipeline_.emplace(context_);
  if (!config_.can_resolve(*pipeline_)) {
    return core::Status::unsupported(who_ +
                                     " cannot run these modes together");
  }
  fps_ = fps;
  modes_ = profiles.size();
  // A camera that goes away fails every later take.
  context_.set_devices_changed_callback(
      [shared = shared_, device = device_,
       who = who_](rs2::event_information& event) {
        if (!event.was_removed(device)) return;
        shared->disconnected.store(true, std::memory_order_release);
        shared->queue.fail(who + " disconnected");
      });
  return {};
}

core::Status Camera::set_queue_depth(std::size_t frames) {
  if (frames == 0) {
    return core::Status::invalid_argument(
        std::string(kind_) + ": a queue holds at least one frame");
  }
  if (running_) {
    return core::Status::invalid_argument(
        std::string(kind_) + ": the queue depth is set before start");
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

core::Status Camera::start() {
  VKC_CHECK(pipeline_.has_value(), "Camera::start before set_streams");
  // Before the idempotent return: a camera gone while running is not
  // running, whatever the flag says.
  if (disconnected()) return core::Status::io_error(who_ + " disconnected");
  if (running_) return {};
  // The counters restart here, not at a stop, so they read the same after
  // one.
  shared_->received.store(0, std::memory_order_relaxed);
  shared_->failed.store(0, std::memory_order_relaxed);
  delivered_.store(0, std::memory_order_relaxed);
  shared_->queue.clear();
  shared_->queue.reset_dropped();
  try {
    // Framesets that miss a mode, or whose frames are of other moments, as
    // while the D455 restarts its depth for its first 1.2 s and the pipeline
    // matches its last depth frame with newer colour, are counted failed and
    // never queued.
    const rs2::pipeline_profile running = pipeline_->start(
        config_, [shared = shared_, fps = fps_,
                  modes = modes_](rs2::frame frame) {
          const std::uint64_t arrival = now_ns();
          shared->received.fetch_add(1, std::memory_order_relaxed);
          try {
            const auto frames = frame.as<rs2::frameset>();
            const std::size_t count = frames ? frames.size() : 0;
            double stamps[kMaxModes] = {};
            for (std::size_t i = 0; i < count && i < kMaxModes; ++i) {
              stamps[i] = frames[i].get_timestamp();
            }
            if (count > kMaxModes || !together(stamps, count, modes, fps)) {
              shared->failed.fetch_add(1, std::memory_order_release);
              return;
            }
            shared->queue.push(Arrived{frames, arrival});
          } catch (...) {
            shared->failed.fetch_add(1, std::memory_order_release);
          }
        });
    // Global time belongs to the device instance that runs, which the
    // pipeline opens itself, so it is set there; librealsense's default is on.
    for (rs2::sensor& s : running.get_device().query_sensors()) {
      if (s.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
        s.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
      }
    }
  } catch (const std::exception& e) {
    return sdk_failure(kind_, "starting the streams", e);
  }
  running_ = true;
  return {};
}

void Camera::stop() noexcept {
  if (running_ && pipeline_) {
    try {
      pipeline_->stop();
    } catch (...) {  // NOLINT(bugprone-empty-catch): the camera may be gone
    }
  }
  running_ = false;
  shared_->queue.clear();
}

SensorStats Camera::stats() const noexcept {
  // A frame is counted received before anything else, so received is read
  // last: the others never sum past it.
  SensorStats s;
  s.delivered = delivered_.load(std::memory_order_acquire);
  s.dropped = shared_->queue.dropped();
  s.failed = shared_->failed.load(std::memory_order_acquire);
  s.received = shared_->received.load(std::memory_order_acquire);
  return s;
}

}  // namespace volumetric_kit::sensor::realsense
