// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <librealsense2/rs.hpp>

#include "conversion.hpp"
#include "frame_queue.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor {

namespace {

constexpr const char* kWho = "RealSenseSensor";

// librealsense reports every failure by throwing; each becomes a Status.
core::Status sdk_failure(const std::string& what, const std::exception& e) {
  std::string message = std::string(kWho) + ": " + what + ": " + e.what();
  if (const auto* rs = dynamic_cast<const rs2::error*>(&e)) {
    message += std::string(" (") + rs->get_failed_function() + ")";
  }
  return core::Status::io_error(message);
}

std::string info_of(const rs2::device& device, rs2_camera_info key) {
  return device.supports(key) ? device.get_info(key) : std::string();
}

std::uint64_t now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// The stream profile of `format` at the size and rate, or none.
std::optional<rs2::video_stream_profile> find_profile(
    const rs2::sensor& sensor, rs2_stream stream, rs2_format format,
    std::uint32_t width, std::uint32_t height, std::uint32_t fps) {
  for (const rs2::stream_profile& p : sensor.get_stream_profiles()) {
    if (p.stream_type() != stream || p.format() != format ||
        p.fps() != static_cast<int>(fps) ||
        !p.is<rs2::video_stream_profile>()) {
      continue;
    }
    const auto v = p.as<rs2::video_stream_profile>();
    if (v.width() == static_cast<int>(width) &&
        v.height() == static_cast<int>(height)) {
      return v;
    }
  }
  return std::nullopt;
}

// The modes `sensor` offers for `stream` in `format`, for an error.
std::string modes_of(const rs2::sensor& sensor, rs2_stream stream,
                     rs2_format format) {
  std::string modes;
  for (const rs2::stream_profile& p : sensor.get_stream_profiles()) {
    if (p.stream_type() != stream || p.format() != format ||
        !p.is<rs2::video_stream_profile>()) {
      continue;
    }
    const auto v = p.as<rs2::video_stream_profile>();
    modes += (modes.empty() ? "" : ", ") + std::to_string(v.width()) + "x" +
             std::to_string(v.height()) + "@" + std::to_string(p.fps());
  }
  return modes.empty() ? "none" : modes;
}

// The colour camera's manual exposure written and read back, or auto
// exposure. Every control is checked before any is written.
core::Result<std::optional<camera::ColorControls>> apply_color_controls(
    rs2::sensor& color, const std::optional<camera::ColorControls>& controls,
    const std::string& who) {
  if (!controls) {
    if (color.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
      color.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f);
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
  if (read.exposure != controls->exposure || read.gain != controls->gain) {
    return core::Status::unsupported(
        who + "'s colour controls read back as exposure " +
        std::to_string(read.exposure) + ", gain " + std::to_string(read.gain) +
        ", not as written");
  }
  return std::optional<camera::ColorControls>(read);
}

// A frameset the SDK delivered, and when.
struct Arrived {
  rs2::frameset frames;
  std::uint64_t arrival_ns = 0;
};

// What the SDK's threads write and the polling thread reads, held by the
// callbacks too, since neither is guaranteed gone when the sensor is.
struct Shared {
  realsense::FrameQueue<Arrived> queue;
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> failed{0};
  std::atomic<bool> disconnected{false};
};

}  // namespace

struct RealSenseSensor::Impl {
  rs2::context context;
  rs2::device device;
  rs2::config config;
  std::optional<rs2::pipeline> pipeline;
  std::shared_ptr<Shared> shared = std::make_shared<Shared>();
  RealSenseDeviceInfo device_info;
  SensorInfo info;
  realsense::FrameSetup setup;
  std::uint32_t fps = 0;
  std::string who;
  bool running = false;
  std::atomic<std::uint64_t> delivered{0};

  ~Impl() { stop(); }

  void stop() noexcept {
    if (running && pipeline) {
      try {
        pipeline->stop();
      } catch (...) {  // NOLINT(bugprone-empty-catch): the camera may be gone
      }
    }
    running = false;
    shared->queue.clear();
  }

  // A frameset taken from the queue as the frame it is.
  core::Result<std::optional<RgbdFrame>> read(Arrived arrived) try {
    const rs2::depth_frame depth = arrived.frames.get_depth_frame();
    const rs2::video_frame color = arrived.frames.get_color_frame();
    if (depth.get_frame_timestamp_domain() ==
        RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK) {
      return core::Status::io_error(
          who + ": a frame is stamped on the camera's clock, not the host's");
    }
    realsense::FrameView view;
    view.depth = depth.get_data();
    view.depth_width = static_cast<std::uint32_t>(depth.get_width());
    view.depth_height = static_cast<std::uint32_t>(depth.get_height());
    view.depth_stride = static_cast<std::size_t>(depth.get_stride_in_bytes());
    view.color = color.get_data();
    view.color_width = static_cast<std::uint32_t>(color.get_width());
    view.color_height = static_cast<std::uint32_t>(color.get_height());
    view.color_stride = static_cast<std::size_t>(color.get_stride_in_bytes());
    view.timestamp_ms = depth.get_timestamp();
    view.sequence = depth.get_frame_number();
    view.arrival_ns = arrived.arrival_ns;
    auto pixels =
        std::make_shared<const rs2::frameset>(std::move(arrived.frames));
    VKC_ASSIGN(RgbdFrame frame,
               realsense::rgbd_frame(view, setup, std::move(pixels), who));
    delivered.fetch_add(1, std::memory_order_relaxed);
    return IRgbdSensor::some_frame(std::move(frame));
  } catch (const std::exception& e) {
    return sdk_failure("reading a frame", e);
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
  // The camera: the named one, or the only one attached.
  const rs2::device_list list = impl->context.query_devices();
  std::vector<rs2::device> found;
  std::string attached;
  for (std::uint32_t i = 0; i < list.size(); ++i) {
    const rs2::device d = list[i];
    const std::string serial = info_of(d, RS2_CAMERA_INFO_SERIAL_NUMBER);
    attached += (attached.empty() ? "" : ", ") +
                info_of(d, RS2_CAMERA_INFO_NAME) + " " + serial;
    if (options.serial.empty() || serial == options.serial) found.push_back(d);
  }
  if (found.empty()) {
    return core::Status::not_found(
        std::string(kWho) + ": " +
        (options.serial.empty()
             ? std::string("no camera is attached")
             : "camera " + options.serial + " is not attached") +
        "; attached: " + (attached.empty() ? "none" : attached));
  }
  if (found.size() > 1) {
    return core::Status::invalid_argument(
        std::string(kWho) + ": more than one camera is attached (" + attached +
        "); name one by its serial");
  }
  impl->device = found.front();
  RealSenseDeviceInfo& about = impl->device_info;
  about.name = info_of(impl->device, RS2_CAMERA_INFO_NAME);
  about.serial = info_of(impl->device, RS2_CAMERA_INFO_SERIAL_NUMBER);
  about.firmware_version =
      info_of(impl->device, RS2_CAMERA_INFO_FIRMWARE_VERSION);
  about.usb_type = info_of(impl->device, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);
  impl->who = std::string(kWho) + ": camera " + about.serial;
  const std::string& who = impl->who;
  if (about.usb_type.empty() || about.usb_type[0] == '2') {
    return core::Status::unsupported(
        who + " is on a USB " +
        (about.usb_type.empty() ? std::string("(unknown)") : about.usb_type) +
        " link, which cannot carry these streams; plug it into a USB 3 port");
  }

  // Its depth and colour cameras, and the requested modes.
  std::optional<rs2::depth_sensor> depth_sensor;
  std::optional<rs2::color_sensor> color_sensor;
  for (rs2::sensor& s : impl->device.query_sensors()) {
    if (!depth_sensor && s.is<rs2::depth_sensor>()) {
      depth_sensor = s.as<rs2::depth_sensor>();
    } else if (!color_sensor && s.is<rs2::color_sensor>()) {
      color_sensor = s.as<rs2::color_sensor>();
    }
  }
  if (!depth_sensor || !color_sensor) {
    return core::Status::unsupported(who + " (" + about.name +
                                     ") has no depth or no colour camera");
  }
  const auto depth_profile =
      find_profile(*depth_sensor, RS2_STREAM_DEPTH, RS2_FORMAT_Z16,
                   options.depth_width, options.depth_height, options.fps);
  if (!depth_profile) {
    return core::Status::unsupported(
        who + " has no Z16 depth mode " + std::to_string(options.depth_width) +
        "x" + std::to_string(options.depth_height) + "@" +
        std::to_string(options.fps) + "; it offers " +
        modes_of(*depth_sensor, RS2_STREAM_DEPTH, RS2_FORMAT_Z16));
  }
  const auto color_profile =
      find_profile(*color_sensor, RS2_STREAM_COLOR, RS2_FORMAT_YUYV,
                   options.color_width, options.color_height, options.fps);
  if (!color_profile) {
    return core::Status::unsupported(
        who + " has no YUYV colour mode " +
        std::to_string(options.color_width) + "x" +
        std::to_string(options.color_height) + "@" +
        std::to_string(options.fps) + "; it offers " +
        modes_of(*color_sensor, RS2_STREAM_COLOR, RS2_FORMAT_YUYV));
  }

  // The factory calibration, as the GPU pass reads it.
  realsense::FrameSetup& setup = impl->setup;
  VKC_ASSIGN(setup.depth_camera,
             realsense::camera_model_from(depth_profile->get_intrinsics(),
                                          false, who));
  VKC_ASSIGN(
      setup.color_camera,
      realsense::camera_model_from(color_profile->get_intrinsics(), true, who));
  VKC_ASSIGN(setup.depth_to_color,
             realsense::transform_from(
                 depth_profile->get_extrinsics_to(*color_profile), who));
  setup.color_to_world = options.color_to_world;
  setup.metres_per_unit = depth_sensor->get_depth_scale();
  setup.min_depth = options.min_depth;
  setup.max_depth = options.max_depth;

  // Timestamps on the host's clock, the projector, the colour exposure: last,
  // so a camera refused above gets no write.
  for (rs2::sensor* s : {static_cast<rs2::sensor*>(&*depth_sensor),
                         static_cast<rs2::sensor*>(&*color_sensor)}) {
    if (!s->supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
      return core::Status::unsupported(
          who +
          " has no global timestamps, which put its frames on the "
          "host's clock");
    }
  }
  if (depth_sensor->supports(RS2_OPTION_EMITTER_ENABLED)) {
    depth_sensor->set_option(RS2_OPTION_EMITTER_ENABLED,
                             options.emitter ? 1.0f : 0.0f);
  } else if (options.emitter) {
    return core::Status::unsupported(who + " has no projector to switch on");
  }
  depth_sensor->set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
  color_sensor->set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
  SensorInfo& info = impl->info;
  VKC_ASSIGN(info.color_controls,
             apply_color_controls(*color_sensor, options.color_controls, who));

  // The streams, which start() opens.
  impl->config.enable_device(about.serial);
  impl->config.enable_stream(RS2_STREAM_DEPTH, depth_profile->stream_index(),
                             static_cast<int>(options.depth_width),
                             static_cast<int>(options.depth_height),
                             RS2_FORMAT_Z16, static_cast<int>(options.fps));
  impl->config.enable_stream(RS2_STREAM_COLOR, color_profile->stream_index(),
                             static_cast<int>(options.color_width),
                             static_cast<int>(options.color_height),
                             RS2_FORMAT_YUYV, static_cast<int>(options.fps));
  impl->pipeline.emplace(impl->context);
  if (!impl->config.can_resolve(*impl->pipeline)) {
    return core::Status::unsupported(
        who + " cannot stream depth and colour together at these modes");
  }
  // A camera that goes away fails every later poll.
  impl->context.set_devices_changed_callback(
      [shared = impl->shared, device = impl->device,
       who](rs2::event_information& event) {
        if (!event.was_removed(device)) return;
        shared->disconnected.store(true, std::memory_order_release);
        shared->queue.fail(who + " disconnected");
      });

  impl->fps = options.fps;
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
  return sdk_failure("opening the camera", e);
}

const RealSenseDeviceInfo& RealSenseSensor::device_info() const noexcept {
  static const RealSenseDeviceInfo kEmpty{};
  return impl_ != nullptr ? impl_->device_info : kEmpty;
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
  if (frames == 0) {
    return core::Status::invalid_argument(
        "RealSenseSensor: a queue holds at least one frame");
  }
  if (impl_->running) {
    return core::Status::invalid_argument(
        "RealSenseSensor: the queue depth is set before start");
  }
  impl_->shared->queue.set_depth(frames);
  return {};
}

core::Status RealSenseSensor::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: start on a moved-from sensor");
  }
  if (impl_->running) return {};
  Shared& shared = *impl_->shared;
  if (shared.disconnected.load(std::memory_order_acquire)) {
    return core::Status::io_error(impl_->who + " disconnected");
  }
  shared.received.store(0, std::memory_order_relaxed);
  shared.failed.store(0, std::memory_order_relaxed);
  impl_->delivered.store(0, std::memory_order_relaxed);
  shared.queue.clear();
  try {
    // Framesets without depth or colour, or whose colour is of another
    // moment, as the syncer's are while the colour camera starts, are counted
    // failed and never queued.
    impl_->pipeline->start(impl_->config, [shared = impl_->shared,
                                           fps = impl_->fps](rs2::frame frame) {
      const std::uint64_t arrival = now_ns();
      shared->received.fetch_add(1, std::memory_order_relaxed);
      try {
        const auto frames = frame.as<rs2::frameset>();
        const rs2::depth_frame depth =
            frames ? frames.get_depth_frame() : rs2::depth_frame(rs2::frame());
        const rs2::video_frame color =
            frames ? frames.get_color_frame() : rs2::video_frame(rs2::frame());
        if (!depth || !color ||
            !realsense::same_moment(depth.get_timestamp(),
                                    color.get_timestamp(), fps)) {
          shared->failed.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        shared->queue.push(Arrived{frames, arrival});
      } catch (...) {
        shared->failed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  } catch (const std::exception& e) {
    return sdk_failure("starting the streams", e);
  }
  impl_->running = true;
  return {};
}

void RealSenseSensor::stop() noexcept {
  if (impl_ != nullptr) impl_->stop();
}

core::Result<std::optional<RgbdFrame>> RealSenseSensor::poll() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: poll on a moved-from sensor");
  }
  VKC_ASSIGN(std::optional<Arrived> arrived,
             impl_->shared->queue.take_newest());
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
  std::vector<Arrived> taken;
  VKC_TRY(impl_->shared->queue.take_all(&taken));
  for (std::size_t i = 0; i < taken.size(); ++i) {
    core::Result<std::optional<RgbdFrame>> frame =
        impl_->read(std::move(taken[i]));
    if (!frame.ok()) {
      // The frames before it stay handed out; the rest were taken and will
      // not be.
      for (std::size_t j = i + 1; j < taken.size(); ++j) {
        impl_->shared->queue.discard();
      }
      return frame.status();
    }
    if (frame.value()) out->push_back(std::move(*frame.value()));
  }
  return {};
}

bool RealSenseSensor::exhausted() const noexcept {
  return impl_ == nullptr ||
         impl_->shared->disconnected.load(std::memory_order_acquire);
}

SensorStats RealSenseSensor::stats() const noexcept {
  if (impl_ == nullptr) return {};
  const Shared& shared = *impl_->shared;
  SensorStats s;
  s.received = shared.received.load(std::memory_order_relaxed);
  s.delivered = impl_->delivered.load(std::memory_order_relaxed);
  s.dropped = shared.queue.dropped();
  s.failed = shared.failed.load(std::memory_order_relaxed);
  return s;
}

core::Status RealSenseSensor::set_frame_notify(std::function<void()> notify) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "RealSenseSensor: set_frame_notify on a moved-from sensor");
  }
  if (impl_->running) {
    return core::Status::invalid_argument(
        "RealSenseSensor: the frame notification is set before start");
  }
  impl_->shared->queue.set_notify(std::move(notify));
  return {};
}

}  // namespace volumetric_kit::sensor
