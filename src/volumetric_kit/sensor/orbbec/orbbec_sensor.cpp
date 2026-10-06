// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "camera_stream.hpp"
#include "frame_conversion.hpp"

namespace volumetric_kit::recon::sensor {

namespace {

SyncRole role_of(OrbbecSyncMode mode) noexcept {
  if (mode == OrbbecSyncMode::Primary) return SyncRole::Primary;
  return waits_for_primary(mode) ? SyncRole::Secondary : SyncRole::FreeRun;
}

}  // namespace

struct OrbbecSensor::Impl {
  // Declared first so it is destroyed last: the stream's SDK objects belong
  // to it.
  std::shared_ptr<ob::Context> context;
  std::unique_ptr<orbbec::CameraStream> stream;
  SensorInfo info;
  std::uint32_t clock_sync_interval_ms = 0;
};

OrbbecSensor::OrbbecSensor(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
OrbbecSensor::OrbbecSensor(OrbbecSensor&& other) noexcept = default;
OrbbecSensor& OrbbecSensor::operator=(OrbbecSensor&& other) noexcept = default;
OrbbecSensor::~OrbbecSensor() = default;

core::Result<OrbbecSensor> OrbbecSensor::open(const Options& options) {
  if (!camera::check_rigid(options.color_to_world).ok()) {
    return core::Status::invalid_argument(
        "OrbbecSensor: color_to_world must be rigid");
  }
  // The capture's checks, for these streams opened raw.
  OrbbecCapture::Options capture;
  static_cast<OrbbecStreamOptions&>(capture) = options;
  capture.raw = true;
  capture.serial = options.serial;
  capture.discovery_timeout_ms = options.discovery_timeout_ms;
  capture.cam_to_world = Mat4f(options.color_to_world);
  capture.configure_sdk_logging = options.configure_sdk_logging;
  VKC_TRY(orbbec::validate(capture));
  VKC_TRY(orbbec::check_color_codec(capture, "OrbbecSensor"));

  auto impl = std::make_unique<Impl>();
  impl->clock_sync_interval_ms = options.clock_sync_interval_ms;
  try {
    if (options.configure_sdk_logging) orbbec::configure_sdk_logging();
    impl->context = std::make_shared<ob::Context>();
    impl->context->enableNetDeviceEnumeration(true);
    std::vector<std::string> serials;
    if (!options.serial.empty()) serials.push_back(options.serial);
    VKC_ASSIGN(const auto devices,
               orbbec::discover(*impl->context, serials,
                                options.discovery_timeout_ms, "OrbbecSensor"));
    VKC_ASSIGN(impl->stream,
               orbbec::CameraStream::create(impl->context, devices.front(),
                                            capture, options.color_to_world,
                                            options.configure_sdk_logging,
                                            "OrbbecSensor"));
  } catch (const std::exception& e) {  // ob::Error is one
    return orbbec::sdk_error("OrbbecSensor", "opening the camera", e);
  }
  const orbbec::CameraStream& stream = *impl->stream;
  SensorInfo& info = impl->info;
  info.id = stream.info().serial;
  info.model = stream.info().name;
  info.color = stream.raw_color_camera();
  info.depth = stream.raw_depth_camera();
  info.depth_to_color = stream.raw_depth_to_color();
  info.role = role_of(stream.info().sync_mode);
  info.clock = options.clock_sync_interval_ms > 0 ? ClockDomain::Host
                                                  : ClockDomain::Device;
  info.pose = PoseSource::Fixed;
  info.fps = stream.fps();
  return OrbbecSensor(std::move(impl));
}

const OrbbecDeviceInfo& OrbbecSensor::device_info() const noexcept {
  static const OrbbecDeviceInfo kEmpty{};
  return impl_ != nullptr ? impl_->stream->info() : kEmpty;
}

OrbbecCaptureStats OrbbecSensor::orbbec_stats() const noexcept {
  return impl_ != nullptr ? impl_->stream->stats() : OrbbecCaptureStats{};
}

const SensorInfo& OrbbecSensor::info() const noexcept {
  static const SensorInfo kEmpty{};
  return impl_ != nullptr ? impl_->info : kEmpty;
}

core::Status OrbbecSensor::set_queue_depth(std::size_t frames) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "OrbbecSensor: set_queue_depth on a moved-from sensor");
  }
  if (frames == 0) {
    return core::Status::invalid_argument(
        "OrbbecSensor: a queue holds at least one frame");
  }
  if (impl_->stream->running()) {
    return core::Status::invalid_argument(
        "OrbbecSensor: the queue depth is set before start");
  }
  impl_->stream->set_queue_depth(frames);
  return {};
}

core::Status OrbbecSensor::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "OrbbecSensor: start on a moved-from sensor");
  }
  if (impl_->clock_sync_interval_ms > 0 && !impl_->stream->running()) {
    try {
      // Before the camera starts, so its first frame is on the host's clock.
      impl_->context->enableDeviceClockSync(impl_->clock_sync_interval_ms);
    } catch (const std::exception& e) {  // ob::Error is one
      return orbbec::sdk_error("OrbbecSensor", "syncing the camera's clock", e);
    }
  }
  return impl_->stream->start();
}

void OrbbecSensor::stop() noexcept {
  if (impl_ != nullptr) impl_->stream->stop();
}

core::Result<std::optional<RgbdFrame>> OrbbecSensor::poll() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "OrbbecSensor: poll on a moved-from sensor");
  }
  VKC_ASSIGN(std::shared_ptr<ob::FrameSet> pair, impl_->stream->take());
  if (pair == nullptr) return no_frame();
  return impl_->stream->process_raw(pair);
}

core::Status OrbbecSensor::drain(std::vector<RgbdFrame>* out) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "OrbbecSensor: drain on a moved-from sensor");
  }
  std::vector<std::shared_ptr<ob::FrameSet>> pairs;
  VKC_TRY(impl_->stream->take_all(&pairs));
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    core::Result<std::optional<RgbdFrame>> frame =
        impl_->stream->process_raw(pairs[i]);
    if (!frame.ok()) {
      // The rest were taken and will not be handed out.
      for (std::size_t j = i + 1; j < pairs.size(); ++j) {
        impl_->stream->discard();
      }
      return frame.status();
    }
    if (frame.value()) out->push_back(std::move(*frame.value()));
  }
  return {};
}

bool OrbbecSensor::exhausted() const noexcept {
  return impl_ == nullptr || impl_->stream->disconnected();
}

SensorStats OrbbecSensor::stats() const noexcept {
  const OrbbecCaptureStats s = orbbec_stats();
  return SensorStats{s.received, s.delivered, s.dropped, s.failed + s.lost};
}

}  // namespace volumetric_kit::recon::sensor
