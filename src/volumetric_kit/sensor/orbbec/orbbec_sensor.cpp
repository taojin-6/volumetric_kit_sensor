// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "camera_stream.hpp"
#include "frame_conversion.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::recon::sensor {

namespace {

SyncRole role_of(OrbbecSyncMode mode) noexcept {
  if (mode == OrbbecSyncMode::Primary) return SyncRole::Primary;
  return waits_for_primary(mode) ? SyncRole::Secondary : SyncRole::FreeRun;
}

}  // namespace

struct OrbbecSensor::Impl {
  std::unique_ptr<orbbec::CameraStream> stream;
  SensorInfo info;
  bool sync_clock_to_host = false;
};

OrbbecSensor::OrbbecSensor(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
OrbbecSensor::OrbbecSensor(OrbbecSensor&& other) noexcept = default;
OrbbecSensor& OrbbecSensor::operator=(OrbbecSensor&& other) noexcept = default;
OrbbecSensor::~OrbbecSensor() = default;

core::Result<OrbbecSensor> OrbbecSensor::open(const Options& options) {
  VKC_TRY(orbbec::validate_streams(options, "OrbbecSensor"));
  const core::Status rigid = camera::check_rigid(options.color_to_world);
  if (!rigid.ok()) {
    return core::Status::invalid_argument("OrbbecSensor: color_to_world: " +
                                          rigid.message());
  }

  if (options.apply_sync && !options.sync) {
    return core::Status::invalid_argument(
        "OrbbecSensor: apply_sync needs sync, the settings to write");
  }

  auto impl = std::make_unique<Impl>();
  impl->sync_clock_to_host = options.sync_clock_to_host;
  VKC_ASSIGN(impl->stream,
             orbbec::open_camera(options.serial, options.discovery_timeout_ms,
                                 options.configure_sdk_logging, options,
                                 options.color_to_world, "OrbbecSensor"));
  orbbec::CameraStream& stream = *impl->stream;
  const std::string who = "OrbbecSensor: camera " + stream.info().serial;
  // Before any write to the camera's flash, so a refusal never follows one.
  if (options.sync_clock_to_host) VKC_TRY(stream.use_host_clock());
  if (options.sync) {
    // Written only where they differ: they persist in the camera's flash.
    const std::vector<std::string> differences =
        orbbec::sync_differences(*options.sync, stream.sync_settings());
    if (!differences.empty() && !options.apply_sync) {
      std::string fields;
      for (const std::string& d : differences) {
        fields += (fields.empty() ? "" : ", ") + d;
      }
      return core::Status::unsupported(
          who + " differs from its sync configuration: " + fields +
          "; set apply_sync to write them");
    }
    if (!differences.empty()) VKC_TRY(stream.apply_sync(*options.sync));
  }
  // The role as the camera has it now, after any write.
  if (stream.info().sync_mode == OrbbecSyncMode::SoftwareTriggering) {
    return core::Status::unsupported(
        who +
        " is in software-triggering mode, which captures only when the host "
        "sends a trigger, and this driver sends none. Set another sync mode "
        "on the camera.");
  }
  if (stream.info().sync_mode == OrbbecSyncMode::Other) {
    // Its role in a rig is unknown, and an array could not order its start.
    return core::Status::unsupported(
        who +
        " is in a sync mode this driver does not know; set it to free-run, "
        "standalone, primary or secondary");
  }
  SensorInfo& info = impl->info;
  info.id = stream.info().serial;
  info.model = stream.info().name;
  info.color = stream.color_camera();
  info.depth = stream.depth_camera();
  info.depth_to_color = stream.depth_to_color();
  info.role = role_of(stream.info().sync_mode);
  info.clock =
      options.sync_clock_to_host ? ClockDomain::Host : ClockDomain::Device;
  info.pose = PoseSource::Fixed;
  info.fps = stream.fps();
  return OrbbecSensor(std::move(impl));
}

const OrbbecDeviceInfo& OrbbecSensor::device_info() const noexcept {
  static const OrbbecDeviceInfo kEmpty{};
  return impl_ != nullptr ? impl_->stream->info() : kEmpty;
}

OrbbecStreamStats OrbbecSensor::orbbec_stats() const noexcept {
  return impl_ != nullptr ? impl_->stream->stats() : OrbbecStreamStats{};
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
  // Once, before it streams, as Orbbec's multi-camera recipe does: the SDK
  // re-fits its global timestamps then, so the first frames are mapped.
  if (impl_->sync_clock_to_host && !impl_->stream->running()) {
    VKC_TRY(impl_->stream->sync_clock_to_host());
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
  return impl_->stream->read(pair);
}

core::Status OrbbecSensor::drain(std::vector<RgbdFrame>* out) {
  VKC_CHECK(out != nullptr, "OrbbecSensor::drain needs somewhere to put them");
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "OrbbecSensor: drain on a moved-from sensor");
  }
  std::vector<std::shared_ptr<ob::FrameSet>> pairs;
  VKC_TRY(impl_->stream->take_all(&pairs));
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    core::Result<std::optional<RgbdFrame>> frame =
        impl_->stream->read(pairs[i]);
    if (!frame.ok()) {
      // The frames before it stay handed out; the rest were taken and will
      // not be.
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
  const OrbbecStreamStats s = orbbec_stats();
  return SensorStats{s.received, s.delivered, s.dropped, s.failed + s.lost};
}

}  // namespace volumetric_kit::recon::sensor
