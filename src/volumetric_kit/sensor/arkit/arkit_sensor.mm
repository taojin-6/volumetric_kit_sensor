// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"

#include <dispatch/dispatch.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "conversion.hpp"
#include "driver.hpp"
#include "feeds.hpp"

namespace volumetric_kit::sensor {

const char* to_string(ArkitTracking tracking) noexcept {
  switch (tracking) {
    case ArkitTracking::NotAvailable:
      return "not available";
    case ArkitTracking::Initializing:
      return "initializing";
    case ArkitTracking::Relocalizing:
      return "relocalizing";
    case ArkitTracking::ExcessiveMotion:
      return "excessive motion";
    case ArkitTracking::InsufficientFeatures:
      return "insufficient features";
    case ArkitTracking::Normal:
      return "normal";
  }
  return "unknown";
}

struct ArkitSensor::Impl {
  // The queue ARKit delivers on, the pattern's timer ticks on: serial, so the
  // driver hears one feed thread at a time. Declared first, so it goes last.
  dispatch_queue_t queue = nil;
  std::unique_ptr<arkit::Driver> driver;
  // After the driver, so it goes first: no capture arrives once it has.
  std::unique_ptr<arkit::Feed> feed;
  bool started = false;
};

ArkitSensor::ArkitSensor(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
ArkitSensor::ArkitSensor(ArkitSensor&& other) noexcept = default;
ArkitSensor& ArkitSensor::operator=(ArkitSensor&& other) noexcept {
  if (this != &other) {
    stop();
    impl_ = std::move(other.impl_);
  }
  return *this;
}
ArkitSensor::~ArkitSensor() { stop(); }

core::Result<ArkitSensor> ArkitSensor::open(const Options& options) {
  VKC_TRY(arkit::check_options(options));
  const bool pattern = options.source == ArkitSource::Pattern;
  auto impl = std::make_unique<Impl>();
  VKC_ASSIGN(
      impl->driver,
      arkit::Driver::create(
          options, std::string(pattern ? "ARKit test pattern, " : "ARKit, ") +
                       arkit::device_model()));
  impl->queue = dispatch_queue_create(
      "volumetric_kit.sensor.arkit",
      dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                              QOS_CLASS_USER_INITIATED, 0));
  if (impl->queue == nil) {
    return core::Status::out_of_memory("ArkitSensor: no dispatch queue");
  }
  VKC_ASSIGN(
      impl->feed,
      pattern ? arkit::open_pattern_feed(*impl->driver, options, impl->queue)
              : arkit::open_camera_feed(*impl->driver, options, impl->queue));
  VKC_TRY(impl->driver->wait_opened(
      std::chrono::milliseconds(options.open_timeout_ms)));
  return ArkitSensor(std::move(impl));
}

ArkitStatus ArkitSensor::status() const {
  return impl_ != nullptr ? impl_->driver->status() : ArkitStatus{};
}

const SensorInfo& ArkitSensor::info() const noexcept {
  static const SensorInfo kEmpty{};
  return impl_ != nullptr ? impl_->driver->info() : kEmpty;
}

core::Status ArkitSensor::set_queue_depth(std::size_t frames) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "ArkitSensor: set_queue_depth on a moved-from sensor");
  }
  return impl_->driver->set_queue_depth(frames);
}

core::Status ArkitSensor::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "ArkitSensor: start on a moved-from sensor");
  }
  if (impl_->started) return {};
  VKC_TRY(impl_->driver->begin());
  const core::Status started = impl_->feed->start();
  if (!started.ok()) {
    impl_->driver->end();
    return started;
  }
  impl_->started = true;
  return {};
}

void ArkitSensor::stop() noexcept {
  if (impl_ == nullptr || !impl_->started) return;
  impl_->feed->stop();
  impl_->driver->end();
  impl_->started = false;
}

core::Result<std::optional<RgbdFrame>> ArkitSensor::poll() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "ArkitSensor: poll on a moved-from sensor");
  }
  return impl_->driver->poll();
}

core::Status ArkitSensor::drain(std::vector<RgbdFrame>* out) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "ArkitSensor: drain on a moved-from sensor");
  }
  return impl_->driver->drain(out);
}

core::Result<std::optional<YuvImage>> ArkitSensor::poll_withheld_picture() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "ArkitSensor: poll_withheld_picture on a moved-from sensor");
  }
  return impl_->driver->poll_withheld_picture();
}

bool ArkitSensor::exhausted() const noexcept {
  return impl_ == nullptr || impl_->driver->failed();
}

SensorStats ArkitSensor::stats() const noexcept {
  return impl_ != nullptr ? impl_->driver->stats() : SensorStats{};
}

core::Status ArkitSensor::set_frame_notify(std::function<void()> notify) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "ArkitSensor: set_frame_notify on a moved-from sensor");
  }
  return impl_->driver->set_frame_notify(std::move(notify));
}

}  // namespace volumetric_kit::sensor
