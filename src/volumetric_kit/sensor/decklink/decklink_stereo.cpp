// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/decklink/decklink_stereo.hpp"

#include <string>
#include <utility>

#include "capture.hpp"
#include "driver.hpp"
#include "ports.hpp"
#include "volumetric_kit/core/base/log.hpp"

namespace volumetric_kit::sensor {

struct DecklinkStereo::Impl {
  std::unique_ptr<decklink::Driver> driver;
};

DecklinkStereo::DecklinkStereo(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
DecklinkStereo::DecklinkStereo(DecklinkStereo&& other) noexcept = default;
DecklinkStereo& DecklinkStereo::operator=(DecklinkStereo&& other) noexcept =
    default;
DecklinkStereo::~DecklinkStereo() = default;

core::Result<DecklinkStereo> DecklinkStereo::open(const Options& options,
                                                  const core::Device& device,
                                                  core::Allocator& allocator) {
  decklink::Config config;
  config.left_id = options.left.id;
  config.right_id = options.right.id;
  config.left_input = options.left.input;
  config.right_input = options.right.input;
  config.calibration = options.calibration;
  config.width = options.width;
  config.height = options.height;
  config.fps = options.fps;
  config.max_skew_ns = options.max_skew_ns;
  config.lock_timeout = options.lock_timeout;
  VKC_TRY(decklink::check_config(config));
  const std::optional<decklink::Mode> mode =
      decklink::mode_of(options.width, options.height, options.fps);
  if (!mode) {
    return core::Status::invalid_argument(
        "DecklinkStereo: no mode of " + std::to_string(options.width) + "x" +
        std::to_string(options.height) + " at " + std::to_string(options.fps) +
        " fps: the cameras send 2160p at 24, 25 or 30");
  }

  decklink::CaptureConfig capture;
  capture.names = {decklink::input_name(config.left_input, config.left_id),
                   decklink::input_name(config.right_input, config.right_id)};
  capture.sub_device = {decklink::sub_device_of(config.left_input),
                        decklink::sub_device_of(config.right_input)};
  capture.mode = mode->code;
  auto impl = std::make_unique<Impl>();
  VKC_ASSIGN(
      impl->driver,
      decklink::Driver::create(
          config, device, allocator,
          [&capture](
              const std::array<std::shared_ptr<decklink::PagePool>, 2>& pages) {
            capture.pages = pages;
            return decklink::open_capture(capture);
          }));
  return DecklinkStereo(std::move(impl));
}

const StereoInfo& DecklinkStereo::info() const noexcept {
  static const StereoInfo kEmpty{};
  return impl_ != nullptr ? impl_->driver->info() : kEmpty;
}

core::Status DecklinkStereo::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "DecklinkStereo: start on a moved-from camera");
  }
  return impl_->driver->start();
}

void DecklinkStereo::stop() noexcept {
  if (impl_ == nullptr) return;
  decklink::Driver& driver = *impl_->driver;
  driver.stop();
  // The pools' high marks, for the hand check (how many buffers the SDK
  // asks for an input, and how many pictures the consumers hold).
  try {
    core::log_message(
        core::LogLevel::Info, "sensor",
        "DecklinkStereo: pages made " + std::to_string(driver.pages_made(0)) +
            " and " + std::to_string(driver.pages_made(1)) + " of " +
            std::to_string(decklink::PagePool::kMaxBuffers) +
            " an input; device pictures made " +
            std::to_string(driver.pictures_made()) + " of " +
            std::to_string(decklink::DevicePictures::kMaxBuffers));
  } catch (...) {  // NOLINT(bugprone-empty-catch): a log line, not a result
  }
}

core::Result<std::optional<StereoFrame>> DecklinkStereo::poll() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "DecklinkStereo: poll on a moved-from camera");
  }
  return impl_->driver->poll();
}

SensorStats DecklinkStereo::stats() const noexcept {
  return impl_ != nullptr ? impl_->driver->stats() : SensorStats{};
}

core::Status DecklinkStereo::set_frame_notify(std::function<void()> notify) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "DecklinkStereo: set_frame_notify on a moved-from camera");
  }
  return impl_->driver->set_frame_notify(std::move(notify));
}

}  // namespace volumetric_kit::sensor
