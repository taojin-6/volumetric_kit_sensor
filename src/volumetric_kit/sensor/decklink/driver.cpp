// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "driver.hpp"

#include <cstdio>
#include <string>
#include <utility>

#include "ports.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"

namespace volumetric_kit::sensor::decklink {

namespace {

core::Status refuse(const std::string& why) {
  return core::Status::invalid_argument("DecklinkStereo: " + why);
}

std::string ms(std::int64_t ns) {
  char text[32];
  std::snprintf(text, sizeof text, "%.2f", static_cast<double>(ns) / 1e6);
  return text;
}

// The mode's name in messages: the SDK's for the driver's modes.
std::string mode_name(const Config& c) {
  if (const std::optional<Mode> mode = mode_of(c.width, c.height, c.fps)) {
    return mode->name;
  }
  return std::to_string(c.width) + "x" + std::to_string(c.height) + "p" +
         std::to_string(c.fps);
}

}  // namespace

std::string input_name(const std::string& input, const std::string& id) {
  return input + " (" + id + ")";
}

core::Status check_config(const Config& c) {
  for (const std::string* input : {&c.left_input, &c.right_input}) {
    if (sub_device_of(*input) < 0) {
      return refuse("no connector \"" + *input +
                    "\" on the card: use \"SDI 1\" to \"SDI 4\"");
    }
  }
  if (c.left_input == c.right_input) {
    return refuse("both cameras on " + c.left_input);
  }
  if (c.width == 0 || c.height == 0 || c.width % 2 != 0 || c.fps == 0) {
    return refuse("a mode of " + std::to_string(c.width) + "x" +
                  std::to_string(c.height) + " at " + std::to_string(c.fps) +
                  " fps");
  }
  if (c.calibration) {
    const StereoCalibration& cal = *c.calibration;
    for (const camera::CameraModel* lens : {&cal.left, &cal.right}) {
      if (lens->size.width != c.width || lens->size.height != c.height) {
        return refuse("a lens calibrated at " +
                      std::to_string(lens->size.width) + "x" +
                      std::to_string(lens->size.height) + ", not the mode's " +
                      std::to_string(c.width) + "x" + std::to_string(c.height));
      }
    }
  }
  const std::uint64_t half_frame = 1'000'000'000ull / (2ull * c.fps);
  if (c.max_skew_ns == 0 || c.max_skew_ns > half_frame) {
    return refuse("max_skew_ns " + std::to_string(c.max_skew_ns) +
                  " is not in 1.." + std::to_string(half_frame) +
                  ", half a frame at " + std::to_string(c.fps) + " fps");
  }
  if (c.lock_timeout.count() <= 0)
    return refuse("lock_timeout is not positive");
  return {};
}

// The capture's calls, handed to the driver.
class Driver::Sink final : public CaptureSink {
 public:
  explicit Sink(Driver& driver) : driver_(driver) {}
  void arrived(Arrival frame) override { driver_.arrived(std::move(frame)); }
  void mode_changed(int input, std::string detected) override {
    driver_.mode_changed(input, detected);
  }

 private:
  Driver& driver_;
};

core::Result<std::unique_ptr<Driver>> Driver::create(const Config& config,
                                                     const core::Device& device,
                                                     core::Allocator& allocator,
                                                     const OpenCapture& open) {
  VKC_TRY(check_config(config));
  std::array<std::shared_ptr<PagePool>, 2> pages;
  for (std::shared_ptr<PagePool>& pool : pages) {
    VKC_ASSIGN(pool, PagePool::create(device, allocator));
  }
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(v210_row_bytes(config.width)) * config.height;
  std::shared_ptr<DevicePictures> pictures =
      DevicePictures::create(allocator, bytes);
  VKC_ASSIGN(std::unique_ptr<Capture> capture, open(pages));
  std::unique_ptr<Driver> driver(
      new Driver(config, pages, std::move(pictures), device, allocator));
  driver->capture_ = std::move(capture);
  return driver;
}

Driver::Driver(const Config& config,
               std::array<std::shared_ptr<PagePool>, 2> pages,
               std::shared_ptr<DevicePictures> pictures,
               const core::Device& device, core::Allocator& allocator)
    : config_(config),
      names_{input_name(config.left_input, config.left_id),
             input_name(config.right_input, config.right_id)},
      pages_(pages),
      pictures_(pictures),
      pairing_(period_ns(config.fps), config.max_skew_ns),
      uploader_(
          device, allocator, std::move(pages), std::move(pictures),
          Picture{config.width, config.height, v210_row_bytes(config.width)}),
      sink_(std::make_unique<Sink>(*this)) {
  if (config.calibration) info_.calibration = *config.calibration;
  info_.id = config.left_id;
  info_.model = "DeckLink stereo pair (left " + config.left_input + ", right " +
                config.right_input + ")";
  info_.clock = ClockDomain::Host;
  info_.fps = config.fps;
  info_.max_skew_ns = config.max_skew_ns;
}

Driver::~Driver() { stop(); }

const StereoInfo& Driver::info() const noexcept { return info_; }

core::Status Driver::start() {
  if (started_) return {};
  const auto deadline = std::chrono::steady_clock::now() + config_.lock_timeout;
  pairing_.reset();
  delivered_ = 0;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    mode_fault_ = {};
  }
  uploader_.start([this] { ready(); });
  const core::Status started = capture_->start(*sink_, config_.lock_timeout);
  if (!started.ok()) {
    stop();
    return started.with_context("DecklinkStereo");
  }

  core::Status mode;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    woken_.wait_until(lock, deadline, [this] {
      return !mode_fault_.ok() || !uploader_.fault().ok() ||
             uploader_.published() > 0;
    });
    mode = mode_fault_;
  }
  if (!mode.ok()) {
    stop();
    return mode;
  }
  if (const core::Status upload = uploader_.fault(); !upload.ok()) {
    stop();
    return upload;
  }
  if (uploader_.published() == 0) {
    std::string why;
    const bool seen[2] = {pairing_.seen(0), pairing_.seen(1)};
    const std::string waited =
        std::to_string(config_.lock_timeout.count()) + " ms";
    if (!seen[0] || !seen[1]) {
      const std::string silent = !seen[0] && !seen[1]
                                     ? names_[0] + " or " + names_[1]
                                 : !seen[0] ? names_[0]
                                            : names_[1];
      why = "no picture from " + silent + " in " + waited +
            ": check that the camera is on and its SDI OUT is cabled to "
            "that input";
    } else {
      const std::optional<std::int64_t> offset = pairing_.offset_ns();
      why = "no two pictures within max_skew_ns (" +
            std::to_string(config_.max_skew_ns / 1000) + " us) in " + waited +
            ": the cameras are " + ms(offset.value_or(0)) +
            " ms apart (right less left); genlock them or raise max_skew_ns";
    }
    stop();
    return core::Status::io_error("DecklinkStereo: " + why);
  }
  started_ = true;
  return {};
}

void Driver::stop() noexcept {
  if (capture_ != nullptr) capture_->stop();
  uploader_.stop();
  pairing_.release();
  started_ = false;
}

core::Result<std::optional<StereoFrame>> Driver::poll() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!mode_fault_.ok()) return mode_fault_;
  }
  VKC_ASSIGN(std::optional<StereoFrame> frame, uploader_.take());
  if (frame) ++delivered_;
  return frame;
}

SensorStats Driver::stats() const noexcept {
  SensorStats s;
  s.received = pairing_.received();
  s.delivered = delivered_;
  s.dropped = uploader_.dropped();
  s.failed = pairing_.failed() + uploader_.failed();
  return s;
}

core::Status Driver::set_frame_notify(std::function<void()> notify) {
  if (started_) {
    return core::Status::invalid_argument(
        "DecklinkStereo: set_frame_notify once started");
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  notify_ = std::move(notify);
  return {};
}

std::size_t Driver::pages_made(int input) const {
  return pages_[input == 0 ? 0 : 1]->made();
}
std::size_t Driver::pages_in_use(int input) const {
  return pages_[input == 0 ? 0 : 1]->in_use();
}
std::size_t Driver::pictures_made() const { return pictures_->made(); }
std::size_t Driver::pictures_in_use() const { return pictures_->in_use(); }

void Driver::arrived(Arrival frame) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!mode_fault_.ok()) return;  // `frame` goes back to the SDK
  }
  if (std::optional<Pair> pair = pairing_.arrive(std::move(frame))) {
    uploader_.submit(std::move(*pair));
  }
}

void Driver::mode_changed(int input, const std::string& detected) {
  std::function<void()> notify;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (mode_fault_.ok()) {
      mode_fault_ = core::Status::unsupported(
          "DecklinkStereo: " + names_[input == 0 ? 0 : 1] + ": " +
          mode_fix(mode_name(config_), config_.fps, detected));
    }
    notify = notify_;
  }
  woken_.notify_all();
  if (notify) notify();
}

void Driver::ready() {
  std::function<void()> notify;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    notify = notify_;
  }
  woken_.notify_all();
  if (notify) notify();
}

}  // namespace volumetric_kit::sensor::decklink
