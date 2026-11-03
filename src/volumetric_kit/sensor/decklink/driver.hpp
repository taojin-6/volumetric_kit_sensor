// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): DecklinkStereo's behaviour over any Capture --
// the option checks, the pairing, the upload, start's wait for the first
// pair, a mode change failing poll, and the counts -- with no SDK type.
// DecklinkStereo hands it the SDK's capture (capture.cpp); a test hands it
// one that writes pages as the card would, so all of it runs on every leg.

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "capture.hpp"
#include "pairing.hpp"
#include "upload.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor::decklink {

// DecklinkStereo::Options, field for field.
struct Config {
  std::string left_id, right_id;        // the cameras' names
  std::string left_input, right_input;  // "SDI 1".."SDI 4"
  std::optional<StereoCalibration> calibration;
  std::uint32_t width = 3840, height = 2160, fps = 30;
  std::uint64_t max_skew_ns = 16'666'000;
  std::chrono::milliseconds lock_timeout{5000};
};

// The refusals DecklinkStereo::open makes before any device
// (InvalidArgument): a connector not on the card, or one twice; a mode
// other than 2160p 24, 25 or 30; a calibration whose lenses are not at the
// mode's size; max_skew_ns 0 or past half a frame; a lock_timeout not
// positive.
core::Status check_config(const Config& config);

// An input as messages name it: "SDI 2 (bm-1)".
std::string input_name(const std::string& input, const std::string& id);

class Driver final : public IStereoSensor {
 public:
  using OpenCapture = std::function<core::Result<std::unique_ptr<Capture>>(
      const std::array<std::shared_ptr<PagePool>, 2>& pages)>;

  // The pages and device pictures for `config`'s picture size (any even
  // width: check_config holds DecklinkStereo to its modes), then the capture
  // over those pages. Unsupported where the device did not enable
  // VK_EXT_external_memory_host; whatever `open` returns.
  static core::Result<std::unique_ptr<Driver>> create(
      const Config& config, const core::Device& device,
      core::Allocator& allocator, const OpenCapture& open);

  Driver(const Driver&) = delete;
  Driver& operator=(const Driver&) = delete;
  Driver(Driver&&) = delete;
  Driver& operator=(Driver&&) = delete;
  ~Driver() override;

  const StereoInfo& info() const noexcept override;
  // Both inputs started; then a wait, to lock_timeout from the call, for
  // the first pair on the device. Unsupported for a camera at another mode,
  // naming it and the fix; IoError when no pair came, naming the input that
  // sent nothing or the cameras' measured offset.
  core::Status start() override;
  void stop() noexcept override;
  // The newest pair; Unsupported once a camera changed mode; IoError once
  // the upload failed.
  core::Result<std::optional<StereoFrame>> poll() override;
  SensorStats stats() const noexcept override;
  core::Status set_frame_notify(std::function<void()> notify) override;

  // The pools' high marks and what frames hold now, for the hand check and
  // the tests.
  std::size_t pages_made(int input) const;
  std::size_t pages_in_use(int input) const;
  std::size_t pictures_made() const;
  std::size_t pictures_in_use() const;

 private:
  class Sink;
  Driver(const Config& config, std::array<std::shared_ptr<PagePool>, 2> pages,
         std::shared_ptr<DevicePictures> pictures, const core::Device& device,
         core::Allocator& allocator);
  void arrived(Arrival frame);
  void mode_changed(int input, const std::string& detected);
  void ready();  // the uploader published a pair or failed

  Config config_;
  StereoInfo info_;
  std::array<std::string, 2> names_;
  std::array<std::shared_ptr<PagePool>, 2> pages_;
  std::shared_ptr<DevicePictures> pictures_;
  Pairing pairing_;
  Uploader uploader_;
  std::unique_ptr<Sink> sink_;
  std::unique_ptr<Capture> capture_;
  bool started_ = false;
  std::uint64_t delivered_ = 0;

  mutable std::mutex mutex_;
  std::condition_variable woken_;
  core::Status mode_fault_;       // guarded by mutex_
  std::function<void()> notify_;  // guarded by mutex_
};

}  // namespace volumetric_kit::sensor::decklink
