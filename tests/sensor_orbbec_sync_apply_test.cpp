// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Exercise OrbbecSensor::open with a camera stub: matching settings must not
// be written, write failures must propagate, and SDK-normalized settings must
// not be reported as applied unchanged. No camera or SDK context is opened.
// The executable compiles orbbec_sensor.cpp so the stub works with static
// and shared libraries without adding a test seam to the public API.

#include <cstdio>
#include <string>

#include "camera_stream.hpp"
#include "frame_conversion.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

struct FakeCamera {
  sensor::OrbbecSyncSettings stored;
  vkc::Status write_status;
  int writes = 0;
};
FakeCamera fake;

sensor::OrbbecSensor::Options primary_options() {
  sensor::OrbbecSensor::Options options;
  options.serial = "TEST-SYNC-CAMERA";
  options.sync.emplace();
  options.sync->mode = sensor::OrbbecSyncMode::Primary;
  options.sync->trigger_out_enable = true;
  options.apply_sync = true;
  return options;
}

}  // namespace

namespace volumetric_kit::recon::sensor::orbbec {

core::Result<std::unique_ptr<CameraStream>> CameraStream::create(
    std::shared_ptr<ob::Context>, std::shared_ptr<ob::Device>,
    const OrbbecStreamOptions& streams, const camera::Mat4d&, bool,
    const std::string&) {
  std::unique_ptr<CameraStream> stream(new CameraStream());
  stream->info_.serial = "TEST-SYNC-CAMERA";
  stream->sync_settings_ = fake.stored;
  stream->info_.sync_mode = fake.stored.mode;
  stream->fps_ = streams.fps;
  return stream;
}

core::Result<std::unique_ptr<CameraStream>> open_camera(
    const std::string&, std::uint32_t, bool, const OrbbecStreamOptions& streams,
    const camera::Mat4d& pose, const std::string& who) {
  return CameraStream::create({}, {}, streams, pose, false, who);
}

CameraStream::~CameraStream() = default;

core::Status CameraStream::apply_sync(const OrbbecSyncSettings& requested) {
  ++fake.writes;
  VKC_TRY(fake.write_status);
  auto applied = sdk_sync_config(requested);
  // SDK 2.10.6's DeviceSyncConfiguratorOldProtocol::setSyncConfig, used by
  // Femto Mega: these fields are normalized before the write AND cache update.
  if (applied.syncMode == OB_MULTI_DEVICE_SYNC_MODE_PRIMARY) {
    applied.triggerOutEnable = true;
    applied.triggerOutDelayUs = 0;
  }
  fake.stored = sync_settings_from(applied);
  sync_settings_ = fake.stored;
  info_.sync_mode = sync_settings_.mode;
  return {};
}

// The remaining methods are needed to link the public sensor implementation;
// the tests only open sensors, never start or poll them.
core::Status CameraStream::use_host_clock() { return {}; }
core::Status CameraStream::sync_clock_to_host() { return {}; }
core::Status CameraStream::start() { return {}; }
void CameraStream::stop() noexcept {}
void CameraStream::set_queue_depth(std::size_t) {}
OrbbecStreamStats CameraStream::stats() const noexcept { return {}; }
void CameraStream::discard() noexcept {}
core::Result<std::shared_ptr<ob::FrameSet>> CameraStream::take() {
  return std::shared_ptr<ob::FrameSet>{};
}
core::Status CameraStream::take_all(
    std::vector<std::shared_ptr<ob::FrameSet>>*) {
  return {};
}
core::Result<std::optional<RgbdFrame>> CameraStream::read(
    const std::shared_ptr<ob::FrameSet>&) {
  return std::optional<RgbdFrame>{};
}

}  // namespace volumetric_kit::recon::sensor::orbbec

namespace {

int test_matching_settings() {
  for (const bool apply : {false, true}) {
    fake = {};
    fake.stored.mode = sensor::OrbbecSyncMode::SecondarySynced;
    fake.stored.depth_delay_us = 160;
    auto options = primary_options();
    options.sync = fake.stored;
    options.sync->mode = sensor::OrbbecSyncMode::Secondary;
    options.apply_sync = apply;
    const auto opened = sensor::OrbbecSensor::open(options);
    CHECK(opened.ok());
    CHECK(opened->info().role == sensor::SyncRole::Secondary);
    CHECK(fake.writes == 0);  // including the equivalent SDK role spelling
  }
  return 0;
}

int test_write() {
  fake = {};
  auto options = primary_options();
  options.apply_sync = false;
  const auto refused = sensor::OrbbecSensor::open(options);
  CHECK(refused.status().domain() == vkc::Status::Code::Unsupported);
  CHECK(refused.status().message().find(options.serial) != std::string::npos);
  CHECK(refused.status().message().find("syncMode") != std::string::npos);
  CHECK(fake.writes == 0);

  // A write can repair software triggering; the role is read after the write.
  fake.stored.mode = sensor::OrbbecSyncMode::SoftwareTriggering;
  options.apply_sync = true;
  const auto opened = sensor::OrbbecSensor::open(options);
  CHECK(opened.ok());
  CHECK(opened->info().role == sensor::SyncRole::Primary);
  CHECK(fake.writes == 1);

  const auto unchanged = sensor::OrbbecSensor::open(options);
  CHECK(unchanged.ok());
  CHECK(fake.writes == 1);
  return 0;
}

int test_normalized_settings() {
  fake = {};
  auto options = primary_options();
  options.sync->trigger_out_enable = false;
  options.sync->trigger_out_delay_us = 250;
  const auto opened = sensor::OrbbecSensor::open(options);
  CHECK(fake.writes == 1);
  CHECK(opened.status().domain() == vkc::Status::Code::Unsupported);
  const std::string& message = opened.status().message();
  CHECK(message.find(options.serial) != std::string::npos);
  CHECK(message.find("triggerOutEnable is true, configured false") !=
        std::string::npos);
  CHECK(message.find("triggerOutDelayUs is 0, configured 250") !=
        std::string::npos);
  CHECK(message.find("set apply_sync") == std::string::npos);
  return 0;
}

int test_write_failure() {
  fake = {};
  fake.write_status = vkc::Status::io_error("test sync write failed");
  const auto opened = sensor::OrbbecSensor::open(primary_options());
  CHECK(fake.writes == 1);
  CHECK(opened.status().domain() == vkc::Status::Code::IoError);
  CHECK(opened.status().message() == "test sync write failed");
  return 0;
}

}  // namespace

int main() {
  if (test_matching_settings() != 0) return 1;
  if (test_write() != 0) return 1;
  if (test_normalized_settings() != 0) return 1;
  if (test_write_failure() != 0) return 1;
  std::puts("orbbec sync apply tests passed");
  return 0;
}
