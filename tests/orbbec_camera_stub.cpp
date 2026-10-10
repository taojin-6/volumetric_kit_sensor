// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "orbbec_camera_stub.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "camera_stream.hpp"
#include "frame_conversion.hpp"

namespace orbbec_stub {

std::map<std::string, FakeCamera> cameras;
int lookups = 0;
int open_streams = 0;

void reset() {
  cameras.clear();
  lookups = 0;
  open_streams = 0;
}

namespace {

// The serial open_camera found, for CameraStream::create, which is not told.
std::string opening;

}  // namespace

}  // namespace orbbec_stub

namespace volumetric_kit::recon::sensor::orbbec {

core::Result<std::unique_ptr<CameraStream>> CameraStream::create(
    std::shared_ptr<ob::Context>, std::shared_ptr<ob::Device>,
    const OrbbecStreamOptions& streams, const camera::Mat4d&, bool,
    const std::string&) {
  const orbbec_stub::FakeCamera& fake =
      orbbec_stub::cameras.at(orbbec_stub::opening);
  std::unique_ptr<CameraStream> stream(new CameraStream());
  stream->info_.serial = orbbec_stub::opening;
  stream->sync_settings_ = fake.stored;
  stream->info_.sync_mode = fake.stored.mode;
  stream->fps_ = streams.fps;
  ++orbbec_stub::open_streams;
  return stream;
}

core::Result<std::unique_ptr<CameraStream>> open_camera(
    const std::string& serial, std::uint32_t discovery_timeout_ms, bool,
    const OrbbecStreamOptions& streams, const camera::Mat4d& pose,
    const std::string& who) {
  ++orbbec_stub::lookups;
  const auto it = orbbec_stub::cameras.find(serial);
  if (it == orbbec_stub::cameras.end()) {
    return core::Status::not_found(who + ": camera " + serial +
                                   " not found (stub)");
  }
  it->second.discovery_timeout_ms = discovery_timeout_ms;
  orbbec_stub::opening = serial;
  return CameraStream::create({}, {}, streams, pose, false, who);
}

CameraStream::~CameraStream() { --orbbec_stub::open_streams; }

core::Status CameraStream::apply_sync(const OrbbecSyncSettings& requested) {
  orbbec_stub::FakeCamera& fake = orbbec_stub::cameras.at(info_.serial);
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

core::Status CameraStream::use_host_clock() {
  orbbec_stub::cameras.at(info_.serial).host_clock = true;
  return {};
}

// The remaining methods are needed to link the public sensor implementation;
// the tests only open sensors, never start or poll them.
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
