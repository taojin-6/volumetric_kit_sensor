// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "orbbec_camera_stub.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
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

// A fake camera's colour properties, as the SDK's device would be read and
// written.
class FakeProperties final
    : public volumetric_kit::sensor::orbbec::CameraProperties {
 public:
  explicit FakeProperties(FakeCamera& camera) : camera_(camera) {}
  bool writable(OBPropertyID id) override { return find(id).second.writable; }
  OBIntPropertyRange int_range(OBPropertyID id) override {
    const FakeProperty& p = use(id).second;
    return OBIntPropertyRange{p.value, p.max, p.min, p.step, p.value};
  }
  std::int32_t get_int(OBPropertyID id) override {
    return use(id).second.value;
  }
  bool get_bool(OBPropertyID id) override { return use(id).second.value != 0; }
  void set_int(OBPropertyID id, std::int32_t value) override {
    write(id, value);
  }
  void set_bool(OBPropertyID id, bool value) override {
    write(id, value ? 1 : 0);
  }

 private:
  std::pair<const char*, FakeProperty&> find(OBPropertyID id) {
    switch (id) {
      case OB_PROP_COLOR_AUTO_EXPOSURE_BOOL:
        return {"auto exposure", camera_.auto_exposure};
      case OB_PROP_COLOR_EXPOSURE_INT:
        return {"exposure", camera_.exposure};
      case OB_PROP_COLOR_GAIN_INT:
        return {"gain", camera_.gain};
      case OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL:
        return {"auto white balance", camera_.auto_white_balance};
      case OB_PROP_COLOR_WHITE_BALANCE_INT:
        return {"white balance", camera_.white_balance};
      default:
        throw std::logic_error("the stub has no property " +
                               std::to_string(static_cast<int>(id)));
    }
  }
  std::pair<const char*, FakeProperty&> use(OBPropertyID id) {
    auto found = find(id);
    if (!found.second.writable) {
      throw std::runtime_error(std::string("the camera has no ") + found.first);
    }
    return found;
  }
  void write(OBPropertyID id, std::int32_t value) {
    auto [name, property] = use(id);
    camera_.color_writes.push_back(std::string(name) + " " +
                                   std::to_string(value));
    if (camera_.failing_write == name) {
      throw std::runtime_error("test colour write failed");
    }
    if (camera_.ignored_write != name) property.value = value;
  }

  FakeCamera& camera_;
};

}  // namespace

}  // namespace orbbec_stub

namespace volumetric_kit::sensor::orbbec {

core::Result<std::unique_ptr<CameraStream>> CameraStream::create(
    std::shared_ptr<ob::Context>, std::shared_ptr<ob::Device>,
    const OrbbecStreamOptions& streams, const camera::Mat4d&, bool,
    const std::string& who) {
  const orbbec_stub::FakeCamera& fake =
      orbbec_stub::cameras.at(orbbec_stub::opening);
  std::unique_ptr<CameraStream> stream(new CameraStream());
  stream->info_.serial = orbbec_stub::opening;
  stream->who_ = who + ": camera " + orbbec_stub::opening;
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

// The driver's own sequence, against the fake's properties.
core::Result<std::optional<camera::ColorControls>>
CameraStream::apply_color_controls(
    const std::optional<camera::ColorControls>& wanted) {
  orbbec_stub::FakeProperties properties(orbbec_stub::cameras.at(info_.serial));
  return write_color_controls(properties, wanted, who_);
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
void CameraStream::set_frame_notify(std::function<void()>) {}
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

}  // namespace volumetric_kit::sensor::orbbec
