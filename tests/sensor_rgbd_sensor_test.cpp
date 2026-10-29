// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sensor vocabulary: defaults, the no_frame/some_frame returns, frame
// ownership through pixels, a sensor that notifies nothing by default, and
// stable enum names. Host-only.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"

namespace sensor = volumetric_kit::recon::sensor;

namespace {

int test_defaults() {
  const sensor::SensorInfo info;
  CHECK(info.id.empty() && !info.depth && !info.color);
  CHECK(info.role == sensor::SyncRole::FreeRun);
  CHECK(info.clock == sensor::ClockDomain::Device);
  CHECK(info.pose == sensor::PoseSource::Fixed);
  return 0;
}

int test_frame_returns_and_ownership() {
  const auto none = sensor::IRgbdSensor::no_frame();
  CHECK(none.ok() && !none.value());

  sensor::RgbdFrame kept;
  std::weak_ptr<const void> buffer;
  {
    auto pixels = std::make_shared<std::vector<std::uint16_t>>(4, 1007);
    buffer = pixels;
    sensor::RgbdFrame frame;
    frame.depth = pixels->data();
    frame.pixels = pixels;
    frame.sequence = 7;
    const auto returned = sensor::IRgbdSensor::some_frame(std::move(frame));
    CHECK(returned.ok() && returned.value());
    CHECK(returned.value()->sequence == 7);
    CHECK(returned.value()->depth == pixels->data());
    kept = *returned.value();
  }
  // Only the copied frame remains after the source and result let go.
  CHECK(!buffer.expired() && kept.depth[3] == 1007);
  kept = sensor::RgbdFrame{};
  CHECK(buffer.expired());
  return 0;
}

// A sensor that overrides nothing optional notifies nothing: its consumer
// polls it.
int test_notify_default() {
  class Polled final : public sensor::IRgbdSensor {
   public:
    const sensor::SensorInfo& info() const noexcept override { return info_; }
    volumetric_kit::core::Status set_queue_depth(std::size_t) override {
      return {};
    }
    volumetric_kit::core::Status start() override { return {}; }
    void stop() noexcept override {}
    volumetric_kit::core::Result<std::optional<sensor::RgbdFrame>> poll()
        override {
      return no_frame();
    }
    volumetric_kit::core::Status drain(
        std::vector<sensor::RgbdFrame>*) override {
      return {};
    }
    sensor::SensorStats stats() const noexcept override { return {}; }

   private:
    sensor::SensorInfo info_;
  } polled;
  CHECK(polled.set_frame_notify([] {}).domain() ==
        volumetric_kit::core::Status::Code::Unsupported);
  CHECK(polled.set_frame_notify({}).domain() ==
        volumetric_kit::core::Status::Code::Unsupported);
  return 0;
}

int test_names() {
  CHECK(std::strcmp(sensor::to_string(sensor::SyncRole::Primary), "primary") ==
        0);
  CHECK(std::strcmp(sensor::to_string(sensor::SyncRole::Secondary),
                    "secondary") == 0);
  CHECK(std::strcmp(sensor::to_string(sensor::SyncRole::FreeRun), "free-run") ==
        0);
  CHECK(std::strcmp(sensor::to_string(sensor::ClockDomain::Host), "host") == 0);
  CHECK(std::strcmp(sensor::to_string(sensor::PoseSource::Tracked),
                    "tracked") == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_defaults() != 0) return 1;
  if (test_frame_returns_and_ownership() != 0) return 1;
  if (test_notify_default() != 0) return 1;
  if (test_names() != 0) return 1;
  std::printf("rgbd sensor contract tests passed\n");
  return 0;
}
