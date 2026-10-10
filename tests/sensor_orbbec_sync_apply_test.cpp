// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Exercise OrbbecSensor::open with a camera stub (orbbec_camera_stub.hpp):
// matching settings must not be written, write failures must propagate, and
// SDK-normalized settings must not be reported as applied unchanged. No camera
// or SDK context is opened. The executable compiles orbbec_sensor.cpp so the
// stub works with static and shared libraries without adding a test seam to
// the public API.

#include <cstdio>
#include <string>

#include "orbbec_camera_stub.hpp"
#include "test_check.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;

namespace {

constexpr const char* kSerial = "TEST-SYNC-CAMERA";

// The one camera that answers, its stored settings the defaults.
orbbec_stub::FakeCamera& attach() {
  orbbec_stub::reset();
  return orbbec_stub::cameras[kSerial];
}

sensor::OrbbecSensor::Options primary_options() {
  sensor::OrbbecSensor::Options options;
  options.serial = kSerial;
  options.sync.emplace();
  options.sync->mode = sensor::OrbbecSyncMode::Primary;
  options.sync->trigger_out_enable = true;
  options.apply_sync = true;
  return options;
}

int test_matching_settings() {
  for (const bool apply : {false, true}) {
    orbbec_stub::FakeCamera& fake = attach();
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
  orbbec_stub::FakeCamera& fake = attach();
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
  const orbbec_stub::FakeCamera& fake = attach();
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
  orbbec_stub::FakeCamera& fake = attach();
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
