// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Opening a rig from its sync file (open_orbbec_sensors) with camera stubs
// (orbbec_camera_stub.hpp): every camera of the file opened in its order,
// with its entry and on the host's clock; a camera that differs from its entry
// refused and left unwritten unless asked; a camera that does not answer
// NotFound, the cameras before it closed again; and a file or options that
// cannot open a rig refused before any camera is looked for. No camera or SDK
// context; the executable compiles orbbec_sensor.cpp, as the sync-apply test
// does.

#include <cstdio>
#include <string>

#include "orbbec_camera_stub.hpp"
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

// A primary and a secondary 160 us behind it.
const std::string kRig = VR_SYNC_FIXTURES "/rig.json";
constexpr const char* kPrimary = "TEST-PRIMARY";
constexpr const char* kSecondary = "TEST-SECONDARY";

bool mentions(const vkc::Status& status, const std::string& text) {
  return status.message().find(text) != std::string::npos;
}

// The fixture's two cameras answering, each stored as its entry says; the
// secondary in the SDK's spelling of the file's mode.
void attach_rig() {
  orbbec_stub::reset();
  orbbec_stub::FakeCamera& primary = orbbec_stub::cameras[kPrimary];
  primary.stored.mode = sensor::OrbbecSyncMode::Primary;
  primary.stored.trigger_out_enable = true;
  orbbec_stub::FakeCamera& secondary = orbbec_stub::cameras[kSecondary];
  secondary.stored.mode = sensor::OrbbecSyncMode::SecondarySynced;
  secondary.stored.depth_delay_us = 160;
  secondary.stored.trigger_out_enable = true;
}

int writes() {
  int n = 0;
  for (const auto& camera : orbbec_stub::cameras) n += camera.second.writes;
  return n;
}

int test_opens_rig() {
  attach_rig();
  sensor::OrbbecSensor::Options options;
  options.fps = 25;
  options.discovery_timeout_ms = 1234;
  options.sync_clock_to_host = false;  // a rig's cameras are on it regardless
  const auto opened = sensor::open_orbbec_sensors(kRig, options);
  CHECK(opened.ok());
  const auto& sensors = opened.value();
  CHECK(sensors.size() == 2);
  CHECK(sensors[0]->info().id == kPrimary);
  CHECK(sensors[0]->info().role == sensor::SyncRole::Primary);
  CHECK(sensors[1]->info().id == kSecondary);
  CHECK(sensors[1]->info().role == sensor::SyncRole::Secondary);
  for (const auto& s : sensors) {
    CHECK(s->info().clock == sensor::ClockDomain::Host);
    CHECK(s->info().fps == 25);  // the options' streams, every camera
  }
  for (const auto& camera : orbbec_stub::cameras) {
    CHECK(camera.second.host_clock);
    CHECK(camera.second.discovery_timeout_ms == 1234);
  }
  CHECK(writes() == 0);  // the file is the cameras' settings already
  CHECK(orbbec_stub::open_streams == 2);
  return 0;
}

int test_difference() {
  for (const bool apply : {false, true}) {
    attach_rig();
    orbbec_stub::cameras[kSecondary].stored.depth_delay_us = 0;
    sensor::OrbbecSensor::Options options;
    options.apply_sync = apply;
    const auto opened = sensor::open_orbbec_sensors(kRig, options);
    if (!apply) {
      CHECK(opened.status().domain() == vkc::Status::Code::Unsupported);
      CHECK(mentions(opened.status(), kSecondary));
      CHECK(mentions(opened.status(), "depthDelayUs is 0, configured 160"));
      CHECK(writes() == 0);
      CHECK(orbbec_stub::open_streams == 0);  // the primary closed again
    } else {
      // Written to the camera that differs, and only to it.
      CHECK(opened.ok());
      CHECK(orbbec_stub::cameras[kPrimary].writes == 0);
      CHECK(orbbec_stub::cameras[kSecondary].writes == 1);
      CHECK(orbbec_stub::cameras[kSecondary].stored.depth_delay_us == 160);
    }
  }
  return 0;
}

int test_absent_camera() {
  attach_rig();
  orbbec_stub::cameras.erase(kSecondary);
  const vkc::Status s = sensor::open_orbbec_sensors(kRig, {}).status();
  CHECK(s.domain() == vkc::Status::Code::NotFound);
  CHECK(mentions(s, kSecondary));
  CHECK(orbbec_stub::lookups == 2);
  CHECK(orbbec_stub::open_streams == 0);  // the primary closed again
  return 0;
}

int test_refusals() {
  attach_rig();
  // The file names the cameras and their settings, not the options.
  sensor::OrbbecSensor::Options named;
  named.serial = kPrimary;
  vkc::Status s = sensor::open_orbbec_sensors(kRig, named).status();
  CHECK(s.domain() == vkc::Status::Code::InvalidArgument);
  CHECK(mentions(s, "serial"));
  sensor::OrbbecSensor::Options synced;
  synced.sync.emplace();
  s = sensor::open_orbbec_sensors(kRig, synced).status();
  CHECK(s.domain() == vkc::Status::Code::InvalidArgument);
  CHECK(mentions(s, "sync"));

  // A file that cannot be read, or holds no rig, named.
  const std::string missing = VR_SYNC_FIXTURES "/missing.json";
  s = sensor::open_orbbec_sensors(missing, {}).status();
  CHECK(s.domain() == vkc::Status::Code::IoError);
  CHECK(mentions(s, missing));
  const std::string empty = VR_SYNC_FIXTURES "/no_devices.json";
  s = sensor::open_orbbec_sensors(empty, {}).status();
  CHECK(s.domain() == vkc::Status::Code::InvalidArgument);
  CHECK(mentions(s, empty));
  CHECK(mentions(s, "devices"));

  CHECK(orbbec_stub::lookups == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_opens_rig() != 0) return 1;
  if (test_difference() != 0) return 1;
  if (test_absent_camera() != 0) return 1;
  if (test_refusals() != 0) return 1;
  std::puts("orbbec open sensors tests passed");
  return 0;
}
