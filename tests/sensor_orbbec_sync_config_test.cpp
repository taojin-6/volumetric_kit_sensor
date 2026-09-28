// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The rig's sync configuration: the SDK's file layout read, its refusals, the
// SDK struct round trip, and the comparison OrbbecRig::open makes -- which must
// call the rig's cameras, as they read back what femto_mega_sync.json wrote,
// in agreement with that file. No camera.

#include <cstdio>
#include <string>
#include <vector>

#include "frame_conversion.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"

namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;
namespace orbbec = volumetric_kit::recon::sensor::orbbec;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

std::string device(const char* sn, const char* mode, int depth_delay) {
  return std::string(R"({"sn": ")") + sn +
         R"(", "syncConfig": {"syncMode": "OB_MULTI_DEVICE_SYNC_MODE_)" + mode +
         R"(", "depthDelayUs": )" + std::to_string(depth_delay) +
         R"(, "colorDelayUs": 0, "trigger2ImageDelayUs": 0,
          "triggerOutEnable": true, "triggerOutDelayUs": 0,
          "framesPerTrigger": 1}})";
}

// femto_mega_sync.json, the rig's file.
std::string rig_file(const std::string& master = "CL2A141000N") {
  return R"({"master_serial": ")" + master + R"(", "devices": [)" +
         device("CL2A141000N", "PRIMARY", 0) + "," +
         device("CL2A141006G", "SECONDARY", 160) + "," +
         device("CL2A14100A4", "SECONDARY", 320) + "," +
         device("CL2A141000G", "SECONDARY", 480) + "]}";
}

bool refused(const std::string& json) {
  const auto r = sensor::parse_orbbec_sync_config(json);
  if (r.ok()) return false;
  std::printf("  refused as expected: %s\n", r.status().message().c_str());
  return r.status().domain() == vr::Status::Code::InvalidArgument;
}

int test_parse() {
  const auto r = sensor::parse_orbbec_sync_config(rig_file());
  CHECK(r.ok());
  const auto& d = r.value().devices;
  CHECK(d.size() == 4);
  CHECK(d[0].serial == "CL2A141000N" &&
        d[0].sync.mode == sensor::OrbbecSyncMode::Primary);
  CHECK(d[3].serial == "CL2A141000G" &&
        d[3].sync.mode == sensor::OrbbecSyncMode::Secondary);
  CHECK(d[1].sync.depth_delay_us == 160 && d[2].sync.depth_delay_us == 320);
  CHECK(d[3].sync.trigger_out_enable && d[3].sync.frames_per_trigger == 1);

  // The SDK's own example has no master_serial.
  CHECK(
      sensor::parse_orbbec_sync_config(R"({"version": "1.0.0", "devices": [)" +
                                       device("A", "PRIMARY", 0) + "]}")
          .ok());
  return 0;
}

int test_refusals() {
  CHECK(refused("{not json"));
  CHECK(refused(R"({"devices": []})"));
  CHECK(refused(rig_file("CL2A141000G")));  // master is not the primary
  CHECK(refused(R"({"devices": [)" + device("A", "PRIMARY", 0) + "," +
                device("B", "PRIMARY", 0) + "]}"));
  CHECK(refused(R"({"devices": [)" + device("A", "PRIMARY", 0) + "," +
                device("A", "SECONDARY", 160) + "]}"));
  CHECK(refused(R"({"devices": [)" + device("A", "SIDEWAYS", 0) + "]}"));
  CHECK(refused(R"({"devices": [)" + device("A", "SECONDARY", -1) + "]}"));
  CHECK(refused(
      R"({"devices": [{"sn": "A", "syncConfig": {"syncMode":
      "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY", "depthDelayUs": 0}}]})"));
  return 0;
}

int test_sdk_round_trip() {
  sensor::OrbbecSyncSettings s;
  s.mode = sensor::OrbbecSyncMode::Secondary;
  s.depth_delay_us = 320;
  s.color_delay_us = 7;
  s.trigger_to_image_delay_us = 9;
  s.trigger_out_enable = true;
  s.trigger_out_delay_us = 11;
  s.frames_per_trigger = 2;
  const OBMultiDeviceSyncConfig sdk = orbbec::sdk_sync_config(s);
  CHECK(sdk.syncMode == OB_MULTI_DEVICE_SYNC_MODE_SECONDARY);
  const sensor::OrbbecSyncSettings back = orbbec::sync_settings_from(sdk);
  CHECK(back.mode == s.mode && back.depth_delay_us == 320 &&
        back.color_delay_us == 7 && back.trigger_to_image_delay_us == 9 &&
        back.trigger_out_enable && back.trigger_out_delay_us == 11 &&
        back.frames_per_trigger == 2);
  return 0;
}

int test_differences() {
  const auto file = sensor::parse_orbbec_sync_config(rig_file());
  CHECK(file.ok());
  const sensor::OrbbecSyncSettings& wanted = file.value().devices[1].sync;

  // What CL2A141006G reads back after the file is written to it, measured:
  // SecondarySynced, trigger2ImageDelayUs = the depth delay, framesPerTrigger
  // 0.
  sensor::OrbbecSyncSettings camera;
  camera.mode = sensor::OrbbecSyncMode::SecondarySynced;
  camera.depth_delay_us = 160;
  camera.trigger_to_image_delay_us = 160;
  camera.trigger_out_enable = true;
  camera.frames_per_trigger = 0;
  CHECK(orbbec::sync_differences(wanted, camera).empty());

  // A delay and a role that really differ are named.
  camera.depth_delay_us = 320;
  auto d = orbbec::sync_differences(wanted, camera);
  CHECK(d.size() == 1 && d[0] == "depthDelayUs is 320, configured 160");
  camera.depth_delay_us = 160;
  camera.mode = sensor::OrbbecSyncMode::Primary;
  d = orbbec::sync_differences(wanted, camera);
  CHECK(d.size() == 1 && d[0] == "syncMode is primary, configured secondary");
  camera.mode = sensor::OrbbecSyncMode::SecondarySynced;
  camera.trigger_out_enable = false;
  CHECK(orbbec::sync_differences(wanted, camera).size() == 1);

  // framesPerTrigger counts only in a triggering mode.
  sensor::OrbbecSyncSettings triggered;
  triggered.mode = sensor::OrbbecSyncMode::HardwareTriggering;
  triggered.frames_per_trigger = 2;
  sensor::OrbbecSyncSettings read_back = triggered;
  read_back.frames_per_trigger = 1;
  CHECK(orbbec::sync_differences(triggered, read_back).size() == 1);
  return 0;
}

int test_committed_rig() {
  // config/femto_mega_sync.json, the lab rig as committed: it reads, has one
  // primary, and staggers every secondary's depth delay so no two ToF
  // exposures overlap.
  const auto r = sensor::read_orbbec_sync_config(VR_RIG_SYNC_CONFIG);
  if (!r.ok()) std::fprintf(stderr, "%s\n", r.status().message().c_str());
  CHECK(r.ok());
  int primaries = 0;
  std::vector<int> delays;
  for (const sensor::OrbbecSyncDevice& d : r.value().devices) {
    if (d.sync.mode == sensor::OrbbecSyncMode::Primary) {
      ++primaries;
    } else {
      CHECK(sensor::waits_for_primary(d.sync.mode));
      for (const int other : delays) CHECK(other != d.sync.depth_delay_us);
      delays.push_back(d.sync.depth_delay_us);
    }
  }
  CHECK(primaries == 1 && !delays.empty());
  return 0;
}

}  // namespace

int main() {
  if (test_parse() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_sdk_round_trip() != 0) return 1;
  if (test_differences() != 0) return 1;
  if (test_committed_rig() != 0) return 1;
  std::printf("orbbec sync config tests passed\n");
  return 0;
}
