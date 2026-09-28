// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"

#include <cstdio>
#include <exception>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace volumetric_kit::recon::sensor {

namespace {

using nlohmann::json;

Status bad(const std::string& what) {
  return Status::invalid_argument("sync config: " + what);
}

// The SDK's names, as its multi-device example and OrbbecViewer write them.
constexpr std::pair<const char*, OrbbecSyncMode> kModes[] = {
    {"OB_MULTI_DEVICE_SYNC_MODE_FREE_RUN", OrbbecSyncMode::FreeRun},
    {"OB_MULTI_DEVICE_SYNC_MODE_STANDALONE", OrbbecSyncMode::Standalone},
    {"OB_MULTI_DEVICE_SYNC_MODE_PRIMARY", OrbbecSyncMode::Primary},
    {"OB_MULTI_DEVICE_SYNC_MODE_SECONDARY", OrbbecSyncMode::Secondary},
    {"OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED",
     OrbbecSyncMode::SecondarySynced},
    {"OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING",
     OrbbecSyncMode::SoftwareTriggering},
    {"OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING",
     OrbbecSyncMode::HardwareTriggering},
};

Result<int> delay(const json& sync, const char* key, const std::string& who) {
  const auto it = sync.find(key);
  if (it == sync.end() || !it->is_number_integer()) {
    return bad(who + "missing integer \"" + key + "\"");
  }
  const long long v = it->get<long long>();
  if (v < 0 || v > 1000000) {
    return bad(who + "\"" + key + "\" is " + std::to_string(v) +
               "; it must be 0 to 1000000");
  }
  return static_cast<int>(v);
}

}  // namespace

Result<OrbbecRigSyncConfig> parse_orbbec_sync_config(const std::string& text) {
  json doc;
  try {
    doc = json::parse(text);
  } catch (const std::exception& e) {
    return bad(std::string("not JSON: ") + e.what());
  }
  const auto devices = doc.is_object() ? doc.find("devices") : doc.end();
  if (!doc.is_object() || devices == doc.end() || !devices->is_array() ||
      devices->empty()) {
    return bad("no \"devices\" array");
  }
  OrbbecRigSyncConfig config;
  std::string primary;
  for (std::size_t i = 0; i < devices->size(); ++i) {
    const json& d = (*devices)[i];
    const std::string where = "device " + std::to_string(i) + ": ";
    const auto sn = d.is_object() ? d.find("sn") : d.end();
    if (!d.is_object() || sn == d.end() || !sn->is_string() ||
        sn->get<std::string>().empty()) {
      return bad(where + "no \"sn\"");
    }
    OrbbecSyncDevice device;
    device.serial = sn->get<std::string>();
    const std::string who = "camera " + device.serial + ": ";
    for (const OrbbecSyncDevice& other : config.devices) {
      if (other.serial == device.serial) return bad(who + "listed twice");
    }
    const auto sync = d.find("syncConfig");
    if (sync == d.end() || !sync->is_object()) {
      return bad(who + "no \"syncConfig\" object");
    }
    const auto mode = sync->find("syncMode");
    if (mode == sync->end() || !mode->is_string()) {
      return bad(who + "missing string \"syncMode\"");
    }
    bool known = false;
    for (const auto& [name, value] : kModes) {
      if (mode->get<std::string>() == name) {
        device.sync.mode = value;
        known = true;
      }
    }
    if (!known) {
      return bad(who + "unknown syncMode \"" + mode->get<std::string>() + "\"");
    }
    OrbbecSyncSettings& s = device.sync;
    VR_ASSIGN(s.depth_delay_us, delay(*sync, "depthDelayUs", who));
    VR_ASSIGN(s.color_delay_us, delay(*sync, "colorDelayUs", who));
    VR_ASSIGN(s.trigger_to_image_delay_us,
              delay(*sync, "trigger2ImageDelayUs", who));
    VR_ASSIGN(s.trigger_out_delay_us, delay(*sync, "triggerOutDelayUs", who));
    VR_ASSIGN(s.frames_per_trigger, delay(*sync, "framesPerTrigger", who));
    const auto out = sync->find("triggerOutEnable");
    if (out == sync->end() || !out->is_boolean()) {
      return bad(who + "missing boolean \"triggerOutEnable\"");
    }
    s.trigger_out_enable = out->get<bool>();
    if (s.mode == OrbbecSyncMode::Primary) {
      if (!primary.empty()) {
        return bad("two primaries, " + primary + " and " + device.serial);
      }
      primary = device.serial;
    }
    config.devices.push_back(std::move(device));
  }
  const auto master = doc.find("master_serial");
  if (master != doc.end()) {
    if (!master->is_string() || master->get<std::string>() != primary) {
      return bad("master_serial is not the primary device" +
                 (primary.empty() ? std::string(" (there is none)")
                                  : " (" + primary + ")"));
    }
  }
  return config;
}

Result<OrbbecRigSyncConfig> read_orbbec_sync_config(const std::string& path) {
  std::FILE* in = std::fopen(path.c_str(), "rb");
  if (in == nullptr)
    return Status::io_error("sync config: cannot open " + path);
  std::string text;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) text.append(buf, n);
  const bool failed = std::ferror(in) != 0;
  std::fclose(in);
  if (failed) return Status::io_error("sync config: cannot read " + path);
  auto config = parse_orbbec_sync_config(text);
  if (!config.ok()) {
    return Status::invalid_argument(path + ": " + config.status().message());
  }
  return config;
}

}  // namespace volumetric_kit::recon::sensor
