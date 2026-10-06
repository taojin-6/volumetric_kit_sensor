// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::recon::sensor {

const char* to_string(SyncRole role) noexcept {
  switch (role) {
    case SyncRole::Primary:
      return "primary";
    case SyncRole::Secondary:
      return "secondary";
    case SyncRole::FreeRun:
      break;
  }
  return "free-run";
}

const char* to_string(ClockDomain clock) noexcept {
  return clock == ClockDomain::Host ? "host" : "device";
}

const char* to_string(PoseSource source) noexcept {
  return source == PoseSource::Tracked ? "tracked" : "fixed";
}

}  // namespace volumetric_kit::recon::sensor
