// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The sync contract every stereo source keeps, checked on what it handed out
// since its start: each frame's skew within its info's bound
// (StereoInfo::max_skew_ns), sequences and times increasing, and the
// counters' rule (SensorStats: delivered + dropped + failed <= received, and
// delivered the frames taken). One check for every source, a test camera's
// and each driver's own unit, so a driver proves the rule rather than a test
// of its own (sensor-robustness-generic).

#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace stereo_contract {

namespace sensor = volumetric_kit::sensor;

// The first rule `frames`, in the order they were taken, and `stats` break,
// named; empty when they keep them all.
inline std::string breach(const sensor::StereoInfo& info,
                          const std::vector<sensor::StereoFrame>& frames,
                          const sensor::SensorStats& stats) {
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const sensor::StereoFrame& f = frames[i];
    const std::string at = "frame " + std::to_string(i) + ": ";
    const std::uint64_t apart =
        f.skew_ns < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(f.skew_ns)
                      : static_cast<std::uint64_t>(f.skew_ns);
    if (apart > info.max_skew_ns) {
      return at + "skew " + std::to_string(f.skew_ns) + " ns past the bound " +
             std::to_string(info.max_skew_ns);
    }
    if (i == 0) continue;
    const sensor::StereoFrame& before = frames[i - 1];
    if (f.sequence <= before.sequence) {
      return at + "sequence " + std::to_string(f.sequence) + " after " +
             std::to_string(before.sequence);
    }
    if (f.timestamp_ns <= before.timestamp_ns) {
      return at + "time " + std::to_string(f.timestamp_ns) + " ns after " +
             std::to_string(before.timestamp_ns);
    }
  }
  if (stats.delivered != frames.size()) {
    return "delivered " + std::to_string(stats.delivered) + ", taken " +
           std::to_string(frames.size());
  }
  if (stats.delivered + stats.dropped + stats.failed > stats.received) {
    return "delivered + dropped + failed past received " +
           std::to_string(stats.received);
  }
  return {};
}

}  // namespace stereo_contract
