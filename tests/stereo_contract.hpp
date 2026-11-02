// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The contract every stereo source keeps, checked on its info and on what it
// handed out since its start: its calibration says how it was opened
// (StereoInfo::calibration: both cameras 0 x 0 without one, neither with
// one), each frame's skew within its info's bound (StereoInfo::max_skew_ns),
// sequences and times increasing, and the counters' rule (SensorStats:
// delivered + dropped + failed <= received, and delivered the frames taken).
// One check for every source, a test camera's and each driver's own unit, so
// a driver proves the rule rather than a test of its own (the same robustness
// rule for every driver).

#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace stereo_contract {

namespace sensor = volumetric_kit::sensor;

// How the source under test was opened: with its pair's calibration, or
// without one, as a pair is opened to be calibrated.
enum class Opened { Calibrated, Uncalibrated };

// The calibration rule `info` breaks, its source `opened` so, named; empty
// when it keeps it. Opened without a calibration, both cameras are exactly
// 0 x 0; with one, neither camera has a side 0.
inline std::string calibration_breach(const sensor::StereoInfo& info,
                                      Opened opened) {
  const sensor::StereoCalibration& c = info.calibration;
  const char* names[2] = {"left", "right"};
  const sensor::camera::ImageSize sizes[2] = {c.left.size, c.right.size};
  for (int k = 0; k < 2; ++k) {
    const sensor::camera::ImageSize& s = sizes[k];
    const std::string is = std::string("the ") + names[k] + " camera is " +
                           std::to_string(s.width) + " x " +
                           std::to_string(s.height);
    if (opened == Opened::Uncalibrated && (s.width != 0 || s.height != 0)) {
      return is + " from a source opened without a calibration, not 0 x 0";
    }
    if (opened == Opened::Calibrated && (s.width == 0 || s.height == 0)) {
      return is + " from a source opened with a calibration";
    }
  }
  return {};
}

// The first rule `info`, its source `opened` so, `frames`, in the order they
// were taken, and `stats` break, named; empty when they keep them all.
inline std::string breach(const sensor::StereoInfo& info, Opened opened,
                          const std::vector<sensor::StereoFrame>& frames,
                          const sensor::SensorStats& stats) {
  const std::string calibration = calibration_breach(info, opened);
  if (!calibration.empty()) return calibration;
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
