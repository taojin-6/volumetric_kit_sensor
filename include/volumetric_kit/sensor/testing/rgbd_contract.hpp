// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/rgbd_contract.hpp
/// @brief The frame contract (rgbd_sensor.hpp) checked on what an RGB-D
///        source handed out. One checker serves every driver's tests and the
///        tools that watch a live source. Header-only, no test framework, no
///        device; its user links `volumetric_kit::sensor_interface`.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor::testing {

/// @return The ordering rule a frame breaks, named, when it is taken right
///         after one at (@p sequence_before, @p time_before) and itself sits
///         at (@p sequence, @p time), times in nanoseconds; empty when both
///         increase. A stereo pair keeps it too.
inline std::string order_breach(std::uint64_t sequence_before,
                                std::uint64_t time_before,
                                std::uint64_t sequence, std::uint64_t time) {
  if (sequence <= sequence_before) {
    return "sequence " + std::to_string(sequence) + " after " +
           std::to_string(sequence_before);
  }
  if (time <= time_before) {
    return "time " + std::to_string(time) + " ns after " +
           std::to_string(time_before);
  }
  return {};
}

/// @return The counter rule @p stats breaks after @p taken frames were handed
///         out, named; empty when `delivered == taken` and
///         `delivered + dropped + failed <= received`.
inline std::string stats_breach(const SensorStats& stats, std::uint64_t taken) {
  if (stats.delivered != taken) {
    return "delivered " + std::to_string(stats.delivered) + ", taken " +
           std::to_string(taken);
  }
  if (stats.delivered + stats.dropped + stats.failed > stats.received) {
    return "delivered + dropped + failed past received " +
           std::to_string(stats.received);
  }
  return {};
}

/// @brief Checks a source's frames one at a time, as it hands them out, so a
///        run of any length keeps none of them.
///
/// @code
/// testing::RgbdContract contract(sensor.info());
/// // ... each frame taken: contract.take(*frame);
/// const std::string breach = contract.breach(sensor.stats());  // empty: kept
/// @endcode
///
/// @warning Not thread-safe: the thread that takes the frames uses it.
class RgbdContract {
 public:
  /// @param info  The source's, from when it opened.
  explicit RgbdContract(SensorInfo info) : info_(std::move(info)) {}

  /// @brief Check @p frame, the next one taken. After the first breach, it
  ///        only counts frames.
  inline void take(const RgbdFrame& frame);

  /// @return The first rule broken, named with its frame's place among those
  ///         taken, from 0 ("frame 12: ..."): the earliest frame's, each
  ///         frame checked in this order:
  ///         - it has depth, and its depth camera is the size of
  ///           `SensorInfo::depth`'s where that names one;
  ///         - with `SensorInfo::color`, it has colour, its colour camera is
  ///           that size, and its picture (`RgbdFrame::color`) is its
  ///           camera's size; without it, it has no colour;
  ///         - its sequence and timestamp increase on the frame's before it
  ///           (@ref order_breach).
  ///         Then the counters against @p stats (@ref stats_breach). Empty
  ///         when every rule holds.
  std::string breach(const SensorStats& stats) const {
    return !first_.empty() ? first_ : stats_breach(stats, taken_);
  }

 private:
  // The rule `frame` breaks, named without its place; empty when it keeps
  // them.
  inline std::string frame_breach(const RgbdFrame& frame) const;

  SensorInfo info_;
  std::uint64_t taken_ = 0;
  std::uint64_t last_sequence_ = 0;
  std::uint64_t last_time_ = 0;
  std::string first_;
};

/// @return @ref RgbdContract run over @p frames, in the order taken, then
///         checked against @p stats.
inline std::string rgbd_breach(const SensorInfo& info,
                               const std::vector<RgbdFrame>& frames,
                               const SensorStats& stats) {
  RgbdContract contract(info);
  for (const RgbdFrame& frame : frames) contract.take(frame);
  return contract.breach(stats);
}

inline std::string RgbdContract::frame_breach(const RgbdFrame& frame) const {
  const auto size_text = [](std::uint32_t width, std::uint32_t height) {
    return std::to_string(width) + " x " + std::to_string(height);
  };
  const auto same_size = [](const camera::ImageSize& a,
                            const camera::ImageSize& b) {
    return a.width == b.width && a.height == b.height;
  };
  if (!frame.has_depth()) return "no depth";
  const camera::ImageSize& depth = frame.depth_camera.size;
  if (info_.depth && !same_size(depth, info_.depth->size)) {
    return "depth camera " + size_text(depth.width, depth.height) +
           ", not the sensor's " +
           size_text(info_.depth->size.width, info_.depth->size.height);
  }
  if (!info_.color) {
    if (frame.has_color()) return "colour from a sensor with no colour camera";
  } else {
    if (!frame.has_color()) {
      return "no colour from a sensor with a colour camera";
    }
    const camera::ImageSize& color = frame.color_camera.size;
    if (!same_size(color, info_.color->size)) {
      return "colour camera " + size_text(color.width, color.height) +
             ", not the sensor's " +
             size_text(info_.color->size.width, info_.color->size.height);
    }
    // Packed host words carry no size of their own: they are the camera's.
    if (frame.color_packed == nullptr && (frame.color.width != color.width ||
                                          frame.color.height != color.height)) {
      return "picture " + size_text(frame.color.width, frame.color.height) +
             ", not its camera's " + size_text(color.width, color.height);
    }
  }
  if (taken_ > 0) {
    return order_breach(last_sequence_, last_time_, frame.sequence,
                        frame.timestamp_ns);
  }
  return {};
}

inline void RgbdContract::take(const RgbdFrame& frame) {
  if (first_.empty()) {
    const std::string why = frame_breach(frame);
    if (!why.empty()) first_ = "frame " + std::to_string(taken_) + ": " + why;
    last_sequence_ = frame.sequence;
    last_time_ = frame.timestamp_ns;
  }
  ++taken_;
}

}  // namespace volumetric_kit::sensor::testing
