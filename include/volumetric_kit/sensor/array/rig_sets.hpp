// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/array/rig_sets.hpp
/// @brief A run of N frames read from a started array: each set waited for on
///        the sensors' notifications and cut to the frames the run still
///        takes.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/recon/sensor/array/export.hpp"
#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief A started array's sets, each waited for on its sensors'
///        notifications (@ref SensorArray::wait_set) and cut to the run's
///        frames, with each secondary's skew to the primary.
///
/// A run of N frames stops mid-set: the set that reaches N keeps its first
/// frames in the array's order and lets the rest go, their pictures with
/// them. The array is borrowed: it must outlive every @ref next. A set's prep
/// stays the caller's (@ref SensorArray::record_process), recorded into the
/// caller's batch with the work that reads it.
///
/// @code
/// VKC_TRY(array.start());
/// for (RigSets sets(array, frames); !sets.done();) {
///   VKC_ASSIGN(FrameSet set, sets.next(&rows));
///   core::CommandBatch front(device, allocator);
///   VKC_ASSIGN(const DeviceFrameSet prepared,
///              array.record_process(front, set, &rows));
///   // the set's fusion and frame, recorded into front after its prep
/// }
/// @endcode
class VR_SENSOR_ARRAY_API RigSets {
 public:
  /// How long @ref next waits for a set before calling the sensors silent.
  /// A streaming camera delivers one every ~33 ms; a secondary with no
  /// primary, or a camera that dropped off the network, delivers none.
  static constexpr std::chrono::seconds kSilenceLimit{10};

  /// @brief A secondary's frame's time less its set's, the primary's, over
  ///        the sets @ref next handed out, before their cut.
  struct Skew {
    double sum_ms = 0.0;       ///< The sum, so the mean is over @ref frames.
    double worst_ms = 0.0;     ///< The largest magnitude.
    std::uint64_t frames = 0;  ///< The sensor's frames among those sets.
  };

  /// @brief A run that has taken nothing yet. Allocates nothing: the skews'
  ///        storage comes with the first @ref next.
  /// @param array   Started; borrowed.
  /// @param frames  The frames the run takes.
  RigSets(SensorArray& array, std::size_t frames) noexcept;

  /// @return Whether the run has taken its frames.
  bool done() const noexcept { return frames_ >= limit_; }
  /// @return The sets @ref next has handed out.
  std::size_t sets() const noexcept { return sets_; }
  /// @return The frames those sets held, as cut.
  std::size_t frames() const noexcept { return frames_; }
  /// @return Sensor @p camera's skew to the primary over the sets so far, a
  ///         copy: empty for the primary and before the first set.
  /// @pre @p camera is below the array's size.
  Skew skew(std::size_t camera) const;

  /// @brief Wait for the next set and cut it to the frames the run still
  ///        takes.
  ///
  /// Blocked on the sensors' notifications (@ref SensorArray::wait_set): the
  /// time blocked is @p metrics' `"wait: cameras"` row, the polls' host time
  /// its `"poll"`. Each secondary's skew to the primary is kept
  /// (@ref skew).
  /// @param metrics  Optional stage rows; null measures nothing.
  /// @return The set; `Status::Code::IoError` after @ref kSilenceLimit with
  ///         none, naming a sync secondary's cause;
  ///         `Status::Code::OutOfMemory` when host memory runs out; or the
  ///         array's error.
  core::Result<FrameSet> next(core::StageMetrics* metrics = nullptr);

 private:
  SensorArray& array_;
  std::size_t size_;  // the array's sensors
  std::size_t limit_;
  std::size_t sets_ = 0;
  std::size_t frames_ = 0;
  std::vector<Skew> skews_;  // per sensor; the primary's stays empty
  std::chrono::steady_clock::time_point last_set_;
};

}  // namespace volumetric_kit::recon::sensor
