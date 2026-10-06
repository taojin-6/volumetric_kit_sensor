// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/array/sensor_array.hpp
/// @brief Several RGB-D sensors read as one array: started in their rig's
///        order, their frames grouped into one set per trigger and posed by
///        the array's calibration.
///
/// Vendor-neutral: an array holds @ref IRgbdSensor s, whatever drives them,
/// and reads what it needs of each from its @ref SensorInfo -- the role that
/// orders the starts, the clock that grouping needs, the id its pose is found
/// by. What a vendor's rig needs beyond that (writing sync settings, the
/// SDK's clock sync) is its driver's, set when the sensor is opened.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"
#include "volumetric_kit/recon/sensor/array/export.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief How an array groups its sensors' frames into sets.
enum class SyncMode : std::uint8_t {
  /// Hardware-triggered: the primary's frames name the triggers, and each
  /// secondary's frame within @ref SensorArray::Options::tolerance_us of one
  /// on the host's clock belongs to it. A set may miss a secondary, never the
  /// primary; a slow consumer gets the newest set.
  Trigger,
  /// Frames already say which set they belong to: equal
  /// `RgbdFrame::sequence` numbers, as a recording or a dataset gives. Every
  /// set is handed out, in order, once each sensor has sent its frame for it
  /// or moved past it or run out; nothing waits on a clock, so a replay is
  /// deterministic.
  Sequence,
};

/// @return A stable lowercase name for @p mode, for logs.
VR_SENSOR_ARRAY_API const char* to_string(SyncMode mode) noexcept;

/// @brief The frames one trigger produced across the array.
struct FrameSet {
  /// The trigger's time (ns): the primary's frame's for `SyncMode::Trigger`,
  /// the first sensor's present for `SyncMode::Sequence`.
  std::uint64_t timestamp_ns = 0;
  /// The set's sequence number for `SyncMode::Sequence`; the primary's
  /// frame's for `SyncMode::Trigger`.
  std::uint64_t sequence = 0;
  /// One per sensor, in the array's order; empty where that sensor's frame
  /// for this trigger never came. Each frame is posed by the array's
  /// calibration (`RgbdFrame::color_to_world`).
  std::vector<std::optional<RgbdFrame>> frames;

  /// @return How many sensors this set holds a frame from.
  std::size_t count() const noexcept {
    std::size_t n = 0;
    for (const std::optional<RgbdFrame>& f : frames) n += f ? 1 : 0;
    return n;
  }
  /// @return `true` when every sensor's frame is here.
  bool complete() const noexcept { return count() == frames.size(); }
};

/// @brief Counters a caller reads to see how the array is keeping up.
struct SensorArrayStats {
  std::uint64_t sets = 0;  ///< Sets handed out since @ref SensorArray::start.
  std::uint64_t incomplete = 0;  ///< Of those, sets missing a sensor.
  /// Frames drained from the sensors that joined no set handed out: their
  /// trigger was overtaken by a newer one, they came too early or too late
  /// for any, or they had no timestamp to group by.
  std::uint64_t unmatched = 0;
  /// Each sensor's own counters, in the array's order.
  std::vector<SensorStats> sensors;
};

/// @brief Several RGB-D sensors read as one array.
///
/// @ref open checks that the sensors can form one under the chosen
/// @ref SyncMode, and that the calibration poses each of them. @ref start
/// starts every secondary before the primary -- the primary's first trigger
/// is what they wait for. @ref poll_set drains every sensor's frames, groups
/// them, and hands out a set, each frame posed by the calibration.
///
/// @code
/// std::vector<std::unique_ptr<IRgbdSensor>> sensors;
/// for (const std::string& serial : serials) {
///   OrbbecSensor::Options o;
///   o.serial = serial;
///   o.clock_sync_interval_ms = 60000;  // Trigger groups on the host clock
///   VKC_ASSIGN(OrbbecSensor s, OrbbecSensor::open(o));
///   sensors.push_back(std::make_unique<OrbbecSensor>(std::move(s)));
/// }
/// SensorArray::Options options;
/// VKC_ASSIGN(options.calibration, camera::read_array_calibration(path));
/// VKC_ASSIGN(SensorArray array, SensorArray::open(std::move(sensors),
///                                                 options));
/// VKC_TRY(array.start());
/// VKC_ASSIGN(std::optional<FrameSet> set, array.poll_set());
/// @endcode
///
/// @warning Not thread-safe: open, start, poll and stop from one thread.
class VR_SENSOR_ARRAY_API SensorArray {
 public:
  /// @brief How the sensors are grouped and posed.
  struct Options {
    SyncMode sync = SyncMode::Trigger;  ///< How frames form sets.
    /// For `SyncMode::Trigger`: a secondary's frame within this of a primary
    /// frame belongs to its trigger. Under half the fastest sensor's frame
    /// period, or neighbouring triggers would share frames.
    std::uint32_t tolerance_us = 5000;
    /// Frames each sensor holds between polls (`IRgbdSensor::set_queue_depth`)
    /// and the grouping holds after; enough that every sensor still holds a
    /// trigger's frame when a slow poll comes to group it.
    std::size_t queue_depth = 4;
    /// Each sensor's pose, by id, stamped on its frames. Every sensor of the
    /// array must be posed in it; no sensors leaves each frame where its
    /// driver put it.
    camera::ArrayCalibration calibration;
  };

  /// @brief Check that @p sensors form an array, and take them.
  ///
  /// None is started. Under `SyncMode::Trigger`, more than one sensor needs
  /// exactly one primary, the rest secondaries, all on the host's clock;
  /// under `SyncMode::Sequence` any sensors do.
  /// @param sensors  The sensors, in the order sets report them.
  /// @param options  How they are grouped and posed.
  /// @return The array; `Status::Code::InvalidArgument` for no sensors, a
  ///         null one, an empty or repeated id, a queue depth of 0, a
  ///         tolerance of 0 or of half a frame period or more, or a
  ///         calibration that is invalid or does not pose a sensor; or
  ///         `Status::Code::Unsupported` for a tracked sensor, or, under
  ///         `SyncMode::Trigger`, no primary or more than one, a free-running
  ///         member, or one whose timestamps are on its own clock.
  static core::Result<SensorArray> open(
      std::vector<std::unique_ptr<IRgbdSensor>> sensors,
      const Options& options);

  SensorArray(SensorArray&& other) noexcept;
  SensorArray& operator=(SensorArray&& other) noexcept;
  /// Stops the sensors if they are running.
  ~SensorArray();

  /// @return How many sensors the array has; 0 on a moved-from array.
  std::size_t size() const noexcept;
  /// @return Sensor @p i, in the array's order, for what its own type tells
  ///         (an `OrbbecSensor`'s device info).
  /// @pre @p i is below @ref size.
  IRgbdSensor& sensor(std::size_t i);
  /// @return The index of the primary under `SyncMode::Trigger`; 0 otherwise.
  std::size_t primary() const noexcept;
  /// @return The counters since the last @ref start.
  SensorArrayStats stats() const;

  /// @brief Start every sensor, each secondary before the primary.
  ///        Idempotent.
  /// @return OK once all run; `Status::Code::InvalidArgument` on a moved-from
  ///         array; or the first sensor's failure, with every sensor stopped.
  core::Status start();

  /// @brief Stop every sensor and drop the frames held. Idempotent.
  void stop() noexcept;

  /// @brief Take the next set ready to hand out.
  /// @return The set; an empty optional when none is ready or the array is
  ///         not started; `Status::Code::InvalidArgument` on a moved-from
  ///         array; or the first sensor's failure.
  core::Result<std::optional<FrameSet>> poll_set();

  /// @return `true` on a moved-from array; once a sensor of a
  ///         `SyncMode::Trigger` array is exhausted (a disconnected camera);
  ///         and once every sensor of a `SyncMode::Sequence` array is
  ///         exhausted and every set handed out.
  bool exhausted() const noexcept;

 private:
  struct Impl;
  explicit SensorArray(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
