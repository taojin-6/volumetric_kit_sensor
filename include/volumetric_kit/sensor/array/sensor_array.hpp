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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"
#include "volumetric_kit/recon/sensor/array/export.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief The frames one trigger produced across the array.
struct FrameSet {
  /// The trigger's time (ns): the primary's frame's.
  std::uint64_t timestamp_ns = 0;
  /// The primary's frame's sequence number.
  std::uint64_t sequence = 0;
  /// One per sensor, in the array's order; empty where that sensor's frame
  /// for this trigger never came. Each frame is posed by the array's
  /// calibration (`RgbdFrame::color_to_world`).
  std::vector<std::optional<RgbdFrame>> frames;
  /// When the set's last frame reached the host: the latest of its frames'
  /// `RgbdFrame::arrival_ns`, on the system clock; 0 when none says.
  std::uint64_t arrival_ns = 0;

  /// @return How many sensors this set holds a frame from.
  std::size_t count() const noexcept {
    std::size_t n = 0;
    for (const std::optional<RgbdFrame>& f : frames) n += f ? 1 : 0;
    return n;
  }
  /// @return `true` when every sensor's frame is here.
  bool complete() const noexcept { return count() == frames.size(); }
  /// @return Sensor @p i's frame's time less the trigger's
  ///         (@ref timestamp_ns), in nanoseconds: negative for a frame
  ///         stamped before the primary's, 0 for the primary's own, and
  ///         saturated at `int64_t`'s range for a gap of 2^63 ns or more;
  ///         empty where the set holds no frame from that sensor.
  /// @pre @p i is below `frames.size()`.
  VR_SENSOR_ARRAY_API std::optional<std::int64_t> skew_ns(std::size_t i) const;
  /// @return The latest frame's time less the earliest's, over the frames the
  ///         set holds, in nanoseconds; 0 for one frame or none.
  VR_SENSOR_ARRAY_API std::uint64_t spread_ns() const noexcept;
};

/// @brief A set prepared on the GPU: each frame undistorted and converted,
///        left on the device for fusion.
struct DeviceFrameSet {
  std::uint64_t timestamp_ns = 0;  ///< As @ref FrameSet::timestamp_ns.
  std::uint64_t sequence = 0;      ///< As @ref FrameSet::sequence.
  /// One per sensor, in the array's order; empty where the set had no frame.
  std::vector<std::optional<DeviceFrame>> frames;
};

/// @brief Counters a caller reads to see how the array is keeping up.
struct SensorArrayStats {
  std::uint64_t sets = 0;  ///< Sets handed out since @ref SensorArray::start.
  std::uint64_t incomplete = 0;  ///< Of those, sets missing a sensor.
  /// Frames drained from the sensors that joined no set handed out: their
  /// trigger was overtaken by a newer one, they came too early or too late
  /// for any, they had no timestamp to group by, or a full queue let them
  /// go.
  std::uint64_t unmatched = 0;
  /// Each sensor's own counters, in the array's order.
  std::vector<SensorStats> sensors;
};

/// @brief Several RGB-D sensors read as one array.
///
/// @ref open checks that the sensors can form one, and that the calibration
/// poses each of them, and asks each sensor to notify the array of its frames
/// (`IRgbdSensor::set_frame_notify`). @ref start starts every secondary
/// before the primary -- the primary's first trigger is what they wait for.
/// @ref poll_set drains every sensor's frames, groups them by trigger, and
/// hands out a set, each frame posed by the calibration; @ref wait_set blocks
/// for the next set on the sensors' notifications. @ref record_process records
/// a set's preparation on the GPU, every stream, into a caller's batch, and
/// @ref process prepares it in one of its own.
///
/// The primary's frames name the triggers, and each secondary's frame within
/// @ref Options::tolerance_us of one on the host's clock belongs to it: a
/// hardware trigger aligns the exposures, so the clocks need only agree well
/// enough to tell one trigger from the next. A set may miss a secondary,
/// never the primary; a slow consumer gets the newest set.
///
/// @code
/// // A rig's cameras, each checked against its sync file, on the host clock.
/// VKC_ASSIGN(auto sensors, open_orbbec_sensors(sync_file, {}));
/// SensorArray::Options options;
/// VKC_ASSIGN(options.calibration, camera::read_array_calibration(path));
/// VKC_ASSIGN(SensorArray array, SensorArray::open(std::move(sensors),
///                                                 options));
/// VKC_TRY(array.start());
/// VKC_ASSIGN(std::optional<FrameSet> set, array.poll_set());
/// if (set) {
///   VKC_ASSIGN(const DeviceFrameSet prepared, array.process(*set));
///   // fuse prepared.frames: allocate_from_depth and integrate, batched
/// }
/// @endcode
///
/// @warning Not thread-safe: open, start, poll and stop from one thread.
class VR_SENSOR_ARRAY_API SensorArray {
 public:
  /// @brief How the sensors are grouped and posed.
  struct Options {
    /// A secondary's frame within this of a primary frame belongs to its
    /// trigger. Under half the fastest sensor's frame period, or neighbouring
    /// triggers would share frames. 0 takes 0.4 of that period (13 333 us at
    /// 30 fps), or 5000 us when no sensor reports a rate.
    std::uint32_t tolerance_us = 0;
    /// Frames each sensor holds between polls (`IRgbdSensor::set_queue_depth`)
    /// and the grouping holds after; enough that every sensor still holds a
    /// trigger's frame when a slow poll comes to group it.
    std::size_t queue_depth = 4;
    /// Each sensor's pose, by id, stamped on its frames, and its depth
    /// offset, which its pass subtracts (`SensorCalibration::depth_offset_m`,
    /// `GpuFramePrepConfig::depth_offset_m`). Every sensor of the array must
    /// be posed in it (`SensorCalibration::color_to_world`), and where it
    /// names the colour cameras' exposure
    /// (`camera::ArrayCalibration::color_controls`), every sensor must have
    /// been opened at it (`SensorInfo::color_controls`); no sensors leaves
    /// each frame where its driver put it, and its depth as read.
    camera::ArrayCalibration calibration;
    /// The device @ref process prepares sets on, with a pass per sensor;
    /// null for an array whose sets are not prepared here. Borrowed: it must
    /// outlive the array and every set it prepares.
    core::Device* device = nullptr;
    /// The allocator the passes' buffers come from; given with @ref device
    /// and only with it. Borrowed, as @ref device is.
    core::Allocator* allocator = nullptr;
    /// Every pass's options, whatever the array's size: the queue families
    /// sharing its colour and its depth, and whether depth is kept only where
    /// colour was recorded (`GpuFramePrepConfig`). Its `depth_offset_m`
    /// stays 0: each pass takes its sensor's from @ref calibration, the one
    /// source of it.
    GpuFramePrepConfig prep{};
  };

  /// @brief Check that @p sensors form an array, and take them.
  ///
  /// None is started. More than one sensor needs exactly one primary, the
  /// rest secondaries, all on the host's clock; one sensor needs neither.
  /// @param sensors  The sensors, in the order sets report them; taken only
  ///                 once the array opens, so a refusal leaves them open
  ///                 with the caller.
  /// @param options  How they are grouped and posed.
  /// @return The array; `Status::Code::InvalidArgument` for no sensors, a
  ///         null one, an empty or repeated id, a queue depth of 0, a
  ///         tolerance of half a frame period or more, a calibration
  ///         that is invalid or does not pose a sensor (lists it unposed, or
  ///         not at all), a sensor whose `SensorInfo::color_controls` differ
  ///         from the calibration's `color_controls` where it names them, a
  ///         `prep.depth_offset_m` other than 0, or a device or an allocator
  ///         without the other;
  ///         what `GpuFramePrep::create` returns for a @ref Options::prep it
  ///         refuses or a device the passes cannot be built on;
  ///         `Status::Code::Unsupported` for a tracked sensor, or, among
  ///         several sensors, no primary or more than one, a free-running
  ///         member, or one whose timestamps are on its own clock; or what a
  ///         sensor's `IRgbdSensor::set_frame_notify` refuses, other than
  ///         `Status::Code::Unsupported` (a started sensor).
  static core::Result<SensorArray> open(
      std::vector<std::unique_ptr<IRgbdSensor>>&& sensors,
      const Options& options);

  SensorArray(SensorArray&& other) noexcept;
  /// Stops the sensors of the array it replaces, as @ref stop does.
  SensorArray& operator=(SensorArray&& other) noexcept;
  /// Stops the sensors if they are running, as @ref stop does.
  ~SensorArray();

  /// @return How many sensors the array has; 0 on a moved-from array.
  std::size_t size() const noexcept;
  /// @return Sensor @p i, in the array's order, for what its own type tells
  ///         (an `OrbbecSensor`'s device info).
  /// @pre @p i is below @ref size.
  IRgbdSensor& sensor(std::size_t i);
  /// @return The index of the primary; 0 for a single sensor.
  std::size_t primary() const noexcept;
  /// @return The counters since the last @ref start.
  SensorArrayStats stats() const;

  /// @brief Start every sensor, each secondary before the primary.
  ///        Idempotent: on a running array each sensor is started again,
  ///        which is OK for a running one and says why one has failed since
  ///        (a disconnected camera).
  /// @return OK once all run; `Status::Code::InvalidArgument` on a moved-from
  ///         array; or the first sensor's failure, with every sensor stopped.
  core::Status start();

  /// @brief Stop every sensor, the primary first, and drop the frames held.
  ///        Idempotent.
  void stop() noexcept;

  /// @brief Take the next set ready to hand out.
  /// @return The set; an empty optional when none is ready or the array is
  ///         not started; `Status::Code::InvalidArgument` on a moved-from
  ///         array; or the first sensor's failure.
  core::Result<std::optional<FrameSet>> poll_set();

  /// @brief @ref poll_set, waiting up to @p timeout for a set: blocked between
  ///        polls on the sensors' frame notifications, never for a fixed
  ///        sleep, and woken too when a trigger stops waiting for a late
  ///        sensor (one and a half frame periods of the slowest).
  /// @param timeout  The longest wait.
  /// @param metrics  Optional: `"wait: cameras"`, the time blocked, and
  ///                 `"poll"`, the polls' host time.
  /// @return The set; an empty optional when @p timeout passed with none ready
  ///         or the array is not started; `Status::Code::Unsupported` when a
  ///         sensor notifies nothing (@ref poll_set it instead);
  ///         `Status::Code::InvalidArgument` on a moved-from array; or the
  ///         first sensor's failure, which wakes the wait.
  core::Result<std::optional<FrameSet>> wait_set(
      std::chrono::milliseconds timeout, core::StageMetrics* metrics = nullptr);

  /// @brief Record preparing @p set into @p batch: every stream of every
  ///        sensor copied into its pass's slot, undistorted and converted
  ///        (`GpuFramePrep::record_set`), with nothing submitted.
  /// @param batch    Records the work; the caller submits it, and keeps @p set
  ///                 until it has resolved, since its pictures are read then.
  /// @param set      A set this array handed out.
  /// @param metrics  Optional: one `"frame prep"` row for the set, its host
  ///                 half now and its device half from
  ///                 `PendingBatch::device_rows`.
  /// @return The prepared set, whose buffers hold it once @p batch has run;
  ///         `Status::Code::InvalidArgument` on a moved-from array, one opened
  ///         without a device, or a set of another size; or what
  ///         `GpuFramePrep::record_set` refuses.
  core::Result<DeviceFrameSet> record_process(
      core::CommandBatch& batch, const FrameSet& set,
      core::StageMetrics* metrics = nullptr);

  /// @brief Prepare @p set on the array's device: @ref record_process, one
  ///        submit and its wait (`GpuFramePrep::prepare_batch`).
  ///
  /// Returns once the set is ready, so a caller fuses it as soon as this
  /// returns; a pass reuses its buffers once no `DeviceFrame` holds them.
  /// @param set      A set this array handed out.
  /// @param metrics  Optional: one `"frame prep"` row for the set.
  /// @return The prepared set; @ref record_process's refusals; otherwise a
  ///         buffer or submit failure.
  core::Result<DeviceFrameSet> process(const FrameSet& set,
                                       core::StageMetrics* metrics = nullptr);

  /// @return `true` on a moved-from array, or once a sensor is exhausted (a
  ///         disconnected camera).
  bool exhausted() const noexcept;

 private:
  struct Impl;
  explicit SensorArray(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
