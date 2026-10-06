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
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"
#include "volumetric_kit/recon/sensor/array/export.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

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
  /// deterministic. A frame for a set already handed out joins none, and a
  /// sensor ahead of one that has fallen behind holds
  /// @ref SensorArray::Options::queue_depth frames, letting its oldest go.
  Sequence,
};

/// @return A stable lowercase name for @p mode, for logs.
VR_SENSOR_ARRAY_API const char* to_string(SyncMode mode) noexcept;

/// @brief The frames one trigger produced across the array.
struct FrameSet {
  /// The trigger's time (ns): the primary's frame's for `SyncMode::Trigger`;
  /// for `SyncMode::Sequence`, the frame's of the first sensor present, in
  /// the array's order.
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
/// @ref open checks that the sensors can form one under the chosen
/// @ref SyncMode, and that the calibration poses each of them. @ref start
/// starts every secondary before the primary -- the primary's first trigger
/// is what they wait for. @ref poll_set drains every sensor's frames, groups
/// them, and hands out a set, each frame posed by the calibration;
/// @ref process prepares a set on the GPU, every stream in one batch.
///
/// @code
/// std::vector<std::unique_ptr<IRgbdSensor>> sensors;
/// for (const std::string& serial : serials) {
///   OrbbecSensor::Options o;
///   o.serial = serial;
///   o.sync_clock_to_host = true;  // Trigger groups on the host clock
///   VKC_ASSIGN(OrbbecSensor s, OrbbecSensor::open(o));
///   sensors.push_back(std::make_unique<OrbbecSensor>(std::move(s)));
/// }
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
    /// The device @ref process prepares sets on, with a pass per sensor;
    /// null for an array whose sets are not prepared here. Borrowed: it must
    /// outlive the array and every set it prepares.
    core::Device* device = nullptr;
    /// The allocator the passes' buffers come from; given with @ref device
    /// and only with it. Borrowed, as @ref device is.
    core::Allocator* allocator = nullptr;
    /// The passes' options: the queue families sharing their colour, and
    /// whether depth is kept only where colour was recorded
    /// (`GpuFramePrepConfig`).
    GpuFramePrepConfig prep{};
  };

  /// @brief Check that @p sensors form an array, and take them.
  ///
  /// None is started. Under `SyncMode::Trigger`, more than one sensor needs
  /// exactly one primary, the rest secondaries, all on the host's clock;
  /// under `SyncMode::Sequence` any sensors do.
  /// @param sensors  The sensors, in the order sets report them; taken only
  ///                 once the array opens, so a refusal leaves them open
  ///                 with the caller.
  /// @param options  How they are grouped and posed.
  /// @return The array; `Status::Code::InvalidArgument` for no sensors, a
  ///         null one, an empty or repeated id, a queue depth of 0, a
  ///         calibration that is invalid or does not pose a sensor, a device
  ///         or an allocator without the other, or, under `SyncMode::Trigger`,
  ///         a tolerance of 0 or of half a frame period or more; what
  ///         `GpuFramePrep::create` returns for a device the passes cannot be
  ///         built on; or `Status::Code::Unsupported` for a tracked sensor,
  ///         or, under `SyncMode::Trigger`, no primary or more than one, a
  ///         free-running member, or one whose timestamps are on its own
  ///         clock.
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
  /// @return The index of the primary under `SyncMode::Trigger`; 0 otherwise.
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

  /// @brief Prepare @p set on the array's device: every stream of every
  ///        sensor undistorted and converted in one batch, one submit and
  ///        one wait for the set (`GpuFramePrep::prepare_batch`).
  ///
  /// Returns once the set is ready, so a caller fuses it as soon as this
  /// returns; a pass reuses its buffers once no `DeviceFrame` holds them.
  /// @param set      A set this array handed out.
  /// @param metrics  Optional: one `"frame prep"` row for the set.
  /// @return The prepared set; `Status::Code::InvalidArgument` on a
  ///         moved-from array, one opened without a device, or a set of
  ///         another size; what `GpuFramePrep::prepare` returns for a frame
  ///         it refuses; otherwise a buffer or submit failure.
  core::Result<DeviceFrameSet> process(const FrameSet& set,
                                       core::StageMetrics* metrics = nullptr);

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
