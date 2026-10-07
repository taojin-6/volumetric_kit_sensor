// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/rgbd_sensor.hpp
/// @brief One RGB-D sensor -- a Femto Mega, an iPhone, a cine camera -- as
///        every driver presents it, used on its own or in a sensor array.
///
/// A driver implements @ref IRgbdSensor and hands out @ref RgbdFrame s as the
/// sensor captured them; `sensor/utils`'s GPU pass undistorts and converts
/// them, so no driver writes either. What a driver knows about its sensor
/// before a frame exists -- its cameras' factory models for the active mode,
/// its extrinsic, its role in a synchronised rig, which clock its timestamps
/// are on -- is its @ref SensorInfo, which a sensor array reads to start the
/// sensors in order and group their frames.
///
/// A driver ships in this repo only if this repo can build and test it (the
/// 2026-08-02 decision); a platform-bound one (ARKit) implements this from
/// its application's repo. Like the frame, this header reaches no Vulkan.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/sensor/export.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief A sensor's role in a hardware-synchronised rig, as the sensor
///        reports it.
enum class SyncRole : std::uint8_t {
  FreeRun,    ///< Not synchronised: captures on its own clock.
  Primary,    ///< Captures on its own and drives the sync line.
  Secondary,  ///< Captures only on the primary's signal.
};

/// @brief Which clock a sensor's timestamps are on.
enum class ClockDomain : std::uint8_t {
  /// The sensor's own: comparable only within its frames.
  Device,
  /// The host's, kept so by the driver: comparable with every sensor on it.
  Host,
};

/// @brief Where a frame's `RgbdFrame::color_to_world` comes from.
enum class PoseSource : std::uint8_t {
  /// The sensor sits still: a calibration poses it, and the driver stamps
  /// the pose it was given.
  Fixed,
  /// The sensor tracks itself (ARKit), or a recording carries its
  /// trajectory: each frame carries its own pose.
  Tracked,
};

/// @return A stable lowercase name for @p role, for logs.
VR_SENSOR_API const char* to_string(SyncRole role) noexcept;
/// @return A stable lowercase name for @p clock, for logs.
VR_SENSOR_API const char* to_string(ClockDomain clock) noexcept;
/// @return A stable lowercase name for @p source, for logs.
VR_SENSOR_API const char* to_string(PoseSource source) noexcept;

/// @brief What a sensor is, known from when it opens: before a frame.
///
/// @code
/// const SensorInfo& info = sensor.info();
/// if (info.color && info.depth) {
///   // The array poses the depth camera at color_to_world * depth_to_color.
///   const camera::Mat4d& extrinsic = info.depth_to_color;
/// }
/// @endcode
struct SensorInfo {
  std::string id;     ///< Its serial: how a calibration file names it.
  std::string model;  ///< What it is, e.g. "Orbbec Femto Mega".
  /// The colour camera's factory model at the mode it streams; empty for a
  /// sensor without colour. Every frame's `RgbdFrame::color_camera` is this
  /// unless the sensor refines it per frame (a tracked one).
  std::optional<camera::CameraModel> color;
  /// The depth camera's factory model at the mode it streams; empty for a
  /// sensor without depth.
  std::optional<camera::CameraModel> depth;
  /// The depth camera's frame to the colour camera's: the factory extrinsic.
  camera::Mat4d depth_to_color = camera::Mat4d(1.0);
  SyncRole role = SyncRole::FreeRun;        ///< Its role in a rig.
  ClockDomain clock = ClockDomain::Device;  ///< Its timestamps' clock.
  PoseSource pose = PoseSource::Fixed;      ///< Where its poses come from.
  std::uint32_t fps = 0;  ///< Frames a second it streams at; 0 if unknown.
};

/// @brief Counters a caller reads to see whether it keeps up with a sensor.
///
/// Each frame the sensor received is counted once:
/// `delivered + dropped + failed <= received`, the difference being frames
/// still held or discarded by a stop.
struct SensorStats {
  std::uint64_t received = 0;   ///< Frames the sensor produced since start.
  std::uint64_t delivered = 0;  ///< Frames handed out by a poll or a drain.
  /// Frames let go unseen: replaced by a newer one before a poll, or pushed
  /// out of a full queue (@ref IRgbdSensor::set_queue_depth).
  std::uint64_t dropped = 0;
  /// Frames that could not be handed out: a decode or SDK failure, or one
  /// the driver refused.
  std::uint64_t failed = 0;
};

/// @brief One RGB-D sensor, polled for the frames it captures.
///
/// **Polled, not called back**, as the rest of the capture code is: a sensor
/// that delivers on a thread of its own holds what arrives, and @ref poll or
/// @ref drain hands it over on the caller's thread. It holds up to
/// @ref set_queue_depth frames, the oldest let go when a newer one arrives on
/// a full queue. A standalone consumer @ref poll s the newest; a sensor array
/// sets a deeper queue and @ref drain s every frame, oldest first, so it can
/// group frames of several sensors by trigger.
///
/// A frame holds its pixels (`RgbdFrame::pixels`), so it may be kept past the
/// next poll, at the cost of the driver's buffers it holds.
///
/// @code
/// VKC_TRY(sensor.start());
/// for (;;) {
///   VKC_ASSIGN(std::optional<RgbdFrame> frame, sensor.poll());
///   if (!frame) {
///     if (sensor.exhausted()) break;  // a recording's end
///     continue;                       // a live sensor's "not yet"
///   }
///   VKC_ASSIGN(const DeviceFrame prepared, prep.prepare(*frame));
/// }
/// @endcode
///
/// @warning Not thread-safe unless an implementation says otherwise: open,
///          start, poll, drain and stop from one thread.
class VR_SENSOR_API IRgbdSensor {
 public:
  virtual ~IRgbdSensor() = default;

  IRgbdSensor(const IRgbdSensor&) = delete;
  IRgbdSensor& operator=(const IRgbdSensor&) = delete;

  /// @return What the sensor is, from when it opened.
  virtual const SensorInfo& info() const noexcept = 0;

  /// @brief How many frames the sensor holds for the next @ref poll or
  ///        @ref drain; 1, the default, holds only the newest.
  /// @param frames  At least 1.
  /// @return OK; `Status::Code::InvalidArgument` for 0, or once started.
  virtual core::Status set_queue_depth(std::size_t frames) = 0;

  /// @brief Begin capturing. Idempotent: starting a running sensor is OK.
  /// @return OK once running, or why the sensor could not start.
  virtual core::Status start() = 0;

  /// @brief Stop capturing and drop the frames held. Idempotent, and safe on
  ///        a sensor that never started: the destructor's fallback.
  virtual void stop() noexcept = 0;

  /// @brief Take the newest frame held, letting the older ones go.
  /// @return The frame; an empty optional (@ref no_frame) when none is held
  ///         -- the ordinary case for a consumer polling faster than the
  ///         sensor runs, not an error; or why the sensor failed.
  virtual core::Result<std::optional<RgbdFrame>> poll() = 0;

  /// @brief Take every frame held, oldest first.
  /// @param out  Receives the frames, appended; not null.
  /// @return OK, with none appended when none is held; or why the sensor
  ///         failed, the frames before the failure appended and counted
  ///         delivered.
  virtual core::Status drain(std::vector<RgbdFrame>* out) = 0;

  /// @brief Whether this sensor will never hand out another frame.
  ///
  /// An empty @ref poll says only "nothing this tick"; a finite source -- a
  /// recording -- overrides this to say its sequence is over, and a live one
  /// keeps the default until it disconnects.
  /// @return `true` once no later poll can return a frame.
  virtual bool exhausted() const noexcept { return false; }

  /// @return The counters since the last @ref start.
  virtual SensorStats stats() const noexcept = 0;

  /// @brief The "nothing this tick" return: `return std::nullopt;` needs two
  ///        user-defined conversions and does not compile, and
  ///        `return core::Status{};` aborts, an OK status not being a
  ///        failure.
  /// @return An OK `Result` holding an empty optional.
  static core::Result<std::optional<RgbdFrame>> no_frame() {
    return std::optional<RgbdFrame>{};
  }

  /// @brief The "here is the frame" return, the counterpart to
  ///        @ref no_frame.
  /// @param frame  The frame to hand over.
  /// @return An OK `Result` holding @p frame.
  static core::Result<std::optional<RgbdFrame>> some_frame(RgbdFrame frame) {
    return std::optional<RgbdFrame>{std::move(frame)};
  }

 protected:
  IRgbdSensor() = default;
  // Move operations stay available to implementations but are not part of
  // the polymorphic interface: callers hold a sensor behind a pointer.
  IRgbdSensor(IRgbdSensor&&) = default;
  IRgbdSensor& operator=(IRgbdSensor&&) = default;
};

}  // namespace volumetric_kit::recon::sensor
