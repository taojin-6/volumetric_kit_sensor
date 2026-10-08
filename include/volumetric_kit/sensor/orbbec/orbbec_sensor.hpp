// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/orbbec/orbbec_sensor.hpp
/// @brief An Orbbec RGB-D camera (the Femto Mega) as an
///        @ref volumetric_kit::recon::sensor::IRgbdSensor: frames as it
///        captured them, for the GPU pass.
///
/// No Orbbec SDK type appears here -- the SDK is held behind a pointer to an
/// implementation -- so a consumer includes this header without the SDK's
/// headers. Linking still needs the SDK's library, which the target carries.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/sensor/orbbec/export.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_stream.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief One Orbbec RGB-D camera, polled for frames as it captured them.
///
/// Each frame is the camera's raw depth and its colour as the decoder left
/// it -- on the device where the hardware decoded it -- with each camera's
/// factory model and the factory depth-to-colour extrinsic (its rotation
/// made one; the 2026-10-06 sensor-frame decision). Nothing on the host
/// undistorts, registers or converts: `sensor/utils`'s GPU pass does. Either
/// codec is decoded here, by the video decoders VR_WITH_ORBBEC builds with.
///
/// A frame holds the SDK pair it was read from, so it outlives the next
/// poll; the SDK has the pair's buffers back once every copy of the frame is
/// gone. A camera wired as a sync secondary starts cleanly and then delivers
/// nothing unless its primary streams (`SensorInfo::role` says so). A camera
/// the SDK reports removed never streams again through this object.
///
/// A synchronised rig is one sensor per camera in a @ref SensorArray, each
/// opened with its entry of the rig's sync configuration (@ref Options::sync)
/// and on the host's clock (@ref Options::sync_clock_to_host):
/// @ref open_orbbec_sensors opens them so.
///
/// @code
/// OrbbecSensor::Options options;
/// options.serial = "CL2A141000N";
/// options.color_codec = OrbbecColorCodec::Hevc;
/// options.device = &device;  // decode onto the GPU the pass runs on
/// options.allocator = &allocator;  // NVDEC's and nvJPEG's pictures
/// VKC_ASSIGN(OrbbecSensor sensor, OrbbecSensor::open(options));
/// VKC_TRY(sensor.start());
/// @endcode
///
/// @warning Not thread-safe: open, start, poll, drain and stop from one
///          thread. The SDK delivers frames on a thread of its own, which
///          this class hands over internally.
class VR_SENSOR_ORBBEC_API OrbbecSensor final : public IRgbdSensor {
 public:
  /// @brief Which camera, which streams, and where it sits.
  struct Options : OrbbecStreamOptions {
    /// Serial number of the camera to open. Empty opens the only camera that
    /// answers, after waiting out all of @ref discovery_timeout_ms, and is
    /// refused when more than one does.
    std::string serial;
    /// How long @ref open re-queries the network for the camera. An Ethernet
    /// camera can take seconds to answer from cold, so one query is not proof
    /// of absence.
    std::uint32_t discovery_timeout_ms = 8000;
    /// The colour camera's frame to the world's, stamped on every frame as
    /// `RgbdFrame::color_to_world`; rigid. Identity places the world at the
    /// camera.
    camera::Mat4d color_to_world = camera::Mat4d(1.0);
    /// This camera's entry of its rig's sync configuration
    /// (@ref read_orbbec_sync_config). @ref open compares it with the
    /// settings stored on the camera, and refuses a camera that differs
    /// unless @ref apply_sync. Empty takes the camera as it is set.
    std::optional<OrbbecSyncSettings> sync;
    /// Write @ref sync to the camera where its stored settings differ, rather
    /// than refusing it. They persist in the camera's flash, so they are
    /// written only on a difference. Refused if the SDK's effective settings
    /// still differ after the write; this checks SDK normalization, not
    /// persistence on the camera.
    bool apply_sync = false;
    /// Stamp each frame with the SDK's global timestamp -- the camera's clock
    /// mapped onto the host's `std::chrono::system_clock` (the SDK's default
    /// host clock, which recon never changes), the mapping re-fitted as the
    /// two drift -- so it compares with other sensors' (`ClockDomain::Host`);
    /// otherwise with the camera's own clock. Each @ref start also sets the
    /// camera's clock to the host's, once, before it streams, so its first
    /// frames are mapped.
    bool sync_clock_to_host = false;
    /// Switch off the SDK's log file (it writes `./Log/` at DEBUG by default)
    /// and route its console sink at WARN, at @ref open, and set FFmpeg's log
    /// level to ERROR at the first @ref start. Process-wide: the SDK and
    /// FFmpeg have one logger each, so an application configuring either
    /// itself turns this off.
    bool configure_sdk_logging = true;
  };

  /// @brief Find the camera, check it can stream the requested modes, and
  ///        read its factory calibration. Does not start streaming.
  /// @param options  The camera and its streams.
  /// @return The sensor, not yet started; or:
  ///         - `Status::Code::InvalidArgument` for options refused before the
  ///           camera is looked for -- a zero size or rate, a depth range
  ///           that is not finite, empty or starts at 0 (which the GPU pass
  ///           would refuse), a @ref Options::color_to_world that is not
  ///           rigid, @ref Options::apply_sync without @ref Options::sync --
  ///           an empty @ref Options::serial with more than one camera
  ///           answering within the discovery window, or a factory
  ///           calibration that cannot be used: intrinsics zeroed, not finite
  ///           or for another size than the mode, or a depth-to-colour
  ///           extrinsic zeroed or reflected, as an uncalibrated unit reports;
  ///         - `Status::Code::NotFound` if no camera (or not the named one)
  ///           answered within @ref Options::discovery_timeout_ms, naming the
  ///           cameras that did;
  ///         - `Status::Code::Unsupported` if the camera has no depth or
  ///           colour mode matching the options (the modes it offers are
  ///           listed), reports its image mirrored, flipped or rotated, is in
  ///           software-triggering mode or a sync mode this driver does not
  ///           know, reports a lens model the GPU pass cannot undistort,
  ///           has sync settings that differ from @ref Options::sync (each
  ///           field named) without @ref Options::apply_sync or after the SDK
  ///           applies them, or has no global timestamps for
  ///           @ref Options::sync_clock_to_host;
  ///         - `Status::Code::IoError` for any other SDK failure, writing
  ///           the sync settings included, with the SDK's message.
  static core::Result<OrbbecSensor> open(const Options& options);

  OrbbecSensor(OrbbecSensor&& other) noexcept;
  OrbbecSensor& operator=(OrbbecSensor&& other) noexcept;
  /// Stops the camera if it is streaming.
  ~OrbbecSensor() override;

  /// @return What the camera reported at @ref open: empty on a moved-from
  ///         sensor.
  const OrbbecDeviceInfo& device_info() const noexcept;
  /// @return The driver's own counters, which split `SensorStats::failed`
  ///         into failed and lost.
  OrbbecStreamStats orbbec_stats() const noexcept;

  /// @return What the sensor is: its serial, its cameras' factory models at
  ///         the opened modes, the factory extrinsic, its rig role, its
  ///         clock. Empty on a moved-from sensor.
  const SensorInfo& info() const noexcept override;
  /// @return As `IRgbdSensor::set_queue_depth`; also
  ///         `Status::Code::InvalidArgument` on a moved-from sensor.
  core::Status set_queue_depth(std::size_t frames) override;
  /// @return OK once streaming, and if already streaming;
  ///         `Status::Code::InvalidArgument` on a moved-from sensor;
  ///         `Status::Code::IoError` if the SDK refuses, the camera has
  ///         disconnected or cannot sync its clock
  ///         (@ref Options::sync_clock_to_host), or the colour decoder's
  ///         thread will not start; and the decoder's error if it does not
  ///         open: `Status::Code::Unsupported` where it has no device path
  ///         (no `OrbbecStreamOptions::device`, say).
  core::Status start() override;
  /// @brief As `IRgbdSensor::stop`, letting go of the frames waiting for
  ///        the colour decoder too. The camera stays open, for another
  ///        @ref start.
  void stop() noexcept override;
  /// @return As `IRgbdSensor::poll`; `Status::Code::IoError` once the camera
  ///         disconnected or its frames stopped processing, or for a pair
  ///         that contradicts the stream; `Status::Code::Unsupported` for a
  ///         stream whose transfer or primaries `ColorEncoding` cannot name,
  ///         and once the colour decoder stops for a stream the hardware
  ///         cannot decode; `Status::Code::Backend` or
  ///         `Status::Code::OutOfMemory` once its device path fails.
  core::Result<std::optional<RgbdFrame>> poll() override;
  /// @return As @ref poll; on a failure, the frames before it stay
  ///         appended, and those after it are counted dropped.
  core::Status drain(std::vector<RgbdFrame>* out) override;
  /// @return `true` on a moved-from sensor and once the camera has
  ///         disconnected.
  bool exhausted() const noexcept override;
  /// @return @ref orbbec_stats, its lost frames -- colour that did not
  ///         decode, or came without depth -- counted failed. Zero on a
  ///         moved-from sensor.
  SensorStats stats() const noexcept override;

 private:
  struct Impl;
  explicit OrbbecSensor(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer so no SDK type reaches this header.
  std::unique_ptr<Impl> impl_;
};

/// @brief Open every camera of a rig's sync configuration file, in the
///        file's order, as the sensors a @ref SensorArray takes.
///
/// Each camera is opened by @ref OrbbecSensor::open with @p options, its
/// @ref OrbbecSensor::Options::serial and @ref OrbbecSensor::Options::sync
/// taken from its entry of the file, and on the host's clock
/// (@ref OrbbecSensor::Options::sync_clock_to_host), on which an array
/// groups several sensors. So a camera whose stored settings differ from its
/// entry is refused unless @ref OrbbecSensor::Options::apply_sync writes
/// them. The file is only read.
///
/// @code
/// OrbbecSensor::Options streams;
/// streams.color_codec = OrbbecColorCodec::Hevc;
/// streams.device = &device;
/// streams.allocator = &allocator;
/// VKC_ASSIGN(auto sensors, open_orbbec_sensors(sync_path, streams));
/// VKC_ASSIGN(SensorArray array, SensorArray::open(std::move(sensors),
///                                                 array_options));
/// @endcode
///
/// @param sync_path  The rig's sync configuration, as
///                   @ref read_orbbec_sync_config reads it.
/// @param options    What every camera streams and how it is found; its
///                   `serial` and `sync` left empty, since the file names
///                   them, and its `sync_clock_to_host` ignored.
/// @return One @ref OrbbecSensor per camera, in the file's order; or
///         - `Status::Code::InvalidArgument` for @p options naming a serial
///           or sync settings, before the file is read;
///         - what @ref read_orbbec_sync_config returns for a file it cannot
///           read or parse;
///         - the first camera's refusal by @ref OrbbecSensor::open, which
///           names the camera (`Status::Code::NotFound` for one that does not
///           answer), the cameras opened before it closed again.
VR_SENSOR_ORBBEC_API core::Result<std::vector<std::unique_ptr<IRgbdSensor>>>
open_orbbec_sensors(const std::string& sync_path,
                    const OrbbecSensor::Options& options);

}  // namespace volumetric_kit::recon::sensor
