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
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"
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
/// codec is decoded here, so this needs a build with VR_WITH_FFMPEG.
///
/// A frame holds the SDK pair it was read from, so it outlives the next
/// poll; the SDK has the pair's buffers back once every copy of the frame is
/// gone. A camera wired as a sync secondary starts cleanly and then delivers
/// nothing unless its primary streams (`SensorInfo::role` says so). A camera
/// the SDK reports removed never streams again through this object.
///
/// @code
/// OrbbecSensor::Options options;
/// options.serial = "CL2A141000N";
/// options.color_codec = OrbbecColorCodec::Hevc;
/// options.device = &device;  // decode onto the GPU the pass runs on
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
  ///
  /// `OrbbecStreamOptions::raw` is ignored: this sensor's frames are always
  /// as captured.
  struct Options : OrbbecStreamOptions {
    /// Serial number of the camera to open. Empty opens the only camera that
    /// answers, after waiting out all of @ref discovery_timeout_ms, and is
    /// refused when more than one does.
    std::string serial;
    /// How long @ref open re-queries the network for the camera.
    std::uint32_t discovery_timeout_ms = 8000;
    /// The colour camera's frame to the world's, stamped on every frame as
    /// `RgbdFrame::color_to_world`; rigid. Identity places the world at the
    /// camera.
    camera::Mat4d color_to_world = camera::Mat4d(1.0);
    /// Keep the camera's clock on the host's, re-synced this often (ms), so
    /// its timestamps compare with other sensors' (`ClockDomain::Host`). 0
    /// leaves it on its own clock.
    std::uint32_t clock_sync_interval_ms = 0;
    /// As `OrbbecCapture::Options::configure_sdk_logging`.
    bool configure_sdk_logging = true;
  };

  /// @brief Find the camera, check it can stream the requested modes, and
  ///        read its factory calibration. Does not start streaming.
  /// @param options  The camera and its streams.
  /// @return The sensor; or what `OrbbecCapture::open` returns for these
  ///         streams opened raw, and `Status::Code::InvalidArgument` also for
  ///         a @ref Options::color_to_world that is not rigid.
  static core::Result<OrbbecSensor> open(const Options& options);

  OrbbecSensor(OrbbecSensor&& other) noexcept;
  OrbbecSensor& operator=(OrbbecSensor&& other) noexcept;
  /// Stops the camera if it is streaming.
  ~OrbbecSensor() override;

  /// @return What the camera reported at @ref open: empty on a moved-from
  ///         sensor.
  const OrbbecDeviceInfo& device_info() const noexcept;
  /// @return The driver's own counters, which split `SensorStats::failed`
  ///         into failed and lost and count the pictures a device-bound
  ///         stream handed out on the host.
  OrbbecCaptureStats orbbec_stats() const noexcept;

  /// @return What the sensor is: its serial, its cameras' factory models at
  ///         the opened modes, the factory extrinsic, its rig role, its
  ///         clock. Empty on a moved-from sensor.
  const SensorInfo& info() const noexcept override;
  /// @return As `IRgbdSensor::set_queue_depth`; also
  ///         `Status::Code::InvalidArgument` on a moved-from sensor.
  core::Status set_queue_depth(std::size_t frames) override;
  /// @return OK once streaming; `Status::Code::InvalidArgument` on a
  ///         moved-from sensor; or what `OrbbecCapture::start` returns, and
  ///         `Status::Code::IoError` if the SDK will not sync the clock.
  core::Status start() override;
  void stop() noexcept override;
  /// @return As `IRgbdSensor::poll`; `Status::Code::IoError` once the camera
  ///         disconnected or its frames stopped processing, or for a pair
  ///         that contradicts the stream; `Status::Code::Unsupported` for a
  ///         stream whose transfer or primaries `ColorEncoding` cannot name.
  core::Result<std::optional<RgbdFrame>> poll() override;
  /// @return As @ref poll; on a failure, the frames not yet handed out are
  ///         counted dropped.
  core::Status drain(std::vector<RgbdFrame>* out) override;
  /// @return `true` on a moved-from sensor and once the camera has
  ///         disconnected.
  bool exhausted() const noexcept override;
  SensorStats stats() const noexcept override;

 private:
  struct Impl;
  explicit OrbbecSensor(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer so no SDK type reaches this header.
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
