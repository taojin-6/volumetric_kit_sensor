// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file realsense/realsense_sensor.hpp
/// @brief An Intel RealSense D400 camera (the D455) as an
///        @ref volumetric_kit::sensor::IRgbdSensor: its own depth and its
///        colour camera's YUYV, as captured, for the GPU pass.
///
/// No librealsense type appears here -- the SDK is held behind a pointer to an
/// implementation -- so a consumer includes this header without the SDK's
/// headers. Linking still needs the SDK's library, which the target carries.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/realsense/export.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief What a RealSense camera reported when it opened.
struct RealSenseDeviceInfo {
  std::string name;              ///< e.g. "RealSense D455".
  std::string serial;            ///< Its serial number.
  std::string firmware_version;  ///< e.g. "5.17.0.10".
  std::string usb_type;          ///< The link it enumerated on, e.g. "3.2".
};

/// @brief One RealSense D400 camera, polled for frames as it captured them.
///
/// Each frame is the camera's own depth (Z16, computed on the camera from its
/// infrared pair, in the left infrared camera: rectified, so no lens) and its
/// colour camera's picture as the camera sends it, YUYV on the host
/// (`YuvImage::host`), with each camera's factory model and the factory
/// depth-to-colour extrinsic. Nothing on the host undistorts, registers or
/// converts: `GpuFramePrep` uploads both and converts the colour on the
/// device. The timestamps are librealsense's global time, on the host's
/// `std::chrono::system_clock` (`ClockDomain::Host`).
///
/// A frame holds the librealsense frameset it was read from, so it outlives
/// the next poll; the SDK has the frames back once every copy of the frame is
/// gone. The SDK keeps a bounded pool of frames per stream, so a consumer that
/// holds many frames starves the camera: keep a few.
///
/// For depth from the infrared pair by Fast-FoundationStereo instead, with
/// the projector off, the camera opens as a stereo source; this class is the
/// camera's own depth, the projector on by default.
///
/// @code
/// RealSenseSensor::Options options;  // depth 1280x720, colour 1280x800
/// VKC_ASSIGN(RealSenseSensor sensor, RealSenseSensor::open(options));
/// VKC_ASSIGN(auto prep, GpuFramePrep::create(device, allocator));
/// VKC_TRY(sensor.start());
/// for (;;) {
///   VKC_ASSIGN(std::optional<RgbdFrame> frame, sensor.poll());
///   if (!frame) continue;
///   VKC_ASSIGN(const DeviceFrame prepared, prep.prepare(*frame));
/// }
/// @endcode
///
/// @warning Not thread-safe: open, start, poll, drain and stop from one
///          thread. The SDK delivers frames on a thread of its own, which
///          this class hands over internally.
class VS_REALSENSE_API RealSenseSensor final : public IRgbdSensor {
 public:
  /// @brief Which camera, which modes, and where it sits.
  struct Options {
    /// Serial number of the camera to open. Empty opens the only camera
    /// attached, and is refused when more than one is.
    std::string serial;
    std::uint32_t depth_width = 1280;  ///< The depth stream's width.
    std::uint32_t depth_height = 720;  ///< The depth stream's height.
    std::uint32_t color_width = 1280;  ///< The colour camera's width (YUYV).
    std::uint32_t color_height = 800;  ///< The colour camera's height.
    std::uint32_t fps = 30;            ///< Both streams' frame rate.
    /// Nearer depth is dropped (metres), as `RgbdFrame::min_depth`; above 0.
    float min_depth = 0.3f;
    /// Farther depth is dropped (metres), as `RgbdFrame::max_depth`.
    float max_depth = 6.0f;
    /// The infrared projector, whose dots give the camera's own stereo
    /// texture to match on blank surfaces.
    bool emitter = true;
    /// The colour camera's manual exposure (auto exposure off, auto white
    /// balance on), in its units: exposure in 100 us, gain 1 to 128 on the
    /// D455 (0 is refused); empty leaves it on auto exposure and auto white
    /// balance.
    std::optional<camera::ColorControls> color_controls;
    /// The colour camera's frame to the world's, stamped on every frame as
    /// `RgbdFrame::color_to_world`; rigid. Identity places the world at the
    /// camera.
    camera::Mat4d color_to_world = camera::Mat4d(1.0);
  };

  /// @brief Find the camera, check it streams the requested modes, read its
  ///        factory calibration, and set its projector and colour exposure.
  ///        Does not start streaming.
  /// @param options  The camera and its streams.
  /// @return The sensor, not yet started; or:
  ///         - `Status::Code::InvalidArgument` for options refused before the
  ///           camera is looked for -- a zero size or rate, a depth range
  ///           that is not finite, empty or starts at 0 (which the GPU pass
  ///           would refuse), `color_controls` with an exposure or a gain of
  ///           0, a @ref Options::color_to_world that is not rigid -- or an
  ///           empty @ref Options::serial with more than one camera attached;
  ///         - `Status::Code::NotFound` if no camera (or not the named one)
  ///           is attached, naming those that are;
  ///         - `Status::Code::Unsupported` for a camera on a USB 2 link, one
  ///           without the requested depth (Z16) or colour (YUYV) mode (the
  ///           modes it offers are listed), a factory calibration the GPU
  ///           pass cannot use (a lens on the depth stream, a colour lens
  ///           model other than Brown-Conrady, intrinsics that are not finite
  ///           and positive or for another size, an extrinsic that is not
  ///           rigid), no global timestamps, or a colour control it lacks or
  ///           whose range excludes the value (nothing written then);
  ///         - `Status::Code::IoError` for any other SDK failure, with its
  ///           message.
  static core::Result<RealSenseSensor> open(const Options& options);

  RealSenseSensor(RealSenseSensor&& other) noexcept;
  RealSenseSensor& operator=(RealSenseSensor&& other) noexcept;
  /// Stops the camera if it is streaming.
  ~RealSenseSensor() override;

  /// @return What the camera reported at @ref open: empty on a moved-from
  ///         sensor.
  const RealSenseDeviceInfo& device_info() const noexcept;

  /// @return What the sensor is: its serial, its cameras' factory models at
  ///         the opened modes, the factory extrinsic, a free-running role on
  ///         the host's clock, its rate and any colour controls read back.
  ///         Empty on a moved-from sensor.
  const SensorInfo& info() const noexcept override;
  /// @return As `IRgbdSensor::set_queue_depth`; also
  ///         `Status::Code::InvalidArgument` on a moved-from sensor.
  core::Status set_queue_depth(std::size_t frames) override;
  /// @return OK once streaming, and if already streaming;
  ///         `Status::Code::InvalidArgument` on a moved-from sensor;
  ///         `Status::Code::IoError` if the SDK refuses or the camera has
  ///         disconnected.
  core::Status start() override;
  /// @brief As `IRgbdSensor::stop`. The camera stays open, for another
  ///        @ref start.
  void stop() noexcept override;
  /// @return As `IRgbdSensor::poll`; `Status::Code::IoError` once the camera
  ///         disconnected, or for a frame that contradicts its stream.
  core::Result<std::optional<RgbdFrame>> poll() override;
  /// @return As @ref poll; on a failure, the frames before it stay
  ///         appended, and those after it are counted dropped.
  core::Status drain(std::vector<RgbdFrame>* out) override;
  /// @return `true` on a moved-from sensor and once the camera has
  ///         disconnected.
  bool exhausted() const noexcept override;
  /// @return The counters since the last @ref start, kept through a
  ///         @ref stop: `SensorStats::failed` counts the SDK's framesets
  ///         without depth or colour, or whose depth and colour lie more than
  ///         half a frame apart, as some do in the first second, and frames
  ///         @ref poll or @ref drain refused. Zero on a moved-from sensor.
  SensorStats stats() const noexcept override;
  /// @brief As `IRgbdSensor::set_frame_notify`: called from the SDK's thread
  ///        as each frame reaches the queue, and when the camera
  ///        disconnects. Each frame's `RgbdFrame::arrival_ns` is the host's
  ///        system clock as the SDK hands the frameset over.
  /// @return As `IRgbdSensor::set_frame_notify`, never
  ///         `Status::Code::Unsupported`; also
  ///         `Status::Code::InvalidArgument` on a moved-from sensor.
  core::Status set_frame_notify(std::function<void()> notify) override;

 private:
  struct Impl;
  explicit RealSenseSensor(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer so no SDK type reaches this header.
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
