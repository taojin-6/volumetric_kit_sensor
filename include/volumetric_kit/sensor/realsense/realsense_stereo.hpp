// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file realsense/realsense_stereo.hpp
/// @brief An Intel RealSense D400 camera's infrared pair (the D455's) as an
///        @ref volumetric_kit::sensor::IStereoSensor: the two infrared
///        pictures as the camera rectified them, and its colour camera's
///        YUYV, for depth from a stereo matcher such as Fast-FoundationStereo.
///
/// No librealsense type appears here -- the SDK is held behind a pointer to an
/// implementation, the one @ref volumetric_kit::sensor::RealSenseSensor runs
/// on -- so a consumer includes this header without the SDK's headers.
/// Linking still needs the SDK's library, which the target carries.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/realsense/export.hpp"
#include "volumetric_kit/sensor/realsense/realsense_sensor.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief One RealSense D400 camera's infrared pair, polled for pairs as it
///        captured them.
///
/// Each pair is the left and right infrared cameras' 8-bit pictures (Y8) as
/// the camera sends them, rectified on the camera, as luma on the host
/// (`YuvLayout::Luma`, full range, @ref StereoFrame::left and
/// @ref StereoFrame::right), and its colour camera's picture as
/// `RealSenseSensor`'s, YUYV on the host (@ref StereoFrame::color). The
/// frame holds the librealsense frameset it was read from, so nothing is
/// copied or converted on the host: `StereoRectifier` copies the pictures to
/// the device once and resamples them there.
///
/// @ref info is the factory calibration: both infrared cameras a pinhole (the
/// camera rectified them, so one camera model, no lens, no rotation between
/// them), the left one's point in the right camera (the baseline, about
/// 95 mm on the D455, along +x) and in the colour camera, and the colour
/// camera with its lens. The timestamps are librealsense's global time, on
/// the host's `std::chrono::system_clock`, the left infrared picture's.
///
/// The projector is off by default: a stereo matcher sees the scene's own
/// texture, which the projector's dots would cover. The camera's own depth is
/// `RealSenseSensor`, the projector on.
///
/// It holds the newest pair, which @ref poll takes; a pair passed over is
/// counted dropped. The SDK keeps a bounded pool of frames per camera, so a
/// consumer that holds many pairs starves the camera: keep a few.
///
/// @code
/// RealSenseStereo::Options options;  // infrared 1280x720, colour 1280x800
/// VKC_ASSIGN(RealSenseStereo camera, RealSenseStereo::open(options));
/// VKC_ASSIGN(auto rectifier,
///            StereoRectifier::create(device, allocator, camera.info(),
///                                    {{960, 544}, 0.0f,
///                                     StereoRectifier::Output::Luma8}));
/// VKC_TRY(camera.start());
/// for (;;) {
///   VKC_ASSIGN(std::optional<StereoFrame> frame, camera.poll());
///   if (!frame) continue;
///   VKC_ASSIGN(RectifiedPair pair, rectifier->rectify(*frame));
/// }
/// @endcode
///
/// @warning Not thread-safe: open, start, poll and stop from one thread. The
///          SDK delivers pairs on a thread of its own, which this class hands
///          over internally.
class VS_REALSENSE_API RealSenseStereo final : public IStereoSensor {
 public:
  /// @brief Which camera, which modes, and where it sits.
  struct Options {
    /// Serial number of the camera to open. Empty opens the only camera
    /// attached, and is refused when more than one is.
    std::string serial;
    std::uint32_t infrared_width = 1280;  ///< Each infrared camera's width.
    std::uint32_t infrared_height = 720;  ///< Each infrared camera's height.
    std::uint32_t color_width = 1280;     ///< The colour camera's width.
    std::uint32_t color_height = 800;     ///< The colour camera's height.
    std::uint32_t fps = 30;               ///< All three cameras' rate.
    /// The infrared projector, whose dots a stereo matcher does not need and
    /// which cover the scene's own texture: off.
    bool emitter = false;
    /// The colour camera's manual exposure, as `RealSenseSensor`'s; empty
    /// leaves it on auto exposure.
    std::optional<camera::ColorControls> color_controls;
    /// The colour camera's frame to the world's, stamped on every pair as
    /// @ref StereoFrame::color_to_world; rigid. Identity places the world at
    /// the camera.
    camera::Mat4d color_to_world = camera::Mat4d(1.0);
  };

  /// @brief Find the camera, check it offers the requested modes, read its
  ///        factory calibration, and set its projector and colour exposure.
  ///        Does not start streaming.
  /// @param options  The camera and its modes.
  /// @return The camera, not yet started; or:
  ///         - `Status::Code::InvalidArgument` for options refused before the
  ///           camera is looked for -- a zero size or rate, `color_controls`
  ///           with an exposure or a gain of 0, a
  ///           @ref Options::color_to_world that is not rigid -- or an empty
  ///           @ref Options::serial with more than one camera attached;
  ///         - `Status::Code::NotFound` if no camera (or not the named one)
  ///           is attached, naming those that are;
  ///         - `Status::Code::Unsupported` for a camera on a USB 2 link, one
  ///           without the requested infrared (Y8, both cameras) or colour
  ///           (YUYV) mode (the modes it offers are listed), a factory
  ///           calibration the stereo tier cannot use (a lens on either
  ///           infrared camera, a colour lens model other than
  ///           Brown-Conrady, intrinsics that are not finite and positive,
  ///           an extrinsic that is not rigid), no global timestamps, a
  ///           projector to switch on that it lacks, or a colour control it
  ///           lacks or whose range excludes the value (nothing written
  ///           then);
  ///         - `Status::Code::IoError` for any other SDK failure, with its
  ///           message.
  static core::Result<RealSenseStereo> open(const Options& options);

  RealSenseStereo(RealSenseStereo&& other) noexcept;
  RealSenseStereo& operator=(RealSenseStereo&& other) noexcept;
  /// Stops the camera if it is streaming.
  ~RealSenseStereo() override;

  /// @return What the camera reported at @ref open: empty on a moved-from
  ///         camera.
  const RealSenseDeviceInfo& device_info() const noexcept;

  /// @return The pair's factory calibration at the opened modes, and the
  ///         colour camera's (see the class). Empty on a moved-from camera.
  const StereoInfo& info() const noexcept override;
  /// @return OK once streaming, and if already streaming;
  ///         `Status::Code::InvalidArgument` on a moved-from camera;
  ///         `Status::Code::IoError` if the SDK refuses or the camera has
  ///         disconnected.
  core::Status start() override;
  /// @brief As `IStereoSensor::stop`. The camera stays open, for another
  ///        @ref start.
  void stop() noexcept override;
  /// @return As `IStereoSensor::poll`; `Status::Code::IoError` once the
  ///         camera disconnected, or for a frameset that contradicts the
  ///         opened modes.
  core::Result<std::optional<StereoFrame>> poll() override;
  /// @return The counters since the last @ref start, kept through a
  ///         @ref stop: `SensorStats::failed` counts the SDK's framesets that
  ///         miss a picture, or whose pictures lie more than half a frame
  ///         apart, as some do in the first second, and those @ref poll
  ///         refused. Zero on a moved-from camera.
  SensorStats stats() const noexcept override;
  /// @brief As `IStereoSensor::set_frame_notify`: called from the SDK's
  ///        thread as each pair is held, and when the camera disconnects.
  ///        Each pair's @ref StereoFrame::arrival_ns is the host's system
  ///        clock as the SDK hands the frameset over.
  /// @return OK, never `Status::Code::Unsupported`;
  ///         `Status::Code::InvalidArgument` once started or on a moved-from
  ///         camera.
  core::Status set_frame_notify(std::function<void()> notify) override;

 private:
  struct Impl;
  explicit RealSenseStereo(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer so no SDK type reaches this header.
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
