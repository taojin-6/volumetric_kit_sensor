// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stereo/stereo_sensor.hpp
/// @brief A calibrated stereo camera -- an infrared pair, a pair of cine
///        cameras -- as its driver presents it: two pictures taken together,
///        and where the two cameras are.
///
/// A stereo camera driver implements @ref IStereoSensor and hands out
/// @ref StereoFrame s as the cameras captured them; nothing on the host
/// undistorts, rectifies or converts them. A depth source built on it
/// (rectification on the GPU, then a stereo matcher) is an `IRgbdSensor`, so
/// a consumer of depth never sees this interface. Like the frame, this header
/// reaches no Vulkan.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/stereo/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @brief Two calibrated cameras: each with its own lens, and where the right
///        one is.
///
/// "Left" and "right" are the matcher's: the right camera sits along the left
/// one's +x, give or take the small rotations and offsets rectification
/// removes.
struct StereoCalibration {
  camera::CameraModel left;   ///< The left camera, lens included.
  camera::CameraModel right;  ///< The right camera, lens included.
  /// A left-camera point in the right camera (OpenCV's `R`, `T`), metres.
  camera::Mat4d left_to_right = camera::Mat4d(1.0);
};

/// @brief What a stereo source delivers, known when it opens: before a frame.
///
/// Besides the cameras, what the source is, as `SensorInfo` has it: a depth
/// source built on it reports these as its own (`StereoRectifier::depth_info`
/// copies them), so a sensor array can name, check and group it.
struct StereoInfo {
  StereoCalibration calibration;  ///< The pair.
  /// A third, colour camera (a RealSense's); empty when `left` is the colour.
  std::optional<camera::CameraModel> color;
  /// A left-camera point in the colour camera; identity when there is none.
  camera::Mat4d left_to_color = camera::Mat4d(1.0);
  std::string id;     ///< Its serial, as `SensorInfo::id`.
  std::string model;  ///< What it is, e.g. "RealSense D455".
  ClockDomain clock = ClockDomain::Device;  ///< Its timestamps' clock.
  std::uint32_t fps = 0;  ///< Its rate, frames a second; 0 if unknown.
  /// The colour camera's manual exposure as read back at open, as
  /// `SensorInfo::color_controls`; empty under auto exposure.
  std::optional<camera::ColorControls> color_controls;
};

/// @brief One picture from each camera, taken together, as the source
///        delivered them, and the colour to fuse.
///
/// It holds what it points to (@ref pixels for host planes, the pictures'
/// own buffers on the device), so a consumer may keep several.
struct StereoFrame {
  /// The left camera's picture: any layout the GPU reads (`YuvLayout::Luma`
  /// for a mono pair, YUYV, NV12, I420), its planes on the host or in a
  /// buffer on the device, at `StereoCalibration::left`'s size. Pictures
  /// that share a buffer (a side-by-side frame) name one `queue_family`, the
  /// one that wrote it.
  YuvImage left;
  YuvImage right;  ///< The right camera's, as @ref left.
  /// A third camera's picture (`StereoInfo::color`); no planes when `left` is
  /// the colour.
  YuvImage color{};
  /// What the colour's R'G'B' is encoded as, as `RgbdFrame::color_encoding`.
  ColorEncoding color_encoding{};
  /// The colour camera's frame to the world's, as
  /// `RgbdFrame::color_to_world`.
  camera::Mat4d color_to_world = camera::Mat4d(1.0);
  std::uint64_t timestamp_ns = 0;  ///< As `RgbdFrame::timestamp_ns`.
  std::uint64_t sequence = 0;      ///< As `RgbdFrame::sequence`.
  std::uint64_t arrival_ns = 0;    ///< As `RgbdFrame::arrival_ns`.
  /// What the pictures' host planes point into, held for as long as the frame
  /// is; null when they need no owner.
  std::shared_ptr<const void> pixels;
};

/// @brief A source of calibrated stereo pairs, polled as an `IRgbdSensor` is.
///
/// What a stereo camera driver implements: a RealSense's infrared pair, a pair
/// of Blackmagic cameras, a ZED. It holds the newest pair, which @ref poll
/// takes on the caller's thread.
///
/// @code
/// VKC_TRY(camera.start());
/// for (;;) {
///   VKC_ASSIGN(std::optional<StereoFrame> frame, camera.poll());
///   if (!frame) continue;  // nothing new yet
///   VKC_ASSIGN(RectifiedPair pair, rectifier->rectify(*frame));
/// }
/// @endcode
///
/// @warning Not thread-safe unless an implementation says otherwise: open,
///          start, poll and stop from one thread at a time, in order -- a
///          thread that polls is created after @ref start and joined before
///          @ref stop.
class VS_STEREO_API IStereoSensor {
 public:
  virtual ~IStereoSensor() = default;

  IStereoSensor(const IStereoSensor&) = delete;
  IStereoSensor& operator=(const IStereoSensor&) = delete;

  /// @return What the source is, from when it opened.
  virtual const StereoInfo& info() const noexcept = 0;

  /// @brief Begin capturing. Idempotent: starting a running source is OK.
  /// @return OK once running, or why it could not start.
  virtual core::Status start() = 0;

  /// @brief Stop capturing and drop the pair held. Idempotent, and safe on a
  ///        source that never started.
  virtual void stop() noexcept = 0;

  /// @brief Take the newest pair held.
  /// @return The pair; an empty optional when none is held -- the ordinary
  ///         case for a consumer polling faster than the cameras run; or why
  ///         the source failed.
  virtual core::Result<std::optional<StereoFrame>> poll() = 0;

  /// @return The counters since the last @ref start, as an `IRgbdSensor`'s.
  virtual SensorStats stats() const noexcept = 0;

  /// @brief Have the source call @p notify each time a pair becomes ready for
  ///        @ref poll, or the source fails, so a consumer -- a depth source's
  ///        own thread, say -- can block between polls instead of sleeping.
  ///        As `IRgbdSensor::set_frame_notify`.
  ///
  /// @p notify runs on whatever thread delivers the pair, after the pair is
  /// held, so the poll it wakes finds it. It must be quick, must not call
  /// back into the source, and may run after a later call replaced it, so it
  /// owns whatever it touches.
  /// @param notify  Called once for each pair held and on a failure; empty
  ///                stops the calls.
  /// @return OK; `Status::Code::InvalidArgument` once started;
  ///         `Status::Code::Unsupported` for a source that notifies nothing,
  ///         the default, whose consumer polls it.
  virtual core::Status set_frame_notify(std::function<void()> notify);

 protected:
  IStereoSensor() = default;
  // Move operations stay available to implementations but are not part of
  // the polymorphic interface: callers hold a source behind a pointer.
  IStereoSensor(IStereoSensor&&) = default;
  IStereoSensor& operator=(IStereoSensor&&) = default;
};

}  // namespace volumetric_kit::sensor
