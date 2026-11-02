// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file decklink/decklink_stereo.hpp
/// @brief Two SDI inputs of a Blackmagic DeckLink card as one stereo camera
///        (@ref volumetric_kit::sensor::IStereoSensor): each camera's 10-bit
///        picture captured straight into host pages the GPU imports, and
///        copied once into a device buffer.
///
/// No DeckLink SDK type appears here, so a consumer includes this header
/// without the SDK. The driver loads Desktop Video's library at run time.

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/decklink/export.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief One camera of a DeckLink-captured stereo camera: the card
///        connector its SDI OUT feeds.
///
/// Connectors are named as on the card's bracket; the driver maps them to
/// sub-devices (DeckLink 8K Pro, profile "four sub-devices half-duplex":
/// SDI 1 = 0, SDI 3 = 1, SDI 2 = 2, SDI 4 = 3).
struct DecklinkPort {
  /// The camera's name: its key in the calibration file. The stereo
  /// camera's id is the left one's.
  std::string id;
  std::string input;  ///< "SDI 1" to "SDI 4".
};

/// @brief Two SDI inputs of one DeckLink card as one stereo camera.
///
/// Each frame is the left and right cameras' pictures as the card captured
/// them: 10-bit V210 (`YuvLayout::V210`) in device buffers
/// (`YuvImage::device`, written on the device's own queue family, the copy
/// finished), BT.709 limited range, chroma co-sited (`ChromaLocation::Left`),
/// encoded BT.709. Each picture's buffer is its own owner, back to the
/// driver once no frame holds it, so @ref StereoFrame::pixels and
/// @ref StereoFrame::right_pixels are null; `StereoRectifier::rectify` lets
/// the right one go, and a depth frame holds the left one alone.
///
/// The card writes each picture into host pages that the device imports
/// (`core::create_imported_host_buffer`), and one GPU copy a picture moves
/// it into a device buffer. The CPU copies nothing; the SDK's frames go
/// back as soon as their copy is done.
///
/// Pairs are matched on the card's frame times: each frame with the other
/// camera's newest, within @ref Options::max_skew_ns, stamped on the host's
/// clock (`ClockDomain::Host`) with the left picture's time, and carrying
/// the measured skew (@ref StereoFrame::skew_ns, right less left). The
/// cameras free-run unless fed a reference from elsewhere, so the skew is
/// whatever their phase is, and drifts.
///
/// It holds the newest pair, which @ref poll takes; a pair passed over is
/// counted dropped.
///
/// @code
/// DecklinkStereo::Options options;
/// options.left = {"bm-1", "SDI 2"};
/// options.right = {"bm-2", "SDI 1"};
/// VKC_ASSIGN(options.calibration, stereo_calibration(file, "bm-1", "bm-2"));
/// VKC_ASSIGN(DecklinkStereo camera,
///            DecklinkStereo::open(options, device, allocator));
/// VKC_TRY(camera.start());
/// VKC_ASSIGN(std::optional<StereoFrame> frame, camera.poll());
/// @endcode
///
/// @warning Not thread-safe: open, start, poll and stop from one thread at a
///          time, in order. The SDK delivers each input's frames on a thread
///          of its own and the driver copies them on another.
class VS_DECKLINK_API DecklinkStereo final : public IStereoSensor {
 public:
  /// @brief Which connectors, which mode, and how far apart a pair may be.
  struct Options {
    DecklinkPort left;   ///< The left camera.
    DecklinkPort right;  ///< The right camera.
    /// The pair's lenses at @ref width x @ref height and the left camera's
    /// point in the right one (`stereo_calibration`). Empty for cameras not
    /// calibrated yet, whose pictures a calibration is made from: @ref info
    /// then has none, which `StereoRectifier` refuses.
    std::optional<StereoCalibration> calibration;
    std::uint32_t width = 3840;   ///< The cameras' mode: 2160p ...
    std::uint32_t height = 2160;  ///< ...
    std::uint32_t fps = 30;       ///< ... at 24, 25 or 30.
    /// Pairs farther apart are refused. The default, just under half a
    /// frame at 30 fps, always pairs free-running cameras (still scenes give
    /// good depth). Genlocked cameras set 500 us, which keeps a 1 m/s
    /// motion's disparity error under 0.08 px at 2.5 m.
    std::uint64_t max_skew_ns = 16'666'000;
    /// How long @ref start waits for the first pair within
    /// @ref max_skew_ns.
    std::chrono::milliseconds lock_timeout{5000};
  };

  /// @brief Find the card and check both inputs' mode. Does not capture.
  /// @param options    The connectors, the mode and the bound.
  /// @param device     The device the pictures go to: the rectifier's and
  ///                   prep's. It must outlive the driver and its frames.
  /// @param allocator  Where their buffers come from; it must outlive the
  ///                   driver and its frames.
  /// @return The stereo camera, not yet started; or:
  ///         - `Status::Code::InvalidArgument` for options refused before
  ///           any device: a connector not on the card or one twice, a mode
  ///           other than 2160p 24, 25 or 30, a lens not at
  ///           @ref Options::width x @ref Options::height,
  ///           @ref Options::max_skew_ns 0 or past half a frame, or a
  ///           @ref Options::lock_timeout not positive;
  ///         - `Status::Code::NotFound` for no Desktop Video, or no DeckLink
  ///           8K Pro (those found named);
  ///         - `Status::Code::Unsupported` for a card whose profile does not
  ///           give each connector its own sub-device (naming the fix:
  ///           "four sub-devices half-duplex"), an input that cannot
  ///           capture the mode as V210, or a device that did not enable
  ///           `VK_EXT_external_memory_host` (`device_requirements` asks for
  ///           it);
  ///         - `Status::Code::IoError` for any other SDK failure.
  static core::Result<DecklinkStereo> open(const Options& options,
                                           const core::Device& device,
                                           core::Allocator& allocator);

  DecklinkStereo(DecklinkStereo&& other) noexcept;
  DecklinkStereo& operator=(DecklinkStereo&& other) noexcept;
  /// Stops the inputs if they are capturing.
  ~DecklinkStereo() override;

  /// @return The calibration given (none without one), the left camera's
  ///         name as the id, the host's clock, @ref Options::fps,
  ///         @ref Options::max_skew_ns, and no colour controls. Empty on a
  ///         moved-from camera.
  const StereoInfo& info() const noexcept override;
  /// @brief Start both inputs, at the opened mode with format detection on,
  ///        and wait for both to lock and for one pair within the bound.
  /// @return OK once a pair is on the device, and if already started;
  ///         `Status::Code::Unsupported` for a camera at another mode,
  ///         naming it and the fix (the camera's menu, or USB PTP property
  ///         0xd006 = fps x 100); `Status::Code::IoError` after
  ///         @ref Options::lock_timeout, naming the input that sent nothing
  ///         or the cameras' measured offset and the fix, or for an SDK or
  ///         device failure; `Status::Code::InvalidArgument` on a moved-from
  ///         camera.
  core::Status start() override;
  /// @brief As `IStereoSensor::stop`; the camera stays open for another
  ///        @ref start.
  void stop() noexcept override;
  /// @return As `IStereoSensor::poll`; `Status::Code::Unsupported` once a
  ///         camera changed mode, naming the mode it sends (open again after
  ///         setting it back); `Status::Code::IoError` once a copy failed.
  core::Result<std::optional<StereoFrame>> poll() override;
  /// @return The counters since the last @ref start, kept through a
  ///         @ref stop, in left frames: received, every left frame; failed,
  ///         a left frame not delivered in a pair, for any cause (no partner,
  ///         no signal, past the bound, every device buffer held); dropped,
  ///         a pair replaced before a poll, on the host or on the device.
  ///         Zero on a moved-from camera.
  SensorStats stats() const noexcept override;
  /// @brief As `IStereoSensor::set_frame_notify`: called on the driver's
  ///        upload thread once a pair is on the device, and on the SDK's
  ///        thread when a camera changes mode.
  /// @return OK, never `Status::Code::Unsupported`;
  ///         `Status::Code::InvalidArgument` once started or on a moved-from
  ///         camera.
  core::Status set_frame_notify(std::function<void()> notify) override;

 private:
  struct Impl;
  explicit DecklinkStereo(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer so no SDK type reaches this header.
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
