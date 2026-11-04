// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file arkit/arkit_sensor.hpp
/// @brief ARKit as an @ref volumetric_kit::sensor::IRgbdSensor: a LiDAR
///        iPad's or iPhone's camera, tracked by ARKit, its depth registered
///        to its picture, as ARKit hands them over (Apple only; the camera on
///        iOS and iPadOS).
///
/// No ARKit or Objective-C type appears here -- ARKit is held behind a
/// pointer to an implementation -- so C++ includes this header as it is. On
/// iOS and iPadOS the target links ARKit; on macOS it builds with the test
/// pattern alone (@ref ArkitSource::Pattern), which the Mac's tests run.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/arkit/export.hpp"
#include "volumetric_kit/sensor/frame/depth_noise.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"  // kMaxColorQueueFamilies

namespace volumetric_kit::sensor {

/// @brief Where an @ref ArkitSensor's frames come from.
enum class ArkitSource : std::uint8_t {
  /// ARKit's world tracking: the back camera, its LiDAR depth and the
  /// device's pose. iOS and iPadOS only.
  Camera,
  /// The test pattern, with no camera opened: each frame's index coded in its
  /// depth (`testing::pattern_depth`) and its picture
  /// (`testing::test_pattern`),
  /// at the camera's sizes and rate, on every Apple platform.
  Pattern,
};

/// @brief Which of ARKit's depth maps the frames carry.
enum class ArkitDepthMap : std::uint8_t {
  Smoothed,  ///< `smoothedSceneDepth`: filtered over the frames before it.
  Raw,       ///< `sceneDepth`: each frame's own.
};

/// @brief ARKit's tracking of the device, as its last frame said.
enum class ArkitTracking : std::uint8_t {
  NotAvailable,          ///< No pose yet, as at a session's start.
  Initializing,          ///< Limited: still finding the world.
  Relocalizing,          ///< Limited: finding it again after an interruption.
  ExcessiveMotion,       ///< Limited: the device moves too fast.
  InsufficientFeatures,  ///< Limited: too little in view to track.
  Normal,                ///< Tracked: frames go out.
};

/// @return A stable lowercase name for @p tracking, for a read-out.
VS_ARKIT_API const char* to_string(ArkitTracking tracking) noexcept;

/// @brief What an @ref ArkitSensor's session is doing, for an app's read-out.
struct ArkitStatus {
  ArkitTracking tracking = ArkitTracking::NotAvailable;  ///< At the last frame.
  /// ARKit has stopped the camera for a while (the app went to the
  /// background, or the system took the camera): no frames until it ends.
  bool interrupted = false;
  /// Frames withheld since the last start because ARKit did not track them
  /// normally; part of `SensorStats::failed`.
  std::uint64_t withheld = 0;
  /// Frames refused since the last start: a depth or confidence map, or a
  /// picture, of a format or size the sensor does not take, colour tags
  /// `ColorEncoding` cannot name, or a pose further from rigid than float
  /// leaves one (`camera::exact_rigid`); part of `SensorStats::failed`.
  std::uint64_t refused = 0;
  /// Capture periods ARKit skipped since the last start, from each frame's
  /// time after the one before at the session's rate, counted as the frame
  /// after the gap arrives: ARKit's own drops, which no count of frames
  /// shows. The pattern's ticks skipped for want of a free picture count
  /// here too.
  std::uint64_t missed = 0;
  std::string last_refusal;   ///< Why the last frame was refused; or empty.
  std::string session_error;  ///< ARKit's failure, which ended the session.
};

/// @brief ARKit's smoothed LiDAR depth's noise at high confidence, which
///        every frame states: about 5 mm at 1 m, 8 mm at 2 m, 20 mm at 3 m and
///        38 mm at 4 m, fitted on one take against ARKit's own poses. Samples
///        kept at a lower confidence are noisier (medium about three times).
inline constexpr DepthNoise kArkitDepthNoise{8.58e-3f, -7.67e-3f, 3.79e-3f};

/// @brief One ARKit depth sample as the sensor writes it: metres to
///        millimetres, rounded to the nearest; 0 (no return) for a sample that
///        is not a number, not above 0, or past 65.535 m. A recording of
///        ARKit's metres converts through this to the depth the sensor would
///        have handed out.
constexpr std::uint16_t arkit_depth_mm(float metres) noexcept {
  return metres > 0.0f && metres < 65.535f
             ? static_cast<std::uint16_t>(metres * 1000.0f + 0.5f)
             : std::uint16_t{0};
}

/// @brief ARKit's camera, polled for frames as ARKit delivered them.
///
/// Each frame is one `ARFrame`. Its depth map (@ref Options::depth_map) is
/// copied once out of ARKit's buffer, as millimetres (@ref arkit_depth_mm),
/// with ARKit's confidence levels beside it (`RgbdFrame::depth_confidence`),
/// which `GpuFramePrep` gates on the GPU at @ref Options::min_confidence. Its
/// picture, `capturedImage`, stays where ARKit left it, 8-bit NV12 on its
/// IOSurface, and becomes the frame's two plane images
/// (`PixelBufferImporter`) with no copy. `pixel_buffer_of(frame.color)`
/// (apple/pixel_buffer_importer.hpp) gives the picture's own buffer back, to
/// code it as it is. The picture's colour is read from its own tags
/// (`color_codes_of`, then video/frame_color.hpp), an untagged matrix being
/// BT.601, ARKit's. HDR is off.
///
/// One camera: ARKit reports the picture's pinhole each frame
/// (`RgbdFrame::color_camera`, no lens), and the depth is registered to it.
/// The depth camera is therefore the same camera at the depth map's size
/// (`depth_from_registered_color`), and `RgbdFrame::depth_to_color` is the
/// identity. Each frame's pose is ARKit's for it (@ref PoseSource::Tracked),
/// in this package's camera axes (`cv_from_gl_camera`), made exactly rigid
/// (`camera::exact_rigid`): its bottom row exactly [0, 0, 0, 1] and its
/// rotation the nearest one, orthonormal to double's rounding, where ARKit's
/// float leaves the corner up to 3 ulp off 1 and the rotation orthonormal to
/// float's. The pose is in ARKit's world:
/// gravity-aligned, with its origin where the camera was when @ref open ran
/// the session. That world lasts as long as the sensor does: the session
/// never resets its tracking, and after an interruption ARKit finds its world
/// again before frames go out again. A frame that ARKit did not track
/// normally is withheld. Timestamps are ARKit's capture times moved onto the
/// host's `std::chrono::system_clock` (@ref ClockDomain::Host), by an offset
/// measured at each @ref start and again when an interruption ends, never
/// smaller than before, so they increase within a start. Each frame states
/// the camera's exposure (`RgbdFrame::color_exposure`) and
/// @ref kArkitDepthNoise.
///
/// A frame holds its depth and ARKit's picture, so it outlives the next poll.
/// ARKit has its picture back once every copy of the frame is gone. ARKit
/// keeps a small pool of pictures, so a consumer that holds many frames
/// starves the camera: keep a few.
///
/// The sensor owns its `ARSession`. It configures and runs the session and
/// asks for the camera. The session is wanted from @ref open until a
/// @ref stop, and from each @ref start: an interruption that ends then runs
/// it again, and a @ref start runs it where it is not running. Apple
/// documents that ARKit resumes by itself after an interruption; the run
/// covers a session left waiting, and keeps its world. An interruption keeps
/// the frame contract: no new frame arrives while it lasts, and the frames
/// after it go on from the next sequence, at later times. An app shows the
/// session's state through @ref status.
///
/// **The test pattern** (@ref ArkitSource::Pattern) runs the same queue,
/// conversion, import and counters with no camera: frame n's depth is
/// `testing::pattern_depth(256, 192, n)` handed over as ARKit's float metres
/// (1 m and 2 m, exact), with its confidence plane, and its picture
/// `testing::test_pattern` at the colour size, written into one of 4 IOSurface
/// '420f' buffers tagged as ARKit tags its own (BT.601 matrix, BT.709
/// transfer and primaries). n is the frame's `RgbdFrame::sequence`. The pose
/// is the identity, the pinhole centred with a focal length of 0.75 times the
/// width, and a timer on the sensor's queue ticks at @ref Options::fps; a
/// tick whose 4 buffers are all held is skipped, which counts as
/// `ArkitStatus::missed`. The model is "ARKit test pattern, " and the
/// device's model.
///
/// @code
/// ArkitSensor::Options options;  // 1920x1440 at 60 Hz, smoothed depth
/// options.device = &device;      // where the pictures become images
/// VKC_ASSIGN(ArkitSensor camera, ArkitSensor::open(options));
/// VKC_ASSIGN(auto prep, GpuFramePrep::create(device, allocator));
/// VKC_TRY(camera.start());
/// for (;;) {
///   VKC_ASSIGN(std::optional<RgbdFrame> frame, camera.poll());
///   if (!frame) continue;
///   VKC_ASSIGN(const DeviceFrame prepared, prep.prepare(*frame));
/// }
/// @endcode
///
/// @warning Not thread-safe, except @ref status and @ref stats: open, start,
///          poll, drain and stop from one thread. ARKit delivers on a queue of
///          the sensor's own, which this class hands over internally.
class VS_ARKIT_API ArkitSensor final : public IRgbdSensor {
 public:
  /// @brief Which source, which picture, which depth, and the device.
  struct Options {
    ArkitSource source = ArkitSource::Camera;  ///< Where frames come from.
    /// How a calibration and a session's track name the sensor
    /// (`SensorInfo::id`): set one per device where several are used.
    std::string id = "arkit";
    /// The picture's size: ARKit's video format of the back wide camera at
    /// this size and @ref fps, the binned one where two have them (the one
    /// ARKit does not recommend for high-resolution captures). The depth
    /// comes at ARKit's own size, 256 x 192.
    std::uint32_t color_width = 1920;
    std::uint32_t color_height = 1440;  ///< See @ref color_width.
    std::uint32_t fps = 60;             ///< See @ref color_width.
    /// The depth map the frames carry; the other is not computed. A session's
    /// first frames, which carry no smoothed map yet, are skipped, uncounted.
    ArkitDepthMap depth_map = ArkitDepthMap::Smoothed;
    /// Samples below this confidence are no return
    /// (`RgbdFrame::min_confidence`): ARKit's 0 low, 1 medium, 2 high. 0
    /// keeps every sample, and the frames then carry no confidence plane.
    std::uint8_t min_confidence = 2;
    /// Nearer depth is dropped (metres), as `RgbdFrame::min_depth`; above 0.
    float min_depth = 0.15f;
    /// Farther depth is dropped (metres), as `RgbdFrame::max_depth`.
    float max_depth = 5.0f;
    /// The device the pictures become images on, which the GPU pass prepares
    /// them on: required. Borrowed: it must outlive the sensor and every
    /// frame.
    const core::Device* device = nullptr;
    /// The queue families that read the pictures, as
    /// `PixelBufferImporter::create` takes them: @ref device's own and, say,
    /// a renderer's that draws them, as `OrbbecStreamOptions` has them.
    std::uint32_t color_queue_families[kMaxColorQueueFamilies] = {};
    /// Entries in @ref color_queue_families: 0 for @ref device's own alone.
    std::uint32_t color_queue_family_count = 0;
    /// How long @ref open waits for ARKit's first frame with depth, once the
    /// camera is allowed.
    std::uint32_t open_timeout_ms = 5000;
  };

  /// @brief Check the options and the device, run the session, and wait for
  ///        its first frame with depth, whose cameras the sensor's are
  ///        (@ref info). The session keeps running: frames go out from
  ///        @ref start, and frames before it are let go, uncounted.
  ///
  /// The first time an app opens the camera, iOS asks the user, and this
  /// waits for the answer, which @ref Options::open_timeout_ms does not
  /// count: call it off the main thread. An interruption that ends while it
  /// waits runs the session again, so a short one does not fail it. With
  /// @ref ArkitSource::Pattern no camera opens and this returns at once.
  ///
  /// A failed open leaves nothing behind: its session is paused and let go,
  /// and it holds none of ARKit's pictures, so another open may follow at
  /// once. Open again after `Status::Code::NotFound`, say once the app is in
  /// front again; `Status::Code::InvalidArgument` and
  /// `Status::Code::Unsupported` come back for the same options on the same
  /// device.
  /// @param options  The source, its streams and the device.
  /// @return The sensor, not yet started; or:
  ///         - `Status::Code::InvalidArgument` for options refused before
  ///           ARKit is asked: no device, a zero size or rate, a depth range
  ///           that is not finite, empty or starts at 0, a `min_confidence`
  ///           above 2, a `color_queue_family_count` past
  ///           @ref kMaxColorQueueFamilies; and
  ///           `PixelBufferImporter::create`'s refusals of the queue
  ///           families;
  ///         - `Status::Code::Unsupported` for @ref ArkitSource::Camera off
  ///           iOS and iPadOS; on a device without LiDAR, which gives no
  ///           scene depth; for a size and rate no video format of the back
  ///           wide camera has (the formats it offers are listed); or where
  ///           the device imports no Metal textures;
  ///         - `Status::Code::NotFound` if no frame with depth arrives within
  ///           @ref Options::open_timeout_ms; where ARKit has the session
  ///           interrupted then (the device locked, the app in the
  ///           background, or the camera in use elsewhere), the message
  ///           says so;
  ///         - `Status::Code::IoError` if the camera is not allowed (the
  ///           user's answer, which Settings changes), ARKit's session fails
  ///           (with ARKit's message), or the first frame is not of the
  ///           format's size.
  static core::Result<ArkitSensor> open(const Options& options);

  ArkitSensor(ArkitSensor&& other) noexcept;
  ArkitSensor& operator=(ArkitSensor&& other) noexcept;
  /// Pauses the session, which lets the camera go.
  ~ArkitSensor() override;

  /// @return What the session is doing now; the default on a moved-from
  ///         sensor.
  ArkitStatus status() const;

  /// @return What the sensor is: its id; "ARKit, " and the device's model;
  ///         the first frame's colour pinhole and the depth camera
  ///         registered to it; the identity extrinsic; free running on the
  ///         host's clock; tracked; its rate. Empty on a moved-from sensor.
  const SensorInfo& info() const noexcept override;
  /// @return As `IRgbdSensor::set_queue_depth`; also
  ///         `Status::Code::InvalidArgument` on a moved-from sensor.
  core::Status set_queue_depth(std::size_t frames) override;
  /// @brief Hand frames out from now: the counters start at 0 and the
  ///        sequences from 0. Where the session is not running -- paused by a
  ///        @ref stop, or interrupted -- it runs it again, its world kept:
  ///        frames are withheld until ARKit has found it.
  /// @return OK, and if already started; `Status::Code::InvalidArgument` on a
  ///         moved-from sensor; `Status::Code::IoError` once ARKit's session
  ///         has failed.
  core::Status start() override;
  /// @brief As `IRgbdSensor::stop`, pausing the session, which lets the
  ///        camera go; a frame ARKit had already queued to the sensor before
  ///        the pause is let go too. The sensor stays open, for another
  ///        @ref start.
  void stop() noexcept override;
  /// @return As `IRgbdSensor::poll`: the newest frame's picture becomes images
  ///         here, on the caller's thread. `Status::Code::IoError` once
  ///         ARKit's session has failed (`ArkitStatus::session_error`); the
  ///         importer's failure (`Status::Code::OutOfMemory`,
  ///         `Status::Code::Backend`), that frame counted failed.
  core::Result<std::optional<RgbdFrame>> poll() override;
  /// @return As @ref poll; on a failure, the frames before it stay appended,
  ///         and those after it are counted dropped.
  core::Status drain(std::vector<RgbdFrame>* out) override;
  /// @return `true` on a moved-from sensor and once ARKit's session has
  ///         failed; never for the pattern.
  bool exhausted() const noexcept override;
  /// @return The counters since the last @ref start, kept through a
  ///         @ref stop: `SensorStats::received` counts the frames carrying
  ///         the requested map, and `SensorStats::failed` is
  ///         `ArkitStatus::withheld` + `ArkitStatus::refused` + the pictures
  ///         that failed to import. Zero on a moved-from sensor.
  SensorStats stats() const noexcept override;
  /// @brief As `IRgbdSensor::set_frame_notify`: called on the sensor's queue
  ///        as each frame is held, and when the session fails. Each frame's
  ///        `RgbdFrame::arrival_ns` is the host's system clock as ARKit hands
  ///        the frame over.
  /// @return As `IRgbdSensor::set_frame_notify`, never
  ///         `Status::Code::Unsupported`; also
  ///         `Status::Code::InvalidArgument` on a moved-from sensor.
  core::Status set_frame_notify(std::function<void()> notify) override;

 private:
  struct Impl;
  explicit ArkitSensor(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer so no ARKit type reaches this header, and so the address
  // ARKit's delegate holds survives a move.
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
