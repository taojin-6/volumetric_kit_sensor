// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the arithmetic between the Orbbec SDK's types and
// this package's -- the camera models, the extrinsic, the sync settings -- the
// option checks, and the colour controls' write and read-back, kept out of the
// driver's .cpp so host tests pin them with no camera attached.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <libobsensor/h/ObTypes.h>
#include <libobsensor/h/Property.h>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/color_controls.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/orbbec/orbbec_stream.hpp"
#include "volumetric_kit/sensor/orbbec/orbbec_sync_config.hpp"

namespace volumetric_kit::sensor::orbbec {

/// A stream's camera as it captures, for the GPU pass: its intrinsics and its
/// lens. The SDK sizes an image in `int16_t`, so a width or height that is not
/// positive is refused here rather than wrapped into a four-billion-pixel
/// `uint32_t`; a focal length that is not finite and positive is refused
/// because every unprojection divides by it. The SDK's Brown-Conrady
/// models are OpenCV's rational one with the missing terms zero -- k4..k6 for
/// the plain model, whatever it reports there, since only the K6 one has
/// them; its modified, inverse and Kannala-Brandt models are refused as
/// `Unsupported`, named (@ref lens_model_name), since the pass samples through
/// that one model. @p what names the stream and @p who the camera, in the
/// errors too.
core::Result<camera::CameraModel> camera_model_from(
    const OBCameraIntrinsic& intrinsic, const OBCameraDistortion& distortion,
    const std::string& what, const std::string& who);

/// The SDK's lens model by this package's name for it, as `OrbbecDeviceInfo`
/// records it and @ref camera_model_from's refusal names it:
/// `OB_DISTORTION_BROWN_CONRADY_K6` is "brown-conrady-k6", and so on; a value
/// this SDK does not name is "unknown-" and its number.
std::string lens_model_name(OBCameraDistortionModel model);

/// The SDK's extrinsic from one stream to another (`p_to = R p_from + t`,
/// `rot` row-major, `trans` in millimetres) as the transform that takes a point
/// in the first camera's frame to the second's, in metres, its rotation made
/// one (`camera::nearest_rotation`): the SDK's need not be. Refused, naming
/// @p who, when it does not come out rigid, as from a zeroed or reflected
/// matrix.
core::Result<camera::Mat4d> transform_from(const OBExtrinsic& extrinsic,
                                           const std::string& who);

/// The SDK's sync-mode bit to the driver's name for it, and back.
OrbbecSyncMode sync_mode_from(OBMultiDeviceSyncMode mode) noexcept;
OBMultiDeviceSyncMode sdk_sync_mode(OrbbecSyncMode mode) noexcept;

/// A camera's sync settings to the driver's struct, and back.
OrbbecSyncSettings sync_settings_from(const OBMultiDeviceSyncConfig& sdk);
OBMultiDeviceSyncConfig sdk_sync_config(const OrbbecSyncSettings& settings);

/// Where a camera's @p actual sync settings differ from the @p wanted ones, one
/// "depthDelayUs is 320, configured 160" entry each; empty when they agree.
/// Only what a Femto Mega reads back as written is compared: it stores one
/// secondary mode and reads it back as SecondarySynced, and reads
/// trigger2ImageDelayUs back as the depth delay and framesPerTrigger as 0
/// outside the triggering modes.
std::vector<std::string> sync_differences(const OrbbecSyncSettings& wanted,
                                          const OrbbecSyncSettings& actual);

/// The streams' checks, refused naming @p who: sizes and rate, a depth range
/// that is finite, non-empty and starts above 0, as the GPU pass requires, a
/// colour queue-family count within its array, and colour controls with an
/// exposure and a gain above 0. The first check a camera's open makes, before
/// it touches the SDK.
core::Status validate_streams(const OrbbecStreamOptions& streams,
                              const std::string& who);

/// The queue families that read a camera's colour pictures, as one value from
/// the options to each start's colour decoder, so no hop copies the list
/// without its count.
struct ColorQueueFamilies {
  std::uint32_t families[kMaxColorQueueFamilies] = {};
  std::uint32_t count = 0;
};

/// @p streams' `color_queue_families` and their count.
ColorQueueFamilies color_queue_families(const OrbbecStreamOptions& streams);

/// A camera's integer and boolean properties, as @ref write_color_controls
/// reads and writes them: the SDK's device, or a test's stand-in. Each call
/// may throw, as the SDK's do.
class CameraProperties {
 public:
  virtual ~CameraProperties() = default;
  /// Whether the camera has @p id to read and to write.
  virtual bool writable(OBPropertyID id) = 0;
  /// @p id's range, an integer property's.
  virtual OBIntPropertyRange int_range(OBPropertyID id) = 0;
  virtual std::int32_t get_int(OBPropertyID id) = 0;
  virtual bool get_bool(OBPropertyID id) = 0;
  virtual void set_int(OBPropertyID id, std::int32_t value) = 0;
  virtual void set_bool(OBPropertyID id, bool value) = 0;
};

/// Set a colour camera's exposure to @p wanted -- a manual exposure, or auto
/// exposure when empty -- with auto white balance on, writing only where the
/// camera differs, and read it back. Always applied, since a camera keeps
/// what was last written to it, in another process too.
///
/// 1. Manual: auto exposure, exposure, gain and auto white balance must all
///    be readable and writable, and exposure and gain within the camera's
///    ranges and on their steps, or nothing is written (`Unsupported`,
///    naming the control).
///    Automatic: a control the camera lacks is left alone, so a camera
///    without these controls still opens.
/// 2. Manual: auto exposure off, then exposure, then gain. Under auto
///    exposure a camera reads its stored values, not the ones it runs at, so
///    a camera switched from auto exposure has both written; one already at a
///    manual exposure has only a value that differs written. Automatic: auto
///    exposure on, exposure and gain untouched. Then auto white balance on.
///    Auto white balance off and the white-balance temperature are never
///    written.
/// 3. Read back: the auto exposure and auto white balance flags, and under a
///    manual exposure the exposure and gain. A camera that reads back
///    otherwise, as one whose SDK changed a value would, is refused
///    (`Unsupported`), naming each field.
///
/// @return The manual exposure as read back; empty under auto exposure;
///         `Unsupported` as above, naming @p who; or `IoError` with the SDK's
///         message for a call that throws.
core::Result<std::optional<camera::ColorControls>> write_color_controls(
    CameraProperties& camera,
    const std::optional<camera::ColorControls>& wanted, const std::string& who);

}  // namespace volumetric_kit::sensor::orbbec
