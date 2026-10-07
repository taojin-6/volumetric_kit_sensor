// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the arithmetic between the Orbbec SDK's types and
// recon's -- the camera models, the extrinsic, the sync settings -- and the
// option checks, kept out of the driver's .cpp so host tests pin them with no
// camera attached.

#include <string>
#include <vector>

#include <libobsensor/h/ObTypes.h>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_stream.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

/// A stream's camera as it captures, for the GPU pass: its intrinsics and its
/// lens. The SDK sizes an image in `int16_t`, so a width or height that is not
/// positive is refused here rather than wrapped into a four-billion-pixel
/// `uint32_t`; a focal length that is not finite and positive is refused
/// because every unprojection divides by it. The SDK's Brown-Conrady
/// models are OpenCV's rational one with the missing terms zero -- k4..k6 for
/// the plain model, whatever it reports there, since only the K6 one has
/// them; its modified, inverse and Kannala-Brandt models are refused as
/// `Unsupported`, since the pass samples through that one model. @p what names
/// the stream, in the errors too.
core::Result<camera::CameraModel> camera_model_from(
    const OBCameraIntrinsic& intrinsic, const OBCameraDistortion& distortion,
    const std::string& what);

/// The SDK's extrinsic from one stream to another (`p_to = R p_from + t`,
/// `rot` row-major, `trans` in millimetres) as the transform that takes a point
/// in the first camera's frame to the second's, in metres, its rotation made
/// one (`camera::nearest_rotation`): the SDK's need not be. Refused when it
/// does not come out rigid, as from a zeroed or reflected matrix.
core::Result<camera::Mat4d> transform_from(const OBExtrinsic& extrinsic);

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

/// The streams' checks, refused naming @p who: sizes and rate, and a depth
/// range that is finite, non-empty and starts above 0, as the GPU pass
/// requires. The first check a camera's open makes, before it touches the SDK.
core::Status validate_streams(const OrbbecStreamOptions& streams,
                              const std::string& who);

}  // namespace volumetric_kit::recon::sensor::orbbec
