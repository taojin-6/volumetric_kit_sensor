// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the arithmetic between the Orbbec SDK's types and
// the contract's -- units, packing, the camera struct -- kept out of the
// capture's .cpp so host tests pin it with no camera attached.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <libobsensor/h/ObTypes.h>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/camera/geometry.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_rig.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sync_config.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

/// The colour camera from the SDK's intrinsics of the (undistorted) colour
/// stream and the caller's pose. The SDK sizes an image in `int16_t`, so a
/// width or height that is not positive is refused here rather than wrapped
/// into a four-billion-pixel `uint32_t`; a focal length that is not finite and
/// positive is refused because every unprojection divides by it.
core::Result<ColorCameraParams> color_camera_from(
    const OBCameraIntrinsic& intrinsic, const Mat4f& cam_to_world);

/// A stream's camera as it captures, for the GPU pass: its intrinsics (checked
/// as @ref color_camera_from checks them) and its lens. The SDK's Brown-Conrady
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
/// one (`camera::nearest_rotation`): the SDK's need not be.
camera::Mat4d transform_from(const OBExtrinsic& extrinsic) noexcept;

/// Whether @p intrinsic is the pinhole camera @p cam: fx, fy, cx and cy each
/// within @p tol pixels. Tested for agreement rather than for a difference, so
/// a NaN on either side is a mismatch instead of passing every comparison.
bool same_pinhole(const OBCameraIntrinsic& intrinsic,
                  const ColorCameraParams& cam, float tol) noexcept;

/// Raw depth units to metres: `metres = raw * value_scale_mm / 1000`, the SDK
/// defining a depth frame's value scale as millimetres per unit. A raw 0 is the
/// camera's "no return" and stays exactly 0, which the fusion kernels skip.
/// @p dst may not overlap @p src.
void depth_to_metres(const std::uint16_t* src, std::size_t count,
                     float value_scale_mm, float* dst);

/// Packed 8-bit RGB triplets (the SDK's `OB_FORMAT_RGB`, R first) to the
/// contract's `R | G<<8 | B<<16` words, high byte 0 -- the layout the `tsdf`
/// and `mesh` tiers read and the one the Replica reader produces.
void pack_rgb(const std::uint8_t* rgb, std::size_t count, std::uint32_t* dst);

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

/// Everything about @p options that can be refused without a camera: sizes and
/// rate, the depth range, the pose. The first check @ref OrbbecCapture::open
/// makes, before it touches the SDK.
core::Status validate(const OrbbecCapture::Options& options);

/// The same for a rig, plus what only a rig has: at least two cameras, a
/// calibration (if any) that @ref camera::validate_array_calibration accepts
/// and that poses every one of them, and a sync tolerance under half a frame
/// period.
core::Status validate(const OrbbecRig::Options& options);

}  // namespace volumetric_kit::recon::sensor::orbbec
