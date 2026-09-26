// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the arithmetic between the Orbbec SDK's types and
// the contract's -- units, packing, and the camera struct -- kept out of the
// capture's .cpp so host tests pin it with no camera attached. Each function is
// a place a unit or a byte order can be silently wrong, which is why they are
// small, separate and tested rather than inlined into the frame path.

#include <cstddef>
#include <cstdint>

#include <libobsensor/h/ObTypes.h>

#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_capture.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

/// The colour camera from the SDK's intrinsics of the (undistorted) colour
/// stream and the caller's pose. The SDK sizes an image in `int16_t`, so a
/// width or height that is not positive is refused here rather than wrapped
/// into a four-billion-pixel `uint32_t`; a focal length that is not finite and
/// positive is refused because every unprojection divides by it.
Result<ColorCameraParams> color_camera_from(const OBCameraIntrinsic& intrinsic,
                                            const Mat4f& cam_to_world);

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

/// The SDK's sync-mode bit to the driver's name for it.
OrbbecSyncMode sync_mode_from(OBMultiDeviceSyncMode mode) noexcept;

/// Everything about @p options that can be refused without a camera: sizes and
/// rate, the depth range, the pose. The first check @ref OrbbecCapture::open
/// makes, before it touches the SDK.
Status validate(const OrbbecCapture::Options& options);

}  // namespace volumetric_kit::recon::sensor::orbbec
