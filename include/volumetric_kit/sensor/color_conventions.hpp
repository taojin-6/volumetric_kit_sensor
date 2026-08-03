// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/color_conventions.hpp
/// @brief The colour half of the capture boundary: what a driver's
///        @ref ColorEncoding declaration *means*, and the one conversion that
///        brings a frame into the canonical form the fusion tiers require.
///
/// The companion to `sensor/camera_conventions.hpp`, and here for the same
/// reason: these are the conversions a capture integration gets *silently*
/// wrong, they are pure arithmetic, and host tests can pin them on any platform
/// -- so they live beside the contract rather than inside whichever
/// platform-bound driver produced the numbers, where a per-driver copy would be
/// a per-driver chance to be wrong.
///
/// The split against `core/color_space.hpp` is deliberate: **`core` owns the
/// curve and the type** -- the curve has a GLSL mirror and `tsdf`/`mesh` decode
/// with it in their kernels, and @ref is_canonical is a property of
/// @ref ColorEncoding that `tsdf` must test to refuse a frame it cannot fuse;
/// neither tier can include from `sensor`, which branches off `core` beside
/// them. **`sensor` owns the boundary conversion** -- @ref to_canonical, which
/// walks a frame, and the cost and limits of doing so.
///
/// A driver states what it produces; it never converts. That is what makes the
/// failure mode of a new integration an inspectable wrong *label* rather than a
/// bespoke curve buried in a platform driver.

#include <cstddef>
#include <cstdint>

#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/export.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief Convert a packed-RGB frame into the canonical encoded form.
///
/// The sensor-boundary conversion, and it is two steps rather than one: decode
/// the declared transfer function to linear, then apply the declared primaries'
/// 3x3 into the working basis (@ref primaries_to_working) before re-encoding.
/// Skipping the second step is the quiet failure this exists to prevent --
/// "linear" alone does not name a space.
///
/// When @ref is_canonical already holds, the colour is carried across verbatim
/// -- no round trip through the curve -- and only the alpha byte is forced.
/// That is the common case by construction: the canonical form was chosen to
/// match what the sensors in hand produce.
///
/// **Wide-gamut sources clip**, knowingly. BT.709 is the narrowest declarable
/// gamut, so a saturated Display P3 or BT.2020 colour lands outside [0, 1] and
/// is clamped on re-encode; an 8-bit canonical attribute could not have held it
/// regardless. See `core/color_space.hpp` for why widening the working space
/// and widening the storage are one move.
///
/// @param src   `count` packed words, R/G/B in the low three bytes. The high
///              byte is ignored, whatever the driver left there.
/// @param count Number of pixels.
/// @param enc   What the driver declared @p src to be.
/// @param dst   `count` packed words, always written with alpha `0xFF` -- on
///              the identity path too, so the guarantee holds for every
///              declaration rather than for the ones that walk the curve. That
///              is what lets a converted frame double as a projective-texturing
///              atlas, and what keeps the `tsdf` tier's "colour unobserved"
///              sentinel exact. May alias @p src exactly (in-place); may not
///              partially overlap.
/// @return OK, or:
///         - @ref Status::Code::InvalidArgument if @p src or @p dst is null
///           with a non-zero @p count;
///         - @ref Status::Code::Unsupported for
///           @ref ColorEncoding::Transfer::Bt2020Pq, whose absolute-luminance
///           HDR curve needs tone mapping into an 8-bit SDR form -- something
///           this repo does not do, and reports rather than approximating into
///           a quietly wrong result.
VR_SENSOR_API Status to_canonical(const std::uint32_t* src, std::size_t count,
                                  const ColorEncoding& enc, std::uint32_t* dst);

}  // namespace volumetric_kit::recon::sensor
