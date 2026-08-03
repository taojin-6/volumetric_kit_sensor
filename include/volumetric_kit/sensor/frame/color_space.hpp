// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file core/color_space.hpp
/// @brief The color-space vocabulary: what encoding a frame declares, and the
///        one transfer curve every tier converts through.
///
/// The rule this header exists to enforce (DESIGN.md, "Color space"):
///
/// > **8-bit color is encoded. Float color is linear. Convert once at the
/// > sensor boundary, encode once at presentation.**
///
/// Averaging is a linear operation and a display encoding deliberately is not,
/// so every average taken on encoded values -- the TSDF running mean, the
/// marching-cubes edge interpolation, texture filtering, a shading multiply --
/// is wrong by construction, and wrong *quietly*: linear `0.0` and `1.0` fuse
/// to `0.214` rather than `0.5`, a plausibly darker surface rather than a
/// visible error.
///
/// **"Linear" alone does not name a space**, so one is named here: the working
/// space is linear **BT.709 primaries, D65 white** (the linear half of sRGB),
/// and its *canonical encoded form* is that space through the exact piecewise
/// sRGB transfer function, full range. Every 8-bit color below the sensor
/// boundary is in that one form -- which is what lets the voxel attribute, the
/// `_SRGB` atlas and the `_SRGB` render target all mean the same thing without
/// carrying a label. Two sensors with different primaries produce linear values
/// in *different RGB bases*, so @ref primaries_to_working exists to make
/// @ref ColorEncoding::Primaries a value something converts rather than a label
/// something reads.
///
/// This lives in `core` rather than `sensor` for the reason the camera
/// parameter blocks do: `sensor` branches off `core` *beside* the fusion tiers,
/// so `tsdf` and `mesh` -- which decode in GLSL -- cannot include from it, and
/// a curve four tiers need is vocabulary. The GLSL mirror is
/// `core/shaders/color_common.glsl`; `tests/core_color_space_gpu_test.cpp` pins
/// the two against each other over all 256 codes, because two implementations
/// of one curve is exactly the drift this header exists to prevent.

#include <cmath>
#include <cstdint>

#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace volumetric_kit::recon {

/// @brief What a capture source declares about the color it produces.
///
/// A driver *states* this; it never converts. The conversion has one
/// implementation (this header, plus its GLSL mirror), so the failure mode of a
/// new sensor integration is a wrong **label** -- inspectable, and testable
/// against a known patch -- rather than a bespoke curve buried in a platform
/// driver.
///
/// It rides **beside** the camera (a field on `sensor::CapturedFrame` and
/// `tsdf::ColorFrame`), deliberately *not* inside @ref ColorCameraParams: that
/// struct is uploaded verbatim to the fusion kernels under scalar block layout,
/// pinned at 88 bytes with GLSL mirrors in two tiers, so a field there would
/// spend a cross-tier shader ABI change to carry something no kernel reads.
///
/// There is no `Range` member, and the omission is deliberate: by the time a
/// frame reaches the capture contract it is packed R'G'B', so the YCbCr matrix
/// and any limited-range expansion have already been applied by the driver --
/// the contract requires full-range R'G'B'. A field nothing consumes is a label
/// free to drift, and @ref Primaries earns its place only because
/// @ref primaries_to_working gives it a consumer.
struct ColorEncoding {
  /// @brief The transfer function the 8-bit values carry.
  enum class Transfer {
    /// The exact piecewise sRGB curve -- the canonical form.
    Srgb,
    /// BT.709's camera OETF. **Accepted as @ref Transfer::Srgb rather than
    /// converted**: the two differ by a couple of codes in the toe, which is
    /// what display pipelines assume anyway (BT.1886), and converting would
    /// cost a per-frame pass over the whole image to buy nothing. A bounded,
    /// stated error beats an unstated one.
    Bt709,
    /// Linear light already -- decoded by nothing, but *encoded* on the way
    /// into the canonical 8-bit form (a linear 16-bit color camera).
    Linear,
    /// BT.2020 with the PQ (SMPTE ST 2084) absolute-luminance HDR curve.
    /// Declarable so a driver can be truthful; mapping it into an 8-bit SDR
    /// canonical form needs tone mapping, which this repo does not do, so
    /// `sensor::to_canonical` reports @ref Status::Code::Unsupported rather
    /// than fusing something quietly wrong.
    Bt2020Pq,
  };

  /// @brief The RGB primaries the values are expressed in (all D65, so no
  ///        chromatic adaptation is involved).
  enum class Primaries {
    /// BT.709 / sRGB -- the working space; conversion is the identity.
    Bt709,
    /// Display P3 (the DCI-P3 primaries at D65) -- Apple's wide gamut.
    DisplayP3,
    /// BT.2020 / BT.2100.
    Bt2020,
  };

  Transfer transfer = Transfer::Srgb;      ///< Defaults to canonical.
  Primaries primaries = Primaries::Bt709;  ///< Defaults to the working space.
};

/// @brief Equality, so a frame's declaration can be compared against the
///        canonical one.
constexpr bool operator==(const ColorEncoding& a,
                          const ColorEncoding& b) noexcept {
  return a.transfer == b.transfer && a.primaries == b.primaries;
}
constexpr bool operator!=(const ColorEncoding& a,
                          const ColorEncoding& b) noexcept {
  return !(a == b);
}

/// @brief Is this declaration already the canonical encoded form, so a frame
///        carrying it can be fused with no conversion at all?
///
/// True for @ref ColorEncoding::Transfer::Srgb and, deliberately, for
/// @ref ColorEncoding::Transfer::Bt709 -- see that enumerator for why accepting
/// it is a bounded, stated error rather than an oversight, and why it is what
/// makes the ARKit path (`{Bt709, Bt709}`) a genuine no-op.
///
/// Both halves must hold: a Display P3 frame with an sRGB transfer is *not*
/// canonical, because its linear values live in a different RGB basis and
/// averaging those is wrong the same way averaging encoded values is.
///
/// This predicate is `core` rather than `sensor` even though the *conversion*
/// (`sensor::to_canonical`) is not: `tsdf::TsdfIntegrator::integrate` refuses a
/// non-canonical frame rather than fusing it through the wrong curve, and
/// `tsdf` cannot include from `sensor` -- that tier branches off `core` beside
/// it. A property of this type belongs with the type; the boundary conversion,
/// and what it costs, stays at the boundary.
constexpr bool is_canonical(const ColorEncoding& e) noexcept {
  const bool transfer_ok = e.transfer == ColorEncoding::Transfer::Srgb ||
                           e.transfer == ColorEncoding::Transfer::Bt709;
  return transfer_ok && e.primaries == ColorEncoding::Primaries::Bt709;
}

/// @brief Decode one exact-piecewise-sRGB-encoded channel to linear.
///
/// The **exact** piecewise function, not a `pow(x, 2.2)` approximation, and the
/// distinction is load-bearing rather than pedantic: hardware `_SRGB` sampling
/// decodes the atlas with the exact curve, and the renderer's hybrid pipeline
/// selects between the atlas and the per-vertex color *per triangle across one
/// surface*, so an approximated voxel decode would show up as a seam exactly
/// where texturing stops.
///
/// @param e Encoded value, nominally in [0, 1]; values outside are carried
///          through the same branches rather than clamped, so the function
///          stays monotone for a caller that clamps later.
/// @return The linear-light value.
inline float srgb_to_linear(float e) noexcept {
  return e <= 0.04045f ? e * (1.0f / 12.92f)
                       : std::pow((e + 0.055f) * (1.0f / 1.055f), 2.4f);
}

/// @brief Encode one linear channel to exact piecewise sRGB.
/// @param l Linear-light value, nominally in [0, 1].
/// @return The encoded value. Inverse of @ref srgb_to_linear.
inline float linear_to_srgb(float l) noexcept {
  return l <= 0.0031308f ? l * 12.92f
                         : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
}

/// @brief Per-channel @ref srgb_to_linear.
inline Vec3f srgb_to_linear(const Vec3f& e) noexcept {
  return Vec3f{srgb_to_linear(e.x), srgb_to_linear(e.y), srgb_to_linear(e.z)};
}

/// @brief Per-channel @ref linear_to_srgb.
inline Vec3f linear_to_srgb(const Vec3f& l) noexcept {
  return Vec3f{linear_to_srgb(l.x), linear_to_srgb(l.y), linear_to_srgb(l.z)};
}

/// @brief Decode a packed canonical-encoded RGB word to linear.
///
/// The packed layout is the one the `tsdf` and `mesh` tiers share: R, G, B in
/// the low three bytes, alpha (if any) in the high byte and ignored here.
/// Mirrors GLSL's `unpackUnorm4x8(...).rgb` fed through @ref srgb_to_linear,
/// which is what `core/shaders/color_common.glsl` does.
inline Vec3f unpack_srgb_to_linear(std::uint32_t packed) noexcept {
  const float inv = 1.0f / 255.0f;
  return srgb_to_linear(
      Vec3f{static_cast<float>(packed & 0xFFu) * inv,
            static_cast<float>((packed >> 8) & 0xFFu) * inv,
            static_cast<float>((packed >> 16) & 0xFFu) * inv});
}

/// @brief Encode a linear RGB triple into a packed canonical-encoded word.
///
/// Clamps to [0, 1] and rounds to nearest, matching GLSL's `packUnorm4x8`.
/// Always sets alpha to `0xFF`, which is what keeps the `tsdf` tier's
/// "color unobserved" sentinel exact: a written color is never `0`, not even
/// pure black.
inline std::uint32_t pack_linear_to_srgb(const Vec3f& linear) noexcept {
  auto to_u8 = [](float v) -> std::uint32_t {
    const float e = linear_to_srgb(v);
    const float c = e < 0.0f ? 0.0f : (e > 1.0f ? 1.0f : e);
    return static_cast<std::uint32_t>(c * 255.0f + 0.5f);
  };
  return to_u8(linear.x) | (to_u8(linear.y) << 8) | (to_u8(linear.z) << 16) |
         0xFF000000u;
}

/// @brief The 3x3 that carries linear values from `p`'s primaries into the
///        working space (linear BT.709, D65).
///
/// All three primary sets are defined at D65, so this is a pure primaries
/// change with no chromatic adaptation. The BT.709 case is the identity, which
/// is why every surface and sensor this repo has run on costs nothing here --
/// and why the conversion is written down before a wide-gamut source makes it
/// something other than identity.
///
/// **Wide-gamut sources clip**, knowingly: BT.709 is the narrowest of the
/// three, so a saturated Display P3 or BT.2020 color lands outside [0, 1] and
/// an 8-bit canonical attribute could not hold it regardless. Widening the
/// working space means widening the storage with it; the two move together.
///
/// Matrices are the standard published conversions, given here column-major as
/// GLM stores them (so `primaries_to_working(p) * v` is the row-major product).
/// Each row sums to one, which is white mapping to white -- the invariant
/// `tests/core_color_space_test.cpp` checks, since it catches a transposed or
/// mistyped matrix that a spot value might not.
inline Mat3f primaries_to_working(ColorEncoding::Primaries p) noexcept {
  switch (p) {
    case ColorEncoding::Primaries::DisplayP3:
      return Mat3f{Vec3f{1.224940f, -0.042056f, -0.019637f},
                   Vec3f{-0.224940f, 1.042056f, -0.078636f},
                   Vec3f{0.000000f, 0.000000f, 1.098273f}};
    case ColorEncoding::Primaries::Bt2020:
      return Mat3f{Vec3f{1.660491f, -0.124551f, -0.018151f},
                   Vec3f{-0.587641f, 1.132900f, -0.100579f},
                   Vec3f{-0.072850f, -0.008349f, 1.118730f}};
    case ColorEncoding::Primaries::Bt709:
      break;
  }
  return Mat3f{1.0f};
}

}  // namespace volumetric_kit::recon
