// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host tests for the color-space vocabulary: the exact piecewise sRGB curve,
// the packed helpers the fusion kernels mirror, the primaries matrices that
// make ColorEncoding::Primaries a value something *converts* rather than a
// label something reads, and the difference between blending linear and
// encoded values.
//
// Pure host math over the core vocabulary, so this always runs: no device, no
// platform, no driver. That is the point of the curve living in `core`.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
// The braced-init range-for below deduces std::initializer_list. libc++ pulls
// this in transitively; libstdc++ does not, so name it.
#include <initializer_list>

#include "test_check.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"

namespace vr = volumetric_kit::recon;

int main() {
  // --- The curve is the EXACT piecewise sRGB function -----------------------
  // Not pow(x, 2.2). The distinction is load-bearing rather than pedantic:
  // hardware _SRGB sampling decodes the atlas with the exact curve, and gfx's
  // hybrid pipeline picks between the atlas and the vertex color per triangle
  // across one surface, so an approximated voxel decode seams exactly where
  // texturing stops. These spot values discriminate the two -- pow(0.5, 2.2) is
  // 0.2176 where the exact curve gives 0.2140, ~1% apart and far outside the
  // tolerance below.
  CHECK(std::fabs(vr::srgb_to_linear(0.0f) - 0.0f) < 1e-7f);
  CHECK(std::fabs(vr::srgb_to_linear(1.0f) - 1.0f) < 1e-6f);
  CHECK(std::fabs(vr::srgb_to_linear(0.5f) - 0.21404f) < 1e-4f);
  CHECK(std::fabs(vr::linear_to_srgb(0.21404f) - 0.5f) < 1e-4f);
  // The linear toe below the 0.04045 knee: slope exactly 1/12.92, which a pure
  // power function has nowhere.
  CHECK(std::fabs(vr::srgb_to_linear(0.02f) - (0.02f / 12.92f)) < 1e-7f);
  CHECK(std::fabs(vr::linear_to_srgb(0.001f) - (0.001f * 12.92f)) < 1e-7f);

  // --- Every one of the 256 codes round-trips exactly -----------------------
  // The property the packed helpers rest on, and the one that lets the tsdf
  // integrator store an 8-bit attribute at all: decode to linear, re-encode,
  // and land on the same code. A curve that drifted (or a pack that truncated
  // instead of rounding) would show up as an off-by-one somewhere in here.
  for (int code = 0; code < 256; ++code) {
    const auto packed = static_cast<std::uint32_t>(code) |
                        (static_cast<std::uint32_t>(code) << 8) |
                        (static_cast<std::uint32_t>(code) << 16);
    const vr::Vec3f linear = vr::unpack_srgb_to_linear(packed);
    const std::uint32_t back = vr::pack_linear_to_srgb(linear);
    CHECK((back & 0xFFu) == static_cast<std::uint32_t>(code));
    CHECK(((back >> 8) & 0xFFu) == static_cast<std::uint32_t>(code));
    CHECK(((back >> 16) & 0xFFu) == static_cast<std::uint32_t>(code));
    // Alpha is always 0xFF, which is what keeps the tsdf tier's "color
    // unobserved" sentinel exact -- a written color is never 0, not even black.
    CHECK((back >> 24) == 0xFFu);
  }
  // Pure black packs non-zero precisely because of that alpha. Checked
  // explicitly because it is the sentinel's only interesting case.
  CHECK(vr::pack_linear_to_srgb(vr::Vec3f(0.0f)) == 0xFF000000u);

  // --- Out-of-range linear values clamp on encode ---------------------------
  // Wide-gamut sources land outside [0,1] after the primaries rotation, and an
  // 8-bit canonical attribute cannot carry that. Clipping is the booked cost.
  CHECK((vr::pack_linear_to_srgb(vr::Vec3f(-0.5f, 2.0f, 0.5f)) & 0xFFu) == 0u);
  CHECK(((vr::pack_linear_to_srgb(vr::Vec3f(-0.5f, 2.0f, 0.5f)) >> 8) &
         0xFFu) == 255u);

  // --- Blending in linear is NOT blending encoded ---------------------------
  // The whole reason this header exists. Two observations of linear 0.0 and 1.0
  // average to 0.5 in the working space; averaging their *codes* and decoding
  // gives 0.214 -- a plausibly darker surface rather than a visible error.
  {
    const float encoded_mean =
        0.5f * (vr::linear_to_srgb(0.0f) + vr::linear_to_srgb(1.0f));
    CHECK(std::fabs(vr::srgb_to_linear(encoded_mean) - 0.21404f) < 1e-3f);
    CHECK(std::fabs(vr::srgb_to_linear(encoded_mean) - 0.5f) > 0.2f);
    // And the smaller, more insidious case the decision quotes: similar samples
    // are off by a few percent, not by a factor.
    const float m2 =
        0.5f * (vr::linear_to_srgb(0.2f) + vr::linear_to_srgb(0.6f));
    CHECK(std::fabs(vr::srgb_to_linear(m2) - 0.3687f) < 1e-3f);
  }

  // --- Primaries: white maps to white, for every declarable set --------------
  // All three are D65, so the conversion is a pure primaries change and pure
  // white must survive it exactly. Each matrix's rows therefore sum to one --
  // an invariant that catches a transposed or mistyped matrix, which a single
  // spot value might not (a transpose preserves the diagonal).
  {
    const vr::ColorEncoding::Primaries all[] = {
        vr::ColorEncoding::Primaries::Bt709,
        vr::ColorEncoding::Primaries::DisplayP3,
        vr::ColorEncoding::Primaries::Bt2020,
    };
    for (vr::ColorEncoding::Primaries p : all) {
      const vr::Vec3f white = vr::primaries_to_working(p) * vr::Vec3f(1.0f);
      CHECK(std::fabs(white.x - 1.0f) < 1e-4f);
      CHECK(std::fabs(white.y - 1.0f) < 1e-4f);
      CHECK(std::fabs(white.z - 1.0f) < 1e-4f);
    }
    // BT.709 is the working space itself, so its conversion is the identity --
    // which is exactly why it would go unnoticed if it were wrong.
    const vr::Vec3f v(0.3f, 0.6f, 0.9f);
    const vr::Vec3f same =
        vr::primaries_to_working(vr::ColorEncoding::Primaries::Bt709) * v;
    CHECK(same.x == v.x && same.y == v.y && same.z == v.z);
    // A saturated P3 green is OUTSIDE BT.709: the rotation must push it past
    // 1.0 (and its off-channels negative), which is the clipping the decision
    // books. If this came back inside the cube the matrix would be doing
    // nothing.
    const vr::Vec3f p3_green =
        vr::primaries_to_working(vr::ColorEncoding::Primaries::DisplayP3) *
        vr::Vec3f(0.0f, 1.0f, 0.0f);
    CHECK(p3_green.y > 1.0f);
    CHECK(p3_green.x < 0.0f);
  }

  // --- is_canonical: both halves must hold ----------------------------------
  CHECK(vr::is_canonical(vr::ColorEncoding{}));  // the default IS canonical
  CHECK(vr::is_canonical({vr::ColorEncoding::Transfer::Srgb,
                          vr::ColorEncoding::Primaries::Bt709}));
  // BT.709's transfer is ACCEPTED as sRGB -- a bounded, stated error, and what
  // makes the ARKit path a genuine no-op rather than a per-frame image pass.
  CHECK(vr::is_canonical({vr::ColorEncoding::Transfer::Bt709,
                          vr::ColorEncoding::Primaries::Bt709}));
  // ...but a wide gamut is not canonical even with an sRGB transfer, because
  // its linear values live in a different RGB basis.
  CHECK(!vr::is_canonical({vr::ColorEncoding::Transfer::Srgb,
                           vr::ColorEncoding::Primaries::DisplayP3}));
  CHECK(!vr::is_canonical({vr::ColorEncoding::Transfer::Linear,
                           vr::ColorEncoding::Primaries::Bt709}));
  CHECK(!vr::is_canonical({vr::ColorEncoding::Transfer::Bt2020Pq,
                           vr::ColorEncoding::Primaries::Bt2020}));

  // --- pack_linear_to_srgb is total over every float, NaN included ----------
  // The clamp was `e < 0 ? 0 : (e > 1 ? 1 : e)`, which NaN slips through --
  // it compares false to both bounds -- straight into a float-to-unsigned
  // conversion that is **undefined** for a value it cannot represent. In a
  // header this library installs, and one the -fno-sanitize-recover UBSan leg
  // would abort on rather than merely produce a wrong colour. A NaN here is a
  // caller bug, but it must not be undefined behaviour.
  //
  // The assertion is only that these are the documented in-range results; the
  // teeth are that the sanitizer legs execute the line at all.
  {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    // NaN takes the same road as every other out-of-range input: 0.
    CHECK(vr::pack_linear_to_srgb(vr::Vec3f{nan, nan, nan}) == 0xFF000000u);
    // ...and the ordinary out-of-range cases still saturate the way they did.
    CHECK(vr::pack_linear_to_srgb(vr::Vec3f{-inf, -1.0f, -0.0f}) ==
          0xFF000000u);
    CHECK(vr::pack_linear_to_srgb(vr::Vec3f{inf, 2.0f, 1.0f}) == 0xFFFFFFFFu);
    // Mixed, so a channel-independent clamp is what is being tested rather than
    // an all-or-nothing early out.
    CHECK(vr::pack_linear_to_srgb(vr::Vec3f{nan, 1.0f, 0.0f}) == 0xFF00FF00u);
    // Alpha is forced on every path, including this one.
    CHECK((vr::pack_linear_to_srgb(vr::Vec3f{nan, nan, nan}) >> 24) == 0xFFu);
  }

  std::printf(
      "recon core color-space test passed: exact piecewise sRGB (discriminated "
      "from pow 2.2), all 256 codes round-trip, primaries preserve white and "
      "P3 green clips outside BT.709, and is_canonical accepts "
      "Bt709-as-sRGB\n");
  return 0;
}
