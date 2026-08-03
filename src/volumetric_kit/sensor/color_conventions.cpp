// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/color_conventions.hpp"

#include <cstring>

namespace volumetric_kit::recon::sensor {

Status to_canonical(const std::uint32_t* src, std::size_t count,
                    const ColorEncoding& enc, std::uint32_t* dst) {
  if (count == 0) {
    return Status{};
  }
  if (src == nullptr || dst == nullptr) {
    return Status::invalid_argument(
        "to_canonical: src and dst must be non-null for a non-zero count");
  }
  if (enc.transfer == ColorEncoding::Transfer::Bt2020Pq) {
    // PQ is an absolute-luminance HDR curve; landing it in an 8-bit SDR
    // canonical form is a tone-mapping problem, not a transfer conversion.
    // Report it rather than approximate, so a driver that declares PQ gets an
    // inspectable error instead of a quietly wrong reconstruction.
    return Status::unsupported(
        "to_canonical: Transfer::Bt2020Pq needs tone mapping into the 8-bit "
        "canonical form, which is not implemented");
  }

  if (is_canonical(enc)) {
    // The common path by construction: the canonical form was chosen to match
    // what the sensors in hand produce, so an ARKit frame lands here. Copy the
    // bytes (or nothing, in place) rather than round-tripping every pixel
    // through the curve -- an identity conversion should cost an identity.
    if (dst != src) {
      std::memcpy(dst, src, count * sizeof(std::uint32_t));
    }
    return Status{};
  }

  // Two steps, and the second is the one that is easy to forget: decode the
  // declared transfer to linear, THEN rotate into the working primaries. A
  // decode alone would leave the values linear but in the wrong RGB basis.
  const bool already_linear = enc.transfer == ColorEncoding::Transfer::Linear;
  const Mat3f to_working = primaries_to_working(enc.primaries);
  const bool rotate = enc.primaries != ColorEncoding::Primaries::Bt709;

  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t p = src[i];
    const float inv = 1.0f / 255.0f;
    Vec3f v{static_cast<float>(p & 0xFFu) * inv,
            static_cast<float>((p >> 8) & 0xFFu) * inv,
            static_cast<float>((p >> 16) & 0xFFu) * inv};
    if (!already_linear) {
      v = srgb_to_linear(v);
    }
    if (rotate) {
      // Clipping is possible and booked: BT.709 is the narrowest declarable
      // gamut, so a saturated wide-gamut colour leaves [0, 1] here and
      // pack_linear_to_srgb clamps it. An 8-bit canonical attribute could not
      // have carried it either way.
      v = to_working * v;
    }
    dst[i] = pack_linear_to_srgb(v);
  }
  return Status{};
}

}  // namespace volumetric_kit::recon::sensor
