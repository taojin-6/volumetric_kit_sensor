// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What a platform's way into HevcEncoder reaches of it: the options it
// opened with, and its send, which applies the rules every way in shares and
// hands the picture to the coding thread. The Apple entry
// (hevc_encoder_apple.cpp) is its one user; the encoder names it its friend.
// Internal.

#include <cstdint>

#include "hevc_encode_session.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"

namespace volumetric_kit::sensor {

struct HevcEncoderAccess {
  // The options @p encoder opened with, colour normalized; null on a
  // moved-from encoder.
  static const HevcEncoder::Options* options(
      const HevcEncoder& encoder) noexcept;

  // @p picture sent: InvalidArgument for kInFlight pictures already in
  // flight, no key while one is owed, or a moved-from encoder; a lost
  // session opened again first, its failure as create's; else queued for
  // the coding thread, returning at once.
  static core::Status send(HevcEncoder& encoder, video::EncodePicture picture,
                           bool key, std::int64_t pts);
};

}  // namespace volumetric_kit::sensor
