// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// One session of this build's hardware HEVC encoder, in the one
// configuration HevcEncoder describes: the seam between HevcEncoder's rules
// and coding thread (hevc_encoder.cpp, every backend's) and the backend. One
// definition a build, no virtual interface: VideoToolbox's on Apple
// (vt_hevc_encode.cpp), and on a build with no encoder the refusal in
// hevc_encoder.cpp. Internal.

#include <cstdint>
#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"

#if defined(__APPLE__)
#include "videotoolbox.hpp"
#endif

namespace volumetric_kit::sensor::video {

// A picture to code, held until it is: on Apple a pixel buffer, retained.
struct EncodePicture {
#if defined(__APPLE__)
  PixelBufferPtr pixels;
#endif
};

// Not thread-safe: HevcEncoder's coding thread codes on it, and its sender
// opens, loses and closes it only while the thread is not coding.
class HevcEncodeSession {
 public:
  // Whether this backend writes a chroma siting other than 0 into the VUI.
  static const bool kWritesSiting;

  // A session for @p options, checked and with its colour normalized by
  // HevcEncoder::create, and what it says of itself into @p facts.
  // Unsupported naming what is missing, or the backend's failure.
  static core::Result<std::unique_ptr<HevcEncodeSession>> open(
      const HevcEncoder::Options& options, HevcEncoder::Facts& facts);

  ~HevcEncodeSession();
  HevcEncodeSession(const HevcEncodeSession&) = delete;
  HevcEncodeSession& operator=(const HevcEncodeSession&) = delete;
  HevcEncodeSession(HevcEncodeSession&&) = delete;
  HevcEncodeSession& operator=(HevcEncodeSession&&) = delete;

  // @p picture coded, an IDR at the key QP when @p key, else a P-picture,
  // into @p access_unit (replaced; Annex B as CodedPicture says), before
  // this returns. True once coded; false if dropped, the hardware's drop
  // or @p drop (a test's fault, through the same path), @p access_unit
  // then empty; or the failure, which loses the session.
  core::Result<bool> encode(EncodePicture& picture, bool key, bool drop,
                            std::vector<std::uint8_t>& access_unit);

  // The session lost, as when the system reclaims it: every later encode
  // fails through the backend's own path.
  void lose() noexcept;

  struct Impl;

 private:
  explicit HevcEncodeSession(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor::video
