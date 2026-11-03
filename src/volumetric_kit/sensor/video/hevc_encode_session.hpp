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
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

#if defined(__APPLE__)
#include "videotoolbox.hpp"
#endif

namespace volumetric_kit::sensor::video {

// A picture to code, held until it is: on Apple a pixel buffer, retained;
// and, for a target the device writes, the value its writer sets once the
// picture is written, which the coding thread waits for on the host (a null
// semaphore for none).
struct EncodePicture {
#if defined(__APPLE__)
  PixelBufferPtr pixels;
#endif
  core::TimelinePoint written;
};

// Not thread-safe: HevcEncoder's coding thread codes on it, and its sender
// opens, loses and closes it only while the thread is not coding. target()
// is the one call the sender makes while the thread codes: it reaches only
// the session's pool and its own state, which encode does not touch.
class HevcEncodeSession {
 public:
  // Whether this backend writes a chroma siting other than 0 into the VUI.
  static const bool kWritesSiting;

  // A session for @p options, checked and with its colour normalized by
  // HevcEncoder::create, and what it says of itself into @p facts; with
  // Options::device, ready to hand out targets on it. With @p previous, the
  // lost session it replaces: its hardware session is closed before this
  // one opens, and the images of the targets it handed out, which a batch
  // may still be writing, pass to this one once it has opened; if it does
  // not, @p previous keeps them. Unsupported naming what is missing, or the
  // backend's failure.
  static core::Result<std::unique_ptr<HevcEncodeSession>> open(
      const HevcEncoder::Options& options, HevcEncoder::Facts& facts,
      HevcEncodeSession* previous = nullptr);

  ~HevcEncodeSession();
  HevcEncodeSession(const HevcEncodeSession&) = delete;
  HevcEncodeSession& operator=(const HevcEncodeSession&) = delete;
  HevcEncodeSession(HevcEncodeSession&&) = delete;
  HevcEncodeSession& operator=(HevcEncodeSession&&) = delete;

  // @p picture coded, an IDR at the key QP when @p key, else a P-picture,
  // into @p access_unit (replaced; Annex B as CodedPicture says), before
  // this returns, and into @p idr whether the hardware marked it a key,
  // which CodedPicture::key reports. True once coded; false if dropped, the
  // hardware's drop or @p drop (a test's fault, through the same path),
  // @p access_unit then empty; or the failure, which loses the session.
  // Never throws: a failure inside is its Status.
  core::Result<bool> encode(EncodePicture& picture, bool key, bool drop,
                            std::vector<std::uint8_t>& access_unit,
                            bool& idr) noexcept;

  // A free picture for the device to write, held in @p picture's backend
  // part until it is sent or given up, its planes into @p out (layout,
  // planes and size; HevcEncoder sets the colour description), and what the
  // planes need recorded into @p batch. Only on a session opened with a
  // device. Its failure, or @p batch's, changes nothing; @p out is then as
  // it was.
  core::Status target(core::CommandBatch& batch, EncodePicture& picture,
                      YuvImage& out);

  // The session lost, as when the system reclaims it: every later encode
  // fails through the backend's own path.
  void lose() noexcept;

  struct Impl;

 private:
  explicit HevcEncodeSession(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor::video
