// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// FFmpeg's helpers (ffmpeg.hpp), with no codec and no GPU: the Status an
// FFmpeg error code becomes (ffmpeg_error) -- out of memory, a hardware call
// that failed inside FFmpeg, which names FFmpeg as the backend and keeps its
// code, and anything else as I/O --, what an NVENC session that did not
// open says (nvenc_open_failure), and the H.273 codes a frame's colour
// fields become (color_codes).

#include <cerrno>
#include <cstdio>
#include <string>

#include "ffmpeg.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace vkc = volumetric_kit::core;
namespace video = volumetric_kit::sensor::video;
using Code = vkc::Status::Code;

namespace {

int test_backend() {
  const vkc::Status s =
      video::ffmpeg_error("HevcDecoder", "decoding", AVERROR_EXTERNAL);
  CHECK(s.domain() == Code::Backend);
  CHECK(s.backend() == vkc::Status::Backend::Ffmpeg);
  CHECK(s.detail() == AVERROR_EXTERNAL);
  CHECK(s.message() ==
        "HevcDecoder: decoding: " + video::ffmpeg_message(AVERROR_EXTERNAL));
  return 0;
}

int test_other_domains() {
  const vkc::Status oom =
      video::ffmpeg_error("HevcDecoder", "decoding", AVERROR(ENOMEM));
  CHECK(oom.domain() == Code::OutOfMemory);
  CHECK(!oom.backend());
  const vkc::Status io =
      video::ffmpeg_error("HevcDecoder", "decoding", AVERROR_INVALIDDATA);
  CHECK(io.domain() == Code::IoError);
  CHECK(!io.backend());
  return 0;
}

// NVENC's open failures: each refusal by NVENC or its driver Unsupported,
// saying which and giving the retry only to what another session closing
// can clear; any other code as ffmpeg_error (EAGAIN, EIO and FFmpeg's
// unknown error IoError, a hardware call's failure Backend).
int test_nvenc_open_failure() {
  const std::string what = "a session of 256x144";
  const auto check = [&](int err, Code want, const std::string& says) {
    const vkc::Status s = video::nvenc_open_failure(what, err);
    if (s.domain() != want || s.message().find(says) == std::string::npos ||
        s.message().find(video::ffmpeg_message(err)) == std::string::npos) {
      std::fprintf(stderr, "err %d: %s\n", err, s.message().c_str());
      return false;
    }
    return true;
  };
  for (const int err : {AVERROR(EINVAL), AVERROR(ENOMEM)}) {
    CHECK(check(err, Code::Unsupported, "session limit"));
    CHECK(check(err, Code::Unsupported, "does not take a parameter of it"));
  }
  for (const int err : {AVERROR(ENOSYS), AVERROR(ENOENT)}) {
    CHECK(check(err, Code::Unsupported, "does not code " + what));
    CHECK(video::nvenc_open_failure(what, err).message().find("closing") ==
          std::string::npos);
  }
  CHECK(check(-1, Code::Unsupported, "libnvidia-encode"));
  CHECK(check(AVERROR_EXTERNAL, Code::Backend, "opening " + what));
  for (const int err : {AVERROR(EAGAIN), AVERROR(EIO), AVERROR_UNKNOWN}) {
    CHECK(check(err, Code::IoError, "opening " + what));
  }
  return 0;
}

// FFmpeg's colour enumerations are H.273's numbers, the chroma sitings one
// past them: an unspecified frame is unspecified codes, limited range and
// left-sited chroma.
int test_color_codes() {
  video::FramePtr frame(av_frame_alloc());
  CHECK(frame != nullptr);
  volumetric_kit::sensor::ColorCodes codes = video::color_codes(*frame);
  CHECK(codes.matrix == 2 && codes.transfer == 2 && codes.primaries == 2);
  CHECK(!codes.full_range && codes.chroma_location == 0);
  frame->colorspace = AVCOL_SPC_BT2020_NCL;
  frame->color_trc = AVCOL_TRC_SMPTE2084;
  frame->color_primaries = AVCOL_PRI_SMPTE432;
  frame->color_range = AVCOL_RANGE_JPEG;
  frame->chroma_location = AVCHROMA_LOC_TOPLEFT;
  codes = video::color_codes(*frame);
  CHECK(codes.matrix == 9 && codes.transfer == 16 && codes.primaries == 12);
  CHECK(codes.full_range && codes.chroma_location == 2);
  frame->chroma_location = AVCHROMA_LOC_LEFT;
  CHECK(video::color_codes(*frame).chroma_location == 0);
  frame->chroma_location = AVCHROMA_LOC_BOTTOM;
  CHECK(video::color_codes(*frame).chroma_location == 5);
  return 0;
}

}  // namespace

int main() {
  if (test_backend() != 0) return 1;
  if (test_other_domains() != 0) return 1;
  if (test_nvenc_open_failure() != 0) return 1;
  if (test_color_codes() != 0) return 1;
  std::printf("ffmpeg helper tests passed\n");
  return 0;
}
