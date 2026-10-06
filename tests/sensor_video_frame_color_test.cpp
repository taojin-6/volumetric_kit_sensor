// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The colour a decoded frame declares, read off frames built by hand, no codec
// and no GPU: the matrix and encoding a stream's tags resolve to, the range,
// the chroma siting, and the unlabelled colour standing in for a frame that
// declares no matrix.

#include <cstdio>
#include <optional>

#include "ffmpeg.hpp"
#include "frame_color.hpp"

namespace sensor = volumetric_kit::recon::sensor;
namespace video = volumetric_kit::recon::sensor::video;
using sensor::ChromaLocation;
using sensor::VideoColorMatrix;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

// What describe_color makes of @p frame, for a picture @p height rows tall.
sensor::DecodedPicture described(
    const AVFrame& frame, int height,
    const std::optional<sensor::VideoColorDescription>& unlabelled =
        std::nullopt) {
  sensor::DecodedPicture picture;
  picture.height = static_cast<std::uint32_t>(height);
  video::describe_color(frame, unlabelled, picture);
  return picture;
}

int test_resolve_matrix() {
  CHECK(video::resolve_matrix(AVCOL_SPC_BT709, 144) == VideoColorMatrix::Bt709);
  CHECK(video::resolve_matrix(AVCOL_SPC_SMPTE170M, 2160) ==
        VideoColorMatrix::Bt601);
  CHECK(video::resolve_matrix(AVCOL_SPC_BT470BG, 720) ==
        VideoColorMatrix::Bt601);
  CHECK(video::resolve_matrix(AVCOL_SPC_BT2020_NCL, 720) ==
        VideoColorMatrix::Bt2020);
  CHECK(video::resolve_matrix(AVCOL_SPC_UNSPECIFIED, 576) ==
        VideoColorMatrix::Bt601);
  CHECK(video::resolve_matrix(AVCOL_SPC_UNSPECIFIED, 720) ==
        VideoColorMatrix::Bt709);
  CHECK(video::resolve_matrix(AVCOL_SPC_SMPTE240M, 1080) ==
        VideoColorMatrix::Smpte240m);
  CHECK(video::resolve_matrix(AVCOL_SPC_FCC, 480) == VideoColorMatrix::Fcc);
  CHECK(video::resolve_matrix(AVCOL_SPC_RGB, 720) == VideoColorMatrix::Bt709);
  return 0;
}

int test_resolve_encoding() {
  using Transfer = volumetric_kit::recon::ColorEncoding::Transfer;
  using Primaries = volumetric_kit::recon::ColorEncoding::Primaries;
  const auto is = [](std::optional<volumetric_kit::recon::ColorEncoding> e,
                     Transfer t, Primaries p) {
    return e.has_value() && e->transfer == t && e->primaries == p;
  };
  CHECK(is(video::resolve_encoding(AVCOL_TRC_BT709, AVCOL_PRI_BT709),
           Transfer::Bt709, Primaries::Bt709));
  CHECK(
      is(video::resolve_encoding(AVCOL_TRC_UNSPECIFIED, AVCOL_PRI_UNSPECIFIED),
         Transfer::Bt709, Primaries::Bt709));
  CHECK(is(video::resolve_encoding(AVCOL_TRC_IEC61966_2_1, AVCOL_PRI_SMPTE432),
           Transfer::Srgb, Primaries::DisplayP3));
  CHECK(is(video::resolve_encoding(AVCOL_TRC_SMPTE2084, AVCOL_PRI_BT2020),
           Transfer::Bt2020Pq, Primaries::Bt2020));
  CHECK(is(video::resolve_encoding(AVCOL_TRC_BT2020_10, AVCOL_PRI_BT2020),
           Transfer::Bt709, Primaries::Bt2020));
  CHECK(is(video::resolve_encoding(AVCOL_TRC_LINEAR, AVCOL_PRI_BT709),
           Transfer::Linear, Primaries::Bt709));
  // Nothing true to declare: HLG, and BT.601's own primaries.
  CHECK(!video::resolve_encoding(AVCOL_TRC_ARIB_STD_B67, AVCOL_PRI_BT2020));
  CHECK(!video::resolve_encoding(AVCOL_TRC_SMPTE170M, AVCOL_PRI_SMPTE170M));

  video::FramePtr frame(av_frame_alloc());
  frame->color_trc = AVCOL_TRC_SMPTE2084;
  frame->color_primaries = AVCOL_PRI_BT2020;
  CHECK(is(described(*frame, 720).encoding, Transfer::Bt2020Pq,
           Primaries::Bt2020));
  frame->color_trc = AVCOL_TRC_ARIB_STD_B67;  // HLG: nothing true to declare
  CHECK(!described(*frame, 720).encoding);
  return 0;
}

// The matrix by the frame's tag or its height, and the range by its tag.
int test_matrix_and_range() {
  video::FramePtr frame(av_frame_alloc());
  frame->colorspace = AVCOL_SPC_BT709;
  frame->color_range = AVCOL_RANGE_MPEG;
  sensor::DecodedPicture p = described(*frame, 144);
  CHECK(p.matrix == VideoColorMatrix::Bt709 && !p.full_range);
  frame->colorspace = AVCOL_SPC_UNSPECIFIED;
  frame->color_range = AVCOL_RANGE_JPEG;
  p = described(*frame, 576);
  CHECK(p.matrix == VideoColorMatrix::Bt601 && p.full_range);
  p = described(*frame, 720);
  CHECK(p.matrix == VideoColorMatrix::Bt709 && p.full_range);
  return 0;
}

// The unlabelled colour stands in for a frame that declares no matrix -- its
// range too, since FFmpeg reads a stream that declares neither as limited. A
// frame that declares one keeps it.
int test_unlabelled_color() {
  const sensor::VideoColorDescription femto{VideoColorMatrix::Bt601, true};
  video::FramePtr frame(av_frame_alloc());
  frame->colorspace = AVCOL_SPC_UNSPECIFIED;
  frame->color_range = AVCOL_RANGE_MPEG;
  sensor::DecodedPicture p = described(*frame, 720, femto);
  CHECK(p.matrix == VideoColorMatrix::Bt601 && p.full_range);
  frame->colorspace = AVCOL_SPC_BT709;
  p = described(*frame, 720, femto);
  CHECK(p.matrix == VideoColorMatrix::Bt709 && !p.full_range);
  return 0;
}

// Every chroma siting FFmpeg names, and Left for one it does not.
int test_chroma_locations() {
  const struct {
    AVChromaLocation tag;
    ChromaLocation want;
  } cases[] = {
      {AVCHROMA_LOC_LEFT, ChromaLocation::Left},
      {AVCHROMA_LOC_CENTER, ChromaLocation::Center},
      {AVCHROMA_LOC_TOPLEFT, ChromaLocation::TopLeft},
      {AVCHROMA_LOC_TOP, ChromaLocation::Top},
      {AVCHROMA_LOC_BOTTOMLEFT, ChromaLocation::BottomLeft},
      {AVCHROMA_LOC_BOTTOM, ChromaLocation::Bottom},
      {AVCHROMA_LOC_UNSPECIFIED, ChromaLocation::Left},
  };
  video::FramePtr frame(av_frame_alloc());
  for (const auto& c : cases) {
    frame->chroma_location = c.tag;
    CHECK(described(*frame, 720).chroma_location == c.want);
  }
  return 0;
}

}  // namespace

int main() {
  if (test_resolve_matrix() != 0) return 1;
  if (test_resolve_encoding() != 0) return 1;
  if (test_matrix_and_range() != 0) return 1;
  if (test_unlabelled_color() != 0) return 1;
  if (test_chroma_locations() != 0) return 1;
  std::printf("frame colour tests passed\n");
  return 0;
}
