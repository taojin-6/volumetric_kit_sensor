// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The colour a stream declares, from H.273 codes set by hand, no codec and no
// GPU: the matrix and encoding the codes resolve to, the weights and range a
// picture is described with, the chroma siting, and the unlabelled colour
// standing in for codes that declare no matrix.

#include <cstdio>
#include <optional>

#include "frame_color.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "yuv_reference.hpp"

namespace sensor = volumetric_kit::sensor;
namespace video = volumetric_kit::sensor::video;
using sensor::ChromaLocation;
using sensor::VideoColorMatrix;

namespace {

// H.273's code points used here.
constexpr int kRgb = 0;
constexpr int kBt709 = 1;
constexpr int kUnspecified = 2;
constexpr int kFcc = 4;
constexpr int kBt470bg = 5;
constexpr int kSmpte170m = 6;
constexpr int kSmpte240m = 7;
constexpr int kBt2020 = 9;  // the matrix's non-constant luminance; primaries
constexpr int kBt2020Ten = 14;
constexpr int kLinear = 8;
constexpr int kSrgb = 13;
constexpr int kPq = 16;
constexpr int kHlg = 18;
constexpr int kDisplayP3 = 12;

// What describe_color makes of @p codes, for a picture @p height rows tall.
sensor::DecodedPicture described(
    const sensor::ColorCodes& codes, int height,
    const std::optional<sensor::VideoColorDescription>& unlabelled =
        std::nullopt) {
  sensor::DecodedPicture picture;
  picture.yuv.height = static_cast<std::uint32_t>(height);
  video::describe_color(codes, unlabelled, picture);
  return picture;
}

int test_resolve_matrix() {
  CHECK(video::resolve_matrix(kBt709, 144) == VideoColorMatrix::Bt709);
  CHECK(video::resolve_matrix(kSmpte170m, 2160) == VideoColorMatrix::Bt601);
  CHECK(video::resolve_matrix(kBt470bg, 720) == VideoColorMatrix::Bt601);
  CHECK(video::resolve_matrix(kBt2020, 720) == VideoColorMatrix::Bt2020);
  CHECK(video::resolve_matrix(kUnspecified, 576) == VideoColorMatrix::Bt601);
  CHECK(video::resolve_matrix(kUnspecified, 720) == VideoColorMatrix::Bt709);
  CHECK(video::resolve_matrix(kSmpte240m, 1080) == VideoColorMatrix::Smpte240m);
  CHECK(video::resolve_matrix(kFcc, 480) == VideoColorMatrix::Fcc);
  CHECK(video::resolve_matrix(kRgb, 720) == VideoColorMatrix::Bt709);
  return 0;
}

int test_resolve_encoding() {
  using Transfer = volumetric_kit::sensor::ColorEncoding::Transfer;
  using Primaries = volumetric_kit::sensor::ColorEncoding::Primaries;
  const auto is = [](std::optional<volumetric_kit::sensor::ColorEncoding> e,
                     Transfer t, Primaries p) {
    return e.has_value() && e->transfer == t && e->primaries == p;
  };
  CHECK(is(video::resolve_encoding(kBt709, kBt709), Transfer::Bt709,
           Primaries::Bt709));
  CHECK(is(video::resolve_encoding(kUnspecified, kUnspecified), Transfer::Bt709,
           Primaries::Bt709));
  CHECK(is(video::resolve_encoding(kSrgb, kDisplayP3), Transfer::Srgb,
           Primaries::DisplayP3));
  CHECK(is(video::resolve_encoding(kPq, kBt2020), Transfer::Bt2020Pq,
           Primaries::Bt2020));
  CHECK(is(video::resolve_encoding(kBt2020Ten, kBt2020), Transfer::Bt709,
           Primaries::Bt2020));
  CHECK(is(video::resolve_encoding(kLinear, kBt709), Transfer::Linear,
           Primaries::Bt709));
  // Nothing true to declare: HLG, and BT.601's own primaries.
  CHECK(!video::resolve_encoding(kHlg, kBt2020));
  CHECK(!video::resolve_encoding(kSmpte170m, kSmpte170m));

  sensor::ColorCodes codes;
  codes.transfer = kPq;
  codes.primaries = kBt2020;
  CHECK(is(described(codes, 720).encoding, Transfer::Bt2020Pq,
           Primaries::Bt2020));
  codes.transfer = kHlg;  // nothing true to declare
  CHECK(!described(codes, 720).encoding);
  return 0;
}

// The matrix by the code or the height, described by its weights, and the
// range by the flag.
int test_matrix_and_range() {
  sensor::ColorCodes codes;
  const struct {
    int code;
    VideoColorMatrix want;
  } cases[] = {
      {kBt709, VideoColorMatrix::Bt709},
      {kSmpte170m, VideoColorMatrix::Bt601},
      {kBt2020, VideoColorMatrix::Bt2020},
      {kSmpte240m, VideoColorMatrix::Smpte240m},
      {kFcc, VideoColorMatrix::Fcc},
  };
  for (const auto& c : cases) {
    codes.matrix = c.code;
    CHECK(yuv_reference::coded_in(described(codes, 144).yuv, c.want, false));
  }
  codes.matrix = kUnspecified;
  codes.full_range = true;
  CHECK(yuv_reference::coded_in(described(codes, 576).yuv,
                                VideoColorMatrix::Bt601, true));
  CHECK(yuv_reference::coded_in(described(codes, 720).yuv,
                                VideoColorMatrix::Bt709, true));
  return 0;
}

// The unlabelled colour stands in for codes that declare no matrix -- its
// range too, since a stream that declares neither reads as limited. Codes
// that declare one keep it.
int test_unlabelled_color() {
  const sensor::VideoColorDescription femto{VideoColorMatrix::Bt601, true};
  sensor::ColorCodes codes;
  CHECK(yuv_reference::coded_in(described(codes, 720, femto).yuv,
                                VideoColorMatrix::Bt601, true));
  codes.matrix = kBt709;
  CHECK(yuv_reference::coded_in(described(codes, 720, femto).yuv,
                                VideoColorMatrix::Bt709, false));
  return 0;
}

// Every chroma siting H.273 names, and Left for a code it does not.
int test_chroma_locations() {
  const struct {
    int code;
    ChromaLocation want;
  } cases[] = {
      {0, ChromaLocation::Left},       {1, ChromaLocation::Center},
      {2, ChromaLocation::TopLeft},    {3, ChromaLocation::Top},
      {4, ChromaLocation::BottomLeft}, {5, ChromaLocation::Bottom},
      {6, ChromaLocation::Left},
  };
  sensor::ColorCodes codes;
  CHECK(described(codes, 720).yuv.chroma_location == ChromaLocation::Left);
  for (const auto& c : cases) {
    codes.chroma_location = c.code;
    CHECK(described(codes, 720).yuv.chroma_location == c.want);
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
