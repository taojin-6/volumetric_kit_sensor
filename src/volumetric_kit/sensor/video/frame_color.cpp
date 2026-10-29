// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "frame_color.hpp"

namespace volumetric_kit::sensor::video {
namespace {

// The H.273 code points named here.
constexpr int kMatrixBt709 = 1;
constexpr int kMatrixUnspecified = 2;
constexpr int kMatrixFcc = 4;
constexpr int kMatrixBt470bg = 5;
constexpr int kMatrixSmpte170m = 6;
constexpr int kMatrixSmpte240m = 7;
constexpr int kMatrixBt2020Ncl = 9;
constexpr int kMatrixBt2020Cl = 10;
constexpr int kTransferBt709 = 1;
constexpr int kTransferUnspecified = 2;
constexpr int kTransferSmpte170m = 6;
constexpr int kTransferLinear = 8;
constexpr int kTransferSrgb = 13;  // IEC 61966-2-1
constexpr int kTransferBt2020Ten = 14;
constexpr int kTransferBt2020Twelve = 15;
constexpr int kTransferPq = 16;  // SMPTE ST 2084
constexpr int kPrimariesBt709 = 1;
constexpr int kPrimariesUnspecified = 2;
constexpr int kPrimariesBt2020 = 9;
constexpr int kPrimariesDisplayP3 = 12;  // SMPTE EG 432-1

}  // namespace

VideoColorMatrix resolve_matrix(int matrix, int height) noexcept {
  switch (matrix) {
    case kMatrixBt709:
      return VideoColorMatrix::Bt709;
    case kMatrixBt470bg:
    case kMatrixSmpte170m:
      return VideoColorMatrix::Bt601;
    case kMatrixBt2020Ncl:
    case kMatrixBt2020Cl:
      return VideoColorMatrix::Bt2020;
    case kMatrixSmpte240m:
      return VideoColorMatrix::Smpte240m;
    case kMatrixFcc:
      return VideoColorMatrix::Fcc;
    default:
      break;
  }
  return height > 576 ? VideoColorMatrix::Bt709 : VideoColorMatrix::Bt601;
}

std::optional<ColorEncoding> resolve_encoding(int transfer,
                                              int primaries) noexcept {
  ColorEncoding encoding;
  switch (transfer) {
    case kTransferUnspecified:
    case kTransferBt709:
    case kTransferSmpte170m:  // BT.601's and BT.2020's SDR curve is BT.709's
    case kTransferBt2020Ten:
    case kTransferBt2020Twelve:
      encoding.transfer = ColorEncoding::Transfer::Bt709;
      break;
    case kTransferSrgb:
      encoding.transfer = ColorEncoding::Transfer::Srgb;
      break;
    case kTransferLinear:
      encoding.transfer = ColorEncoding::Transfer::Linear;
      break;
    case kTransferPq:
      encoding.transfer = ColorEncoding::Transfer::Bt2020Pq;
      break;
    default:
      return std::nullopt;
  }
  switch (primaries) {
    case kPrimariesUnspecified:
    case kPrimariesBt709:
      encoding.primaries = ColorEncoding::Primaries::Bt709;
      break;
    case kPrimariesDisplayP3:
      encoding.primaries = ColorEncoding::Primaries::DisplayP3;
      break;
    case kPrimariesBt2020:
      encoding.primaries = ColorEncoding::Primaries::Bt2020;
      break;
    default:
      return std::nullopt;
  }
  return encoding;
}

void describe_color(
    const ColorCodes& codes,
    const std::optional<VideoColorDescription>& unlabelled_color,
    DecodedPicture& picture) {
  YuvImage& yuv = picture.yuv;
  // A stream that declares no matrix reads as limited range whether or not
  // it declares one, so the range is taken with the matrix.
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  if (unlabelled_color && codes.matrix == kMatrixUnspecified) {
    matrix = unlabelled_color->matrix;
    yuv.full_range = unlabelled_color->full_range;
  } else {
    matrix = resolve_matrix(codes.matrix, static_cast<int>(yuv.height));
    yuv.full_range = codes.full_range;
  }
  const YcbcrWeights weights = ycbcr_weights(matrix);
  yuv.kr = weights.kr;
  yuv.kb = weights.kb;
  picture.encoding = resolve_encoding(codes.transfer, codes.primaries);
  switch (codes.chroma_location) {
    case 1:
      yuv.chroma_location = ChromaLocation::Center;
      break;
    case 2:
      yuv.chroma_location = ChromaLocation::TopLeft;
      break;
    case 3:
      yuv.chroma_location = ChromaLocation::Top;
      break;
    case 4:
      yuv.chroma_location = ChromaLocation::BottomLeft;
      break;
    case 5:
      yuv.chroma_location = ChromaLocation::Bottom;
      break;
    default:
      yuv.chroma_location = ChromaLocation::Left;
      break;
  }
}

}  // namespace volumetric_kit::sensor::video
