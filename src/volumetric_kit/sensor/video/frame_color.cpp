// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "frame_color.hpp"

namespace volumetric_kit::recon::sensor::video {
VideoColorMatrix resolve_matrix(AVColorSpace space, int height) noexcept {
  switch (space) {
    case AVCOL_SPC_BT709:
      return VideoColorMatrix::Bt709;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      return VideoColorMatrix::Bt601;
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
      return VideoColorMatrix::Bt2020;
    case AVCOL_SPC_SMPTE240M:
      return VideoColorMatrix::Smpte240m;
    case AVCOL_SPC_FCC:
      return VideoColorMatrix::Fcc;
    default:
      break;
  }
  return height > 576 ? VideoColorMatrix::Bt709 : VideoColorMatrix::Bt601;
}

std::optional<ColorEncoding> resolve_encoding(
    AVColorTransferCharacteristic transfer,
    AVColorPrimaries primaries) noexcept {
  ColorEncoding encoding;
  switch (transfer) {
    case AVCOL_TRC_UNSPECIFIED:
    case AVCOL_TRC_BT709:
    case AVCOL_TRC_SMPTE170M:  // BT.601's and BT.2020's SDR curve is BT.709's
    case AVCOL_TRC_BT2020_10:
    case AVCOL_TRC_BT2020_12:
      encoding.transfer = ColorEncoding::Transfer::Bt709;
      break;
    case AVCOL_TRC_IEC61966_2_1:
      encoding.transfer = ColorEncoding::Transfer::Srgb;
      break;
    case AVCOL_TRC_LINEAR:
      encoding.transfer = ColorEncoding::Transfer::Linear;
      break;
    case AVCOL_TRC_SMPTE2084:
      encoding.transfer = ColorEncoding::Transfer::Bt2020Pq;
      break;
    default:
      return std::nullopt;
  }
  switch (primaries) {
    case AVCOL_PRI_UNSPECIFIED:
    case AVCOL_PRI_BT709:
      encoding.primaries = ColorEncoding::Primaries::Bt709;
      break;
    case AVCOL_PRI_SMPTE432:
      encoding.primaries = ColorEncoding::Primaries::DisplayP3;
      break;
    case AVCOL_PRI_BT2020:
      encoding.primaries = ColorEncoding::Primaries::Bt2020;
      break;
    default:
      return std::nullopt;
  }
  return encoding;
}

void describe_color(
    const AVFrame& frame,
    const std::optional<VideoColorDescription>& unlabelled_color,
    DecodedPicture& picture) {
  // A stream that declares no matrix reads as limited range whether or not
  // it declares one, so the range is taken with the matrix.
  if (unlabelled_color && frame.colorspace == AVCOL_SPC_UNSPECIFIED) {
    picture.matrix = unlabelled_color->matrix;
    picture.full_range = unlabelled_color->full_range;
  } else {
    // The displayed height: a hardware frame's still counts its top crop.
    picture.matrix =
        resolve_matrix(frame.colorspace, static_cast<int>(picture.height));
    picture.full_range = frame.color_range == AVCOL_RANGE_JPEG;
  }
  picture.encoding = resolve_encoding(frame.color_trc, frame.color_primaries);
  switch (frame.chroma_location) {
    case AVCHROMA_LOC_CENTER:
      picture.chroma_location = ChromaLocation::Center;
      break;
    case AVCHROMA_LOC_TOPLEFT:
      picture.chroma_location = ChromaLocation::TopLeft;
      break;
    case AVCHROMA_LOC_TOP:
      picture.chroma_location = ChromaLocation::Top;
      break;
    case AVCHROMA_LOC_BOTTOMLEFT:
      picture.chroma_location = ChromaLocation::BottomLeft;
      break;
    case AVCHROMA_LOC_BOTTOM:
      picture.chroma_location = ChromaLocation::Bottom;
      break;
    default:
      picture.chroma_location = ChromaLocation::Left;
      break;
  }
}

}  // namespace volumetric_kit::recon::sensor::video
