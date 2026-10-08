// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/decoded_picture.hpp
/// @brief A decoded picture, on the device and held by the picture.

#include <cstdint>
#include <optional>

#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief The YCbCr-to-RGB matrix of a coded picture.
enum class VideoColorMatrix {
  Bt601,      ///< BT.601 (SMPTE 170M, BT.470 B/G): SD.
  Bt709,      ///< BT.709: HD.
  Bt2020,     ///< BT.2020 non-constant luminance: UHD.
  Smpte240m,  ///< SMPTE 240M, the interim HD matrix.
  Fcc,        ///< FCC 73.682, the 1953 NTSC matrix.
};

/// @brief A matrix's red and blue luma weights; green's is the rest.
struct YcbcrWeights {
  float kr = 0.0f;  ///< Red's weight in luma.
  float kb = 0.0f;  ///< Blue's weight in luma.
};

/// @return @p matrix's weights, as the standards give them: what the GPU pass
///         converts by.
constexpr YcbcrWeights ycbcr_weights(VideoColorMatrix matrix) noexcept {
  switch (matrix) {
    case VideoColorMatrix::Bt601:
      return {0.299f, 0.114f};
    case VideoColorMatrix::Bt709:
      return {0.2126f, 0.0722f};
    case VideoColorMatrix::Bt2020:
      return {0.2627f, 0.0593f};
    case VideoColorMatrix::Smpte240m:
      return {0.212f, 0.087f};
    case VideoColorMatrix::Fcc:
      return {0.30f, 0.11f};
  }
  return {0.2126f, 0.0722f};
}

/// @brief A stream's YCbCr matrix and range.
struct VideoColorDescription {
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  bool full_range = false;  ///< Y in 0..255 rather than 16..235.
};

/// @brief One decoded picture, on the device, held by the picture: a frame's
///        colour as it is (`RgbdFrame::color = picture.yuv`), with the
///        stream's timestamp and encoding beside it.
struct DecodedPicture {
  /// The picture and how it is coded. Its size is the stream's display size,
  /// after its crop. From NVDEC or nvJPEG (Linux, through CUDA) it is in a
  /// buffer (`YuvImage::device`), NV12 from @ref HevcDecoder or I420 from
  /// @ref JpegDecoder; CUDA wrote it, so `YuvImage::queue_family` is
  /// @ref kQueueFamilyExternal, and the decoder reuses the buffer only once
  /// nothing holds it. From VideoToolbox (Apple), for either decoder, it is
  /// NV12 images (`YuvImage::image`) that hold the decoder's picture, which
  /// VideoToolbox does not reuse while they are held. Either way the decoder
  /// has finished writing it; drop it before the device it is on is
  /// destroyed.
  ///
  /// The matrix and range are the ones it is coded in. A stream that declares
  /// no matrix takes the decoder's `Options::unlabelled_color` when it has
  /// one; without one, or for a matrix @ref VideoColorMatrix cannot name, it
  /// is taken as BT.709 above 576 rows and BT.601 at or below, as players
  /// do. Chroma sits as JPEG places it, centred, or as an H.265 stream
  /// declares, left when it declares nothing.
  YuvImage yuv;
  std::int64_t pts = 0;  ///< The one sent with its access unit.
  /// The transfer function and primaries the stream declares, as a capture
  /// source declares them (`core/color_space.hpp`); a stream that declares
  /// neither is taken as BT.709. Empty when the stream declares one
  /// @ref ColorEncoding cannot name (HLG, the SD primaries of BT.601, a
  /// gamma curve), so a driver has nothing true to declare and refuses the
  /// picture rather than fusing it through the wrong curve or basis.
  std::optional<ColorEncoding> encoding = ColorEncoding{
      ColorEncoding::Transfer::Bt709, ColorEncoding::Primaries::Bt709};
};

}  // namespace volumetric_kit::recon::sensor
