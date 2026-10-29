// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/decoded_picture.hpp
/// @brief A decoded picture, on the device and held by the picture.

#include <cstdint>
#include <optional>

#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

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
  /// after its crop. Every HEVC picture, and VideoToolbox's JPEG, is NV12's
  /// two plane images (`YuvImage::image`): from VideoToolbox (Apple) they
  /// hold the decoder's picture, which VideoToolbox does not reuse while they
  /// are held; from NVDEC (Linux) CUDA copied the picture into them, so
  /// `YuvImage::queue_family` is @ref kQueueFamilyExternal, and the decoder
  /// reuses them only once nothing holds either. From nvJPEG (Linux) it is
  /// I420 in a buffer (`YuvImage::device`), which CUDA wrote and the decoder
  /// reuses on the same terms. Either way the decoder has finished writing
  /// it; drop it before the device it is on is destroyed.
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
  /// source declares them (`sensor/frame/color_space.hpp`); a stream that
  /// declares neither is taken as BT.709. Empty when the stream declares one
  /// @ref ColorEncoding cannot name (HLG, the SD primaries of BT.601, a
  /// gamma curve), so a driver has nothing true to declare and refuses the
  /// picture rather than fusing it through the wrong curve or basis.
  std::optional<ColorEncoding> encoding = ColorEncoding{
      ColorEncoding::Transfer::Bt709, ColorEncoding::Primaries::Bt709};
};

}  // namespace volumetric_kit::sensor
