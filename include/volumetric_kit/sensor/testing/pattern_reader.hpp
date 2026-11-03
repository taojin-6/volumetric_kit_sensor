// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/pattern_reader.hpp
/// @brief The index a test picture (`test_pattern`, testing/nv12_pattern.hpp)
///        or test depth (`pattern_depth`, testing/rgbd_pattern.hpp) carries,
///        read on the GPU where the picture or depth is: a small kernel over
///        the index band and 4 bytes a read back, for the tools and tests
///        that check each frame carries its own picture.
///
/// The band is testing/index_band.hpp's. A read takes each square's central
/// half, its middle 8 x 8:
/// - **A picture's luma:** the mean, which survives lossy coding and a change
///   of range. Square 0's mean must be at least 64 above square 1's, the
///   references. A square three quarters of the way from square 1's mean to
///   square 0's or more is set, one a quarter of the way or less is clear,
///   and one between is no index. So a picture blended from two frames reads
///   as none rather than as a wrong frame; a clean or lightly coded one names
///   what `pattern_index` names.
/// - **Depth**, in `pattern_depth`'s millimetres: every sample holds one
///   value, as `depth_index` reads it. The references are
///   @ref kPatternNearMm and @ref kPatternFarMm, each bit square one of the
///   two; anything else is none.
///
/// Built in `sensor_prep`: its user links `volumetric_kit::sensor_prep`.

#include <cstdint>
#include <memory>
#include <optional>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/prep/export.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/testing/index_band.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief The index a frame's picture and its depth carry, each
///        `index % 2^kIndexBits`, as @ref PatternReader::read found them.
struct FrameIndex {
  /// The colour picture's; empty for a frame without a picture (no colour,
  /// or a dataset's packed words), a picture under 256 x 16 or a band that
  /// names no index.
  std::optional<std::uint32_t> picture;
  /// The depth's; empty for a frame without depth, depth under 256 x 16 or a
  /// band that names no index.
  std::optional<std::uint32_t> depth;
};

/// @brief Reads the index band of pictures and depth on the GPU.
///
/// It reads them where the decoder or driver left them:
/// - NV12 plane images, as VideoToolbox's and NVDEC's pictures arrive: the
///   luma image's corner copied on the device;
/// - I420, NV12 or luma-alone planes in a device buffer, as nvJPEG's arrive,
///   and device depth (`RgbdFrame::depth_device`): the band's 16 rows copied
///   on the device;
/// - planes and depth on the host: the band's 16 rows uploaded, 4 or 8 KiB.
///
/// One kernel then decides each band, and only its result words come back.
/// Device planes and depth are taken over from the queue family they name,
/// each buffer and image once, images in the layout they are in. After a
/// successful read that family is this device's for an EXCLUSIVE buffer or
/// images, or `kQueueFamilyIgnored` for CONCURRENT ones, as
/// `PictureReadback::read` leaves it: pass the updated picture on. A read is
/// one batch, one submit and its wait, which waits on the device for the
/// picture's `YuvImage::ready` value. Everything is checked before anything
/// is recorded, so a refused read leaves the reader and its inputs as they
/// were.
///
/// @code
/// VKC_ASSIGN(auto reader, testing::PatternReader::create(device, allocator));
/// // Each frame a source hands out:
/// VKC_ASSIGN(const testing::FrameIndex index, reader->read(*frame));
/// if (!index.depth || index.picture != index.depth) ++wrong;
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object; it stores references to them. Not thread-safe.
class VS_PREP_API PatternReader {
 public:
  /// @brief Build the reader's kernel and its band and result buffers on
  ///        @p device.
  /// @param device     The device the pictures and depth are on (must
  ///                   outlive this).
  /// @param allocator  Where the buffers come from (must outlive this).
  /// @return The reader; `Status::Code::InvalidArgument` for an empty device
  ///         or allocator; a device that lacks what the package's kernels
  ///         need; or a pipeline, buffer or `Status::Code::OutOfMemory`
  ///         failure.
  static core::Result<std::unique_ptr<PatternReader>> create(
      const core::Device& device, core::Allocator& allocator);

  ~PatternReader();
  PatternReader(const PatternReader&) = delete;
  PatternReader& operator=(const PatternReader&) = delete;
  PatternReader(PatternReader&&) = delete;
  PatternReader& operator=(PatternReader&&) = delete;

  /// @brief The index @p picture's luma carries.
  /// @param[in,out] picture  NV12, I420 or luma alone, on the device, its
  ///                         writer finished or `YuvImage::ready` set, or
  ///                         on the host; its ownership is updated on
  ///                         success, as described above.
  /// @return `index % 2^kIndexBits`; empty for a picture under 256 x 16,
  ///         which nothing reads, or a band that names no index;
  ///         `Status::Code::InvalidArgument` for a picture of no size or with
  ///         no planes, YUYV or V210, planes `PictureReadback::read` refuses
  ///         (in more than one place, overlapping, outside their buffer,
  ///         images that are not NV12's or not in a layout a copy reads), a
  ///         buffer that is empty or without `TRANSFER_SRC` usage, a ready
  ///         value of another device, a queue family the device lacks, or
  ///         `EXCLUSIVE` images from another of its families; or a submit or
  ///         `Status::Code::OutOfMemory` failure. After a failed submit or
  ///         wait every read is refused, as the device may still run that
  ///         batch.
  core::Result<std::optional<std::uint32_t>> read(YuvImage& picture);

  /// @brief The index @p frame's colour picture and its depth carry, both
  ///        read in one batch.
  /// @param[in,out] frame  Depth on the host (`RgbdFrame::depth`) or the
  ///                       device (`RgbdFrame::depth_device`, its writer
  ///                       finished), `depth_camera.size` samples, rows
  ///                       packed; colour as the other overload takes it, or
  ///                       none. Its picture's and its device depth's
  ///                       ownership is updated on success.
  /// @return The two indices; the other overload's refusals for the picture;
  ///         `Status::Code::InvalidArgument` for depth both on the host and on
  ///         the device, or device depth at an odd offset, past its buffer, or
  ///         in one that is empty or without `TRANSFER_SRC` usage, or in a
  ///         buffer the picture names with another queue family; or as the
  ///         other overload.
  core::Result<FrameIndex> read(RgbdFrame& frame);

 private:
  struct Impl;
  explicit PatternReader(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor::testing
