// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/utils/picture_readback.hpp
/// @brief A set's colour pictures copied from the device to the host, where
///        the decoder left them: a detector's luma, or every plane.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/export.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief One picture's planes on the host, as @ref PictureReadback::read
///        left them: views into the memory the read copied them to.
///
/// Rows run @ref stride bytes apart, as the picture's buffer had them, or
/// packed when the planes were images.
///
/// @code
/// // An OpenCV detector's input, without a copy:
/// cv::Mat luma(view.height, view.width, CV_8UC1,
///              const_cast<std::uint8_t*>(view.luma), view.stride);
/// @endcode
struct PictureView {
  std::uint32_t width = 0;   ///< Luma width (pixels).
  std::uint32_t height = 0;  ///< Luma height (rows).
  std::size_t stride = 0;    ///< Bytes from one luma row to the next.
  /// Y in 0..255 rather than 16..235: the picture's `YuvImage::full_range`.
  bool full_range = true;
  /// @ref height rows of @ref width bytes, @ref stride apart.
  const std::uint8_t* luma = nullptr;
  /// How @ref chroma is laid out: NV12 for planes that were images.
  YuvLayout layout = YuvLayout::Nv12;
  /// With `PictureReadback::Planes::All`, the chroma, `(height + 1) / 2` rows
  /// each: NV12's interleaved Cb and Cr in `chroma[0]`, `(width + 1) / 2` pairs
  /// a row, or I420's Cb and Cr, `(width + 1) / 2` bytes a row. Null otherwise.
  const std::uint8_t* chroma[2] = {};
  std::size_t chroma_stride[2] = {};  ///< Bytes between @ref chroma rows.
};

/// @brief Copies a set's colour pictures to the host in one submit, wherever
///        the decoder left them on the device.
///
/// NV12 plane images, as VideoToolbox's and NVDEC's pictures arrive, are
/// copied from their corner; planes in a buffer, as nvJPEG's arrive, are
/// copied where they sit; either after the batch takes them over from the
/// queue family named by `YuvImage::queue_family`, images in the layout they
/// are in. After a successful read, that field names this device's family
/// for an EXCLUSIVE buffer or images, or `kQueueFamilyIgnored` for CONCURRENT
/// ones; images that needed no transfer keep `kQueueFamilyIgnored`. Pass the
/// updated picture to subsequent stages, including `GpuFramePrep`; copies
/// made before the read retain the old ownership metadata. Each read checks
/// every picture before it records anything, records every copy into one
/// `CommandBatch`, submits it and waits. The copies land in one host
/// buffer the readback keeps (`MemoryUsage::Staging`, cached for reading),
/// grown to the largest read and reused, so a steady stream of sets
/// allocates nothing.
///
/// @code
/// VKC_ASSIGN(auto readback, PictureReadback::create(device, allocator));
/// // Each set, as SensorArray::poll_set hands it out:
/// VKC_ASSIGN(auto views,
///            readback->read(set.frames, PictureReadback::Planes::Luma));
/// for (const std::optional<PictureView>& view : views) {
///   if (view) detect(view->luma, view->width, view->height, view->stride);
/// }
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object; it stores references to them. Not thread-safe,
///          though readbacks on several threads may share one device.
class VR_SENSOR_UTILS_API PictureReadback {
 public:
  /// @brief Which planes @ref read copies.
  enum class Planes : std::uint8_t {
    Luma,  ///< Y alone: what a detector reads.
    All,   ///< Y and the chroma.
  };

  /// @brief A readback on @p device; its host memory is allocated at the
  ///        first read.
  /// @param device     The device the pictures are on (must outlive this).
  /// @param allocator  The allocator its host memory comes from (must
  ///                   outlive this).
  /// @return The readback; `Status::Code::InvalidArgument` for an empty
  ///         device or allocator; `Status::Code::OutOfMemory` for a host
  ///         allocation failure, including timer creation.
  static core::Result<std::unique_ptr<PictureReadback>> create(
      const core::Device& device, core::Allocator& allocator);

  ~PictureReadback() = default;
  PictureReadback(const PictureReadback&) = delete;
  PictureReadback& operator=(const PictureReadback&) = delete;
  PictureReadback(PictureReadback&&) = delete;
  PictureReadback& operator=(PictureReadback&&) = delete;

  /// @brief Copy each frame's colour picture's @p planes to the host.
  ///
  /// The views point into this object's memory, so they are valid until
  /// the next read or this object's destruction, whatever the frames do:
  /// the frames, and their pictures, may go as soon as this returns.
  /// @param[in,out] frames  One entry per camera, as a `FrameSet` holds them;
  ///                       each picture's ownership is updated on success,
  ///                       as described above.
  /// @param planes   Luma alone, or every plane.
  /// @param metrics  Optional: a `"readback"` row, the call on the host and
  ///                 the copies on the device. `nullptr` measures nothing.
  /// @return One view per frame, empty where the frame is, or has no
  ///         picture on the device (no colour, or colour as host words);
  ///         what the other overload returns for a picture it refuses; or a
  ///         buffer, submit or `Status::Code::OutOfMemory` failure.
  core::Result<std::vector<std::optional<PictureView>>> read(
      std::vector<std::optional<RgbdFrame>>& frames, Planes planes,
      core::StageMetrics* metrics = nullptr);

  /// @brief Copy one picture's @p planes to the host: a decoder's
  ///        `DecodedPicture::yuv`, say.
  /// @param[in,out] picture  On the device, its writer finished; its
  ///                        ownership is updated on success, as above.
  /// @param planes   Luma alone, or every plane.
  /// @param metrics  As the other overload.
  /// @return The view, valid as the other overload's;
  ///         `Status::Code::InvalidArgument` for a picture that is empty or
  ///         not on the device, planes both in a buffer and as images, a
  ///         buffer that is empty or without `TRANSFER_SRC` usage, a plane
  ///         stride shorter than its rows, planes that overlap or lie
  ///         outside their buffer, images that are not NV12's R8 and R8G8
  ///         planes at least the picture's size with `TRANSFER_SRC` usage,
  ///         in a layout a copy reads, a `queue_family` the device lacks, or
  ///         `EXCLUSIVE` images from another of its families; or a buffer,
  ///         submit or `Status::Code::OutOfMemory` failure.
  core::Result<PictureView> read(YuvImage& picture, Planes planes,
                                 core::StageMetrics* metrics = nullptr);

 private:
  PictureReadback(const core::Device& device, core::Allocator& allocator,
                  core::GpuTimer timer);

  // The one path both reads take: `pictures[i]` into `views[i]`, a null
  // picture skipped, for `count` pictures.
  core::Status read_pictures(YuvImage* const* pictures, std::size_t count,
                             Planes planes, std::optional<PictureView>* views,
                             core::StageMetrics* metrics);

  const core::Device& device_;
  core::Allocator& allocator_;
  core::GpuTimer timer_;
  // Where the copies land: host memory, mapped, grown to the largest read.
  core::Buffer staging_;
};

}  // namespace volumetric_kit::recon::sensor
