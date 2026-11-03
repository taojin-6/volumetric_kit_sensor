// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file prep/luma_pass.hpp
/// @brief Any picture's 8-bit luma and its half, made on the GPU and left
///        there: a detector's input, whatever layout the camera gives.

#include <cstdint>
#include <memory>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/prep/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @brief One picture's luma on the device, as @ref LumaPass::run made it:
///        at full size and halved.
///
/// Both are `YuvLayout::Luma` pictures in one device buffer of the pass's,
/// rows packed (the stride is the width), the picture's range kept
/// (`YuvImage::full_range`: a V210 picture's is limited), and
/// `YuvImage::queue_family` `kQueueFamilyIgnored`: written on the pass's
/// family, or shared CONCURRENT with the families the pass was made for. So
/// `PictureReadback::read` copies them to the host as they are, and a renderer
/// on a named family reads them in place.
///
/// @code
/// VKC_ASSIGN(std::vector<DeviceLuma> luma, pass->run({&left, &right}));
/// VKC_ASSIGN(std::vector<PictureView> host,
///            readback->read({&luma[0].full, &luma[0].half, &luma[1].full,
///                            &luma[1].half},
///                           PictureReadback::Planes::Luma));
/// @endcode
struct DeviceLuma {
  /// The picture's luma, its size: an 8-bit layout's Y codes as they were,
  /// or V210's 10-bit Y rounded once, `(code + 2) / 4`, at most 255.
  YuvImage full;
  /// @ref full halved: `(width / 2) x (height / 2)`, each pixel the mean of a
  /// 2 x 2 block of @ref full rounded half up, `(a + b + c + d + 2) / 4`, so
  /// pixel `u` here is pixel `2u + 0.5` there. An odd last column or row of
  /// @ref full is in no block.
  YuvImage half;
  /// Holds the buffer @ref full and @ref half are in, as each of their
  /// `YuvImage::device` does. The pass writes the buffer again only once the
  /// last of them is gone, so a picture kept past the next run is still
  /// itself. Drop the last only once every device read of it has completed:
  /// the next run may write it from then on.
  std::shared_ptr<const void> owner;
};

/// @brief Makes each picture's 8-bit luma and its half (@ref DeviceLuma) on
///        the GPU, every picture of a list in one batch and one submit.
///
/// It reads any layout a driver hands out, where the driver left it:
/// - planes in a buffer on the device -- a decoder's I420 or NV12, a capture
///   card's V210, a camera's YUYV or luma alone -- read where they are, through
///   the unpack `GpuFramePrep` and `StereoRectifier` share
///   (`color_planes.glsl`): an 8-bit layout's Y codes copied, V210's 10-bit
///   codes rounded once to 8 bits;
/// - planes on the host, as a camera's SDK hands them over
///   (`YuvImage::host`), copied into one persistent slot of the pass's
///   (`core::HostRing`) and read there the same way; a 4:2:0 picture's luma
///   plane alone is copied;
/// - NV12's planes as images, as VideoToolbox's and NVDEC's pictures arrive,
///   whose luma image is copied from its corner.
///
/// Device planes are taken over from the queue family that wrote them
/// (`YuvImage::queue_family`), each buffer and image once, however many
/// pictures name it, images in the layout they are in. After a successful run
/// that field names this device's family for an EXCLUSIVE buffer or images,
/// or `kQueueFamilyIgnored` for CONCURRENT ones, as `PictureReadback::read`
/// leaves it; pass the updated picture on. A first kernel writes each
/// picture's full luma, a second its half from the full one; every
/// picture's work is recorded into one `CommandBatch`, submitted once and
/// waited on; it waits on the device for each picture's `YuvImage::ready`
/// value before it reads.
///
/// The outputs come from the pass's pool of device buffers: each picture
/// takes the smallest free one that holds it, else a free one too small is
/// replaced, else one is added. A buffer is free once no @ref DeviceLuma
/// holds it, so a steady stream of lists, each dropped before the next run,
/// allocates nothing after the first. Every picture is checked before
/// anything is allocated or recorded, so a refused list leaves the pass and
/// the pictures as they were.
///
/// @code
/// VKC_ASSIGN(auto pass, LumaPass::create(device, allocator));
/// // Each stereo pair, as an IStereoSensor hands it out:
/// VKC_ASSIGN(std::vector<DeviceLuma> luma,
///            pass->run({&frame.left, &frame.right}));
/// // luma[0].full is the left picture's 8-bit luma, luma[0].half its half.
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object and every @ref DeviceLuma it hands out. Not
///          thread-safe; a @ref DeviceLuma may be dropped on any thread.
class VS_PREP_API LumaPass {
 public:
  /// @brief Build the pass's kernels on @p device; its buffers are made at
  ///        the first run.
  /// @param device              The device the pictures are on (must outlive
  ///                            this object and its outputs).
  /// @param allocator           Where the outputs and the host slot come from
  ///                            (must outlive this object and its outputs).
  /// @param queue_families      The queue families that read the outputs, as
  ///                            `BufferDesc::queue_families`: @p device's own
  ///                            and a renderer's, say, which makes them
  ///                            CONCURRENT across both. Null, the default,
  ///                            leaves them EXCLUSIVE to @p device's family.
  ///                            Copied here.
  /// @param queue_family_count  The entries in @p queue_families.
  /// @return The pass; `Status::Code::InvalidArgument` for an empty device or
  ///         allocator, a count with no families, more distinct families
  ///         than `BufferDesc::kMaxQueueFamilies`, a family @p device does
  ///         not have, or families that leave out @p device's own; a device
  ///         that lacks what the package's kernels need; or a pipeline,
  ///         timer or `Status::Code::OutOfMemory` failure.
  static core::Result<std::unique_ptr<LumaPass>> create(
      const core::Device& device, core::Allocator& allocator,
      const std::uint32_t* queue_families = nullptr,
      std::uint32_t queue_family_count = 0);

  ~LumaPass();
  LumaPass(const LumaPass&) = delete;
  LumaPass& operator=(const LumaPass&) = delete;
  LumaPass(LumaPass&&) = delete;
  LumaPass& operator=(LumaPass&&) = delete;

  /// @brief Make each picture's luma and half: one batch, one submit and its
  ///        wait.
  /// @param[in,out] pictures  The pictures, each non-null and at least 2 x 2.
  ///                          Host planes are copied here; device planes,
  ///                          their writer finished or `YuvImage::ready`
  ///                          set, are read before this returns, and each
  ///                          picture's ownership is updated on success, as
  ///                          described above.
  /// @param metrics  Optional: a `"luma"` row, the call on the host and the
  ///                 copies and kernels on the device. `nullptr` measures
  ///                 nothing.
  /// @return One @ref DeviceLuma per picture, in order (none for an empty
  ///         list); `Status::Code::InvalidArgument` for a null picture, one
  ///         under 2 x 2, one with no planes or with planes in more than one
  ///         of a buffer, the host and images, an unknown layout, YUYV or
  ///         V210 of odd width, V210 in full range or off 32-bit words, a
  ///         plane stride shorter than its rows, planes that overlap or lie
  ///         outside their buffer, a buffer that is empty or without storage
  ///         usage, images that are not NV12's R8 and R8G8 planes at least
  ///         the picture's size with `TRANSFER_SRC` usage in `GENERAL` or
  ///         `TRANSFER_SRC_OPTIMAL`, a picture past a single dispatch or a
  ///         storage binding, a `queue_family` the device lacks, `EXCLUSIVE`
  ///         images from another of its families, or pictures in one buffer
  ///         or image that name different writers' families; or a buffer,
  ///         submit or `Status::Code::OutOfMemory` failure. After a failed
  ///         submit or wait the pass refuses every run, as the device may
  ///         still run that batch.
  core::Result<std::vector<DeviceLuma>> run(
      const std::vector<YuvImage*>& pictures,
      core::StageMetrics* metrics = nullptr);

 private:
  struct Impl;
  explicit LumaPass(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
