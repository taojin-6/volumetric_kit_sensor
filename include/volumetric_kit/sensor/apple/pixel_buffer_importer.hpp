// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file apple/pixel_buffer_importer.hpp
/// @brief Apple's pictures handed to Vulkan where they are: a `CVPixelBuffer`'s
///        NV12 planes on its IOSurface, as images on the device (Apple only).

#include <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <memory>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/apple/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @brief Imports 8-bit NV12 `CVPixelBuffer`s on an IOSurface -- a
///        VideoToolbox decoder's pictures, or a camera's such as ARKit's
///        `capturedImage` -- as the two plane images of a @ref YuvImage,
///        without a copy.
///
/// Each plane of the surface becomes a Metal texture, imported into Vulkan as
/// an image (`VK_EXT_metal_objects`), so the picture never crosses to the
/// host. A whole NV12 surface does not import as one image: MoltenVK refuses
/// its bytes per element. A surface's images are made the first time a
/// picture arrives on it and kept while pictures keep arriving, as a decoder
/// or a camera cycles its pictures through a few surfaces; one no picture has
/// arrived on in 64 is let go, and the pictures still holding it keep it. A
/// surface is known by its `IOSurfaceGetID`: one from another process comes
/// under a new `IOSurfaceRef` each time. Making its images waits on the
/// device's queue, behind whatever is queued there; a known surface's import
/// is a lookup. @ref stats says which a producer's pictures take.
///
/// The images hold the pixel buffer, so its producer does not reuse the
/// surface while anything reads them. Its writer must have finished: the
/// images carry no `YuvImage::ready` for a reader to wait on.
///
/// An importer made for @ref Access::Write hands out images a compute shader
/// writes in place as well, such as an encoder's pool pictures, which
/// `HevcEncoder::target` hands out this way: the surface the encoder then
/// reads holds what the shader stored, with no copy.
///
/// The images are shared across the queue families @ref create is given, as
/// when a renderer on another family draws the pictures (MoltenVK hands a
/// compute consumer and a renderer different families).
///
/// @code
/// auto importer = PixelBufferImporter::create(device, "ARKitCapture");
/// RgbdFrame frame;
/// VKC_TRY(importer->import(captured_image, width, height, frame.color));
/// // the colour description is the caller's
/// const YcbcrWeights weights = ycbcr_weights(VideoColorMatrix::Bt601);
/// frame.color.kr = weights.kr;
/// frame.color.kb = weights.kb;
/// frame.color.full_range = true;
/// frame.color.chroma_location = ChromaLocation::Left;
/// @endcode
///
/// @warning Not thread-safe: use from one thread.
class VS_APPLE_API PixelBufferImporter {
 public:
  /// @brief What the imported images are for.
  enum class Access : std::uint8_t {
    /// `SAMPLED` and `TRANSFER_SRC` usage: a decoder's or a camera's
    /// pictures, which the device reads.
    Read,
    /// Also `STORAGE` and `TRANSFER_DST`, the Metal textures written as well
    /// as read: an encoder's pictures, which the device writes in place.
    Write,
  };

  /// @param device The device the images are made on. Borrowed: it must
  ///        outlive the importer and every picture imported.
  /// @param who Names the importer's user in errors.
  /// @param queue_families The queue families that will read the images, as
  ///        `BufferDesc::queue_families`: null for @p device's own alone.
  ///        Two or more distinct give `VK_SHARING_MODE_CONCURRENT` images,
  ///        which those families read with no ownership transfer; otherwise
  ///        they are EXCLUSIVE. Duplicates count once, so a caller may pass
  ///        its compute and render families unconditionally. When given, they
  ///        include @p device's own family, on which the importer readies the
  ///        images. Read during the call only.
  /// @param queue_family_count The entries in @p queue_families.
  /// @param access What the images are for: @ref Access::Read, or
  ///        @ref Access::Write for images the device writes too.
  /// @return The importer; `Status::Code::InvalidArgument` for a count with
  ///         no families, more than `BufferDesc::kMaxQueueFamilies` distinct,
  ///         one @p device does not have, or none that is @p device's own;
  ///         or `Status::Code::Unsupported` where @p device imports no Metal
  ///         textures (`Device::imports_metal_textures`) or names no Metal
  ///         device, or, for @ref Access::Write, has no R8 or R8G8 storage
  ///         images (`VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT`).
  static core::Result<PixelBufferImporter> create(
      const core::Device& device, std::string who = "PixelBufferImporter",
      const std::uint32_t* queue_families = nullptr,
      std::uint32_t queue_family_count = 0, Access access = Access::Read);

  PixelBufferImporter(PixelBufferImporter&& other) noexcept;
  PixelBufferImporter& operator=(PixelBufferImporter&& other) noexcept;
  ~PixelBufferImporter();
  PixelBufferImporter(const PixelBufferImporter&) = delete;
  PixelBufferImporter& operator=(const PixelBufferImporter&) = delete;

  /// @brief Hand @p pixels' planes to Vulkan as @p out's.
  /// @param pixels 8-bit NV12, video or full range, on an IOSurface. Held by
  ///        the images until the last of them is dropped.
  /// @param width The picture's width: it sits at the buffer's corner.
  /// @param height The picture's height.
  /// @param out Given its planes: NV12, `image[0]` and `image[1]` in
  ///        `GENERAL` with views and `SAMPLED` and `TRANSFER_SRC` usage, so
  ///        `GpuFramePrep` samples them and `PictureReadback` copies them,
  ///        and for @ref Access::Write `STORAGE` and `TRANSFER_DST` too, so
  ///        a compute shader stores into them (as `r8` and `rg8` images) and
  ///        a copy writes them;
  ///        @p width x @p height, no @ref YuvImage::device. Its
  ///        matrix, range and chroma siting are left as they are, for the
  ///        caller to set: the pixel buffer's format need not say how its
  ///        samples are coded.
  /// @return OK; `Status::Code::Unsupported` for a picture this does not take:
  ///         not 8-bit NV12, not on an IOSurface, or smaller than
  ///         @p width x @p height; `Status::Code::OutOfMemory` or
  ///         `Status::Code::Backend` if the import fails; or
  ///         `Status::Code::InvalidArgument` for no @p pixels or a moved-from
  ///         importer. @p out is left as it was unless this is OK.
  core::Status import(CVPixelBufferRef pixels, std::uint32_t width,
                      std::uint32_t height, YuvImage& out);

  /// @brief What an importer has done since @ref create, for a read-out.
  struct Stats {
    /// Pictures that reached their surface: every @ref import but the
    /// refusals of a picture it does not take, a failed one counted.
    std::uint64_t pictures = 0;
    /// Surfaces whose images were made, one let go and back counting again. A
    /// producer's pool keeps this near its size; a count that keeps pace with
    /// @ref pictures means images made for every picture.
    std::uint64_t surfaces = 0;
  };

  /// @return The counts; zero on a moved-from importer.
  Stats stats() const noexcept;

 private:
  struct Impl;
  explicit PixelBufferImporter(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

/// @brief The pixel buffer whose planes @p picture's images are, when a
///        @ref PixelBufferImporter made them: a camera's picture (ARKit's
///        `capturedImage`, through `ArkitSensor`) or a VideoToolbox decoder's,
///        to hand on as it is, such as to `send_pixel_buffer`
///        (video/hevc_encoder_apple.hpp), with no copy.
///
/// Found through what the images hold, so it answers for any picture and
/// takes no cast: a picture whose images another producer made, or whose
/// planes are on the host or in a buffer, has none.
///
/// @code
/// VKC_ASSIGN(std::optional<RgbdFrame> frame, camera.poll());  // ArkitSensor
/// if (frame) {
///   VKC_TRY(send_pixel_buffer(encoder, pixel_buffer_of(frame->color),
///                             encoder.key_owed(), pts));
/// }
/// @endcode
/// @param picture  Any picture.
/// @return Its pixel buffer, not retained: valid while @p picture's images
///         are held. Null when @ref PixelBufferImporter::import did not make
///         them.
VS_APPLE_API CVPixelBufferRef pixel_buffer_of(const YuvImage& picture) noexcept;

}  // namespace volumetric_kit::sensor
