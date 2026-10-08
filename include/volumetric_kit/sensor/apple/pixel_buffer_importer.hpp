// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/apple/pixel_buffer_importer.hpp
/// @brief Apple's pictures handed to Vulkan where they are: a `CVPixelBuffer`'s
///        NV12 planes on its IOSurface, as images on the device (Apple only).

#include <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <memory>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/apple/export.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace volumetric_kit::recon::sensor {

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
/// arrived on in 64 is let go, and the pictures still holding it keep it.
///
/// The images hold the pixel buffer, so its producer does not reuse the
/// surface while anything reads them. Its writer must have finished, as for
/// any @ref YuvImage: the GPU pass waits on nothing.
///
/// @code
/// auto importer = PixelBufferImporter::create(device, "ARKitCapture");
/// RgbdFrame frame;
/// VKC_TRY(importer->import(captured_image, width, height, frame.color));
/// frame.color.kr = weights.kr;  // the colour description is the caller's
/// frame.color.kb = weights.kb;
/// frame.color.full_range = true;
/// frame.color.chroma_location = ChromaLocation::Left;
/// @endcode
///
/// @warning Not thread-safe: use from one thread.
class VR_SENSOR_APPLE_API PixelBufferImporter {
 public:
  /// @param device The device the images are made on. Borrowed: it must
  ///        outlive the importer and every picture imported.
  /// @param who Names the importer's user in errors.
  /// @return The importer; `Status::Code::Unsupported` where @p device imports
  ///         no Metal textures (`Device::imports_metal_textures`) or names no
  ///         Metal device.
  static core::Result<PixelBufferImporter> create(
      const core::Device& device, std::string who = "PixelBufferImporter");

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
  ///        `GENERAL`, @p width x @p height, no @ref YuvImage::device. Its
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

 private:
  struct Impl;
  explicit PixelBufferImporter(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::recon::sensor
