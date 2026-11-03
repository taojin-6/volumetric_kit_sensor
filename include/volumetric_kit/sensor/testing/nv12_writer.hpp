// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/nv12_writer.hpp
/// @brief Test support: an 8-bit NV12 picture written into a picture's two
///        plane images on the device by a compute kernel, in the caller's
///        batch, as a producer's own kernel writes an encoder's target
///        (`HevcEncoder::target`) (`volumetric_kit::sensor_video_testing`,
///        with no test framework).
///
/// The planes are storage images, `r8` and `rg8` in GLSL, so the device
/// needs `hevc_encoder_device_requirements()`; the picture's codes go in as
/// they are, whatever the target's matrix and range say they mean. Writing in
/// the batch, ordered by the timeline values it waits for and sets, is what a
/// test of the encoder's wait for its writer needs.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// VKC_ASSIGN(testing::Nv12Writer writer, testing::Nv12Writer::create(device));
/// core::CommandBatch batch(device, allocator);
/// VKC_ASSIGN(const YuvImage target, encoder.target(batch));
/// VKC_TRY(writer.record_upload(batch, allocator,
///                              testing::test_pattern(1920, 1440, i), target));
/// VKC_ASSIGN(core::PendingBatch written,
///            batch.submit_async({}, {{&timeline, n}}));
/// VKC_TRY(encoder.send(encoder.key_owed(), pts, {&timeline, n}));
/// @endcode

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/video/testing_export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief Writes NV12 pictures into plane images on the device from one
///        compute kernel, recorded into the caller's batches.
///
/// @warning Not thread-safe: use from one thread. At most @ref kPending of
///          the batches it recorded into may be unfinished at once.
class VS_VIDEO_TESTING_API Nv12Writer {
 public:
  /// Batches it recorded into that may be unfinished at once: it takes its
  /// descriptor sets in turn, one a record, and a set is rewritten
  /// @ref kPending records later.
  static constexpr std::uint32_t kPending = 8;

  /// @return The writer, its kernel built on @p device.
  ///         `Status::Code::InvalidArgument` for a moved-from @p device;
  ///         `Status::Code::Unsupported` where @p device did not enable
  ///         `hevc_encoder_device_requirements()` (the `r8` and `rg8`
  ///         storage images among them); or the kernel's failure. @p device
  ///         is borrowed: it must outlive the writer.
  static core::Result<Nv12Writer> create(const core::Device& device);

  Nv12Writer(Nv12Writer&& other) noexcept;
  Nv12Writer& operator=(Nv12Writer&& other) noexcept;
  Nv12Writer(const Nv12Writer&) = delete;
  Nv12Writer& operator=(const Nv12Writer&) = delete;
  ~Nv12Writer();

  /// @return @p picture's planes in a new device buffer for @ref record:
  ///         its luma rows, then its chroma rows, packed, as @ref HostNv12
  ///         holds them, the upload recorded into @p batch (so the buffer
  ///         holds them once the batch has run). `Status::Code::
  ///         InvalidArgument` for a picture of no size or with planes of
  ///         other lengths; or the buffer's or the upload's failure.
  static core::Result<core::Buffer> upload(core::CommandBatch& batch,
                                           core::Allocator& allocator,
                                           const HostNv12& picture);

  /// @brief Record into @p batch the kernel that stores @p planes, a
  ///        picture of @p target's size as @ref upload lays it out, into
  ///        @p target's planes.
  /// @param batch   The batch; @p planes and @p target's images must stay
  ///                alive until it completes.
  /// @param planes  The picture's codes, a storage buffer.
  /// @param target  NV12 plane images: `image[0]` R8 and `image[1]` R8G8,
  ///                each in `GENERAL` with a view and `STORAGE` usage, at
  ///                least the picture's size (chroma halved, rounded up),
  ///                written from their corner.
  /// @return OK; `Status::Code::InvalidArgument` for another target (not
  ///         NV12 images, another format, layout or usage, too small),
  ///         @p planes too small for @p target's size, or a moved-from
  ///         writer; or @p batch's refusal.
  core::Status record(core::CommandBatch& batch, const core::Buffer& planes,
                      const YuvImage& target);

  /// @brief @ref upload, then @ref record: @p picture, which must be
  ///        @p target's size, into @p target's planes, the buffer kept by
  ///        @p batch until it completes.
  /// @return As @ref upload and @ref record, and
  ///         `Status::Code::InvalidArgument` for a picture of another size.
  core::Status record_upload(core::CommandBatch& batch,
                             core::Allocator& allocator,
                             const HostNv12& picture, const YuvImage& target);

 private:
  struct Impl;
  explicit Nv12Writer(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor::testing
