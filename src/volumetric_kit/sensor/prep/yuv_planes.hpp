// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A device picture's planes and the checks every reader of one makes:
// GpuFramePrep and the stereo rectifier, which convert it (color_pass.hpp),
// and PictureReadback, which copies it to the host; and that the timeline
// value each one's submission waits for is one it can. Internal to
// sensor_prep.

#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/prep/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::detail {

// A picture's planes: Y, then NV12's CbCr or I420's Cb and Cr, the chroma
// at half the size, rounded up; or YUYV's one plane, two bytes a pixel; or
// V210's, 16 bytes each six pixels or part of six; or Luma's, one.
struct YuvPlanes {
  int count = 0;  // 2 for NV12, 3 for I420, 1 for YUYV, V210 and Luma
  std::uint64_t rows[3] = {};       // each plane's rows
  std::uint64_t row_bytes[3] = {};  // the bytes of a row the picture uses
  std::uint64_t begin[3] = {};      // planes in a buffer or on the host:
  std::uint64_t end[3] = {};        // the first byte and one past the last
};

// How a reader takes plane images: GpuFramePrep samples them where they are,
// PictureReadback copies them.
enum class ImageRead { Sample, Copy };

// `image`'s planes, checked as `read` takes them: in a buffer, on the host or
// as images, one of them, in a layout this package knows; YUYV's and V210's
// width is even, and V210's offset and stride are whole 32-bit words.
// Images are NV12's R8 luma and R8G8 chroma, each at least the
// picture's size: to sample, 2D, of one layer and one sample, with SAMPLED
// usage and a view, in GENERAL or SHADER_READ_ONLY_OPTIMAL; to copy, with
// TRANSFER_SRC usage, in GENERAL or TRANSFER_SRC_OPTIMAL. Planes in
// YuvImage::device or from YuvImage::host each have a stride at least their
// rows and overlap no other; in a buffer they lie inside it, on the host
// their bytes are counted in 64 bits. The buffer itself (empty, its usage) is
// the caller's to check first. `who` starts every message.
core::Result<YuvPlanes> check_yuv_planes(const YuvImage& image, const char* who,
                                         ImageRead read);

// InvalidArgument where `image`'s YuvImage::ready is an empty semaphore or one
// made on another VkDevice than `device`, which a reader's submission could
// not wait for; OK without one. `who` starts the message. Exported: the stereo
// rectifier checks its views with it too.
VS_PREP_API core::Status check_ready(const YuvImage& image,
                                     const core::Device& device,
                                     const char* who);

// What holds `image` on the device -- its buffer or plane images, and the
// timeline its readers wait on -- kept with `batch`'s work (the core's
// CommandBatch::retain): past a failed wait until the device has waited for
// it, so the picture's holder may drop it, and a decoder reuse it, as soon as
// the reader returns. Host planes are copied as the batch is recorded and need
// nothing. Exported, as check_ready.
VS_PREP_API void retain_picture(core::CommandBatch& batch,
                                const YuvImage& image) noexcept;

}  // namespace volumetric_kit::sensor::detail
