// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A device picture's planes and the checks every reader of one makes:
// GpuFramePrep, which converts it, and PictureReadback, which copies it to
// the host. Internal to recon_sensor_utils.

#include <cstdint>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace volumetric_kit::recon::sensor::detail {

// A picture's planes: Y, then NV12's CbCr or I420's Cb and Cr, the chroma
// at half the size, rounded up.
struct YuvPlanes {
  int count = 0;                    // 2 for NV12, 3 for I420
  std::uint64_t rows[3] = {};       // each plane's rows
  std::uint64_t row_bytes[3] = {};  // the bytes of a row the picture uses
  std::uint64_t begin[3] = {};      // planes in a buffer: the first byte
  std::uint64_t end[3] = {};        // and one past the last
};

// `image`'s planes, checked as a copy reads them: in a buffer or as images,
// not both. Images are NV12's R8 luma and R8G8 chroma, each at least the
// picture's size, with TRANSFER_SRC usage, in GENERAL or
// TRANSFER_SRC_OPTIMAL. Planes in YuvImage::device each have a stride at
// least their rows, lie inside the buffer and overlap no other; the buffer
// itself (empty, its usage) is the caller's to check first. `who` starts
// every message.
core::Result<YuvPlanes> check_yuv_planes(const YuvImage& image,
                                         const char* who);

}  // namespace volumetric_kit::recon::sensor::detail
