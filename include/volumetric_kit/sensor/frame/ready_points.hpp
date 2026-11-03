// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame/ready_points.hpp
/// @brief What a submission that reads a picture waits for: the picture's
///        ready value (`YuvImage::ready`), gathered one a timeline.
///
/// This package's readers wait for it themselves, and `GpuFramePrep`'s callers
/// have `ready_points`; this is for a reader of its own, which samples or
/// copies a picture's planes in a batch it submits.

#include <algorithm>
#include <vector>

#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @brief Add @p image's ready value to @p points, the waits of the
///        submission that reads it.
///
/// @code
/// std::vector<core::TimelinePoint> waits;
/// for (const DecodedPicture& p : pictures) add_ready_point(p.yuv, waits);
/// VKC_ASSIGN(core::PendingBatch read, batch.submit_async(waits, {}));
/// @endcode
/// @param image   A picture that may carry @ref YuvImage::ready, which must be
///                a timeline of the device the submission is on.
/// @param points  The waits, one a timeline: where they name @p image's
///                already, the higher of the two values. Unchanged for an
///                image without one.
inline void add_ready_point(const YuvImage& image,
                            std::vector<core::TimelinePoint>& points) {
  if (image.ready == nullptr) return;
  for (core::TimelinePoint& point : points) {
    if (point.semaphore == image.ready.get()) {
      point.value = std::max(point.value, image.ready_value);
      return;
    }
  }
  points.push_back({image.ready.get(), image.ready_value});
}

}  // namespace volumetric_kit::sensor
