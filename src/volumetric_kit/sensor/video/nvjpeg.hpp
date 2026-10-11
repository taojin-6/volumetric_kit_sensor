// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The one place this target includes nvJPEG (VR_WITH_CUDA): its C header and
// its status as a Status. Internal.

#include <nvjpeg.h>

#include <cstdint>
#include <string>

#include "volumetric_kit/core/base/result.hpp"

namespace volumetric_kit::recon::sensor::video {

// An NvJpeg Backend error naming @p who, @p what it was doing and nvJPEG's
// @p status, which is not NVJPEG_STATUS_SUCCESS, kept as its detail.
inline core::Status nvjpeg_error(const char* who, nvjpegStatus_t status,
                                 const std::string& what) {
  return core::Status::backend_error(
      core::Status::Backend::NvJpeg, static_cast<std::int64_t>(status),
      std::string(who) + ": " + what + ": nvJPEG status " +
          std::to_string(static_cast<int>(status)));
}

}  // namespace volumetric_kit::recon::sensor::video
