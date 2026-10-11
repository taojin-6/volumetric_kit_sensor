// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Status a failed CUDA driver call becomes (cuda_error): a backend failure
// that names CUDA and keeps its code, with the driver's name for it. Loads
// libcuda for that name, and opens no device; skips where libcuda does not
// load.

#include <cstdint>
#include <cstdio>
#include <string>

#include "cuda_pictures.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"

namespace vkc = volumetric_kit::core;
namespace video = volumetric_kit::recon::sensor::video;

int main() {
  if (video::cuda_driver() == nullptr) {
    std::printf("libcuda does not load; skipped\n");
    return vkc::test::kSkipExitCode;
  }
  const vkc::Status s =
      video::cuda_error("JpegDecoder", CUDA_ERROR_INVALID_VALUE, "decoding");
  CHECK(s.domain() == vkc::Status::Code::Backend);
  CHECK(s.backend() == vkc::Status::Backend::Cuda);
  CHECK(s.detail() == static_cast<std::int64_t>(CUDA_ERROR_INVALID_VALUE));
  CHECK(s.message() == "JpegDecoder: decoding: CUDA_ERROR_INVALID_VALUE");
  std::printf("CUDA error tests passed\n");
  return 0;
}
