// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Status a failed CUDA driver or nvJPEG call becomes (cuda_error,
// nvjpeg_error): a backend failure that names its library and keeps its code,
// with the driver's name for a CUDA code. nvjpeg_error needs only nvjpeg.h;
// cuda_error loads libcuda for the name, and opens no device, so its check
// skips where libcuda does not load.

#include <cstdint>
#include <cstdio>
#include <string>

#include "cuda_driver.hpp"
#include "nvjpeg.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"

namespace vkc = volumetric_kit::core;
namespace video = volumetric_kit::recon::sensor::video;

int main() {
  const vkc::Status jpeg = video::nvjpeg_error(
      "JpegDecoder", NVJPEG_STATUS_EXECUTION_FAILED, "decoding");
  CHECK(jpeg.domain() == vkc::Status::Code::Backend);
  CHECK(jpeg.backend() == vkc::Status::Backend::NvJpeg);
  CHECK(jpeg.detail() ==
        static_cast<std::int64_t>(NVJPEG_STATUS_EXECUTION_FAILED));
  CHECK(jpeg.message() ==
        "JpegDecoder: decoding: nvJPEG status " +
            std::to_string(static_cast<int>(NVJPEG_STATUS_EXECUTION_FAILED)));

  if (video::cuda_driver() == nullptr) {
    std::printf("libcuda does not load; the CUDA check skipped\n");
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
