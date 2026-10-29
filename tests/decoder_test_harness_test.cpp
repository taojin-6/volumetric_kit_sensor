// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The decoder tests' skips (decoder_test.hpp): a missing device path skips,
// or fails where VR_TEST_HEVC_BACKEND promises one, and a GPU with no hardware
// JPEG engine always skips. The GPU harness itself is the core's, tested
// there.

#include <cstdio>

#include "decoder_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"

namespace test = volumetric_kit::core::test;

namespace {

int decoder_skips() {
  {
    const test::ScopedEnv none("VR_TEST_HEVC_BACKEND", nullptr);
    CHECK(vr_test::no_decoder("none here") == test::kSkipExitCode);
    CHECK(vr_test::no_jpeg_decoder("none here") == test::kSkipExitCode);
  }
  const test::ScopedEnv promised("VR_TEST_HEVC_BACKEND", "cuda");
  CHECK(vr_test::no_decoder("none here") == 1);
  CHECK(vr_test::no_jpeg_decoder("none here") == 1);
  CHECK(vr_test::no_jpeg_decoder("no hardware JPEG engine") ==
        test::kSkipExitCode);
  return 0;
}

}  // namespace

int main() {
  if (decoder_skips() != 0) return 1;
  std::puts("decoder_test_harness: OK");
  return 0;
}
