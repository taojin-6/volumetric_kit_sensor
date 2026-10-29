// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What a decoder test needs beyond the core's GPU test harness
// (volumetric_kit/core/testing/gpu_test.hpp): whether a missing device path
// skips or fails, and a device the decoders refuse.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace vr_test {

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;

// What a decoder test returns when the decoders have no device path here (no
// GPU this build's hardware path reaches): it skips, or fails where CI sets
// VR_TEST_HEVC_BACKEND, as on every leg that promises one.
inline int no_decoder(const std::string& why) {
  const char* promise = std::getenv("VR_TEST_HEVC_BACKEND");
  const bool required = promise != nullptr && *promise != '\0';
  std::fprintf(stderr, "no device path for the decoders (%s); %s\n",
               why.c_str(), required ? "failing" : "skipping");
  return required ? 1 : vkc::test::kSkipExitCode;
}

// As no_decoder for JpegDecoder, except that a GPU with no hardware JPEG
// engine always skips: NVIDIA's are few (the RTX 5090 has one, the RTX 4090
// on one CI host none), and nothing stands in for it.
inline int no_jpeg_decoder(const std::string& why) {
  if (why.find("no hardware JPEG engine") != std::string::npos) {
    std::fprintf(stderr, "%s; skipping\n", why.c_str());
    return vkc::test::kSkipExitCode;
  }
  return no_decoder(why);
}

// `gpu`'s device again, adopted with none of its optional extensions declared,
// so a decoder handed it can keep no picture on the GPU: the device path fails
// to open on any machine, which is how a test reaches a decoder's refusal of
// it. Borrowed: `gpu`'s device must outlive the result.
inline vkc::Result<vkc::Device> bare_device(const vkc::test::GpuContext& gpu) {
  vkc::AdoptedDevice view;
  view.instance = gpu.instance.handle();
  view.instance_api_version = gpu.instance.api_version();
  view.physical_device = gpu.device.physical_device();
  view.device = gpu.device.handle();
  view.queue_family = gpu.device.queue_family();
  view.queue = gpu.device.queue();
  view.submit_mutex = gpu.device.submit_mutex();
  view.enabled_features.timeline_semaphore = true;
  view.enabled_features.scalar_block_layout = true;
  return vkc::Device::adopt(view, vr::device_requirements());
}

}  // namespace vr_test
