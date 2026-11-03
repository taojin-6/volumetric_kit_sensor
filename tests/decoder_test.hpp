// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// What a decoder test needs beyond the core's GPU test harness
// (volumetric_kit/core/testing/gpu_test.hpp): whether a missing device path
// skips or fails, and a device the decoders refuse.

#include <cstdint>
#include <cstdio>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/testing/hevc_clip.hpp"

namespace vs_test {

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

// What a decoder test returns when the decoders have no device path here (no
// GPU this build's hardware path reaches): it skips, or fails where the run
// promises one (testing/hevc_clip.hpp's hevc_decoder_promised), as CI's
// VS_TEST_HEVC_BACKEND does on every leg that has one.
inline int no_decoder(const std::string& why) {
  const bool required = sensor::testing::hevc_decoder_promised();
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

// `gpu`'s device again, adopted with only the `count` extensions at
// `extensions` declared among those it enabled. Borrowed: `gpu`'s device must
// outlive the result.
inline vkc::Result<vkc::Device> adopted_device(const vkc::test::GpuContext& gpu,
                                               const char* const* extensions,
                                               std::uint32_t count) {
  vkc::AdoptedDevice view;
  view.instance = gpu.instance.handle();
  view.instance_api_version = gpu.instance.api_version();
  view.physical_device = gpu.device.physical_device();
  view.device = gpu.device.handle();
  view.queue_family = gpu.device.queue_family();
  view.queue = gpu.device.queue();
  view.submit_mutex = gpu.device.submit_mutex();
  view.enabled_extensions = extensions;
  view.enabled_extension_count = count;
  view.enabled_features.timeline_semaphore = true;
  view.enabled_features.scalar_block_layout = true;
  return vkc::Device::adopt(view, sensor::device_requirements());
}

// `gpu`'s device again, adopted with none of its optional extensions declared,
// so a decoder handed it can keep no picture on the GPU: the device path fails
// to open on any machine, which is how a test reaches a decoder's refusal of
// it. Borrowed: `gpu`'s device must outlive the result.
inline vkc::Result<vkc::Device> bare_device(const vkc::test::GpuContext& gpu) {
  return adopted_device(gpu, nullptr, 0);
}

// `gpu`'s device again, declared to export memory and not semaphores, so
// NVIDIA's decoders, which hand a picture over with a timeline value CUDA sets
// (YuvImage::ready), refuse it: only where `gpu`'s device exports memory.
// Borrowed as bare_device's.
inline vkc::Result<vkc::Device> memory_only_device(
    const vkc::test::GpuContext& gpu) {
  static const char* const kMemory[] = {
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
  return adopted_device(gpu, kMemory, 1);
}

}  // namespace vs_test
