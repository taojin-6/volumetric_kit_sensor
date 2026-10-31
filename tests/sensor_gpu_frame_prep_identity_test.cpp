// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// GpuFramePrep's words for one picture in every layout its colour pass reads
// (prep_layouts.hpp), through a lens, as MD5s of each output: a dump to hold
// one build's shaders against another's byte for byte, on the same device, as
// a change to the colour pass that must leave today's words alone is compared.
// The test itself checks that each layout's words come back the same from a
// second prepare, most of them covered. Skips where no device is present.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "md5.hpp"
#include "prep_layouts.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;

namespace {

constexpr std::uint32_t kWidth = 320;
constexpr std::uint32_t kHeight = 240;

// A barrel lens, so pixels sample between the picture's.
camera::CameraModel lensed() {
  camera::CameraModel c;
  c.size = {kWidth, kHeight};
  c.intrinsics = {260.0f, 258.0f, 161.3f, 118.7f};
  c.distortion = {-0.28f, 0.09f, 0.0012f, -0.0009f, -0.01f, 0.05f, 0.0f, 0.0f};
  return c;
}

template <typename T>
std::string md5_of(const std::vector<T>& v) {
  vs_test::Md5 md5;
  md5.update(reinterpret_cast<const std::uint8_t*>(v.data()),
             v.size() * sizeof(T));
  return md5.hex();
}

int gpu_main(vkc::test::GpuContext& gpu) {
  auto picture =
      prep_layouts::make_picture(gpu.device, gpu.allocator, kWidth, kHeight);
  CHECK(picture->layouts.size() == 7);
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  CHECK(prep.ok());
  const camera::CameraModel cam = lensed();
  std::vector<std::uint16_t> raw(std::size_t{kWidth} * kHeight);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::uint16_t>(1200 + (i * 2654435761u) % 900u);
  }
  sensor::RgbdFrame f;
  f.depth = raw.data();
  f.metres_per_unit = 0.001f;
  f.depth_camera = cam;
  f.color_camera = cam;
  f.min_depth = 0.1f;
  f.max_depth = 10.0f;
  for (const prep_layouts::Layout& layout : picture->layouts) {
    f.color = layout.image;
    f.color_packed = layout.packed;
    std::string hashes[2];
    int inside = 0;
    for (std::string& hash : hashes) {
      auto out = prep->prepare(f);
      if (!out) std::fprintf(stderr, "%s\n", out.status().message().c_str());
      CHECK(out.ok());
      auto color = vkc::test::read_back<std::uint32_t>(
          gpu.device, gpu.allocator, *out->color, raw.size());
      auto depth = vkc::test::read_back<float>(gpu.device, gpu.allocator,
                                               *out->depth, raw.size());
      CHECK(color.ok() && depth.ok());
      inside = 0;
      for (const std::uint32_t word : color.value()) {
        inside += (word >> 24) == 0xFFu ? 1 : 0;
      }
      hash = md5_of(color.value()) + " depth " + md5_of(depth.value());
    }
    std::printf("  %-12s colour %s (%d covered)\n", layout.name.c_str(),
                hashes[0].c_str(), inside);
    CHECK(hashes[0] == hashes[1]);
    CHECK(inside > static_cast<int>(raw.size() / 2));
  }
  std::puts("sensor_gpu_frame_prep_identity: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
