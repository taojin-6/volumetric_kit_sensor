// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The calls an iOS app makes to open ARKit's camera and code its pictures,
// and pictures its kernels write on the device, linked into an iOS executable
// by tools/ios_build_check.sh, so a symbol iOS lacks, an FFmpeg or CUDA one
// among them, fails the check. Never run: it opens no camera or session
// unless given a picture.

#include <CoreVideo/CoreVideo.h>

#include <cstdint>
#include <cstdio>

#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/sensor/apple/core_video_color.hpp"
#include "volumetric_kit/sensor/apple/pixel_buffer_importer.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"
#include "volumetric_kit/sensor/testing/hevc_conformance.hpp"
#include "volumetric_kit/sensor/testing/nv12_writer.hpp"
#include "volumetric_kit/sensor/video/backend.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"

namespace core = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;

int main(int argc, char** argv) {
  if (sensor::video_backend() != sensor::VideoBackend::VideoToolbox) return 1;
  // A picture only where one is handed in, which no run of this does.
  auto* picture =
      argc > 99 ? reinterpret_cast<CVPixelBufferRef>(argv[0]) : nullptr;
  if (picture == nullptr) return 0;
  // ARKit's camera, as the scanner opens it (the device is the app's).
  sensor::ArkitSensor::Options camera_options;
  camera_options.device = reinterpret_cast<const core::Device*>(argv[1]);
  auto camera = sensor::ArkitSensor::open(camera_options);
  if (camera.ok() && camera->start().ok()) {
    auto frame = camera->poll();
    if (frame.ok() && frame.value()) {
      picture = sensor::pixel_buffer_of(frame.value()->color);
    }
    auto withheld = camera->poll_withheld_picture();
    if (withheld.ok() && withheld.value()) {
      picture = sensor::pixel_buffer_of(*withheld.value());
    }
    std::printf("%s\n", sensor::to_string(camera->status().tracking));
  }
  const auto tags = sensor::color_codes_of(picture);
  if (!tags) return 1;
  sensor::HevcEncoder::Options o;
  o.width = static_cast<std::uint32_t>(CVPixelBufferGetWidth(picture));
  o.height = static_cast<std::uint32_t>(CVPixelBufferGetHeight(picture));
  o.color = *tags;
  auto encoder = sensor::HevcEncoder::create(o);
  if (!encoder) return 1;
  if (!sensor::send_pixel_buffer(*encoder, picture, true, 0).ok()) return 1;
  sensor::HevcEncoder::CodedPicture coded;
  const auto got = encoder->receive(coded);
  sensor::testing::HevcEncoderCase c;
  c.options = o;
  c.pixel_buffers = true;
  const auto breach = sensor::testing::hevc_encoder_breach(c);
  std::printf("%zu %d\n", coded.access_unit.size(),
              breach.ok() && breach->empty());
  // A target its kernel writes, where a device is handed in too.
  if (argc > 100) {
    const auto* device = reinterpret_cast<const core::Device*>(argv[1]);
    auto* allocator = reinterpret_cast<core::Allocator*>(argv[2]);
    o.device = device;
    o.allocator = allocator;
    auto written = sensor::HevcEncoder::create(o);
    auto writer = sensor::testing::Nv12Writer::create(*device);
    if (!written || !writer) return 1;
    core::CommandBatch batch(*device, *allocator);
    const auto target = written->target(batch);
    if (!target.ok() ||
        !writer
             ->record_upload(
                 batch, *allocator,
                 sensor::testing::test_pattern(o.width, o.height, 0), *target)
             .ok() ||
        !batch.submit().ok() || !written->send(true, 0, {}).ok()) {
      return 1;
    }
  }
  return got.ok() ? 0 : 1;
}
