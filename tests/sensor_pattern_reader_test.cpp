// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// PatternReader (testing/pattern_reader.hpp, sensor_prep) on the family's test
// pattern and test depth: across the band's span, and past it, the index read
// on the GPU equals pattern_index's and depth_index's on the host, for NV12
// plane images (padded or not), NV12, I420 and luma-alone planes in a device
// buffer at odd offsets and strides and on the host, in full and limited
// range, at 256 x 16, 300 x 21 and 1920 x 1440; depth on the host and in a
// device buffer at an even lead. A frame's picture and depth in one read, one
// buffer holding both. None for a band that does not fit, a blend of two
// frames (where pattern_index's midpoint names a wrong one), references under
// 64 apart, a square past either margin, torn depth and swapped references;
// the index still read through noise and blur. A picture's ready value waited
// for on the device; ownership from outside Vulkan published; every refusal,
// each leaving the inputs as they were. Skips without a device.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/testing/nv12_picture.hpp"
#include "volumetric_kit/sensor/testing/pattern_reader.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace sensor = volumetric_kit::sensor;
namespace st = sensor::testing;
namespace vkc = volumetric_kit::core;
using sensor::RgbdFrame;
using sensor::YuvImage;
using sensor::YuvLayout;
using st::HostNv12;
using st::PatternReader;
using namespace std::chrono_literals;

// Made by create and held where it was made: never copied or moved.
static_assert(!std::is_copy_constructible_v<PatternReader> &&
                  !std::is_copy_assignable_v<PatternReader> &&
                  !std::is_move_constructible_v<PatternReader> &&
                  !std::is_move_assignable_v<PatternReader>,
              "PatternReader is an aggregate");

namespace {

constexpr std::uint8_t kJunk = 0xEE;
constexpr std::uint32_t kSpan = 1u << st::kIndexBits;

// A stride across the band's span, its ends, and two past it.
std::vector<std::uint32_t> indices() {
  std::vector<std::uint32_t> out;
  for (std::uint32_t i = 0; i < kSpan; i += 331) out.push_back(i);
  for (std::uint32_t i : {1u, 2u, 5u, 1000u, kSpan - 1, kSpan, 70000u}) {
    out.push_back(i);
  }
  return out;
}

std::unique_ptr<PatternReader> reader_on(const vkc::test::Gpu& gpu) {
  auto made = PatternReader::create(gpu.device, gpu.allocator);
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  return made.ok() ? std::move(made).value() : nullptr;
}

// `bytes` in a device storage buffer; null if it cannot be made.
std::shared_ptr<const vkc::Buffer> on_device(
    const vkc::test::Gpu& gpu, const std::vector<std::uint8_t>& bytes) {
  auto made = vkc::device_storage_buffer(gpu.allocator, bytes.size());
  if (!made.ok() ||
      !vkc::test::write_back(gpu.device, gpu.allocator, *made, bytes).ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

// `rows` rows of `row_bytes` from `data` appended to `blob` after 3 junk
// bytes, `stride` apart, the padding junk; their offset.
std::uint64_t place(std::vector<std::uint8_t>& blob, const std::uint8_t* data,
                    std::size_t row_bytes, std::size_t rows,
                    std::size_t stride) {
  blob.resize(blob.size() + 3, kJunk);
  const std::size_t at = blob.size();
  blob.resize(at + stride * rows, kJunk);
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(&blob[at + r * stride], data + r * row_bytes, row_bytes);
  }
  return at;
}

// Where a picture's planes are.
enum class Where { Images, Buffer, Host };

// A picture as a reader finds it, and the bytes host planes point into.
struct Planted {
  YuvImage image;
  std::vector<std::uint8_t> bytes;
};

// `p` as `layout`'s planes at `where`: NV12 images `pad` wider and taller
// than the picture; planes in a buffer or on the host after a lead, at odd
// offsets and strides. `image.width` is 0 if it cannot be made.
Planted plant(const vkc::test::Gpu& gpu, const HostNv12& p, YuvLayout layout,
              Where where, std::uint32_t pad = 0) {
  Planted out;
  YuvImage& image = out.image;
  if (where == Where::Images) {
    auto made = st::device_picture(gpu.device, gpu.allocator, p, pad);
    if (!made) {
      std::fprintf(stderr, "%s\n", made.status().message().c_str());
      return out;
    }
    image = std::move(made).value();
    return out;
  }
  image.layout = layout;
  const std::uint32_t cw = (p.width + 1) / 2, ch = (p.height + 1) / 2;
  std::vector<std::uint8_t>& blob = out.bytes;
  image.stride[0] = p.width + 5;
  image.offset[0] =
      place(blob, p.luma.data(), p.width, p.height, image.stride[0]);
  if (layout == YuvLayout::Nv12) {
    image.stride[1] = 2 * std::size_t{cw} + 3;
    image.offset[1] =
        place(blob, p.chroma.data(), 2 * std::size_t{cw}, ch, image.stride[1]);
  } else if (layout == YuvLayout::I420) {
    std::vector<std::uint8_t> cb, cr;
    for (std::size_t k = 0; k < p.chroma.size(); k += 2) {
      cb.push_back(p.chroma[k]);
      cr.push_back(p.chroma[k + 1]);
    }
    image.stride[1] = cw + 1;
    image.offset[1] = place(blob, cb.data(), cw, ch, image.stride[1]);
    image.stride[2] = cw + 6;
    image.offset[2] = place(blob, cr.data(), cw, ch, image.stride[2]);
  }
  image.width = p.width;
  image.height = p.height;
  if (where == Where::Host) {
    image.host = blob.data();
  } else {
    image.device = on_device(gpu, blob);
    if (image.device == nullptr) image.width = 0;
  }
  return out;
}

// `p`'s luma squeezed to limited range, 16 to 235, as a camera codes it.
HostNv12 limited(HostNv12 p) {
  for (std::uint8_t& v : p.luma) {
    v = static_cast<std::uint8_t>(16 + (v * 219 + 127) / 255);
  }
  return p;
}

// What the reader reads in `planted`; prints a failure, and returns -2 for
// one, -1 for none.
long read_of(PatternReader& reader, Planted& planted) {
  auto got = reader.read(planted.image);
  if (!got) {
    std::fprintf(stderr, "%s\n", got.status().message().c_str());
    return -2;
  }
  return got->has_value() ? static_cast<long>(**got) : -1;
}

// Across the span, each picture read on the GPU as pattern_index reads it
// on the host: the sweep as images and as luma alone in a buffer; every
// other form, limited range and the decoder's size on a few.
int test_pictures(const vkc::test::Gpu& gpu, PatternReader& reader) {
  int reads = 0;
  for (const auto [w, h] : {std::pair{256u, 16u}, std::pair{300u, 21u}}) {
    for (std::uint32_t i : indices()) {
      const HostNv12 p = st::test_pattern(w, h, i);
      const long want = st::pattern_index(p.luma.data(), w);
      CHECK(want == static_cast<long>(i % kSpan));
      Planted images = plant(gpu, p, YuvLayout::Nv12, Where::Images, i % 4);
      Planted luma = plant(gpu, p, YuvLayout::Luma, Where::Buffer);
      CHECK(images.image.width != 0 && luma.image.width != 0);
      CHECK(read_of(reader, images) == want);
      CHECK(read_of(reader, luma) == want);
      reads += 2;
    }
  }
  for (std::uint32_t i : {0u, 6u, 9999u, kSpan - 1}) {
    const HostNv12 p = st::test_pattern(300, 21, i);
    const HostNv12 squeezed = limited(p);
    const long want = static_cast<long>(i % kSpan);
    CHECK(st::pattern_index(squeezed.luma.data(), 300) == want);
    for (const HostNv12* q : {&p, &squeezed}) {
      for (const YuvLayout layout :
           {YuvLayout::Nv12, YuvLayout::I420, YuvLayout::Luma}) {
        for (const Where where : {Where::Buffer, Where::Host}) {
          Planted planted = plant(gpu, *q, layout, where);
          CHECK(planted.image.width != 0);
          planted.image.full_range = q == &p;
          CHECK(read_of(reader, planted) == want);
          ++reads;
        }
      }
      Planted images = plant(gpu, *q, YuvLayout::Nv12, Where::Images, 3);
      CHECK(read_of(reader, images) == want);
      ++reads;
    }
  }
  for (std::uint32_t i : {77u, 12345u}) {
    Planted images = plant(gpu, st::test_pattern(1920, 1440, i),
                           YuvLayout::Nv12, Where::Images);
    CHECK(read_of(reader, images) == static_cast<long>(i));
    ++reads;
  }
  std::printf("  pictures: %d reads equal pattern_index\n", reads);
  return 0;
}

// `d`'s samples as bytes.
std::vector<std::uint8_t> bytes_of(const std::vector<std::uint16_t>& d) {
  std::vector<std::uint8_t> out(d.size() * 2);
  std::memcpy(out.data(), d.data(), out.size());
  return out;
}

// A frame of `d`'s depth on the host, no colour.
RgbdFrame host_frame(const st::PatternDepth& d) {
  RgbdFrame f;
  f.depth = d.depth.data();
  f.metres_per_unit = 0.001f;
  f.depth_camera.size = {d.width, d.height};
  return f;
}

// A frame of `d`'s depth in a device buffer after a lead of `lead` bytes;
// its buffer null if it cannot be made.
RgbdFrame device_frame(const vkc::test::Gpu& gpu, const st::PatternDepth& d,
                       std::size_t lead) {
  RgbdFrame f;
  std::vector<std::uint8_t> blob(lead, kJunk);
  const std::vector<std::uint8_t> samples = bytes_of(d.depth);
  blob.insert(blob.end(), samples.begin(), samples.end());
  blob.resize(blob.size() + 2, kJunk);
  f.depth_device.buffer = on_device(gpu, blob);
  f.depth_device.offset = lead;
  f.metres_per_unit = 0.001f;
  f.depth_camera.size = {d.width, d.height};
  return f;
}

// Across the span, depth read on the GPU as depth_index reads it on the
// host, on the host and in a device buffer; a frame without colour has no
// picture index.
int test_depth(const vkc::test::Gpu& gpu, PatternReader& reader) {
  int reads = 0;
  for (const auto [w, h] : {std::pair{256u, 192u}, std::pair{300u, 20u}}) {
    for (std::uint32_t i : indices()) {
      const st::PatternDepth d = st::pattern_depth(w, h, i);
      const std::optional<std::uint32_t> want =
          st::depth_index(d.depth.data(), w, h);
      CHECK(want == i % kSpan);
      RgbdFrame host = host_frame(d);
      RgbdFrame device = device_frame(gpu, d, 2 + 4 * (i % 3));
      CHECK(device.depth_device.buffer != nullptr);
      for (RgbdFrame* f : {&host, &device}) {
        auto got = reader.read(*f);
        if (!got) std::fprintf(stderr, "%s\n", got.status().message().c_str());
        CHECK(got.ok() && got->depth == want && !got->picture);
        ++reads;
      }
    }
  }
  std::printf("  depth: %d reads equal depth_index\n", reads);
  return 0;
}

// A frame's picture and depth in one read, apart or alike; both in one
// device buffer, taken over once.
int test_frames(const vkc::test::Gpu& gpu, PatternReader& reader) {
  const st::PatternDepth d = st::pattern_depth(256, 192, 6);
  RgbdFrame f = host_frame(d);
  Planted images = plant(gpu, st::test_pattern(320, 240, 5), YuvLayout::Nv12,
                         Where::Images, 3);
  f.color = images.image;
  auto got = reader.read(f);
  CHECK(got.ok() && got->picture == 5u && got->depth == 6u);

  // Depth, then NV12's planes, in one EXCLUSIVE buffer CUDA wrote.
  const HostNv12 p = st::test_pattern(256, 16, 9);
  Planted host = plant(gpu, p, YuvLayout::Nv12, Where::Host);
  const st::PatternDepth near = st::pattern_depth(256, 16, 9);
  std::vector<std::uint8_t> blob = bytes_of(near.depth);
  const std::size_t planes_at = blob.size() + 2;
  blob.resize(planes_at, kJunk);
  blob.insert(blob.end(), host.bytes.begin(), host.bytes.end());
  const auto buffer = on_device(gpu, blob);
  CHECK(buffer != nullptr);
  RgbdFrame one = host_frame(near);
  one.depth = nullptr;
  one.depth_device.buffer = buffer;
  one.depth_device.queue_family = sensor::kQueueFamilyExternal;
  one.color = host.image;
  one.color.host = nullptr;
  one.color.device = buffer;
  for (int k = 0; k < 2; ++k) one.color.offset[k] += planes_at;
  one.color.queue_family = sensor::kQueueFamilyExternal;
  // Named with two writers, it is refused, and nothing changes.
  one.depth_device.queue_family = sensor::kQueueFamilyIgnored;
  CHECK(reader.read(one).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(one.color.queue_family == sensor::kQueueFamilyExternal);
  one.depth_device.queue_family = sensor::kQueueFamilyExternal;
  got = reader.read(one);
  if (!got) std::fprintf(stderr, "%s\n", got.status().message().c_str());
  CHECK(got.ok() && got->picture == 9u && got->depth == 9u);
  CHECK(one.color.queue_family == gpu.device.queue_family());
  CHECK(one.depth_device.queue_family == gpu.device.queue_family());
  got = reader.read(one);  // nothing to take over now
  CHECK(got.ok() && got->picture == 9u && got->depth == 9u);
  std::printf(
      "  frames: picture and depth apart and alike, one buffer taken over "
      "once\n");
  return 0;
}

// `luma`'s square `s` (0 to 15) of a 256-wide band set to `v` throughout.
void fill_square(std::vector<std::uint8_t>& luma, std::uint32_t width,
                 std::uint32_t s, std::uint8_t v) {
  for (std::uint32_t y = 0; y < st::kIndexSquare; ++y) {
    std::fill_n(&luma[std::size_t{y} * width + s * st::kIndexSquare],
                st::kIndexSquare, v);
  }
}

// What the reader reads in host luma `p`, as an NV12 image.
long read_luma(const vkc::test::Gpu& gpu, PatternReader& reader,
               const HostNv12& p) {
  Planted planted = plant(gpu, p, YuvLayout::Nv12, Where::Images);
  if (planted.image.width == 0) return -2;
  return read_of(reader, planted);
}

// None where the band does not fit or names no index: a blend, references
// too close, a square past a margin, torn depth, swapped references. The
// index read through noise and a blur.
int test_none(const vkc::test::Gpu& gpu, PatternReader& reader) {
  CHECK(read_luma(gpu, reader, st::test_pattern(255, 16, 3)) == -1);
  CHECK(read_luma(gpu, reader, st::test_pattern(256, 15, 3)) == -1);

  // Frames 5 and 10 averaged: their four low squares at the middle, which
  // pattern_index's midpoint reads as clear, naming frame 0.
  HostNv12 blend = st::test_pattern(256, 16, 5);
  const HostNv12 ten = st::test_pattern(256, 16, 10);
  for (std::size_t k = 0; k < blend.luma.size(); ++k) {
    blend.luma[k] =
        static_cast<std::uint8_t>((blend.luma[k] + ten.luma[k]) / 2);
  }
  CHECK(st::pattern_index(blend.luma.data(), 256) == 0);
  CHECK(read_luma(gpu, reader, blend) == -1);

  // The margins, exactly: a square at three quarters of the span or more is
  // set, at a quarter or less clear, between them none.
  const auto with_square = [&](std::uint8_t v) {
    HostNv12 p = st::test_pattern(256, 16, 0);
    fill_square(p.luma, 256, 2, v);
    return read_luma(gpu, reader, p);
  };
  CHECK(with_square(192) == 1);  // 4 * 192 >= 3 * 255
  CHECK(with_square(191) == -1);
  CHECK(with_square(64) == -1);
  CHECK(with_square(63) == 0);  // 4 * 63 <= 255
  // References 64 apart read; 63 apart, as a grey picture's, do not.
  for (const auto [white, want] : {std::pair{96, 0L}, std::pair{95, -1L}}) {
    HostNv12 p = st::test_pattern(256, 16, 0);
    for (std::uint32_t s = 0; s < 16; ++s) fill_square(p.luma, 256, s, 32);
    fill_square(p.luma, 256, 0, static_cast<std::uint8_t>(white));
    CHECK(read_luma(gpu, reader, p) == want);
  }
  HostNv12 grey = st::test_pattern(256, 16, 0);
  std::fill(grey.luma.begin(), grey.luma.end(), 128);
  CHECK(st::pattern_index(grey.luma.data(), 256) == -1);
  CHECK(read_luma(gpu, reader, grey) == -1);

  // +-40 noise, then a 4 x 4 box blur: still index 12345.
  HostNv12 noisy = st::test_pattern(320, 32, 12345);
  std::uint32_t state = 7;
  for (std::uint8_t& v : noisy.luma) {
    state = state * 1664525u + 1013904223u;
    const int n = static_cast<int>((state >> 8) % 81) - 40;
    v = static_cast<std::uint8_t>(std::clamp(v + n, 0, 255));
  }
  HostNv12 blurred = noisy;
  for (std::uint32_t y = 0; y + 4 <= 32; ++y) {
    for (std::uint32_t x = 0; x + 4 <= 320; ++x) {
      unsigned sum = 0;
      for (std::uint32_t j = 0; j < 4; ++j) {
        for (std::uint32_t i = 0; i < 4; ++i) {
          sum += noisy.luma[std::size_t{y + j} * 320 + x + i];
        }
      }
      blurred.luma[std::size_t{y + 1} * 320 + x + 1] =
          static_cast<std::uint8_t>((sum + 8) / 16);
    }
  }
  CHECK(read_luma(gpu, reader, noisy) == 12345);
  CHECK(read_luma(gpu, reader, blurred) == 12345);

  // Depth: under 256 x 16, one sample of a square's central half changed,
  // the references swapped; a sample off the central half changes nothing.
  const auto depth_read = [&](const st::PatternDepth& d) {
    RgbdFrame f = host_frame(d);
    auto got = reader.read(f);
    return got.ok() && !got->picture ? got->depth
                                     : std::optional<std::uint32_t>{99999};
  };
  CHECK(!depth_read(st::pattern_depth(255, 192, 3)));
  CHECK(!depth_read(st::pattern_depth(256, 15, 3)));
  st::PatternDepth torn = st::pattern_depth(256, 192, 3);
  torn.depth[std::size_t{9} * 256 + 7 * 16 + 10] = 1500;
  CHECK(!st::depth_index(torn.depth.data(), 256, 192));
  CHECK(!depth_read(torn));
  st::PatternDepth edge = st::pattern_depth(256, 192, 3);
  edge.depth[std::size_t{2} * 256 + 7 * 16 + 1] = 1500;
  CHECK(depth_read(edge) == 3u);
  st::PatternDepth swapped = st::pattern_depth(256, 192, 3);
  for (std::uint32_t y = 0; y < 16; ++y) {
    for (std::uint32_t x = 0; x < 32; ++x) {
      std::uint16_t& v = swapped.depth[std::size_t{y} * 256 + x];
      v = v == st::kPatternNearMm ? st::kPatternFarMm : st::kPatternNearMm;
    }
  }
  CHECK(!st::depth_index(swapped.depth.data(), 256, 192));
  CHECK(!depth_read(swapped));
  std::printf(
      "  none: small, blended, past a margin, references too close, torn "
      "and swapped; read through noise and blur\n");
  return 0;
}

// A picture whose ready value the host sets 50 ms after the read starts is
// read only after it, by a batch that waits on the device.
int test_ready(const vkc::test::Gpu& gpu, PatternReader& reader) {
  auto made = vkc::TimelineSemaphore::create(gpu.device);
  CHECK(made.ok());
  const auto ready =
      std::make_shared<vkc::TimelineSemaphore>(std::move(made).value());
  Planted planted =
      plant(gpu, st::test_pattern(256, 16, 77), YuvLayout::Nv12, Where::Images);
  CHECK(planted.image.width != 0);
  planted.image.ready = ready;
  planted.image.ready_value = 2;
  std::atomic<bool> done{false};
  long got = -3;
  std::thread read([&] {
    got = read_of(reader, planted);
    done.store(true);
  });
  // The value set before any check, so a failing one leaves no read waiting.
  std::this_thread::sleep_for(50ms);
  const bool waited = !done.load();
  const bool set = ready->signal(2).ok();
  read.join();
  CHECK(waited && set);
  CHECK(got == 77);
  // Set already: read at once.
  CHECK(read_of(reader, planted) == 77);
  std::printf("  ready: the read waited for the picture's value\n");
  return 0;
}

// Pictures and depth from outside Vulkan taken over, their ownership
// published, and read again with nothing to take over.
int test_ownership(const vkc::test::Gpu& gpu, PatternReader& reader) {
  const std::uint32_t own = gpu.device.queue_family();
  const HostNv12 p = st::test_pattern(256, 16, 31);
  Planted images = plant(gpu, p, YuvLayout::Nv12, Where::Images);
  Planted buffer = plant(gpu, p, YuvLayout::I420, Where::Buffer);
  for (Planted* planted : {&images, &buffer}) {
    planted->image.queue_family = sensor::kQueueFamilyExternal;
    CHECK(read_of(reader, *planted) == 31);
    CHECK(planted->image.queue_family == own);
    CHECK(read_of(reader, *planted) == 31);
    CHECK(planted->image.queue_family == own);
  }
  RgbdFrame f = device_frame(gpu, st::pattern_depth(256, 16, 32), 4);
  f.depth_device.queue_family = sensor::kQueueFamilyExternal;
  auto got = reader.read(f);
  CHECK(got.ok() && got->depth == 32u);
  CHECK(f.depth_device.queue_family == own);
  // A host picture has no family to publish.
  Planted host = plant(gpu, p, YuvLayout::Nv12, Where::Host);
  CHECK(read_of(reader, host) == 31);
  CHECK(host.image.queue_family == sensor::kQueueFamilyIgnored);
  std::printf("  ownership: taken over from outside Vulkan and published\n");
  return 0;
}

bool refused(const vkc::Status& status) {
  return status.domain() == vkc::Status::Code::InvalidArgument;
}

// Each refusal, with the inputs as they were; the reader reads on after.
int test_refusals(const vkc::test::Gpu& gpu, PatternReader& reader) {
  const HostNv12 p = st::test_pattern(256, 16, 44);
  YuvImage none;
  none.width = 256;
  none.height = 16;
  CHECK(refused(reader.read(none).status()));

  Planted base = plant(gpu, p, YuvLayout::Luma, Where::Buffer);
  CHECK(base.image.width != 0);
  const auto refuses = [&](YuvImage picture) {
    const std::uint32_t family = picture.queue_family;
    const vkc::Status status = reader.read(picture).status();
    return refused(status) && picture.queue_family == family;
  };
  YuvImage empty = base.image;
  empty.width = 0;
  CHECK(refuses(empty));
  for (const YuvLayout layout : {YuvLayout::Yuyv, YuvLayout::V210}) {
    YuvImage other = base.image;
    other.layout = layout;
    CHECK(refuses(other));
  }
  YuvImage both = base.image;
  both.host = base.bytes.data();
  CHECK(refuses(both));
  YuvImage past = base.image;
  past.stride[0] = 4096;
  CHECK(refuses(past));
  YuvImage lacks = base.image;
  lacks.queue_family = 99;
  CHECK(refuses(lacks));
  vkc::BufferDesc desc;
  desc.size = base.bytes.size();
  desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto storage_only = gpu.allocator.create_buffer(desc);
  CHECK(storage_only.ok());
  YuvImage no_copy = base.image;
  no_copy.device =
      std::make_shared<const vkc::Buffer>(std::move(storage_only).value());
  CHECK(refuses(no_copy));
  YuvImage unready = base.image;
  unready.ready = std::make_shared<vkc::TimelineSemaphore>();
  unready.ready_value = 1;
  CHECK(refuses(unready));
  Planted images = plant(gpu, p, YuvLayout::Nv12, Where::Images);
  YuvImage not_nv12 = images.image;
  not_nv12.layout = YuvLayout::Luma;
  CHECK(refuses(not_nv12));

  // Depth.
  const st::PatternDepth d = st::pattern_depth(256, 16, 45);
  RgbdFrame twice = device_frame(gpu, d, 2);
  twice.depth = d.depth.data();
  CHECK(refused(reader.read(twice).status()));
  RgbdFrame odd = device_frame(gpu, d, 2);
  odd.depth_device.offset = 3;
  CHECK(refused(reader.read(odd).status()));
  RgbdFrame beyond = device_frame(gpu, d, 2);
  beyond.depth_device.offset = 8;
  CHECK(refused(reader.read(beyond).status()));
  RgbdFrame copyless = device_frame(gpu, d, 0);
  auto depth_storage = gpu.allocator.create_buffer(desc);
  CHECK(depth_storage.ok());
  copyless.depth_device.buffer =
      std::make_shared<const vkc::Buffer>(std::move(depth_storage).value());
  CHECK(refused(reader.read(copyless).status()));
  RgbdFrame bad_picture = host_frame(d);
  bad_picture.color = empty;
  CHECK(refused(reader.read(bad_picture).status()));

  // The reader reads on.
  CHECK(read_of(reader, base) == 44);
  RgbdFrame good = device_frame(gpu, d, 2);
  auto got = reader.read(good);
  CHECK(got.ok() && got->depth == 45u);
  std::printf("  refusals: each named, the inputs as they were\n");
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const auto reader = reader_on(gpu);
  CHECK(reader != nullptr);
  CHECK(test_pictures(gpu, *reader) == 0);
  CHECK(test_depth(gpu, *reader) == 0);
  CHECK(test_frames(gpu, *reader) == 0);
  CHECK(test_none(gpu, *reader) == 0);
  CHECK(test_ready(gpu, *reader) == 0);
  CHECK(test_ownership(gpu, *reader) == 0);
  CHECK(test_refusals(gpu, *reader) == 0);
  std::puts("sensor_pattern_reader: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
