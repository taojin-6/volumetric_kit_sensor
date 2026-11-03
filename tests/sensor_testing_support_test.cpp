// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The package's test support (include/volumetric_kit/sensor/testing/): a
// clip's access units at the delimiters, from three- and four-byte start
// codes, bytes before the first left out, and a committed clip's from its
// file; the decoder and encoder promises as VS_TEST_HEVC_BACKEND and
// VS_TEST_HEVC_ENCODER set them; the test pattern's planes, its 0 and 255,
// its index band read back by pattern_index for indices across its span, a
// different picture at each index, its bytes pinned by MD5, and the test
// depth's band at the same pixels; and,
// on the GPU, its device picture's two plane images, their usage, layout and
// size, each plane at their corner with 0xEE past it, and the refusals. The
// host parts run without a device.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "md5.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/vulkan_policy.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/testing/hevc_clip.hpp"
#include "volumetric_kit/sensor/testing/nv12_picture.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;
namespace vkc = volumetric_kit::core;
using Bytes = std::vector<std::uint8_t>;

namespace {

// Three access units behind two junk bytes: a delimiter after a four-byte
// start code, then one after a three-byte start code with a parameter set's
// NAL unit, then a delimiter alone at the end.
int test_split() {
  const Bytes first = {0, 0, 0, 1, 0x46, 0x01, 0x50, 0, 0, 1, 0x40, 0x01, 9};
  const Bytes second = {0, 0, 1, 0x46, 0x01, 0x30, 0, 0, 1, 0x42, 0x01, 7};
  const Bytes third = {0, 0, 1, 0x46, 0x01, 0x10};
  Bytes clip = {0x12, 0x34};
  for (const Bytes* unit : {&first, &second, &third}) {
    clip.insert(clip.end(), unit->begin(), unit->end());
  }
  const auto units = testing::split_access_units(clip.data(), clip.size());
  CHECK(units.size() == 3);
  CHECK(units[0] == first && units[1] == second && units[2] == third);
  // No delimiter, no bytes, and a start code cut short: no units.
  const Bytes vps = {0, 0, 1, 0x40, 0x01, 9};
  CHECK(testing::split_access_units(vps.data(), vps.size()).empty());
  CHECK(testing::split_access_units(nullptr, 0).empty());
  CHECK(testing::split_access_units(third.data(), 3).empty());

  // A committed clip: 8 units that make the file up.
  const char* patches = VS_HEVC_DATA "/patches_256x144.h265";
  const auto clip_units = testing::access_units_of_file(patches);
  CHECK(clip_units.size() == 8);
  for (const Bytes& unit : clip_units) {
    CHECK(unit.size() > 5 && unit[0] == 0 && unit[1] == 0);
  }
  CHECK(testing::access_units_of_file(VS_HEVC_DATA "/none.h265").empty());
  return 0;
}

int test_promise() {
  namespace test = vkc::test;
  {
    const test::ScopedEnv none("VS_TEST_HEVC_BACKEND", nullptr);
    CHECK(!testing::hevc_decoder_promised());
  }
  {
    const test::ScopedEnv empty("VS_TEST_HEVC_BACKEND", "");
    CHECK(!testing::hevc_decoder_promised());
  }
  const test::ScopedEnv promised("VS_TEST_HEVC_BACKEND", "videotoolbox");
  CHECK(testing::hevc_decoder_promised());
  return 0;
}

int test_encoder_promise() {
  namespace test = vkc::test;
  {
    const test::ScopedEnv none("VS_TEST_HEVC_ENCODER", nullptr);
    CHECK(!testing::hevc_encoder_promised());
  }
  {
    const test::ScopedEnv empty("VS_TEST_HEVC_ENCODER", "");
    CHECK(!testing::hevc_encoder_promised());
  }
  const test::ScopedEnv promised("VS_TEST_HEVC_ENCODER", "videotoolbox");
  CHECK(testing::hevc_encoder_promised());
  return 0;
}

// The index the band of @p p names (testing::pattern_index).
long read_index(const testing::HostNv12& p) {
  return testing::pattern_index(p.luma.data(), p.width);
}

// The planes' sizes at even, odd and single-pixel sizes; 0 and 255 in each
// picture at least 32 wide; the band names the index modulo 16384 and its
// chroma is neutral; every index a different picture, the same index the
// same one.
int test_pattern() {
  struct Size {
    std::uint32_t w, h;
  };
  for (const Size s : {Size{256, 144}, Size{37, 21}, Size{1, 1}}) {
    const testing::HostNv12 p = testing::test_pattern(s.w, s.h, 5);
    CHECK(p.width == s.w && p.height == s.h);
    CHECK(p.luma.size() == std::size_t{s.w} * s.h);
    CHECK(p.chroma.size() == std::size_t{(s.w + 1) / 2} * ((s.h + 1) / 2) * 2);
  }
  for (const Size s : {Size{256, 144}, Size{32, 1}, Size{1920, 1440}}) {
    const testing::HostNv12 p = testing::test_pattern(s.w, s.h, 0);
    bool zero = false, full = false;
    for (const std::uint8_t v : p.luma) {
      zero = zero || v == 0;
      full = full || v == 255;
    }
    CHECK(zero && full);
  }
  for (const std::uint32_t i : {0u, 1u, 5u, 12345u, 16383u, 16384u, 70000u}) {
    const testing::HostNv12 p = testing::test_pattern(256, 144, i);
    CHECK(read_index(p) == static_cast<long>(i % 16384));
    // Read with a stride past the width, as from a device picture's rows.
    Bytes strided(std::size_t{300} * 16);
    for (std::uint32_t y = 0; y < 16; ++y) {
      std::copy_n(p.luma.data() + std::size_t{y} * 256, 256,
                  strided.data() + std::size_t{y} * 300);
    }
    CHECK(testing::pattern_index(strided.data(), 300) ==
          static_cast<long>(i % 16384));
    for (std::uint32_t y = 0; y < 8; ++y) {
      for (std::uint32_t x = 0; x < 256; ++x) {  // 128 pairs, 2 bytes each
        CHECK(p.chroma[std::size_t{y} * 256 + x] == 128);
      }
    }
  }
  // A flat picture is not the pattern.
  const Bytes grey(std::size_t{256} * 16, 128);
  CHECK(testing::pattern_index(grey.data(), 256) == -1);
  // The bytes are the ones the band's constants were written with, before
  // they moved to index_band.hpp (sensor 0.23.0's).
  struct Pinned {
    std::uint32_t w, h, index;
    const char* md5;  // of the luma, then the chroma
  };
  for (const Pinned& pin :
       {Pinned{256, 144, 0, "cb5721796bf623b92b4fc22f768bf280"},
        Pinned{256, 144, 12345, "2d2b6cbab77b5585307b31076b605cf5"},
        Pinned{37, 21, 9, "b70cbfa67f31b1e068a7472691e527ac"},
        Pinned{1920, 1440, 70000, "f45e7232dbfccd7b3647c75dc676fbc5"}}) {
    const testing::HostNv12 p = testing::test_pattern(pin.w, pin.h, pin.index);
    vs_test::Md5 md5;
    md5.update(p.luma.data(), p.luma.size());
    md5.update(p.chroma.data(), p.chroma.size());
    CHECK(md5.hex() == pin.md5);
  }
  // The test depth carries the band at the same pixels: a square is near
  // where the picture's is 255, far where it is 0.
  for (const std::uint32_t i : {5u, 12345u}) {
    const testing::HostNv12 p = testing::test_pattern(256, 192, i);
    const testing::PatternDepth d = testing::pattern_depth(256, 192, i);
    for (std::size_t k = 0; k < std::size_t{testing::kIndexSquare} * 256; ++k) {
      CHECK(d.depth[k] == (p.luma[k] == 255 ? testing::kPatternNearMm
                                            : testing::kPatternFarMm));
    }
    CHECK(read_index(p) == static_cast<long>(i) &&
          testing::depth_index(d.depth.data(), 256, 192) == i);
  }
  const testing::HostNv12 a = testing::test_pattern(256, 144, 7);
  const testing::HostNv12 b = testing::test_pattern(256, 144, 8);
  const testing::HostNv12 again = testing::test_pattern(256, 144, 7);
  CHECK(a.luma != b.luma && a.chroma != b.chroma);
  CHECK(a.luma == again.luma && a.chroma == again.chroma);
  // Below the band the picture moves with the index too.
  const std::size_t below = std::size_t{testing::kIndexSquare} * 256;
  CHECK(!std::equal(a.luma.begin() + static_cast<std::ptrdiff_t>(below),
                    a.luma.end(),
                    b.luma.begin() + static_cast<std::ptrdiff_t>(below)));
  return 0;
}

// The whole of @p image, rows packed, read back; empty on a failure.
Bytes whole(const vkc::test::Gpu& gpu, const vkc::Image& image,
            std::uint32_t bytes_per_column) {
  const std::size_t bytes =
      std::size_t{image.width()} * image.height() * bytes_per_column;
  auto buffer = vkc::device_storage_buffer(gpu.allocator, (bytes + 3) / 4 * 4);
  if (!buffer) return {};
  Bytes out(bytes);
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  if (!batch.copy(image, image.width(), image.height(), *buffer, 0).ok() ||
      !batch.readback(*buffer, 0, bytes, out.data()).ok() ||
      !batch.submit().ok()) {
    return {};
  }
  return out;
}

// Each plane's image: its format and usage, a view, GENERAL, pad wider and
// taller than the plane, the plane at its corner and 0xEE past it.
int check_plane(const vkc::test::Gpu& gpu, const vkc::Image& image,
                VkFormat format, std::uint32_t bytes_per_column,
                const Bytes& plane, std::uint32_t columns, std::uint32_t rows,
                std::uint32_t pad) {
  CHECK(image.format() == format && image.view() != VK_NULL_HANDLE);
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  CHECK((image.usage() & usage) == usage);
  CHECK(image.layout() == VK_IMAGE_LAYOUT_GENERAL);
  CHECK(image.width() == columns + pad && image.height() == rows + pad);
  const Bytes image_bytes = whole(gpu, image, bytes_per_column);
  CHECK(!image_bytes.empty());
  const std::size_t row = std::size_t{image.width()} * bytes_per_column;
  for (std::uint32_t y = 0; y < image.height(); ++y) {
    for (std::size_t x = 0; x < row; ++x) {
      const std::uint8_t got = image_bytes[y * row + x];
      const bool inside =
          y < rows && x < std::size_t{columns} * bytes_per_column;
      CHECK(got == (inside ? plane[y * columns * bytes_per_column + x] : 0xEE));
    }
  }
  return 0;
}

int test_device_picture(const vkc::test::Gpu& gpu) {
  for (const std::uint32_t pad : {0u, 3u}) {
    const testing::HostNv12 p = testing::test_pattern(37, 21, 9);
    auto made = testing::device_picture(gpu.device, gpu.allocator, p, pad);
    if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
    CHECK(made.ok());
    const sensor::YuvImage& yuv = *made;
    CHECK(yuv.layout == sensor::YuvLayout::Nv12 && yuv.width == 37 &&
          yuv.height == 21 && yuv.device == nullptr && yuv.host == nullptr);
    CHECK(yuv.queue_family == sensor::kQueueFamilyIgnored);
    CHECK(yuv.image[0] != nullptr && yuv.image[1] != nullptr);
    if (check_plane(gpu, *yuv.image[0], VK_FORMAT_R8_UNORM, 1, p.luma, 37, 21,
                    pad) != 0) {
      return 1;
    }
    if (check_plane(gpu, *yuv.image[1], VK_FORMAT_R8G8_UNORM, 2, p.chroma, 19,
                    11, pad) != 0) {
      return 1;
    }
  }
  testing::HostNv12 bad = testing::test_pattern(8, 4, 0);
  bad.chroma.pop_back();
  const auto code = [&gpu](const testing::HostNv12& picture) {
    return testing::device_picture(gpu.device, gpu.allocator, picture)
        .status()
        .domain();
  };
  CHECK(code(bad) == vkc::Status::Code::InvalidArgument);
  CHECK(code(testing::HostNv12{}) == vkc::Status::Code::InvalidArgument);
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  if (test_device_picture(gpu) != 0) return 1;
  std::puts("sensor_testing_support: OK");
  return 0;
}

}  // namespace

int main() {
  if (test_split() != 0 || test_promise() != 0 || test_encoder_promise() != 0 ||
      test_pattern() != 0) {
    return 1;
  }
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
