// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// LumaPass (sensor_prep) on synthetic pictures: every layout a driver hands
// out -- V210, NV12, I420, YUYV and luma alone, in a buffer and on the host,
// and NV12 plane images -- held bit for bit to a host reference of its luma
// (V210's 10-bit codes from tests/v210.hpp rounded once, (code + 2) / 4) and
// of the 2x2 half, odd edges included, at odd sizes, offsets and strides and
// at widths 3840 and 3842; a mixed list in one submit; the outputs pooled,
// reused after warm-up with nothing allocated, the smallest free one replaced
// when none holds a picture, held while owned and past the pass; buffers and
// images taken over from outside Vulkan and from another family, once each,
// the outputs read on a renderer's family; every refusal, the list refused
// whole; and the stage row. VS_TEST_LUMA_TIMING=<runs> times a DeckLink
// pair's pictures (two 3840x2160 V210 in buffers) and prints the device
// span's p50 and p95. Skips without a device.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "test_image.hpp"
#include "v210.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/testing/two_family_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/luma_pass.hpp"
#include "volumetric_kit/sensor/prep/picture_readback.hpp"
#include "volumetric_kit/sensor/testing/nv12_picture.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;
using sensor::DeviceLuma;
using sensor::LumaPass;
using sensor::PictureReadback;
using sensor::PictureView;
using sensor::YuvImage;
using sensor::YuvLayout;
using Planes = PictureReadback::Planes;

// Made by create and held where it was made: never copied or moved.
static_assert(!std::is_copy_constructible_v<LumaPass> &&
                  !std::is_copy_assignable_v<LumaPass> &&
                  !std::is_move_constructible_v<LumaPass> &&
                  !std::is_move_assignable_v<LumaPass>,
              "LumaPass is an aggregate");

namespace {

constexpr std::uint8_t kJunk = 0xEE;

// Where a picture's planes are.
enum class Where { Device, Host, Images };

const char* name(YuvLayout layout) {
  switch (layout) {
    case YuvLayout::I420:
      return "I420";
    case YuvLayout::Nv12:
      return "NV12";
    case YuvLayout::Yuyv:
      return "YUYV";
    case YuvLayout::Luma:
      return "luma";
    case YuvLayout::V210:
      return "V210";
  }
  return "?";
}

const char* name(Where where) {
  return where == Where::Device ? "buffer"
         : where == Where::Host ? "host"
                                : "images";
}

// A picture as a driver hands it out, and the luma the pass must make of it.
struct Source {
  YuvImage image;
  std::vector<std::uint8_t> bytes;  // the planes, kept for host pictures
  std::vector<std::uint8_t> luma;   // the full luma, w x h, rows packed
};

std::uint32_t next(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state >> 8;
}

// `count` codes below `limit`, from `seed`, neighbours unrelated.
std::vector<std::uint16_t> codes(std::size_t count, std::uint32_t limit,
                                 std::uint32_t seed) {
  std::vector<std::uint16_t> out(count);
  std::uint32_t state = seed * 2654435761u + 1u;
  for (std::uint16_t& c : out) {
    c = static_cast<std::uint16_t>(next(state) % limit);
  }
  return out;
}

std::vector<std::uint8_t> bytes_of(const std::vector<std::uint16_t>& c) {
  return std::vector<std::uint8_t>(c.begin(), c.end());
}

// `rows` rows of `row_bytes` from `data`, rows packed, appended to `blob`
// after 3 junk bytes at `stride` apart, the padding junk; their offset.
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

// `bytes` in a device storage buffer shared among `families` (exclusive for
// none); null if it cannot be made.
std::shared_ptr<const vkc::Buffer> on_device(
    const vkc::Device& device, vkc::Allocator& allocator,
    const std::vector<std::uint8_t>& bytes,
    const std::uint32_t* families = nullptr, std::uint32_t family_count = 0) {
  auto made = vkc::device_storage_buffer(allocator, bytes.size(), 0, families,
                                         family_count);
  if (!made.ok() ||
      !vkc::test::write_back(device, allocator, *made, bytes).ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

// A `w` x `h` picture of `layout` at `where`, its codes from `seed`: planes
// after a lead, at odd offsets and strides (V210's on words), padding junk;
// NV12 images 3 texels wider and taller than the picture
// (testing/nv12_picture.hpp). Its range
// alternates with the seed (V210's is limited). `image.width` is 0 if it
// cannot be made.
Source make_source(const vkc::Device& device, vkc::Allocator& allocator,
                   YuvLayout layout, Where where, std::uint32_t w,
                   std::uint32_t h, std::uint32_t seed) {
  Source s;
  YuvImage& image = s.image;
  image.layout = layout;
  image.width = w;
  image.height = h;
  image.full_range = layout != YuvLayout::V210 && seed % 2 == 0;
  const std::size_t pixels = std::size_t{w} * h;
  const std::uint32_t cw = (w + 1) / 2, ch = (h + 1) / 2;
  std::vector<std::uint8_t>& blob = s.bytes;
  blob.assign(4096 + 1, kJunk);
  if (layout == YuvLayout::V210) {
    const std::vector<std::uint16_t> y = codes(pixels, 1024, seed);
    const std::vector<std::uint16_t> cb =
        codes(std::size_t{cw} * h, 1024, seed + 101);
    const std::vector<std::uint16_t> cr =
        codes(std::size_t{cw} * h, 1024, seed + 202);
    image.offset[0] = 20;
    image.stride[0] = v210::row_bytes(w) + 8;
    blob = v210::pack(y, cb, cr, w, h, image.offset[0], image.stride[0]);
    s.luma.resize(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
      s.luma[i] = static_cast<std::uint8_t>(std::min(255, (y[i] + 2) / 4));
    }
  } else {
    s.luma = bytes_of(codes(pixels, 256, seed));
    if (layout == YuvLayout::Yuyv) {
      const std::vector<std::uint8_t> c =
          bytes_of(codes(std::size_t{w} * h, 256, seed + 7));
      std::vector<std::uint8_t> packed(2 * pixels);
      for (std::size_t i = 0; i < pixels; ++i) {
        packed[2 * i] = s.luma[i];  // Y0 Cb Y1 Cr
        packed[2 * i + 1] = c[i];
      }
      image.stride[0] = 2 * std::size_t{w} + 6;
      image.offset[0] = place(blob, packed.data(), 2 * w, h, image.stride[0]);
    } else {
      image.stride[0] = w + 5;
      image.offset[0] = place(blob, s.luma.data(), w, h, image.stride[0]);
      const std::vector<std::uint8_t> cb =
          bytes_of(codes(std::size_t{cw} * ch, 256, seed + 11));
      const std::vector<std::uint8_t> cr =
          bytes_of(codes(std::size_t{cw} * ch, 256, seed + 13));
      if (layout == YuvLayout::Nv12) {
        std::vector<std::uint8_t> cbcr;
        for (std::size_t i = 0; i < cb.size(); ++i) {
          cbcr.push_back(cb[i]);
          cbcr.push_back(cr[i]);
        }
        if (where == Where::Images) {
          auto images = sensor::testing::device_picture(
              device, allocator, {w, h, s.luma, cbcr}, 3);
          if (images.ok()) {
            image.image[0] = images->image[0];
            image.image[1] = images->image[1];
          } else {
            image.width = 0;
          }
          image.stride[0] = 0;
          image.offset[0] = 0;
          return s;
        }
        image.stride[1] = 2 * std::size_t{cw} + 3;
        image.offset[1] = place(blob, cbcr.data(), 2 * cw, ch, image.stride[1]);
      } else if (layout == YuvLayout::I420) {
        image.stride[1] = cw + 1;
        image.offset[1] = place(blob, cb.data(), cw, ch, image.stride[1]);
        image.stride[2] = cw + 6;
        image.offset[2] = place(blob, cr.data(), cw, ch, image.stride[2]);
      }
    }
  }
  if (where == Where::Host) {
    image.host = blob.data();
  } else {
    image.device = on_device(device, allocator, blob);
    if (image.device == nullptr) image.width = 0;
  }
  return s;
}

Source make_source(const vkc::test::Gpu& gpu, YuvLayout layout, Where where,
                   std::uint32_t w, std::uint32_t h, std::uint32_t seed) {
  return make_source(gpu.device, gpu.allocator, layout, where, w, h, seed);
}

// The 2x2 half of `full`, `w` x `h`: each the rounded mean of its block, an
// odd last column or row left out.
std::vector<std::uint8_t> half_of(const std::vector<std::uint8_t>& full,
                                  std::uint32_t w, std::uint32_t h) {
  const std::uint32_t hw = w / 2, hh = h / 2;
  std::vector<std::uint8_t> out(std::size_t{hw} * hh);
  for (std::uint32_t v = 0; v < hh; ++v) {
    for (std::uint32_t u = 0; u < hw; ++u) {
      const auto at = [&](std::uint32_t x, std::uint32_t y) {
        return unsigned{full[std::size_t{y} * w + x]};
      };
      const unsigned sum = at(2 * u, 2 * v) + at(2 * u + 1, 2 * v) +
                           at(2 * u, 2 * v + 1) + at(2 * u + 1, 2 * v + 1);
      out[std::size_t{v} * hw + u] = static_cast<std::uint8_t>((sum + 2) / 4);
    }
  }
  return out;
}

// `view`'s rows equal `want`'s, `w` x `h` packed; the first miss printed.
bool rows_equal(const PictureView& view, const std::vector<std::uint8_t>& want,
                std::uint32_t w, std::uint32_t h, const char* what) {
  if (view.width != w || view.height != h || view.luma == nullptr) {
    std::fprintf(stderr, "%s: %ux%u, want %ux%u\n", what, view.width,
                 view.height, w, h);
    return false;
  }
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      const std::uint8_t got = view.luma[std::size_t{y} * view.stride + x];
      const std::uint8_t exp = want[std::size_t{y} * w + x];
      if (got != exp) {
        std::fprintf(stderr, "%s: (%u, %u) is %u, want %u\n", what, x, y, got,
                     exp);
        return false;
      }
    }
  }
  return true;
}

// `luma` is `s`'s luma and half, bit for bit, read back in one list read,
// and described as the pass's outputs are.
int check_luma(PictureReadback& readback, DeviceLuma& luma, const Source& s,
               const char* what) {
  const std::uint32_t w = s.image.width, h = s.image.height;
  CHECK(luma.owner != nullptr && luma.full.device != nullptr);
  CHECK(luma.full.device == luma.half.device);
  for (const YuvImage* out : {&luma.full, &luma.half}) {
    CHECK(out->layout == YuvLayout::Luma && out->host == nullptr);
    CHECK(out->image[0] == nullptr && out->image[1] == nullptr);
    CHECK(out->stride[0] == out->width);
    CHECK(out->full_range == s.image.full_range);
    CHECK(out->queue_family == sensor::kQueueFamilyIgnored);
  }
  CHECK(luma.full.width == w && luma.full.height == h);
  CHECK(luma.full.offset[0] == 0);
  CHECK(luma.half.width == w / 2 && luma.half.height == h / 2);
  CHECK(luma.half.offset[0] >= std::uint64_t{w} * h);
  auto views = readback.read({&luma.full, &luma.half}, Planes::Luma);
  if (!views) std::fprintf(stderr, "%s\n", views.status().message().c_str());
  CHECK(views.ok() && views->size() == 2);
  CHECK(rows_equal((*views)[0], s.luma, w, h, what));
  CHECK(rows_equal((*views)[1], half_of(s.luma, w, h), w / 2, h / 2, what));
  return 0;
}

std::unique_ptr<LumaPass> pass_on(const vkc::Device& device,
                                  vkc::Allocator& allocator,
                                  const std::uint32_t* families = nullptr,
                                  std::uint32_t count = 0) {
  auto made = LumaPass::create(device, allocator, families, count);
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  return made.ok() ? std::move(made).value() : nullptr;
}

std::unique_ptr<PictureReadback> readback_on(const vkc::Device& device,
                                             vkc::Allocator& allocator) {
  auto made = PictureReadback::create(device, allocator);
  return made.ok() ? std::move(made).value() : nullptr;
}

struct Form {
  YuvLayout layout;
  Where where;
};

// Every form a driver hands out.
const Form kForms[] = {
    {YuvLayout::V210, Where::Device}, {YuvLayout::V210, Where::Host},
    {YuvLayout::Nv12, Where::Device}, {YuvLayout::Nv12, Where::Host},
    {YuvLayout::Nv12, Where::Images}, {YuvLayout::I420, Where::Device},
    {YuvLayout::I420, Where::Host},   {YuvLayout::Yuyv, Where::Device},
    {YuvLayout::Yuyv, Where::Host},   {YuvLayout::Luma, Where::Device},
    {YuvLayout::Luma, Where::Host}};

bool even_only(YuvLayout layout) {
  return layout == YuvLayout::V210 || layout == YuvLayout::Yuyv;
}

// Each form at odd and even sizes, the smallest 2x2, and at 3840 and 3842
// wide (V210's last group holding all six pixels and then two), each run
// alone: its luma and half exact.
int test_forms(const vkc::test::Gpu& gpu) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  struct Size {
    std::uint32_t w, h;
  };
  std::uint32_t seed = 0;
  int count = 0;
  for (const Size size : {Size{37, 21}, Size{64, 48}, Size{2, 2}, Size{3, 5},
                          Size{3840, 3}, Size{3842, 3}}) {
    for (const Form form : kForms) {
      const std::uint32_t w =
          even_only(form.layout) ? size.w + size.w % 2 : size.w;
      Source s = make_source(gpu, form.layout, form.where, w, size.h, ++seed);
      CHECK(s.image.width != 0);
      auto luma = pass->run({&s.image});
      if (!luma) std::fprintf(stderr, "%s\n", luma.status().message().c_str());
      CHECK(luma.ok() && luma->size() == 1);
      char what[96];
      std::snprintf(what, sizeof(what), "%s %s %ux%u", name(form.layout),
                    name(form.where), w, size.h);
      CHECK(check_luma(*readback, (*luma)[0], s, what) == 0);
      ++count;
    }
  }
  std::printf("  forms: %d pictures exact, luma and half, 2x2 to 3842 wide\n",
              count);
  return 0;
}

// V210's 10-bit codes at the rounding's edges: every code 0..1023 in a
// 3842-wide row, so each two low bits and the clamp past 1019 are seen.
int test_v210_codes(const vkc::test::Gpu& gpu) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  const std::uint32_t w = 3842, h = 2;
  std::vector<std::uint16_t> y(std::size_t{w} * h);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = static_cast<std::uint16_t>((i * 7) % 1024);
  }
  const std::vector<std::uint16_t> c(std::size_t{(w + 1) / 2} * h, 512);
  Source s;
  s.image.layout = YuvLayout::V210;
  s.image.width = w;
  s.image.height = h;
  s.image.full_range = false;
  s.image.offset[0] = 0;
  s.image.stride[0] = v210::row_bytes(w);
  s.bytes = v210::pack(y, c, c, w, h, 0, s.image.stride[0]);
  s.image.device = on_device(gpu.device, gpu.allocator, s.bytes);
  CHECK(s.image.device != nullptr);
  s.luma.resize(y.size());
  int up = 0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    s.luma[i] = static_cast<std::uint8_t>(std::min(255, (y[i] + 2) / 4));
    up += (y[i] % 4 >= 2) ? 1 : 0;
  }
  auto luma = pass->run({&s.image});
  CHECK(luma.ok());
  CHECK(check_luma(*readback, (*luma)[0], s, "V210 codes") == 0);
  std::printf("  V210 codes: 0..1023 rounded once, %d up, 1020..1023 to 255\n",
              up);
  return 0;
}

// One run over every form, two pictures sharing one buffer among them (a
// side-by-side pair), is one submit; each output exact; and the list read of
// all the outputs equals each one's single read, byte for byte.
int test_list(const vkc::test::Gpu& gpu) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  std::vector<Source> sources;
  std::uint32_t seed = 40;
  for (const Form form : kForms) {
    ++seed;
    sources.push_back(
        make_source(gpu, form.layout, form.where, 38, 21 + seed % 3, seed));
    CHECK(sources.back().image.width != 0);
  }
  // Two luma pictures side by side in one buffer: its left and right halves.
  Source pair = make_source(gpu, YuvLayout::Luma, Where::Device, 64, 9, 77);
  CHECK(pair.image.width != 0);
  for (int half = 0; half < 2; ++half) {
    Source side;
    side.image = pair.image;
    side.image.width = 32;
    side.image.offset[0] += 32 * half;
    for (std::uint32_t r = 0; r < 9; ++r) {
      side.luma.insert(side.luma.end(), pair.luma.begin() + r * 64 + 32 * half,
                       pair.luma.begin() + r * 64 + 32 * half + 32);
    }
    sources.push_back(std::move(side));
  }
  std::vector<YuvImage*> pictures;
  for (Source& s : sources) pictures.push_back(&s.image);

  const vkc::DeviceCounters before = gpu.device.counters();
  auto luma = pass->run(pictures);
  const vkc::DeviceCounters after = gpu.device.counters();
  if (!luma) std::fprintf(stderr, "%s\n", luma.status().message().c_str());
  CHECK(luma.ok() && luma->size() == sources.size());
  CHECK(after.submits - before.submits == 1);
  for (std::size_t i = 0; i < sources.size(); ++i) {
    CHECK(check_luma(*readback, (*luma)[i], sources[i], "list") == 0);
  }

  // Every full and half luma in one list read: each view's bytes are its
  // single read's.
  std::vector<YuvImage*> outputs;
  for (DeviceLuma& l : *luma) {
    outputs.push_back(&l.full);
    outputs.push_back(&l.half);
  }
  auto listed = readback->read(outputs, Planes::Luma);
  CHECK(listed.ok() && listed->size() == outputs.size());
  std::vector<std::vector<std::uint8_t>> copies;
  for (const PictureView& v : *listed) {
    std::vector<std::uint8_t> rows;
    for (std::uint32_t r = 0; r < v.height; ++r) {
      rows.insert(rows.end(), v.luma + r * v.stride,
                  v.luma + r * v.stride + v.width);
    }
    copies.push_back(std::move(rows));
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto single = readback->read(*outputs[i], Planes::Luma);
    CHECK(single.ok());
    std::vector<std::uint8_t> rows;
    for (std::uint32_t r = 0; r < single->height; ++r) {
      rows.insert(rows.end(), single->luma + r * single->stride,
                  single->luma + r * single->stride + single->width);
    }
    CHECK(rows == copies[i]);
  }
  // An empty list makes nothing and submits nothing.
  const vkc::DeviceCounters idle = gpu.device.counters();
  auto none = pass->run({});
  CHECK(none.ok() && none->empty());
  CHECK(gpu.device.counters().submits == idle.submits);
  std::printf(
      "  list: %zu pictures, every form and a side-by-side pair, one "
      "submit; the list read is the single reads' bytes\n",
      sources.size());
  return 0;
}

// The device memory, summed over every heap.
std::uint64_t allocated(const vkc::Allocator& allocator) {
  const vkc::MemoryStats stats = allocator.memory_stats();
  std::uint64_t bytes = 0;
  for (std::uint32_t h = 0; h < stats.heap_count; ++h) {
    bytes += stats.heaps[h].allocation_bytes;
  }
  return bytes;
}

// The device bytes a storage buffer of `size` takes, as the pass makes its
// outputs; 0 if it cannot be made.
std::uint64_t allocation_of(vkc::Allocator& allocator, VkDeviceSize size) {
  const std::uint64_t before = allocated(allocator);
  auto made = vkc::device_storage_buffer(allocator, size);
  return made.ok() ? allocated(allocator) - before : 0;
}

// The buffers of `luma`'s outputs, in order.
std::vector<VkBuffer> buffers_of(const std::vector<DeviceLuma>& luma) {
  std::vector<VkBuffer> out;
  for (const DeviceLuma& l : luma) out.push_back(l.full.device->handle());
  return out;
}

// After warm-up a stream of runs, each dropped before the next, reuses the
// same buffers and allocates nothing, staging included. A picture no free
// buffer holds replaces the smallest free one rather than adding one. A kept
// output keeps its contents across the next run, which takes other buffers,
// and is reused once dropped; outputs kept past the pass stay readable.
int test_pool(const vkc::test::Gpu& gpu) {
  auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  Source a = make_source(gpu, YuvLayout::V210, Where::Device, 96, 8, 5);
  Source b = make_source(gpu, YuvLayout::Luma, Where::Host, 37, 21, 6);
  Source c = make_source(gpu, YuvLayout::Nv12, Where::Images, 37, 21, 7);
  CHECK(a.image.width != 0 && b.image.width != 0 && c.image.width != 0);
  const std::vector<YuvImage*> list = {&a.image, &b.image, &c.image};
  std::vector<VkBuffer> first;
  std::vector<VkDeviceSize> sizes;
  for (int warm = 0; warm < 3; ++warm) {
    auto luma = pass->run(list);
    CHECK(luma.ok());
    CHECK(check_luma(*readback, (*luma)[0], a, "pool") == 0);
    first = buffers_of(*luma);
    sizes.clear();
    for (const DeviceLuma& l : *luma) sizes.push_back(l.full.device->size());
  }
  const std::uint64_t before = allocated(gpu.allocator);
  const vkc::DeviceCounters counted = gpu.device.counters();
  for (int run = 0; run < 20; ++run) {
    auto luma = pass->run(list);
    CHECK(luma.ok() && buffers_of(*luma) == first);
  }
  CHECK(allocated(gpu.allocator) == before);
  CHECK(gpu.device.counters().staging_buffers == counted.staging_buffers);

  // A larger picture alone, the three free and none holding it: the smallest
  // is replaced, so the device bytes change by the new buffer's less its.
  {
    Source big = make_source(gpu, YuvLayout::Luma, Where::Device, 128, 64, 8);
    CHECK(big.image.width != 0);
    const VkDeviceSize smallest = *std::min_element(sizes.begin(), sizes.end());
    const std::uint64_t start = allocated(gpu.allocator);
    auto grown = pass->run({&big.image});
    CHECK(grown.ok());
    const std::uint64_t end = allocated(gpu.allocator);
    const VkDeviceSize made = (*grown)[0].full.device->size();
    CHECK(made > *std::max_element(sizes.begin(), sizes.end()));
    const std::uint64_t made_bytes = allocation_of(gpu.allocator, made);
    const std::uint64_t smallest_bytes = allocation_of(gpu.allocator, smallest);
    CHECK(made_bytes != 0 && smallest_bytes != 0);
    CHECK(end + smallest_bytes == start + made_bytes);
    CHECK(check_luma(*readback, (*grown)[0], big, "grown") == 0);
  }

  // Kept across the next run, whose pictures differ: the kept outputs still
  // hold the first pictures', the new ones are elsewhere and hold the new.
  auto kept = pass->run(list);
  CHECK(kept.ok());
  Source a2 = make_source(gpu, YuvLayout::V210, Where::Device, 96, 8, 15);
  Source b2 = make_source(gpu, YuvLayout::Luma, Where::Host, 37, 21, 16);
  Source c2 = make_source(gpu, YuvLayout::Nv12, Where::Images, 37, 21, 17);
  CHECK(a2.image.width != 0 && b2.image.width != 0 && c2.image.width != 0);
  {
    auto later = pass->run({&a2.image, &b2.image, &c2.image});
    CHECK(later.ok());
    for (const VkBuffer held : buffers_of(*kept)) {
      for (const VkBuffer made : buffers_of(*later)) CHECK(held != made);
    }
    CHECK(check_luma(*readback, (*later)[0], a2, "later") == 0);
    CHECK(check_luma(*readback, (*later)[1], b2, "later") == 0);
    CHECK(check_luma(*readback, (*later)[2], c2, "later") == 0);
  }
  CHECK(check_luma(*readback, (*kept)[0], a, "kept") == 0);
  CHECK(check_luma(*readback, (*kept)[1], b, "kept") == 0);
  CHECK(check_luma(*readback, (*kept)[2], c, "kept") == 0);
  // Copies of one picture's output hold it, the owner alone or a YuvImage.
  YuvImage half = (*kept)[1].half;
  const std::vector<VkBuffer> held = buffers_of(*kept);
  kept->clear();
  {
    auto again = pass->run(list);
    CHECK(again.ok());
    const std::vector<VkBuffer> now = buffers_of(*again);
    CHECK(now[1] != held[1]);
    auto view = readback->read(half, Planes::Luma);
    CHECK(view.ok());
    CHECK(rows_equal(*view, half_of(b.luma, 37, 21), 18, 10, "held half"));
  }
  // Outputs kept past the pass stay what they were.
  auto last = pass->run(list);
  CHECK(last.ok());
  pass.reset();
  CHECK(check_luma(*readback, (*last)[0], a, "past the pass") == 0);
  CHECK(check_luma(*readback, (*last)[2], c, "past the pass") == 0);
  std::printf(
      "  pool: reused after warm-up, nothing allocated; a larger picture "
      "replaces the smallest free buffer; kept outputs held across runs and "
      "past the pass\n");
  return 0;
}

// Planes in a buffer taken over from outside Vulkan, then named as the pass's
// own family after; images likewise.
int test_external(const vkc::test::Gpu& gpu) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  Source s = make_source(gpu, YuvLayout::V210, Where::Device, 38, 5, 9);
  Source i = make_source(gpu, YuvLayout::Nv12, Where::Images, 37, 5, 10);
  CHECK(s.image.width != 0 && i.image.width != 0);
  s.image.queue_family = sensor::kQueueFamilyExternal;
  i.image.queue_family = sensor::kQueueFamilyExternal;
  auto luma = pass->run({&s.image, &i.image});
  CHECK(luma.ok());
  CHECK(s.image.queue_family == gpu.device.queue_family());
  CHECK(i.image.queue_family == gpu.device.queue_family());
  CHECK(check_luma(*readback, (*luma)[0], s, "external") == 0);
  CHECK(check_luma(*readback, (*luma)[1], i, "external") == 0);
  // The updated pictures run again with nothing to take over.
  auto again = pass->run({&s.image, &i.image});
  CHECK(again.ok());
  CHECK(check_luma(*readback, (*again)[0], s, "external again") == 0);
  // One image pair named by two pictures, as the pass's family and as
  // kQueueFamilyIgnored: both mean nothing to transfer, so one writer.
  YuvImage ignored = i.image;
  ignored.queue_family = sensor::kQueueFamilyIgnored;
  auto both = pass->run({&i.image, &ignored});
  if (!both) std::fprintf(stderr, "%s\n", both.status().message().c_str());
  CHECK(both.ok());
  CHECK(check_luma(*readback, (*both)[0], i, "own family") == 0);
  CHECK(check_luma(*readback, (*both)[1], i, "ignored") == 0);
  std::printf(
      "  external: buffer and images taken over, ownership updated; an image "
      "named as own and as ignored is one writer\n");
  return 0;
}

// Two queue families: the outputs CONCURRENT across the pass's and a
// renderer's, read on the renderer's family; and an EXCLUSIVE picture the
// renderer's family wrote and released, taken over and named the pass's
// family after.
int test_families() {
  auto made =
      vkc::test::TwoFamilyDevice::create(sensor::device_requirements(), {});
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  vkc::test::TwoFamilyDevice& two = **made;
  if (!two.crosses_families()) {
    std::printf(
        "  families: one queue family, cross-family case skipped (%s)\n",
        two.shared().summary().c_str());
    return 0;
  }
  vkc::Device& compute = two.compute();
  vkc::Device& graphics = two.graphics();
  const auto pass =
      pass_on(compute, two.compute_allocator(), two.families().data(),
              vkc::test::TwoFamilyDevice::kFamilyCount);
  const auto reader = readback_on(graphics, two.graphics_allocator());
  CHECK(pass != nullptr && reader != nullptr);

  Source s = make_source(compute, two.compute_allocator(), YuvLayout::Luma,
                         Where::Host, 38, 7, 21);
  // A YUYV picture the renderer's family wrote, EXCLUSIVE, released to the
  // pass's family.
  Source w = make_source(compute, two.compute_allocator(), YuvLayout::Yuyv,
                         Where::Host, 38, 7, 22);
  auto buffer =
      vkc::device_storage_buffer(two.graphics_allocator(), w.bytes.size());
  CHECK(buffer.ok() && buffer->sharing_mode() == VK_SHARING_MODE_EXCLUSIVE);
  vkc::CommandBatch upload(graphics, two.graphics_allocator());
  CHECK(upload.upload(*buffer, 0, w.bytes.data(), w.bytes.size()).ok());
  CHECK(upload.release(*buffer, compute.queue_family()).ok());
  CHECK(upload.submit().ok());
  w.image.host = nullptr;
  w.image.device = std::make_shared<const vkc::Buffer>(std::move(*buffer));
  w.image.queue_family = graphics.queue_family();

  auto luma = pass->run({&s.image, &w.image});
  if (!luma) std::fprintf(stderr, "%s\n", luma.status().message().c_str());
  CHECK(luma.ok());
  CHECK(w.image.queue_family == compute.queue_family());
  for (std::size_t k = 0; k < 2; ++k) {
    DeviceLuma& l = (*luma)[k];
    CHECK(l.full.device->sharing_mode() == VK_SHARING_MODE_CONCURRENT);
    CHECK(check_luma(*reader, l, k == 0 ? s : w, "renderer's family") == 0);
  }
  auto again = pass->run({&w.image});
  CHECK(again.ok());
  CHECK(check_luma(*reader, (*again)[0], w, "again") == 0);
  std::printf(
      "  families: outputs CONCURRENT, read on the renderer's family; an "
      "EXCLUSIVE picture taken over\n");
  return 0;
}

// What a run refuses, each naming why, a list refused whole with nothing
// submitted, allocated or measured and its pictures as they were; the pass
// runs again afterwards. create's refusals.
int test_refusals(vkc::test::GpuContext& gpu) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  Source good = make_source(gpu, YuvLayout::I420, Where::Device, 37, 21, 31);
  Source v210 = make_source(gpu, YuvLayout::V210, Where::Device, 36, 4, 32);
  Source images = make_source(gpu, YuvLayout::Nv12, Where::Images, 37, 21, 33);
  CHECK(good.image.width != 0 && v210.image.width != 0 &&
        images.image.width != 0);
  // Warm, so a refusal that allocated would show.
  CHECK(pass->run({&good.image}).ok());
  const auto refused = [&](YuvImage picture, const char* why) {
    YuvImage first = good.image;
    first.queue_family = sensor::kQueueFamilyExternal;
    const std::uint64_t bytes = allocated(gpu.allocator);
    const std::uint64_t submits = gpu.device.counters().submits;
    vkc::StageMetrics metrics;
    const vkc::Status status = pass->run({&first, &picture}, &metrics).status();
    const bool ok = status.domain() == vkc::Status::Code::InvalidArgument &&
                    status.message().find(why) != std::string::npos &&
                    status.message().find("picture 1") != std::string::npos &&
                    first.queue_family == sensor::kQueueFamilyExternal &&
                    allocated(gpu.allocator) == bytes &&
                    gpu.device.counters().submits == submits &&
                    metrics.rows().empty();
    if (!ok) std::fprintf(stderr, "%s\n", status.message().c_str());
    return ok;
  };
  {
    YuvImage first = good.image;
    const vkc::Status status = pass->run({&first, nullptr}).status();
    CHECK(status.domain() == vkc::Status::Code::InvalidArgument &&
          status.message().find("null") != std::string::npos);
  }
  YuvImage bad = good.image;
  bad.width = 1;
  CHECK(refused(bad, "under 2x2"));
  bad = good.image;
  bad.height = 1;
  CHECK(refused(bad, "under 2x2"));
  YuvImage none;
  none.width = none.height = 4;
  CHECK(refused(none, "has no planes"));
  bad = good.image;
  bad.host = good.bytes.data();
  CHECK(refused(bad, "one of them"));
  bad = good.image;
  bad.layout = static_cast<YuvLayout>(9);
  CHECK(refused(bad, "unknown colour layout"));
  bad = v210.image;
  bad.width = 35;
  CHECK(refused(bad, "even"));
  bad = v210.image;
  bad.full_range = true;
  CHECK(refused(bad, "limited range"));
  bad = v210.image;
  bad.offset[0] = 2;
  CHECK(refused(bad, "multiples of 4"));
  bad = good.image;
  bad.stride[0] = 36;
  CHECK(refused(bad, "stride is shorter"));
  bad = good.image;
  bad.stride[2] = 1;  // the chroma, which the pass does not read, still checked
  CHECK(refused(bad, "stride is shorter"));
  bad = good.image;
  bad.offset[1] = bad.offset[0];  // Cb on the luma, which is read
  CHECK(refused(bad, "overlap"));
  bad = good.image;
  bad.offset[0] = good.image.device->size() - 10;
  CHECK(refused(bad, "runs past its buffer"));
  bad = good.image;
  bad.device = std::make_shared<const vkc::Buffer>();
  CHECK(refused(bad, "empty"));
  {
    vkc::BufferDesc transfer_only;
    transfer_only.size = good.image.device->size();
    transfer_only.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto unbound = gpu.allocator.create_buffer(transfer_only);
    CHECK(unbound.ok());
    bad = good.image;
    bad.device = std::make_shared<const vkc::Buffer>(std::move(*unbound));
    CHECK(refused(bad, "storage"));
  }
  bad = good.image;
  bad.queue_family =
      static_cast<std::uint32_t>(gpu.device.caps().queue_families().size());
  CHECK(refused(bad, "does not have"));
  bad = images.image;
  bad.image[1] = nullptr;
  CHECK(refused(bad, "NV12's two"));
  {
    // A luma image the pass cannot copy.
    const std::vector<std::uint8_t> luma(37 * 21, 1);
    bad = images.image;
    bad.image[0] = test_image::with_corner(
        gpu.device, gpu.allocator, VK_FORMAT_R8_UNORM, 1, 37, 21, luma.data(),
        37, 21, VK_IMAGE_USAGE_SAMPLED_BIT);
    CHECK(bad.image[0] != nullptr);
    CHECK(refused(bad, "TRANSFER_SRC usage"));
  }
  if (const auto other = vkc::test::other_queue_family(gpu.device)) {
    bad = images.image;
    bad.queue_family = *other;
    CHECK(refused(bad, "EXCLUSIVE images"));
  }
  {
    // Two pictures in one buffer naming two writers.
    YuvImage first = good.image;
    YuvImage second = good.image;
    first.queue_family = sensor::kQueueFamilyExternal;
    const vkc::Status status = pass->run({&first, &second}).status();
    CHECK(status.domain() == vkc::Status::Code::InvalidArgument &&
          status.message().find("two writers") != std::string::npos);
    CHECK(first.queue_family == sensor::kQueueFamilyExternal);
    // And two pictures of one image pair, one from outside Vulkan.
    YuvImage outside = images.image;
    YuvImage inside = images.image;
    outside.queue_family = sensor::kQueueFamilyExternal;
    const vkc::Status image_status = pass->run({&inside, &outside}).status();
    CHECK(image_status.domain() == vkc::Status::Code::InvalidArgument &&
          image_status.message().find("two writers") != std::string::npos);
    CHECK(outside.queue_family == sensor::kQueueFamilyExternal);
  }

  // Refusals leave the pass running.
  auto luma = pass->run({&good.image, &images.image});
  CHECK(luma.ok());
  CHECK(check_luma(*readback, (*luma)[0], good, "after refusals") == 0);
  CHECK(check_luma(*readback, (*luma)[1], images, "after refusals") == 0);

  // create: an empty device or allocator, and families it cannot share with.
  auto bare = vs_test::bare_device(gpu);
  CHECK(bare.ok());
  const vkc::Device taken = std::move(*bare);
  CHECK(LumaPass::create(*bare, gpu.allocator).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  auto spare = vkc::Allocator::create(gpu.instance.handle(), gpu.device);
  CHECK(spare.ok());
  const vkc::Allocator kept = std::move(*spare);
  CHECK(LumaPass::create(gpu.device, *spare).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(taken.handle() == gpu.device.handle() && kept.valid());
  CHECK(LumaPass::create(gpu.device, gpu.allocator, nullptr, 1)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  const std::uint32_t missing =
      static_cast<std::uint32_t>(gpu.device.caps().queue_families().size());
  const std::uint32_t with_missing[2] = {gpu.device.queue_family(), missing};
  CHECK(LumaPass::create(gpu.device, gpu.allocator, with_missing, 2)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  if (const auto other = vkc::test::other_queue_family(gpu.device)) {
    CHECK(LumaPass::create(gpu.device, gpu.allocator, &*other, 1)
              .status()
              .domain() == vkc::Status::Code::InvalidArgument);
  }
  std::printf("  refusals: each named, a list refused whole\n");
  return 0;
}

// A measured run adds one "luma" row, its device half inside its host half;
// an unmeasured one adds none.
int test_metrics(const vkc::test::Gpu& gpu) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr);
  Source a = make_source(gpu, YuvLayout::V210, Where::Device, 96, 16, 51);
  Source b = make_source(gpu, YuvLayout::Nv12, Where::Images, 64, 48, 52);
  CHECK(a.image.width != 0 && b.image.width != 0);
  vkc::StageMetrics metrics;
  CHECK(pass->run({&a.image, &b.image}, &metrics).ok());
  CHECK(metrics.rows().size() == 1);
  const vkc::StageRow& row = metrics.rows()[0];
  CHECK(std::strcmp(row.name, "luma") == 0 && row.cpu_ms > 0.0);
  if (gpu.device.timestamp_valid_bits() != 0) {
    CHECK(row.has_gpu && row.gpu_ms > 0.0 && row.gpu_ms < row.cpu_ms);
  }
  CHECK(pass->run({&a.image, &b.image}).ok());
  CHECK(metrics.rows().size() == 1);
  std::printf("  metrics: one row, %.3f ms host, %.3f ms device\n", row.cpu_ms,
              row.gpu_ms);
  return 0;
}

// `sorted`'s nearest-rank percentile `p`.
double percentile(const std::vector<double>& sorted, double p) {
  const std::size_t rank = static_cast<std::size_t>(
      std::max(1.0, std::ceil(p / 100.0 * static_cast<double>(sorted.size()))));
  return sorted[std::min(rank, sorted.size()) - 1];
}

// By hand (VS_TEST_LUMA_TIMING=<runs>): a DeckLink pair's pictures, two
// 3840x2160 V210 in buffers, run after 50 warm-up runs; the pass's device
// span and host span, and the readback of both lumas and halves after it,
// as p50 and p95.
int time_pair(const vkc::test::GpuContext& gpu, int runs) {
  const auto pass = pass_on(gpu.device, gpu.allocator);
  const auto readback = readback_on(gpu.device, gpu.allocator);
  CHECK(pass != nullptr && readback != nullptr);
  Source left = make_source(gpu, YuvLayout::V210, Where::Device, 3840, 2160, 1);
  Source right =
      make_source(gpu, YuvLayout::V210, Where::Device, 3840, 2160, 2);
  CHECK(left.image.width != 0 && right.image.width != 0);
  std::vector<double> device_ms, host_ms, read_ms;
  for (int i = -50; i < runs; ++i) {
    vkc::StageMetrics metrics;
    auto luma = pass->run({&left.image, &right.image}, &metrics);
    CHECK(luma.ok());
    std::vector<YuvImage*> outputs;
    for (DeviceLuma& l : *luma) {
      outputs.push_back(&l.full);
      outputs.push_back(&l.half);
    }
    CHECK(readback->read(outputs, Planes::Luma, &metrics).ok());
    if (i < 0) continue;
    CHECK(metrics.rows().size() == 2);
    device_ms.push_back(metrics.rows()[0].gpu_ms);
    host_ms.push_back(metrics.rows()[0].cpu_ms);
    read_ms.push_back(metrics.rows()[1].cpu_ms);
  }
  for (std::vector<double>* v : {&device_ms, &host_ms, &read_ms}) {
    std::sort(v->begin(), v->end());
  }
  std::printf(
      "  timing (%s, %d runs): run() device p50 %.3f ms, p95 %.3f ms; host "
      "p50 %.3f, p95 %.3f; readback host p50 %.3f, p95 %.3f\n",
      gpu.physical.properties().deviceName, runs, percentile(device_ms, 50),
      percentile(device_ms, 95), percentile(host_ms, 50),
      percentile(host_ms, 95), percentile(read_ms, 50),
      percentile(read_ms, 95));
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  if (const char* runs = std::getenv("VS_TEST_LUMA_TIMING")) {
    CHECK(time_pair(gpu, std::max(1, std::atoi(runs))) == 0);
    std::puts("sensor_luma_pass: timing OK");
    return 0;
  }
  CHECK(test_forms(gpu) == 0);
  CHECK(test_v210_codes(gpu) == 0);
  CHECK(test_list(gpu) == 0);
  CHECK(test_pool(gpu) == 0);
  CHECK(test_external(gpu) == 0);
  CHECK(test_families() == 0);
  CHECK(test_refusals(gpu) == 0);
  CHECK(test_metrics(gpu) == 0);
  std::puts("sensor_luma_pass: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
