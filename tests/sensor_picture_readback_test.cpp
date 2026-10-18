// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// PictureReadback (recon_sensor_utils) on synthetic pictures, no decoder:
// NV12 plane images larger than their picture, and NV12 and I420 planes in
// buffers at odd sizes, offsets and strides, read back byte for byte, luma
// alone or every plane, one picture or a set; buffers taken over from the
// readback's own family, a CONCURRENT buffer's other family and from outside
// Vulkan; the host memory grown and reused; views that outlive their
// pictures; every refusal; and the stage row. Skips without a device.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "gpu_test.hpp"
#include "test_check.hpp"
#include "test_image.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/testing/two_family_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"
#include "volumetric_kit/recon/sensor/utils/picture_readback.hpp"
#include "volumetric_kit/recon/sensor/yuv_image.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = vr::sensor;
using sensor::PictureReadback;
using sensor::PictureView;
using sensor::YuvImage;
using sensor::YuvLayout;
using Planes = PictureReadback::Planes;

// Made by create and held where it was made: never copied or moved.
static_assert(!std::is_copy_constructible_v<PictureReadback> &&
                  !std::is_copy_assignable_v<PictureReadback> &&
                  !std::is_move_constructible_v<PictureReadback> &&
                  !std::is_move_assignable_v<PictureReadback>,
              "PictureReadback is an aggregate");

namespace {

constexpr std::uint8_t kJunk = 0xEE;

// A picture's planes, rows packed: Y w x h, Cb and Cr cw x ch, each byte
// differing from its neighbours' and from another seed's.
struct Packed {
  std::uint32_t w = 0, h = 0, cw = 0, ch = 0;
  std::vector<std::uint8_t> y, cb, cr;

  // Cb and Cr interleaved, as NV12 holds them.
  std::vector<std::uint8_t> cbcr() const {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i < cb.size(); ++i) {
      out.push_back(cb[i]);
      out.push_back(cr[i]);
    }
    return out;
  }
};

Packed pattern(std::uint32_t w, std::uint32_t h, int seed) {
  Packed p;
  p.w = w;
  p.h = h;
  p.cw = (w + 1) / 2;
  p.ch = (h + 1) / 2;
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      p.y.push_back(static_cast<std::uint8_t>(x * 7 + y * 13 + seed * 31));
    }
  }
  for (std::uint32_t y = 0; y < p.ch; ++y) {
    for (std::uint32_t x = 0; x < p.cw; ++x) {
      p.cb.push_back(
          static_cast<std::uint8_t>(x * 11 + y * 3 + seed * 17 + 64));
      p.cr.push_back(
          static_cast<std::uint8_t>(x * 5 + y * 19 + seed * 23 + 128));
    }
  }
  return p;
}

// `rows` rows of `row_bytes` at `at`, `stride` apart, equal `want`'s packed
// rows.
bool rows_equal(const std::uint8_t* at, std::size_t stride,
                const std::uint8_t* want, std::size_t row_bytes,
                std::size_t rows) {
  for (std::size_t r = 0; r < rows; ++r) {
    if (std::memcmp(at + r * stride, want + r * row_bytes, row_bytes) != 0) {
      return false;
    }
  }
  return true;
}

// `v` holds `want`'s planes exactly: luma alone, or every plane in `layout`.
int check_view(const PictureView& v, const Packed& want, YuvLayout layout,
               Planes planes) {
  CHECK(v.width == want.w && v.height == want.h);
  CHECK(v.luma != nullptr && v.stride >= v.width);
  CHECK(rows_equal(v.luma, v.stride, want.y.data(), want.w, want.h));
  CHECK(v.layout == layout);
  if (planes == Planes::Luma) {
    CHECK(v.chroma[0] == nullptr && v.chroma[1] == nullptr);
    return 0;
  }
  if (layout == YuvLayout::Nv12) {
    const std::vector<std::uint8_t> cbcr = want.cbcr();
    CHECK(v.chroma[0] != nullptr && v.chroma[1] == nullptr);
    CHECK(v.chroma_stride[0] >= 2 * std::size_t{want.cw});
    CHECK(rows_equal(v.chroma[0], v.chroma_stride[0], cbcr.data(),
                     2 * std::size_t{want.cw}, want.ch));
    return 0;
  }
  CHECK(v.chroma[0] != nullptr && v.chroma[1] != nullptr);
  CHECK(rows_equal(v.chroma[0], v.chroma_stride[0], want.cb.data(), want.cw,
                   want.ch));
  CHECK(rows_equal(v.chroma[1], v.chroma_stride[1], want.cr.data(), want.cw,
                   want.ch));
  return 0;
}

// `bytes` in a device storage buffer shared among `families` (exclusive for
// none), as a decoder leaves its picture; null if it cannot be made.
std::shared_ptr<const vkc::Buffer> on_device(
    const vr_test::Gpu& gpu, const std::vector<std::uint8_t>& bytes,
    const std::uint32_t* families = nullptr, std::uint32_t family_count = 0) {
  auto made = vkc::device_storage_buffer(gpu.allocator, bytes.size(), 0,
                                         families, family_count);
  if (!made.ok() ||
      !vr_test::write_back(gpu.device, gpu.allocator, *made, bytes).ok()) {
    return nullptr;
  }
  return std::make_shared<const vkc::Buffer>(std::move(made).value());
}

// `p`'s planes as a decoder leaves them in a buffer, `layout`'s, laid out in
// the bytes returned: after a lead longer than any alignment, each plane at
// an odd offset, its rows padded past the plane's to odd strides, the padding
// junk. `image` describes them, all but its buffer.
std::vector<std::uint8_t> laid_out(const Packed& p, YuvLayout layout,
                                   YuvImage& image) {
  std::vector<std::uint8_t> blob(4096 + 1, kJunk);
  const auto place = [&blob](const std::uint8_t* rows, std::size_t row_bytes,
                             std::size_t count, std::size_t stride) {
    blob.resize(blob.size() + 3, kJunk);
    const std::size_t at = blob.size();
    blob.resize(at + stride * count, kJunk);
    for (std::size_t r = 0; r < count; ++r) {
      std::memcpy(&blob[at + r * stride], rows + r * row_bytes, row_bytes);
    }
    return static_cast<std::uint64_t>(at);
  };
  image = YuvImage{};
  image.layout = layout;
  image.width = p.w;
  image.height = p.h;
  image.stride[0] = p.w + 5;
  image.offset[0] = place(p.y.data(), p.w, p.h, image.stride[0]);
  if (layout == YuvLayout::Nv12) {
    const std::vector<std::uint8_t> cbcr = p.cbcr();
    image.stride[1] = 2 * std::size_t{p.cw} + 3;
    image.offset[1] = place(cbcr.data(), 2 * p.cw, p.ch, image.stride[1]);
  } else {
    image.stride[1] = p.cw + 1;
    image.offset[1] = place(p.cb.data(), p.cw, p.ch, image.stride[1]);
    image.stride[2] = p.cw + 6;
    image.offset[2] = place(p.cr.data(), p.cw, p.ch, image.stride[2]);
  }
  return blob;
}

// `p` in a buffer, as laid_out places it; no buffer if it cannot be made.
YuvImage in_buffer(const vr_test::Gpu& gpu, const Packed& p, YuvLayout layout,
                   const std::uint32_t* families = nullptr,
                   std::uint32_t family_count = 0) {
  YuvImage image;
  image.device =
      on_device(gpu, laid_out(p, layout, image), families, family_count);
  return image;
}

// `p` as VideoToolbox leaves it: NV12's planes as images 3 texels wider and
// 2 rows taller than the planes, junk in the rest; no images if they cannot
// be made.
YuvImage as_images(const vr_test::Gpu& gpu, const Packed& p) {
  const std::vector<std::uint8_t> cbcr = p.cbcr();
  YuvImage image;
  image.layout = YuvLayout::Nv12;
  image.width = p.w;
  image.height = p.h;
  image.image[0] =
      test_image::with_corner(gpu.device, gpu.allocator, VK_FORMAT_R8_UNORM, 1,
                              p.w + 3, p.h + 2, p.y.data(), p.w, p.h);
  image.image[1] = test_image::with_corner(
      gpu.device, gpu.allocator, VK_FORMAT_R8G8_UNORM, 2, p.cw + 3, p.ch + 2,
      cbcr.data(), 2 * p.cw, p.ch);
  return image;
}

bool made(const YuvImage& image) {
  return image.device != nullptr ||
         (image.image[0] != nullptr && image.image[1] != nullptr);
}

std::unique_ptr<PictureReadback> readback_on(const vr_test::Gpu& gpu) {
  auto made = PictureReadback::create(gpu.device, gpu.allocator);
  return made.ok() ? std::move(made).value() : nullptr;
}

// Each layout, at odd and even sizes and a single pixel, read alone, luma
// alone and every plane: exact, rows at the buffer's strides or packed from
// the images' corner; the range passed through.
int test_layouts(const vr_test::Gpu& gpu) {
  const auto readback = readback_on(gpu);
  CHECK(readback != nullptr);
  struct Size {
    std::uint32_t w, h;
  };
  int seed = 0;
  for (const Size size : {Size{37, 21}, Size{64, 48}, Size{1, 1}}) {
    const Packed p = pattern(size.w, size.h, ++seed);
    YuvImage pictures[3] = {in_buffer(gpu, p, YuvLayout::Nv12),
                            in_buffer(gpu, p, YuvLayout::I420),
                            as_images(gpu, p)};
    for (YuvImage& picture : pictures) {
      CHECK(made(picture));
      picture.full_range = seed % 2 == 0;
      for (const Planes planes : {Planes::Luma, Planes::All}) {
        auto view = readback->read(picture, planes);
        if (!view) {
          std::fprintf(stderr, "%s\n", view.status().message().c_str());
        }
        CHECK(view.ok());
        CHECK(check_view(*view, p, picture.layout, planes) == 0);
        CHECK(view->full_range == picture.full_range);
        // Rows as the buffer had them, or packed from an image.
        CHECK(view->stride ==
              (picture.device ? picture.stride[0] : std::size_t{p.w}));
      }
    }
  }
  std::printf("  layouts: NV12 and I420 buffers and NV12 images exact\n");
  return 0;
}

// A set: a view per frame that has a picture on the device, every picture
// exact in one read; empty where the frame is missing, has no colour, or has
// it as host words.
int test_set(const vr_test::Gpu& gpu) {
  const auto readback = readback_on(gpu);
  CHECK(readback != nullptr);
  const Packed a = pattern(37, 21, 7), b = pattern(64, 48, 8),
               c = pattern(33, 17, 9);
  const std::uint32_t words[4] = {};
  std::vector<std::optional<sensor::RgbdFrame>> frames(6);
  frames[0].emplace().color = as_images(gpu, a);
  frames[2].emplace();  // no colour
  frames[3].emplace().color_packed = words;
  frames[4].emplace().color = in_buffer(gpu, b, YuvLayout::I420);
  frames[5].emplace().color = in_buffer(gpu, c, YuvLayout::Nv12);
  CHECK(made(frames[0]->color) && made(frames[4]->color) &&
        made(frames[5]->color));
  for (const Planes planes : {Planes::Luma, Planes::All}) {
    auto views = readback->read(frames, planes);
    CHECK(views.ok() && views->size() == frames.size());
    const std::vector<std::optional<PictureView>>& v = *views;
    CHECK(v[0] && !v[1] && !v[2] && !v[3] && v[4] && v[5]);
    CHECK(check_view(*v[0], a, YuvLayout::Nv12, planes) == 0);
    CHECK(check_view(*v[4], b, YuvLayout::I420, planes) == 0);
    CHECK(check_view(*v[5], c, YuvLayout::Nv12, planes) == 0);
  }
  // A set with no picture on the device reads nothing.
  std::vector<std::optional<sensor::RgbdFrame>> none(2);
  none[1].emplace();
  auto empty = readback->read(none, Planes::Luma);
  CHECK(empty.ok() && empty->size() == 2 && !(*empty)[0] && !(*empty)[1]);
  std::printf("  set: three frames without a device picture, three exact\n");
  return 0;
}

// Planes in a buffer the readback's own family wrote, named as ignored or as
// that family; a CONCURRENT buffer another family wrote; and one an API
// outside Vulkan wrote: the same bytes.
int test_families(const vr_test::Gpu& gpu) {
  const auto readback = readback_on(gpu);
  CHECK(readback != nullptr);
  const Packed p = pattern(37, 21, 3);
  YuvImage picture = in_buffer(gpu, p, YuvLayout::Nv12);
  CHECK(made(picture));
  const std::uint32_t own = gpu.device.queue_family();
  for (const std::uint32_t family :
       {sensor::kQueueFamilyIgnored, own, sensor::kQueueFamilyExternal}) {
    picture.queue_family = family;
    auto view = readback->read(picture, Planes::All);
    CHECK(view.ok());
    CHECK(check_view(*view, p, YuvLayout::Nv12, Planes::All) == 0);
    CHECK(picture.queue_family == own);
  }
  if (const auto other = vr_test::other_queue_family(gpu.device)) {
    const std::uint32_t shared[2] = {own, *other};
    YuvImage concurrent = in_buffer(gpu, p, YuvLayout::I420, shared, 2);
    CHECK(made(concurrent));
    CHECK(concurrent.device->sharing_mode() == VK_SHARING_MODE_CONCURRENT);
    concurrent.queue_family = shared[1];
    auto view = readback->read(concurrent, Planes::All);
    CHECK(view.ok());
    CHECK(check_view(*view, p, YuvLayout::I420, Planes::All) == 0);
    CHECK(concurrent.queue_family == sensor::kQueueFamilyIgnored);
  } else {
    std::printf("  families: one queue family, no CONCURRENT buffer\n");
  }
  // A family the device lacks.
  picture.queue_family =
      static_cast<std::uint32_t>(gpu.device.caps().queue_families().size());
  CHECK(readback->read(picture, Planes::Luma).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  std::printf("  families: own, CONCURRENT and external exact\n");
  return 0;
}

// An EXCLUSIVE buffer released by another family stays usable after either
// read overload: preparation and a repeated read must not acquire it again.
int test_read_then_prepare() {
  auto made = vkc::test::TwoFamilyDevice::create(vr::device_requirements(), {});
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  vkc::test::TwoFamilyDevice& two = **made;
  if (!two.crosses_families()) {
    std::printf("  read then prepare: cross-family case skipped (%s)\n",
                two.shared().summary().c_str());
    return 0;
  }
  vkc::Device& reader = two.compute();
  vkc::Device& writer = two.graphics();
  vkc::Allocator& allocator = two.compute_allocator();
  auto readback = PictureReadback::create(reader, allocator);
  auto prep = sensor::GpuFramePrep::create(reader, allocator);
  CHECK(readback.ok() && prep.ok());
  const std::vector<std::uint8_t> bytes(96, 128);
  const std::vector<std::uint16_t> depth(64, 1000);
  for (const bool single : {true, false}) {
    auto buffer = vkc::device_storage_buffer(allocator, bytes.size());
    CHECK(buffer.ok());
    CHECK(buffer->sharing_mode() == VK_SHARING_MODE_EXCLUSIVE);
    vkc::CommandBatch upload(writer, two.graphics_allocator());
    CHECK(upload.upload(*buffer, 0, bytes.data(), bytes.size()).ok());
    CHECK(upload.release(*buffer, reader.queue_family()).ok());
    CHECK(upload.submit().ok());

    std::vector<std::optional<sensor::RgbdFrame>> frames(3);
    sensor::RgbdFrame& frame = frames[1].emplace();
    frame.depth = depth.data();
    frame.depth_camera = {{8, 8}, {8.0, 8.0, 3.5, 3.5}, {}};
    frame.color_camera = frame.depth_camera;
    frame.min_depth = 0.1f;
    frame.max_depth = 4.0f;
    frame.color.layout = YuvLayout::Nv12;
    frame.color.width = frame.color.height = 8;
    frame.color.stride[0] = frame.color.stride[1] = 8;
    frame.color.offset[1] = 64;
    frame.color.queue_family = writer.queue_family();
    frame.color.device =
        std::make_shared<const vkc::Buffer>(std::move(*buffer));

    // Validation failures leave ownership with the writer, so the valid
    // picture can still consume its pending release on the next read.
    frame.color.stride[0] = 1;
    const auto refused =
        single ? (*readback)->read(frame.color, Planes::Luma).status()
               : (*readback)->read(frames, Planes::Luma).status();
    CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
    CHECK(frame.color.queue_family == writer.queue_family());
    frame.color.stride[0] = 8;
    if (single) {
      auto view = (*readback)->read(frame.color, Planes::Luma);
      CHECK(view.ok() && std::memcmp(view->luma, bytes.data(), 64) == 0);
    } else {
      auto views = (*readback)->read(frames, Planes::Luma);
      CHECK(views.ok() && views->size() == 3);
      CHECK(!(*views)[0] && (*views)[1] && !(*views)[2]);
      CHECK(std::memcmp((*views)[1]->luma, bytes.data(), 64) == 0);
    }
    auto prepared = prep->prepare(frame);
    CHECK(prepared.ok() && prepared->has_color());
    auto view = (*readback)->read(frame.color, Planes::All);
    CHECK(view.ok() && std::memcmp(view->luma, bytes.data(), 64) == 0);
    CHECK(std::memcmp(view->chroma[0], bytes.data() + 64, 32) == 0);
    CHECK(frame.color.queue_family == reader.queue_family());
  }
  std::puts(
      "  read then prepare: both overloads, two families, repeated reads");
  return 0;
}

// The host memory, summed over every heap.
std::uint64_t allocated(const vkc::Allocator& allocator) {
  const vkc::MemoryStats stats = allocator.memory_stats();
  std::uint64_t bytes = 0;
  for (std::uint32_t h = 0; h < stats.heap_count; ++h) {
    bytes += stats.heaps[h].allocation_bytes;
  }
  return bytes;
}

// The views are copies the readback holds: a picture rewritten or dropped
// after the read leaves them as they were. The memory grows for a larger read
// and is reused by every read that fits: a steady stream allocates nothing.
int test_memory(const vr_test::Gpu& gpu) {
  const auto readback = readback_on(gpu);
  CHECK(readback != nullptr);
  const Packed small = pattern(37, 21, 4), large = pattern(96, 64, 5);
  YuvImage s = in_buffer(gpu, small, YuvLayout::I420);
  YuvImage l = in_buffer(gpu, large, YuvLayout::Nv12);
  CHECK(made(s) && made(l));

  auto first = readback->read(s, Planes::All);
  CHECK(first.ok());
  // Rewritten, then dropped: the view still holds what was read.
  std::vector<std::uint8_t> zeros(s.device->size(), 0);
  CHECK(vr_test::write_back(gpu.device, gpu.allocator, *s.device, zeros).ok());
  s.device.reset();
  CHECK(check_view(*first, small, YuvLayout::I420, Planes::All) == 0);
  s = in_buffer(gpu, small, YuvLayout::I420);
  CHECK(made(s));

  // The same read again: the same memory, nothing allocated.
  const std::uint64_t before = allocated(gpu.allocator);
  auto again = readback->read(s, Planes::All);
  CHECK(again.ok() && again->luma == first->luma);
  CHECK(allocated(gpu.allocator) == before);
  CHECK(check_view(*again, small, YuvLayout::I420, Planes::All) == 0);

  // Larger: grown, the old memory freed first.
  auto grown = readback->read(l, Planes::All);
  CHECK(grown.ok());
  CHECK(check_view(*grown, large, YuvLayout::Nv12, Planes::All) == 0);
  const std::uint64_t after_growth = allocated(gpu.allocator);
  CHECK(after_growth > before);

  // Smaller again, and luma alone: kept, nothing allocated or freed.
  auto kept = readback->read(s, Planes::Luma);
  CHECK(kept.ok() && kept->luma == grown->luma);
  CHECK(allocated(gpu.allocator) == after_growth);
  CHECK(check_view(*kept, small, YuvLayout::I420, Planes::Luma) == 0);
  auto large_again = readback->read(l, Planes::All);
  CHECK(large_again.ok());
  CHECK(check_view(*large_again, large, YuvLayout::Nv12, Planes::All) == 0);
  CHECK(allocated(gpu.allocator) == after_growth);
  std::printf("  memory: grown once, reused, views outlive their pictures\n");
  return 0;
}

// What a read refuses, each naming why, a set refused whole; the readback
// reads again afterwards. create refuses an empty device or allocator.
int test_refusals(vr_test::GpuContext& gpu) {
  const auto readback = readback_on(gpu);
  CHECK(readback != nullptr);
  const Packed p = pattern(37, 21, 6);
  const YuvImage buffer = in_buffer(gpu, p, YuvLayout::I420);
  YuvImage images = as_images(gpu, p);
  CHECK(made(buffer) && made(images));
  const auto refused = [&](YuvImage picture, const char* why) {
    const vkc::Status status = readback->read(picture, Planes::Luma).status();
    const bool ok = status.domain() == vkc::Status::Code::InvalidArgument &&
                    status.message().find(why) != std::string::npos;
    if (!ok) std::fprintf(stderr, "%s\n", status.message().c_str());
    return ok;
  };

  CHECK(refused(YuvImage{}, "not on the device"));
  YuvImage bad = buffer;
  bad.width = 0;
  CHECK(refused(bad, "is empty"));
  bad = buffer;
  bad.image[0] = images.image[0];
  bad.image[1] = images.image[1];
  CHECK(refused(bad, "not both"));
  bad = buffer;
  bad.device = std::make_shared<const vkc::Buffer>();
  CHECK(refused(bad, "buffer is empty"));
  vkc::BufferDesc storage_only;
  storage_only.size = buffer.device->size();
  storage_only.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  auto unreadable = gpu.allocator.create_buffer(storage_only);
  CHECK(unreadable.ok());
  bad.device =
      std::make_shared<const vkc::Buffer>(std::move(unreadable).value());
  CHECK(refused(bad, "buffer needs TRANSFER_SRC"));
  bad = buffer;
  bad.stride[2] = p.cw - 1;
  CHECK(refused(bad, "stride is shorter"));
  bad = buffer;
  bad.offset[2] = buffer.device->size() - bad.stride[2];  // last rows past it
  CHECK(refused(bad, "runs past its buffer"));
  bad = buffer;
  bad.offset[0] = bad.offset[1] = bad.offset[2] = 0;
  CHECK(refused(bad, "overlap"));

  bad = images;
  bad.layout = YuvLayout::I420;
  CHECK(refused(bad, "NV12's two"));
  bad = images;
  bad.image[1] = nullptr;
  CHECK(refused(bad, "NV12's two"));
  bad = images;
  std::swap(bad.image[0], bad.image[1]);
  CHECK(refused(bad, "at least the picture's size"));
  bad = images;
  bad.width = p.w + 4;  // past the luma image
  CHECK(refused(bad, "at least the picture's size"));
  bad = images;
  const std::vector<std::uint8_t> luma(std::size_t{p.w} * p.h, 1);
  bad.image[0] =
      test_image::with_corner(gpu.device, gpu.allocator, VK_FORMAT_R8_UNORM, 1,
                              p.w, p.h, luma.data(), p.w, p.h, 0);
  CHECK(bad.image[0] != nullptr);
  CHECK(refused(bad, "TRANSFER_SRC usage"));
  bad = images;
  vkc::ImageInfo shader_read = images.image[1]->info();
  shader_read.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  bad.image[1] =
      std::make_shared<const vkc::Image>(shader_read, nullptr);  // borrowed
  CHECK(refused(bad, "TRANSFER_SRC_OPTIMAL"));

  // One bad picture refuses the set, measured or not.
  std::vector<std::optional<sensor::RgbdFrame>> frames(2);
  frames[0].emplace().color = images;
  frames[1].emplace().color = buffer;
  frames[1]->color.stride[0] = 1;
  vkc::StageMetrics metrics;
  CHECK(readback->read(frames, Planes::Luma, &metrics).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(metrics.rows().empty());

  // Refusals leave the readback reading.
  auto view = readback->read(images, Planes::All);
  CHECK(view.ok());
  CHECK(check_view(*view, p, YuvLayout::Nv12, Planes::All) == 0);

  // An empty device or allocator.
  auto bare = vr_test::bare_device(gpu);
  CHECK(bare.ok());
  const vkc::Device taken = std::move(*bare);
  CHECK(PictureReadback::create(*bare, gpu.allocator).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  auto spare = vkc::Allocator::create(gpu.instance.handle(), gpu.device);
  CHECK(spare.ok());
  const vkc::Allocator kept = std::move(*spare);
  CHECK(PictureReadback::create(gpu.device, *spare).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(taken.handle() == gpu.device.handle() && kept.valid());
  std::printf("  refusals: each named, a set refused whole\n");
  return 0;
}

// A measured read adds one "readback" row, its device half inside its host
// half; an unmeasured one adds none.
int test_metrics(const vr_test::Gpu& gpu) {
  const auto readback = readback_on(gpu);
  CHECK(readback != nullptr);
  const Packed p = pattern(64, 48, 2);
  std::vector<std::optional<sensor::RgbdFrame>> frames(2);
  frames[0].emplace().color = as_images(gpu, p);
  frames[1].emplace().color = in_buffer(gpu, p, YuvLayout::I420);
  CHECK(made(frames[0]->color) && made(frames[1]->color));
  vkc::StageMetrics metrics;
  CHECK(readback->read(frames, Planes::All, &metrics).ok());
  CHECK(metrics.rows().size() == 1);
  const vkc::StageRow& row = metrics.rows()[0];
  CHECK(std::strcmp(row.name, "readback") == 0 && row.cpu_ms > 0.0);
  if (gpu.device.timestamp_valid_bits() != 0) {
    CHECK(row.has_gpu && row.gpu_ms > 0.0 && row.gpu_ms < row.cpu_ms);
  }
  CHECK(readback->read(frames, Planes::All).ok());
  CHECK(metrics.rows().size() == 1);
  std::printf("  metrics: one row, %.3f ms host, %.3f ms device\n", row.cpu_ms,
              row.gpu_ms);
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  CHECK(test_layouts(gpu) == 0);
  CHECK(test_set(gpu) == 0);
  CHECK(test_families(gpu) == 0);
  CHECK(test_read_then_prepare() == 0);
  CHECK(test_memory(gpu) == 0);
  CHECK(test_refusals(gpu) == 0);
  CHECK(test_metrics(gpu) == 0);
  std::puts("sensor_picture_readback: OK");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
