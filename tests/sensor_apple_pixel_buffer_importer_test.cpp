// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// PixelBufferImporter on pixel buffers made here, as a camera hands them out,
// with no decoder and no FFmpeg: an NV12 IOSurface's planes reach Vulkan as
// images without a copy -- they read what the surface holds when they are
// read, not when it was imported -- at the picture's size from the buffer's
// corner, each surface's images made once; the images hold the buffer; what
// the images allocate in Metal; the picture prepares as a frame's colour by
// the description its caller gives; the images' sharing across queue
// families, and a renderer's family copying them on a device shared with
// recon; what it refuses; and moves.
//
// Skips where no Vulkan device is present, unless VKC_REQUIRE_VULKAN_DEVICE
// says this machine has one.

#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "device_picture_readback.hpp"
#include "gpu_test.hpp"
#include "metal_allocation.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/testing/two_family_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/apple/pixel_buffer_importer.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"
#include "yuv_reference.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::PixelBufferImporter;
using sensor::VideoColorMatrix;
using sensor::YuvImage;

namespace {

constexpr OSType kFull = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
constexpr OSType kVideo = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;

using Gpu = vr_test::GpuContext;

// A pixel buffer, released with it.
struct Pixels {
  CVPixelBufferRef ref = nullptr;
  Pixels() = default;
  explicit Pixels(CVPixelBufferRef r) : ref(r) {}
  ~Pixels() {
    if (ref != nullptr) CVPixelBufferRelease(ref);
  }
  Pixels(const Pixels&) = delete;
  Pixels& operator=(const Pixels&) = delete;
};

// A `width` x `height` buffer of `format`, on an IOSurface Metal reads unless
// `surface` is false.
CVPixelBufferRef make_pixels(std::size_t width, std::size_t height,
                             OSType format, bool surface = true) {
  CFMutableDictionaryRef attributes = CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CFDictionaryRef none = CFDictionaryCreate(
      kCFAllocatorDefault, nullptr, nullptr, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  if (surface) {
    CFDictionarySetValue(attributes, kCVPixelBufferIOSurfacePropertiesKey,
                         none);
    CFDictionarySetValue(attributes, kCVPixelBufferMetalCompatibilityKey,
                         kCFBooleanTrue);
  }
  CVPixelBufferRef out = nullptr;
  const CVReturn made = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                            format, attributes, &out);
  CFRelease(none);
  CFRelease(attributes);
  return made == kCVReturnSuccess ? out : nullptr;
}

// The sample a plane holds at a column and row, for pattern `seed`.
std::uint8_t luma(int seed, std::uint32_t x, std::uint32_t y) {
  return static_cast<std::uint8_t>(x + 3 * y + 41 * seed);
}
std::uint8_t cb(int seed, std::uint32_t x, std::uint32_t y) {
  return static_cast<std::uint8_t>(5 * x + y + 17 * seed);
}
std::uint8_t cr(int seed, std::uint32_t x, std::uint32_t y) {
  return static_cast<std::uint8_t>(x + 7 * y + 29 * seed);
}

// Writes pattern `seed` over every sample of an NV12 buffer; with `flat`,
// one colour everywhere, Y = seed, Cb and Cr from `flat`.
bool fill(CVPixelBufferRef pixels, int seed,
          const std::uint8_t* flat = nullptr) {
  if (CVPixelBufferLockBaseAddress(pixels, 0) != kCVReturnSuccess) return false;
  for (std::size_t plane = 0; plane < 2; ++plane) {
    auto* base = static_cast<std::uint8_t*>(
        CVPixelBufferGetBaseAddressOfPlane(pixels, plane));
    const std::size_t row = CVPixelBufferGetBytesPerRowOfPlane(pixels, plane);
    const auto w =
        static_cast<std::uint32_t>(CVPixelBufferGetWidthOfPlane(pixels, plane));
    const auto h = static_cast<std::uint32_t>(
        CVPixelBufferGetHeightOfPlane(pixels, plane));
    for (std::uint32_t y = 0; y < h; ++y) {
      std::uint8_t* line = base + y * row;
      for (std::uint32_t x = 0; x < w; ++x) {
        if (plane == 0) {
          line[x] = flat ? static_cast<std::uint8_t>(seed) : luma(seed, x, y);
        } else {
          line[2 * x] = flat ? flat[0] : cb(seed, x, y);
          line[2 * x + 1] = flat ? flat[1] : cr(seed, x, y);
        }
      }
    }
  }
  CVPixelBufferUnlockBaseAddress(pixels, 0);
  return true;
}

// The picture read back on `device`'s queue holds pattern `seed`, `width` x
// `height` from the buffer's corner.
int check_holds(YuvImage& yuv, vkc::Device& device, vkc::Allocator& allocator,
                int seed) {
  std::vector<std::uint8_t> planes[3];
  vr_test::read_device_picture(yuv, device, allocator, planes);
  const std::uint32_t cw = (yuv.width + 1) / 2;
  const std::uint32_t ch = (yuv.height + 1) / 2;
  CHECK(planes[0].size() == std::size_t{yuv.width} * yuv.height);
  CHECK(planes[1].size() == std::size_t{cw} * ch);
  CHECK(planes[2].size() == planes[1].size());
  for (std::uint32_t y = 0; y < yuv.height; ++y) {
    for (std::uint32_t x = 0; x < yuv.width; ++x) {
      CHECK(planes[0][y * yuv.width + x] == luma(seed, x, y));
    }
  }
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t x = 0; x < cw; ++x) {
      CHECK(planes[1][y * cw + x] == cb(seed, x, y));
      CHECK(planes[2][y * cw + x] == cr(seed, x, y));
    }
  }
  return 0;
}

int check_holds(YuvImage& yuv, const Gpu& gpu, int seed) {
  return check_holds(yuv, gpu.device, gpu.allocator, seed);
}

// Both planes' sharing mode, or VK_SHARING_MODE_MAX_ENUM where they differ.
VkSharingMode sharing(const YuvImage& yuv) {
  const VkSharingMode mode = yuv.image[0]->sharing_mode();
  return yuv.image[1]->sharing_mode() == mode ? mode : VK_SHARING_MODE_MAX_ENUM;
}

// Its planes as images, read in place: what the surface holds when they are
// read, so writing the surface after the import changes what they read. A
// surface's images are made once, and a picture smaller than its buffer sits
// at the corner.
int test_planes(const Gpu& gpu) {
  auto importer = PixelBufferImporter::create(gpu.device, "test");
  CHECK(importer.ok());
  const Pixels pixels(make_pixels(256, 144, kFull));
  CHECK(pixels.ref != nullptr && fill(pixels.ref, 0));

  YuvImage yuv;
  CHECK(importer->import(pixels.ref, 256, 144, yuv).ok());
  CHECK(yuv.layout == sensor::YuvLayout::Nv12);
  CHECK(yuv.image[0] != nullptr && yuv.image[1] != nullptr);
  CHECK(yuv.device == nullptr);
  CHECK(yuv.width == 256 && yuv.height == 144);
  CHECK(check_holds(yuv, gpu, 0) == 0);

  // No copy: the images are the surface.
  CHECK(fill(pixels.ref, 1));
  CHECK(check_holds(yuv, gpu, 1) == 0);

  YuvImage again;
  CHECK(importer->import(pixels.ref, 255, 143, again).ok());
  CHECK(again.image[0] == yuv.image[0] && again.image[1] == yuv.image[1]);
  CHECK(again.width == 255 && again.height == 143);
  CHECK(check_holds(again, gpu, 1) == 0);

  // A '420v' buffer imports the same way.
  const Pixels video(make_pixels(64, 48, kVideo));
  CHECK(video.ref != nullptr && fill(video.ref, 2));
  YuvImage limited;
  CHECK(importer->import(video.ref, 64, 48, limited).ok());
  CHECK(limited.image[0] != yuv.image[0]);  // another surface's
  CHECK(check_holds(limited, gpu, 2) == 0);
  return 0;
}

// The import sets the planes and nothing else: a buffer the image named goes,
// and the colour description stays the caller's.
int test_sets_only_the_planes(const Gpu& gpu) {
  auto importer = PixelBufferImporter::create(gpu.device, "test");
  CHECK(importer.ok());
  const Pixels pixels(make_pixels(32, 32, kFull));
  CHECK(pixels.ref != nullptr && fill(pixels.ref, 3));
  auto buffer = vkc::device_storage_buffer(gpu.allocator, 4096);
  CHECK(buffer.ok());
  YuvImage yuv;
  yuv.device = std::make_shared<const vkc::Buffer>(std::move(buffer).value());
  yuv.layout = sensor::YuvLayout::I420;
  yuv.kr = 0.25f;
  yuv.kb = 0.125f;
  yuv.full_range = false;
  yuv.chroma_location = sensor::ChromaLocation::Center;
  CHECK(importer->import(pixels.ref, 32, 32, yuv).ok());
  CHECK(yuv.device == nullptr && yuv.layout == sensor::YuvLayout::Nv12);
  CHECK(yuv.kr == 0.25f && yuv.kb == 0.125f && !yuv.full_range);
  CHECK(yuv.chroma_location == sensor::ChromaLocation::Center);
  return check_holds(yuv, gpu, 3);
}

// The images hold the pixel buffer, so its producer cannot reuse the surface
// while anything reads them, and let it go with the last of them.
int test_holds_the_buffer(const Gpu& gpu) {
  auto importer = PixelBufferImporter::create(gpu.device, "test");
  CHECK(importer.ok());
  const Pixels pixels(make_pixels(32, 32, kFull));
  CHECK(pixels.ref != nullptr && fill(pixels.ref, 4));
  const CFIndex alone = CFGetRetainCount(pixels.ref);
  YuvImage yuv;
  CHECK(importer->import(pixels.ref, 32, 32, yuv).ok());
  CHECK(CFGetRetainCount(pixels.ref) == alone + 1);
  std::shared_ptr<const vkc::Image> chroma = yuv.image[1];
  yuv = {};
  CHECK(CFGetRetainCount(pixels.ref) == alone + 1);  // the chroma holds it
  chroma.reset();
  CHECK(CFGetRetainCount(pixels.ref) == alone);
  return 0;
}

// The images allocate nothing in Metal beyond the textures on the surface's
// planes: importing four distinct 4K surfaces, a camera's size, grows the
// Metal device's allocation by no more than bare textures on the same planes
// do, plus less than one luma plane. Memory not dedicated to the images would
// add a heap per plane, 13.2 MB a surface on an M4. A surface imported once
// more adds nothing.
int test_allocates_only_textures(const Gpu& gpu) {
  constexpr std::uint32_t kW = 3840;
  constexpr std::uint32_t kH = 2160;
  constexpr std::int64_t kLuma = std::int64_t{kW} * kH;
  constexpr std::size_t kSurfaces = 4;
  auto importer = PixelBufferImporter::create(gpu.device, "test");
  CHECK(importer.ok());
  // What a first import alone makes is not counted.
  const Pixels first(make_pixels(64, 48, kFull));
  CHECK(first.ref != nullptr);
  YuvImage warm;
  CHECK(importer->import(first.ref, 64, 48, warm).ok());

  const Pixels pixels[kSurfaces] = {
      Pixels(make_pixels(kW, kH, kFull)), Pixels(make_pixels(kW, kH, kFull)),
      Pixels(make_pixels(kW, kH, kFull)), Pixels(make_pixels(kW, kH, kFull))};
  std::vector<CVPixelBufferRef> refs;
  for (const Pixels& p : pixels) {
    CHECK(p.ref != nullptr);
    refs.push_back(p.ref);
  }
  const std::optional<std::int64_t> bare =
      vr_test::bare_plane_textures_size(gpu.device, refs);
  CHECK(bare.has_value());

  const std::optional<std::int64_t> before =
      vr_test::metal_allocated_size(gpu.device);
  CHECK(before.has_value());
  YuvImage yuv[kSurfaces];
  for (std::size_t i = 0; i < kSurfaces; ++i) {
    CHECK(importer->import(refs[i], kW, kH, yuv[i]).ok());
  }
  const std::optional<std::int64_t> imported =
      vr_test::metal_allocated_size(gpu.device);
  CHECK(imported.has_value());
  const std::int64_t grown = *imported - *before;
  std::printf(
      "%zu surfaces of %ux%u: imported +%lld B, bare textures +%lld B\n",
      kSurfaces, kW, kH, static_cast<long long>(grown),
      static_cast<long long>(*bare));
  CHECK(grown < *bare + kLuma);

  YuvImage again;
  CHECK(importer->import(refs[0], kW, kH, again).ok());
  CHECK(again.image[0] == yuv[0].image[0]);
  CHECK(vr_test::metal_allocated_size(gpu.device) == imported);
  return 0;
}

// The imported picture is a frame's colour as it is, with the matrix and range
// its caller describes, taking a named matrix's weights from yuv_image.hpp, as
// a camera's caller does with no decoder: one colour throughout, prepared on
// the GPU, comes out as the standards convert it, for full and video range.
int test_prepares(const Gpu& gpu) {
  auto importer = PixelBufferImporter::create(gpu.device, "test");
  CHECK(importer.ok());
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  CHECK(prep.ok());
  const struct {
    OSType format;
    VideoColorMatrix matrix;
    bool full_range;
    std::uint8_t y;
    std::uint8_t cbcr[2];
  } cases[] = {
      {kFull, VideoColorMatrix::Bt601, true, 150, {90, 170}},
      {kVideo, VideoColorMatrix::Bt709, false, 100, {140, 110}},
  };
  constexpr std::uint32_t kW = 64;
  constexpr std::uint32_t kH = 48;
  for (const auto& c : cases) {
    const Pixels pixels(make_pixels(kW, kH, c.format));
    CHECK(pixels.ref != nullptr && fill(pixels.ref, c.y, c.cbcr));
    sensor::RgbdFrame raw;
    raw.depth_camera.size = {kW, kH};
    raw.depth_camera.intrinsics.fx = raw.depth_camera.intrinsics.fy = 64.0;
    raw.depth_camera.intrinsics.cx = kW / 2.0;
    raw.depth_camera.intrinsics.cy = kH / 2.0;
    raw.color_camera = raw.depth_camera;
    std::vector<std::uint16_t> depth(std::size_t{kW} * kH, 1000);
    raw.depth = depth.data();
    raw.min_depth = 0.1f;
    raw.max_depth = 5.0f;
    CHECK(importer->import(pixels.ref, kW, kH, raw.color).ok());
    const sensor::YcbcrWeights weights = sensor::ycbcr_weights(c.matrix);
    raw.color.kr = weights.kr;
    raw.color.kb = weights.kb;
    raw.color.full_range = c.full_range;
    CHECK(yuv_reference::coded_in(raw.color, c.matrix, c.full_range));
    auto frame = prep->prepare(raw);
    CHECK(frame.ok());
    auto got = vr_test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                 *frame->color, depth.size());
    CHECK(got.ok());
    const std::array<int, 3> want =
        yuv_reference::rgb(c.y, c.cbcr[0], c.cbcr[1], c.matrix, c.full_range);
    int worst = 0;
    for (const std::uint32_t word : got.value()) {
      for (int i = 0; i < 3; ++i) {
        const int value = static_cast<int>(word >> (8 * i) & 255);
        worst = std::max(worst, std::abs(value - want[i]));
      }
    }
    if (worst > 1) std::fprintf(stderr, "worst RGB error %d\n", worst);
    CHECK(worst <= 1);
  }
  return 0;
}

// What it does not take: Unsupported for another format, a buffer on no
// IOSurface, or one smaller than the picture; InvalidArgument for no buffer.
// A refused import leaves its image as it was. A device that imports no
// Metal textures has no importer.
int test_refusals(const Gpu& gpu) {
  auto importer = PixelBufferImporter::create(gpu.device, "test");
  CHECK(importer.ok());
  const auto refused = [&](CVPixelBufferRef pixels, std::uint32_t w,
                           std::uint32_t h) {
    YuvImage yuv;
    yuv.width = 7;
    const vkc::Status status = importer->import(pixels, w, h, yuv);
    if (yuv.width != 7 || yuv.image[0] != nullptr) {
      return vkc::Status::Code::Ok;  // touched: not a refusal
    }
    return status.domain();
  };
  const Pixels bgra(make_pixels(64, 64, kCVPixelFormatType_32BGRA));
  const Pixels ten(
      make_pixels(64, 64, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange));
  const Pixels host(make_pixels(64, 64, kFull, false));
  const Pixels small(make_pixels(64, 64, kFull));
  CHECK(bgra.ref != nullptr && ten.ref != nullptr && host.ref != nullptr &&
        small.ref != nullptr);
  CHECK(CVPixelBufferGetIOSurface(host.ref) == nullptr);
  CHECK(refused(bgra.ref, 64, 64) == vkc::Status::Code::Unsupported);
  CHECK(refused(ten.ref, 64, 64) == vkc::Status::Code::Unsupported);
  CHECK(refused(host.ref, 64, 64) == vkc::Status::Code::Unsupported);
  CHECK(refused(small.ref, 65, 64) == vkc::Status::Code::Unsupported);
  CHECK(refused(small.ref, 64, 66) == vkc::Status::Code::Unsupported);
  CHECK(refused(nullptr, 64, 64) == vkc::Status::Code::InvalidArgument);

  vkc::Result<vkc::Device> bare = vr_test::bare_device(gpu);
  CHECK(bare.ok());
  CHECK(PixelBufferImporter::create(bare.value()).status().domain() ==
        vkc::Status::Code::Unsupported);
  return 0;
}

// The images are EXCLUSIVE unless the importer is given two or more distinct
// queue families, and CONCURRENT across those, up to kMaxQueueFamilies of
// them; duplicates count once, so a list longer than kMaxQueueFamilies with
// two distinct is taken. Refused, InvalidArgument: a count with no list, more
// than kMaxQueueFamilies distinct, a family the device does not have, and a
// list without the device's own family, on which the importer readies the
// images.
int test_sharing(const Gpu& gpu) {
  const std::uint32_t own = gpu.device.queue_family();
  const auto family_count =
      static_cast<std::uint32_t>(gpu.device.caps().queue_families().size());
  const Pixels pixels(make_pixels(32, 32, kFull));
  CHECK(pixels.ref != nullptr && fill(pixels.ref, 6));
  const auto imported = [&](const std::uint32_t* families, std::uint32_t count,
                            YuvImage& yuv) {
    auto importer =
        PixelBufferImporter::create(gpu.device, "test", families, count);
    return importer.ok() && importer->import(pixels.ref, 32, 32, yuv).ok();
  };

  YuvImage alone;
  CHECK(imported(nullptr, 0, alone));
  CHECK(sharing(alone) == VK_SHARING_MODE_EXCLUSIVE);
  const std::uint32_t twice[] = {own, own};
  YuvImage one;
  CHECK(imported(twice, 1, one));
  CHECK(sharing(one) == VK_SHARING_MODE_EXCLUSIVE);
  YuvImage duplicated;
  CHECK(imported(twice, 2, duplicated));
  CHECK(sharing(duplicated) == VK_SHARING_MODE_EXCLUSIVE);

  const auto refusal = [&](const std::uint32_t* families, std::uint32_t count) {
    const vkc::Status s =
        PixelBufferImporter::create(gpu.device, "test", families, count)
            .status();
    return s.domain() == vkc::Status::Code::InvalidArgument ? s.message()
                                                            : std::string();
  };
  const auto says = [](const std::string& message, const char* what) {
    return message.find(what) != std::string::npos;
  };
  CHECK(says(refusal(nullptr, 1), "queue_families is null"));
  const std::uint32_t five[] = {own, own + 1, own + 2, own + 3, own + 4};
  CHECK(says(refusal(five, 5), "kMaxQueueFamilies"));
  const std::uint32_t past[] = {own, family_count};
  CHECK(says(refusal(past, 2), "does not have"));

  const std::optional<std::uint32_t> other =
      vr_test::other_queue_family(gpu.device);
  if (!other) {
    std::puts("one queue family: no CONCURRENT images to check");
    return 0;
  }
  CHECK(says(refusal(&*other, 1), "own family"));
  const std::uint32_t both[] = {*other, own, *other, own, *other};
  YuvImage shared;
  CHECK(imported(both, 5, shared));
  CHECK(sharing(shared) == VK_SHARING_MODE_CONCURRENT);
  CHECK(check_holds(shared, gpu, 6) == 0);

  // As many as kMaxQueueFamilies: the device's own and the first others.
  if (family_count < vkc::BufferDesc::kMaxQueueFamilies) {
    std::printf("%u queue families: kMaxQueueFamilies of them not checked\n",
                family_count);
    return 0;
  }
  std::vector<std::uint32_t> most = {own};
  for (std::uint32_t f = 0; most.size() < vkc::BufferDesc::kMaxQueueFamilies;
       ++f) {
    if (f != own) most.push_back(f);
  }
  YuvImage widest;
  CHECK(imported(most.data(), vkc::BufferDesc::kMaxQueueFamilies, widest));
  CHECK(sharing(widest) == VK_SHARING_MODE_CONCURRENT);
  return check_holds(widest, gpu, 6);
}

// What CONCURRENT is for: on one device shared by recon and a renderer on
// another queue family, as MoltenVK hands them, a picture imported on recon's
// family is copied on the renderer's, byte for byte. Windowless: the
// renderer's queue does graphics and presents nothing. The validation layer
// checks the CONCURRENT create info but not which family owns an EXCLUSIVE
// image, so the sharing mode recorded on the image, which is the one the
// image was created with, is what shows the read is legal.
int test_cross_family() {
  auto made = vkc::test::TwoFamilyDevice::create(vr::device_requirements(), {});
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  vkc::test::TwoFamilyDevice& two = **made;
  if (!two.crosses_families()) {
    std::printf("cross-family read skipped: %s\n",
                two.shared().summary().c_str());
    return 0;
  }
  CHECK(two.compute().imports_metal_textures());
  auto importer =
      PixelBufferImporter::create(two.compute(), "test", two.families().data(),
                                  vkc::test::TwoFamilyDevice::kFamilyCount);
  CHECK(importer.ok());
  const Pixels pixels(make_pixels(255, 143, kFull));
  CHECK(pixels.ref != nullptr && fill(pixels.ref, 7));
  YuvImage yuv;
  CHECK(importer->import(pixels.ref, 255, 143, yuv).ok());
  CHECK(sharing(yuv) == VK_SHARING_MODE_CONCURRENT);
  return check_holds(yuv, two.graphics(), two.graphics_allocator(), 7);
}

int test_moves(const Gpu& gpu) {
  const Pixels pixels(make_pixels(32, 32, kFull));
  CHECK(pixels.ref != nullptr && fill(pixels.ref, 5));
  YuvImage yuv;

  auto created = PixelBufferImporter::create(gpu.device, "test");
  CHECK(created.ok());
  PixelBufferImporter a = std::move(created).value();
  PixelBufferImporter b(std::move(a));
  CHECK(a.import(pixels.ref, 32, 32, yuv).domain() ==  // NOLINT: moved from
        vkc::Status::Code::InvalidArgument);

  auto other = PixelBufferImporter::create(gpu.device, "test");
  CHECK(other.ok());
  PixelBufferImporter c = std::move(other).value();
  // Over a live importer.
  c = std::move(b);
  CHECK(b.import(pixels.ref, 32, 32, yuv).domain() ==  // NOLINT: moved from
        vkc::Status::Code::InvalidArgument);

  PixelBufferImporter* alias = &c;
  c = std::move(*alias);  // self-move
  CHECK(c.import(pixels.ref, 32, 32, yuv).ok());
  return check_holds(yuv, gpu, 5);
}

int gpu_main(vr_test::GpuContext& gpu) {
  if (!gpu.device.imports_metal_textures()) {
    return vkc::test::no_device_exit_code(
        "the device imports no Metal textures");
  }

  if (test_planes(gpu) != 0) return 1;
  if (test_sets_only_the_planes(gpu) != 0) return 1;
  if (test_holds_the_buffer(gpu) != 0) return 1;
  if (test_allocates_only_textures(gpu) != 0) return 1;
  if (test_prepares(gpu) != 0) return 1;
  if (test_sharing(gpu) != 0) return 1;
  if (test_cross_family() != 0) return 1;
  if (test_refusals(gpu) != 0) return 1;
  if (test_moves(gpu) != 0) return 1;
  std::puts("sensor_apple_pixel_buffer_importer: OK");
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  return vr_test::run_on_gpu(gpu_main);
}
