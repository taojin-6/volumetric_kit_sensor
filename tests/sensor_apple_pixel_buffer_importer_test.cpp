// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// PixelBufferImporter on pixel buffers made here, as a camera hands them out,
// with no decoder and no FFmpeg: an NV12 IOSurface's planes reach Vulkan as
// images without a copy -- they read what the surface holds when they are
// read, not when it was imported -- at the picture's size from the buffer's
// corner, each surface's images made once; the images hold the buffer; the
// picture prepares as a frame's colour by the description its caller gives;
// what it refuses; and moves.
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
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "device_picture_readback.hpp"
#include "gpu_test.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
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

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

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

// The picture read back holds pattern `seed`, `width` x `height` from the
// buffer's corner.
int check_holds(const YuvImage& yuv, const Gpu& gpu, int seed) {
  std::vector<std::uint8_t> planes[3];
  vr_test::read_device_picture(yuv, gpu.device, gpu.allocator, planes);
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

// The imported picture is a frame's colour as it is, with the matrix and range
// its caller describes: one colour throughout, prepared on the GPU, comes out
// as the standards convert it, for full and video range.
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
    const std::array<double, 2> k = yuv_reference::weights(c.matrix);
    raw.color.kr = static_cast<float>(k[0]);
    raw.color.kb = static_cast<float>(k[1]);
    raw.color.full_range = c.full_range;
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
  if (test_prepares(gpu) != 0) return 1;
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
