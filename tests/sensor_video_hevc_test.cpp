// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC decoder on committed clips (tools/make_hevc_fixtures.sh): the
// software decoder against the pattern the clip was made from, each hardware
// back end against software (a conformant decoder is bit-exact), a display
// window off the coded picture's corner, the colour an unlabelled stream
// takes, a reset mid-stream and after the end, Auto's move to software with
// pictures still held for display, Auto's choice, the refusals, and moves.
//
// VR_TEST_HEVC_BACKEND=<name> requires that back end: it must decode here and
// Auto must choose it. CI sets it where a leg promises hardware (cuda on the
// NVIDIA containers, videotoolbox on macOS), so a runner that silently falls
// back to software fails rather than passes.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"
#include "yuv_reference.hpp"

namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::HevcDecoder;
using sensor::VideoDecodeBackend;
using sensor::VideoPixelLayout;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 144;
constexpr int kFrames = 8;
constexpr const char* kPatches = VR_HEVC_DATA "/patches_256x144.h265";
constexpr const char* kCropped = VR_HEVC_DATA "/cropped_240x128.h265";
constexpr const char* kFallback = VR_HEVC_DATA "/fallback_256x144.h265";
constexpr const char* kUnlabelled = VR_HEVC_DATA "/unlabelled_256x144.h265";
constexpr int kCropLeft = 16;  // the cropped clip's window
constexpr int kCropTop = 16;

std::int64_t pts_of(int frame) { return 1000 + 33 * frame; }

// The clip's pattern: patch p = (column + frame + 3 * row) mod 8 over 32x72
// luma patches; Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch(int column, int row, int frame) {
  return (column + frame + 3 * row) % 8;
}
int patch_y(int p) { return 40 + 24 * p; }
int patch_u(int p) { return 64 + 16 * ((3 * p) % 8); }
int patch_v(int p) { return 64 + 16 * ((5 * p) % 8); }

// A picture copied out of the decoder, planes packed.
struct Picture {
  sensor::DecodedPicture meta;
  std::vector<std::uint8_t> planes[3];
};

Picture copy(const sensor::DecodedPicture& p) {
  Picture out;
  out.meta = p;
  const bool rgb = p.layout == VideoPixelLayout::Rgb24;
  for (int i = 0; i < (rgb ? 1 : 3); ++i) {
    const std::size_t row =
        rgb ? 3u * p.width : (i == 0 ? p.width : p.width / 2);
    const std::size_t rows = i == 0 ? p.height : p.height / 2;
    for (std::size_t r = 0; r < rows; ++r) {
      const std::uint8_t* line = p.plane[i] + r * p.stride[i];
      out.planes[i].insert(out.planes[i].end(), line, line + row);
    }
    out.meta.plane[i] = nullptr;
  }
  return out;
}

using AccessUnits = std::vector<std::vector<std::uint8_t>>;

// A clip's access units, split at each access unit delimiter (NAL type 35).
AccessUnits access_units(const char* path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
  std::vector<std::size_t> starts;
  for (std::size_t i = 0; i + 3 < bytes.size(); ++i) {
    if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 &&
        ((bytes[i + 3] >> 1) & 0x3f) == 35) {
      starts.push_back(i > 0 && bytes[i - 1] == 0 ? i - 1 : i);
    }
  }
  AccessUnits units;
  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t end =
        k + 1 < starts.size() ? starts[k + 1] : bytes.size();
    units.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(starts[k]),
                       bytes.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return units;
}

// Every picture of a clip, taking each as soon as it is ready.
vr::Result<std::vector<Picture>> decode_clip(HevcDecoder& decoder,
                                             const AccessUnits& units) {
  std::vector<Picture> pictures;
  const auto drain = [&]() -> vr::Status {
    for (;;) {
      VR_ASSIGN(auto picture, decoder.receive());
      if (!picture) return {};
      pictures.push_back(copy(*picture));
    }
  };
  for (std::size_t i = 0; i < units.size(); ++i) {
    VR_TRY(decoder.send(units[i].data(), units[i].size(),
                        pts_of(static_cast<int>(i))));
    VR_TRY(drain());
  }
  VR_TRY(decoder.send(nullptr, 0, 0));
  VR_TRY(drain());
  return pictures;
}

vr::Result<std::vector<Picture>> decode_clip(HevcDecoder& decoder) {
  return decode_clip(decoder, access_units(kPatches));
}

vr::Result<std::vector<Picture>> decode_with(
    VideoDecodeBackend backend, VideoPixelLayout layout,
    std::optional<sensor::VideoColorDescription> unlabelled_color =
        std::nullopt,
    const char* clip = kPatches) {
  HevcDecoder::Options options;
  options.backend = backend;
  options.layout = layout;
  options.unlabelled_color = unlabelled_color;
  VR_ASSIGN(HevcDecoder decoder, HevcDecoder::create(options));
  if (backend != VideoDecodeBackend::Auto && decoder.backend() != backend) {
    return vr::Status::io_error("decoder runs elsewhere");
  }
  return decode_clip(decoder, access_units(clip));
}

// Software, then every hardware back end found.
std::vector<VideoDecodeBackend> every_backend() {
  std::vector<VideoDecodeBackend> backends = {VideoDecodeBackend::Software};
  for (const VideoDecodeBackend b : HevcDecoder::hardware_backends()) {
    backends.push_back(b);
  }
  return backends;
}

int check_clip_shape(const std::vector<Picture>& pictures) {
  CHECK(pictures.size() == static_cast<std::size_t>(kFrames));
  for (int f = 0; f < kFrames; ++f) {
    const auto& meta = pictures[static_cast<std::size_t>(f)].meta;
    CHECK(meta.width == kWidth && meta.height == kHeight);
    CHECK(meta.pts == pts_of(f));
    CHECK(meta.matrix == sensor::VideoColorMatrix::Bt709);
    CHECK(!meta.full_range);
    CHECK(meta.encoding.has_value() &&
          meta.encoding->transfer == vr::ColorEncoding::Transfer::Bt709 &&
          meta.encoding->primaries == vr::ColorEncoding::Primaries::Bt709);
  }
  return 0;
}

// Software, Yuv420: every patch's interior against the pattern (qp 4, so
// within a code or two), which also pins the order and pts of the pictures.
int test_software_yuv() {
  auto decoded =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Yuv420);
  if (!decoded)
    std::fprintf(stderr, "%s\n", decoded.status().message().c_str());
  CHECK(decoded.ok());
  if (check_clip_shape(decoded.value()) != 0) return 1;
  int worst = 0;
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.value()[static_cast<std::size_t>(f)];
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int p = patch(col, row, f);
        for (int dy = -8; dy < 8; ++dy) {
          for (int dx = -8; dx < 8; ++dx) {
            const int x = 32 * col + 16 + dx;
            const int y = 72 * row + 36 + dy;
            const int cx = x / 2;
            const int cy = y / 2;
            const int got[3] = {pic.planes[0][y * kWidth + x],
                                pic.planes[1][cy * (kWidth / 2) + cx],
                                pic.planes[2][cy * (kWidth / 2) + cx]};
            const int want[3] = {patch_y(p), patch_u(p), patch_v(p)};
            for (int k = 0; k < 3; ++k) {
              worst = std::max(worst, std::abs(got[k] - want[k]));
            }
          }
        }
      }
    }
  }
  if (worst > 2) std::fprintf(stderr, "worst YUV error %d\n", worst);
  CHECK(worst <= 2);
  return 0;
}

// Software, Rgb24: each patch's centre against BT.709 limited range.
int test_software_rgb() {
  auto decoded =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Rgb24);
  CHECK(decoded.ok());
  if (check_clip_shape(decoded.value()) != 0) return 1;
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.value()[static_cast<std::size_t>(f)];
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int p = patch(col, row, f);
        const auto want =
            yuv_reference::rgb(patch_y(p), patch_u(p), patch_v(p),
                               sensor::VideoColorMatrix::Bt709, false);
        const int x = 32 * col + 16;
        const int y = 72 * row + 36;
        for (int k = 0; k < 3; ++k) {
          const int got = pic.planes[0][3 * (y * kWidth + x) + k];
          if (std::abs(got - want[k]) > 3) {
            std::fprintf(stderr, "frame %d patch %d,%d channel %d: %d vs %d\n",
                         f, col, row, k, got, want[k]);
            CHECK(false);
          }
        }
      }
    }
  }
  return 0;
}

// Options::unlabelled_color on every back end. The unlabelled clip takes it:
// it is reported on each picture and followed by Rgb24, and the Yuv420 bytes
// are those decoded without it. The labelled clip keeps its BT.709 limited.
int test_unlabelled_color() {
  const sensor::VideoColorDescription femto{sensor::VideoColorMatrix::Bt601,
                                            true};
  auto plain =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Yuv420);
  CHECK(plain.ok());
  for (const VideoDecodeBackend backend : every_backend()) {
    std::printf("  unlabelled colour on %s\n", sensor::to_string(backend));
    auto yuv =
        decode_with(backend, VideoPixelLayout::Yuv420, femto, kUnlabelled);
    auto rgb =
        decode_with(backend, VideoPixelLayout::Rgb24, femto, kUnlabelled);
    CHECK(yuv.ok() && rgb.ok());
    CHECK(yuv.value().size() == plain.value().size());
    for (std::size_t f = 0; f < yuv.value().size(); ++f) {
      const Picture& y = yuv.value()[f];
      CHECK(y.meta.matrix == sensor::VideoColorMatrix::Bt601);
      CHECK(y.meta.full_range);
      for (int i = 0; i < 3; ++i)
        CHECK(y.planes[i] == plain.value()[f].planes[i]);
      const Picture& pic = rgb.value()[f];
      CHECK(pic.meta.matrix == sensor::VideoColorMatrix::Bt601);
      CHECK(pic.meta.full_range);
      for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 8; ++col) {
          const int p = patch(col, row, static_cast<int>(f));
          const auto want =
              yuv_reference::rgb(patch_y(p), patch_u(p), patch_v(p),
                                 sensor::VideoColorMatrix::Bt601, true);
          const int x = 32 * col + 16;
          const int yy = 72 * row + 36;
          for (int k = 0; k < 3; ++k) {
            CHECK(std::abs(pic.planes[0][3 * (yy * kWidth + x) + k] -
                           want[k]) <= 3);
          }
        }
      }
    }
    auto labelled = decode_with(backend, VideoPixelLayout::Rgb24, femto);
    CHECK(labelled.ok());
    if (check_clip_shape(labelled.value()) != 0) return 1;
  }
  return 0;
}

// reset() on every back end: the pictures held for display are dropped, a
// stream ended with size 0 takes data again, and the next key frame decodes
// as the first.
int test_reset() {
  const AccessUnits b_frames = access_units(kFallback);
  const AccessUnits patches = access_units(kPatches);
  for (const VideoDecodeBackend backend : every_backend()) {
    std::printf("  reset on %s\n", sensor::to_string(backend));
    HevcDecoder::Options options;
    options.backend = backend;
    options.layout = VideoPixelLayout::Yuv420;
    auto decoder = HevcDecoder::create(options);
    CHECK(decoder.ok());
    int out = 0;
    for (std::size_t i = 0; i < 4; ++i) {
      CHECK(decoder
                ->send(b_frames[i].data(), b_frames[i].size(),
                       pts_of(static_cast<int>(i)))
                .ok());
      for (;;) {
        auto picture = decoder->receive();
        CHECK(picture.ok());
        if (!picture.value()) break;
        ++out;
      }
    }
    CHECK(out < 4);  // B-frames: some are held for display
    CHECK(decoder->reset().ok());
    auto dropped = decoder->receive();
    CHECK(dropped.ok() && !dropped.value());
    CHECK(decoder->send(nullptr, 0, 0).ok());
    CHECK(decoder->send(patches[0].data(), patches[0].size(), 0).domain() ==
          vr::Status::Code::InvalidArgument);  // after the end
    CHECK(decoder->reset().ok());
    auto pictures = decode_clip(decoder.value(), patches);
    CHECK(pictures.ok());
    if (check_clip_shape(pictures.value()) != 0) return 1;
  }
  return 0;
}

// Each hardware back end: Yuv420 identical to software, Rgb24 within a code.
int test_hardware_matches_software() {
  auto soft_yuv =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Yuv420);
  auto soft_rgb =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Rgb24);
  CHECK(soft_yuv.ok() && soft_rgb.ok());
  for (const VideoDecodeBackend backend : HevcDecoder::hardware_backends()) {
    std::printf("  hardware back end: %s\n", sensor::to_string(backend));
    auto yuv = decode_with(backend, VideoPixelLayout::Yuv420);
    if (!yuv) std::fprintf(stderr, "%s\n", yuv.status().message().c_str());
    CHECK(yuv.ok());
    if (check_clip_shape(yuv.value()) != 0) return 1;
    for (int f = 0; f < kFrames; ++f) {
      for (int i = 0; i < 3; ++i) {
        CHECK(yuv.value()[static_cast<std::size_t>(f)].planes[i] ==
              soft_yuv.value()[static_cast<std::size_t>(f)].planes[i]);
      }
    }
    std::printf("  hardware back end: %s, rgb\n", sensor::to_string(backend));
    auto rgb = decode_with(backend, VideoPixelLayout::Rgb24);
    CHECK(rgb.ok());
    if (check_clip_shape(rgb.value()) != 0) return 1;
    int worst = 0;
    for (int f = 0; f < kFrames; ++f) {
      const auto& a = rgb.value()[static_cast<std::size_t>(f)].planes[0];
      const auto& b = soft_rgb.value()[static_cast<std::size_t>(f)].planes[0];
      CHECK(a.size() == b.size());
      for (std::size_t k = 0; k < a.size(); ++k) {
        worst = std::max(worst, std::abs(a[k] - b[k]));
      }
    }
    CHECK(worst <= 1);
  }
  return 0;
}

// The cropped clip is the uncropped one's pixels, 16 right and 16 down, on
// every back end that can crop there. VideoToolbox cannot (FFmpeg sizes its
// output at the display size, filled from the coded corner), so named it
// refuses the stream and Auto moves it to software.
int check_cropped(const std::vector<Picture>& cropped,
                  const std::vector<Picture>& full) {
  CHECK(cropped.size() == full.size());
  const int width = kWidth - kCropLeft;
  const int height = kHeight - kCropTop;
  for (std::size_t f = 0; f < cropped.size(); ++f) {
    const Picture& c = cropped[f];
    const Picture& u = full[f];
    CHECK(c.meta.width == static_cast<std::uint32_t>(width));
    CHECK(c.meta.height == static_cast<std::uint32_t>(height));
    CHECK(c.meta.pts == u.meta.pts);
    for (int i = 0; i < 3; ++i) {
      const int shift = i == 0 ? 0 : 1;
      const int w = width >> shift;
      for (int y = 0; y < height >> shift; ++y) {
        for (int x = 0; x < w; ++x) {
          const int want = u.planes[i][static_cast<std::size_t>(
              ((y + (kCropTop >> shift)) * (kWidth >> shift)) + x +
              (kCropLeft >> shift))];
          CHECK(c.planes[i][static_cast<std::size_t>(y * w + x)] == want);
        }
      }
    }
  }
  return 0;
}

// Given a device on the GPU NVDEC decodes on, a picture stays on the device:
// NV12 in a buffer it holds, the same samples as software (hardware is
// bit-exact), cropped at the left and top as the host path crops it. Without
// CUDA in the build, or on any other back end, pictures still come to the
// host.
Picture from_device(const sensor::DecodedPicture& p, vr::Device& device,
                    vr::Allocator& allocator) {
  Picture out;
  out.meta = p;
  // CUDA wrote the buffer, so the batch takes it over before reading it.
  std::vector<std::uint8_t> b(static_cast<std::size_t>(p.device->size()));
  vr::CommandBatch batch(device, allocator);
  if (!batch.acquire(*p.device, VK_QUEUE_FAMILY_EXTERNAL).ok() ||
      !batch.readback(*p.device, 0, b.size(), b.data()).ok() ||
      !batch.submit().ok()) {
    return out;
  }
  for (std::uint32_t r = 0; r < p.height; ++r) {
    const std::uint8_t* row = b.data() + p.offset[0] + r * p.stride[0];
    out.planes[0].insert(out.planes[0].end(), row, row + p.width);
  }
  for (std::uint32_t r = 0; r < p.height / 2; ++r) {
    const std::uint8_t* row = b.data() + p.offset[1] + r * p.stride[1];
    for (std::uint32_t x = 0; x < p.width / 2; ++x) {
      out.planes[1].push_back(row[2 * x]);
      out.planes[2].push_back(row[2 * x + 1]);
    }
  }
  return out;
}

int test_device_pictures() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) return 0;
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) return 0;
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  CHECK(device.ok());
  vr::Result<vr::Allocator> allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  const auto hardware = HevcDecoder::hardware_backends();
  const bool cuda = std::find(hardware.begin(), hardware.end(),
                              VideoDecodeBackend::Cuda) != hardware.end();
  // The decoder also needs the Vulkan device to be a GPU CUDA sees, which an
  // NVIDIA one is.
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(gpu.value(), &props);
  constexpr std::uint32_t kNvidia = 0x10DE;
  const bool on_device = VR_TEST_WITH_CUDA && cuda &&
                         device.value().exports_memory() &&
                         props.vendorID == kNvidia;
  const char* required = std::getenv("VR_TEST_HEVC_BACKEND");
  if (VR_TEST_WITH_CUDA && required != nullptr &&
      std::string(required) == "cuda") {
    CHECK(on_device);  // a CUDA leg must hand pictures out on the device
  }

  const char* clips[] = {kPatches, kCropped};
  for (const char* clip : clips) {
    auto soft = decode_with(VideoDecodeBackend::Software,
                            VideoPixelLayout::Yuv420, std::nullopt, clip);
    CHECK(soft.ok());
    HevcDecoder::Options options;
    options.backend =
        cuda ? VideoDecodeBackend::Cuda : VideoDecodeBackend::Auto;
    options.layout = VideoPixelLayout::Yuv420;
    options.device = &device.value();
    auto decoder = HevcDecoder::create(options);
    CHECK(decoder.ok());
    const AccessUnits units = access_units(clip);
    std::vector<Picture> pictures;
    for (std::size_t i = 0; i <= units.size(); ++i) {
      const bool end = i == units.size();
      CHECK(decoder.value()
                .send(end ? nullptr : units[i].data(),
                      end ? 0 : units[i].size(), pts_of(static_cast<int>(i)))
                .ok());
      for (;;) {
        auto picture = decoder.value().receive();
        CHECK(picture.ok());
        if (!picture.value()) break;
        const sensor::DecodedPicture& p = *picture.value();
        CHECK((p.device != nullptr) == on_device);
        if (p.device == nullptr) {
          pictures.push_back(copy(p));
          continue;
        }
        CHECK(p.layout == VideoPixelLayout::Nv12 && p.plane[0] == nullptr);
        pictures.push_back(from_device(p, device.value(), allocator.value()));
      }
    }
    CHECK(pictures.size() == soft.value().size());
    for (std::size_t f = 0; f < pictures.size(); ++f) {
      const Picture& a = pictures[f];
      const Picture& b = soft.value()[f];
      CHECK(a.meta.width == b.meta.width && a.meta.height == b.meta.height);
      CHECK(a.meta.pts == b.meta.pts);
      CHECK(a.meta.matrix == b.meta.matrix);
      CHECK(a.meta.full_range == b.meta.full_range);
      for (int i = 0; i < 3; ++i) CHECK(a.planes[i] == b.planes[i]);
    }
  }
  std::printf("  device pictures: %s\n",
              on_device ? "on the device, as software decodes them"
                        : "not offered here; they come to the host");
  return 0;
}

int test_cropped() {
  auto full =
      decode_with(VideoDecodeBackend::Software, VideoPixelLayout::Yuv420);
  CHECK(full.ok());
  const AccessUnits units = access_units(kCropped);
  std::vector<VideoDecodeBackend> backends = every_backend();
  backends.push_back(VideoDecodeBackend::Auto);
  for (const VideoDecodeBackend backend : backends) {
    HevcDecoder::Options options;
    options.backend = backend;
    options.layout = VideoPixelLayout::Yuv420;
    std::printf("  cropped on %s\n", sensor::to_string(backend));
    auto decoder = HevcDecoder::create(options);
    CHECK(decoder.ok());
    auto pictures = decode_clip(decoder.value(), units);
    if (backend == VideoDecodeBackend::VideoToolbox) {
      CHECK(pictures.status().domain() == vr::Status::Code::Unsupported);
      continue;
    }
    if (!pictures) {
      std::fprintf(stderr, "%s: %s\n", sensor::to_string(backend),
                   pictures.status().message().c_str());
    }
    CHECK(pictures.ok());
    if (check_cropped(pictures.value(), full.value()) != 0) {
      std::fprintf(stderr, "cropped on %s\n",
                   sensor::to_string(decoder->backend()));
      return 1;
    }
    if (backend == VideoDecodeBackend::Auto) {
      std::printf("  auto, cropped: %s\n",
                  sensor::to_string(decoder->backend()));
    }
  }
  return 0;
}

// The fallback clip: 8 frames with B-frames, then 2 of 4:0:0 grey that no
// hardware back end decodes. Every picture comes out, in display order and
// each pts sent once, including those the hardware still held when Auto
// moved to software; a named hardware back end refuses at the grey.
int test_fallback() {
  const AccessUnits units = access_units(kFallback);
  CHECK(units.size() == static_cast<std::size_t>(kFrames + 2));
  std::vector<VideoDecodeBackend> backends = every_backend();
  backends.push_back(VideoDecodeBackend::Auto);
  for (const VideoDecodeBackend backend : backends) {
    HevcDecoder::Options options;
    options.backend = backend;
    options.layout = VideoPixelLayout::Yuv420;
    std::printf("  fallback on %s\n", sensor::to_string(backend));
    auto decoder = HevcDecoder::create(options);
    CHECK(decoder.ok());
    auto pictures = decode_clip(decoder.value(), units);
    if (backend != VideoDecodeBackend::Auto &&
        backend != VideoDecodeBackend::Software) {
      CHECK(pictures.status().domain() == vr::Status::Code::Unsupported);
      continue;
    }
    if (!pictures) {
      std::fprintf(stderr, "%s: %s\n", sensor::to_string(backend),
                   pictures.status().message().c_str());
    }
    CHECK(pictures.ok());
    CHECK(decoder->backend() == VideoDecodeBackend::Software);
    const auto& got = pictures.value();
    CHECK(got.size() == units.size());
    std::vector<std::int64_t> sent;
    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i < got.size(); ++i) {
      sent.push_back(pts_of(static_cast<int>(i)));
      out.push_back(got[i].meta.pts);
      const std::size_t y = 36 * kWidth + 16;  // patch (0, 0)'s centre
      const std::size_t c = 18 * (kWidth / 2) + 8;
      if (i < static_cast<std::size_t>(kFrames)) {
        const int want = patch_y(patch(0, 0, static_cast<int>(i)));
        CHECK(std::abs(got[i].planes[0][y] - want) <= 2);
      } else {
        CHECK(got[i].planes[1][c] == 128 && got[i].planes[2][c] == 128);
      }
    }
    std::sort(out.begin(), out.end());
    CHECK(out == sent);
  }
  return 0;
}

// Auto takes the first hardware back end, else software; the environment can
// demand which.
int test_auto_choice() {
  const auto hardware = HevcDecoder::hardware_backends();
  for (const VideoDecodeBackend b : hardware) {
    CHECK(b != VideoDecodeBackend::Auto && b != VideoDecodeBackend::Software);
  }
  CHECK(HevcDecoder::hardware_backends() == hardware);  // probed once

  HevcDecoder::Options options;
  auto decoder = HevcDecoder::create(options);
  CHECK(decoder.ok());
  const VideoDecodeBackend chosen = decoder->backend();
  CHECK(chosen ==
        (hardware.empty() ? VideoDecodeBackend::Software : hardware.front()));
  auto pictures = decode_clip(decoder.value());
  CHECK(pictures.ok());
  if (check_clip_shape(pictures.value()) != 0) return 1;
  CHECK(decoder->backend() == chosen);  // the clip is plain 4:2:0 Main
  std::printf("  auto: %s\n", sensor::to_string(chosen));

  const char* required = std::getenv("VR_TEST_HEVC_BACKEND");
  if (required != nullptr && *required != '\0') {
    if (std::strcmp(required, sensor::to_string(chosen)) != 0) {
      std::fprintf(stderr, "VR_TEST_HEVC_BACKEND=%s but auto chose %s\n",
                   required, sensor::to_string(chosen));
      CHECK(false);
    }
  }
  return 0;
}

int test_refusals() {
  HevcDecoder::Options options;
#if defined(__APPLE__)
  options.backend = VideoDecodeBackend::Cuda;
#else
  options.backend = VideoDecodeBackend::VideoToolbox;
#endif
  auto missing = HevcDecoder::create(options);
  CHECK(missing.status().domain() == vr::Status::Code::Unsupported);
  CHECK(missing.status().message().find("available: software") !=
        std::string::npos);

  options.backend = VideoDecodeBackend::Software;
  options.threads = -1;
  CHECK(HevcDecoder::create(options).status().domain() ==
        vr::Status::Code::InvalidArgument);
  options.threads = 0;
  options.layout = VideoPixelLayout::Nv12;  // only a device picture is
  CHECK(HevcDecoder::create(options).status().domain() ==
        vr::Status::Code::InvalidArgument);
  options.layout = VideoPixelLayout::Rgb24;

  auto decoder = HevcDecoder::create(options);
  CHECK(decoder.ok());
  auto nothing = decoder->receive();  // before any input
  CHECK(nothing.ok() && !nothing.value());
  CHECK(decoder->send(nullptr, 16, 0).domain() ==
        vr::Status::Code::InvalidArgument);
  CHECK(decoder->send(nullptr, 0, 0).ok());
  const std::uint8_t byte = 0;
  CHECK(decoder->send(&byte, 1, 0).domain() ==
        vr::Status::Code::InvalidArgument);  // after the end
  auto drained = decoder->receive();
  CHECK(drained.ok() && !drained.value());
  return 0;
}

int test_moves() {
  HevcDecoder::Options options;
  options.backend = VideoDecodeBackend::Software;
  auto created = HevcDecoder::create(options);
  CHECK(created.ok());
  HevcDecoder a = std::move(created).value();

  HevcDecoder b(std::move(a));
  CHECK(a.backend() == VideoDecodeBackend::Auto);  // NOLINT: moved from
  CHECK(a.send(nullptr, 0, 0).domain() == vr::Status::Code::InvalidArgument);
  CHECK(a.receive().status().domain() == vr::Status::Code::InvalidArgument);
  CHECK(a.reset().domain() == vr::Status::Code::InvalidArgument);
  CHECK(b.backend() == VideoDecodeBackend::Software);

  auto other = HevcDecoder::create(options);
  CHECK(other.ok());
  HevcDecoder c = std::move(other).value();
  c = std::move(b);                                // over a live decoder
  CHECK(b.backend() == VideoDecodeBackend::Auto);  // NOLINT: moved from

  HevcDecoder* alias = &c;
  c = std::move(*alias);  // self-move
  CHECK(c.backend() == VideoDecodeBackend::Software);
  auto pictures = decode_clip(c);
  CHECK(pictures.ok());
  return check_clip_shape(pictures.value());
}

int test_names() {
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::Auto), "auto") == 0);
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::Cuda), "cuda") == 0);
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::VideoToolbox),
                    "videotoolbox") == 0);
  CHECK(std::strcmp(sensor::to_string(VideoDecodeBackend::Vaapi), "vaapi") ==
        0);
  return 0;
}

}  // namespace

int main() {
  // Unbuffered, so a crash inside FFmpeg or a driver still shows which back
  // end and clip it was on.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (access_units(kPatches).size() != static_cast<std::size_t>(kFrames)) {
    std::fprintf(stderr, "FAIL: cannot split %s into %d access units\n",
                 kPatches, kFrames);
    return 1;
  }
  if (test_names() != 0) return 1;
  if (test_device_pictures() != 0) return 1;
  if (test_software_yuv() != 0) return 1;
  if (test_software_rgb() != 0) return 1;
  if (test_hardware_matches_software() != 0) return 1;
  if (test_unlabelled_color() != 0) return 1;
  if (test_reset() != 0) return 1;
  if (test_cropped() != 0) return 1;
  if (test_fallback() != 0) return 1;
  if (test_auto_choice() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_moves() != 0) return 1;
  std::puts("sensor_video_hevc: OK");
  return 0;
}
