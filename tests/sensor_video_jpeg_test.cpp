// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The JPEG decoder on committed JPEGs (tools/make_jpeg_fixtures.sh), decoded
// on this machine's hardware onto the device and read back: the pattern they
// were made from, at an odd size too, prepared on the GPU with centred chroma;
// the buffers nvJPEG's pictures hold; the JPEGs the hardware does not take
// (4:2:2, and one past its size); corrupt bytes, which cost only themselves;
// the decoders create refuses; and moves.
//
// Where no device path opens the test skips; VR_TEST_HEVC_BACKEND, which CI
// sets on the legs that promise one (nvJPEG on the NVIDIA containers,
// VideoToolbox on macOS), makes it fail instead.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <utility>
#include <vector>

#include "bare_device.hpp"
#include "buffer_readback.hpp"
#include "device_picture_readback.hpp"
#include "no_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"
#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"
#include "yuv_reference.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::JpegDecoder;
using sensor::VideoPixelLayout;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr const char* k420 = VR_JPEG_DATA "/patches_256x144.jpg";
constexpr const char* kOdd = VR_JPEG_DATA "/patches_255x143.jpg";
constexpr const char* k422 = VR_JPEG_DATA "/patches_422_256x144.jpg";
constexpr const char* kWide = VR_JPEG_DATA "/patches_16400x72.jpg";

// The pattern: patch p = (column + 3 * row) mod 8 over 32x72 luma patches;
// Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch_y(int p) { return 40 + 24 * p; }
int patch_u(int p) { return 64 + 16 * ((3 * p) % 8); }
int patch_v(int p) { return 64 + 16 * ((5 * p) % 8); }

// The device and allocator the decoders hand their pictures out on.
struct Gpu {
  vkc::Instance& instance;
  vkc::Device& device;
  vkc::Allocator& allocator;
  std::uint32_t extent;  // maxImageDimension2D

  JpegDecoder::Options options() const {
    JpegDecoder::Options o;
    o.device = &device;
    o.allocator = &allocator;
    return o;
  }
};

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

vkc::Result<sensor::DecodedPicture> decode(JpegDecoder& decoder,
                                           const std::vector<std::uint8_t>& b) {
  return decoder.decode(b.data(), b.size());
}

// An I420 picture's planes, packed.
struct Planes {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> plane[3];
};

// nvJPEG's buffer or VideoToolbox's images, read back.
Planes from_device(const sensor::DecodedPicture& p, vkc::Device& device,
                   vkc::Allocator& allocator) {
  Planes out{p.width, p.height, {}};
  vr_test::read_device_picture(p, device, allocator, out.plane);
  return out;
}

// Each patch's centre in the first 256 columns holds the pattern's value,
// within the JPEG's loss.
int check_pattern(const Planes& p) {
  const auto near = [](int got, int want) {
    return got >= want - 3 && got <= want + 3;
  };
  const std::uint32_t cw = (p.width + 1) / 2;
  const std::size_t chroma = std::size_t{cw} * ((p.height + 1) / 2);
  // Every plane whole, so a read that failed fails here rather than index an
  // empty one, and wide enough for the last patch's centre.
  CHECK(p.width > 32 * 7 + 16);
  CHECK(p.plane[0].size() == std::size_t{p.width} * p.height);
  CHECK(p.plane[1].size() == chroma && p.plane[2].size() == chroma);
  for (std::uint32_t row = 0; row < 2 && 72 * row + 36 < p.height; ++row) {
    for (int column = 0; column < 8; ++column) {
      const int k = (column + 3 * row) % 8;
      const std::uint32_t x = 32u * column + 16;
      const std::uint32_t y = 72u * row + 36;
      CHECK(near(p.plane[0][y * p.width + x], patch_y(k)));
      CHECK(near(p.plane[1][(y / 2) * cw + x / 2], patch_u(k)));
      CHECK(near(p.plane[2][(y / 2) * cw + x / 2], patch_v(k)));
    }
  }
  return 0;
}

int check_meta(const sensor::DecodedPicture& p, std::uint32_t w,
               std::uint32_t h) {
  CHECK(p.width == w && p.height == h);
  CHECK(p.layout == (p.image[0] != nullptr ? VideoPixelLayout::Nv12
                                           : VideoPixelLayout::Yuv420));
  CHECK(p.matrix == sensor::VideoColorMatrix::Bt601 && p.full_range);
  CHECK(p.chroma_location == sensor::ChromaLocation::Center);
  return 0;
}

// At integer luma pixels, centred 4:2:0 chroma gives the nearest sample
// weight 3/4 and its neighbour 1/4 on each axis. Keep fractional values
// until RGB conversion. Left alignment is a control for the old shift.
double chroma_at(const Planes& p, int plane, int x, int y, bool centred) {
  const int w = static_cast<int>((p.width + 1) / 2);
  const int h = static_cast<int>((p.height + 1) / 2);
  const int cx = x / 2;
  const int cy = y / 2;
  const int nx = std::clamp(cx + (centred && x % 2 == 0 ? -1 : 1), 0, w - 1);
  const int ny = std::clamp(cy + (y % 2 == 0 ? -1 : 1), 0, h - 1);
  const double wx = centred ? 0.25 : (x % 2) * 0.5;
  const auto& bytes = p.plane[plane];
  return 0.75 * ((1 - wx) * bytes[cy * w + cx] + wx * bytes[cy * w + nx]) +
         0.25 * ((1 - wx) * bytes[ny * w + cx] + wx * bytes[ny * w + nx]);
}

// Decode-to-preparation on the device planes. The reference uses the sampling
// weights and colour-matrix constants; the committed patch edges distinguish
// centred from left-aligned chroma.
int check_preprocessing(const sensor::DecodedPicture& p, const Planes& planes,
                        vkc::Device& device, vkc::Allocator& allocator,
                        sensor::GpuFramePrep& prep) {
  CHECK(planes.plane[0].size() == std::size_t{p.width} * p.height);
  CHECK(planes.plane[1].size() ==
        std::size_t{(p.width + 1) / 2} * ((p.height + 1) / 2));
  CHECK(planes.plane[2].size() == planes.plane[1].size());
  sensor::RgbdFrame raw;
  raw.depth_camera.size = {p.width, p.height};
  raw.depth_camera.intrinsics.fx = raw.depth_camera.intrinsics.fy = 256.0;
  raw.color_camera = raw.depth_camera;
  std::vector<std::uint16_t> depth(std::size_t{p.width} * p.height, 1000);
  raw.depth = depth.data();
  raw.min_depth = 0.1f;
  raw.max_depth = 5.0f;
  raw.color.width = p.width;
  raw.color.height = p.height;
  raw.color.layout = p.layout == VideoPixelLayout::Nv12
                         ? sensor::YuvLayout::Nv12
                         : sensor::YuvLayout::I420;
  raw.color.chroma_location = p.chroma_location;
  raw.color.device = p.device;
  // from_device already acquired a CUDA buffer for the reference readback.
  // Preparation now reads it on the same queue family.
  raw.color.queue_family = sensor::kQueueFamilyIgnored;
  for (int i = 0; i < 2; ++i) raw.color.image[i] = p.image[i];
  for (int i = 0; i < 3; ++i) {
    raw.color.stride[i] = p.stride[i];
    raw.color.offset[i] = p.offset[i];
  }
  auto frame = prep.prepare(raw);
  CHECK(frame.ok());
  auto gpu = vr_test::read_back<std::uint32_t>(device, allocator, *frame->color,
                                               depth.size());
  CHECK(gpu.ok());
  for (bool centred : {false, true}) {
    int max_error = 0;
    for (std::size_t i = 0; i < depth.size(); ++i) {
      const int x = static_cast<int>(i % p.width);
      const int y = static_cast<int>(i / p.width);
      const auto rgb = yuv_reference::rgb(
          planes.plane[0][i], chroma_at(planes, 1, x, y, centred),
          chroma_at(planes, 2, x, y, centred), p.matrix, p.full_range);
      for (int k = 0; k < 3; ++k) {
        const int error = std::abs(
            static_cast<int>(gpu.value()[i] >> (8 * k) & 255) - rgb[k]);
        max_error = std::max(max_error, error);
      }
    }
    CHECK(centred ? max_error <= 1 : max_error >= 30);
  }
  return 0;
}

// On the device, where this platform's hardware leaves it.
int check_on_device(const sensor::DecodedPicture& p) {
#if defined(__APPLE__)
  CHECK(p.image[0] != nullptr && p.image[1] != nullptr && p.device == nullptr);
#else
  CHECK(p.device != nullptr && p.image[0] == nullptr);
#endif
  return 0;
}

// A 4:2:0 JPEG decodes onto the device, the pattern at any size, and the
// preparation centres its chroma. On Apple the wide one decodes too where the
// device's images reach it (an M5's 32768, not an M4's 16384); nvJPEG's
// hardware engine stops at 16384. 4:2:2 is refused. A buffer is reused only
// once no picture holds it.
int test_device(const Gpu& gpu) {
  auto decoder = JpegDecoder::create(gpu.options());
  CHECK(decoder.ok());
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  CHECK(prep.ok());
  const struct {
    const char* path;
    std::uint32_t width;
    std::uint32_t height;
  } fixtures[] = {{k420, 256, 144}, {kOdd, 255, 143}};
  for (const auto& f : fixtures) {
    auto p = decode(decoder.value(), read_file(f.path));
    CHECK(p.ok());
    if (check_on_device(p.value()) != 0) return 1;
    CHECK(check_meta(p.value(), f.width, f.height) == 0);
    const Planes got = from_device(p.value(), gpu.device, gpu.allocator);
    CHECK(check_pattern(got) == 0);
    if (f.path == k420)
      CHECK(check_preprocessing(p.value(), got, gpu.device, gpu.allocator,
                                prep.value()) == 0);
  }

  auto wide = decode(decoder.value(), read_file(kWide));
#if defined(__APPLE__)
  if (16400 > gpu.extent) {
    CHECK(wide.status().domain() == vkc::Status::Code::Unsupported);
  } else {
    CHECK(wide.ok());
    CHECK(check_meta(wide.value(), 16400, 72) == 0);
    CHECK(check_pattern(from_device(wide.value(), gpu.device, gpu.allocator)) ==
          0);
  }
#else
  CHECK(wide.status().domain() == vkc::Status::Code::Unsupported);
#endif
  CHECK(decode(decoder.value(), read_file(k422)).status().domain() ==
        vkc::Status::Code::Unsupported);

#if !defined(__APPLE__)
  const std::vector<std::uint8_t> bytes = read_file(k420);
  auto first = decode(decoder.value(), bytes);
  auto second = decode(decoder.value(), bytes);
  CHECK(first.ok() && second.ok());
  CHECK(first->device != second->device);  // the first still holds its own
  const vkc::Buffer* freed = first->device.get();
  first.value() = {};
  auto third = decode(decoder.value(), bytes);
  CHECK(third.ok() && third->device.get() == freed);
#endif
  return 0;
}

// No device, a device the hardware path cannot reach, and on nvJPEG no
// allocator: each Unsupported.
int test_create_refusals(const Gpu& gpu) {
  CHECK(JpegDecoder::create({}).status().domain() ==
        vkc::Status::Code::Unsupported);
  vkc::Result<vkc::Device> bare =
      vr_test::bare_device(gpu.instance, gpu.device);
  CHECK(bare.ok());
  JpegDecoder::Options options = gpu.options();
  options.device = &bare.value();
  CHECK(JpegDecoder::create(options).status().domain() ==
        vkc::Status::Code::Unsupported);
#if !defined(__APPLE__)
  options = gpu.options();
  options.allocator = nullptr;
  CHECK(JpegDecoder::create(options).status().domain() ==
        vkc::Status::Code::Unsupported);
#endif
  return 0;
}

// A corrupt frame is the camera's, not the GPU's: IoError for it alone, and
// the next JPEG decodes on the device.
int test_corrupt(const Gpu& gpu) {
  auto decoder = JpegDecoder::create(gpu.options());
  CHECK(decoder.ok());
  const std::vector<std::uint8_t> good = read_file(k420);
  CHECK(decoder->decode(nullptr, 16).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(decoder->decode(good.data(), 0).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  const std::vector<std::uint8_t> garbage(64, 0x5a);
  CHECK(decode(decoder.value(), garbage).status().domain() ==
        vkc::Status::Code::IoError);
  const std::vector<std::uint8_t> cut(good.begin(),
                                      good.begin() + good.size() / 2);
  const auto partial = decode(decoder.value(), cut);  // decodes, or IoError
  if (!partial && partial.status().domain() != vkc::Status::Code::IoError)
    std::fprintf(stderr, "%s\n", partial.status().message().c_str());
  CHECK(partial.ok() ||
        partial.status().domain() == vkc::Status::Code::IoError);
  auto next = decode(decoder.value(), good);
  CHECK(next.ok());
  return check_on_device(next.value());
}

int test_moves(const Gpu& gpu) {
  auto created = JpegDecoder::create(gpu.options());
  CHECK(created.ok());
  JpegDecoder a = std::move(created).value();

  JpegDecoder b(std::move(a));
  const std::vector<std::uint8_t> good = read_file(k420);
  CHECK(decode(a, good).status().domain() ==  // NOLINT: moved from
        vkc::Status::Code::InvalidArgument);

  auto other = JpegDecoder::create(gpu.options());
  CHECK(other.ok());
  JpegDecoder c = std::move(other).value();
  c = std::move(b);                           // over a live decoder
  CHECK(decode(b, good).status().domain() ==  // NOLINT: moved from
        vkc::Status::Code::InvalidArgument);

  JpegDecoder* alias = &c;
  c = std::move(*alias);  // self-move
  auto p = decode(c, good);
  CHECK(p.ok());
  return check_pattern(from_device(p.value(), gpu.device, gpu.allocator));
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (read_file(k420).empty()) {
    std::fprintf(stderr, "FAIL: cannot read %s\n", k420);
    return 1;
  }
  vkc::Result<vkc::Instance> instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  vkc::Result<vkc::PhysicalDeviceInfo> physical =
      instance.value().select_physical_device(vr::device_requirements());
  if (!physical) {
    return vr_test::no_device("no compute-capable device",
                              physical.status().message());
  }
  vkc::Result<vkc::Device> device = vkc::Device::create(
      instance.value(), physical.value(), vr::device_requirements());
  CHECK(device.ok());
  vkc::Result<vkc::Allocator> allocator =
      vkc::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  const Gpu gpu{instance.value(), device.value(), allocator.value(),
                physical.value().properties().limits.maxImageDimension2D};
  if (auto probe = JpegDecoder::create(gpu.options()); !probe) {
    if (probe.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", probe.status().message().c_str());
      return 1;
    }
    return vr_test::no_jpeg_decoder(probe.status().message());
  }

  if (test_device(gpu) != 0) return 1;
  if (test_create_refusals(gpu) != 0) return 1;
  if (test_corrupt(gpu) != 0) return 1;
  if (test_moves(gpu) != 0) return 1;
  std::puts("sensor_video_jpeg: OK");
  return 0;
}
