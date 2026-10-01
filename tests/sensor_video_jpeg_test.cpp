// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The JPEG decoder on committed JPEGs (tools/make_jpeg_fixtures.sh): software
// against the pattern they were made from, 4:2:2 converted to 4:2:0, nvJPEG's
// device pictures against software and the buffers they hold, one past the
// hardware engine's size on the GPU's cores, VideoToolbox's images against
// software, the refusals, and moves.
//
// VR_TEST_HEVC_BACKEND=cuda, the NVIDIA legs' promise, also requires nvJPEG in
// a VR_WITH_CUDA build, and =videotoolbox requires VideoToolbox, so a runner
// that silently decodes on the host fails.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "bare_device.hpp"
#include "device_picture_readback.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "volumetric_kit/recon/core/log.hpp"
#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"

namespace vr = volumetric_kit::recon;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::JpegDecodeBackend;
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

// The 4:2:0 JPEGs and their sizes.
struct Fixture {
  const char* path;
  std::uint32_t width;
  std::uint32_t height;
};
constexpr Fixture k420s[] = {
    {k420, 256, 144}, {kOdd, 255, 143}, {kWide, 16400, 72}};

// The pattern: patch p = (column + 3 * row) mod 8 over 32x72 luma patches;
// Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch_y(int p) { return 40 + 24 * p; }
int patch_u(int p) { return 64 + 16 * ((3 * p) % 8); }
int patch_v(int p) { return 64 + 16 * ((5 * p) % 8); }

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

vr::Result<sensor::DecodedPicture> decode(JpegDecoder& decoder,
                                          const std::vector<std::uint8_t>& b) {
  return decoder.decode(b.data(), b.size());
}

// An I420 picture's planes, packed.
struct Planes {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> plane[3];
};

Planes from_host(const sensor::DecodedPicture& p) {
  Planes out{p.width, p.height, {}};
  for (int i = 0; i < 3; ++i) {
    const std::uint32_t w = i == 0 ? p.width : (p.width + 1) / 2;
    const std::uint32_t h = i == 0 ? p.height : (p.height + 1) / 2;
    for (std::uint32_t r = 0; r < h; ++r) {
      const std::uint8_t* row = p.plane[i] + r * p.stride[i];
      out.plane[i].insert(out.plane[i].end(), row, row + w);
    }
  }
  return out;
}

// nvJPEG's buffer or VideoToolbox's images, read back.
Planes from_device(const sensor::DecodedPicture& p, vr::Device& device,
                   vr::Allocator& allocator) {
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
  return 0;
}

int test_names() {
  CHECK(std::strcmp(sensor::to_string(JpegDecodeBackend::Software),
                    "software") == 0);
  CHECK(std::strcmp(sensor::to_string(JpegDecodeBackend::NvjpegHardware),
                    "nvjpeg-hardware") == 0);
  CHECK(std::strcmp(sensor::to_string(JpegDecodeBackend::NvjpegGpu),
                    "nvjpeg-gpu") == 0);
  CHECK(std::strcmp(sensor::to_string(JpegDecodeBackend::VideoToolbox),
                    "videotoolbox") == 0);
  return 0;
}

// Without a device, in software to host planes: the pattern, at an odd size,
// and 4:2:2 converted to 4:2:0.
int test_software() {
  auto decoder = JpegDecoder::create({});
  CHECK(decoder.ok());
  CHECK(decoder->backend() == JpegDecodeBackend::Software);
  const Fixture files[] = {k420s[0], k420s[1], k420s[2], {k422, 256, 144}};
  for (const Fixture& f : files) {
    auto p = decode(decoder.value(), read_file(f.path));
    CHECK(p.ok());
    CHECK(check_meta(p.value(), f.width, f.height) == 0);
    CHECK(p->device == nullptr && p->plane[0] != nullptr);
    CHECK(check_pattern(from_host(p.value())) == 0);
  }
  return 0;
}

// Given a device on an NVIDIA GPU, a 4:2:0 JPEG decodes into a buffer the
// picture holds, as software decodes it: the wide one on the GPU's cores, as
// the hardware engine refuses it. On Apple, into images the picture holds,
// the wide one too where the device's extent takes it, and on the host,
// keeping VideoToolbox, where it does not. 4:2:2 still comes to the host. A
// buffer is reused only once no picture holds it.
int test_device() {
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

  JpegDecoder::Options options;
  options.device = &device.value();
  auto decoder = JpegDecoder::create(options);
  CHECK(decoder.ok());
  const bool on_device = decoder->backend() != JpegDecodeBackend::Software;
  const bool vt = decoder->backend() == JpegDecodeBackend::VideoToolbox;
#if defined(__APPLE__)
  CHECK(!on_device || vt);
#else
  CHECK(VR_TEST_WITH_CUDA || !on_device);
#endif
  const char* required = std::getenv("VR_TEST_HEVC_BACKEND");
  if (VR_TEST_WITH_CUDA && required != nullptr &&
      std::string(required) == "cuda") {
    CHECK(on_device);  // a CUDA leg must decode on the GPU
  }
  if (required != nullptr && std::string(required) == "videotoolbox") {
    CHECK(vt);  // and so must a VideoToolbox one
  }
  auto software = JpegDecoder::create({});
  CHECK(software.ok());
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(gpu.value(), &props);
  const std::uint32_t extent = props.limits.maxImageDimension2D;

  for (const Fixture& f : k420s) {
    const std::vector<std::uint8_t> bytes = read_file(f.path);
    auto p = decode(decoder.value(), bytes);
    CHECK(p.ok());
    // Images stop at the device's extent, which the wide one passes on an M4
    // (16384) and not on an M5 (32768).
    const bool image = vt && f.width <= extent && f.height <= extent;
    CHECK((p->device != nullptr) == (on_device && !vt));
    CHECK((p->image[0] != nullptr && p->image[1] != nullptr) == image);
    if (!image && p->device == nullptr) continue;  // on the host
    CHECK(p->plane[0] == nullptr);
    CHECK(check_meta(p.value(), f.width, f.height) == 0);
    const Planes got =
        from_device(p.value(), device.value(), allocator.value());
    CHECK(check_pattern(got) == 0);
    auto s = decode(software.value(), bytes);
    CHECK(s.ok());
    const Planes want = from_host(s.value());
    CHECK(got.width == want.width && got.height == want.height);
    // Two conformant decoders, whose inverse DCTs may round apart.
    for (int i = 0; i < 3; ++i) {
      CHECK(got.plane[i].size() == want.plane[i].size());
      for (std::size_t k = 0; k < got.plane[i].size(); ++k) {
        CHECK(std::abs(got.plane[i][k] - want.plane[i][k]) <= 2);
      }
    }
  }

  CHECK(!vt || decoder->backend() == JpegDecodeBackend::VideoToolbox);
  auto held = decode(decoder.value(), read_file(k422));
  CHECK(held.ok() && held->device == nullptr && held->plane[0] != nullptr);
  CHECK(held->image[0] == nullptr);
  CHECK(check_pattern(from_host(held.value())) == 0);

  if (on_device && !vt) {
    const std::vector<std::uint8_t> bytes = read_file(k420);
    auto first = decode(decoder.value(), bytes);
    auto second = decode(decoder.value(), bytes);
    CHECK(first.ok() && second.ok());
    CHECK(first->device != second->device);  // the first still holds its own
    const vr::Buffer* freed = first->device.get();
    first.value() = {};
    auto third = decode(decoder.value(), bytes);
    CHECK(third.ok() && third->device.get() == freed);
  }
  std::printf("  device pictures: %s\n",
              on_device ? sensor::to_string(decoder->backend())
                        : "not offered here; they come to the host");
  return 0;
}

// A device the decoder can keep no picture on is said once, as a warning
// naming whose decoder it is; no device says nothing.
int test_host_warning() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  if (!instance) return 0;
  vr::Result<VkPhysicalDevice> gpu = instance.value().select_physical_device();
  if (!gpu) return 0;
  vr::Result<vr::Device> device =
      vr::Device::create(instance.value(), gpu.value(), {});
  CHECK(device.ok());
  vr::Result<vr::Device> bare =
      vr_test::bare_device(instance.value(), device.value());
  CHECK(bare.ok());
  std::vector<std::string> warnings;
  vr::set_log_handler([&warnings](vr::LogLevel level, std::string_view m) {
    if (level == vr::LogLevel::Warning) warnings.emplace_back(m);
  });
  JpegDecoder::Options options;
  options.label = "camera 7";
  const bool quiet = JpegDecoder::create(options).ok() && warnings.empty();
  options.device = &bare.value();
  auto decoder = JpegDecoder::create(options);
  vr::set_log_handler({});
  CHECK(quiet && decoder.ok());
  CHECK(decoder->backend() == JpegDecodeBackend::Software);
  CHECK(warnings.size() == 1);
  CHECK(warnings[0].rfind("camera 7: JpegDecoder: no device path opened", 0) ==
        0);
  return 0;
}

int test_refusals() {
  vr::Result<vr::Instance> instance = vr::Instance::create({});
  vr::Result<VkPhysicalDevice> gpu =
      instance ? instance.value().select_physical_device()
               : vr::Result<VkPhysicalDevice>(instance.status());
  vr::Result<vr::Device> device =
      gpu ? vr::Device::create(instance.value(), gpu.value(), {})
          : vr::Result<vr::Device>(gpu.status());
  JpegDecoder::Options options;
  options.device = device ? &device.value() : nullptr;
  auto decoder = JpegDecoder::create(options);
  CHECK(decoder.ok());

  const std::vector<std::uint8_t> good = read_file(k420);
  CHECK(decoder->decode(nullptr, 16).status().domain() ==
        vr::Status::Code::InvalidArgument);
  CHECK(decoder->decode(good.data(), 0).status().domain() ==
        vr::Status::Code::InvalidArgument);
  // A corrupt frame is the camera's, not the GPU's, so neither costs the
  // device path: nvJPEG refuses the garbage, and decodes the cut one or not.
  const JpegDecodeBackend before = decoder->backend();
  const std::vector<std::uint8_t> garbage(64, 0x5a);
  CHECK(decode(decoder.value(), garbage).status().domain() ==
        vr::Status::Code::IoError);
  const std::vector<std::uint8_t> cut(good.begin(),
                                      good.begin() + good.size() / 2);
  (void)decode(decoder.value(), cut);
  CHECK(decoder->backend() == before);
  auto next = decode(decoder.value(), good);  // and on to the next
  CHECK(next.ok());
  CHECK((next->device != nullptr || next->image[0] != nullptr) ==
        (before != JpegDecodeBackend::Software));
  return 0;
}

int test_moves() {
  auto created = JpegDecoder::create({});
  CHECK(created.ok());
  JpegDecoder a = std::move(created).value();

  JpegDecoder b(std::move(a));
  const std::vector<std::uint8_t> good = read_file(k420);
  CHECK(decode(a, good).status().domain() ==  // NOLINT: moved from
        vr::Status::Code::InvalidArgument);
  CHECK(a.backend() == JpegDecodeBackend::Software);  // NOLINT: moved from

  auto other = JpegDecoder::create({});
  CHECK(other.ok());
  JpegDecoder c = std::move(other).value();
  c = std::move(b);                           // over a live decoder
  CHECK(decode(b, good).status().domain() ==  // NOLINT: moved from
        vr::Status::Code::InvalidArgument);

  JpegDecoder* alias = &c;
  c = std::move(*alias);  // self-move
  auto p = decode(c, good);
  CHECK(p.ok());
  return check_pattern(from_host(p.value()));
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (read_file(k420).empty()) {
    std::fprintf(stderr, "FAIL: cannot read %s\n", k420);
    return 1;
  }
  if (test_names() != 0) return 1;
  if (test_software() != 0) return 1;
  if (test_device() != 0) return 1;
  if (test_host_warning() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_moves() != 0) return 1;
  std::puts("sensor_video_jpeg: OK");
  return 0;
}
