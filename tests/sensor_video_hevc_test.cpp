// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The HEVC decoder on committed clips (tools/make_hevc_fixtures.sh), decoded
// on this machine's hardware onto the device and read back: the pattern the
// clip was made from, the colour an unlabelled stream takes, a reset
// mid-stream and after the end, a display window off the coded picture's
// corner, a stream the hardware refuses, the decoders create refuses, the
// argument checks, and moves.
//
// Where no device path opens the test skips; VR_TEST_HEVC_BACKEND, which CI
// sets on the legs that promise one (NVDEC on the NVIDIA containers,
// VideoToolbox on macOS), makes it fail instead.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "bare_device.hpp"
#include "device_picture_readback.hpp"
#include "no_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
using sensor::HevcDecoder;
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
constexpr const char* kRefused = VR_HEVC_DATA "/refused_256x144.h265";
constexpr const char* kUnlabelled = VR_HEVC_DATA "/unlabelled_256x144.h265";
constexpr const char* kFullRange = VR_HEVC_DATA "/fullrange_256x144.h265";

std::int64_t pts_of(int frame) { return 1000 + 33 * frame; }

// The clip's pattern: patch p = (column + frame + 3 * row) mod 8 over 32x72
// luma patches; Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch(int column, int row, int frame) {
  return (column + frame + 3 * row) % 8;
}
int patch_y(int p) { return 40 + 24 * p; }
int patch_u(int p) { return 64 + 16 * ((3 * p) % 8); }
int patch_v(int p) { return 64 + 16 * ((5 * p) % 8); }

// The device and allocator the decoders hand their pictures out on.
struct Gpu {
  vkc::Instance& instance;
  vkc::Device& device;
  vkc::Allocator& allocator;

  HevcDecoder::Options options() const {
    HevcDecoder::Options o;
    o.device = &device;
    o.allocator = &allocator;
    return o;
  }
};

// A picture read back from the device, planes packed.
struct Picture {
  sensor::DecodedPicture meta;
  std::vector<std::uint8_t> planes[3];
};

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

// The pictures a clip decoded to, each read back as soon as it is ready, and
// the first error, after which the pictures still waiting are taken too.
struct Decoded {
  std::vector<Picture> pictures;
  vkc::Status status;
};

Decoded decode_clip(HevcDecoder& decoder, const AccessUnits& units,
                    const Gpu& gpu) {
  Decoded out;
  const auto drain = [&]() -> vkc::Status {
    for (;;) {
      VKC_ASSIGN(auto picture, decoder.receive());
      if (!picture) return {};
      Picture p;
      p.meta = *picture;
      vr_test::read_device_picture(*picture, gpu.device, gpu.allocator,
                                   p.planes);
      if (p.planes[0].size() != std::size_t{picture->width} * picture->height) {
        return vkc::Status::io_error("reading a picture back");
      }
      out.pictures.push_back(std::move(p));
    }
  };
  for (std::size_t i = 0; i <= units.size() && out.status.ok(); ++i) {
    const bool end = i == units.size();
    out.status =
        decoder.send(end ? nullptr : units[i].data(), end ? 0 : units[i].size(),
                     pts_of(static_cast<int>(i)));
    const vkc::Status drained = drain();
    if (out.status.ok()) out.status = drained;
  }
  return out;
}

Decoded decode(
    const Gpu& gpu, const char* clip,
    std::optional<sensor::VideoColorDescription> unlabelled = std::nullopt) {
  HevcDecoder::Options options = gpu.options();
  options.unlabelled_color = unlabelled;
  auto decoder = HevcDecoder::create(options);
  if (!decoder) return {{}, decoder.status()};
  return decode_clip(decoder.value(), access_units(clip), gpu);
}

// On the device, where this platform's hardware leaves it.
int check_on_device(const sensor::DecodedPicture& p) {
  CHECK(p.layout == VideoPixelLayout::Nv12);
#if defined(__APPLE__)
  CHECK(p.image[0] != nullptr && p.image[1] != nullptr && p.device == nullptr);
#else
  CHECK(p.device != nullptr && p.image[0] == nullptr);
#endif
  return 0;
}

int check_clip_shape(const Decoded& decoded) {
  if (!decoded.status.ok()) {
    std::fprintf(stderr, "%s\n", decoded.status.message().c_str());
  }
  CHECK(decoded.status.ok());
  CHECK(decoded.pictures.size() == static_cast<std::size_t>(kFrames));
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.pictures[static_cast<std::size_t>(f)];
    const auto& meta = pic.meta;
    if (check_on_device(meta) != 0) return 1;
    CHECK(meta.width == kWidth && meta.height == kHeight);
    CHECK(meta.pts == pts_of(f));
    CHECK(meta.matrix == sensor::VideoColorMatrix::Bt709);
    CHECK(!meta.full_range);
    CHECK(meta.encoding.has_value() &&
          meta.encoding->transfer == vr::ColorEncoding::Transfer::Bt709 &&
          meta.encoding->primaries == vr::ColorEncoding::Primaries::Bt709);
    CHECK(pic.planes[0].size() == static_cast<std::size_t>(kWidth * kHeight));
  }
  return 0;
}

// Every patch's interior against the pattern (qp 4, so within a code or two),
// which also pins the order and pts of the pictures.
int test_pattern(const Gpu& gpu) {
  const Decoded decoded = decode(gpu, kPatches);
  if (check_clip_shape(decoded) != 0) return 1;
  int worst = 0;
  for (int f = 0; f < kFrames; ++f) {
    const Picture& pic = decoded.pictures[static_cast<std::size_t>(f)];
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

// Options::unlabelled_color: the unlabelled clip takes it, reported on each
// picture, the bytes those decoded without it. The labelled clip keeps its
// BT.709 limited, and one labelled full range decodes as such.
int test_unlabelled_color(const Gpu& gpu) {
  const sensor::VideoColorDescription femto{sensor::VideoColorMatrix::Bt601,
                                            true};
  const Decoded plain = decode(gpu, kUnlabelled);
  const Decoded taken = decode(gpu, kUnlabelled, femto);
  CHECK(plain.status.ok() && taken.status.ok());
  CHECK(taken.pictures.size() == static_cast<std::size_t>(kFrames));
  CHECK(taken.pictures.size() == plain.pictures.size());
  for (std::size_t f = 0; f < taken.pictures.size(); ++f) {
    const Picture& p = taken.pictures[f];
    CHECK(p.meta.matrix == sensor::VideoColorMatrix::Bt601);
    CHECK(p.meta.full_range);
    for (int i = 0; i < 3; ++i)
      CHECK(p.planes[i] == plain.pictures[f].planes[i]);
  }
  const Decoded patches = decode(gpu, kPatches);
  const Decoded full = decode(gpu, kFullRange);
  CHECK(patches.status.ok() && full.status.ok());
  CHECK(full.pictures.size() == static_cast<std::size_t>(kFrames));
  CHECK(full.pictures.size() == patches.pictures.size());
  for (std::size_t f = 0; f < full.pictures.size(); ++f) {
    const Picture& p = full.pictures[f];
    CHECK(p.meta.matrix == sensor::VideoColorMatrix::Bt709);
    CHECK(p.meta.full_range);
    for (int i = 0; i < 3; ++i)
      CHECK(p.planes[i] == patches.pictures[f].planes[i]);
  }
  return check_clip_shape(decode(gpu, kPatches, femto));
}

// reset(): the pictures held for display are dropped, a stream ended with
// size 0 takes data again, and the next key frame decodes as the first.
int test_reset(const Gpu& gpu) {
  const AccessUnits b_frames = access_units(kRefused);
  const AccessUnits patches = access_units(kPatches);
  auto decoder = HevcDecoder::create(gpu.options());
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
        vkc::Status::Code::InvalidArgument);  // after the end
  CHECK(decoder->reset().ok());
  return check_clip_shape(decode_clip(decoder.value(), patches, gpu));
}

// The cropped clip is the uncropped one's pixels, 16 right and 16 down, from
// NVDEC. VideoToolbox cannot crop there (FFmpeg sizes its output at the
// display size, filled from the coded corner), so it refuses the stream, and
// a reset does not undo the refusal.
int test_cropped(const Gpu& gpu) {
  auto decoder = HevcDecoder::create(gpu.options());
  CHECK(decoder.ok());
  const Decoded cropped =
      decode_clip(decoder.value(), access_units(kCropped), gpu);
#if defined(__APPLE__)
  CHECK(cropped.pictures.empty());
  CHECK(cropped.status.domain() == vkc::Status::Code::Unsupported);
  CHECK(decoder->reset().ok());
  const AccessUnits units = access_units(kCropped);
  CHECK(decoder->send(units[0].data(), units[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
#else
  constexpr int kCropLeft = 16;  // the cropped clip's window
  constexpr int kCropTop = 16;
  CHECK(cropped.status.ok());
  const Decoded full = decode(gpu, kPatches);
  CHECK(full.status.ok());
  CHECK(cropped.pictures.size() == full.pictures.size());
  const int width = kWidth - kCropLeft;
  const int height = kHeight - kCropTop;
  for (std::size_t f = 0; f < cropped.pictures.size(); ++f) {
    const Picture& c = cropped.pictures[f];
    const Picture& u = full.pictures[f];
    if (check_on_device(c.meta) != 0) return 1;
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
#endif
  return 0;
}

#if defined(__APPLE__)
// The crop prefix is not proof of a valid SPS. Truncate the committed SPS
// after its crop offsets but before the remaining syntax, retaining its VPS.
// Check recovery both with and without reset, and with the rest of the access
// unit still present. No rejected bytes may latch a permanent refusal.
int test_malformed_cropped_sps(const Gpu& gpu) {
  const AccessUnits cropped = access_units(kCropped);
  const AccessUnits patches = access_units(kPatches);
  CHECK(!cropped.empty());
  const auto& unit = cropped[0];
  std::size_t sps = 0;
  std::size_t end = 0;
  for (std::size_t i = 0; i + 3 < unit.size(); ++i) {
    if (unit[i] != 0 || unit[i + 1] != 0 || unit[i + 2] != 1) continue;
    if (sps != 0) {
      end = i;
      break;
    }
    if (((unit[i + 3] >> 1) & 0x3f) == 33) sps = i + 3;
  }
  CHECK(sps != 0 && end > sps + 30);
  for (const std::size_t kept : {25u, 30u}) {
    for (const bool slices : {false, true}) {
      auto damaged = std::vector<std::uint8_t>(
          unit.begin(), unit.begin() + static_cast<std::ptrdiff_t>(sps + kept));
      if (slices)
        damaged.insert(damaged.end(),
                       unit.begin() + static_cast<std::ptrdiff_t>(end),
                       unit.end());
      for (const bool reset : {false, true}) {
        auto decoder = HevcDecoder::create(gpu.options());
        CHECK(decoder.ok());
        CHECK(decoder->send(damaged.data(), damaged.size(), 0).domain() ==
              vkc::Status::Code::IoError);
        auto waiting = decoder->receive();
        CHECK(waiting.ok() && !waiting.value());
        if (reset) CHECK(decoder->reset().ok());
        CHECK(check_clip_shape(decode_clip(*decoder, patches, gpu)) == 0);
      }
    }
  }

  // A complete cropped SPS sent without a slice still refuses the stream;
  // resetting must not turn a genuine refusal into an accepted crop.
  auto decoder = HevcDecoder::create(gpu.options());
  CHECK(decoder.ok());
  CHECK(decoder->send(unit.data(), end, 0).domain() ==
        vkc::Status::Code::Unsupported);
  CHECK(decoder->reset().ok());
  CHECK(decoder->send(patches[0].data(), patches[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
  return 0;
}
#endif

// The refused clip: 8 frames with B-frames, then 2 of 4:0:0 grey that no
// hardware path hands out. The pictures decoded before the grey come out, in
// display order, each a distinct pts sent before it; then the refusal, which
// stands after a reset.
int test_refused(const Gpu& gpu) {
  const AccessUnits units = access_units(kRefused);
  CHECK(units.size() == static_cast<std::size_t>(kFrames + 2));
  auto decoder = HevcDecoder::create(gpu.options());
  CHECK(decoder.ok());
  const Decoded decoded = decode_clip(decoder.value(), units, gpu);
  CHECK(decoded.status.domain() == vkc::Status::Code::Unsupported);
  const auto& got = decoded.pictures;
  CHECK(!got.empty() && got.size() <= static_cast<std::size_t>(kFrames));
  std::vector<std::int64_t> out;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (check_on_device(got[i].meta) != 0) return 1;
    out.push_back(got[i].meta.pts);
    const std::size_t y = 36 * kWidth + 16;  // patch (0, 0)'s centre
    const int want = patch_y(patch(0, 0, static_cast<int>(i)));
    CHECK(std::abs(got[i].planes[0][y] - want) <= 2);
  }
  std::sort(out.begin(), out.end());
  CHECK(std::adjacent_find(out.begin(), out.end()) == out.end());
  CHECK(out.back() < pts_of(kFrames));
  CHECK(decoder->reset().ok());
  CHECK(decoder->send(units[0].data(), units[0].size(), 0).domain() ==
        vkc::Status::Code::Unsupported);
  return 0;
}

// No device, a device the hardware path cannot reach, and on NVDEC no
// allocator: each Unsupported.
int test_create_refusals(const Gpu& gpu) {
  CHECK(HevcDecoder::create({}).status().domain() ==
        vkc::Status::Code::Unsupported);
  vkc::Result<vkc::Device> bare =
      vr_test::bare_device(gpu.instance, gpu.device);
  CHECK(bare.ok());
  HevcDecoder::Options options = gpu.options();
  options.device = &bare.value();
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::Unsupported);
#if !defined(__APPLE__)
  options = gpu.options();
  options.allocator = nullptr;
  CHECK(HevcDecoder::create(options).status().domain() ==
        vkc::Status::Code::Unsupported);
#endif
  return 0;
}

int test_arguments(const Gpu& gpu) {
  auto decoder = HevcDecoder::create(gpu.options());
  CHECK(decoder.ok());
  auto nothing = decoder->receive();  // before any input
  CHECK(nothing.ok() && !nothing.value());
  CHECK(decoder->send(nullptr, 16, 0).domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(decoder->send(nullptr, 0, 0).ok());
  const std::uint8_t byte = 0;
  CHECK(decoder->send(&byte, 1, 0).domain() ==
        vkc::Status::Code::InvalidArgument);  // after the end
  auto drained = decoder->receive();
  CHECK(drained.ok() && !drained.value());
  return 0;
}

int test_moves(const Gpu& gpu) {
  auto created = HevcDecoder::create(gpu.options());
  CHECK(created.ok());
  HevcDecoder a = std::move(created).value();

  HevcDecoder b(std::move(a));
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is tested
  CHECK(a.send(nullptr, 0, 0).domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.receive().status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.reset().domain() == vkc::Status::Code::InvalidArgument);

  auto other = HevcDecoder::create(gpu.options());
  CHECK(other.ok());
  HevcDecoder c = std::move(other).value();
  c = std::move(b);  // over a live decoder
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from state is tested
  CHECK(b.reset().domain() == vkc::Status::Code::InvalidArgument);

  HevcDecoder* alias = &c;
  c = std::move(*alias);  // self-move
  return check_clip_shape(decode_clip(c, access_units(kPatches), gpu));
}

}  // namespace

int main() {
  // Unbuffered, so a crash inside FFmpeg or a driver still shows which clip
  // it was on.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (access_units(kPatches).size() != static_cast<std::size_t>(kFrames)) {
    std::fprintf(stderr, "FAIL: cannot split %s into %d access units\n",
                 kPatches, kFrames);
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
  const Gpu gpu{instance.value(), device.value(), allocator.value()};
  if (auto probe = HevcDecoder::create(gpu.options()); !probe) {
    if (probe.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", probe.status().message().c_str());
      return 1;
    }
    return vr_test::no_decoder(probe.status().message());
  }

  if (test_pattern(gpu) != 0) return 1;
  if (test_unlabelled_color(gpu) != 0) return 1;
  if (test_reset(gpu) != 0) return 1;
  if (test_cropped(gpu) != 0) return 1;
#if defined(__APPLE__)
  if (test_malformed_cropped_sps(gpu) != 0) return 1;
#endif
  if (test_refused(gpu) != 0) return 1;
  if (test_create_refusals(gpu) != 0) return 1;
  if (test_arguments(gpu) != 0) return 1;
  if (test_moves(gpu) != 0) return 1;
  std::puts("sensor_video_hevc: OK");
  return 0;
}
