// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only image I/O contract tests. PNG fixtures below were encoded
// independently of the production writer: one unfiltered row, zlib DEFLATE.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_allocation_failure.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/recon/io/image_io.hpp"

namespace vkc = volumetric_kit::core;
namespace vr = volumetric_kit::recon;
namespace fs = std::filesystem;

namespace {

using vkc::test::read_bytes;

template <std::size_t N>
bool write_fixture(const std::string& path, const std::uint8_t (&data)[N]) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data), N);
  out.close();
  return static_cast<bool>(out);
}

// Each fixture is 2x1: gray16 [0,65535] and gray8 [0,255].
constexpr std::uint8_t kDepth16[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x10, 0x00, 0x00, 0x00, 0x00, 0x81, 0xd9, 0xfc, 0x15, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0x60, 0xf8, 0xff,
    0x1f, 0x00, 0x03, 0x02, 0x01, 0xff, 0xe6, 0x77, 0x0b, 0xae, 0x00, 0x00,
    0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};
constexpr std::uint8_t kDepth8[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x00, 0x00, 0x00, 0x00, 0xd1, 0x49, 0x20, 0x56, 0x00, 0x00, 0x00,
    0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0xf8, 0x0f, 0x00,
    0x01, 0x02, 0x01, 0x00, 0x42, 0xbe, 0xbc, 0x68, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

// IHDR of a 3x2 16-bit gray image: length, type, data, CRC. stb_image skips
// CRCs, so this CRC was computed independently (Python's zlib.crc32).
constexpr std::uint8_t kGray16Ihdr3x2[] = {
    0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x10, 0x00,
    0x00, 0x00, 0x00, 0xe8, 0x8f, 0xe5, 0x85,
};

bool has_code(const vkc::Status& status, vkc::Status::Code code) {
  return status.domain() == code;
}

// Every 16-bit edge value first, then noise, ramps, flat runs and products,
// so stb picks each of the five PNG filters on some row of the round trip's
// files (checked by inflating them with Python's zlib). Paeth's up-left
// neighbour is a sample's other byte two bytes back.
std::vector<std::uint16_t> depth_pattern(std::uint32_t w, std::uint32_t h) {
  std::vector<std::uint16_t> samples(static_cast<std::size_t>(w) * h);
  std::uint32_t state = 12345;
  for (std::uint32_t y = 0; y < h; ++y) {
    for (std::uint32_t x = 0; x < w; ++x) {
      state = state * 1664525u + 1013904223u;
      auto& s = samples[static_cast<std::size_t>(y) * w + x];
      switch ((x / 7 + y / 5) % 4) {
        case 0:
          s = static_cast<std::uint16_t>(state >> 16);
          break;
        case 1:
          s = static_cast<std::uint16_t>(x * 97 + y * 31);
          break;
        case 2:
          s = static_cast<std::uint16_t>(1000 + y);
          break;
        default:
          s = static_cast<std::uint16_t>(x * y * 37);
      }
    }
  }
  constexpr std::uint16_t kEdges[] = {0,      1,      255,    256,
                                      0x7fff, 0x8000, 0xfffe, 0xffff};
  std::copy_n(kEdges, std::min(std::size(kEdges), samples.size()),
              samples.begin());
  samples.back() = 0xffff;
  return samples;
}

// What write_png_u16 writes, load_depth_u16 reads back exactly, and it is
// what encode_png_u16 returns.
int u16_round_trip(const fs::path& dir) {
  struct Extent {
    std::uint32_t w, h;
  };
  for (const Extent e : {Extent{1, 1}, Extent{3, 2}, Extent{7, 5},
                         Extent{640, 576}, Extent{641, 577}}) {
    const auto samples = depth_pattern(e.w, e.h);
    const auto path =
        (dir / ("depth_" + std::to_string(e.w) + ".png")).string();
    CHECK(
        vr::io::write_png_u16(path, samples.data(), samples.size(), e.w, e.h));
    const auto decoded = vr::io::load_depth_u16(path, e.w, e.h);
    CHECK(decoded && decoded.value() == samples);
    const auto encoded =
        vr::io::encode_png_u16(samples.data(), samples.size(), e.w, e.h);
    CHECK(encoded && encoded.value() == read_bytes(path));
  }
  const std::uint16_t zero = 0;
  const auto path = (dir / "zero.png").string();
  CHECK(vr::io::write_png_u16(path, &zero, 1, 1, 1));
  const auto decoded = vr::io::load_depth_u16(path, 1, 1);
  CHECK(decoded && decoded.value() == std::vector<std::uint16_t>{0});
  return 0;
}

// PNG stores a sample's most significant byte first; a writer storing the
// low byte first would swap these. The 8-bit decode keeps the high byte.
int u16_big_endian(const fs::path& dir) {
  const std::uint16_t samples[] = {0x0102, 0xff00, 0x00ff};
  const auto path = (dir / "order.png").string();
  CHECK(vr::io::write_png_u16(path, samples, 3, 3, 1));
  const auto depth = vr::io::load_depth_u16(path, 3, 1);
  CHECK(depth &&
        depth.value() == std::vector<std::uint16_t>(samples, samples + 3));
  const auto color = vr::io::load_color_packed(path, 3, 1);
  CHECK(color &&
        color.value() == std::vector<std::uint32_t>({0x010101, 0xffffff, 0}));
  return 0;
}

// The file starts with the PNG signature and a 16-bit gray IHDR whose CRC
// matches it.
int u16_header() {
  constexpr std::uint8_t kSignature[] = {0x89, 0x50, 0x4e, 0x47,
                                         0x0d, 0x0a, 0x1a, 0x0a};
  const std::uint16_t samples[6] = {};
  const auto png = vr::io::encode_png_u16(samples, 6, 3, 2);
  CHECK(png && png.value().size() > 8 + std::size(kGray16Ihdr3x2));
  const auto& bytes = png.value();
  CHECK(
      std::equal(std::begin(kSignature), std::end(kSignature), bytes.begin()));
  CHECK(std::equal(std::begin(kGray16Ihdr3x2), std::end(kGray16Ihdr3x2),
                   bytes.begin() + 8));
  return 0;
}

// Invalid input is refused before anything is written.
int u16_refusals(const fs::path& dir) {
  using Code = vkc::Status::Code;
  const std::vector<std::uint16_t> samples(6, 1000);
  const auto refused = [&](const std::uint16_t* p, std::size_t n,
                           std::uint32_t w, std::uint32_t h) {
    const auto png = vr::io::encode_png_u16(p, n, w, h);
    return !png && has_code(png.status(), Code::InvalidArgument);
  };
  CHECK(refused(nullptr, 6, 3, 2));
  CHECK(refused(samples.data(), 5, 3, 2));
  CHECK(refused(samples.data(), 6, 0, 2));
  CHECK(refused(samples.data(), 6, 3, 0));
  // The count claims to cover the extent, so only the extent is refused;
  // nothing is read.
  constexpr auto kAll = std::numeric_limits<std::size_t>::max();
  CHECK(refused(samples.data(), kAll, 65536, 65536));
  // Only stb's int-sized filter and compressor buffers refuse this extent.
  CHECK(refused(samples.data(), kAll, 8192, 65535));
  // A row too long for stb's int filter score.
  const std::vector<std::uint16_t> row(std::size_t{1} << 23);
  CHECK(refused(row.data(), row.size(), 1u << 23, 1));
  // Each refusal names what it refuses.
  const auto message = [](const vkc::Result<std::vector<std::uint8_t>>& png) {
    return png.status().message();
  };
  CHECK(message(vr::io::encode_png_u16(nullptr, 6, 3, 2)) ==
        "encode_png_u16: null buffer");
  CHECK(message(vr::io::encode_png_u16(samples.data(), 5, 3, 2)) ==
        "encode_png_u16: 5 samples for 3x2");
  CHECK(message(vr::io::encode_png_u16(row.data(), row.size(), 1u << 23, 1)) ==
        "encode_png_u16: 8388608x1 exceeds the encoder's bounds");

  CHECK(has_code(vr::io::write_png_u16("", samples.data(), 6, 3, 2),
                 Code::InvalidArgument));
  CHECK(has_code(vr::io::write_png_u16(std::string("a\0b.png", 7),
                                       samples.data(), 6, 3, 2),
                 Code::InvalidArgument));
  const auto missing = (dir / "missing" / "d.png").string();
  const auto unwritable =
      vr::io::write_png_u16(missing, samples.data(), 6, 3, 2);
  CHECK(has_code(unwritable, Code::IoError) &&
        unwritable.message() == "write_png_u16: " + missing + ": cannot write");
  // A refused write leaves the existing file intact.
  const auto path = (dir / "kept.png").string();
  CHECK(vr::io::write_png_u16(path, samples.data(), 6, 3, 2));
  const auto kept = read_bytes(path);
  CHECK(!vr::io::write_png_u16(path, samples.data(), 5, 3, 2));
  CHECK(read_bytes(path) == kept);
  return 0;
}

// An allocation failure while encoding is a Status, not an exception, and a
// write that fails so leaves the existing file intact.
int u16_out_of_memory(const fs::path& dir) {
  using Code = vkc::Status::Code;
  constexpr std::uint32_t kW = 37;
  constexpr std::uint32_t kH = 29;
  constexpr std::size_t kBigEndian = std::size_t{kW} * kH * 2;
  const auto samples = depth_pattern(kW, kH);
  const auto png =
      vr::io::encode_png_u16(samples.data(), samples.size(), kW, kH);
  CHECK(png && png.value().size() != kBigEndian);
  const auto path = (dir / "oom.png").string();
  CHECK(vr::io::write_png_u16(path, samples.data(), samples.size(), kW, kH));
  const auto kept = read_bytes(path);
  // Fail the first allocation of `bytes` in `call`. Returns the OutOfMemory
  // message, or nothing if no such failure was injected and returned.
  auto& failure = vkc::test::allocation_failure;
  const auto fail = [&](std::size_t bytes, const auto& call) -> std::string {
    failure = {};
    failure.bytes = bytes;
    failure.armed = true;
    const vkc::Status status = call();
    const bool injected = failure.injected;
    failure = {};
    if (!injected || !has_code(status, Code::OutOfMemory)) return {};
    return status.message();
  };
  const auto encode = [&] {
    return vr::io::encode_png_u16(samples.data(), samples.size(), kW, kH)
        .status();
  };
  const auto write = [&] {
    return vr::io::write_png_u16(path, samples.data(), samples.size(), kW, kH);
  };
  // The big-endian copy of the samples, then the copy of stb's PNG.
  for (const std::size_t bytes : {kBigEndian, png.value().size()}) {
    CHECK(!fail(bytes, encode).empty());
    CHECK(!fail(bytes, write).empty());
    CHECK(read_bytes(path) == kept);
  }
  // A failed encode names the file being written.
  CHECK(fail(png.value().size(), write) ==
        "write_png_u16: " + path + ": cannot encode");
  return 0;
}

int rgba8_and_depth_fixtures(const fs::path& dir) {
  const auto color_path = (dir / "color.png").string();
  const auto depth_path = (dir / "depth.png").string();
  const std::array<std::uint8_t, 16> rgba{128, 19, 231, 0,   9, 241, 17, 127,
                                          255, 0,  128, 255, 1, 2,   3,  64};
  CHECK(vr::io::write_png_rgba8(color_path, rgba.data(), rgba.size(), 2, 2));
  auto color = vr::io::load_color_packed(color_path, 2, 2);
  CHECK(color &&
        color.value() == std::vector<std::uint32_t>(
                             {0x00e71380, 0x0011f109, 0x008000ff, 0x00030201}));
  CHECK(!vr::io::load_color_packed(color_path, 1, 2));
  // A refused write leaves the existing file intact.
  const auto encoded = read_bytes(color_path);
  CHECK(
      !vr::io::write_png_rgba8(color_path, rgba.data(), rgba.size() - 1, 2, 2));
  // A row too long for stb's int filter score, in a real buffer, so accepting
  // it would encode rather than read out of bounds.
  const std::vector<std::uint8_t> row(std::size_t{4} << 22);
  CHECK(!vr::io::write_png_rgba8(color_path, row.data(), row.size(), 1u << 22,
                                 1));
  CHECK(read_bytes(color_path) == encoded);

  CHECK(write_fixture(depth_path, kDepth16));
  auto depth = vr::io::load_depth_u16(depth_path, 2, 1);
  CHECK(depth && depth.value() == std::vector<std::uint16_t>({0, 65535}));
  CHECK(!vr::io::load_depth_u16(depth_path, 1, 1));
  // An 8-bit depth PNG would otherwise decode about 257x too far.
  CHECK(write_fixture(depth_path, kDepth8));
  CHECK(!vr::io::load_depth_u16(depth_path, 2, 1));
  return 0;
}

}  // namespace

int main() {
  vkc::test::Scratch scratch("recon_io_image_");
  CHECK(rgba8_and_depth_fixtures(scratch.path) == 0);
  CHECK(u16_round_trip(scratch.path) == 0);
  CHECK(u16_big_endian(scratch.path) == 0);
  CHECK(u16_header() == 0);
  CHECK(u16_refusals(scratch.path) == 0);
  CHECK(u16_out_of_memory(scratch.path) == 0);
  std::puts("io image tests passed");
  return 0;
}
