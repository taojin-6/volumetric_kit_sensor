// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only image I/O contract tests. PNG fixtures below were encoded
// independently of the production writer: one unfiltered row, zlib DEFLATE.

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include "volumetric_kit/recon/io/image_io.hpp"

namespace vr = volumetric_kit::recon;
namespace fs = std::filesystem;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

struct Scratch {
  fs::path path =
      fs::temp_directory_path() /
      ("recon_io_image_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Scratch() { fs::create_directory(path); }
  ~Scratch() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }
};

std::vector<std::uint8_t> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

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
}  // namespace

int main() {
  Scratch scratch;
  const auto color_path = (scratch.path / "color.png").string();
  const auto depth_path = (scratch.path / "depth.png").string();
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
  CHECK(read_bytes(color_path) == encoded);

  CHECK(write_fixture(depth_path, kDepth16));
  auto depth = vr::io::load_depth_u16(depth_path, 2, 1);
  CHECK(depth && depth.value() == std::vector<std::uint16_t>({0, 65535}));
  CHECK(!vr::io::load_depth_u16(depth_path, 1, 1));
  // An 8-bit depth PNG would otherwise decode about 257x too far.
  CHECK(write_fixture(depth_path, kDepth8));
  CHECK(!vr::io::load_depth_u16(depth_path, 2, 1));
  std::puts("io image tests passed");
  return 0;
}
