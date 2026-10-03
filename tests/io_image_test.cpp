// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// Host-only image I/O contract tests. PNG fixtures below were encoded
// independently of the production writer: one unfiltered row, zlib DEFLATE.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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

// Each fixture is 2x1: gray16 [0,65535], gray8 [0,255], RGB16, and
// gray-alpha16.
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
constexpr std::uint8_t kRgb16[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x10, 0x02, 0x00, 0x00, 0x00, 0x2b, 0xd0, 0x34, 0x9e, 0x00, 0x00, 0x00,
    0x15, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0x60, 0x60, 0x60,
    0x64, 0x60, 0xfa, 0xff, 0xbf, 0x81, 0x81, 0xa1, 0x01, 0x00, 0x0d, 0x99,
    0x03, 0x02, 0x16, 0xe6, 0x79, 0xc4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
    0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};
constexpr std::uint8_t kGrayAlpha16[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
    0x00, 0x01, 0x10, 0x04, 0x00, 0x00, 0x00, 0x0e, 0xbb, 0x6b, 0x42,
    0x00, 0x00, 0x00, 0x11, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63,
    0x60, 0xe0, 0xf9, 0xff, 0x9f, 0x41, 0xe9, 0xff, 0x7f, 0x00, 0x0e,
    0xb5, 0x04, 0x2b, 0xba, 0xfe, 0x3f, 0x8b, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
};

}  // namespace

int main() {
  Scratch scratch;
  const auto color_path = (scratch.path / "color.png").string();
  const auto depth_path = (scratch.path / "depth.png").string();
  const auto invalid_path = (scratch.path / "invalid.png").string();
  const std::array<std::uint8_t, 16> rgba{128, 19, 231, 0,   9, 241, 17, 127,
                                          255, 0,  128, 255, 1, 2,   3,  64};
  CHECK(vr::io::write_png_rgba8(color_path, rgba.data(), rgba.size(), 2, 2));
  auto color = vr::io::load_color_packed(color_path, 2, 2);
  CHECK(color && color->size() == 4);
  CHECK(color.value() == std::vector<std::uint32_t>(
                             {0x00e71380, 0x0011f109, 0x008000ff, 0x00030201}));
  const auto encoded = read_bytes(color_path);
  CHECK(encoded.size() > 26 && encoded[24] == 8 && encoded[25] == 6);
  // Invalid dimensions/buffer/path must leave an existing destination intact.
  CHECK(!vr::io::write_png_rgba8(color_path, nullptr, rgba.size(), 2, 2));
  CHECK(
      !vr::io::write_png_rgba8(color_path, rgba.data(), rgba.size() - 1, 2, 2));
  CHECK(!vr::io::write_png_rgba8(color_path, rgba.data(), rgba.size(), 0, 2));
  CHECK(!vr::io::write_png_rgba8(color_path, rgba.data(), rgba.size(), 2, 0));
  CHECK(!vr::io::write_png_rgba8(color_path, rgba.data(),
                                 std::numeric_limits<std::size_t>::max(),
                                 std::numeric_limits<std::uint32_t>::max(), 2));
  // A row can fit in int while stb's signed-byte filter-score sum cannot.
  CHECK(!vr::io::write_png_rgba8(color_path, rgba.data(),
                                 std::numeric_limits<std::size_t>::max(),
                                 1u << 23, 1));
  CHECK(!vr::io::write_png_rgba8(color_path + std::string("\0tail", 5),
                                 rgba.data(), rgba.size(), 2, 2));
  CHECK(read_bytes(color_path) == encoded);
  CHECK(!vr::io::write_png_rgba8("", rgba.data(), rgba.size(), 2, 2));
  CHECK(!vr::io::write_png_rgba8(scratch.path.string(), rgba.data(),
                                 rgba.size(), 2, 2));
  CHECK(
      !vr::io::write_png_rgba8((scratch.path / "missing" / "out.png").string(),
                               rgba.data(), rgba.size(), 2, 2));
  if (fs::exists("/dev/full")) {
    CHECK(
        !vr::io::write_png_rgba8("/dev/full", rgba.data(), rgba.size(), 2, 2));
  }

  CHECK(!vr::io::load_color_packed(color_path, 1, 2));
  CHECK(!vr::io::load_color_packed(color_path, 0, 2));
  CHECK(!vr::io::load_color_packed(color_path, 2, 0));
  CHECK(!vr::io::load_color_packed(
      color_path, std::numeric_limits<std::uint32_t>::max(), 2));
  CHECK(!vr::io::load_color_packed("", 2, 2));
  CHECK(
      !vr::io::load_color_packed(color_path + std::string("\0tail", 5), 2, 2));
  CHECK(!vr::io::load_color_packed(invalid_path, 2, 2));
  constexpr std::uint8_t bad[] = {'n', 'o', 't', ' ', 'p', 'n', 'g'};
  CHECK(write_fixture(invalid_path, bad));
  CHECK(!vr::io::load_color_packed(invalid_path, 2, 2));
  CHECK(!vr::io::load_depth_metres(invalid_path, 2, 1, 1.0f));

  CHECK(write_fixture(depth_path, kDepth16));
  auto depth = vr::io::load_depth_metres(depth_path, 2, 1, 6553.5f);
  CHECK(depth && depth->size() == 2);
  CHECK(depth.value()[0] == 0.0f && depth.value()[1] == 10.0f);
  CHECK(!vr::io::load_depth_metres(depth_path, 1, 1, 6553.5f));
  CHECK(!vr::io::load_depth_metres(depth_path, 0, 1, 6553.5f));
  for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::denorm_min()}) {
    CHECK(!vr::io::load_depth_metres(depth_path, 2, 1, scale));
  }
  auto tiny = vr::io::load_depth_metres(depth_path, 2, 1,
                                        std::numeric_limits<float>::max());
  CHECK(tiny && tiny.value()[0] == 0.0f && tiny.value()[1] > 0.0f &&
        std::isfinite(tiny.value()[1]));
  CHECK(write_fixture(depth_path, kDepth8));
  CHECK(!vr::io::load_depth_metres(depth_path, 2, 1, 1.0f));
  auto gray = vr::io::load_color_packed(depth_path, 2, 1);
  CHECK(gray && gray.value() == std::vector<std::uint32_t>({0, 0x00ffffff}));
  CHECK(write_fixture(depth_path, kRgb16));
  CHECK(!vr::io::load_depth_metres(depth_path, 2, 1, 1.0f));
  auto rgb16 = vr::io::load_color_packed(depth_path, 2, 1);
  CHECK(rgb16 && rgb16.value() == std::vector<std::uint32_t>({0, 0x000080ff}));
  CHECK(write_fixture(depth_path, kGrayAlpha16));
  CHECK(!vr::io::load_depth_metres(depth_path, 2, 1, 1.0f));
  CHECK(!vr::io::load_depth_metres(color_path, 2, 2, 1.0f));
  // A recognized PNG header followed by truncated image data fails decoding.
  fs::resize_file(depth_path, 40);
  CHECK(!vr::io::load_color_packed(depth_path, 2, 1));
  CHECK(!vr::io::load_depth_metres(depth_path, 2, 1, 1.0f));
  std::puts("io image tests passed");
  return 0;
}
