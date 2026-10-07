// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/image_io.hpp"

#include <fstream>
#include <limits>
#include <memory>

#include "errors.hpp"
#include "stb_backend.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::recon::io {
namespace {

namespace stb = detail::stb;

struct StbiFree {
  void operator()(void* p) const noexcept { stb::free_image(p); }
};

core::Status check_dimensions(const char* api, std::uint32_t w,
                              std::uint32_t h) {
  // stb uses signed int products internally. Output arrays use four bytes per
  // pixel, and these limits also keep their size representable on 32-bit hosts.
  constexpr auto limit =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  if (w == 0 || h == 0 || static_cast<std::uint64_t>(w) * h > limit / 4) {
    return core::Status::invalid_argument(std::string(api) +
                                          ": invalid or excessive dimensions");
  }
  return {};
}

// Read the encoded file once, then check its header against the expected
// dimensions before anything is decoded.
core::Result<std::vector<std::uint8_t>> read_image(const char* api,
                                                   const std::string& path,
                                                   std::uint32_t expected_w,
                                                   std::uint32_t expected_h,
                                                   int& channels) {
  VKC_TRY(detail::check_path(api, path));
  VKC_TRY(check_dimensions(api, expected_w, expected_h));
  const std::string where = std::string(api) + ": " + path;
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return core::Status::io_error(where + ": cannot open");
  const auto size = static_cast<std::streamoff>(in.tellg());
  if (size <= 0 || size > std::numeric_limits<int>::max()) {
    return core::Status::invalid_argument(where + ": empty or oversized file");
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!in) return core::Status::io_error(where + ": read failed");
  int w = 0;
  int h = 0;
  if (!stb::info(bytes.data(), static_cast<int>(size), &w, &h, &channels)) {
    return core::Status::io_error(where + ": " + stb::failure_reason());
  }
  if (w != static_cast<int>(expected_w) || h != static_cast<int>(expected_h)) {
    return core::Status::invalid_argument(
        where + ": " + std::to_string(w) + "x" + std::to_string(h) +
        " image, expected " + std::to_string(expected_w) + "x" +
        std::to_string(expected_h));
  }
  return bytes;
}

struct PngOutput {
  const std::string* path = nullptr;
  std::ofstream stream;
  bool failed = false;
};

// Never throw through stb: it must finish releasing its own encoding buffers.
void write_png_bytes(void* context, void* data, int size) noexcept {
  auto& out = *static_cast<PngOutput*>(context);
  if (out.failed) return;
  try {
    // stb calls this only after encoding succeeds, so an encoder failure
    // leaves an existing file untouched.
    if (!out.stream.is_open()) out.stream.open(*out.path, std::ios::binary);
    out.stream.write(static_cast<const char*>(data), size);
    out.failed = !out.stream;
  } catch (...) {
    out.failed = true;
  }
}

}  // namespace

core::Result<std::vector<std::uint32_t>> load_color_packed(
    const std::string& path, std::uint32_t expected_w,
    std::uint32_t expected_h) try {
  int channels = 0;
  VKC_ASSIGN(const auto encoded, read_image("load_color_packed", path,
                                            expected_w, expected_h, channels));
  int w = 0;
  int h = 0;
  std::unique_ptr<std::uint8_t, StbiFree> pixels(stb::load8(
      encoded.data(), static_cast<int>(encoded.size()), &w, &h, &channels, 3));
  if (!pixels) {
    return core::Status::io_error("load_color_packed: " + path + ": " +
                                  stb::failure_reason());
  }
  VKC_CHECK(
      w == static_cast<int>(expected_w) && h == static_cast<int>(expected_h),
      "stb decoded other dimensions than its header reported");
  const std::size_t count = static_cast<std::size_t>(w) * h;
  std::vector<std::uint32_t> packed(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t r = pixels.get()[i * 3];
    const std::uint32_t g = pixels.get()[i * 3 + 1];
    const std::uint32_t b = pixels.get()[i * 3 + 2];
    packed[i] = r | (g << 8) | (b << 16);
  }
  return packed;
} catch (...) {
  return detail::exception_status("load_color_packed");
}

core::Result<std::vector<std::uint16_t>> load_depth_u16(
    const std::string& path, std::uint32_t expected_w,
    std::uint32_t expected_h) try {
  int channels = 0;
  VKC_ASSIGN(const auto encoded, read_image("load_depth_u16", path, expected_w,
                                            expected_h, channels));
  const int size = static_cast<int>(encoded.size());
  if (channels != 1 || !stb::is_16_bit(encoded.data(), size)) {
    return core::Status::invalid_argument("load_depth_u16: " + path +
                                          ": expected a 16-bit grayscale PNG");
  }
  int w = 0;
  int h = 0;
  std::unique_ptr<std::uint16_t, StbiFree> raw(
      stb::load16(encoded.data(), size, &w, &h, &channels, 1));
  if (!raw) {
    return core::Status::io_error("load_depth_u16: " + path + ": " +
                                  stb::failure_reason());
  }
  VKC_CHECK(
      w == static_cast<int>(expected_w) && h == static_cast<int>(expected_h),
      "stb decoded other dimensions than its header reported");
  const std::size_t count = static_cast<std::size_t>(w) * h;
  return std::vector<std::uint16_t>(raw.get(), raw.get() + count);
} catch (...) {
  return detail::exception_status("load_depth_u16");
}

core::Status write_png_rgba8(const std::string& path,
                             const std::uint8_t* pixels, std::size_t byte_count,
                             std::uint32_t width, std::uint32_t height) try {
  VKC_TRY(detail::check_path("write_png_rgba8", path));
  VKC_TRY(check_dimensions("write_png_rgba8", width, height));
  const std::size_t stride = static_cast<std::size_t>(width) * 4;
  const std::size_t bytes = stride * height;
  const auto int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  // stb's filter score sums abs(signed byte) in int. Its compressed buffer
  // grows geometrically, with signed-int length/capacity and PNG framing.
  if (pixels == nullptr || byte_count < bytes || stride > int_max / 128 ||
      (stride + 1) * height > int_max / 2 - 64) {
    return core::Status::invalid_argument(
        "write_png_rgba8: invalid buffer or extent");
  }
  PngOutput out;
  out.path = &path;
  const bool encoded = stb::write_png(
      write_png_bytes, &out, static_cast<int>(width), static_cast<int>(height),
      4, pixels, static_cast<int>(stride));
  // stb fails only when its encoding buffers cannot be allocated.
  if (!encoded) {
    return core::Status::out_of_memory("write_png_rgba8: cannot encode " +
                                       path);
  }
  out.stream.close();
  if (out.failed || !out.stream) {
    return core::Status::io_error("write_png_rgba8: cannot write " + path);
  }
  return {};
} catch (...) {
  return detail::exception_status("write_png_rgba8");
}

}  // namespace volumetric_kit::recon::io
