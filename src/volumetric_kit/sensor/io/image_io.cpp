// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/image_io.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <utility>

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

// Whether stb's PNG encoder can take a tightly packed image of `pixel_bytes`
// per pixel. Its filter score sums abs(signed byte) in int. Its compressed
// buffer grows geometrically, with signed-int length/capacity and PNG framing.
bool fits_png_encoder(std::uint32_t width, std::uint32_t height,
                      std::uint64_t pixel_bytes) {
  const std::uint64_t stride = width * pixel_bytes;
  const auto int_max =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  return stride <= int_max / 128 && (stride + 1) * height <= int_max / 2 - 64;
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

struct PngBytes {
  std::vector<std::uint8_t> bytes;
  bool failed = false;
};

// Never throw through stb; it calls this once with the whole file.
void copy_png_bytes(void* context, void* data, int size) noexcept {
  auto& out = *static_cast<PngBytes*>(context);
  try {
    const auto* first = static_cast<const std::uint8_t*>(data);
    out.bytes.assign(first, first + size);
  } catch (...) {
    out.failed = true;
  }
}

// stb writes 8-bit samples only. Two 8-bit channels holding each sample's
// bytes, most significant first, filter and compress exactly as 16-bit gray
// does: PNG filters work on bytes, two per pixel either way. Only IHDR then
// differs: its bit depth, color type and CRC are rewritten.
core::Result<std::vector<std::uint8_t>> encode_gray16(
    const char* api, const std::uint16_t* samples, std::size_t sample_count,
    std::uint32_t width, std::uint32_t height) {
  VKC_TRY(check_dimensions(api, width, height));
  const std::size_t count = static_cast<std::size_t>(width) * height;
  if (samples == nullptr || sample_count < count ||
      !fits_png_encoder(width, height, 2)) {
    return core::Status::invalid_argument(std::string(api) +
                                          ": invalid buffer or extent");
  }
  std::vector<std::uint8_t> big_endian(count * 2);
  for (std::size_t i = 0; i < count; ++i) {
    big_endian[i * 2] = static_cast<std::uint8_t>(samples[i] >> 8);
    big_endian[i * 2 + 1] = static_cast<std::uint8_t>(samples[i] & 0xff);
  }
  PngBytes out;
  // stb fails only when its encoding buffers cannot be allocated.
  if (!stb::write_png(copy_png_bytes, &out, static_cast<int>(width),
                      static_cast<int>(height), 2, big_endian.data(),
                      static_cast<int>(width) * 2) ||
      out.failed) {
    return core::Status::out_of_memory(std::string(api) + ": cannot encode");
  }
  // The 8-byte signature, then IHDR: length, type, width, height, bit depth,
  // color type, compression, filter, interlace, and the CRC of type and data.
  constexpr std::size_t kType = 12;
  constexpr std::size_t kDepth = 24;
  constexpr std::size_t kColor = 25;
  constexpr std::size_t kCrc = 29;
  auto& png = out.bytes;
  VKC_CHECK(png.size() > kCrc + 4 && std::memcmp(&png[kType], "IHDR", 4) == 0 &&
                png[kDepth] == 8 && png[kColor] == 4,
            "stb wrote other than an 8-bit gray-alpha IHDR first");
  png[kDepth] = 16;
  png[kColor] = 0;  // Gray.
  const std::uint32_t crc =
      stb::png_crc(&png[kType], static_cast<int>(kCrc - kType));
  for (std::size_t i = 0; i < 4; ++i) {
    png[kCrc + i] = static_cast<std::uint8_t>(crc >> (24 - 8 * i));
  }
  return std::move(png);
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
  if (pixels == nullptr || byte_count < stride * height ||
      !fits_png_encoder(width, height, 4)) {
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

core::Result<std::vector<std::uint8_t>> encode_png_u16(
    const std::uint16_t* samples, std::size_t sample_count, std::uint32_t width,
    std::uint32_t height) try {
  return encode_gray16("encode_png_u16", samples, sample_count, width, height);
} catch (...) {
  return detail::exception_status("encode_png_u16");
}

core::Status write_png_u16(const std::string& path,
                           const std::uint16_t* samples,
                           std::size_t sample_count, std::uint32_t width,
                           std::uint32_t height) try {
  VKC_TRY(detail::check_path("write_png_u16", path));
  VKC_ASSIGN(const auto png, encode_gray16("write_png_u16", samples,
                                           sample_count, width, height));
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(png.data()),
            static_cast<std::streamsize>(png.size()));
  out.close();
  if (!out) {
    return core::Status::io_error("write_png_u16: cannot write " + path);
  }
  return {};
} catch (...) {
  return detail::exception_status("write_png_u16");
}

}  // namespace volumetric_kit::recon::io
