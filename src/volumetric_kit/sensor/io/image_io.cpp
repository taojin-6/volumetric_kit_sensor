// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/io/image_io.hpp"

#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <new>

#include "stb_backend.hpp"

namespace volumetric_kit::recon::io {
namespace {

namespace stb = detail::stb;

struct StbiFree {
  void operator()(void* p) const noexcept { stb::free_image(p); }
};

Status check_path(const std::string& path) {
  if (path.empty() || path.find('\0') != std::string::npos) {
    return Status::invalid_argument("image I/O: empty path or embedded NUL");
  }
  return {};
}

Status check_dimensions(std::uint32_t w, std::uint32_t h) {
  // stb uses signed int products internally. Output arrays use four bytes per
  // pixel, and these limits also keep their size representable on 32-bit hosts.
  constexpr auto limit =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  if (w == 0 || h == 0 || static_cast<std::uint64_t>(w) * h > limit / 4) {
    return Status::invalid_argument(
        "image I/O: invalid or excessive dimensions");
  }
  return {};
}

Status inspect(const std::string& path, std::uint32_t expected_w,
               std::uint32_t expected_h, int& channels) {
  VR_TRY(check_path(path));
  VR_TRY(check_dimensions(expected_w, expected_h));
  int w = 0;
  int h = 0;
  if (!stb::info(path.c_str(), &w, &h, &channels)) {
    return Status::io_error("image I/O: cannot inspect " + path);
  }
  if (w != static_cast<int>(expected_w) || h != static_cast<int>(expected_h)) {
    return Status::invalid_argument("image I/O: unexpected image dimensions");
  }
  return {};
}

struct PngOutput {
  std::ofstream stream;
  bool failed = false;
};

// Never throw through stb: it must finish releasing its own encoding buffers.
void write_png_bytes(void* context, void* data, int size) noexcept {
  auto& out = *static_cast<PngOutput*>(context);
  if (out.failed) return;
  try {
    out.stream.write(static_cast<const char*>(data), size);
    out.failed = !out.stream;
  } catch (...) {
    out.failed = true;
  }
}

}  // namespace

Result<std::vector<std::uint32_t>> load_color_packed(
    const std::string& path, std::uint32_t expected_w,
    std::uint32_t expected_h) try {
  int channels = 0;
  VR_TRY(inspect(path, expected_w, expected_h, channels));
  int w = 0;
  int h = 0;
  std::unique_ptr<std::uint8_t, StbiFree> pixels(
      stb::load8(path.c_str(), &w, &h, &channels, 3));
  if (!pixels) {
    return Status::io_error("load_color_packed: cannot decode " + path);
  }
  // Recheck the decoded dimensions in case the file changed since inspection.
  if (w != static_cast<int>(expected_w) || h != static_cast<int>(expected_h)) {
    return Status::invalid_argument("load_color_packed: unexpected dimensions");
  }
  const std::size_t count = static_cast<std::size_t>(w) * h;
  std::vector<std::uint32_t> packed(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t r = pixels.get()[i * 3];
    const std::uint32_t g = pixels.get()[i * 3 + 1];
    const std::uint32_t b = pixels.get()[i * 3 + 2];
    packed[i] = r | (g << 8) | (b << 16);
  }
  return packed;
} catch (const std::bad_alloc&) {
  return Status::out_of_memory({});
} catch (...) {
  return Status::io_error({});
}

Result<std::vector<float>> load_depth_metres(const std::string& path,
                                             std::uint32_t expected_w,
                                             std::uint32_t expected_h,
                                             float depth_scale) try {
  if (!std::isfinite(depth_scale) || !(depth_scale > 0.0f)) {
    return Status::invalid_argument("load_depth_metres: invalid depth scale");
  }
  int channels = 0;
  VR_TRY(inspect(path, expected_w, expected_h, channels));
  if (channels != 1 || !stb::is_16_bit(path.c_str())) {
    return Status::invalid_argument(
        "load_depth_metres: expected a 16-bit grayscale PNG");
  }
  int w = 0;
  int h = 0;
  std::unique_ptr<std::uint16_t, StbiFree> raw(
      stb::load16(path.c_str(), &w, &h, &channels, 0));
  if (!raw) {
    return Status::io_error("load_depth_metres: cannot decode " + path);
  }
  if (w != static_cast<int>(expected_w) || h != static_cast<int>(expected_h) ||
      channels != 1) {
    return Status::invalid_argument(
        "load_depth_metres: unexpected image layout");
  }
  const std::size_t count = static_cast<std::size_t>(w) * h;
  std::vector<float> metres(count);
  for (std::size_t i = 0; i < count; ++i) {
    // Divide in double: an overflowing reciprocal must not turn raw zero into
    // NaN, and a small but valid divisor can still represent small samples.
    const double depth = static_cast<double>(raw.get()[i]) / depth_scale;
    if (depth > std::numeric_limits<float>::max()) {
      return Status::invalid_argument(
          "load_depth_metres: depth overflows float");
    }
    metres[i] = static_cast<float>(depth);
  }
  return metres;
} catch (const std::bad_alloc&) {
  return Status::out_of_memory({});
} catch (...) {
  return Status::io_error({});
}

Status write_png_rgba8(const std::string& path, const std::uint8_t* pixels,
                       std::size_t byte_count, std::uint32_t width,
                       std::uint32_t height) try {
  VR_TRY(check_path(path));
  VR_TRY(check_dimensions(width, height));
  const std::size_t stride = static_cast<std::size_t>(width) * 4;
  const std::size_t bytes = stride * height;
  const auto int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  // stb's filter score sums abs(signed byte) in int. Its compressed buffer
  // grows geometrically, with signed-int length/capacity and PNG framing.
  if (pixels == nullptr || byte_count < bytes || stride > int_max / 128 ||
      (stride + 1) * height > int_max / 2 - 64) {
    return Status::invalid_argument(
        "write_png_rgba8: invalid buffer or extent");
  }
  PngOutput out;
  out.stream.open(path, std::ios::binary);
  if (!out.stream) {
    return Status::io_error("write_png_rgba8: cannot open " + path);
  }
  const bool result = stb::write_png(
      write_png_bytes, &out, static_cast<int>(width), static_cast<int>(height),
      4, pixels, static_cast<int>(stride));
  out.stream.close();
  if (!result || out.failed || !out.stream) {
    return Status::io_error("write_png_rgba8: encode or write failed for " +
                            path);
  }
  return {};
} catch (const std::bad_alloc&) {
  return Status::out_of_memory({});
} catch (...) {
  return Status::io_error({});
}

}  // namespace volumetric_kit::recon::io
