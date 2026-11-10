// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/v210_rgb.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "volumetric_kit/sensor/chroma_location.hpp"

namespace volumetric_kit::sensor {
namespace {

core::Status refuse(const char* why) {
  return core::Status::invalid_argument(std::string("v210_to_rgb8: ") + why);
}

// The little-endian word at `p`, whatever its alignment and the host's order.
std::uint32_t word_at(const std::uint8_t* p) noexcept {
  return std::uint32_t{p[0]} | std::uint32_t{p[1]} << 8 |
         std::uint32_t{p[2]} << 16 | std::uint32_t{p[3]} << 24;
}

// `picture` checked as `v210_to_rgb8` documents it, before anything is
// written; the bytes its rows end at past the host pointer are counted in 64
// bits.
core::Status check(const YuvImage& picture, const std::uint8_t* rgb,
                   std::size_t rgb_bytes) {
  if (picture.layout != YuvLayout::V210) {
    return refuse("the picture is not V210");
  }
  if (picture.host == nullptr) {
    return refuse("the picture is not on the host");
  }
  if (picture.device != nullptr || picture.image[0] != nullptr ||
      picture.image[1] != nullptr) {
    return refuse(
        "the picture's planes must be on the host alone, not also in a "
        "buffer or images");
  }
  if (picture.width == 0 || picture.height == 0) {
    return refuse("the picture is empty");
  }
  if (picture.width % 2 != 0) {
    return refuse("V210 pairs pixels, so its width must be even");
  }
  const std::uint64_t row_bytes = v210_row_bytes(picture.width);
  if (picture.stride[0] < row_bytes) {
    return refuse("the picture's stride is shorter than its rows");
  }
  if (picture.offset[0] % 4 != 0 || picture.stride[0] % 4 != 0) {
    return refuse(
        "V210's rows are 32-bit words, so its offset and stride must be "
        "multiples of 4");
  }
  constexpr std::uint64_t kAll = std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t stride = picture.stride[0];
  if (picture.offset[0] > kAll - row_bytes ||
      (picture.height - 1) > (kAll - row_bytes - picture.offset[0]) / stride) {
    return refuse("the picture's rows run past 64 bits");
  }
  // Its 10-bit codes are read as code / 4 on the 8-bit scale, which only
  // limited range's levels keep exact, as the colour pass refuses it.
  if (picture.full_range) {
    return refuse("the picture is V210, which is limited range (SDI's)");
  }
  if (!(std::isfinite(picture.kr) && std::isfinite(picture.kb) &&
        picture.kr > 0.0f && picture.kb > 0.0f &&
        picture.kr + picture.kb < 1.0f)) {
    return refuse("the matrix's kr and kb must be positive and sum below 1");
  }
  if (static_cast<unsigned>(picture.chroma_location) >
      static_cast<unsigned>(ChromaLocation::Bottom)) {
    return refuse("unknown chroma location");
  }
  if (rgb == nullptr) return refuse("null output");
  const std::uint64_t pixels =
      std::uint64_t{picture.width} * std::uint64_t{picture.height};
  if (pixels > kAll / 3 || pixels * 3 > rgb_bytes) {
    return refuse(("the output holds " + std::to_string(rgb_bytes) +
                   " bytes, fewer than 3 a pixel")
                      .c_str());
  }
  return {};
}

// GLSL's mix: x (1 - a) + y a.
float mix(float x, float y, float a) noexcept { return x * (1.0f - a) + y * a; }

// undistort_color.glsl's code: a level, clamped to [0, 1], as 8 bits, half up.
std::uint8_t code(float c) noexcept {
  return static_cast<std::uint8_t>(std::min(std::max(c, 0.0f), 1.0f) * 255.0f +
                                   0.5f);
}

}  // namespace

core::Status v210_to_rgb8(const YuvImage& picture, std::uint8_t* rgb,
                          std::size_t rgb_bytes) {
  VKC_TRY(check(picture, rgb, rgb_bytes));
  try {
    const std::uint32_t w = picture.width;
    const std::uint32_t h = picture.height;
    const std::uint32_t cw = w / 2;
    const std::uint32_t groups = (w + 5) / 6;
    // Each pixel's two chroma samples, and the weight of the second, are
    // where the pass's bilinear_taps puts them for a pixel left where it is:
    // chroma coordinate (x - chroma_x) / 2, clamped to the row's samples.
    // chroma_x is 0 or 1/2, so that is 2 x - 2 chroma_x quarter samples,
    // worked out in integers: exact at any width, where float loses x's low
    // bits past 2^24 pixels across (past 2^23 centred) and picks the wrong
    // samples, at some widths one past the row. Every row takes the same:
    // 4:2:2 has chroma on every row.
    const std::uint64_t centred =
        chroma_offset(picture.chroma_location)[0] != 0.0f ? 1 : 0;
    const std::uint64_t last_quarter = std::uint64_t{cw - 1} * 4;
    // One row's samples on the 8-bit scale, code / 4, a whole last group's
    // worth (the samples past the width unused).
    std::vector<float> y(std::size_t{groups} * 6), cb(std::size_t{groups} * 3),
        cr(std::size_t{groups} * 3);
    const float kr = picture.kr;
    const float kb = picture.kb;
    const float kg = 1.0f - kr - kb;
    const float r_from_pr = 2.0f * (1.0f - kr);
    const float b_from_pb = 2.0f * (1.0f - kb);
    const std::uint8_t* row = picture.host + picture.offset[0];
    for (std::uint32_t r = 0; r < h; ++r, row += picture.stride[0]) {
      // Cb0 Y0 Cr0 | Y1 Cb2 Y2 | Cr2 Y3 Cb4 | Y4 Cr4 Y5, from the low bits.
      for (std::uint32_t g = 0; g < groups; ++g) {
        const std::uint8_t* at = row + std::size_t{g} * 16;
        std::uint32_t s[12];
        for (int k = 0; k < 4; ++k) {
          const std::uint32_t word = word_at(at + 4 * k);
          s[3 * k] = word & 0x3FFu;
          s[3 * k + 1] = (word >> 10) & 0x3FFu;
          s[3 * k + 2] = (word >> 20) & 0x3FFu;
        }
        float* yg = &y[std::size_t{g} * 6];
        float* cbg = &cb[std::size_t{g} * 3];
        float* crg = &cr[std::size_t{g} * 3];
        for (int i = 0; i < 6; ++i) {
          yg[i] = static_cast<float>(s[2 * i + 1]) * 0.25f;
        }
        for (int i = 0; i < 3; ++i) {
          cbg[i] = static_cast<float>(s[4 * i]) * 0.25f;
          crg[i] = static_cast<float>(s[4 * i + 2]) * 0.25f;
        }
      }
      std::uint8_t* out = rgb + std::size_t{r} * w * 3;
      for (std::uint32_t x = 0; x < w; ++x) {
        const std::uint64_t twice = std::uint64_t{x} * 2;
        const std::uint64_t quarter =
            std::min(twice < centred ? 0 : twice - centred, last_quarter);
        const std::size_t lo = static_cast<std::size_t>(quarter / 4);
        const std::size_t hi = std::min<std::size_t>(lo + 1, cw - 1);
        const float f = static_cast<float>(quarter % 4) * 0.25f;
        // luma_level, ycbcr_to_word: limited range.
        const float luma = (y[x] - 16.0f) / 219.0f;
        const float pb = (mix(cb[lo], cb[hi], f) - 128.0f) / 224.0f;
        const float pr = (mix(cr[lo], cr[hi], f) - 128.0f) / 224.0f;
        const float red = luma + r_from_pr * pr;
        const float blue = luma + b_from_pb * pb;
        const float green = (luma - kr * red - kb * blue) / kg;
        out[3 * x] = code(red);
        out[3 * x + 1] = code(green);
        out[3 * x + 2] = code(blue);
      }
    }
  } catch (const std::bad_alloc&) {
    return core::Status::out_of_memory("v210_to_rgb8: out of host memory");
  }
  return {};
}

}  // namespace volumetric_kit::sensor
