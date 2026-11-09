// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// v210_to_rgb8 (v210_rgb.hpp), a V210 picture's 8-bit R'G'B' on the host, no
// device, held to BT.709's limited-range conversion of the 10-bit samples in
// double, its chroma sampled as the colour pass samples it, written out here
// apart from the implementation:
//   - levels: black 64 is 0 and white 940 is 255, exactly; codes past them
//     clamp; a channel that converts to a half code rounds up;
//   - BT.709's 100% bars come out 0 and 255 and its 75% bars within a code of
//     191 and 0, their codes from the standard's equations (the 100%
//     primaries' checked against the published codes);
//   - a chroma step pins where each pixel's chroma comes from: co-sited, an
//     even pixel's own sample and an odd one's mean of the two either side;
//     centred, 3/4 and 1/4; the last pixel clamped to the last sample;
//   - random 10-bit codes off the 8-bit grid at widths 2 to 4096 (last groups
//     of 2, 4 and 6 pixels) within half a code of the reference, which the
//     same codes cut to 8 bits miss by more; the same picture after a lead
//     and with padded rows of junk gives the same bytes, and nothing past
//     the output's pixels is written;
//   - every refusal, naming why and writing nothing.
// `--timing` (the manual entry sensor_v210_rgb_timing) prints the conversion's
// time on a noisy 3840x2160 picture, and its PNG's time and size, and asserts
// nothing more.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "v210.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/sensor/io/image_io.hpp"
#include "volumetric_kit/sensor/v210_rgb.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
using sensor::ChromaLocation;
using sensor::YuvImage;

static_assert(sensor::v210_row_bytes(3840) == 10240 &&
                  sensor::v210_row_bytes(3842) == 10256 &&
                  sensor::v210_row_bytes(6) == 16 &&
                  sensor::v210_row_bytes(8) == 32,
              "16 bytes each six pixels or part of six");

namespace {

constexpr double kKr = 0.2126, kKb = 0.0722;  // BT.709

// A picture's 10-bit samples: luma `w` a row, Cb and Cr `w / 2` a row, every
// row.
struct Samples {
  std::uint32_t w = 0, h = 0;
  std::vector<std::uint16_t> y, cb, cr;
  std::uint32_t cw() const { return w / 2; }
};

Samples flat(std::uint32_t w, std::uint32_t h, std::uint16_t y,
             std::uint16_t cb, std::uint16_t cr) {
  Samples s;
  s.w = w;
  s.h = h;
  s.y.assign(std::size_t{w} * h, y);
  s.cb.assign(std::size_t{w / 2} * h, cb);
  s.cr.assign(s.cb.size(), cr);
  return s;
}

// `s` packed (tests/v210.hpp) from byte `lead`, rows `stride` apart.
std::vector<std::uint8_t> pack(const Samples& s, std::size_t lead = 0,
                               std::size_t stride = 0) {
  return v210::pack(s.y, s.cb, s.cr, s.w, s.h, lead,
                    stride != 0 ? stride : v210::row_bytes(s.w));
}

// A picture of `bytes` as a DeckLink capture states one: BT.709, limited
// range, chroma co-sited.
YuvImage picture(const std::vector<std::uint8_t>& bytes, std::uint32_t w,
                 std::uint32_t h, std::size_t lead = 0,
                 std::size_t stride = 0) {
  YuvImage p;
  p.layout = sensor::YuvLayout::V210;
  p.host = bytes.data();
  p.offset[0] = lead;
  p.stride[0] = stride != 0 ? stride : v210::row_bytes(w);
  p.width = w;
  p.height = h;
  p.kr = static_cast<float>(kKr);
  p.kb = static_cast<float>(kKb);
  p.full_range = false;
  p.chroma_location = ChromaLocation::Left;
  return p;
}

// `p` converted; empty, the reason printed, if refused.
std::vector<std::uint8_t> convert(const YuvImage& p) {
  std::vector<std::uint8_t> rgb(std::size_t{p.width} * p.height * 3);
  const vkc::Status status = sensor::v210_to_rgb8(p, rgb.data(), rgb.size());
  if (!status.ok()) {
    std::fprintf(stderr, "%s\n", status.message().c_str());
    return {};
  }
  return rgb;
}

// Pixel (x, r)'s chroma sample of `c`, `cw` a row: co-sited, an even pixel's
// own and an odd one's mean of the two either side; centred, x = 2i takes
// 1/4 of sample i - 1 and 3/4 of sample i, x = 2i + 1 3/4 of i and 1/4 of
// i + 1; a sample past either end the end's.
double chroma(const std::vector<std::uint16_t>& c, std::uint32_t cw,
              std::uint32_t r, std::uint32_t x, bool centred,
              bool drop = false) {
  const auto at = [&](long i) {
    i = std::clamp<long>(i, 0, static_cast<long>(cw) - 1);
    const std::uint16_t v =
        c[std::size_t{r} * cw + static_cast<std::size_t>(i)];
    return static_cast<double>(drop ? v & ~3u : v);
  };
  const long i = x / 2;
  if (!centred) return x % 2 == 0 ? at(i) : (at(i) + at(i + 1)) / 2;
  return x % 2 == 0 ? (at(i - 1) + 3 * at(i)) / 4 : (3 * at(i) + at(i + 1)) / 4;
}

// BT.709 limited range of pixel (x, r) in double from its 10-bit samples:
// Y' = (Y - 64) / 876, Pb and Pr = (C - 512) / 896; each channel clamped,
// times 255, unrounded. `drop` cuts each sample to 8 bits first.
std::array<double, 3> reference(const Samples& s, std::uint32_t r,
                                std::uint32_t x, bool centred,
                                bool drop = false) {
  const std::uint16_t yc = s.y[std::size_t{r} * s.w + x];
  const double luma = ((drop ? yc & ~3u : yc) - 64.0) / 876.0;
  const double pb = (chroma(s.cb, s.cw(), r, x, centred, drop) - 512) / 896;
  const double pr = (chroma(s.cr, s.cw(), r, x, centred, drop) - 512) / 896;
  const double red = luma + 2 * (1 - kKr) * pr;
  const double blue = luma + 2 * (1 - kKb) * pb;
  const double green = (luma - kKr * red - kKb * blue) / (1 - kKr - kKb);
  std::array<double, 3> out = {red, green, blue};
  for (double& c : out) c = std::clamp(c, 0.0, 1.0) * 255.0;
  return out;
}

// The farthest any byte of `rgb` lies from the reference of `s`.
double worst(const Samples& s, const std::vector<std::uint8_t>& rgb,
             bool centred, bool drop = false) {
  double out = 0;
  for (std::uint32_t r = 0; r < s.h; ++r) {
    for (std::uint32_t x = 0; x < s.w; ++x) {
      const std::array<double, 3> want = reference(s, r, x, centred, drop);
      for (int c = 0; c < 3; ++c) {
        const double got = rgb[(std::size_t{r} * s.w + x) * 3 + c];
        out = std::max(out, std::fabs(got - want[c]));
      }
    }
  }
  return out;
}

// 10-bit codes from xorshift, none a multiple of 4: off the 8-bit grid.
struct OffGrid {
  std::uint32_t seed = 0x9E3779B9u;
  std::uint16_t operator()(std::uint16_t lo, std::uint16_t hi) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    auto c = static_cast<std::uint16_t>(lo + seed % (hi - lo + 1u));
    if (c % 4 == 0) c = static_cast<std::uint16_t>(c == hi ? c - 1 : c + 1);
    return c;
  }
};

Samples random(std::uint32_t w, std::uint32_t h, OffGrid& codes) {
  Samples s = flat(w, h, 0, 0, 0);
  for (std::uint16_t& c : s.y) c = codes(4, 1019);  // past black and white
  for (std::size_t i = 0; i < s.cb.size(); ++i) {
    s.cb[i] = codes(4, 1019);
    s.cr[i] = codes(4, 1019);
  }
  return s;
}

// Black and white exactly, past them clamped, a half rounded up.
int test_levels() {
  struct Level {
    std::uint16_t y;
    std::uint8_t rgb;
  };
  // 502: Y' = 438 / 876 = 1/2 exactly, so R' and B' (Pb and Pr 0) are
  // 127.5, half up to 128. G' is the matrix's float arithmetic, which may
  // leave it an ulp either side of the half.
  for (const Level l : {Level{64, 0}, Level{940, 255}, Level{4, 0},
                        Level{1019, 255}, Level{502, 128}}) {
    const Samples s = flat(6, 2, l.y, 512, 512);
    const std::vector<std::uint8_t> bytes = pack(s);
    const std::vector<std::uint8_t> rgb = convert(picture(bytes, 6, 2));
    CHECK(rgb.size() == 36);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
      if (l.y == 502 && i % 3 == 1) {
        CHECK(rgb[i] == 127 || rgb[i] == 128);
      } else {
        CHECK(rgb[i] == l.rgb);
      }
    }
  }
  std::printf("  levels: black 0, white 255, clamped past them, half up\n");
  return 0;
}

// BT.709's codes of R'G'B' at 10 bits, rounded: Y = 64 + 876 Y',
// C = 512 + 896 P.
std::array<std::uint16_t, 3> codes709(double r, double g, double b) {
  const double y = kKr * r + (1 - kKr - kKb) * g + kKb * b;
  const double pb = (b - y) / (2 * (1 - kKb));
  const double pr = (r - y) / (2 * (1 - kKr));
  return {static_cast<std::uint16_t>(std::lround(64 + 876 * y)),
          static_cast<std::uint16_t>(std::lround(512 + 896 * pb)),
          static_cast<std::uint16_t>(std::lround(512 + 896 * pr))};
}

// The 100% and 75% bars, each 12 pixels wide (six chroma samples) in one
// picture of two rows; every pixel's chroma inside its bar but the last of
// each, which is the next bar's half.
int test_bars() {
  // The published 10-bit BT.709 primaries (SMPTE RP 219).
  CHECK((codes709(1, 0, 0) == std::array<std::uint16_t, 3>{250, 409, 960}));
  CHECK((codes709(0, 1, 0) == std::array<std::uint16_t, 3>{691, 167, 105}));
  CHECK((codes709(0, 0, 1) == std::array<std::uint16_t, 3>{127, 960, 471}));
  const int bars[8][3] = {{1, 1, 1}, {1, 1, 0}, {0, 1, 1}, {0, 1, 0},
                          {1, 0, 1}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}};
  for (const double level : {1.0, 0.75}) {
    Samples s = flat(8 * 12, 2, 0, 0, 0);
    for (std::uint32_t r = 0; r < s.h; ++r) {
      for (std::uint32_t x = 0; x < s.w; ++x) {
        const int* bar = bars[x / 12];
        const auto c = codes709(level * bar[0], level * bar[1], level * bar[2]);
        s.y[std::size_t{r} * s.w + x] = c[0];
        if (x % 2 == 0) {
          s.cb[std::size_t{r} * s.cw() + x / 2] = c[1];
          s.cr[std::size_t{r} * s.cw() + x / 2] = c[2];
        }
      }
    }
    const std::vector<std::uint8_t> bytes = pack(s);
    const std::vector<std::uint8_t> rgb = convert(picture(bytes, s.w, s.h));
    CHECK(rgb.size() == std::size_t{s.w} * s.h * 3);
    for (std::uint32_t r = 0; r < s.h; ++r) {
      for (std::uint32_t x = 0; x < s.w; ++x) {
        if (x % 12 == 11 && x + 1 < s.w) continue;  // the next bar's half
        for (int c = 0; c < 3; ++c) {
          const double want = level * bars[x / 12][c] * 255.0;
          const int got = rgb[(std::size_t{r} * s.w + x) * 3 + c];
          if (level == 1.0) {
            CHECK(got == static_cast<int>(want));
          } else {
            CHECK(std::fabs(got - want) <= 1.0);
          }
        }
      }
    }
    CHECK(worst(s, rgb, false) <= 0.5 + 1e-3);
  }
  std::printf("  bars: 100%% exact, 75%% within a code of 191\n");
  return 0;
}

// Red's code at luma 200 for chroma Cr, Cb neutral, as the reference has it.
double red_at(double cr) {
  const double luma = (200 - 64.0) / 876.0;
  return std::clamp(luma + 2 * (1 - kKr) * (cr - 512) / 896, 0.0, 1.0) * 255.0;
}

// A step in Cr, its last sample apart, through both sitings: each pixel's
// red is that of the Cr written out for it.
int test_chroma_step() {
  Samples s = flat(12, 1, 200, 512, 512);
  s.cr = {512, 512, 512, 960, 960, 700};
  const std::vector<std::uint8_t> bytes = pack(s);
  struct Siting {
    ChromaLocation location;
    double cr[12];
  };
  for (const Siting& k :
       {Siting{ChromaLocation::Left,
               {512, 512, 512, 512, 512, 736, 960, 960, 960, 830, 700, 700}},
        Siting{ChromaLocation::TopLeft,
               {512, 512, 512, 512, 512, 736, 960, 960, 960, 830, 700, 700}},
        Siting{ChromaLocation::Center,
               {512, 512, 512, 512, 512, 624, 848, 960, 960, 895, 765, 700}},
        Siting{ChromaLocation::Bottom,
               {512, 512, 512, 512, 512, 624, 848, 960, 960, 895, 765, 700}}}) {
    YuvImage p = picture(bytes, 12, 1);
    p.chroma_location = k.location;
    const std::vector<std::uint8_t> rgb = convert(p);
    CHECK(rgb.size() == 36);
    for (std::uint32_t x = 0; x < 12; ++x) {
      CHECK(std::fabs(rgb[3 * x] - red_at(k.cr[x])) <= 0.5 + 1e-3);
      CHECK(rgb[3 * x + 2] == rgb[2]);  // blue: luma and Cb alone
    }
  }
  std::printf(
      "  chroma step: co-sited mean at odd pixels, centred 3/4 and 1/4, the "
      "last clamped\n");
  return 0;
}

// Random codes at widths ending in every partial group, within half a code
// of the reference at either siting, its two low bits read; after a lead and
// with padded rows, the same bytes, and nothing past the pixels written.
int test_codes_and_rows() {
  OffGrid codes;
  double worst_seen = 0, worst_cut = 0;
  for (const std::uint32_t w :
       {2u, 4u, 6u, 8u, 10u, 38u, 40u, 42u, 3840u, 3842u, 3844u, 4096u}) {
    const std::uint32_t h = w > 64 ? 2 : 5;
    const Samples s = random(w, h, codes);
    const std::vector<std::uint8_t> tight = pack(s);
    for (const bool centred : {false, true}) {
      YuvImage p = picture(tight, w, h);
      p.chroma_location =
          centred ? ChromaLocation::Center : ChromaLocation::Left;
      const std::vector<std::uint8_t> rgb = convert(p);
      CHECK(rgb.size() == std::size_t{w} * h * 3);
      const double off = worst(s, rgb, centred);
      if (off > 0.5 + 1e-3) {
        std::fprintf(stderr, "  %u wide, %s: %.4f codes off\n", w,
                     centred ? "centred" : "co-sited", off);
      }
      CHECK(off <= 0.5 + 1e-3);
      worst_seen = std::max(worst_seen, off);
      worst_cut = std::max(worst_cut, worst(s, rgb, centred, true));

      // From byte 20, rows 36 bytes past theirs, the gaps junk (0xCD).
      const std::size_t stride = v210::row_bytes(w) + 36;
      const std::vector<std::uint8_t> padded = pack(s, 20, stride);
      YuvImage q = picture(padded, w, h, 20, stride);
      q.chroma_location = p.chroma_location;
      CHECK(convert(q) == rgb);
    }
  }
  // An output larger than the picture: only its pixels are written.
  {
    const Samples s = random(8, 2, codes);
    const std::vector<std::uint8_t> bytes = pack(s);
    std::vector<std::uint8_t> rgb(8 * 2 * 3 + 5, 0xAB);
    CHECK(sensor::v210_to_rgb8(picture(bytes, 8, 2), rgb.data(), rgb.size())
              .ok());
    for (std::size_t i = 48; i < rgb.size(); ++i) CHECK(rgb[i] == 0xAB);
    CHECK(std::vector<std::uint8_t>(rgb.begin(), rgb.begin() + 48) ==
          convert(picture(bytes, 8, 2)));
  }
  std::printf(
      "  codes: 2 to 4096 wide, worst %.4f codes from BT.709 in double, %.4f "
      "from the codes cut to 8 bits; lead and padded rows the same\n",
      worst_seen, worst_cut);
  CHECK(worst_cut > 0.5 + 1e-3);
  return 0;
}

// Each refusal names why and writes nothing.
int test_refusals() {
  const Samples s = flat(12, 2, 502, 512, 512);
  const std::vector<std::uint8_t> bytes = pack(s, 0, 64);
  const YuvImage good = picture(bytes, 12, 2, 0, 64);
  std::vector<std::uint8_t> rgb(12 * 2 * 3, 0xAB);
  CHECK(sensor::v210_to_rgb8(good, rgb.data(), rgb.size()).ok());
  const auto refused = [&](const YuvImage& p, const char* why,
                           std::uint8_t* out, std::size_t out_bytes) {
    std::fill(rgb.begin(), rgb.end(), 0xAB);
    const vkc::Status status = sensor::v210_to_rgb8(p, out, out_bytes);
    const bool ok =
        status.domain() == vkc::Status::Code::InvalidArgument &&
        status.message().rfind("v210_to_rgb8: ", 0) == 0 &&
        status.message().find(why) != std::string::npos &&
        std::all_of(rgb.begin(), rgb.end(), [](auto b) { return b == 0xAB; });
    if (!ok) std::fprintf(stderr, "%s\n", status.message().c_str());
    return ok;
  };
  const auto bad = [&](const YuvImage& p, const char* why) {
    return refused(p, why, rgb.data(), rgb.size());
  };
  YuvImage p = good;
  p.layout = sensor::YuvLayout::Yuyv;
  CHECK(bad(p, "not V210"));
  p = good;
  p.host = nullptr;
  CHECK(bad(p, "not on the host"));
  p = good;
  p.device = std::make_shared<const vkc::Buffer>();
  CHECK(bad(p, "on the host alone"));
  p = good;
  p.width = 0;
  CHECK(bad(p, "empty"));
  p = good;
  p.height = 0;
  CHECK(bad(p, "empty"));
  p = good;
  p.width = 11;
  CHECK(bad(p, "width must be even"));
  p = good;
  p.stride[0] = 31;  // 12 pixels take 32
  CHECK(bad(p, "stride is shorter"));
  p = good;
  p.width = 26;  // 5 groups, 80 bytes, past the stride of 64
  CHECK(bad(p, "stride is shorter"));
  p = good;
  p.offset[0] = 2;
  CHECK(bad(p, "multiples of 4"));
  p = good;
  p.stride[0] = 66;
  CHECK(bad(p, "multiples of 4"));
  p = good;
  p.offset[0] = std::numeric_limits<std::uint64_t>::max() - 15;
  CHECK(bad(p, "past 64 bits"));
  p = good;
  p.offset[0] = 0;
  p.stride[0] = std::size_t{1} << 62;
  p.height = 5;
  CHECK(bad(p, "past 64 bits"));
  p = good;
  p.full_range = true;
  CHECK(bad(p, "limited range"));
  for (const std::array<float, 2> k :
       {std::array<float, 2>{0.0f, 0.0722f},
        {0.2126f, -0.1f},
        {0.6f, 0.4f},
        {std::nanf(""), 0.0722f},
        {0.2126f, std::numeric_limits<float>::infinity()}}) {
    p = good;
    p.kr = k[0];
    p.kb = k[1];
    CHECK(bad(p, "kr and kb"));
  }
  p = good;
  p.chroma_location = static_cast<ChromaLocation>(6);
  CHECK(bad(p, "unknown chroma location"));
  CHECK(refused(good, "null output", nullptr, rgb.size()));
  CHECK(refused(good, "fewer than 3 a pixel", rgb.data(), rgb.size() - 1));
  std::printf("  refusals: each named, nothing written\n");
  return 0;
}

// A noisy 3840x2160 picture, more like a camera's than a pattern: smooth
// shading, a few hundred patches of colour with hard edges, fine stripes,
// and noise of `sigma` codes on every sample (10-bit; 4 is about one 8-bit
// code). Deterministic.
Samples scene(double sigma) {
  const std::uint32_t w = 3840, h = 2160;
  Samples s = flat(w, h, 0, 0, 0);
  std::uint32_t seed = 12345;
  const auto uniform = [&seed] {
    seed = seed * 1664525u + 1013904223u;
    return (seed >> 8) * (1.0 / 16777216.0);
  };
  const auto gauss = [&] {
    // The sum of four uniforms, scaled to unit variance.
    return (uniform() + uniform() + uniform() + uniform() - 2.0) *
           std::sqrt(3.0);
  };
  // Each pixel's patch, the last drawn over it, or none (-1).
  std::vector<std::array<double, 3>> colours(300);
  std::vector<int> patch(std::size_t{w} * h, -1);
  for (std::size_t i = 0; i < colours.size(); ++i) {
    const auto x0 = static_cast<std::uint32_t>(uniform() * w);
    const auto y0 = static_cast<std::uint32_t>(uniform() * h);
    const auto x1 = std::min<std::uint32_t>(
        w, x0 + 40 + static_cast<std::uint32_t>(uniform() * 600));
    const auto y1 = std::min<std::uint32_t>(
        h, y0 + 40 + static_cast<std::uint32_t>(uniform() * 400));
    colours[i] = {uniform(), uniform(), uniform()};
    for (std::uint32_t r = y0; r < y1; ++r) {
      std::fill(&patch[std::size_t{r} * w + x0],
                &patch[std::size_t{r} * w + x1], static_cast<int>(i));
    }
  }
  for (std::uint32_t r = 0; r < h; ++r) {
    for (std::uint32_t x = 0; x < w; ++x) {
      const double shade = 0.35 + 0.4 * std::sin(x * 0.0011 + r * 0.0007) *
                                      std::cos(r * 0.0013 - x * 0.0003);
      std::array<double, 3> rgb = {shade, shade * 0.95, shade * 0.85};
      if (const int i = patch[std::size_t{r} * w + x]; i >= 0) {
        for (int c = 0; c < 3; ++c) rgb[c] = colours[i][c] * shade * 1.6;
      }
      if ((x / 700 + r / 500) % 3 == 0) {
        const double stripe = 0.08 * std::sin(x * 0.9 + r * 0.2);
        for (double& c : rgb) c += stripe;
      }
      for (double& c : rgb) c = std::clamp(c, 0.0, 1.0);
      const auto c = codes709(rgb[0], rgb[1], rgb[2]);
      const auto noisy = [&](std::uint16_t v, double sd) {
        return static_cast<std::uint16_t>(
            std::clamp(std::lround(v + sd * gauss()), 4L, 1019L));
      };
      s.y[std::size_t{r} * w + x] = noisy(c[0], sigma);
      if (x % 2 == 0) {
        s.cb[std::size_t{r} * s.cw() + x / 2] = noisy(c[1], sigma / 2);
        s.cr[std::size_t{r} * s.cw() + x / 2] = noisy(c[2], sigma / 2);
      }
    }
  }
  return s;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// The conversion's time, p50 and the fastest of 15 after 2 warm-up runs; then
// the PNG's encode and write, 3 runs each, and its size; for two noise levels.
int timing() {
  const vkc::test::Scratch scratch("sensor_v210_rgb_");
  for (const double sigma : {4.0, 8.0}) {
    const Samples s = scene(sigma);
    const std::vector<std::uint8_t> bytes = pack(s);
    const YuvImage p = picture(bytes, s.w, s.h);
    std::vector<std::uint8_t> rgb(std::size_t{s.w} * s.h * 3);
    std::vector<double> times;
    for (int run = 0; run < 17; ++run) {
      const auto t0 = std::chrono::steady_clock::now();
      CHECK(sensor::v210_to_rgb8(p, rgb.data(), rgb.size()).ok());
      if (run >= 2) times.push_back(ms_since(t0));
    }
    std::sort(times.begin(), times.end());
    std::printf(
        "  %ux%u V210 to RGB8, noise %.0f codes: p50 %.1f ms, fastest %.1f "
        "ms (15 runs)\n",
        s.w, s.h, sigma, times[times.size() / 2], times.front());
    const std::string path =
        (scratch.path / ("still_" + std::to_string(int(sigma)) + ".png"))
            .string();
    std::vector<double> encodes, writes;
    std::size_t size = 0;
    for (int run = 0; run < 3; ++run) {
      auto t0 = std::chrono::steady_clock::now();
      const auto png =
          sensor::io::encode_png_rgb8(rgb.data(), rgb.size(), s.w, s.h);
      encodes.push_back(ms_since(t0));
      CHECK(png.ok());
      size = png->size();
      t0 = std::chrono::steady_clock::now();
      CHECK(sensor::io::write_png_rgb8(path, rgb.data(), rgb.size(), s.w, s.h)
                .ok());
      writes.push_back(ms_since(t0));
    }
    std::sort(encodes.begin(), encodes.end());
    std::sort(writes.begin(), writes.end());
    std::printf(
        "  PNG of it: %zu bytes (%.1f MB, %.0f%% of the %zu raw), encode p50 "
        "%.0f ms, write p50 %.0f ms (3 runs)\n",
        size, size / 1e6, 100.0 * size / rgb.size(), rgb.size(), encodes[1],
        writes[1]);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--timing") == 0) {
    CHECK(timing() == 0);
    std::puts("sensor_v210_rgb timing: done");
    return 0;
  }
  CHECK(test_levels() == 0);
  CHECK(test_bars() == 0);
  CHECK(test_chroma_step() == 0);
  CHECK(test_codes_and_rows() == 0);
  CHECK(test_refusals() == 0);
  std::puts("sensor_v210_rgb: OK");
  return 0;
}
