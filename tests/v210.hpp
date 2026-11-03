// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// V210 pictures for the tests (YuvLayout::V210): 10-bit 4:2:2 samples packed
// as SDI carries them, word by word from the format's table, apart from the
// shader's arithmetic.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace v210 {

// The bytes a row of `width` pixels takes: 16 each six pixels or part of six.
inline std::size_t row_bytes(std::uint32_t width) {
  return (std::size_t{width} + 5) / 6 * 16;
}

// A `width` x `height` picture's 10-bit luma `y` (`width` a row) and chroma
// `cb`, `cr` (`(width + 1) / 2` a row, every row), packed from byte `lead`,
// rows `stride` bytes apart, ending at the last row's last group; each six
// pixels' four little-endian words Cb0 Y0 Cr0 | Y1 Cb2 Y2 | Cr2 Y3 Cb4 |
// Y4 Cr4 Y5, the first sample in the low bits, samples past the width 0.
// Bytes outside the rows are 0xCD.
inline std::vector<std::uint8_t> pack(const std::vector<std::uint16_t>& y,
                                      const std::vector<std::uint16_t>& cb,
                                      const std::vector<std::uint16_t>& cr,
                                      std::uint32_t width, std::uint32_t height,
                                      std::size_t lead, std::size_t stride) {
  const std::size_t cw = (std::size_t{width} + 1) / 2;
  std::vector<std::uint8_t> out(lead + stride * (height - 1) + row_bytes(width),
                                0xCD);
  for (std::uint32_t r = 0; r < height; ++r) {
    const auto Y = [&](std::size_t x) -> std::uint32_t {
      return x < width ? y[r * std::size_t{width} + x] & 0x3FFu : 0u;
    };
    const auto B = [&](std::size_t i) -> std::uint32_t {
      return i < cw ? cb[r * cw + i] & 0x3FFu : 0u;
    };
    const auto R = [&](std::size_t i) -> std::uint32_t {
      return i < cw ? cr[r * cw + i] & 0x3FFu : 0u;
    };
    for (std::size_t g = 0; 6 * g < width; ++g) {
      const std::size_t x = 6 * g, c = 3 * g;
      const std::uint32_t words[4] = {
          B(c) | Y(x) << 10 | R(c) << 20,
          Y(x + 1) | B(c + 1) << 10 | Y(x + 2) << 20,
          R(c + 1) | Y(x + 3) << 10 | B(c + 2) << 20,
          Y(x + 4) | R(c + 2) << 10 | Y(x + 5) << 20};
      std::uint8_t* at = &out[lead + r * stride + 16 * g];
      for (int w = 0; w < 4; ++w) {
        for (int b = 0; b < 4; ++b) {
          at[4 * w + b] = static_cast<std::uint8_t>(words[w] >> (8 * b));
        }
      }
    }
  }
  return out;
}

// 8-bit codes as 10-bit ones, times 4: the same levels.
inline std::vector<std::uint16_t> widen(const std::vector<std::uint8_t>& c) {
  std::vector<std::uint16_t> out(c.size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    out[i] = static_cast<std::uint16_t>(4 * c[i]);
  }
  return out;
}

}  // namespace v210
