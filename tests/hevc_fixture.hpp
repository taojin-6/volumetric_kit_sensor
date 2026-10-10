// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

namespace vr_test {

// Committed clips carry access unit delimiters (NAL type 35). Keep both
// decoder tests on the same split, including the three/four-byte start code.
inline std::vector<std::vector<std::uint8_t>> hevc_access_units(
    const char* path) {
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
  std::vector<std::vector<std::uint8_t>> units;
  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t end =
        k + 1 < starts.size() ? starts[k + 1] : bytes.size();
    units.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(starts[k]),
                       bytes.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return units;
}

}  // namespace vr_test
