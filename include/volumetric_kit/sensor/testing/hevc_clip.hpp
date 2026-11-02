// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/hevc_clip.hpp
/// @brief Test support for H.265 clips, with no test framework, codec or
///        GPU: a clip's access units, and whether a run promises a hardware
///        decoder. Header-only, on the standard library alone.
///
/// The clips sensor commits (`tests/data/hevc`, made by
/// `tools/make_hevc_fixtures.sh`) open every access unit with a delimiter,
/// so a test sends them to a decoder one unit at a time.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
/// namespace test = volumetric_kit::core::test;  // the core's test support
///
/// const auto units = testing::access_units_of_file(clip);
/// auto decoder = HevcDecoder::create(options);
/// if (!decoder) {
///   // No device path: skip, unless this run promises one.
///   return testing::hevc_decoder_promised() ? 1 : test::kSkipExitCode;
/// }
/// for (const std::vector<std::uint8_t>& unit : units) {
///   VKC_TRY(decoder->send(unit.data(), unit.size(), pts));
/// }
/// @endcode

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

namespace volumetric_kit::sensor::testing {

/// @return The access units of the Annex B bytes @p data (@p size of them)
///         of a clip that carries access unit delimiters (NAL type 35):
///         each from its delimiter's start code, three bytes or four, to the
///         next delimiter's, the last to the end. Bytes before the first
///         delimiter are left out; empty for bytes with none.
inline std::vector<std::vector<std::uint8_t>> split_access_units(
    const std::uint8_t* data, std::size_t size) {
  std::vector<std::size_t> starts;
  for (std::size_t i = 0; i + 3 < size; ++i) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 &&
        ((data[i + 3] >> 1) & 0x3f) == 35) {
      starts.push_back(i > 0 && data[i - 1] == 0 ? i - 1 : i);
    }
  }
  std::vector<std::vector<std::uint8_t>> units;
  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t end = k + 1 < starts.size() ? starts[k + 1] : size;
    units.emplace_back(data + starts[k], data + end);
  }
  return units;
}

/// @return @ref split_access_units of the file at @p path; empty if it
///         cannot be read.
inline std::vector<std::vector<std::uint8_t>> access_units_of_file(
    const char* path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
  return split_access_units(bytes.data(), bytes.size());
}

/// @return Whether this run promises a hardware HEVC decoder:
///         `VS_TEST_HEVC_BACKEND` set and not empty, as CI sets it on every
///         leg that has one (`cuda`, `videotoolbox`). A test that opens no
///         device path then fails rather than skips.
inline bool hevc_decoder_promised() noexcept {
  const char* promise = std::getenv("VS_TEST_HEVC_BACKEND");
  return promise != nullptr && *promise != '\0';
}

}  // namespace volumetric_kit::sensor::testing
