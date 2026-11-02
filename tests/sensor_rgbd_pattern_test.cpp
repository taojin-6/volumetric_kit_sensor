// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The index-coded test depth (testing/rgbd_pattern.hpp), host-only and on
// the standard library alone, so camera-only builds, which have no Vulkan,
// run it too: the index read back modulo 16384 across its span, at ARKit's
// size, a 16:9 one and a picture's; a size that cannot hold the band, and
// no samples, read as none; one sample changed in a square's central half,
// torn depth (two frames' halves) and swapped references read as none,
// while a change past the central half still reads; the depth's values, the
// hole, and the confidence plane as documented; and a size too small for
// the band made without reading or writing past it.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"

namespace testing = volumetric_kit::sensor::testing;

namespace {

struct Size {
  std::uint32_t w, h;
};

std::optional<std::uint32_t> read(const testing::PatternDepth& d) {
  return testing::depth_index(d.depth.data(), d.width, d.height);
}

int test_index() {
  for (const Size s : {Size{256, 192}, Size{256, 144}, Size{1920, 1440}}) {
    for (const std::uint32_t i : {0u, 1u, 16383u, 16384u, 70000u}) {
      const testing::PatternDepth d = testing::pattern_depth(s.w, s.h, i);
      CHECK(d.width == s.w && d.height == s.h);
      CHECK(d.depth.size() == std::size_t{s.w} * s.h);
      CHECK(d.confidence.size() == d.depth.size());
      CHECK(read(d) == i % 16384);
    }
  }
  // A size that cannot hold the band, and no samples, read as none.
  for (const Size s : {Size{255, 192}, Size{256, 15}}) {
    const testing::PatternDepth d = testing::pattern_depth(s.w, s.h, 5);
    CHECK(!read(d));
  }
  CHECK(!testing::depth_index(nullptr, 256, 192));
  std::puts("the index modulo 16384 at three sizes; none below 256 x 16");
  return 0;
}

int test_breaks() {
  const testing::PatternDepth five = testing::pattern_depth(256, 192, 5);
  const testing::PatternDepth ten = testing::pattern_depth(256, 192, 10);
  CHECK(read(five) == 5u && read(ten) == 10u);

  // One sample changed in square 7's central half, away from the sample a
  // reader would start from and from the square's centre: none.
  testing::PatternDepth one = five;
  const std::size_t at = std::size_t{5} * 256 + 7 * 16 + 9;
  one.depth[at] = 1001;
  CHECK(!read(one));
  // The same sample set to the other value a square may hold: still none,
  // the square no longer one value.
  one.depth[at] = testing::kPatternNearMm;
  CHECK(!read(one));
  // Past the central half, a change leaves the index as it was.
  testing::PatternDepth edge = five;
  edge.depth[std::size_t{1} * 256 + 7 * 16 + 1] = 1234;
  CHECK(read(edge) == 5u);

  // Torn depth: the band's top half from frame 5, its bottom half from
  // frame 10. Squares where they differ hold both values: none.
  testing::PatternDepth torn = five;
  for (std::size_t k = 8 * 256; k < 16 * 256; ++k) torn.depth[k] = ten.depth[k];
  CHECK(!read(torn));

  // Swapped references: none, though each square is one value.
  testing::PatternDepth swapped = five;
  for (std::uint32_t y = 0; y < 16; ++y) {
    for (std::uint32_t x = 0; x < 16; ++x) {
      std::swap(swapped.depth[std::size_t{y} * 256 + x],
                swapped.depth[std::size_t{y} * 256 + 16 + x]);
    }
  }
  CHECK(!read(swapped));
  std::puts("a changed sample, torn depth and swapped references: none");
  return 0;
}

// The depth and the confidence as documented, at ARKit's 256 x 192: the band
// 1000 where a square is set and 2000 where it is clear, confidence 2 on it;
// elsewhere the plane at 2000, 0 across the left half of the bottom eighth,
// confidence x % 3.
int test_planes() {
  const std::uint32_t index = 0x2A5B;  // bits 0, 1, 3, 4, 6, 9, 11, 13
  const testing::PatternDepth d = testing::pattern_depth(256, 192, index);
  for (std::uint32_t y = 0; y < 192; ++y) {
    for (std::uint32_t x = 0; x < 256; ++x) {
      const std::size_t at = std::size_t{y} * 256 + x;
      if (y < 16) {
        const std::uint32_t square = x / 16;
        const bool set =
            square == 0 || (square >= 2 && ((index >> (square - 2)) & 1));
        CHECK(d.depth[at] == (set ? 1000 : 2000));
        CHECK(d.confidence[at] == 2);
        continue;
      }
      const bool hole = y >= 168 && x < 128;
      CHECK(d.depth[at] == (hole ? 0 : 2000));
      CHECK(d.confidence[at] == x % 3);
    }
  }
  // A floor at each level keeps every square and drops part of the plane.
  for (std::uint8_t floor = 1; floor <= 2; ++floor) {
    std::size_t dropped = 0;
    for (std::size_t k = 0; k < d.confidence.size(); ++k) {
      if (d.confidence[k] < floor) {
        CHECK(k >= std::size_t{16} * 256);
        ++dropped;
      }
    }
    CHECK(dropped > 0);
  }
  // A size too small for the band holds what fits; the index reads as none.
  const testing::PatternDepth small = testing::pattern_depth(37, 21, 3);
  CHECK(small.depth.size() == 37u * 21u &&
        small.confidence.size() == 37u * 21u);
  CHECK(small.depth[0] == 1000 && small.depth[16] == 2000);
  CHECK(small.depth[std::size_t{20} * 37] == 0);  // the hole, below the band
  CHECK(!read(small));
  const testing::PatternDepth empty = testing::pattern_depth(0, 0, 3);
  CHECK(empty.depth.empty() && empty.confidence.empty());
  std::puts("the depth, its hole and the confidence plane as documented");
  return 0;
}

}  // namespace

int main() {
  if (test_index() != 0 || test_breaks() != 0 || test_planes() != 0) return 1;
  std::puts("sensor_rgbd_pattern: OK");
  return 0;
}
