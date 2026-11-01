// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): what decklink_skew_report prints, from each
// input's frame times on the card's clock -- the frames with a picture, in
// arrival order. No SDK type, so a test feeds it synthetic logs.

#include <cstdint>
#include <vector>

namespace volumetric_kit::sensor::decklink {

struct SkewReport {
  // Each input's frames, and the frames lost between them: a step of n
  // whole periods is n - 1 lost.
  std::uint64_t frames[2] = {};
  std::uint64_t lost[2] = {};
  // Each input's rate against the card clock, parts per million slow (a
  // fitted period longer than nominal is positive).
  double slow_ppm[2] = {};
  // Each left frame with its nearest right frame: within the bound, a pair;
  // past it, refused.
  std::uint64_t pairs = 0;
  std::uint64_t refused = 0;
  // The pairs' skew, right less left, ms: mean, standard deviation, the
  // 99.9th percentile of its size, and a line fitted against the left
  // frame's time: its value at the first left frame and its slope, ms a
  // minute.
  double skew_mean_ms = 0.0;
  double skew_std_ms = 0.0;
  double skew_p999_ms = 0.0;
  double skew_at_start_ms = 0.0;
  double drift_ms_per_min = 0.0;
};

// The report of `left` and `right`'s card times (ns, rising), at the
// mode's nominal `period_ns`, pairs refused past `max_skew_ns`.
SkewReport skew_stats(const std::vector<std::int64_t>& left,
                      const std::vector<std::int64_t>& right,
                      std::uint64_t period_ns, std::uint64_t max_skew_ns);

}  // namespace volumetric_kit::sensor::decklink
