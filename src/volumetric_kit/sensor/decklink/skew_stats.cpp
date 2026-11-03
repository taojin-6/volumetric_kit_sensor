// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "skew_stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace volumetric_kit::sensor::decklink {

namespace {

// y = a + b x by least squares, about the first x for precision.
std::pair<double, double> fit_line(const std::vector<double>& x,
                                   const std::vector<double>& y) {
  const std::size_t n = x.size();
  if (n < 2) return {n == 1 ? y[0] : 0.0, 0.0};
  double mx = 0.0;
  double my = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    mx += x[i];
    my += y[i];
  }
  mx /= static_cast<double>(n);
  my /= static_cast<double>(n);
  double sxx = 0.0;
  double sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sxx += (x[i] - mx) * (x[i] - mx);
    sxy += (x[i] - mx) * (y[i] - my);
  }
  const double b = sxx > 0.0 ? sxy / sxx : 0.0;
  return {my - b * mx, b};
}

// One input's frames, losses and rate.
void time_input(const std::vector<std::int64_t>& t, double period, int side,
                SkewReport& r) {
  r.frames[side] = t.size();
  if (t.size() < 2) return;
  std::vector<double> index(t.size());
  std::vector<double> at(t.size());
  double k = 0.0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (i > 0) {
      const double steps = std::max(
          1.0, std::round(static_cast<double>(t[i] - t[i - 1]) / period));
      r.lost[side] += static_cast<std::uint64_t>(steps) - 1;
      k += steps;
    }
    index[i] = k;
    at[i] = static_cast<double>(t[i] - t[0]);
  }
  const double fitted = fit_line(index, at).second;
  r.slow_ppm[side] = (fitted / period - 1.0) * 1e6;
}

}  // namespace

SkewReport skew_stats(const std::vector<std::int64_t>& left,
                      const std::vector<std::int64_t>& right,
                      std::uint64_t period_ns, std::uint64_t max_skew_ns) {
  SkewReport r;
  const double period = static_cast<double>(period_ns);
  time_input(left, period, 0, r);
  time_input(right, period, 1, r);
  if (left.empty() || right.empty()) return r;

  // Each left frame's nearest right frame, both rising.
  std::vector<double> when;
  std::vector<double> skew;
  std::size_t j = 0;
  for (const std::int64_t t : left) {
    while (j + 1 < right.size() &&
           std::llabs(right[j + 1] - t) <= std::llabs(right[j] - t)) {
      ++j;
    }
    const std::int64_t s = right[j] - t;
    if (static_cast<std::uint64_t>(std::llabs(s)) > max_skew_ns) {
      ++r.refused;
      continue;
    }
    ++r.pairs;
    when.push_back(static_cast<double>(t - left.front()) / 60e9);  // minutes
    skew.push_back(static_cast<double>(s) / 1e6);                  // ms
  }
  if (skew.empty()) return r;

  double sum = 0.0;
  for (const double s : skew) sum += s;
  r.skew_mean_ms = sum / static_cast<double>(skew.size());
  double var = 0.0;
  for (const double s : skew) {
    var += (s - r.skew_mean_ms) * (s - r.skew_mean_ms);
  }
  r.skew_std_ms = std::sqrt(var / static_cast<double>(skew.size()));
  std::vector<double> size(skew.size());
  for (std::size_t i = 0; i < skew.size(); ++i) size[i] = std::fabs(skew[i]);
  std::sort(size.begin(), size.end());
  const std::size_t at = static_cast<std::size_t>(std::ceil(
                             0.999 * static_cast<double>(size.size()))) -
                         1;
  r.skew_p999_ms = size[std::min(at, size.size() - 1)];
  const std::pair<double, double> line = fit_line(when, skew);
  r.skew_at_start_ms = line.first;
  r.drift_ms_per_min = line.second;
  return r;
}

}  // namespace volumetric_kit::sensor::decklink
