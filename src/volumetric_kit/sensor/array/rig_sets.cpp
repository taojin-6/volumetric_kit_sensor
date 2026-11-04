// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/array/rig_sets.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::sensor {
namespace {

// Keep @p set's first @p frames frames, in the array's order, and drop the
// rest. Returns the frames kept.
std::size_t cut_set(FrameSet& set, std::size_t frames) {
  std::size_t kept = 0;
  for (std::optional<RgbdFrame>& frame : set.frames) {
    if (!frame) continue;
    if (kept == frames) {
      frame.reset();
    } else {
      ++kept;
    }
  }
  return kept;
}

// No set in `limit`: the IoError naming the rig or its one sensor, the limit
// (in whole seconds where it is whole), and why a lone sync secondary
// delivers none.
core::Status silence(SensorArray& array, std::chrono::milliseconds limit) {
  const bool one = array.size() == 1;
  const bool secondary =
      one && array.sensor(0).info().role == SyncRole::Secondary;
  const std::string within = limit.count() % 1000 == 0
                                 ? std::to_string(limit.count() / 1000) + " s"
                                 : std::to_string(limit.count()) + " ms";
  return core::Status::io_error(
      "no frame from " +
      (one ? "camera " + array.sensor(0).info().id : std::string("the rig")) +
      " in " + within +
      (secondary ? std::string(" -- it is a sync secondary, which captures "
                               "only on another camera's signal; start the "
                               "camera that drives it")
                 : std::string()));
}

}  // namespace

RigSets::RigSets(SensorArray& array, std::size_t frames) noexcept
    : RigSets(array, frames, kSilenceLimit) {}

RigSets::RigSets(SensorArray& array, std::size_t frames,
                 std::chrono::milliseconds silence_limit) noexcept
    : array_(array),
      size_(array.size()),
      limit_(frames),
      silence_limit_(silence_limit),
      last_set_(std::chrono::steady_clock::now()) {
  VKC_CHECK(silence_limit.count() > 0,
            "RigSets: the silence limit must be positive");
}

RigSets::Skew RigSets::skew(std::size_t camera) const {
  VKC_CHECK(camera < size_, "RigSets::skew: index past the array");
  return camera < skews_.size() ? skews_[camera] : Skew{};
}

core::Result<FrameSet> RigSets::next(core::StageMetrics* metrics) try {
  // Before the wait, so a failed allocation takes no set.
  if (skews_.empty()) skews_.resize(size_);
  const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
      silence_limit_ - (std::chrono::steady_clock::now() - last_set_));
  VKC_ASSIGN(
      std::optional<FrameSet> set,
      array_.wait_set(std::max(left, std::chrono::milliseconds(0)), metrics));
  if (!set) return silence(array_, silence_limit_);
  last_set_ = std::chrono::steady_clock::now();
  for (std::size_t c = 0; c < set->frames.size(); ++c) {
    const std::optional<std::int64_t> skew_ns = set->skew_ns(c);
    if (c == array_.primary() || !skew_ns) continue;
    const double ms = static_cast<double>(*skew_ns) / 1e6;
    skews_[c].sum_ms += ms;
    skews_[c].worst_ms = std::max(skews_[c].worst_ms, std::fabs(ms));
    ++skews_[c].frames;
  }
  frames_ += cut_set(*set, limit_ - frames_);
  ++sets_;
  return std::move(*set);
} catch (const std::bad_alloc&) {
  return core::exception_status("RigSets::next",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor
