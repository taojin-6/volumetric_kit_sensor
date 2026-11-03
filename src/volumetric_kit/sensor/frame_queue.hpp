// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The frames a driver's thread delivers and its polling thread takes, as
// `IRgbdSensor` describes them -- the newest wins at depth 1, a deeper queue
// is drained oldest first, a frame pushed out of a full queue or passed over
// by a poll is counted dropped, and a failure is reported to every later
// take. The one copy of that rule: the RealSense and Orbbec drivers hold one
// for their SDK's threads, StereoDepthSensor for its own. Generic over what
// it holds, so a host test pins it with no SDK. Header-only, and internal to
// the targets whose sources include it.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"

namespace volumetric_kit::sensor::detail {

template <typename T>
class FrameQueue {
 public:
  // How many frames it holds (at least 1) and whom it wakes. Set before the
  // producer delivers.
  void set_depth(std::size_t depth) {
    std::lock_guard<std::mutex> lock(mutex_);
    depth_ = depth > 0 ? depth : 1;
  }
  std::size_t depth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return depth_;
  }
  void set_notify(std::function<void()> notify) {
    std::lock_guard<std::mutex> lock(mutex_);
    notify_ = std::move(notify);
  }

  // From the producer's thread: a frame in, the oldest pushed out of a full
  // queue (counted dropped), then the consumer woken, after the lock is
  // released, so its wake-up may take from the queue.
  void push(T frame) {
    std::function<void()> notify;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.push_back(std::move(frame));
      while (pending_.size() > depth_) {
        pending_.pop_front();
        dropped_.fetch_add(1, std::memory_order_release);
      }
      notify = notify_;
    }
    if (notify) notify();
  }

  // The consumer woken with no frame: a failure held elsewhere (a colour
  // decoder's), which its next take asks for.
  void wake() {
    std::function<void()> notify;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      notify = notify_;
    }
    if (notify) notify();
  }

  // A failure (not OK) every later take reports, the first one kept, then
  // the consumer woken.
  void fail(core::Status why) {
    std::function<void()> notify;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (failure_.ok()) failure_ = std::move(why);
      failed_.store(!failure_.ok(), std::memory_order_release);
      notify = notify_;
    }
    if (notify) notify();
  }

  // The newest frame, the older ones counted dropped; empty when none is
  // held; the failure once there is one.
  core::Result<std::optional<T>> take_newest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.ok()) return failure_;
    if (pending_.empty()) return std::optional<T>{};
    std::optional<T> newest(std::move(pending_.back()));
    dropped_.fetch_add(pending_.size() - 1, std::memory_order_release);
    pending_.clear();
    return newest;
  }

  // Every frame held, oldest first, appended to `out`; or the failure, with
  // `out` untouched.
  core::Status take_all(std::vector<T>* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.ok()) return failure_;
    for (T& frame : pending_) out->push_back(std::move(frame));
    pending_.clear();
    return {};
  }

  // A frame taken but never handed out, counted dropped.
  void discard() noexcept { dropped_.fetch_add(1, std::memory_order_release); }

  // At a stop or a start: the frames held let go uncounted.
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
  }
  // At a start: the dropped count restarted, as every counter is.
  void reset_dropped() noexcept {
    dropped_.store(0, std::memory_order_relaxed);
  }
  // At a start, for a producer whose failure ends one run rather than the
  // device (StereoDepthSensor's thread): the failure forgotten. A driver
  // whose failure is a disconnect never calls it.
  void reset_failure() {
    std::lock_guard<std::mutex> lock(mutex_);
    failure_ = {};
    failed_.store(false, std::memory_order_release);
  }

  // Each drop is counted with release and read with acquire, so a reader
  // that reads the frame's arrival (received) after its drops sees it, as
  // every driver's stats does.
  std::uint64_t dropped() const noexcept {
    return dropped_.load(std::memory_order_acquire);
  }
  bool failed() const noexcept {
    return failed_.load(std::memory_order_acquire);
  }
  // The failure; OK while there is none.
  core::Status failure() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_;
  }

 private:
  mutable std::mutex mutex_;
  std::deque<T> pending_;  // guarded by mutex_, oldest first
  std::size_t depth_ = 1;  // guarded by mutex_
  core::Status failure_;   // guarded by mutex_
  // The consumer's wake-up, copied under the lock and called after it.
  std::function<void()> notify_;  // guarded by mutex_
  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<bool> failed_{false};
};

}  // namespace volumetric_kit::sensor::detail
