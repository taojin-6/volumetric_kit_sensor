// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the frames a driver's SDK thread delivers and its
// polling thread takes, as `IRgbdSensor` describes them -- the newest wins at
// depth 1, a deeper queue is drained oldest first, a frame pushed out of a
// full queue or passed over by a poll is counted dropped, and a failure is
// reported to every later take. Generic over what it holds (a librealsense
// frameset), so a host test pins it with no SDK.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"

namespace volumetric_kit::sensor::realsense {

template <typename T>
class FrameQueue {
 public:
  // How many frames it holds (at least 1) and whom it wakes. Set before the
  // SDK delivers.
  void set_depth(std::size_t depth) {
    std::lock_guard<std::mutex> lock(mutex_);
    depth_ = depth > 0 ? depth : 1;
  }
  void set_notify(std::function<void()> notify) {
    std::lock_guard<std::mutex> lock(mutex_);
    notify_ = std::move(notify);
  }

  // From the SDK's thread: a frame in, the oldest pushed out of a full queue
  // (counted dropped), then the consumer woken, after the lock is released.
  void push(T frame) {
    std::function<void()> notify;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_.push_back(std::move(frame));
      while (pending_.size() > depth_) {
        pending_.pop_front();
        dropped_.fetch_add(1, std::memory_order_relaxed);
      }
      notify = notify_;
    }
    if (notify) notify();
  }

  // A failure every later take reports (the first one kept), then the
  // consumer woken.
  void fail(std::string why) {
    std::function<void()> notify;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (fault_.empty()) fault_ = std::move(why);
      failed_.store(true, std::memory_order_release);
      notify = notify_;
    }
    if (notify) notify();
  }

  // The newest frame, the older ones counted dropped; empty when none is
  // held; the failure once there is one.
  core::Result<std::optional<T>> take_newest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fault_.empty()) return core::Status::io_error(fault_);
    if (pending_.empty()) return std::optional<T>{};
    std::optional<T> newest(std::move(pending_.back()));
    dropped_.fetch_add(pending_.size() - 1, std::memory_order_relaxed);
    pending_.clear();
    return newest;
  }

  // Every frame held, oldest first, appended to `out`; or the failure.
  core::Status take_all(std::vector<T>* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fault_.empty()) return core::Status::io_error(fault_);
    for (T& frame : pending_) out->push_back(std::move(frame));
    pending_.clear();
    return {};
  }

  // A frame taken but never handed out, counted dropped.
  void discard() noexcept { dropped_.fetch_add(1, std::memory_order_relaxed); }

  // At a stop: the frames held let go uncounted, and the count restarted.
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
    dropped_.store(0, std::memory_order_relaxed);
  }

  std::uint64_t dropped() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }
  bool failed() const noexcept {
    return failed_.load(std::memory_order_acquire);
  }

 private:
  std::mutex mutex_;
  std::deque<T> pending_;  // guarded by mutex_, oldest first
  std::size_t depth_ = 1;  // guarded by mutex_
  std::string fault_;      // guarded by mutex_
  // The consumer's wake-up, copied under the lock and called after it.
  std::function<void()> notify_;  // guarded by mutex_
  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<bool> failed_{false};
};

}  // namespace volumetric_kit::sensor::realsense
