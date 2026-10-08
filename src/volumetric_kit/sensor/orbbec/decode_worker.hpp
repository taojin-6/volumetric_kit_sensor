// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the thread a camera's colour decoder decodes on,
// for JpegColorDecoder and HevcColorDecoder alike. The SDK's thread pushes
// each pair and never waits for a decode; the worker hands the pairs to the
// decoder's function one at a time, in order, on a thread of its own. It owns
// the queue and what a full one drops, the thread's start and stop, the
// counts, and the failure that ends decoding; what decoding a pair means is
// the decoder's. It names no SDK type, so its test pushes plain values.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/result.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

template <typename Item>
class DecodeWorker {
 public:
  // What a push does when `depth` items already wait.
  enum class Overflow {
    // The oldest go, counted dropped: each item decodes on its own (MJPEG).
    DropOldest,
    // They all go, counted lost: each item is predicted from the ones before
    // it (H.265), so decoding picks up afresh after them.
    DropAll,
  };

  // Decodes one item, on the worker's thread. `after_gap`: an item before it
  // never reached this function -- the queue dropped it, or had no memory to
  // hold it -- or reached it and threw.
  using Decode = std::function<void(Item item, bool after_gap)>;

  // At most `depth` items wait (at least one: the newest always does).
  DecodeWorker(std::size_t depth, Overflow overflow)
      : depth_(depth > 0 ? depth : 1), overflow_(overflow) {}
  DecodeWorker(const DecodeWorker&) = delete;
  DecodeWorker& operator=(const DecodeWorker&) = delete;
  ~DecodeWorker() { stop(); }

  // Start the thread, which hands each item pushed to `decode`. IoError when
  // the thread cannot start. Call once.
  core::Status start(Decode decode) {
    VKC_CHECK(!thread_.joinable(), "DecodeWorker::start called twice");
    decode_ = std::move(decode);
    try {
      thread_ = std::thread([this] { run(); });
    } catch (const std::system_error& e) {
      return core::Status::io_error(
          std::string("starting the colour decoding thread: ") + e.what());
    }
    return {};
  }

  // Queue `item` for decoding; it never waits for a decode. A full queue
  // drops as the Overflow says, and an item it has no memory to hold is
  // lost. Ignored once stopped or failed.
  void push(Item item) noexcept {
    std::uint64_t lost = 0;
    std::uint64_t dropped = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_ || !failure_.ok()) return;
      if (queue_.size() >= depth_) {
        if (overflow_ == Overflow::DropOldest) {
          while (queue_.size() >= depth_) {
            queue_.pop_front();
            ++dropped;
          }
        } else {
          lost = queue_.size();
          queue_.clear();
        }
        gap_ = true;
      }
      try {
        queue_.push_back(std::move(item));
      } catch (...) {  // out of memory: this item goes, as an overflow's do
        ++lost;
        gap_ = true;
      }
    }
    if (dropped != 0) dropped_.fetch_add(dropped, std::memory_order_relaxed);
    if (lost != 0) lose(lost);
    wake_.notify_one();
  }

  // Stop the thread once its decode in hand returns, dropping what waits,
  // uncounted. True for the call that stopped it; false, at once, for any
  // other.
  bool stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) return false;
      stopping_ = true;
      queue_.clear();
    }
    wake_.notify_one();
    if (thread_.joinable()) thread_.join();
    return true;
  }

  // Count `items` lost: handed to the decode and never decoded.
  void lose(std::uint64_t items = 1) noexcept {
    lost_.fetch_add(items, std::memory_order_relaxed);
  }

  // Stop decoding for good, for `why` (not OK; the first one stands): what
  // waits is lost with it, and pushes are ignored from then on.
  void fail(core::Status why) {
    VKC_CHECK(!why.ok(), "DecodeWorker::fail needs a failure");
    std::lock_guard<std::mutex> lock(mutex_);
    // Counted before the failure shows, so whoever sees it sees them.
    lose(queue_.size());
    queue_.clear();
    if (failure_.ok()) failure_ = std::move(why);
  }

  // Why decoding stopped for good (fail); OK while it runs.
  core::Status failure() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_;
  }
  // Items lost: dropped by DropAll or for want of memory, one whose decode
  // threw, those waiting at fail(), and whatever the decode counts (lose).
  std::uint64_t lost() const noexcept {
    return lost_.load(std::memory_order_relaxed);
  }
  // Items DropOldest replaced with newer ones.
  std::uint64_t dropped() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  void run() {
    bool threw = false;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_) return;
      Item item = std::move(queue_.front());
      queue_.pop_front();
      const bool after_gap = std::exchange(gap_, false) || threw;
      lock.unlock();
      threw = false;
      try {
        decode_(std::move(item), after_gap);
      } catch (...) {  // an SDK call that threw; never out of this thread
        lose();
        threw = true;
      }
    }
  }

  const std::size_t depth_;
  const Overflow overflow_;
  Decode decode_;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Item> queue_;  // guarded by mutex_
  bool stopping_ = false;   // guarded by mutex_
  bool gap_ = false;        // guarded by mutex_: dropped since the last pop
  core::Status failure_;    // guarded by mutex_
  std::atomic<std::uint64_t> lost_{0};
  std::atomic<std::uint64_t> dropped_{0};
  std::thread thread_;
};

}  // namespace volumetric_kit::recon::sensor::orbbec
