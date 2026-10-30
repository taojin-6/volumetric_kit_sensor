// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the chunks an audio device's thread captures and
// the caller drains, one writer and one reader. Every entry is allocated at
// construction, and a push neither locks nor allocates, so the writer may be a
// realtime audio callback.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace volumetric_kit::sensor::audio {

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "the audio callback must not lock");

class ChunkRing {
 public:
  // `entries` chunks (at least 1) of up to `entry_samples` samples each.
  ChunkRing(std::size_t entries, std::size_t entry_samples)
      : entries_(entries > 0 ? entries : 1),
        entry_samples_(entry_samples),
        samples_(entries_.size() * entry_samples) {}

  ChunkRing(const ChunkRing&) = delete;
  ChunkRing& operator=(const ChunkRing&) = delete;

  // Writer: a chunk of `count` samples stamped `pts_ns`, zeros where
  // `samples` is null. False, with nothing queued, when every entry is held or
  // `count` exceeds an entry.
  bool push(std::uint64_t pts_ns, const std::int16_t* samples,
            std::size_t count) noexcept {
    const std::uint64_t written = written_.load(std::memory_order_relaxed);
    const std::uint64_t read = read_.load(std::memory_order_acquire);
    if (written - read >= entries_.size() || count > entry_samples_) {
      return false;
    }
    const std::size_t at = static_cast<std::size_t>(written % entries_.size());
    std::int16_t* to = samples_.data() + at * entry_samples_;
    if (samples != nullptr) {
      std::memcpy(to, samples, count * sizeof(std::int16_t));
    } else {
      std::memset(to, 0, count * sizeof(std::int16_t));
    }
    // Field by field: ThreadSanitizer misses a struct copy at -O0 (CI's Debug).
    entries_[at].pts_ns = pts_ns;
    entries_[at].count = count;
    written_.store(written + 1, std::memory_order_release);
    return true;
  }

  // Reader: each held chunk, oldest first, through
  // `take(pts_ns, const std::int16_t* samples, std::size_t count)`; the
  // entries are the writer's again once it returns. Returns how many.
  template <typename Take>
  std::size_t pop_all(Take&& take) {
    const std::uint64_t read = read_.load(std::memory_order_relaxed);
    const std::uint64_t written = written_.load(std::memory_order_acquire);
    for (std::uint64_t i = read; i < written; ++i) {
      const std::size_t at = static_cast<std::size_t>(i % entries_.size());
      take(entries_[at].pts_ns, samples_.data() + at * entry_samples_,
           entries_[at].count);
    }
    read_.store(written, std::memory_order_release);
    return static_cast<std::size_t>(written - read);
  }

  // Reader: every held chunk let go.
  void clear() noexcept {
    read_.store(written_.load(std::memory_order_acquire),
                std::memory_order_release);
  }

  std::size_t entries() const noexcept { return entries_.size(); }
  std::size_t entry_samples() const noexcept { return entry_samples_; }

 private:
  struct Entry {
    std::uint64_t pts_ns = 0;
    std::size_t count = 0;
  };

  std::vector<Entry> entries_;
  std::size_t entry_samples_;
  std::vector<std::int16_t> samples_;  // entry i at i * entry_samples_
  // Chunks ever pushed (the writer's) and ever taken or let go (the
  // reader's); held: the difference, at most entries().
  std::atomic<std::uint64_t> written_{0};
  std::atomic<std::uint64_t> read_{0};
};

}  // namespace volumetric_kit::sensor::audio
