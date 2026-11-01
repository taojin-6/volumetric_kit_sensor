// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): two free-running inputs' frames made into pairs
// by the card's frame times, on the SDK's two callback threads; no pixel is
// touched and no SDK type named.
//
// - Time. A frame's host time is its card time plus the two clocks' offset
//   sampled together in its own callback (Arrival), so the card clock's
//   drift against the host's (about 19 ppm on the lab desktop) is tracked
//   frame by frame rather than fixed at the first.
// - Pairs. The newest unpaired frame of each input is kept; a frame pairs
//   with the other input's when their card times lie within max_skew_ns,
//   and takes the left frame's time, sequence and the skew between them
//   (right less left).
// - Sequence. Each left frame with a picture adds the whole periods since
//   the last one, at least one, so a gap is a lost frame; increments, not
//   the time since the first, which a camera 27 ppm off nominal would slip
//   by half a frame in about 10 minutes.
// - Counts, in left frames: received, every left frame; failed, a left
//   frame not delivered in a pair, for any cause: no partner before the
//   next left frame, no signal, or the partner past the bound.

#include <cstdint>
#include <mutex>
#include <optional>

#include "capture.hpp"

namespace volumetric_kit::sensor::decklink {

// A left and a right frame within the bound.
struct Pair {
  Arrival left;
  Arrival right;
  std::uint64_t timestamp_ns = 0;  // the left frame's host time
  std::int64_t skew_ns = 0;        // the right's card time less the left's
  std::uint64_t sequence = 0;      // the left frame's
  std::uint64_t arrival_ns = 0;    // the host's clock as the pair formed
};

// The frame's time on the host's clock: its card time plus the offset its
// own callback sampled.
std::int64_t host_time_ns(const Arrival& frame);

class Pairing {
 public:
  // `period_ns`: the mode's nominal frame period; `max_skew_ns`: the bound.
  Pairing(std::uint64_t period_ns, std::uint64_t max_skew_ns);

  // A frame in, from either input's thread; the pair it completes, if any.
  // A frame it replaces or refuses is let go here.
  std::optional<Pair> arrive(Arrival frame);

  // At a start: the frames held let go, the counts and the sequence
  // started again.
  void reset();
  // At a stop: the frames held let go; the counts kept.
  void release();

  std::uint64_t received() const;
  std::uint64_t failed() const;
  // Whether `input` has sent a frame with a picture since the reset.
  bool seen(int input) const;
  // The two inputs' newest pictures apart, right less left, wrapped into
  // half a frame either way: the cameras' phase. Empty until both sent one.
  std::optional<std::int64_t> offset_ns() const;

 private:
  struct Held {
    Arrival frame;
    std::int64_t host_ns = 0;
    std::uint64_t sequence = 0;
  };

  const std::int64_t period_ns_;
  const std::int64_t max_skew_ns_;
  mutable std::mutex mutex_;
  std::optional<Held> held_[2];  // guarded by mutex_
  std::optional<std::int64_t> newest_card_ns_[2];
  std::optional<std::int64_t> last_left_card_ns_;
  std::uint64_t sequence_ = 0;
  std::uint64_t received_ = 0;
  std::uint64_t failed_ = 0;
};

}  // namespace volumetric_kit::sensor::decklink
