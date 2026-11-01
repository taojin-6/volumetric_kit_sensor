// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "pairing.hpp"

#include <cmath>
#include <utility>

namespace volumetric_kit::sensor::decklink {

namespace {

std::int64_t whole_periods(std::int64_t ns, std::int64_t period) {
  return static_cast<std::int64_t>(
      std::llround(static_cast<double>(ns) / static_cast<double>(period)));
}

}  // namespace

std::int64_t host_time_ns(const Arrival& frame) {
  return frame.card_ns + (frame.host_now_ns - frame.card_now_ns);
}

Pairing::Pairing(std::uint64_t period_ns, std::uint64_t max_skew_ns)
    : period_ns_(static_cast<std::int64_t>(period_ns)),
      max_skew_ns_(static_cast<std::int64_t>(max_skew_ns)) {}

std::optional<Pair> Pairing::arrive(Arrival frame) {
  // Frames let go are handed back to the SDK after the lock is released.
  std::optional<Held> gone[2];
  std::optional<Pair> pair;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const int input = frame.input == 0 ? 0 : 1;
    const int other = 1 - input;
    if (input == 0) ++received_;
    if (frame.no_signal) {
      // No picture: it fails, and so does a frame of this input held from
      // before the signal went.
      if (input == 0) ++failed_;
      gone[0] = std::move(held_[input]);
      held_[input].reset();
      if (gone[0] && gone[0]->frame.input == 0) ++failed_;
    } else {
      Held arrived{std::move(frame), 0, 0};
      arrived.host_ns = host_time_ns(arrived.frame);
      newest_card_ns_[input] = arrived.frame.card_ns;
      if (input == 0) {
        if (last_left_card_ns_) {
          const std::int64_t periods = whole_periods(
              arrived.frame.card_ns - *last_left_card_ns_, period_ns_);
          sequence_ += static_cast<std::uint64_t>(periods > 1 ? periods : 1);
        }
        last_left_card_ns_ = arrived.frame.card_ns;
        arrived.sequence = sequence_;
      }
      // The newest of this input replaces the one held: it found no
      // partner before it.
      gone[0] = std::move(held_[input]);
      held_[input].reset();
      if (gone[0] && gone[0]->frame.input == 0) ++failed_;

      std::optional<Held>& partner = held_[other];
      const Held* left = input == 0 ? &arrived : partner ? &*partner : nullptr;
      const Held* right = input == 1 ? &arrived : partner ? &*partner : nullptr;
      if (left != nullptr && right != nullptr) {
        const std::int64_t skew = right->frame.card_ns - left->frame.card_ns;
        if (skew <= max_skew_ns_ && -skew <= max_skew_ns_) {
          Pair made;
          made.timestamp_ns = static_cast<std::uint64_t>(left->host_ns);
          made.skew_ns = skew;
          made.sequence = left->sequence;
          made.arrival_ns =
              static_cast<std::uint64_t>(arrived.frame.host_now_ns);
          Held other_frame = std::move(*partner);
          partner.reset();
          made.left = std::move(input == 0 ? arrived.frame : other_frame.frame);
          made.right =
              std::move(input == 1 ? arrived.frame : other_frame.frame);
          pair = std::move(made);
        }
      }
      if (!pair) held_[input] = std::move(arrived);
    }
  }
  return pair;
}

void Pairing::reset() {
  std::optional<Held> gone[2];
  const std::lock_guard<std::mutex> lock(mutex_);
  for (int i = 0; i < 2; ++i) {
    gone[i] = std::move(held_[i]);
    held_[i].reset();
    newest_card_ns_[i].reset();
  }
  last_left_card_ns_.reset();
  sequence_ = 0;
  received_ = 0;
  failed_ = 0;
}

void Pairing::release() {
  std::optional<Held> gone[2];
  const std::lock_guard<std::mutex> lock(mutex_);
  for (int i = 0; i < 2; ++i) {
    gone[i] = std::move(held_[i]);
    held_[i].reset();
  }
}

std::uint64_t Pairing::received() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return received_;
}

std::uint64_t Pairing::failed() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return failed_;
}

bool Pairing::seen(int input) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return newest_card_ns_[input == 0 ? 0 : 1].has_value();
}

std::optional<std::int64_t> Pairing::offset_ns() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!newest_card_ns_[0] || !newest_card_ns_[1]) return std::nullopt;
  const std::int64_t apart = *newest_card_ns_[1] - *newest_card_ns_[0];
  return apart - whole_periods(apart, period_ns_) * period_ns_;
}

}  // namespace volumetric_kit::sensor::decklink
