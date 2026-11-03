// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The public key gate (sensor/video/key_gate.hpp), with no codec and no GPU,
// so it runs on every leg: a stream starts at its first key, which restarts
// it; a skipped, repeated or backward number waits for the next key, as
// lose() does; a key within an unbroken stream is taken; an unnumbered unit
// is never a gap and the one after it is not checked, while 0 is a number
// like any other.

#include "volumetric_kit/sensor/video/key_gate.hpp"

#include <cstdint>
#include <cstdio>
#include <optional>

#include "volumetric_kit/core/testing/test_check.hpp"

namespace sensor = volumetric_kit::sensor;
using A = sensor::KeyGate::Admission;

namespace {

constexpr std::optional<std::uint64_t> kUnnumbered = std::nullopt;

// The Orbbec colour gate's cases, which this gate took over: key frames start
// a stream, a skipped or repeated number stops it until the next, an
// unnumbered unit is never a gap, and lose() waits too. The key that ends a
// wait restarts the stream; one within an unbroken stream continues it.
int test_orbbec_cases() {
  sensor::KeyGate gate;
  CHECK(gate.admit(1, false) == A::Skip);  // before the first key
  CHECK(gate.admit(2, true) == A::Restart);
  CHECK(gate.admit(3, false) == A::Take);
  CHECK(gate.admit(4, true) == A::Take);
  CHECK(gate.admit(6, false) == A::Skip);  // 5 went missing
  CHECK(gate.admit(7, false) == A::Skip);
  CHECK(gate.admit(8, true) == A::Restart);
  CHECK(gate.admit(9, false) == A::Take);
  CHECK(gate.admit(9, false) == A::Skip);  // repeated
  CHECK(gate.admit(10, true) == A::Restart);
  CHECK(gate.admit(2, false) == A::Skip);  // backwards: a restarted camera
  CHECK(gate.admit(3, true) == A::Restart);
  gate.lose();
  CHECK(gate.admit(4, false) == A::Skip);
  CHECK(gate.admit(5, true) == A::Restart);
  sensor::KeyGate unnumbered;
  CHECK(unnumbered.admit(kUnnumbered, true) == A::Restart);
  CHECK(unnumbered.admit(kUnnumbered, false) == A::Take);
  CHECK(unnumbered.admit(kUnnumbered, false) == A::Take);
  return 0;
}

// A new gate waits, whatever the first number, and its first key restarts
// it; 0 is a number, so 2 after it is a gap.
int test_start() {
  sensor::KeyGate gate;
  CHECK(gate.waiting());
  CHECK(gate.admit(1000, false) == A::Skip);
  CHECK(gate.waiting());
  CHECK(gate.admit(1001, true) == A::Restart);
  CHECK(!gate.waiting());
  CHECK(gate.admit(1002, false) == A::Take);
  sensor::KeyGate zero;
  CHECK(zero.admit(0, true) == A::Restart);
  CHECK(zero.admit(1, false) == A::Take);
  CHECK(zero.admit(3, false) == A::Skip);
  sensor::KeyGate from_zero;
  CHECK(from_zero.admit(0, true) == A::Restart);
  CHECK(from_zero.admit(2, false) == A::Skip);  // 1 went missing
  return 0;
}

// A gap, seen while running or while waiting, waits for the next key; the
// units it skips set the number the next is checked against.
int test_gap() {
  sensor::KeyGate gate;
  CHECK(gate.admit(1, true) == A::Restart);
  CHECK(gate.admit(3, true) == A::Restart);  // a key right after the gap
  CHECK(gate.admit(5, false) == A::Skip);
  CHECK(gate.waiting());
  CHECK(gate.admit(6, false) == A::Skip);    // one up, but still waiting
  CHECK(gate.admit(8, true) == A::Restart);  // a gap while waiting, a key
  CHECK(gate.admit(9, false) == A::Take);
  return 0;
}

// An unnumbered unit resets the last number: the unit after it is not
// checked against the one before it, and the one after that is checked
// against it.
int test_unnumbered_resets_the_last_number() {
  sensor::KeyGate gate;
  CHECK(gate.admit(5, true) == A::Restart);
  CHECK(gate.admit(kUnnumbered, false) == A::Take);
  CHECK(gate.admit(9, false) == A::Take);  // not checked against 5
  CHECK(gate.admit(10, false) == A::Take);
  CHECK(gate.admit(kUnnumbered, false) == A::Take);
  CHECK(gate.admit(kUnnumbered, false) == A::Take);
  CHECK(gate.admit(20, false) == A::Take);
  CHECK(gate.admit(22, false) == A::Skip);  // checked against 20
  return 0;
}

// lose() waits for the next key, whatever the numbers, and called again,
// or while waiting, still costs one key.
int test_lose() {
  sensor::KeyGate gate;
  CHECK(gate.admit(1, true) == A::Restart);
  gate.lose();
  CHECK(gate.waiting());
  CHECK(gate.admit(2, false) == A::Skip);
  gate.lose();
  CHECK(gate.admit(3, true) == A::Restart);
  CHECK(!gate.waiting());
  gate.lose();
  CHECK(gate.admit(4, true) == A::Restart);
  CHECK(gate.admit(5, false) == A::Take);
  gate.lose();
  gate.lose();
  CHECK(gate.admit(7, true) == A::Restart);  // 6 missing too: still one key
  CHECK(gate.admit(8, false) == A::Take);
  return 0;
}

}  // namespace

int main() {
  if (test_orbbec_cases() != 0) return 1;
  if (test_start() != 0) return 1;
  if (test_gap() != 0) return 1;
  if (test_unnumbered_resets_the_last_number() != 0) return 1;
  if (test_lose() != 0) return 1;
  std::printf("key gate tests passed\n");
  return 0;
}
