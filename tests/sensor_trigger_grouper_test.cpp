// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// How the trigger grouper turns per-sensor frames into one set per sync
// trigger -- with synthetic streams, no camera. The streams
// are shaped like the rig's: 30 fps, secondaries a fraction of a millisecond
// to ~2 ms off the primary, frames arriving in any order across cameras. Each
// set is built around a primary frame.

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/recon/sensor/trigger_grouper.hpp"

namespace sensor = volumetric_kit::recon::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr std::uint64_t kPeriod = 33333;  // us, 30 fps
constexpr std::size_t kCameras = 4;
constexpr std::size_t kPrimary = 2;
// Each camera's offset from the primary on the synced clock, as measured.
constexpr std::uint64_t kOffset[kCameras] = {1300, 400, 0, 1600};

sensor::TriggerGrouper::Config config() {
  sensor::TriggerGrouper::Config c;
  c.cameras = kCameras;
  c.anchor = kPrimary;
  c.tolerance_us = 5000;
  c.max_wait_us = 50000;
  c.queue_depth = 4;
  return c;
}

// The id of camera `c`'s frame for trigger `t`.
std::uint64_t id_of(std::size_t c, std::uint64_t t) { return t * 10 + c; }
std::uint64_t ts_of(std::size_t c, std::uint64_t t) {
  return 1000000 + t * kPeriod + kOffset[c];
}

int test_perfect_stream() {
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  std::uint64_t now = 0;
  for (std::uint64_t t = 0; t < 10; ++t) {
    // Frames of one trigger arrive in scrambled camera order.
    for (const std::size_t c : {3u, 0u, 2u, 1u}) {
      CHECK(!g.take(now, &released));  // nothing complete until the last
      g.add(c, ts_of(c, t), id_of(c, t), now, &released);
      now += 1000;
    }
    const auto group = g.take(now, &released);
    CHECK(group);
    for (std::size_t c = 0; c < kCameras; ++c) {
      CHECK(group->ids[c] == id_of(c, t));
    }
    // Named by the primary's frame, not the earliest.
    CHECK(group->timestamp_us == ts_of(kPrimary, t));
    now += kPeriod - 4000;
  }
  CHECK(released.empty());
  return 0;
}

int test_dropped_frame() {
  // Camera 1 drops trigger 1's frame. The set goes out without it once camera
  // 1's frame for trigger 2 shows it has moved past -- not a whole wait later.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  std::uint64_t now = 0;
  for (std::size_t c = 0; c < kCameras; ++c) {
    g.add(c, ts_of(c, 0), id_of(c, 0), now, &released);
  }
  CHECK(g.take(now, &released));
  for (std::size_t c : {0u, 2u, 3u}) {
    g.add(c, ts_of(c, 1), id_of(c, 1), now, &released);
  }
  CHECK(!g.take(now + 1000, &released));  // camera 1 may still send it
  g.add(1, ts_of(1, 2), id_of(1, 2), now + 2000, &released);
  const auto group = g.take(now + 2000, &released);
  CHECK(group);
  CHECK(group->ids[0] == id_of(0, 1) && group->ids[2] == id_of(2, 1) &&
        group->ids[3] == id_of(3, 1));
  CHECK(!group->ids[1]);
  CHECK(released.empty());
  return 0;
}

int test_silent_camera() {
  // Camera 3 sends nothing at all: each set waits max_wait for it, then goes
  // out without it.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  for (std::size_t c : {0u, 1u, 2u}) {
    g.add(c, ts_of(c, 0), id_of(c, 0), 100, &released);
  }
  CHECK(!g.take(100 + 49000, &released));
  const auto group = g.take(100 + 50000, &released);
  CHECK(group && group->ids[0] && group->ids[1] && group->ids[2] &&
        !group->ids[3]);
  return 0;
}

int test_slow_poll() {
  // Six triggers arrive before one poll: the newest goes out; the older five
  // are released, and the queues never hold more than their depth.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  for (std::uint64_t t = 0; t < 6; ++t) {
    for (std::size_t c = 0; c < kCameras; ++c) {
      g.add(c, ts_of(c, t), id_of(c, t), t * kPeriod, &released);
    }
  }
  // Depth 4: triggers 0 and 1 were let go as trigger 4 and 5 arrived.
  CHECK(released.size() == 2 * kCameras);
  released.clear();
  const auto group = g.take(6 * kPeriod, &released);
  CHECK(group);
  for (std::size_t c = 0; c < kCameras; ++c) {
    CHECK(group->ids[c] == id_of(c, 5));
  }
  CHECK(released.size() == 3 * kCameras);  // triggers 2, 3 and 4
  CHECK(!g.take(6 * kPeriod, &released));
  return 0;
}

int clock_beyond_tolerance(std::int64_t offset_us) {
  // Camera 0's clock runs `offset_us` off, past the 5 ms tolerance, so its
  // frames never join a trigger. Polled every millisecond, as fuse_orbbec
  // polls, with each trigger's frames arriving a few hundred microseconds
  // apart: every trigger still goes out once, holding the other three cameras,
  // and camera 0's frames are let go rather than handed out on their own.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  constexpr std::uint64_t kTriggers = 30;
  constexpr std::uint64_t kArrival[kCameras] = {300, 100, 0, 200};
  std::uint64_t next[kCameras] = {};  // each camera's next trigger to add
  std::uint64_t sets = 0, last_ts = 0;
  for (std::uint64_t now = 0; now < (kTriggers + 3) * kPeriod; now += 1000) {
    for (std::size_t c = 0; c < kCameras; ++c) {
      while (next[c] < kTriggers &&
             20000 + next[c] * kPeriod + kArrival[c] <= now) {
        const std::uint64_t t = next[c]++;
        const std::uint64_t ts =
            c == 0 ? static_cast<std::uint64_t>(
                         static_cast<std::int64_t>(ts_of(c, t)) + offset_us)
                   : ts_of(c, t);
        g.add(c, ts, id_of(c, t), now, &released);
      }
    }
    if (const auto group = g.take(now, &released)) {
      ++sets;
      CHECK(!group->ids[0]);
      CHECK(group->ids[1] && group->ids[kPrimary] && group->ids[3]);
      CHECK(group->timestamp_us > last_ts);
      last_ts = group->timestamp_us;
    }
  }
  CHECK(sets == kTriggers);
  return 0;
}

int test_clock_beyond_tolerance() {
  if (clock_beyond_tolerance(+8000) != 0) return 1;
  return clock_beyond_tolerance(-8000);
}

int test_late_frame_after_its_set() {
  // Camera 1's frame for trigger 0 arrives after trigger 0 went out without
  // it, while the other cameras' trigger-1 frames wait on camera 1. It is let
  // go -- not handed out as a set of its own, older than the one before it.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  for (std::size_t c : {0u, 2u, 3u}) {
    g.add(c, ts_of(c, 0), id_of(c, 0), 0, &released);
  }
  const auto first = g.take(50000, &released);
  CHECK(first && !first->ids[1]);
  for (std::size_t c : {0u, 2u, 3u}) {
    g.add(c, ts_of(c, 1), id_of(c, 1), 55000, &released);
  }
  g.add(1, ts_of(1, 0), id_of(1, 0), 56000, &released);
  CHECK(!g.take(56000, &released));
  g.add(1, ts_of(1, 1), id_of(1, 1), 57000, &released);
  const auto second = g.take(57000, &released);
  CHECK(second && second->ids[1] == id_of(1, 1));
  CHECK(second->timestamp_us > first->timestamp_us);
  CHECK((released == std::vector<std::uint64_t>{id_of(1, 0)}));
  return 0;
}

int test_secondaries_before_primary() {
  // The primary's first triggers reach the secondaries ~0.5 s before its own
  // first frame reaches the host (measured: ~16 triggers). None of those
  // frames makes a set; the first set is the primary's first frame's.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  for (std::uint64_t t = 0; t < 6; ++t) {
    for (std::size_t c : {0u, 1u, 3u}) {
      g.add(c, ts_of(c, t), id_of(c, t), t * kPeriod, &released);
    }
    CHECK(!g.take(t * kPeriod + 1000, &released));
  }
  for (std::size_t c = 0; c < kCameras; ++c) {
    g.add(c, ts_of(c, 6), id_of(c, 6), 6 * kPeriod, &released);
  }
  const auto group = g.take(6 * kPeriod, &released);
  CHECK(group);
  for (std::size_t c = 0; c < kCameras; ++c) {
    CHECK(group->ids[c] == id_of(c, 6));
  }
  CHECK(released.size() == 3 * 6);  // every start-up frame, let go
  return 0;
}

int test_dropped_primary_frame() {
  // The primary's frame names a trigger, so a trigger it dropped is not
  // handed out; the secondaries' frames for it are let go.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  for (std::size_t c = 0; c < kCameras; ++c) {
    g.add(c, ts_of(c, 0), id_of(c, 0), 0, &released);
  }
  CHECK(g.take(0, &released));
  for (std::size_t c : {0u, 1u, 3u}) {
    g.add(c, ts_of(c, 1), id_of(c, 1), kPeriod, &released);
  }
  CHECK(!g.take(kPeriod + 60000, &released));  // long past any wait
  for (std::size_t c = 0; c < kCameras; ++c) {
    g.add(c, ts_of(c, 2), id_of(c, 2), 2 * kPeriod, &released);
  }
  const auto group = g.take(2 * kPeriod, &released);
  CHECK(group && group->timestamp_us == ts_of(kPrimary, 2));
  CHECK(released.size() == 3);
  return 0;
}

int test_out_of_order_within_camera() {
  // A clock re-sync can step a camera's clock back a little; its frames are
  // still grouped by timestamp.
  sensor::TriggerGrouper g(config());
  std::vector<std::uint64_t> released;
  g.add(1, ts_of(1, 1), id_of(1, 1), 0, &released);
  g.add(1, ts_of(1, 0), id_of(1, 0), 0, &released);
  for (std::size_t c : {0u, 2u, 3u}) {
    g.add(c, ts_of(c, 0), id_of(c, 0), 0, &released);
  }
  const auto group = g.take(0, &released);
  CHECK(group && group->ids[1] == id_of(1, 0));
  // clear() gives everything held back.
  g.clear(&released);
  CHECK(released.size() == 1 && released[0] == id_of(1, 1));
  return 0;
}

}  // namespace

int main() {
  if (test_perfect_stream() != 0) return 1;
  if (test_dropped_frame() != 0) return 1;
  if (test_silent_camera() != 0) return 1;
  if (test_slow_poll() != 0) return 1;
  if (test_clock_beyond_tolerance() != 0) return 1;
  if (test_late_frame_after_its_set() != 0) return 1;
  if (test_secondaries_before_primary() != 0) return 1;
  if (test_dropped_primary_frame() != 0) return 1;
  if (test_out_of_order_within_camera() != 0) return 1;
  std::printf("trigger grouper tests passed\n");
  return 0;
}
