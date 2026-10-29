// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The RealSense driver's frame queue (realsense/frame_queue.hpp), with no SDK:
//   - at depth 1 the newest frame wins, each one passed over counted dropped;
//   - a deeper queue drains oldest first, a frame pushed out of a full queue
//     counted dropped;
//   - the consumer is woken once a frame or a failure, after it is held;
//   - a failure is reported to every later take, the first one kept;
//   - a stop clears it and keeps the count, which a start restarts; a
//     discarded frame is counted dropped;
//   - with a producer thread pushing, every frame is taken or dropped once.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "frame_queue.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace vkc = volumetric_kit::core;
using Queue = volumetric_kit::sensor::realsense::FrameQueue<int>;

namespace {

int test_newest_wins() {
  Queue q;
  int woken = 0;
  q.set_notify([&woken] { ++woken; });
  auto none = q.take_newest();
  CHECK(none.ok() && !none.value());
  for (int i = 1; i <= 4; ++i) q.push(i);
  CHECK(woken == 4);
  CHECK(q.dropped() == 3);  // 1, 2 and 3 pushed out at depth 1
  auto newest = q.take_newest();
  CHECK(newest.ok() && newest.value() && *newest.value() == 4);
  CHECK(q.dropped() == 3);
  return 0;
}

int test_drain_oldest_first() {
  Queue q;
  q.set_depth(3);
  for (int i = 1; i <= 5; ++i) q.push(i);
  CHECK(q.dropped() == 2);
  std::vector<int> out{0};
  CHECK(q.take_all(&out).ok());
  CHECK((out == std::vector<int>{0, 3, 4, 5}));  // appended, oldest first
  CHECK(q.take_all(&out).ok() && out.size() == 4);
  q.push(6);
  q.push(7);
  auto newest = q.take_newest();  // passes 6 over
  CHECK(newest.ok() && *newest.value() == 7 && q.dropped() == 3);
  q.discard();
  CHECK(q.dropped() == 4);
  q.set_depth(0);  // held at 1
  q.push(8);
  q.push(9);
  CHECK(q.dropped() == 5);
  return 0;
}

int test_failure() {
  Queue q;
  int woken = 0;
  q.set_notify([&woken] { ++woken; });
  q.push(1);
  CHECK(!q.failed());
  q.fail("camera 123 disconnected");
  q.fail("a later failure");
  CHECK(q.failed() && woken == 3);
  for (int i = 0; i < 2; ++i) {
    const vkc::Status s = q.take_newest().status();
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message() == "camera 123 disconnected");
  }
  std::vector<int> out;
  CHECK(q.take_all(&out).domain() == vkc::Status::Code::IoError);
  CHECK(out.empty());
  return 0;
}

int test_clear() {
  Queue q;
  q.set_depth(2);
  for (int i = 0; i < 5; ++i) q.push(i);
  CHECK(q.dropped() == 3);
  q.clear();
  CHECK(q.dropped() == 3);  // read after a stop
  auto none = q.take_newest();
  CHECK(none.ok() && !none.value());
  q.reset_dropped();
  CHECK(q.dropped() == 0);
  return 0;
}

int test_threads() {
  Queue q;
  q.set_depth(4);
  constexpr int kFrames = 20000;
  std::atomic<bool> done{false};
  std::thread producer([&] {
    for (int i = 0; i < kFrames; ++i) q.push(i);
    done.store(true);
  });
  std::uint64_t taken = 0;
  int last = -1;
  bool ordered = true;
  std::vector<int> out;
  for (;;) {
    const bool finished = done.load();
    out.clear();
    if (!q.take_all(&out).ok()) return 1;
    for (const int v : out) {
      ordered = ordered && v > last;
      last = v;
    }
    taken += out.size();
    if (finished && out.empty()) break;
  }
  producer.join();
  std::printf("  threads: %llu taken, %llu dropped of %d\n",
              static_cast<unsigned long long>(taken),
              static_cast<unsigned long long>(q.dropped()), kFrames);
  CHECK(ordered && last == kFrames - 1);
  CHECK(taken + q.dropped() == static_cast<std::uint64_t>(kFrames));
  return 0;
}

}  // namespace

int main() {
  if (test_newest_wins() != 0) return 1;
  if (test_drain_oldest_first() != 0) return 1;
  if (test_failure() != 0) return 1;
  if (test_clear() != 0) return 1;
  if (test_threads() != 0) return 1;
  std::puts("sensor_realsense_queue: OK");
  return 0;
}
