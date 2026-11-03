// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The frame queue every threaded driver holds (frame_queue.hpp), with no SDK:
//   - at depth 1 the newest frame wins, each one passed over counted dropped;
//   - a deeper queue drains oldest first, a frame pushed out of a full queue
//     counted dropped; its depth reads back, held at 1 or more;
//   - the consumer is woken once a frame or a failure, after it is held, and
//     by a wake with nothing held;
//   - a failure is reported to every later take as it was given, the first
//     one kept, until a reset forgets it;
//   - a stop clears it and keeps the count, which a start restarts; a
//     discarded frame is counted dropped;
//   - with a producer thread pushing, every frame is taken or dropped once;
//     and a consumer asleep until woken, as a driver's is, sees each frame
//     and the failure that ends the run.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "frame_queue.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace vkc = volumetric_kit::core;
using Queue = volumetric_kit::sensor::detail::FrameQueue<int>;

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
  q.wake();  // nothing held: woken, and nothing to take
  CHECK(woken == 5 && q.dropped() == 3);
  none = q.take_newest();
  CHECK(none.ok() && !none.value());
  return 0;
}

int test_drain_oldest_first() {
  Queue q;
  CHECK(q.depth() == 1);
  q.set_depth(3);
  CHECK(q.depth() == 3);
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
  CHECK(q.depth() == 1);
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
  CHECK(!q.failed() && q.failure().ok());
  q.fail(vkc::Status::io_error("camera 123 disconnected"));
  q.fail(vkc::Status::unsupported("a later failure"));
  CHECK(q.failed() && woken == 3);
  CHECK(q.failure().domain() == vkc::Status::Code::IoError);
  for (int i = 0; i < 2; ++i) {
    const vkc::Status s = q.take_newest().status();
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message() == "camera 123 disconnected");
  }
  std::vector<int> out;
  CHECK(q.take_all(&out).domain() == vkc::Status::Code::IoError);
  CHECK(out.empty());
  // A run's failure, as a GPU stage's thread ends one, kept as given; a
  // reset at the next start forgets it, and the frame it held is still
  // there for a clear to let go.
  Queue run;
  run.set_depth(2);
  run.push(1);
  run.fail(vkc::Status::out_of_memory("FFS: out of device memory"));
  CHECK(run.take_newest().status().domain() == vkc::Status::Code::OutOfMemory);
  run.reset_failure();
  CHECK(!run.failed() && run.failure().ok());
  run.clear();
  run.push(2);
  auto next = run.take_newest();
  CHECK(next.ok() && next.value() && *next.value() == 2);
  run.fail(vkc::Status::io_error("the camera stopped"));
  CHECK(run.take_all(&out).message() == "the camera stopped");
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

// A driver's consumer: asleep until the queue's notification, then a poll,
// until the producer's failure ends the run. Every wake-up is from a push,
// a wake or the failure, so it never sleeps through the last of them.
int test_woken_consumer() {
  Queue q;
  std::mutex mutex;
  std::condition_variable cv;
  bool woken = false;  // guarded by mutex
  q.set_notify([&] {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      woken = true;
    }
    cv.notify_one();
  });
  constexpr int kFrames = 5000;
  std::thread producer([&] {
    for (int i = 0; i < kFrames; ++i) {
      q.push(i);
      if (i % 64 == 0) q.wake();
    }
    q.fail(vkc::Status::io_error("the run ended"));
  });
  std::uint64_t taken = 0;
  int last = -1;
  bool ordered = true;
  vkc::Status ended;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      cv.wait(lock, [&] { return woken; });
      woken = false;
    }
    auto polled = q.take_newest();
    if (!polled) {
      ended = polled.status();
      break;
    }
    if (!polled.value()) continue;  // a wake with nothing new
    ordered = ordered && *polled.value() > last;
    last = *polled.value();
    ++taken;
  }
  producer.join();
  std::printf("  woken consumer: %llu taken, %llu dropped of %d\n",
              static_cast<unsigned long long>(taken),
              static_cast<unsigned long long>(q.dropped()), kFrames);
  CHECK(ordered && ended.message() == "the run ended");
  // The one frame a depth of 1 may still hold at the failure is neither
  // taken nor dropped.
  const std::uint64_t counted = taken + q.dropped();
  const auto frames = static_cast<std::uint64_t>(kFrames);
  CHECK(counted == frames || counted + 1 == frames);
  return 0;
}

}  // namespace

int main() {
  if (test_newest_wins() != 0) return 1;
  if (test_drain_oldest_first() != 0) return 1;
  if (test_failure() != 0) return 1;
  if (test_clear() != 0) return 1;
  if (test_threads() != 0) return 1;
  if (test_woken_consumer() != 0) return 1;
  std::puts("sensor_frame_queue: OK");
  return 0;
}
