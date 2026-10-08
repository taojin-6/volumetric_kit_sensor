// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The thread the Orbbec driver's colour decoders decode on, driven by a fake
// decode. Items are decoded one at a time, in order. A full queue drops its
// oldest (counted dropped) or all of it (counted lost), and the next item is
// told it follows a gap. A decode that throws costs only its item. fail()
// ends decoding, the waiting items lost with it, and the first failure
// stands. stop() and destruction wait for the decode in hand and drop what
// waits, uncounted. No push waits for a wake-up that never comes.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "decode_worker.hpp"

namespace vkc = volumetric_kit::core;
namespace orbbec = volumetric_kit::recon::sensor::orbbec;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

using Item = std::shared_ptr<int>;
using Worker = orbbec::DecodeWorker<Item>;
// Each item decoded, in order, and whether it came after a gap.
using Seen = std::vector<std::pair<int, bool>>;

constexpr auto kTimeout = std::chrono::seconds(10);

Item item(int value) { return std::make_shared<int>(value); }

// Items 0 to count - 1, each decoded in turn, none after a gap.
Seen in_order(int count) {
  Seen seen;
  for (int i = 0; i < count; ++i) seen.emplace_back(i, false);
  return seen;
}

// Whether `done` comes true within the timeout, polled.
template <typename Done>
bool eventually(Done done) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

// The fake decode: it records what it sees, holds at item `hold` until
// released, and throws at item `fault`.
class Recorder {
 public:
  int hold = -1;
  int fault = -1;

  Worker::Decode decoder() {
    return [this](Item i, bool after_gap) { decode(std::move(i), after_gap); };
  }

  void decode(Item i, bool after_gap) {
    std::unique_lock<std::mutex> lock(mutex_);
    seen_.emplace_back(*i, after_gap);
    if (*i == hold) {
      holding_ = true;
      changed_.notify_all();
      changed_.wait(lock, [this] { return released_; });
      finished_ = true;  // the held decode returns next
    }
    changed_.notify_all();
    if (*i == fault) throw std::runtime_error("an SDK call threw");
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    changed_.notify_all();
  }

  // Whether `count` items have been seen within the timeout.
  bool wait_for_seen(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, kTimeout,
                             [&] { return seen_.size() >= count; });
  }
  // Whether the decode holds within the timeout.
  bool wait_for_hold() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(lock, kTimeout, [&] { return holding_; });
  }

  Seen seen() {
    std::lock_guard<std::mutex> lock(mutex_);
    return seen_;
  }
  bool finished() {
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  Seen seen_;
  bool holding_ = false;
  bool released_ = false;
  bool finished_ = false;
};

// Releases the held decode on the way out, so that a failed check does not
// leave the worker's join waiting on it. Declared after the worker.
struct Releaser {
  Recorder& recorder;
  ~Releaser() { recorder.release(); }
};

// Every item, in order, none after a gap; once stopped, a push is neither
// held nor counted, and stop() reports that it had already stopped.
int test_decodes_in_order() {
  Recorder r;
  Worker worker(64, Worker::Overflow::DropAll);
  CHECK(worker.start(r.decoder()).ok());
  for (int i = 0; i < 50; ++i) worker.push(item(i));
  CHECK(r.wait_for_seen(50));
  CHECK(r.seen() == in_order(50));
  CHECK(worker.lost() == 0 && worker.dropped() == 0 && worker.failure().ok());
  CHECK(worker.stop());
  CHECK(!worker.stop());
  Item late = item(50);
  const std::weak_ptr<int> held = late;
  worker.push(std::move(late));
  CHECK(held.expired());
  CHECK(r.seen().size() == 50);
  CHECK(worker.lost() == 0 && worker.dropped() == 0);
  return 0;
}

// Two deep, the decode busy with 0: 3 and 4 replace 1 and 2, counted
// dropped, and 3 comes after a gap, 4 not.
int test_drop_oldest() {
  Recorder r;
  r.hold = 0;
  Worker worker(2, Worker::Overflow::DropOldest);
  CHECK(worker.start(r.decoder()).ok());
  Releaser releaser{r};
  worker.push(item(0));
  CHECK(r.wait_for_hold());
  for (int i = 1; i <= 4; ++i) worker.push(item(i));
  CHECK(worker.dropped() == 2 && worker.lost() == 0);
  r.release();
  CHECK(r.wait_for_seen(3));
  CHECK(worker.stop());
  CHECK((r.seen() == Seen{{0, false}, {3, true}, {4, false}}));
  CHECK(worker.dropped() == 2 && worker.lost() == 0);

  // A depth of 0 still holds the newest.
  Worker shallow(0, Worker::Overflow::DropOldest);
  shallow.push(item(1));
  shallow.push(item(2));
  CHECK(shallow.dropped() == 1);
  return 0;
}

// Two deep, the decode busy with 0: 3 finds 1 and 2 waiting, and all of
// them go, counted lost; 3 comes after a gap, 4 not.
int test_drop_all() {
  Recorder r;
  r.hold = 0;
  Worker worker(2, Worker::Overflow::DropAll);
  CHECK(worker.start(r.decoder()).ok());
  Releaser releaser{r};
  worker.push(item(0));
  CHECK(r.wait_for_hold());
  for (int i = 1; i <= 4; ++i) worker.push(item(i));
  CHECK(worker.lost() == 2 && worker.dropped() == 0);
  r.release();
  CHECK(r.wait_for_seen(3));
  CHECK(worker.stop());
  CHECK((r.seen() == Seen{{0, false}, {3, true}, {4, false}}));
  CHECK(worker.lost() == 2 && worker.dropped() == 0);
  return 0;
}

// The decode of 1 throws: it is lost, decoding goes on, and 2 comes after a
// gap.
int test_throw_costs_its_item() {
  Recorder r;
  r.fault = 1;
  Worker worker(8, Worker::Overflow::DropAll);
  CHECK(worker.start(r.decoder()).ok());
  for (int i = 0; i < 4; ++i) worker.push(item(i));
  CHECK(r.wait_for_seen(4));
  CHECK(worker.stop());
  CHECK((r.seen() == Seen{{0, false}, {1, false}, {2, true}, {3, false}}));
  CHECK(worker.lost() == 1 && worker.failure().ok());
  return 0;
}

// The decode of 0 fails for good with 1 and 2 waiting: they are lost, a
// later push is ignored, and a later failure does not replace the first.
int test_fail_ends_decoding() {
  Recorder r;
  r.hold = 0;
  Worker worker(8, Worker::Overflow::DropAll);
  CHECK(worker
            .start([&](Item i, bool after_gap) {
              const int value = *i;
              r.decode(std::move(i), after_gap);
              if (value == 0) worker.fail(vkc::Status::unsupported("refused"));
            })
            .ok());
  Releaser releaser{r};
  worker.push(item(0));
  CHECK(r.wait_for_hold());
  Item one = item(1);
  Item two = item(2);
  const std::weak_ptr<int> waiting_one = one;
  const std::weak_ptr<int> waiting_two = two;
  worker.push(std::move(one));
  worker.push(std::move(two));
  CHECK(worker.failure().ok());
  r.release();
  CHECK(eventually([&] { return !worker.failure().ok(); }));
  const vkc::Status why = worker.failure();
  CHECK(why.domain() == vkc::Status::Code::Unsupported);
  CHECK(why.message() == "refused");
  CHECK(worker.lost() == 2);
  CHECK(waiting_one.expired() && waiting_two.expired());

  Item late = item(3);
  const std::weak_ptr<int> held = late;
  worker.push(std::move(late));
  CHECK(held.expired());
  worker.fail(vkc::Status::io_error("later"));
  CHECK(worker.failure().message() == "refused");
  CHECK(worker.stop());
  CHECK((r.seen() == Seen{{0, false}}));
  CHECK(worker.lost() == 2 && worker.dropped() == 0);
  return 0;
}

// stop() with the decode busy and an item waiting drops the item, uncounted,
// before it waits, and returns only once the decode in hand has.
int test_stop_with_work_queued() {
  Recorder r;
  r.hold = 0;
  Worker worker(8, Worker::Overflow::DropAll);
  CHECK(worker.start(r.decoder()).ok());
  Releaser releaser{r};
  worker.push(item(0));
  CHECK(r.wait_for_hold());
  Item one = item(1);
  const std::weak_ptr<int> waiting = one;
  worker.push(std::move(one));
  bool stopped = false;
  bool finished = false;
  std::thread stopper([&] {
    stopped = worker.stop();
    finished = r.finished();
  });
  const bool dropped = eventually([&] { return waiting.expired(); });
  r.release();
  stopper.join();
  CHECK(dropped);
  CHECK(stopped && finished);
  CHECK((r.seen() == Seen{{0, false}}));
  CHECK(worker.lost() == 0 && worker.dropped() == 0);
  CHECK(!worker.stop());
  return 0;
}

// Destroyed while busy, as stop(): what waits is dropped, and the destructor
// returns only once the decode in hand has.
int test_destroyed_while_busy() {
  Recorder r;
  r.hold = 0;
  auto worker = std::make_unique<Worker>(8, Worker::Overflow::DropAll);
  CHECK(worker->start(r.decoder()).ok());
  Releaser releaser{r};
  worker->push(item(0));
  CHECK(r.wait_for_hold());
  Item one = item(1);
  const std::weak_ptr<int> waiting = one;
  worker->push(std::move(one));
  bool finished = false;
  std::thread destroyer([&] {
    worker.reset();
    finished = r.finished();
  });
  const bool dropped = eventually([&] { return waiting.expired(); });
  r.release();
  destroyer.join();
  CHECK(dropped && finished);
  CHECK((r.seen() == Seen{{0, false}}));
  return 0;
}

// Each push waits for its item to be decoded before the next, so the worker
// is asleep, or on its way there, at every push: a wake-up lost leaves an
// item undecoded until the wait times out. Then a burst from another thread,
// the queue deep enough to hold it: every item decoded, in order.
int test_no_lost_wakeups() {
  {
    Recorder r;
    Worker worker(4, Worker::Overflow::DropAll);
    CHECK(worker.start(r.decoder()).ok());
    constexpr int kRounds = 10000;
    for (int i = 0; i < kRounds; ++i) {
      worker.push(item(i));
      CHECK(r.wait_for_seen(static_cast<std::size_t>(i) + 1));
    }
    CHECK(worker.lost() == 0 && worker.dropped() == 0);
  }
  constexpr int kBurst = 100000;
  Recorder r;
  Worker worker(kBurst, Worker::Overflow::DropAll);
  CHECK(worker.start(r.decoder()).ok());
  std::thread producer([&] {
    for (int i = 0; i < kBurst; ++i) worker.push(item(i));
  });
  producer.join();
  CHECK(r.wait_for_seen(kBurst));
  CHECK(r.seen() == in_order(kBurst));
  CHECK(worker.lost() == 0 && worker.dropped() == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_decodes_in_order() != 0) return 1;
  if (test_drop_oldest() != 0) return 1;
  if (test_drop_all() != 0) return 1;
  if (test_throw_costs_its_item() != 0) return 1;
  if (test_fail_ends_decoding() != 0) return 1;
  if (test_stop_with_work_queued() != 0) return 1;
  if (test_destroyed_while_busy() != 0) return 1;
  if (test_no_lost_wakeups() != 0) return 1;
  std::puts("sensor_orbbec_decode_worker: OK");
  return 0;
}
