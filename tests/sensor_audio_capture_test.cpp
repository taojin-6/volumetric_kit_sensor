// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// What the microphone driver does with each chunk, apart from PortAudio
// (audio/capture.hpp, audio/chunk_ring.hpp), on synthetic tones; no device:
//   - a chunk's stamp is the host time less its first sample's wait, never
//     below the chunk before's, a host clock that steps back included; a wait
//     the backend cannot report counts as none;
//   - the formats taken and refused;
//   - the queue hands chunks out oldest first, lets the chunk arriving go when
//     full, and wraps around its entries;
//   - the capture counts what it received, dropped, refused and lost to an
//     overflow;
//   - fed from a second thread, as PortAudio's callback feeds it, every chunk
//     is handed out or dropped once, in order, with its own samples and stamp
//     (CI runs this under ThreadSanitizer).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

#include "capture.hpp"
#include "chunk_ring.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace audio = volumetric_kit::sensor::audio;
namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;
using Clock = audio::Capture::Clock;

namespace {

constexpr std::int64_t kSecond = 1'000'000'000;

// `count` samples of a 1 kHz tone at 48 kHz, from sample `first` on.
std::vector<std::int16_t> tone(std::size_t count, std::size_t first = 0) {
  std::vector<std::int16_t> samples(count);
  for (std::size_t i = 0; i < count; ++i) {
    const double t = static_cast<double>(first + i) / 48000.0;
    samples[i] = static_cast<std::int16_t>(
        std::lround(12000.0 * std::sin(2.0 * 3.14159265358979 * 1000.0 * t)));
  }
  return samples;
}

int test_pts_mapping() {
  const std::int64_t now = 1'800'000'000 * kSecond;  // 2027, on system_clock
  // The first sample waited 25 ms: 100.025 - 100.000 on the backend's clock.
  CHECK(audio::chunk_pts_ns(now, 100.025, 100.0, 0) ==
        static_cast<std::uint64_t>(now - 25'000'000));
  // Only the difference counts, whatever the backend clock's origin.
  CHECK(audio::chunk_pts_ns(now, 5.5, 5.49, 0) ==
        static_cast<std::uint64_t>(now - 10'000'000));
  // No wait: the host time itself.
  CHECK(audio::chunk_pts_ns(now, 7.0, 7.0, 0) ==
        static_cast<std::uint64_t>(now));
  // A wait the backend cannot report counts as none: no ADC time (0, so the
  // whole uptime), an ADC time after the callback, NaN, 1 s or more.
  CHECK(audio::chunk_pts_ns(now, 86400.0, 0.0, 0) ==
        static_cast<std::uint64_t>(now));
  CHECK(audio::chunk_pts_ns(now, 10.0, 10.5, 0) ==
        static_cast<std::uint64_t>(now));
  CHECK(audio::chunk_pts_ns(now, std::nan(""), 1.0, 0) ==
        static_cast<std::uint64_t>(now));
  CHECK(audio::chunk_pts_ns(now, 11.0, 10.0, 0) ==
        static_cast<std::uint64_t>(now));
  CHECK(audio::chunk_pts_ns(now, 10.999, 10.0, 0) ==
        static_cast<std::uint64_t>(now - 999'000'000));
  // Never below 0.
  CHECK(audio::chunk_pts_ns(1000, 0.5, 0.0, 0) == 0);
  return 0;
}

int test_pts_never_go_back() {
  // 20-ms chunks, each waiting 5 ms, on a host clock that steps back 30 ms
  // before the third: that chunk takes the second's stamp, and the fourth
  // goes on from the stepped clock.
  const std::int64_t t0 = 1'800'000'000 * kSecond;
  const std::int64_t ms = 1'000'000;
  std::uint64_t last = 0;
  const std::int64_t now[] = {t0, t0 + 20 * ms, t0 + 10 * ms, t0 + 60 * ms};
  const std::uint64_t want[] = {
      static_cast<std::uint64_t>(t0 - 5 * ms),
      static_cast<std::uint64_t>(t0 + 15 * ms),
      static_cast<std::uint64_t>(t0 + 15 * ms),  // 5 ms would go back 10
      static_cast<std::uint64_t>(t0 + 55 * ms)};
  for (int i = 0; i < 4; ++i) {
    const double adc = 50.0 + 0.020 * i;
    last = audio::chunk_pts_ns(now[i], adc + 0.005, adc, last);
    CHECK(last == want[i]);
  }
  // A shorter wait on the same host time cannot move a stamp back either.
  CHECK(audio::chunk_pts_ns(t0, 1.010, 1.0, static_cast<std::uint64_t>(t0)) ==
        static_cast<std::uint64_t>(t0));
  return 0;
}

int test_check_format() {
  CHECK(audio::check_format(48000, 1, 960, 512).ok());
  CHECK(audio::check_format(44100, 2, 1, 1).ok());
  CHECK(audio::check_format(48000, 2, 24000, 1).ok());  // half a second
  const auto refused = [](const vkc::Status& s) {
    return s.domain() == vkc::Status::Code::InvalidArgument;
  };
  CHECK(refused(audio::check_format(0, 1, 960, 512)));
  CHECK(refused(audio::check_format(48000, 0, 960, 512)));
  CHECK(refused(audio::check_format(48000, 3, 960, 512)));
  CHECK(refused(audio::check_format(48000, 1, 0, 512)));
  CHECK(refused(audio::check_format(48000, 1, 24001, 512)));
  CHECK(refused(audio::check_format(48000, 1, 960, 0)));
  return 0;
}

int test_ring_order_and_full() {
  audio::ChunkRing ring(3, 4);
  CHECK(ring.entries() == 3 && ring.entry_samples() == 4);
  const std::int16_t a[] = {1, 2, 3, 4};
  CHECK(ring.push(10, a, 4));
  CHECK(ring.push(20, a + 1, 3));
  CHECK(ring.push(30, nullptr, 2));  // silence
  CHECK(!ring.push(40, a, 4));       // full: the chunk arriving goes
  CHECK(!ring.push(50, a, 5));       // longer than an entry
  std::vector<std::uint64_t> pts;
  std::vector<std::vector<std::int16_t>> got;
  CHECK(
      ring.pop_all([&](std::uint64_t p, const std::int16_t* s, std::size_t n) {
        pts.push_back(p);
        got.emplace_back(s, s + n);
      }) == 3);
  CHECK((pts == std::vector<std::uint64_t>{10, 20, 30}));
  CHECK((got[0] == std::vector<std::int16_t>{1, 2, 3, 4}));
  CHECK((got[1] == std::vector<std::int16_t>{2, 3, 4}));
  CHECK((got[2] == std::vector<std::int16_t>{0, 0}));
  CHECK(ring.pop_all([](std::uint64_t, const std::int16_t*, std::size_t) {}) ==
        0);
  // Room again once taken; clear lets the held ones go.
  CHECK(ring.push(60, a, 4) && ring.push(70, a, 4));
  ring.clear();
  CHECK(ring.pop_all([](std::uint64_t, const std::int16_t*, std::size_t) {}) ==
        0);
  return 0;
}

int test_ring_wraps_around() {
  // 3 entries, 1000 chunks pushed in runs of 1 and 2: the entries are reused
  // hundreds of times, every chunk comes back whole and in order.
  audio::ChunkRing ring(3, 8);
  std::uint64_t pushed = 0;
  std::uint64_t next = 0;
  bool whole = true;
  while (pushed < 1000) {
    const int run = 1 + static_cast<int>(pushed % 2);
    for (int i = 0; i < run && pushed < 1000; ++i, ++pushed) {
      const std::vector<std::int16_t> s = tone(8, pushed * 8);
      CHECK(ring.push(pushed, s.data(), s.size()));
    }
    ring.pop_all([&](std::uint64_t p, const std::int16_t* s, std::size_t n) {
      whole = whole && p == next && n == 8 &&
              std::vector<std::int16_t>(s, s + n) == tone(8, next * 8);
      ++next;
    });
  }
  CHECK(whole && next == 1000);
  return 0;
}

int test_capture_counts() {
  // Mono, 4-frame chunks, 2 held.
  audio::Capture capture(1, 4, 2);
  const Clock::time_point t0{};
  const std::int64_t now = 1'800'000'000 * kSecond;
  const std::vector<std::int16_t> s = tone(4);
  std::vector<sensor::AudioChunk> out;
  capture.start(t0);
  capture.on_input(s.data(), 4, now, 1.0, 1.0, false);
  capture.on_input(s.data(), 4, now + 1, 1.0, 1.0, true);   // overflow before
  capture.on_input(s.data(), 4, now + 2, 1.0, 1.0, false);  // full: dropped
  capture.on_input(s.data(), 5, now + 3, 1.0, 1.0, false);  // too long: failed
  capture.on_input(s.data(), 0, now + 4, 1.0, 1.0, false);  // empty: failed
  sensor::SensorStats st = capture.stats();
  CHECK(st.received == 5 && st.dropped == 1 && st.failed == 2);
  CHECK(st.delivered == 0 && capture.overflows() == 1);
  CHECK(capture.drain(&out, t0).ok());
  CHECK(out.size() == 2);
  // The held ones stayed; the chunk arriving at a full queue went.
  CHECK(out[0].timestamp_ns == static_cast<std::uint64_t>(now));
  CHECK(out[1].timestamp_ns == static_cast<std::uint64_t>(now + 1));
  CHECK(out[0].samples == s && out[1].samples == s);
  st = capture.stats();
  CHECK(st.delivered == 2);
  CHECK(st.delivered + st.dropped + st.failed == st.received);
  // A stereo capture takes frames of two samples.
  audio::Capture stereo(2, 4, 2);
  stereo.start(t0);
  const std::vector<std::int16_t> s8 = tone(8);
  stereo.on_input(s8.data(), 4, now, 1.0, 1.0, false);
  out.clear();
  CHECK(stereo.drain(&out, t0).ok());
  CHECK(out.size() == 1 && out[0].samples == s8);
  return 0;
}

int test_threads() {
  // A producer thread feeds 20 ms stereo chunks of a tone, as PortAudio's
  // callback does, stepping the host clock back now and then, while this
  // thread drains. Each chunk carries its index in its first sample. After a
  // dropped push the producer waits for a drain, so the 8 entries are reused
  // hundreds of times while both threads run. Every chunk is handed out or
  // dropped once; the delivered ones come in order, each with its own samples
  // and stamp.
  constexpr std::uint32_t kFrames = 960;
  constexpr int kChunks = 4000;
  constexpr std::int64_t kT0 = 1'800'000'000 * kSecond;
  constexpr std::int64_t kMs = 1'000'000;
  audio::Capture capture(2, kFrames, 8);
  // One steady time throughout: the silence rule, the contract test's, never
  // fires on a slow runner.
  const Clock::time_point t{};
  capture.start(t);
  std::vector<std::vector<std::int16_t>> tones;
  for (int i = 0; i < 16; ++i) tones.push_back(tone(2 * kFrames, i * 7));
  // Chunk i's host time; every tenth steps back 30 ms. Each waited 4 ms.
  const auto host_ns = [](int i) {
    return kT0 + i * 20 * kMs - (i % 10 == 9 ? 30 * kMs : 0);
  };
  // Its stamp: the latest host time so far, less the wait.
  const auto stamp = [&](int i) {
    const std::int64_t latest = i % 10 == 9 ? host_ns(i - 1) : host_ns(i);
    return static_cast<std::uint64_t>(latest - 4 * kMs);
  };
  std::atomic<std::uint64_t> drained{0};  // chunks this thread has taken
  std::atomic<bool> done{false};
  std::thread producer([&] {
    std::vector<std::int16_t> chunk;
    for (int i = 0; i < kChunks; ++i) {
      chunk = tones[i % 16];
      chunk[0] = static_cast<std::int16_t>(i);
      const std::uint64_t seen = drained.load();
      if (!capture.on_input(chunk.data(), kFrames, host_ns(i), 2.0 + 0.004, 2.0,
                            i % 100 == 0)) {
        while (drained.load() == seen) std::this_thread::yield();
      }
    }
    done.store(true);
  });
  std::vector<sensor::AudioChunk> out;
  std::uint64_t taken = 0;
  int last = -1;
  bool drains_ok = true;
  bool whole = true;
  for (;;) {
    const bool finished = done.load();
    out.clear();
    drains_ok = drains_ok && capture.drain(&out, t).ok();
    for (const sensor::AudioChunk& c : out) {
      const int i = c.samples.size() == 2 * kFrames ? c.samples[0] : -1;
      whole = whole && i > last && i < kChunks && c.sequence == taken &&
              c.timestamp_ns == stamp(i) &&
              std::equal(c.samples.begin() + 1, c.samples.end(),
                         tones[static_cast<std::size_t>(i % 16)].begin() + 1);
      last = i;
      ++taken;
    }
    drained.store(taken);
    if (finished && out.empty()) break;
  }
  producer.join();
  const sensor::SensorStats st = capture.stats();
  std::printf("  threads: %llu delivered, %llu dropped of %d\n",
              static_cast<unsigned long long>(st.delivered),
              static_cast<unsigned long long>(st.dropped), kChunks);
  CHECK(drains_ok && whole);
  CHECK(st.received == static_cast<std::uint64_t>(kChunks));
  CHECK(st.delivered == taken && st.failed == 0);
  CHECK(st.delivered + st.dropped == st.received);
  // Each drop waits for a drain of at least one chunk.
  CHECK(st.delivered >= st.dropped);
  CHECK(capture.overflows() == kChunks / 100);
  return 0;
}

}  // namespace

int main() {
  if (test_pts_mapping() != 0) return 1;
  if (test_pts_never_go_back() != 0) return 1;
  if (test_check_format() != 0) return 1;
  if (test_ring_order_and_full() != 0) return 1;
  if (test_ring_wraps_around() != 0) return 1;
  if (test_capture_counts() != 0) return 1;
  if (test_threads() != 0) return 1;
  std::puts("sensor_audio_capture: OK");
  return 0;
}
