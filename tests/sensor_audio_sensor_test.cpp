// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The audio source contract (audio_sensor.hpp) on a small fake built on the
// microphone driver's capture (audio/capture.hpp), its device a test function
// and its clock a variable; no device:
//   - the vocabulary's defaults; a source is neither copied nor moved through
//     the interface;
//   - drain appends every held chunk, oldest first, numbered, and nothing
//     before a start or after a stop;
//   - start and stop are idempotent: a second start keeps what is held and the
//     counters, a stop drops what is held and keeps the counters, a start
//     restarts them;
//   - a running source that delivers nothing for 1 s fails its drains until a
//     chunk arrives again.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

#include "capture.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/audio_sensor.hpp"

namespace audio = volumetric_kit::sensor::audio;
namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;
using Clock = audio::Capture::Clock;
using std::chrono::milliseconds;

static_assert(!std::is_copy_constructible_v<sensor::IAudioSensor>);
static_assert(!std::is_copy_assignable_v<sensor::IAudioSensor>);
static_assert(!std::is_move_constructible_v<sensor::IAudioSensor>);

namespace {

// A mono source of 4-frame chunks, 3 held, at 8 kHz, on a clock the test
// sets.
class FakeSource final : public sensor::IAudioSensor {
 public:
  FakeSource() : capture_(1, 4, 3) {
    info_.id = "fake-0";
    info_.model = "a test's tones";
    info_.sample_rate = 8000;
    info_.channels = 1;
    info_.chunk_frames = 4;
  }

  const sensor::AudioInfo& info() const noexcept override { return info_; }
  vkc::Status start() override {
    capture_.start(now);
    return {};
  }
  void stop() noexcept override { capture_.stop(); }
  vkc::Status drain(std::vector<sensor::AudioChunk>* out) override {
    return capture_.drain(out, now);
  }
  sensor::SensorStats stats() const noexcept override {
    return capture_.stats();
  }

  // As the device's thread would: a chunk of `value`s, its first sample
  // captured at `pts_ns`.
  void capture(std::int16_t value, std::int64_t pts_ns) {
    const std::int16_t s[4] = {value, value, value, value};
    capture_.on_input(s, 4, pts_ns, 0.0, 0.0, false);
  }

  Clock::time_point now{};

 private:
  sensor::AudioInfo info_;
  audio::Capture capture_;
};

int test_defaults() {
  const sensor::AudioInfo info;
  CHECK(info.id.empty() && info.model.empty());
  CHECK(info.clock == sensor::ClockDomain::Host);
  CHECK(info.sample_rate == 0 && info.channels == 0 && info.chunk_frames == 0);
  const sensor::AudioChunk chunk;
  CHECK(chunk.timestamp_ns == 0 && chunk.sequence == 0 &&
        chunk.samples.empty());
  return 0;
}

int test_drain_order() {
  auto owned = std::make_unique<FakeSource>();
  FakeSource& fake = *owned;
  std::unique_ptr<sensor::IAudioSensor> source = std::move(owned);
  std::vector<sensor::AudioChunk> out(1);  // appended after this one
  // Not started: nothing, and OK.
  CHECK(source->drain(&out).ok() && out.size() == 1);

  CHECK(source->start().ok());
  CHECK(source->drain(&out).ok() && out.size() == 1);  // none held yet
  fake.capture(1, 1000);
  fake.capture(2, 2000);
  fake.capture(3, 3000);
  CHECK(source->drain(&out).ok());
  CHECK(out.size() == 4);
  for (int i = 0; i < 3; ++i) {
    const sensor::AudioChunk& c = out[1 + i];
    CHECK(c.sequence == static_cast<std::uint64_t>(i));
    CHECK(c.timestamp_ns == 1000u * (i + 1));
    CHECK((c.samples == std::vector<std::int16_t>(4, i + 1)));
  }
  // Numbering goes on across drains.
  fake.capture(4, 4000);
  out.clear();
  CHECK(source->drain(&out).ok() && out.size() == 1 && out[0].sequence == 3);
  const sensor::SensorStats st = source->stats();
  CHECK(st.received == 4 && st.delivered == 4 && st.dropped == 0);
  return 0;
}

int test_start_stop_idempotent() {
  FakeSource fake;
  sensor::IAudioSensor& source = fake;
  source.stop();  // never started
  source.stop();
  CHECK(source.start().ok());
  fake.capture(1, 1000);
  fake.capture(2, 2000);
  CHECK(source.start().ok());  // running: what is held and the counts stay
  CHECK(source.stats().received == 2);
  std::vector<sensor::AudioChunk> out;
  CHECK(source.drain(&out).ok() && out.size() == 2);

  fake.capture(3, 3000);
  fake.capture(4, 4000);
  fake.capture(5, 5000);
  fake.capture(6, 6000);  // 3 held: dropped
  source.stop();          // drops the 3 held, keeps the counts
  source.stop();
  sensor::SensorStats st = source.stats();
  CHECK(st.received == 6 && st.delivered == 2 && st.dropped == 1);
  out.clear();
  CHECK(source.drain(&out).ok() && out.empty());  // stopped: nothing

  CHECK(source.start().ok());  // the counts restart, numbering too
  st = source.stats();
  CHECK(st.received == 0 && st.delivered == 0 && st.dropped == 0);
  CHECK(source.drain(&out).ok() && out.empty());  // the held ones went
  fake.capture(7, 7000);
  CHECK(source.drain(&out).ok() && out.size() == 1 && out[0].sequence == 0);
  CHECK(out[0].timestamp_ns == 7000);
  return 0;
}

int test_silence_means_gone() {
  FakeSource fake;
  fake.now = Clock::time_point{} + std::chrono::hours(1);
  const Clock::time_point t0 = fake.now;
  std::vector<sensor::AudioChunk> out;
  CHECK(fake.start().ok());
  // Nothing yet: OK until a whole second has passed since the start.
  fake.now = t0 + milliseconds(999);
  CHECK(fake.drain(&out).ok());
  fake.now = t0 + milliseconds(1000);
  vkc::Status s = fake.drain(&out);
  CHECK(s.domain() == vkc::Status::Code::IoError);
  // Again at every drain, until a chunk arrives.
  fake.now = t0 + milliseconds(1500);
  CHECK(!fake.drain(&out).ok());
  fake.capture(1, 1000);
  CHECK(fake.drain(&out).ok() && out.size() == 1);
  // The second is counted from the drain that saw it.
  fake.now = t0 + milliseconds(2499);
  CHECK(fake.drain(&out).ok());
  fake.capture(2, 2000);  // arrives; seen at 2.499 s
  CHECK(fake.drain(&out).ok());
  fake.now = t0 + milliseconds(3498);
  CHECK(fake.drain(&out).ok());
  fake.now = t0 + milliseconds(3499);
  CHECK(!fake.drain(&out).ok());
  // A stopped source never fails a drain; a start measures from itself.
  fake.stop();
  fake.now = t0 + milliseconds(9000);
  CHECK(fake.drain(&out).ok());
  CHECK(fake.start().ok());
  fake.now = t0 + milliseconds(9999);
  CHECK(fake.drain(&out).ok());
  fake.now = t0 + milliseconds(10000);
  CHECK(!fake.drain(&out).ok());
  return 0;
}

}  // namespace

int main() {
  if (test_defaults() != 0) return 1;
  if (test_drain_order() != 0) return 1;
  if (test_start_stop_idempotent() != 0) return 1;
  if (test_silence_means_gone() != 0) return 1;
  std::puts("sensor_audio_sensor: OK");
  return 0;
}
