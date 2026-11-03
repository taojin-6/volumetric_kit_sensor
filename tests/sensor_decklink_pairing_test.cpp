// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The DeckLink driver's SDK-free host units, on synthetic arrivals, no card:
//   - ports: the connectors' sub-devices, the modes, v210 rows, the period;
//   - check_config: each option DecklinkStereo::open refuses before any
//     device, named;
//   - the pairing: free-running cameras 8.2 ms apart always pair, either
//     arrival order; a lost left or right frame, an input without signal
//     and back, a skew past the bound and then within it, the newest
//     unpaired frame winning, a restart; the counts in left frames and the
//     sync contract (stereo_contract.hpp) on what it paired;
//   - the clock map: a card clock 18.8 ppm off the host's mapped within
//     10 us over an hour; a camera 27 ppm off nominal gives no false
//     sequence gap in an hour;
//   - the skew report on synthetic logs of today's free run: SDI 2 lagging
//     SDI 1 by 8.20 ms, falling 0.0367 ms a minute, and lost frames.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "driver.hpp"
#include "pairing.hpp"
#include "ports.hpp"
#include "skew_stats.hpp"
#include "stereo_contract.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace dl = volumetric_kit::sensor::decklink;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace vkc = volumetric_kit::core;

namespace {

constexpr std::int64_t kMs = 1'000'000;
constexpr std::uint64_t kPeriod = 33'333'333;  // 30 fps
constexpr std::uint64_t kHalfFrame = 16'666'666;

bool says(const vkc::Status& s, const char* text) {
  if (s.ok() || s.domain() != vkc::Status::Code::InvalidArgument) return false;
  if (s.message().find(text) == std::string::npos) {
    std::fprintf(stderr, "  message: %s\n", s.message().c_str());
    return false;
  }
  return true;
}

// A deterministic jitter in [-span, span] ns.
struct Jitter {
  std::uint64_t state = 0x9E3779B97F4A7C15ull;
  std::int64_t operator()(std::int64_t span) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const double u = static_cast<double>(state >> 11) * 0x1.0p-53;  // [0, 1)
    return static_cast<std::int64_t>((2.0 * u - 1.0) *
                                     static_cast<double>(span));
  }
};

// Hands frames back as the SDK would get them; counts how many came back.
struct Holds {
  std::shared_ptr<int> back = std::make_shared<int>(0);
  std::shared_ptr<void> make() {
    return std::shared_ptr<void>(new int(0), [back = back](void* p) {
      ++*back;
      delete static_cast<int*>(p);
    });
  }
};

// A frame at card time `card`, its callback 3.6 ms later on a card clock
// that equals the host's.
dl::Arrival frame(int input, std::int64_t card, Holds& holds,
                  bool no_signal = false) {
  dl::Arrival a;
  a.input = input;
  a.card_ns = card;
  a.card_now_ns = card + 3'600'000;
  a.host_now_ns = a.card_now_ns + 1'700'000'000'000'000'000;
  a.no_signal = no_signal;
  a.hold = holds.make();
  return a;
}

// What the pairing made, as the frames a source would hand out.
struct Taken {
  std::vector<dl::Pair> pairs;
  void add(std::optional<dl::Pair> p) {
    if (p) pairs.push_back(std::move(*p));
  }
  std::vector<sensor::StereoFrame> frames() const {
    std::vector<sensor::StereoFrame> out;
    for (const dl::Pair& p : pairs) {
      sensor::StereoFrame f;
      f.timestamp_ns = p.timestamp_ns;
      f.skew_ns = p.skew_ns;
      f.sequence = p.sequence;
      out.push_back(f);
    }
    return out;
  }
};

int test_ports() {
  CHECK(dl::sub_device_of("SDI 1") == 0);
  CHECK(dl::sub_device_of("SDI 3") == 1);
  CHECK(dl::sub_device_of("SDI 2") == 2);
  CHECK(dl::sub_device_of("SDI 4") == 3);
  CHECK(dl::sub_device_of("SDI 5") == -1);
  CHECK(dl::sub_device_of("sdi 1") == -1);
  const std::optional<dl::Mode> p30 = dl::mode_of(3840, 2160, 30);
  CHECK(p30 && p30->code == 0x346B3330 && std::string(p30->name) == "2160p30");
  CHECK(dl::mode_of(3840, 2160, 24)->code == 0x346B3234);
  CHECK(dl::mode_of(3840, 2160, 25)->code == 0x346B3235);
  CHECK(!dl::mode_of(3840, 2160, 60));
  CHECK(!dl::mode_of(1920, 1080, 30));
  CHECK(dl::v210_row_bytes(3840) == 10240);
  CHECK(dl::v210_row_bytes(1920) == 5120);
  CHECK(dl::v210_row_bytes(1280) == 3456);  // 27 blocks of 48 pixels
  CHECK(dl::v210_row_bytes(96) == 256);
  CHECK(dl::period_ns(30) == kPeriod);
  CHECK(dl::period_ns(25) == 40'000'000);
  const std::string fix = dl::mode_fix("2160p30", 30, "2160p24");
  CHECK(fix.find("sends 2160p24, not 2160p30") != std::string::npos);
  CHECK(fix.find("0xd006 = 3000") != std::string::npos);
  std::puts("ports: connectors, modes, rows and the mode fix");
  return 0;
}

int test_options() {
  dl::Config ok;
  ok.left_id = "bm-1";
  ok.right_id = "bm-2";
  ok.left_input = "SDI 2";
  ok.right_input = "SDI 1";
  CHECK(dl::check_config(ok).ok());
  CHECK(ok.max_skew_ns == 16'666'000);

  dl::Config c = ok;
  c.right_input = "SDI 5";
  CHECK(says(dl::check_config(c), "no connector \"SDI 5\""));
  c = ok;
  c.right_input = "SDI 2";
  CHECK(says(dl::check_config(c), "both cameras on SDI 2"));
  c = ok;
  c.fps = 0;
  CHECK(says(dl::check_config(c), "at 0 fps"));
  c = ok;
  c.max_skew_ns = 0;
  CHECK(says(dl::check_config(c), "max_skew_ns 0 is not in 1..16666666"));
  c = ok;
  c.max_skew_ns = kHalfFrame + 1;
  CHECK(says(dl::check_config(c), "is not in 1..16666666"));
  c.max_skew_ns = kHalfFrame;
  CHECK(dl::check_config(c).ok());
  c = ok;
  c.lock_timeout = std::chrono::milliseconds(0);
  CHECK(says(dl::check_config(c), "lock_timeout"));

  // Lenses are checked only when given: a pair to be calibrated has none.
  c = ok;
  c.calibration = sensor::StereoCalibration{};
  c.calibration->left.size = {3840, 2160};
  c.calibration->right.size = {1920, 1080};
  CHECK(says(dl::check_config(c), "a lens calibrated at 1920x1080"));
  c.calibration->right.size = {3840, 2160};
  CHECK(dl::check_config(c).ok());
  std::puts("options: each refusal named; lenses checked only when given");
  return 0;
}

int test_pairing() {
  Holds holds;
  // Free-running cameras: the right 8.2 ms ahead of the left (it is SDI 1,
  // the left SDI 2), arrivals in time order.
  {
    dl::Pairing pairing(kPeriod, 16'666'000);
    Taken taken;
    for (int k = 0; k < 20; ++k) {
      const std::int64_t t =
          1'000 * kMs + k * static_cast<std::int64_t>(kPeriod);
      taken.add(pairing.arrive(frame(1, t - 8'200'000, holds)));
      taken.add(pairing.arrive(frame(0, t, holds)));
    }
    CHECK(taken.pairs.size() == 20);
    for (std::size_t i = 0; i < taken.pairs.size(); ++i) {
      const dl::Pair& p = taken.pairs[i];
      CHECK(p.skew_ns == -8'200'000);
      CHECK(p.sequence == i);
      CHECK(p.left.input == 0 && p.right.input == 1);
      // The left frame's time, on the host's clock.
      CHECK(p.timestamp_ns ==
            static_cast<std::uint64_t>(dl::host_time_ns(p.left)));
    }
    CHECK(pairing.received() == 20 && pairing.failed() == 0);
  }
  // The two inputs' callbacks race when the cameras are near in phase:
  // either order pairs.
  {
    dl::Pairing pairing(kPeriod, 500'000);
    Taken taken;
    Jitter jitter;
    for (int k = 0; k < 200; ++k) {
      const std::int64_t t = k * static_cast<std::int64_t>(kPeriod);
      const std::int64_t r = t + jitter(200'000);
      if (k % 2 == 0) {
        taken.add(pairing.arrive(frame(0, t, holds)));
        taken.add(pairing.arrive(frame(1, r, holds)));
      } else {
        taken.add(pairing.arrive(frame(1, r, holds)));
        taken.add(pairing.arrive(frame(0, t, holds)));
      }
    }
    CHECK(taken.pairs.size() == 200 && pairing.failed() == 0);
  }
  // Losses: a left frame lost (a sequence gap), a right one lost (that left
  // frame fails), one input without signal and back; and the contract.
  {
    dl::Pairing pairing(kPeriod, 16'666'000);
    Taken taken;
    for (int k = 0; k < 30; ++k) {
      const std::int64_t t =
          1'000 * kMs + k * static_cast<std::int64_t>(kPeriod);
      const bool right_signal = k < 20 || k >= 24;
      if (k != 12)
        taken.add(pairing.arrive(frame(1, t + 3 * kMs, holds, !right_signal)));
      if (k != 7) taken.add(pairing.arrive(frame(0, t, holds)));
    }
    // 30 left frames, 1 lost; 29 received; of those, frame 12 (no right) and
    // 20..23 (no signal on the right) failed.
    CHECK(pairing.received() == 29);
    CHECK(pairing.failed() == 5);
    CHECK(taken.pairs.size() == 24);
    std::vector<std::uint64_t> seqs;
    for (const dl::Pair& p : taken.pairs) seqs.push_back(p.sequence);
    CHECK(seqs[6] == 6 && seqs[7] == 8);  // the lost left frame is a gap
    CHECK(seqs.back() == 29);
    sensor::StereoInfo info;
    info.max_skew_ns = 16'666'000;
    sensor::SensorStats stats;
    stats.received = pairing.received();
    stats.failed = pairing.failed();
    stats.delivered = taken.pairs.size();
    const std::string breach = stereo_contract::breach(
        info, stereo_contract::Opened::Uncalibrated, taken.frames(), stats);
    if (!breach.empty()) std::fprintf(stderr, "  %s\n", breach.c_str());
    CHECK(breach.empty());
    CHECK(stats.delivered + stats.failed == stats.received);
  }
  // No signal on the left: each such frame is received and failed, and a
  // left frame held from before the signal went fails too.
  {
    dl::Pairing pairing(kPeriod, 16'666'000);
    CHECK(!pairing.arrive(frame(0, 0, holds)));
    CHECK(!pairing.arrive(frame(0, kPeriod, holds, true)));
    CHECK(pairing.received() == 2 && pairing.failed() == 2);
    CHECK(!pairing.arrive(frame(1, 2 * kPeriod + kMs, holds)));
    CHECK(pairing.arrive(frame(0, 2 * kPeriod, holds)).has_value());
  }
  // A genlock-sized bound: 8 ms apart pairs nothing, then 0.2 ms pairs
  // again.
  {
    dl::Pairing pairing(kPeriod, 500'000);
    Taken taken;
    for (int k = 0; k < 20; ++k) {
      const std::int64_t t = k * static_cast<std::int64_t>(kPeriod);
      const std::int64_t apart = k < 10 ? 8 * kMs : 200'000;
      taken.add(pairing.arrive(frame(0, t, holds)));
      taken.add(pairing.arrive(frame(1, t + apart, holds)));
    }
    CHECK(taken.pairs.size() == 10);
    CHECK(taken.pairs.front().sequence == 10);
    CHECK(pairing.failed() == 10);
    CHECK(pairing.offset_ns() && *pairing.offset_ns() == 200'000);
  }
  // The newest unpaired frame wins: two left frames before any right one,
  // the first fails and the second pairs; a right frame replaced is no
  // left-frame failure.
  {
    dl::Pairing pairing(kPeriod, 16'666'000);
    CHECK(!pairing.arrive(frame(0, 0, holds)));
    CHECK(!pairing.arrive(frame(0, kPeriod, holds)));
    std::optional<dl::Pair> p = pairing.arrive(frame(1, kPeriod + kMs, holds));
    CHECK(p && p->left.card_ns == static_cast<std::int64_t>(kPeriod));
    CHECK(pairing.failed() == 1);
    CHECK(!pairing.arrive(frame(1, 3 * kPeriod, holds)));
    CHECK(!pairing.arrive(frame(1, 4 * kPeriod, holds)));
    p = pairing.arrive(frame(0, 4 * kPeriod - kMs, holds));
    CHECK(p && p->right.card_ns == 4 * static_cast<std::int64_t>(kPeriod));
    CHECK(pairing.failed() == 1);
    // A restart: the held frames go back, the counts and sequence restart;
    // a stop's release keeps the counts.
    CHECK(!pairing.arrive(frame(0, 10 * kPeriod, holds)));
    const int back = *holds.back;
    pairing.release();
    CHECK(*holds.back == back + 1);
    CHECK(pairing.received() == 4);
    pairing.reset();
    CHECK(pairing.received() == 0 && pairing.failed() == 0);
    CHECK(!pairing.seen(0) && !pairing.offset_ns());
    CHECK(!pairing.arrive(frame(0, 20 * kPeriod, holds)));
    p = pairing.arrive(frame(1, 20 * kPeriod + kMs, holds));
    CHECK(p && p->sequence == 0);
  }
  std::puts(
      "pairing: free run, either order, losses, no signal, the bound, "
      "newest wins, restart, the contract");
  return 0;
}

// The card clock runs 18.78 ppm slow against the host's (measured, the
// host's raw clock against its NTP-disciplined one), the cameras 27 ppm
// slow against nominal. Over an hour each frame's host time is the true one
// within 10 us, and the left frames' sequence has no false gap.
int test_clock_map() {
  constexpr double kCardSlow = 18.78e-6;
  const double camera_period = static_cast<double>(kPeriod) * (1.0 + 27e-6);
  const std::int64_t host0 = 1'793'500'000'000'000'000;  // 2026-11
  const std::int64_t card0 = 6'000'000'000'000;  // about 100 min of uptime
  const auto card_of = [&](double host) {
    return static_cast<std::int64_t>(
        std::llround(static_cast<double>(card0) +
                     (host - static_cast<double>(host0)) * (1.0 - kCardSlow)));
  };
  const auto host_of = [&](std::int64_t card) {
    return static_cast<double>(host0) +
           static_cast<double>(card - card0) / (1.0 - kCardSlow);
  };
  Holds holds;
  dl::Pairing pairing(kPeriod, 16'666'000);
  Jitter jitter;
  const int frames = 30 * 3600;
  std::uint64_t pairs = 0;
  std::uint64_t last_sequence = 0;
  double worst = 0.0;
  for (int k = 0; k < frames; ++k) {
    const double base = static_cast<double>(card0) + k * camera_period;
    for (int input = 1; input >= 0; --input) {
      dl::Arrival a;
      a.input = input;
      a.card_ns = static_cast<std::int64_t>(base) -
                  (input == 1 ? 8'200'000 : 0) + jitter(40'000);
      // The callback 3.6 ms later, both clocks read in it, 1 us apart.
      const double callback = host_of(a.card_ns) + 3.6e6 + jitter(200'000);
      a.host_now_ns = static_cast<std::int64_t>(callback);
      a.card_now_ns = card_of(callback + 1'000.0);
      a.hold = holds.make();
      const std::int64_t card = a.card_ns;
      const std::optional<dl::Pair> p = pairing.arrive(std::move(a));
      if (input == 0) {
        CHECK(p.has_value());
        const double error =
            std::fabs(static_cast<double>(p->timestamp_ns) - host_of(card));
        if (error > worst) worst = error;
        CHECK(pairs == 0 || p->sequence == last_sequence + 1);
        last_sequence = p->sequence;
        ++pairs;
      }
    }
  }
  std::printf(
      "  an hour: %llu pairs, host time within %.2f us, sequence "
      "0..%llu\n",
      static_cast<unsigned long long>(pairs), worst / 1e3,
      static_cast<unsigned long long>(last_sequence));
  CHECK(pairs == static_cast<std::uint64_t>(frames));
  CHECK(worst < 10'000.0);
  CHECK(last_sequence == static_cast<std::uint64_t>(frames) - 1);
  std::puts(
      "clock map: 18.8 ppm tracked within 10 us over an hour; 27 ppm "
      "gives no false gap");
  return 0;
}

// A measured free run: SDI 1 (the right camera) at
// 33.334257 ms a frame, SDI 2 (the left) 8.20 ms behind it, the gap falling
// 0.0367 ms a minute, 32 us of jitter, over 8.22 minutes.
int test_skew_stats() {
  const double right_period = 33'334'257.0;
  const double per_frame = -0.0367e6 / 60e9 * right_period;  // ns a frame
  const double left_period = right_period + per_frame;
  const int frames = static_cast<int>(8.22 * 60 * 30);
  Jitter jitter;
  std::vector<std::int64_t> left;
  std::vector<std::int64_t> right;
  const double t0 = 6e12;
  for (int k = 0; k < frames; ++k) {
    right.push_back(static_cast<std::int64_t>(t0 + k * right_period) +
                    jitter(55'000));
    left.push_back(static_cast<std::int64_t>(t0 + 8.2e6 + k * left_period) +
                   jitter(55'000));
  }
  dl::SkewReport r = dl::skew_stats(left, right, kPeriod, 16'666'000);
  std::printf(
      "  skew at start %.4f ms, drift %.5f ms/min, mean %.4f, std "
      "%.4f, p99.9 %.4f; slow %.2f / %.2f ppm\n",
      r.skew_at_start_ms, r.drift_ms_per_min, r.skew_mean_ms, r.skew_std_ms,
      r.skew_p999_ms, r.slow_ppm[0], r.slow_ppm[1]);
  // Right less left: -8.20 ms, rising 0.0367 ms a minute -- SDI 2 lagging
  // SDI 1 by 8.20 ms, falling 0.0367 ms a minute.
  CHECK(std::fabs(r.skew_at_start_ms + 8.20) < 0.005);
  CHECK(std::fabs(r.drift_ms_per_min - 0.0367) < 0.001);
  CHECK(r.pairs == static_cast<std::uint64_t>(frames) && r.refused == 0);
  CHECK(r.lost[0] == 0 && r.lost[1] == 0);
  CHECK(std::fabs(r.slow_ppm[1] - 27.7) < 0.5);
  CHECK(r.skew_std_ms > 0.03 && r.skew_std_ms < 0.12);
  CHECK(r.skew_p999_ms > 8.1 && r.skew_p999_ms < 8.4);

  // Two right frames lost; a bound under the cameras' gap refuses them all.
  right.erase(right.begin() + 1000, right.begin() + 1002);
  r = dl::skew_stats(left, right, kPeriod, 16'666'000);
  CHECK(r.lost[1] == 2 && r.lost[0] == 0);
  CHECK(r.frames[1] == static_cast<std::uint64_t>(frames) - 2);
  r = dl::skew_stats(left, right, kPeriod, 500'000);
  CHECK(r.pairs == 0 && r.refused == static_cast<std::uint64_t>(frames));
  std::puts(
      "skew report: today's 8.20 ms and -0.0367 ms/min, losses, refusals");
  return 0;
}

}  // namespace

int main() {
  if (test_ports() != 0 || test_options() != 0 || test_pairing() != 0 ||
      test_clock_map() != 0 || test_skew_stats() != 0) {
    return 1;
  }
  std::puts("sensor_decklink_pairing: OK");
  return 0;
}
