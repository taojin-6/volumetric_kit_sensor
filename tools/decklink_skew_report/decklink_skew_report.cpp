// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A DeckLink stereo pair's timing, by hand: captures both inputs through the
// driver's own capture unit (the SDK's buffers; no pixel is read) for a few
// seconds and prints each input's frames, losses and rate against the card
// clock, and the pairs' skew, right less left: its mean, spread, 99.9th
// percentile, its value at the start and its drift a minute, and the pairs
// past the bound. It opens the cameras, so no test runs it.
//
//   decklink_skew_report --left "SDI 2" --right "SDI 1" [--seconds <n>]
//                        [--fps 24|25|30] [--max-skew-us <n>]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "capture.hpp"
#include "ports.hpp"
#include "skew_stats.hpp"
#include "volumetric_kit/core/base/log.hpp"

namespace dl = volumetric_kit::sensor::decklink;
namespace vkc = volumetric_kit::core;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: decklink_skew_report --left \"SDI n\" --right \"SDI n\" "
               "[--seconds <n>] [--fps 24|25|30] [--max-skew-us <n>]\n");
  return 2;
}

int fail(const vkc::Status& s) {
  std::fprintf(stderr, "decklink_skew_report: %s\n", s.message().c_str());
  return 1;
}

// Each input's frame times, and what failed, from the SDK's threads.
class Log final : public dl::CaptureSink {
 public:
  void arrived(dl::Arrival frame) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    const int i = frame.input == 0 ? 0 : 1;
    if (frame.no_signal) {
      ++no_signal_[i];
    } else {
      times_[i].push_back(frame.card_ns);
    }
  }
  void mode_changed(int input, std::string detected) override {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (changed_.empty()) {
      changed_ = (input == 0 ? "the left" : "the right") +
                 std::string(" input: the camera sends ") + detected;
    }
  }
  std::vector<std::int64_t> times(int i) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return times_[i];
  }
  std::uint64_t no_signal(int i) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return no_signal_[i];
  }
  std::string changed() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return changed_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::int64_t> times_[2];
  std::uint64_t no_signal_[2] = {};
  std::string changed_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string inputs[2];
  int seconds = 60;
  std::uint32_t fps = 30;
  std::uint64_t max_skew_us = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i + 1 >= argc) return usage();
    const char* value = argv[++i];
    if (arg == "--left") {
      inputs[0] = value;
    } else if (arg == "--right") {
      inputs[1] = value;
    } else if (arg == "--seconds") {
      seconds = std::atoi(value);
    } else if (arg == "--fps") {
      fps = static_cast<std::uint32_t>(std::atoi(value));
    } else if (arg == "--max-skew-us") {
      max_skew_us = std::strtoull(value, nullptr, 10);
    } else {
      return usage();
    }
  }
  const std::optional<dl::Mode> mode = dl::mode_of(3840, 2160, fps);
  if (inputs[0].empty() || inputs[1].empty() || seconds <= 0 || !mode) {
    return usage();
  }
  const std::uint64_t period = dl::period_ns(fps);
  const std::uint64_t max_skew =
      max_skew_us != 0 ? max_skew_us * 1000 : 1'000'000'000ull / (2ull * fps);

  dl::CaptureConfig config;
  for (int i = 0; i < 2; ++i) {
    config.names[i] = inputs[i] + (i == 0 ? " (left)" : " (right)");
    config.sub_device[i] = dl::sub_device_of(inputs[i]);
    if (config.sub_device[i] < 0) return usage();
  }
  config.mode = mode->code;
  // The capture's log on stderr, its Info too: which SDK device each
  // connector opened.
  vkc::set_log_handler(
      [](vkc::LogLevel level, std::string_view, std::string_view message) {
        if (level == vkc::LogLevel::Debug) return;
        std::fprintf(stderr, "%.*s\n", static_cast<int>(message.size()),
                     message.data());
      });
  vkc::Result<std::unique_ptr<dl::Capture>> capture = dl::open_capture(config);
  if (!capture.ok()) return fail(capture.status());
  Log log;
  if (const vkc::Status s =
          (*capture)->start(log, std::chrono::milliseconds(5000));
      !s.ok()) {
    return fail(s);
  }
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  (*capture)->stop();
  if (!log.changed().empty()) {
    std::fprintf(stderr, "decklink_skew_report: %s, not %s\n",
                 log.changed().c_str(), mode->name);
    return 1;
  }

  const std::vector<std::int64_t> left = log.times(0);
  const std::vector<std::int64_t> right = log.times(1);
  const dl::SkewReport r = dl::skew_stats(left, right, period, max_skew);
  std::printf("%s and %s, %s, %d s\n", config.names[0].c_str(),
              config.names[1].c_str(), mode->name, seconds);
  for (int i = 0; i < 2; ++i) {
    std::printf(
        "  %-12s %llu frames, %llu lost, %llu without signal; %+.2f ppm slow "
        "against the card clock\n",
        config.names[i].c_str(), static_cast<unsigned long long>(r.frames[i]),
        static_cast<unsigned long long>(r.lost[i]),
        static_cast<unsigned long long>(log.no_signal(i)), r.slow_ppm[i]);
  }
  std::printf("  pairs: %llu within %llu us, %llu refused\n",
              static_cast<unsigned long long>(r.pairs),
              static_cast<unsigned long long>(max_skew / 1000),
              static_cast<unsigned long long>(r.refused));
  std::printf(
      "  skew, right less left: mean %+.4f ms, std %.4f ms, p99.9 of its size "
      "%.4f ms; %+.4f ms at the start, drifting %+.5f ms a minute\n",
      r.skew_mean_ms, r.skew_std_ms, r.skew_p999_ms, r.skew_at_start_ms,
      r.drift_ms_per_min);
  return 0;
}
