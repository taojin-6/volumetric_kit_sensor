// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A microphone checked by hand: lists the inputs, or records from one for a
// few seconds and prints what came: chunks, frames, losses, the step between
// chunk stamps against the nominal chunk length, the last chunk's age against
// system_clock, and the loudest sample. It saves no audio.
//
//   mic_check --list
//   mic_check [--device <name or part>] [--seconds <n>]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "volumetric_kit/sensor/audio/microphone.hpp"

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: mic_check --list\n"
               "       mic_check [--device <name or part>] [--seconds <n>]\n");
  return 2;
}

int fail(const vkc::Status& s) {
  std::fprintf(stderr, "mic_check: %s\n", s.message().c_str());
  return 1;
}

std::int64_t system_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int list() {
  auto inputs = sensor::Microphone::inputs();
  if (!inputs.ok()) return fail(inputs.status());
  for (const sensor::Microphone::Input& in : inputs.value()) {
    std::printf("%c %s (%s, %u ch, %.0f Hz)\n", in.is_default ? '*' : ' ',
                in.name.c_str(), in.host_api.c_str(),
                static_cast<unsigned>(in.channels), in.default_sample_rate);
  }
  std::printf("* the default input\n");
  return 0;
}

int record(const std::string& device, int seconds) {
  sensor::Microphone::Options options;
  options.device = device;
  auto opened = sensor::Microphone::open(options);
  if (!opened.ok()) return fail(opened.status());
  sensor::Microphone mic = std::move(opened).value();
  const sensor::AudioInfo& info = mic.info();
  std::printf("recording %d s from %s: %u Hz, %u ch, %u frames a chunk\n",
              seconds, info.model.c_str(), info.sample_rate,
              static_cast<unsigned>(info.channels), info.chunk_frames);

  if (const vkc::Status s = mic.start(); !s) return fail(s);
  const auto end =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::vector<sensor::AudioChunk> chunks;
  std::uint64_t count = 0;
  std::uint64_t frames = 0;
  std::uint64_t last_pts = 0;
  double step_sum_ms = 0.0;
  double step_min_ms = std::numeric_limits<double>::infinity();
  double step_max_ms = 0.0;
  std::uint64_t steps = 0;
  int peak = 0;
  std::int64_t last_age_ns = 0;
  while (std::chrono::steady_clock::now() < end) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    chunks.clear();
    if (const vkc::Status s = mic.drain(&chunks); !s) return fail(s);
    if (chunks.empty()) continue;
    last_age_ns =
        system_now_ns() - static_cast<std::int64_t>(chunks.back().timestamp_ns);
    for (const sensor::AudioChunk& c : chunks) {
      if (count > 0) {
        const double step_ms = (c.timestamp_ns - last_pts) / 1e6;
        step_sum_ms += step_ms;
        step_min_ms = std::min(step_min_ms, step_ms);
        step_max_ms = std::max(step_max_ms, step_ms);
        ++steps;
      }
      last_pts = c.timestamp_ns;
      ++count;
      frames += c.samples.size() / info.channels;
      for (const std::int16_t v : c.samples) {
        peak = std::max(peak, std::abs(static_cast<int>(v)));
      }
    }
  }
  const sensor::SensorStats stats = mic.stats();
  const std::uint64_t overflows = mic.overflows();
  mic.stop();

  const double nominal_ms = 1e3 * info.chunk_frames / info.sample_rate;
  std::printf("chunks %llu, frames %llu (%.2f s)\n",
              static_cast<unsigned long long>(count),
              static_cast<unsigned long long>(frames),
              static_cast<double>(frames) / info.sample_rate);
  std::printf(
      "received %llu, delivered %llu, dropped %llu, failed %llu, "
      "overflows %llu\n",
      static_cast<unsigned long long>(stats.received),
      static_cast<unsigned long long>(stats.delivered),
      static_cast<unsigned long long>(stats.dropped),
      static_cast<unsigned long long>(stats.failed),
      static_cast<unsigned long long>(overflows));
  if (steps > 0) {
    std::printf("pts step: mean %.3f ms, min %.3f, max %.3f (nominal %.3f)\n",
                step_sum_ms / steps, step_min_ms, step_max_ms, nominal_ms);
  }
  if (count > 0) {
    std::printf("last chunk's age at its drain: %.3f ms\n", last_age_ns / 1e6);
  }
  std::printf("peak |sample| %d of 32767%s\n", peak,
              peak == 0 ? ": silence, so check the microphone permission" : "");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool list_inputs = false;
  std::string device;
  int seconds = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--list") {
      list_inputs = true;
    } else if (arg == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else if (arg == "--seconds" && i + 1 < argc) {
      seconds = std::atoi(argv[++i]);
      if (seconds <= 0) return usage();
    } else {
      return usage();
    }
  }
  std::printf(
      "note: macOS asks for microphone access the first time a process "
      "opens an input (allow the terminal); without it the input is "
      "silent.\n");
  return list_inputs ? list() : record(device, seconds);
}
