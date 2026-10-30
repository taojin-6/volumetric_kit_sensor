// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "capture.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor::audio {

std::uint64_t chunk_pts_ns(std::int64_t now_ns, double current_time,
                           double adc_time,
                           std::uint64_t last_pts_ns) noexcept {
  double wait_s = current_time - adc_time;
  if (!(wait_s >= 0.0 && wait_s < 1.0)) wait_s = 0.0;  // NaN fails both
  const std::int64_t pts =
      now_ns - static_cast<std::int64_t>(std::llround(wait_s * 1e9));
  const std::uint64_t mapped = pts > 0 ? static_cast<std::uint64_t>(pts) : 0;
  return std::max(mapped, last_pts_ns);
}

core::Status check_format(std::uint32_t sample_rate, std::uint16_t channels,
                          std::uint32_t chunk_frames,
                          std::size_t queue_chunks) {
  if (sample_rate == 0) {
    return core::Status::invalid_argument("a sample rate of 0");
  }
  if (channels != 1 && channels != 2) {
    return core::Status::invalid_argument(std::to_string(channels) +
                                          " channels: 1 or 2 are taken");
  }
  if (chunk_frames == 0 || chunk_frames > sample_rate / 2) {
    return core::Status::invalid_argument(
        "a chunk of " + std::to_string(chunk_frames) +
        " frames: 1 to half a second's, " + std::to_string(sample_rate / 2) +
        ", are taken");
  }
  if (queue_chunks == 0) {
    return core::Status::invalid_argument("a queue of 0 chunks");
  }
  return {};
}

Capture::Capture(std::uint16_t channels, std::uint32_t chunk_frames,
                 std::size_t queue_chunks)
    : channels_(channels),
      ring_(queue_chunks, std::size_t{chunk_frames} * channels) {}

bool Capture::on_input(const std::int16_t* samples, std::size_t frames,
                       std::int64_t now_ns, double current_time,
                       double adc_time, bool overflow) noexcept {
  received_.fetch_add(1, std::memory_order_relaxed);
  if (overflow) overflows_.fetch_add(1, std::memory_order_relaxed);
  const std::size_t count = frames * channels_;
  if (frames == 0 || count > ring_.entry_samples()) {
    failed_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  last_pts_ns_ = chunk_pts_ns(now_ns, current_time, adc_time, last_pts_ns_);
  if (!ring_.push(last_pts_ns_, samples, count)) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

void Capture::start(Clock::time_point now) noexcept {
  if (running_) return;
  ring_.clear();
  received_.store(0, std::memory_order_relaxed);
  dropped_.store(0, std::memory_order_relaxed);
  failed_.store(0, std::memory_order_relaxed);
  overflows_.store(0, std::memory_order_relaxed);
  delivered_ = 0;
  seen_received_ = 0;
  last_arrival_ = now;
  running_ = true;
}

void Capture::stop() noexcept {
  if (!running_) return;
  running_ = false;
  ring_.clear();
}

core::Status Capture::drain(std::vector<AudioChunk>* out,
                            Clock::time_point now) {
  VKC_CHECK(out != nullptr, "Capture::drain needs somewhere to put them");
  if (!running_) return {};
  ring_.pop_all([&](std::uint64_t pts_ns, const std::int16_t* samples,
                    std::size_t count) {
    AudioChunk chunk;
    chunk.timestamp_ns = pts_ns;
    chunk.sequence = delivered_++;
    chunk.samples.assign(samples, samples + count);
    out->push_back(std::move(chunk));
  });
  const std::uint64_t received = received_.load(std::memory_order_relaxed);
  if (received != seen_received_) {
    seen_received_ = received;
    last_arrival_ = now;
  } else if (now - last_arrival_ >= kSilenceLimit) {
    return core::Status::io_error(
        "no audio for 1 s or more: the input has gone");
  }
  return {};
}

SensorStats Capture::stats() const noexcept {
  SensorStats stats;
  stats.received = received_.load(std::memory_order_relaxed);
  stats.delivered = delivered_;
  stats.dropped = dropped_.load(std::memory_order_relaxed);
  stats.failed = failed_.load(std::memory_order_relaxed);
  return stats;
}

}  // namespace volumetric_kit::sensor::audio
