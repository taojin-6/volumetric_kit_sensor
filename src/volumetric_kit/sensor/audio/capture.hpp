// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): what Microphone does with each chunk, apart from
// PortAudio. The clocks come in as arguments, so a test drives it with no
// device.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "chunk_ring.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/audio_sensor.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor::audio {

// How long a running source may deliver nothing before drain says it has gone.
inline constexpr std::chrono::seconds kSilenceLimit{1};

// The host time of a chunk's first sample, in ns on system_clock: `now_ns`
// (system_clock, read as the callback began) less how long that sample waited,
// `current_time - adc_time` (the backend's clock, in seconds; only the
// difference is used). A wait that is not finite, negative or 1 s or more
// counts as 0: a backend that reports no ADC time gives 0, which would put the
// chunk at the backend clock's origin. Never below `last_pts_ns`, so the
// stamps never go back, a step back of the host clock included.
std::uint64_t chunk_pts_ns(std::int64_t now_ns, double current_time,
                           double adc_time, std::uint64_t last_pts_ns) noexcept;

// InvalidArgument unless the rate is above 0, the channels 1 or 2, a chunk 1
// frame to half a second (so kSilenceLimit never passes between two chunks),
// and the queue at least 1 chunk.
core::Status check_format(std::uint32_t sample_rate, std::uint16_t channels,
                          std::uint32_t chunk_frames, std::size_t queue_chunks);

class Capture {
 public:
  using Clock = std::chrono::steady_clock;

  // `queue_chunks` chunks of up to `chunk_frames` frames of `channels`
  // samples, all allocated here (check_format's ranges).
  Capture(std::uint16_t channels, std::uint32_t chunk_frames,
          std::size_t queue_chunks);

  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;

  // The device's thread (the callback seam): one chunk of `frames` frames,
  // interleaved (zeros where `samples` is null), stamped by chunk_pts_ns;
  // `overflow` when the backend lost input before it. Counted received;
  // dropped when the queue is full (ChunkRing::push); failed for no frames or
  // more than a chunk holds. True when queued. Locks, allocates and logs
  // nothing.
  bool on_input(const std::int16_t* samples, std::size_t frames,
                std::int64_t now_ns, double current_time, double adc_time,
                bool overflow) noexcept;

  // The caller's thread, while the device delivers nothing (before its stream
  // starts): the chunks held let go, the counters restarted, the silence
  // measured from `now`. Nothing on a running capture.
  void start(Clock::time_point now) noexcept;
  // The caller's thread, once the device delivers no more (its stream
  // stopped): the chunks held let go; the counters kept.
  void stop() noexcept;
  bool running() const noexcept { return running_; }

  // The caller's thread: every chunk held appended to `out`, oldest first,
  // each given the next sequence number. OK, with nothing appended, when not
  // running; IoError when nothing has been received for kSilenceLimit or more
  // up to `now` (from the start, or the first drain that saw the last chunk).
  core::Status drain(std::vector<AudioChunk>* out, Clock::time_point now);

  // The caller's thread: as IAudioSensor::stats.
  SensorStats stats() const noexcept;
  // Chunks the backend lost input before (paInputOverflow), since start.
  std::uint64_t overflows() const noexcept {
    return overflows_.load(std::memory_order_relaxed);
  }

 private:
  const std::uint16_t channels_;
  ChunkRing ring_;

  // The device's thread alone: the last chunk's stamp, kept across restarts.
  std::uint64_t last_pts_ns_ = 0;

  // Written on the device's thread (and zeroed by start, before it runs).
  std::atomic<std::uint64_t> received_{0};
  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<std::uint64_t> failed_{0};
  std::atomic<std::uint64_t> overflows_{0};

  // The caller's thread alone.
  bool running_ = false;
  std::uint64_t delivered_ = 0;
  std::uint64_t seen_received_ = 0;
  Clock::time_point last_arrival_{};
};

}  // namespace volumetric_kit::sensor::audio
