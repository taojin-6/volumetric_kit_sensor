// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file audio_sensor.hpp
/// @brief Any audio source -- a USB or built-in mic, a camera's mic, a capture
///        card's embedded audio -- as every driver presents it.
///
/// A driver implements @ref IAudioSensor and hands out @ref AudioChunk s as
/// captured: signed 16-bit PCM in host memory, each chunk stamped with its
/// first sample's capture time. Audio is host I/O: nothing here reaches the
/// GPU. A camera with a mic hands out an @ref IAudioSensor of its own beside
/// its @ref IRgbdSensor, which does not change.

#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/export.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief What an audio source is, fixed when it opens.
struct AudioInfo {
  /// A stable device id: the OS's name for an input, or a camera's serial
  /// when the mic is that camera's.
  std::string id;
  std::string model;  ///< What it is, e.g. "Wireless GO II RX".
  /// The clock its chunks' timestamps are on.
  ClockDomain clock = ClockDomain::Host;
  std::uint32_t sample_rate = 0;  ///< Sample frames a second, e.g. 48000.
  std::uint16_t channels = 0;     ///< Samples a frame, interleaved; at least 1.
  /// Frames a chunk, nominally (a stream's record rate); a source may hand
  /// out chunks of other lengths.
  std::uint32_t chunk_frames = 0;
};

/// @brief One chunk as captured: signed 16-bit samples, interleaved by
///        channel, `samples.size() / AudioInfo::channels` frames.
struct AudioChunk {
  /// The capture time of its first sample, in nanoseconds on
  /// `AudioInfo::clock` (on `ClockDomain::Host`: `std::chrono::system_clock`
  /// since its epoch, as `RgbdFrame::timestamp_ns`). Never below the chunk
  /// before's.
  std::uint64_t timestamp_ns = 0;
  /// Chunks handed out before it since the last start: one more than the
  /// chunk before's. A chunk lost is placed by its neighbours' timestamps,
  /// not by this.
  std::uint64_t sequence = 0;
  std::vector<std::int16_t> samples;  ///< Interleaved by channel.
};

/// @brief An audio source, drained of every chunk it captured.
///
/// **Polled, not called back**, as @ref IRgbdSensor is: a source that
/// captures on a thread of its own holds what arrives, and @ref drain hands
/// it over on the caller's thread, oldest first. A source that holds as many
/// chunks as it can lets further ones go, counted in `SensorStats::dropped`;
/// which ones is the implementation's to say.
///
/// @code
/// VKC_TRY(mic.start());
/// std::vector<AudioChunk> chunks;
/// for (;;) {  // the caller's loop, e.g. once a camera set
///   chunks.clear();
///   VKC_TRY(mic.drain(&chunks));  // fails once the device has gone
///   for (const AudioChunk& c : chunks) write(c.timestamp_ns, c.samples);
/// }
/// @endcode
///
/// @warning Not thread-safe unless an implementation says otherwise: open,
///          start, drain and stop from one thread.
class VS_INTERFACE_API IAudioSensor {
 public:
  virtual ~IAudioSensor() = default;

  IAudioSensor(const IAudioSensor&) = delete;
  IAudioSensor& operator=(const IAudioSensor&) = delete;

  /// @return What the source is, from when it opened.
  virtual const AudioInfo& info() const noexcept = 0;

  /// @brief Begin capturing, the counters restarted. Idempotent: starting a
  ///        running source is OK.
  /// @return OK once running, or why the source could not start.
  virtual core::Status start() = 0;

  /// @brief Stop capturing and drop the chunks held. Idempotent, and safe on
  ///        a source that never started: the destructor's fallback.
  virtual void stop() noexcept = 0;

  /// @brief Take every chunk held, oldest first.
  /// @param out  Receives the chunks, appended; not null.
  /// @return OK, with none appended when none is held or the source is not
  ///         running; an error once started and nothing has arrived for 1 s
  ///         or more: the device has gone.
  virtual core::Status drain(std::vector<AudioChunk>* out) = 0;

  /// @return The counters since the last @ref start, in chunks:
  ///         `SensorStats::received` captured, `delivered` handed out by
  ///         @ref drain, `dropped` let go unseen, `failed` refused.
  virtual SensorStats stats() const noexcept = 0;

 protected:
  IAudioSensor() = default;
  // Move operations stay available to implementations but are not part of
  // the polymorphic interface: callers hold a source behind a pointer, as an
  // IRgbdSensor.
  IAudioSensor(IAudioSensor&&) = default;
  IAudioSensor& operator=(IAudioSensor&&) = default;
};

}  // namespace volumetric_kit::sensor
