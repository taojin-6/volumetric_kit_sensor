// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file audio/microphone.hpp
/// @brief An OS audio input -- any USB or built-in mic -- as an
///        @ref volumetric_kit::sensor::IAudioSensor, through PortAudio: Core
///        Audio on macOS, ALSA on Linux.
///
/// No PortAudio type appears here -- the library is held behind a pointer to
/// an implementation -- so a consumer includes this header without
/// PortAudio's. Linking still needs its library, which the target carries.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/audio/microphone_export.hpp"
#include "volumetric_kit/sensor/audio_sensor.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief One OS audio input, drained of its chunks, each stamped on the
///        host's clock.
///
/// The input is opened as signed 16-bit PCM at the rate and channel count
/// asked for, in chunks of @ref Options::chunk_frames frames. Each chunk's
/// `AudioChunk::timestamp_ns` (`ClockDomain::Host`) is
/// `std::chrono::system_clock` as PortAudio hands the chunk over, less how
/// long its first sample waited since the converter captured it.
///
/// Up to @ref Options::queue_chunks chunks are held for @ref drain; a full
/// queue lets the chunk arriving go, keeping those it holds, counted in
/// `SensorStats::dropped`. Input the backend lost before a chunk
/// (`paInputOverflow`) is counted in @ref overflows; the chunk is kept, and
/// its stamp places it after the gap.
///
/// @code
/// VKC_ASSIGN(Microphone mic, Microphone::open({}));  // the default input
/// VKC_TRY(mic.start());
/// std::vector<AudioChunk> chunks;
/// VKC_TRY(mic.drain(&chunks));  // every chunk since, oldest first
/// @endcode
///
/// macOS asks the user for microphone access the first time a process opens
/// an input; until it is granted, the input delivers silence or nothing.
///
/// @warning Not thread-safe: open, start, drain and stop from one thread, and
///          list @ref inputs from that thread too (PortAudio's set-up is not
///          thread-safe). PortAudio delivers chunks on a thread of its own,
///          which this class hands over internally.
class VS_MICROPHONE_API Microphone final : public IAudioSensor {
 public:
  /// @brief Which input, and the format it is opened in.
  struct Options {
    /// The input's name as @ref inputs lists it, or a part of a name that
    /// names one input (e.g. `hw:1,0` on Linux); empty: the default input.
    std::string device;
    std::uint32_t sample_rate = 48000;  ///< Frames a second; above 0.
    std::uint16_t channels = 1;         ///< 1 or 2.
    /// Frames a chunk: 20 ms at 48 kHz. 1 to half a second's.
    std::uint32_t chunk_frames = 960;
    /// Chunks held for @ref drain: about 10 s at the defaults, to ride out a
    /// caller's stall. At least 1.
    std::size_t queue_chunks = 512;
  };

  /// @brief An audio input the OS offers.
  struct Input {
    std::string name;            ///< What @ref Options::device names it by.
    std::string host_api;        ///< PortAudio's backend, e.g. "Core Audio".
    std::uint16_t channels = 0;  ///< Its most input channels.
    double default_sample_rate = 0;  ///< Its default rate.
    bool is_default = false;         ///< The default input.
  };

  /// @return Every input the OS offers; `Status::Code::IoError` if PortAudio
  ///         cannot start.
  static core::Result<std::vector<Input>> inputs();

  /// @brief Find the input and open it in the format asked for. Does not
  ///        start capturing.
  /// @param options  The input and its format.
  /// @return The source, not yet started; or:
  ///         - `Status::Code::InvalidArgument` for options refused before
  ///           PortAudio starts (the ranges at @ref Options), or a
  ///           @ref Options::device that names several inputs, naming them;
  ///         - `Status::Code::NotFound` if no input matches (or there is no
  ///           default input), naming the inputs found;
  ///         - `Status::Code::Unsupported` for an input with fewer channels
  ///           than asked for, or that refuses the rate in 16-bit samples;
  ///         - `Status::Code::IoError` for any other PortAudio failure, with
  ///           its message.
  static core::Result<Microphone> open(const Options& options);

  Microphone(Microphone&& other) noexcept;
  Microphone& operator=(Microphone&& other) noexcept;
  /// Stops capturing and closes the input.
  ~Microphone() override;

  /// @return What the source is: the input's name as its id, the name and
  ///         PortAudio's backend as its model, `ClockDomain::Host`, and the
  ///         format opened. Empty on a moved-from source.
  const AudioInfo& info() const noexcept override;
  /// @return As `IAudioSensor::start`; `Status::Code::InvalidArgument` on a
  ///         moved-from source; `Status::Code::IoError` if PortAudio cannot
  ///         start the stream.
  core::Status start() override;
  /// @brief As `IAudioSensor::stop`. The input stays open, for another
  ///        @ref start.
  void stop() noexcept override;
  /// @return As `IAudioSensor::drain`: `Status::Code::IoError` once started
  ///         and no chunk has arrived for 1 s or more (the device has gone),
  ///         again at every later drain until one arrives;
  ///         `Status::Code::InvalidArgument` on a moved-from source.
  core::Status drain(std::vector<AudioChunk>* out) override;
  /// @return As `IAudioSensor::stats`, kept through a @ref stop:
  ///         `SensorStats::dropped` counts chunks a full queue let go, and
  ///         `failed` chunks of no frames or more than
  ///         @ref Options::chunk_frames (PortAudio hands over exactly that
  ///         many). Zero on a moved-from source.
  SensorStats stats() const noexcept override;
  /// @return Chunks since the last @ref start before which the backend lost
  ///         input (`paInputOverflow`): the device ran ahead of the callback.
  ///         Zero on a moved-from source.
  std::uint64_t overflows() const noexcept;

 private:
  struct Impl;
  explicit Microphone(std::unique_ptr<Impl> impl) noexcept;
  // Behind a pointer (see the file comment), which also keeps the
  // callback's state at one address when the source moves.
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
