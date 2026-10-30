// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): Microphone's PortAudio stream callback, apart from
// the stream, so a test calls it with made-up times and flags and no
// Pa_Initialize. It runs on PortAudio's realtime thread: it reads the host
// clock and hands the chunk to Capture::on_input, and locks, allocates and
// logs nothing.

#include <chrono>
#include <cstdint>

#include <portaudio.h>

#include "capture.hpp"

namespace volumetric_kit::sensor::audio {

// A PaStreamCallback for an input stream of paInt16 samples whose user data is
// a Capture. A chunk's first sample waited `currentTime - inputBufferAdcTime`
// (both on Pa_GetStreamTime's clock) before the callback read system_clock;
// paInputOverflow is counted. Always paContinue.
inline int on_portaudio_input(const void* input, void* /*output*/,
                              unsigned long frames,
                              const PaStreamCallbackTimeInfo* time,
                              PaStreamCallbackFlags flags, void* capture) {
  const std::int64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  const double current_time = time != nullptr ? time->currentTime : 0.0;
  const double adc_time = time != nullptr ? time->inputBufferAdcTime : 0.0;
  static_cast<Capture*>(capture)->on_input(
      static_cast<const std::int16_t*>(input), frames, now_ns, current_time,
      adc_time, (flags & paInputOverflow) != 0);
  return paContinue;
}

}  // namespace volumetric_kit::sensor::audio
