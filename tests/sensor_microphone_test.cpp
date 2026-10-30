// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The microphone driver's PortAudio side (VS_WITH_PORTAUDIO only), with no
// device and no Pa_Initialize, which would look for one:
//   - the PortAudio that loads is a v19;
//   - open refuses options out of range before PortAudio starts;
//   - Options::device names the default input when empty, else the input
//     named exactly, else the one whose name contains it; several matches
//     are refused naming them, none as not found naming every input
//     (audio/choose_input.hpp);
//   - the stream callback (audio/portaudio_callback.hpp), called as PortAudio
//     calls it with made-up times and flags, stamps a chunk at the host clock
//     less its first sample's wait, never earlier than the chunk before,
//     counts an input overflow and nothing else, takes a missing buffer as
//     silence, and always continues.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include <portaudio.h>

#include "capture.hpp"
#include "choose_input.hpp"
#include "portaudio_callback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/audio/microphone.hpp"

namespace audio = volumetric_kit::sensor::audio;
namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;
using Clock = audio::Capture::Clock;

static_assert(!std::is_copy_constructible_v<sensor::Microphone>);
static_assert(std::is_nothrow_move_constructible_v<sensor::Microphone>);
static_assert(std::is_nothrow_move_assignable_v<sensor::Microphone>);

namespace {

std::uint64_t system_now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

int test_version() {
  const PaVersionInfo* v = Pa_GetVersionInfo();
  CHECK(v != nullptr && v->versionMajor == 19);
  std::printf("  %s\n", v->versionText);
  return 0;
}

int test_open_refuses() {
  // Refused before PortAudio starts, so no device is looked for.
  sensor::Microphone::Options options;
  options.channels = 3;
  auto three = sensor::Microphone::open(options);
  CHECK(!three.ok() &&
        three.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(three.status().message().rfind("Microphone: ", 0) == 0);
  options = {};
  options.chunk_frames = 0;
  auto empty = sensor::Microphone::open(options);
  CHECK(!empty.ok() &&
        empty.status().domain() == vkc::Status::Code::InvalidArgument);
  return 0;
}

int test_choose_input() {
  const auto input = [](const char* name, bool is_default) {
    sensor::Microphone::Input in;
    in.name = name;
    in.is_default = is_default;
    return in;
  };
  const std::vector<sensor::Microphone::Input> inputs = {
      input("MacBook Pro Microphone", true), input("USB Mic", false),
      input("USB Mic (hw:1,0)", false), input("Line In", false)};
  const auto chosen = [&](const std::string& device) {
    const vkc::Result<std::size_t> r = audio::choose_input(inputs, device);
    return r.ok() ? static_cast<int>(*r) : -1;
  };
  const auto has = [](const vkc::Status& s, const std::string& text) {
    return s.message().find(text) != std::string::npos;
  };
  CHECK(chosen("") == 0);         // the default
  CHECK(chosen("USB Mic") == 1);  // exact, though two contain it
  CHECK(chosen("hw:1,0") == 2);   // the one that contains it
  CHECK(chosen("Line In") == 3);

  // Several contain it: refused, naming them and only them.
  const vkc::Status several = audio::choose_input(inputs, "Mic").status();
  CHECK(several.domain() == vkc::Status::Code::InvalidArgument);
  CHECK(has(several,
            "\"Mic\" names several inputs: \"MacBook Pro "
            "Microphone\", \"USB Mic\", \"USB Mic (hw:1,0)\""));
  CHECK(!has(several, "Line In"));

  // None does: not found, naming every input.
  const vkc::Status none = audio::choose_input(inputs, "Zoom").status();
  CHECK(none.domain() == vkc::Status::Code::NotFound);
  CHECK(has(none,
            "no input \"Zoom\"; inputs found: \"MacBook Pro "
            "Microphone\", \"USB Mic\", \"USB Mic (hw:1,0)\", "
            "\"Line In\""));

  // No default input, or no input at all.
  const std::vector<sensor::Microphone::Input> no_default(inputs.begin() + 1,
                                                          inputs.end());
  const vkc::Status no_def = audio::choose_input(no_default, "").status();
  CHECK(no_def.domain() == vkc::Status::Code::NotFound);
  CHECK(has(no_def, "no default input; inputs found: \"USB Mic\""));
  const vkc::Status empty = audio::choose_input({}, "Mic").status();
  CHECK(empty.domain() == vkc::Status::Code::NotFound);
  CHECK(has(empty, "inputs found: none"));
  return 0;
}

int test_callback() {
  // Mono, 4-frame chunks, 4 held.
  audio::Capture capture(1, 4, 4);
  capture.start(Clock::time_point{});
  const std::int16_t s[4] = {100, -200, 300, -400};
  PaStreamCallbackTimeInfo time{};
  time.currentTime = 250.040;  // the backend's clock: uptime, say
  time.inputBufferAdcTime = 250.0;
  time.outputBufferDacTime = 0.0;

  // A first sample that waited 40 ms.
  std::uint64_t before = system_now_ns();
  CHECK(audio::on_portaudio_input(s, nullptr, 4, &time, 0, &capture) ==
        paContinue);
  std::uint64_t after = system_now_ns();
  std::vector<sensor::AudioChunk> out;
  CHECK(capture.drain(&out, Clock::time_point{}).ok() && out.size() == 1);
  const std::uint64_t wait = 40'000'000;
  CHECK(out[0].timestamp_ns >= before - wait - 1000);
  CHECK(out[0].timestamp_ns <= after - wait + 1000);
  CHECK((out[0].samples == std::vector<std::int16_t>{100, -200, 300, -400}));
  const std::uint64_t first = out[0].timestamp_ns;

  // An overflow is counted; an underflow, priming or the output's flags are
  // not.
  CHECK(audio::on_portaudio_input(s, nullptr, 4, &time, paInputOverflow,
                                  &capture) == paContinue);
  CHECK(capture.overflows() == 1);
  CHECK(audio::on_portaudio_input(s, nullptr, 4, &time,
                                  paInputUnderflow | paOutputUnderflow |
                                      paOutputOverflow | paPrimingOutput,
                                  &capture) == paContinue);
  CHECK(capture.overflows() == 1);

  // No buffer: silence; no times: stamped at the host clock.
  before = system_now_ns();
  CHECK(audio::on_portaudio_input(nullptr, nullptr, 4, nullptr, 0, &capture) ==
        paContinue);
  after = system_now_ns();
  out.clear();
  CHECK(capture.drain(&out, Clock::time_point{}).ok() && out.size() == 3);
  CHECK(out[0].timestamp_ns >= first && out[1].timestamp_ns >= first);
  CHECK(out[2].timestamp_ns >= before && out[2].timestamp_ns <= after);
  CHECK((out[2].samples == std::vector<std::int16_t>(4, 0)));
  const std::uint64_t third = out[2].timestamp_ns;

  // A wait of 900 ms would put the next chunk before that one: it takes that
  // one's stamp instead.
  time.inputBufferAdcTime = time.currentTime - 0.9;
  CHECK(audio::on_portaudio_input(s, nullptr, 4, &time, 0, &capture) ==
        paContinue);
  out.clear();
  CHECK(capture.drain(&out, Clock::time_point{}).ok() && out.size() == 1);
  CHECK(out[0].timestamp_ns == third);
  const sensor::SensorStats st = capture.stats();
  CHECK(st.received == 5 && st.delivered == 5 && st.failed == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_version() != 0) return 1;
  if (test_open_refuses() != 0) return 1;
  if (test_choose_input() != 0) return 1;
  if (test_callback() != 0) return 1;
  std::puts("sensor_microphone: OK");
  return 0;
}
