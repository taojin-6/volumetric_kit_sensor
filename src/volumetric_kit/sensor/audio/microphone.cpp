// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/audio/microphone.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <portaudio.h>

#include "capture.hpp"
#include "choose_input.hpp"
#include "portaudio_callback.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor {

namespace {

constexpr const char* kWho = "Microphone";

core::Status portaudio_failure(const std::string& what, PaError error) {
  return core::Status::io_error(std::string(kWho) + ": " + what + ": " +
                                Pa_GetErrorText(error));
}

// One Pa_Initialize, and its Pa_Terminate once it succeeded; PortAudio counts
// them, so each user holds its own.
class PortAudio {
 public:
  PortAudio() : error_(Pa_Initialize()) {}
  ~PortAudio() {
    if (error_ == paNoError) Pa_Terminate();
  }
  PortAudio(const PortAudio&) = delete;
  PortAudio& operator=(const PortAudio&) = delete;

  core::Status status() const {
    return error_ == paNoError ? core::Status{}
                               : portaudio_failure("Pa_Initialize", error_);
  }

 private:
  PaError error_;
};

struct Found {
  PaDeviceIndex index = paNoDevice;
  Microphone::Input input;
};

// Every input device, in PortAudio's order.
core::Result<std::vector<Found>> find_inputs() {
  const PaDeviceIndex count = Pa_GetDeviceCount();
  if (count < 0) return portaudio_failure("Pa_GetDeviceCount", count);
  const PaDeviceIndex default_input = Pa_GetDefaultInputDevice();
  std::vector<Found> found;
  for (PaDeviceIndex i = 0; i < count; ++i) {
    const PaDeviceInfo* device = Pa_GetDeviceInfo(i);
    if (device == nullptr || device->maxInputChannels <= 0) continue;
    const PaHostApiInfo* api = Pa_GetHostApiInfo(device->hostApi);
    Found f;
    f.index = i;
    f.input.name = device->name != nullptr ? device->name : "";
    f.input.host_api = api != nullptr && api->name != nullptr ? api->name : "";
    f.input.channels = static_cast<std::uint16_t>(device->maxInputChannels);
    f.input.default_sample_rate = device->defaultSampleRate;
    f.input.is_default = i == default_input;
    found.push_back(std::move(f));
  }
  return found;
}

// What each one is, in the same order.
std::vector<Microphone::Input> inputs_of(const std::vector<Found>& found) {
  std::vector<Microphone::Input> inputs;
  inputs.reserve(found.size());
  for (const Found& f : found) inputs.push_back(f.input);
  return inputs;
}

}  // namespace

struct Microphone::Impl {
  // First, so PortAudio terminates after the stream closes.
  PortAudio library;
  audio::Capture capture;
  AudioInfo info;
  PaStream* stream = nullptr;

  explicit Impl(const Options& options)
      : capture(options.channels, options.chunk_frames, options.queue_chunks) {}
  ~Impl() {
    // Aborts a running stream: no callback runs once it returns.
    if (stream != nullptr) Pa_CloseStream(stream);
  }
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
};

Microphone::Microphone(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
Microphone::Microphone(Microphone&& other) noexcept = default;
Microphone& Microphone::operator=(Microphone&& other) noexcept = default;
Microphone::~Microphone() { stop(); }

core::Result<std::vector<Microphone::Input>> Microphone::inputs() {
  const PortAudio library;
  VKC_TRY(library.status());
  VKC_ASSIGN(const std::vector<Found> found, find_inputs());
  return inputs_of(found);
}

core::Result<Microphone> Microphone::open(const Options& options) {
  if (const core::Status s =
          audio::check_format(options.sample_rate, options.channels,
                              options.chunk_frames, options.queue_chunks);
      !s) {
    return s.with_context(kWho);
  }
  auto impl = std::make_unique<Impl>(options);
  VKC_TRY(impl->library.status());
  VKC_ASSIGN(const std::vector<Found> found, find_inputs());
  const core::Result<std::size_t> chosen =
      audio::choose_input(inputs_of(found), options.device);
  if (!chosen) return chosen.status().with_context(kWho);
  const Found& input = found[*chosen];
  const std::string& name = input.input.name;
  if (input.input.channels < options.channels) {
    return core::Status::unsupported(
        std::string(kWho) + ": \"" + name + "\" has " +
        std::to_string(input.input.channels) + " input channel(s), not " +
        std::to_string(options.channels));
  }

  // The device's default high latency: a deeper host buffer rides out a late
  // callback, and each chunk's ADC time places it whatever the latency.
  PaStreamParameters params{};
  params.device = input.index;
  params.channelCount = options.channels;
  params.sampleFormat = paInt16;
  params.suggestedLatency =
      Pa_GetDeviceInfo(input.index)->defaultHighInputLatency;
  params.hostApiSpecificStreamInfo = nullptr;
  if (const PaError e =
          Pa_IsFormatSupported(&params, nullptr, options.sample_rate);
      e != paFormatIsSupported) {
    return core::Status::unsupported(
        std::string(kWho) + ": \"" + name + "\" refuses " +
        std::to_string(options.channels) + " channel(s) of 16-bit samples at " +
        std::to_string(options.sample_rate) + " Hz: " + Pa_GetErrorText(e));
  }
  if (const PaError e =
          Pa_OpenStream(&impl->stream, &params, nullptr, options.sample_rate,
                        options.chunk_frames, paNoFlag,
                        &audio::on_portaudio_input, &impl->capture);
      e != paNoError) {
    impl->stream = nullptr;
    return portaudio_failure("opening \"" + name + "\"", e);
  }

  AudioInfo& info = impl->info;
  info.id = name;
  info.model = name + " (" + input.input.host_api + ")";
  info.clock = ClockDomain::Host;
  info.sample_rate = options.sample_rate;
  info.channels = options.channels;
  info.chunk_frames = options.chunk_frames;
  return Microphone(std::move(impl));
}

const AudioInfo& Microphone::info() const noexcept {
  static const AudioInfo kEmpty{};
  return impl_ != nullptr ? impl_->info : kEmpty;
}

core::Status Microphone::start() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "Microphone: start on a moved-from source");
  }
  if (impl_->capture.running()) return {};
  // Before the stream starts, so its first callback finds the counters
  // restarted.
  impl_->capture.start(audio::Capture::Clock::now());
  if (const PaError e = Pa_StartStream(impl_->stream); e != paNoError) {
    impl_->capture.stop();
    return portaudio_failure("starting \"" + impl_->info.id + "\"", e);
  }
  return {};
}

void Microphone::stop() noexcept {
  if (impl_ == nullptr || !impl_->capture.running()) return;
  // Returns once the callback has run for the last time.
  Pa_StopStream(impl_->stream);
  impl_->capture.stop();
}

core::Status Microphone::drain(std::vector<AudioChunk>* out) {
  VKC_CHECK(out != nullptr, "Microphone::drain needs somewhere to put them");
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(
        "Microphone: drain on a moved-from source");
  }
  if (core::Status s = impl_->capture.drain(out, audio::Capture::Clock::now());
      !s) {
    return std::move(s).with_context(std::string(kWho) + " \"" +
                                     impl_->info.id + "\"");
  }
  return {};
}

SensorStats Microphone::stats() const noexcept {
  return impl_ != nullptr ? impl_->capture.stats() : SensorStats{};
}

std::uint64_t Microphone::overflows() const noexcept {
  return impl_ != nullptr ? impl_->capture.overflows() : 0;
}

}  // namespace volumetric_kit::sensor
