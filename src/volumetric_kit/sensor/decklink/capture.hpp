// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the two SDI inputs as the stereo driver and the
// skew report see them, with no SDK type. capture.cpp, the one SDK user,
// implements Capture over the DeckLink API (VS_WITH_DECKLINK only); a test
// implements it by writing pages as the card would, so everything above it
// builds and runs on every leg.

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "volumetric_kit/core/base/result.hpp"

namespace volumetric_kit::sensor::decklink {

class PagePool;

// One input's frame as the card stamped it and the SDK's callback thread
// saw it. Copies share the SDK's frame, which goes back to the SDK when the
// last one goes.
struct Arrival {
  int input = 0;  // 0: the left camera's input; 1: the right's
  // The frame's time on the card's clock (GetHardwareReferenceTimestamp).
  std::int64_t card_ns = 0;
  // The card's clock as the callback began (GetHardwareReferenceClock), and
  // the host's std::chrono::system_clock sampled with it: the two clocks'
  // offset at this frame.
  std::int64_t card_now_ns = 0;
  std::int64_t host_now_ns = 0;
  // No picture in it (bmdFrameHasNoInputSource), or no time the card gave.
  bool no_signal = false;
  // The picture, where the card wrote it: the start of a PagePool buffer's
  // pages, or the SDK's own buffer where no pool was given.
  const std::uint8_t* bytes = nullptr;
  std::shared_ptr<void> hold;  // the SDK's frame
};

// What a capture tells its user, on the SDK's callback threads, one an
// input. Quick, and never calls back into the capture.
class CaptureSink {
 public:
  virtual void arrived(Arrival frame) = 0;
  // The input's camera sends another mode than the one opened (format
  // detection), named as the SDK names it, e.g. "2160p24".
  virtual void mode_changed(int input, std::string detected) = 0;

 protected:
  CaptureSink() = default;
  ~CaptureSink() = default;
  CaptureSink(const CaptureSink&) = default;
  CaptureSink& operator=(const CaptureSink&) = default;
};

// The two inputs, captured.
class Capture {
 public:
  virtual ~Capture() = default;
  Capture(const Capture&) = delete;
  Capture& operator=(const Capture&) = delete;
  Capture(Capture&&) = delete;
  Capture& operator=(Capture&&) = delete;

  // Both inputs enabled at the opened mode with format detection on, and
  // streaming into `sink` once each has locked to its signal, within
  // `timeout` (IoError naming an input that did not lock).
  virtual core::Status start(CaptureSink& sink,
                             std::chrono::milliseconds timeout) = 0;
  // Both inputs stopped and disabled; no call reaches the sink after this
  // returns. Safe when not started.
  virtual void stop() noexcept = 0;

 protected:
  Capture() = default;
};

// What open_capture opens: the left and right inputs, as names for messages
// ("SDI 2 (bm-1)") and sub-devices, and the mode, in the SDK's codes.
struct CaptureConfig {
  std::array<std::string, 2> names;
  std::array<int, 2> sub_device{};
  std::uint32_t mode = 0;  // BMDDisplayMode; always v210
  // Where each input's pictures go, through the SDK's capture allocator;
  // null for the SDK's own buffers (the skew report, which reads no pixel).
  std::array<std::shared_ptr<PagePool>, 2> pages;
};

// The DeckLink API's capture (capture.cpp, VS_WITH_DECKLINK only): the card
// found, its profile and both inputs' mode checked, and the SDK device each
// connector opened logged (Info); nothing captured yet.
// NotFound: no Desktop Video, or no DeckLink 8K Pro (those found named);
// Unsupported: another profile than four sub-devices half-duplex, or an
// input that cannot capture the mode as v210; IoError: an SDK failure.
core::Result<std::unique_ptr<Capture>> open_capture(
    const CaptureConfig& config);

}  // namespace volumetric_kit::sensor::decklink
