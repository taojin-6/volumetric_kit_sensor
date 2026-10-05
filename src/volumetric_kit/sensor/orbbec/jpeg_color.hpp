// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed), and built only with the video decoders: one
// camera's MJPEG colour for a raw stream, decoded on a thread of its own
// before the mailbox, onto the GPU pass's device where nvJPEG or VideoToolbox
// takes the JPEG, in software to I420 planes otherwise. On its own thread, so
// a rig's cameras decode at once rather than one after another on the
// polling thread. A JPEG depends on no other frame, so, unlike H.265's, a
// pair lost here costs only itself, and a decoder slower than the camera
// skips pairs rather than falling behind.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <libobsensor/ObSensor.hpp>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

class JpegColorDecoder {
 public:
  using Sink = std::function<void(std::shared_ptr<ob::FrameSet>)>;

  struct Options {
    // Pairs waiting to be decoded, at most, newest kept: the mailbox's
    // depth, so the queue holds no older a pair than the mailbox would.
    std::size_t depth = 1;
    // The device the GPU pass runs on, which the JPEGs are decoded onto
    // where the hardware takes them; null decodes in software. Borrowed: it
    // must outlive the decoder and every frame it hands on.
    const core::Device* device = nullptr;
    // With device, the allocator nvJPEG's pictures are made through. Borrowed
    // as device is.
    core::Allocator* allocator = nullptr;
    bool configure_ffmpeg_logging = true;
    std::string who;
  };

  // Open the decoder and start its thread. `sink` gets each decoded pair --
  // its depth, and its colour in a raw_color_frame -- on that thread.
  static core::Result<std::unique_ptr<JpegColorDecoder>> start(
      const Options& options, Sink sink);

  JpegColorDecoder(const JpegColorDecoder&) = delete;
  JpegColorDecoder& operator=(const JpegColorDecoder&) = delete;
  ~JpegColorDecoder();

  // Queue a pair (a JPEG colour frame and its depth) for decoding. Called on
  // the SDK's thread; it never waits for a decode. A call after stop() is
  // ignored.
  void push(std::shared_ptr<ob::FrameSet> pair) noexcept;

  // Stop the thread, drop what is queued, uncounted, and release the decoder.
  // Idempotent.
  void stop() noexcept;

  // Pairs that will not be handed on: one missing either frame or with an
  // empty colour frame, a JPEG that does not decode, and the queue's oldest
  // when a pair arrives with it full.
  std::uint64_t lost() const noexcept {
    return lost_.load(std::memory_order_relaxed);
  }

 private:
  JpegColorDecoder() = default;
  void run();
  void decode(const std::shared_ptr<ob::FrameSet>& pair);
  void lose(std::uint64_t pairs = 1) noexcept {
    lost_.fetch_add(pairs, std::memory_order_relaxed);
  }

  Options options_;
  Sink sink_;
  std::optional<JpegDecoder> decoder_;

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<ob::FrameSet>> queue_;  // guarded by mutex_
  bool stopping_ = false;                            // guarded by mutex_
  std::thread thread_;
  std::atomic<std::uint64_t> lost_{0};
};

}  // namespace volumetric_kit::recon::sensor::orbbec
