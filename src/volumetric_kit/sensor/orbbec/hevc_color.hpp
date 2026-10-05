// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed), and built only with VR_WITH_FFMPEG: one camera's
// H.265 colour, decoded on a thread of its own. A pair cannot wait in the
// mailbox as compressed colour, since the mailbox drops pairs and an H.265
// frame dropped before decoding corrupts the frames after it. So every pair
// is decoded, in order, before the mailbox, and handed on with its colour as
// an RGB frame, or for the GPU pass as I420 planes or a picture the hardware
// left on the device (picture_frames.hpp); everything after the mailbox is
// as for MJPEG.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <libobsensor/ObSensor.hpp>

#include "picture_frames.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

// Whether an Annex B access unit holds a key frame: an IRAP slice (BLA, IDR
// or CRA; NAL types 16-23), from which decoding can start.
bool is_key_frame(const std::uint8_t* data, std::size_t size) noexcept;

// Which H.265 frames can be decoded. A frame lost on the way leaves the ones
// after it predicted from a picture the decoder never had, until the next key
// frame; the camera's frame index, one up per frame it encodes, is what shows
// the loss. Its timestamps cannot: a sync secondary's first frame comes as
// soon as it starts and the rest only once the primary triggers it, up to
// seconds later, with no frame between (measured on the rig).
class ColorStreamGate {
 public:
  enum class Admission {
    Drop,     // waiting for a key frame: not decodable
    Decode,   // the next frame of an unbroken stream
    Restart,  // the key frame that ends a wait: the stream starts afresh
  };
  // What to do with frame `index` (0: unnumbered, never a gap).
  Admission admit(std::uint64_t index, bool key_frame) noexcept {
    if (index != 0 && last_index_ != 0 && index != last_index_ + 1) {
      waiting_for_key_ = true;
    }
    last_index_ = index;
    if (!waiting_for_key_) return Admission::Decode;
    if (!key_frame) return Admission::Drop;
    waiting_for_key_ = false;
    return Admission::Restart;
  }
  // Wait for the next key frame: a decode failed, or frames were dropped.
  void resync() noexcept { waiting_for_key_ = true; }

 private:
  bool waiting_for_key_ = true;  // the first frame decoded is a key frame
  std::uint64_t last_index_ = 0;
};

// The Femto Mega's colour stream: BT.601 full range, which it writes no VUI
// to say (the 2026-09-28 decision). Taken for a stream that declares no
// matrix; one that declares its own is decoded by it.
constexpr VideoColorDescription kFemtoMegaHevcColor{VideoColorMatrix::Bt601,
                                                    true};

class HevcColorDecoder {
 public:
  using Sink = std::function<void(std::shared_ptr<ob::FrameSet>)>;

  struct Options {
    std::uint32_t fps = 30;  // the stream's rate, which sizes the queue
    // Stamped on each RGB frame, so the SDK's filters see the colour
    // camera's calibration; null leaves the frame without one (the tests).
    // Never on an I420 frame, whose format it would restamp as RGB.
    std::shared_ptr<ob::StreamProfile> rgb_profile;
    bool configure_ffmpeg_logging = true;
    std::string who;
    // A colour frame's number, by which a loss shows: the SDK's index when
    // unset. A test numbers its own frames, since an SDK frame's index
    // cannot be set.
    std::function<std::uint64_t(const ob::Frame&)> frame_index;
    // Hand colour on as the decoded Y'CbCr planes, an I420 frame (Y, then Cb
    // and Cr at half size, rows packed) carrying its PlanesColor, for the GPU
    // pass; RGB otherwise.
    bool yuv = false;
    // With yuv, the device the GPU pass runs on: a picture NVDEC or
    // VideoToolbox leaves there is handed on in its raw_color_frame, and
    // only the others as I420. Borrowed: it must outlive the decoder and
    // every frame it hands on.
    const core::Device* device = nullptr;
    // With device, the allocator NVDEC's pictures are made through. Borrowed
    // as device is.
    core::Allocator* allocator = nullptr;
  };

  // Open the decoder and start its thread. `sink` gets each decoded pair, on
  // that thread.
  static core::Result<std::unique_ptr<HevcColorDecoder>> start(
      const Options& options, Sink sink);

  HevcColorDecoder(const HevcColorDecoder&) = delete;
  HevcColorDecoder& operator=(const HevcColorDecoder&) = delete;
  ~HevcColorDecoder();

  // Queue a pair (H.265 colour, and depth when the SDK paired one) for
  // decoding. Every colour frame must come through here, a lone one too: the
  // frames after it are predicted from it. Called on the SDK's thread; it
  // never waits for a decode, and a pair it cannot queue is lost, not
  // thrown. A call after stop() is ignored.
  void push(std::shared_ptr<ob::FrameSet> pair) noexcept;

  // Stop the thread, drop what is queued, uncounted, and release the decoder
  // and the RGB profile. Idempotent.
  void stop() noexcept;

  // Pairs that will not be handed on: from a gap in the colour stream's frame
  // indices, an empty colour frame, a decode error or an overflowing queue
  // until the next key frame, and any without both frames (decoded, if it
  // has colour, and dropped after). The first key frame is waited for too.
  std::uint64_t lost() const noexcept {
    return lost_.load(std::memory_order_relaxed);
  }

 private:
  HevcColorDecoder() = default;
  void run();
  void decode(const std::shared_ptr<ob::FrameSet>& pair);
  void hand_on(const DecodedPicture& picture);
  void lose(std::uint64_t pairs = 1) noexcept {
    lost_.fetch_add(pairs, std::memory_order_relaxed);
  }

  Options options_;
  Sink sink_;
  std::optional<HevcDecoder> decoder_;

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<std::shared_ptr<ob::FrameSet>> queue_;  // guarded by mutex_
  bool stopping_ = false;                            // guarded by mutex_
  bool resync_ = false;  // guarded by mutex_: the queue overflowed
  std::thread thread_;
  std::atomic<std::uint64_t> lost_{0};

  // The decode thread's alone.
  ColorStreamGate gate_;
  std::int64_t next_pts_ = 0;
  // Pairs sent and not yet decoded, by the pts they were sent with.
  std::map<std::int64_t, std::shared_ptr<ob::FrameSet>> in_flight_;
};

}  // namespace volumetric_kit::recon::sensor::orbbec
