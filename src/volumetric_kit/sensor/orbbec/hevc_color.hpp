// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): one camera's H.265 colour, decoded on a thread of
// its own. A pair cannot wait in the mailbox as compressed colour, since the
// mailbox drops pairs and an H.265 frame dropped before decoding corrupts the
// frames after it. So every pair is decoded, in order, before the mailbox,
// and handed on with its colour as the picture the hardware left on the
// device (picture_frames.hpp); everything after the mailbox is as for MJPEG.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <libobsensor/ObSensor.hpp>

#include "decode_worker.hpp"
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
    bool configure_ffmpeg_logging = true;
    std::string who;
    // A colour frame's number, by which a loss shows: the SDK's index when
    // unset. A test numbers its own frames, since an SDK frame's index
    // cannot be set.
    std::function<std::uint64_t(const ob::Frame&)> frame_index;
    // The device the pictures are decoded onto and handed on in their
    // picture_frame: required. Borrowed: it must outlive the decoder and
    // every frame it hands on.
    const core::Device* device = nullptr;
    // The allocator NVDEC's pictures are made through. Borrowed as device is.
    core::Allocator* allocator = nullptr;
  };

  // Open the decoder and start its thread. `sink` gets each decoded pair, on
  // that thread. Unsupported where the decoder has no device path.
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

  // Stop the thread, drop what is queued, uncounted, and release the decoder.
  // Idempotent.
  void stop() noexcept;

  // Pairs that will not be handed on: from a gap in the colour stream's frame
  // indices, an empty colour frame, a decode error (IoError) or an
  // overflowing queue until the next key frame, and any without both frames
  // (decoded, if it has colour, and dropped after). The first key frame is
  // waited for too.
  std::uint64_t lost() const noexcept { return worker_.lost(); }
  // Why decoding stopped for good -- a stream the hardware refuses
  // (Unsupported), or a device path that failed (Backend, OutOfMemory) --
  // after which nothing more is handed on; OK while it runs.
  core::Status failure() const { return worker_.failure(); }

 private:
  using Worker = DecodeWorker<std::shared_ptr<ob::FrameSet>>;

  HevcColorDecoder(const Options& options, Sink sink);
  void decode(const std::shared_ptr<ob::FrameSet>& pair, bool after_gap);
  void hand_on(const DecodedPicture& picture);
  // After a decoder error, whose pair the caller has counted lost: IoError
  // waits for the next key frame, any other stops decoding. Decode thread
  // only.
  void decoder_error(core::Status why);

  Options options_;
  Sink sink_;
  std::optional<HevcDecoder> decoder_;

  // The decode thread's alone.
  ColorStreamGate gate_;
  std::int64_t next_pts_ = 0;
  // Pairs sent and not yet decoded, by the pts they were sent with.
  std::map<std::int64_t, std::shared_ptr<ob::FrameSet>> in_flight_;

  Worker worker_;  // last: its thread uses the rest
};

}  // namespace volumetric_kit::recon::sensor::orbbec
