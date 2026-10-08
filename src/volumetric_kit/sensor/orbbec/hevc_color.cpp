// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "hevc_color.hpp"

#include <cstddef>
#include <memory>
#include <utility>

namespace volumetric_kit::recon::sensor::orbbec {

namespace {

// Pairs waiting for the decode thread, at most: two seconds at the stream's
// rate. A decoder that far behind is not catching up, and what is queued is
// dropped and the stream picked up at the next key frame.
constexpr std::uint32_t kQueueSeconds = 2;

// Access units after its own by which a picture has come out, if it ever
// will. H.265 holds at most 16 pictures back for display, and one decoding
// thread holds none; but an FFmpeg before 7.1 hands out one picture per
// access unit, so the pictures a new sequence holds back can also wait
// behind the last one's, as many again. A pair sent longer ago with no
// picture is one the decoder skipped -- a CRA's leading pictures after a
// restart, say.
constexpr std::int64_t kMaxPictureDelay = 32;

}  // namespace

bool is_key_frame(const std::uint8_t* data, std::size_t size) noexcept {
  for (std::size_t i = 0; i + 3 < size; ++i) {
    if (data[i] != 0 || data[i + 1] != 0 || data[i + 2] != 1) continue;
    const int type = (data[i + 3] >> 1) & 0x3f;
    if (type < 32) return type >= 16 && type <= 23;  // the first slice
    i += 2;
  }
  return false;
}

HevcColorDecoder::HevcColorDecoder(const Options& options, Sink sink)
    : options_(options),
      sink_(std::move(sink)),
      worker_(static_cast<std::size_t>(options.fps) * kQueueSeconds,
              Worker::Overflow::DropAll) {}

core::Result<std::unique_ptr<HevcColorDecoder>> HevcColorDecoder::start(
    const Options& options, Sink sink) {
  std::unique_ptr<HevcColorDecoder> d(
      new HevcColorDecoder(options, std::move(sink)));
  HevcDecoder::Options decoding;
  decoding.unlabelled_color = kFemtoMegaHevcColor;
  decoding.device = options.device;
  decoding.allocator = options.allocator;
  decoding.configure_ffmpeg_logging = options.configure_ffmpeg_logging;
  auto decoder = HevcDecoder::create(decoding);
  if (!decoder) {
    return decoder.status().with_context(options.who +
                                         ": opening the HEVC decoder");
  }
  d->decoder_.emplace(std::move(decoder).value());
  core::Status started = d->worker_.start(
      [raw = d.get()](std::shared_ptr<ob::FrameSet> pair, bool after_gap) {
        raw->decode(pair, after_gap);
      });
  if (!started.ok()) return std::move(started).with_context(options.who);
  return d;
}

HevcColorDecoder::~HevcColorDecoder() { stop(); }

void HevcColorDecoder::decoder_error(core::Status why) {
  if (why.domain() == core::Status::Code::IoError) {
    gate_.resync();
    return;
  }
  // The pairs in flight are lost with the stream, and the worker counts
  // those waiting.
  worker_.lose(in_flight_.size());
  in_flight_.clear();
  worker_.fail(std::move(why).with_context(options_.who));
}

void HevcColorDecoder::push(std::shared_ptr<ob::FrameSet> pair) noexcept {
  if (pair == nullptr) return;
  worker_.push(std::move(pair));
}

void HevcColorDecoder::stop() noexcept {
  if (!worker_.stop()) return;
  // Now, not with the last reference: the SDK may keep the frame callback
  // that holds one past the context, and the decoder may hold a hardware
  // session. Nothing reads them once the thread is gone.
  in_flight_.clear();
  decoder_.reset();
}

void HevcColorDecoder::decode(const std::shared_ptr<ob::FrameSet>& pair,
                              bool after_gap) {
  // A pair dropped before it, or one that threw, leaves the stream broken.
  if (after_gap) gate_.resync();
  // Decoded even without its depth, which only drops it after: the frames
  // after it are predicted from it.
  const auto color = pair->getColorFrame();
  if (color == nullptr) {
    worker_.lose();
    return;
  }
  const std::uint8_t* data = color->getData();
  const std::size_t size = color->getDataSize();
  const std::uint64_t index =
      options_.frame_index ? options_.frame_index(*color) : color->getIndex();
  // No access unit, and sending none would end the stream: as a frame lost
  // on the wire.
  if (data == nullptr || size == 0) {
    worker_.lose();
    gate_.resync();
    return;
  }
  switch (gate_.admit(index, is_key_frame(data, size))) {
    case ColorStreamGate::Admission::Drop:
      worker_.lose();
      return;
    case ColorStreamGate::Admission::Restart:
      // What the decoder still holds belongs to pairs from before the loss,
      // and a CRA's leading pictures refer to frames it never had: reset, so
      // it drops the one and skips the other.
      worker_.lose(in_flight_.size());
      in_flight_.clear();
      if (core::Status reset = decoder_->reset(); !reset.ok()) {
        worker_.lose();
        decoder_error(std::move(reset));
        return;
      }
      break;
    case ColorStreamGate::Admission::Decode:
      break;
  }

  const std::int64_t pts = next_pts_++;
  in_flight_.emplace(pts, pair);
  if (core::Status sent = decoder_->send(data, size, pts); !sent.ok()) {
    in_flight_.erase(pts);
    worker_.lose();
    decoder_error(std::move(sent));
    return;
  }
  for (;;) {
    auto picture = decoder_->receive();
    if (!picture) {
      worker_.lose(in_flight_.size());
      in_flight_.clear();
      decoder_error(picture.status());
      return;
    }
    if (!picture.value()) break;
    hand_on(*picture.value());
  }
  while (!in_flight_.empty() &&
         in_flight_.begin()->first + kMaxPictureDelay < next_pts_) {
    in_flight_.erase(in_flight_.begin());
    worker_.lose();
  }
}

void HevcColorDecoder::hand_on(const DecodedPicture& picture) {
  // Pictures come out in display order, which B-frames are not sent in, so a
  // picture settles its own pair and no other.
  const auto found = in_flight_.find(picture.pts);
  if (found == in_flight_.end()) return;
  const std::shared_ptr<ob::FrameSet> pair = std::move(found->second);
  in_flight_.erase(found);
  if (pair->getDepthFrame() == nullptr) {
    worker_.lose();
    return;
  }

  // For the GPU pass: the picture on the device where the hardware left it.
  sink_(rebuilt_pair(pair->getDepthFrame(), *pair->getColorFrame(),
                     picture_frame(picture)));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
