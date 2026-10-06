// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "hevc_color.hpp"

#include <memory>
#include <system_error>
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

core::Result<std::unique_ptr<HevcColorDecoder>> HevcColorDecoder::start(
    const Options& options, Sink sink) {
  std::unique_ptr<HevcColorDecoder> d(new HevcColorDecoder());
  d->options_ = options;
  d->sink_ = std::move(sink);
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
  try {
    d->thread_ = std::thread([raw = d.get()] { raw->run(); });
  } catch (const std::system_error& e) {
    return core::Status::io_error(
        options.who + ": starting the colour decoding thread: " + e.what());
  }
  return d;
}

HevcColorDecoder::~HevcColorDecoder() { stop(); }

core::Status HevcColorDecoder::failure() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failure_;
}

void HevcColorDecoder::decoder_error(core::Status why) {
  if (why.domain() == core::Status::Code::IoError) {
    gate_.resync();
    return;
  }
  // The pairs waiting and in flight are lost with the stream.
  std::uint64_t lost = in_flight_.size();
  in_flight_.clear();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    lost += queue_.size();
    queue_.clear();
    failure_ = std::move(why).with_context(options_.who);
  }
  lose(lost);
}

void HevcColorDecoder::push(std::shared_ptr<ob::FrameSet> pair) noexcept {
  if (pair == nullptr) return;
  std::uint64_t lost = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || !failure_.ok()) return;
    const std::size_t limit =
        static_cast<std::size_t>(options_.fps) * kQueueSeconds;
    if (queue_.size() >= limit) {
      lost = queue_.size();
      queue_.clear();
      resync_ = true;  // and so is everything up to the next key frame
    }
    try {
      queue_.push_back(std::move(pair));
    } catch (...) {  // out of memory: this pair goes, as an overflow's do
      ++lost;
      resync_ = true;
    }
  }
  if (lost != 0) lose(lost);
  wake_.notify_one();
}

void HevcColorDecoder::stop() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return;
    stopping_ = true;
    queue_.clear();
  }
  wake_.notify_one();
  if (thread_.joinable()) thread_.join();
  // Now, not with the last reference: the SDK may keep the frame callback
  // that holds one past the context, and the decoder may hold a hardware
  // session. Nothing reads them once the thread is gone.
  in_flight_.clear();
  decoder_.reset();
}

void HevcColorDecoder::run() {
  for (;;) {
    std::shared_ptr<ob::FrameSet> pair;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_) return;
      pair = std::move(queue_.front());
      queue_.pop_front();
      if (resync_) {
        gate_.resync();
        resync_ = false;
      }
    }
    try {
      decode(pair);
    } catch (...) {  // an SDK call that threw; never out of this thread
      lose();
      gate_.resync();
    }
  }
}

void HevcColorDecoder::decode(const std::shared_ptr<ob::FrameSet>& pair) {
  // Decoded even without its depth, which only drops it after: the frames
  // after it are predicted from it.
  const auto color = pair->getColorFrame();
  if (color == nullptr) {
    lose();
    return;
  }
  const std::uint8_t* data = color->getData();
  const std::size_t size = color->getDataSize();
  const std::uint64_t index =
      options_.frame_index ? options_.frame_index(*color) : color->getIndex();
  // No access unit, and sending none would end the stream: as a frame lost
  // on the wire.
  if (data == nullptr || size == 0) {
    lose();
    gate_.resync();
    return;
  }
  switch (gate_.admit(index, is_key_frame(data, size))) {
    case ColorStreamGate::Admission::Drop:
      lose();
      return;
    case ColorStreamGate::Admission::Restart:
      // What the decoder still holds belongs to pairs from before the loss,
      // and a CRA's leading pictures refer to frames it never had: reset, so
      // it drops the one and skips the other.
      lose(in_flight_.size());
      in_flight_.clear();
      if (core::Status reset = decoder_->reset(); !reset.ok()) {
        lose();
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
    lose();
    decoder_error(std::move(sent));
    return;
  }
  for (;;) {
    auto picture = decoder_->receive();
    if (!picture) {
      lose(in_flight_.size());
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
    lose();
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
    lose();
    return;
  }

  // For the GPU pass: the picture on the device where the hardware left it.
  sink_(rebuilt_pair(pair->getDepthFrame(), *pair->getColorFrame(),
                     picture_frame(picture)));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
