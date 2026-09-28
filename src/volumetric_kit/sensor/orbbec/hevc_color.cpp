// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "hevc_color.hpp"

#include <cstring>
#include <utility>

namespace volumetric_kit::recon::sensor::orbbec {

namespace {

// Pairs waiting for the decode thread, at most: two seconds at the stream's
// rate. A decoder that far behind is not catching up, and what is queued is
// dropped and the stream picked up at the next key frame.
constexpr std::uint32_t kQueueSeconds = 2;

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

Result<std::unique_ptr<HevcColorDecoder>> HevcColorDecoder::start(
    const Options& options, Sink sink) {
  std::unique_ptr<HevcColorDecoder> d(new HevcColorDecoder());
  d->options_ = options;
  d->sink_ = std::move(sink);
  HevcDecoder::Options decoding;
  decoding.layout = VideoPixelLayout::Rgb24;
  decoding.threads = 1;  // a live stream: no picture held back
  decoding.color = kFemtoMegaHevcColor;
  decoding.configure_ffmpeg_logging = options.configure_ffmpeg_logging;
  auto decoder = HevcDecoder::create(decoding);
  if (!decoder) {
    return Status::unsupported(
        options.who + ": no HEVC decoder: " + decoder.status().message());
  }
  d->decoder_.emplace(std::move(decoder).value());
  d->thread_ = std::thread([raw = d.get()] { raw->run(); });
  return d;
}

HevcColorDecoder::~HevcColorDecoder() { stop(); }

void HevcColorDecoder::push(std::shared_ptr<ob::FrameSet> pair) noexcept {
  if (pair == nullptr) return;
  std::size_t overflow = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return;
    const std::size_t limit =
        static_cast<std::size_t>(options_.fps) * kQueueSeconds;
    if (queue_.size() >= limit) {
      overflow = queue_.size();
      queue_.clear();
      resync_ = true;  // and so is everything up to the next key frame
    }
    queue_.push_back(std::move(pair));
  }
  if (overflow != 0) lose(overflow);
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
  in_flight_.clear();
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
  if (!gate_.admit(color->getIndex(), is_key_frame(data, size))) {
    lose();
    return;
  }

  const std::int64_t pts = next_pts_++;
  in_flight_.emplace(pts, pair);
  if (!decoder_->send(data, size, pts)) {
    in_flight_.erase(pts);
    lose();
    gate_.resync();
    return;
  }
  for (;;) {
    auto picture = decoder_->receive();
    if (!picture) {
      lose(in_flight_.size());
      in_flight_.clear();
      gate_.resync();
      return;
    }
    if (!picture.value()) return;
    hand_on(*picture.value());
  }
}

void HevcColorDecoder::hand_on(const DecodedPicture& picture) {
  const auto found = in_flight_.find(picture.pts);
  if (found == in_flight_.end()) return;
  // Pairs sent before this one that produced no picture never will.
  lose(static_cast<std::uint64_t>(std::distance(in_flight_.begin(), found)));
  const std::shared_ptr<ob::FrameSet> pair = found->second;
  in_flight_.erase(in_flight_.begin(), std::next(found));
  if (pair->getDepthFrame() == nullptr) {
    lose();
    return;
  }

  const auto color = pair->getColorFrame();
  const std::size_t row = 3u * picture.width;
  auto rgb = ob::FrameFactory::createVideoFrame(
      OB_FRAME_COLOR, OB_FORMAT_RGB, picture.width, picture.height,
      static_cast<std::uint32_t>(row));
  std::uint8_t* out = rgb->getData();
  for (std::uint32_t y = 0; y < picture.height; ++y) {
    std::memcpy(out + y * row, picture.plane[0] + y * picture.stride[0], row);
  }
  ob::FrameHelper::setFrameDeviceTimestampUs(rgb, color->getTimeStampUs());
  rgb->setSystemTimestampUs(color->getSystemTimeStampUs());
  if (options_.rgb_profile != nullptr) {
    rgb->setStreamProfile(options_.rgb_profile);
  }
  auto rebuilt = ob::FrameFactory::createFrameSet();
  rebuilt->pushFrame(pair->getDepthFrame());
  rebuilt->pushFrame(rgb);
  sink_(std::move(rebuilt));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
