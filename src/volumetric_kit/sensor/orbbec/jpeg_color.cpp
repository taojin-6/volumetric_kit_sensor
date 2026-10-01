// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "jpeg_color.hpp"

#include <system_error>
#include <utility>

#include "picture_frames.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

Result<std::unique_ptr<JpegColorDecoder>> JpegColorDecoder::start(
    const Options& options, Sink sink) {
  std::unique_ptr<JpegColorDecoder> d(new JpegColorDecoder());
  d->options_ = options;
  d->sink_ = std::move(sink);
  JpegDecoder::Options decoding;
  decoding.device = options.device;
  decoding.configure_ffmpeg_logging = options.configure_ffmpeg_logging;
  decoding.label = options.who;
  auto decoder = JpegDecoder::create(decoding);
  if (!decoder) {
    return Status::io_error(options.who + ": opening the JPEG decoder: " +
                            decoder.status().message());
  }
  d->decoder_.emplace(std::move(decoder).value());
  try {
    d->thread_ = std::thread([raw = d.get()] { raw->run(); });
  } catch (const std::system_error& e) {
    return Status::io_error(
        options.who + ": starting the colour decoding thread: " + e.what());
  }
  return d;
}

JpegColorDecoder::~JpegColorDecoder() { stop(); }

void JpegColorDecoder::push(std::shared_ptr<ob::FrameSet> pair) noexcept {
  if (pair == nullptr) return;
  std::uint64_t lost = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return;
    // The oldest go, a JPEG needing no other.
    while (!queue_.empty() && queue_.size() >= options_.depth) {
      queue_.pop_front();
      ++lost;
    }
    try {
      queue_.push_back(std::move(pair));
    } catch (...) {  // out of memory: this pair goes
      ++lost;
    }
  }
  if (lost != 0) lose(lost);
  wake_.notify_one();
}

void JpegColorDecoder::stop() noexcept {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return;
    stopping_ = true;
    queue_.clear();
  }
  wake_.notify_one();
  if (thread_.joinable()) thread_.join();
  // Now, not with the last reference: the SDK may keep the frame callback
  // that holds one past the context, and the decoder holds a GPU session.
  decoder_.reset();
}

void JpegColorDecoder::run() {
  for (;;) {
    std::shared_ptr<ob::FrameSet> pair;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (stopping_) return;
      pair = std::move(queue_.front());
      queue_.pop_front();
    }
    try {
      decode(pair);
    } catch (...) {  // an SDK call that threw; never out of this thread
      lose();
    }
  }
}

void JpegColorDecoder::decode(const std::shared_ptr<ob::FrameSet>& pair) {
  const auto depth = pair->getDepthFrame();
  const auto color = pair->getColorFrame();
  if (depth == nullptr || color == nullptr || color->getData() == nullptr ||
      color->getDataSize() == 0) {
    lose();
    return;
  }
  auto picture = decoder_->decode(color->getData(), color->getDataSize());
  if (!picture) {  // a corrupt JPEG, or one the software cannot read
    lose();
    return;
  }
  sink_(rebuilt_pair(depth, *color, raw_color_frame(picture.value())));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
