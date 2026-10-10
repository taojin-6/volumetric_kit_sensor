// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "jpeg_color.hpp"

#include <utility>

#include "picture_frames.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

JpegColorDecoder::JpegColorDecoder(const Options& options, Sink sink)
    : options_(options),
      sink_(std::move(sink)),
      // The oldest go, a JPEG needing no other.
      worker_(options.depth, Worker::Overflow::DropOldest) {}

core::Result<std::unique_ptr<JpegColorDecoder>> JpegColorDecoder::start(
    const Options& options, Sink sink) {
  std::unique_ptr<JpegColorDecoder> d(
      new JpegColorDecoder(options, std::move(sink)));
  JpegDecoder::Options decoding;
  decoding.device = options.device;
  decoding.allocator = options.allocator;
  auto decoder = JpegDecoder::create(decoding);
  if (!decoder) {
    return decoder.status().with_context(options.who +
                                         ": opening the JPEG decoder");
  }
  d->decoder_.emplace(std::move(decoder).value());
  core::Status started =
      d->worker_.start([raw = d.get()](std::shared_ptr<ob::FrameSet> pair,
                                       bool) { raw->decode(pair); });
  if (!started.ok()) return std::move(started).with_context(options.who);
  return d;
}

JpegColorDecoder::~JpegColorDecoder() { stop(); }

void JpegColorDecoder::push(std::shared_ptr<ob::FrameSet> pair) noexcept {
  if (pair == nullptr) return;
  worker_.push(std::move(pair));
}

void JpegColorDecoder::stop() noexcept {
  if (!worker_.stop()) return;
  // Now, not with the last reference: the SDK may keep the frame callback
  // that holds one past the context, and the decoder holds a GPU session.
  decoder_.reset();
}

void JpegColorDecoder::decode(const std::shared_ptr<ob::FrameSet>& pair) {
  const auto depth = pair->getDepthFrame();
  const auto color = pair->getColorFrame();
  if (depth == nullptr || color == nullptr || color->getData() == nullptr ||
      color->getDataSize() == 0) {
    worker_.lose();
    return;
  }
  auto picture = decoder_->decode(color->getData(), color->getDataSize());
  if (!picture) {
    worker_.lose();
    // A corrupt JPEG costs itself; anything else, every JPEG after it.
    if (picture.status().domain() != core::Status::Code::IoError) {
      worker_.fail(picture.status().with_context(options_.who));
    }
    return;
  }
  sink_(rebuilt_pair(depth, *color, picture_frame(picture.value())));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
