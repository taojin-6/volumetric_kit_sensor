// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "hevc_color.hpp"

#include <cstring>
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

// A Yuv420 picture as an I420 frame: Y, then Cb and Cr at half size (rounded
// up), each plane's rows packed. The frame owns the copy.
std::shared_ptr<ob::VideoFrame> i420_frame(const DecodedPicture& picture) {
  const std::uint32_t w = picture.width;
  const std::uint32_t h = picture.height;
  const std::uint32_t widths[3] = {w, (w + 1) / 2, (w + 1) / 2};
  const std::uint32_t heights[3] = {h, (h + 1) / 2, (h + 1) / 2};
  const std::size_t bytes =
      std::size_t{w} * h + 2 * std::size_t{widths[1]} * heights[1];
  std::unique_ptr<std::uint8_t[]> buffer(new std::uint8_t[bytes]);
  std::uint8_t* out = buffer.get();
  for (int p = 0; p < 3; ++p) {
    for (std::uint32_t y = 0; y < heights[p]; ++y) {
      std::memcpy(out, picture.plane[p] + y * picture.stride[p], widths[p]);
      out += widths[p];
    }
  }
  auto frame = ob::FrameFactory::createVideoFrameFromBuffer(
      OB_FRAME_COLOR, OB_FORMAT_I420, w, h, buffer.get(),
      [](std::uint8_t* b) { delete[] b; }, static_cast<std::uint32_t>(bytes),
      w);
  buffer.release();  // the frame's now, freed by the callback
  return frame;
}

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
  decoding.layout =
      options.yuv ? VideoPixelLayout::Yuv420 : VideoPixelLayout::Rgb24;
  // One thread, so no picture is held back. That leaves software decoding
  // little headroom at 4K25 (the 2026-09-28 decision).
  // TODO(sensor): frame threads in software alone, or the conversion on the
  // GPU (hevc_decoder.cpp's TODO), once a host without a hardware HEVC
  // decoder needs 4K.
  decoding.threads = 1;
  decoding.unlabelled_color = kFemtoMegaHevcColor;
  decoding.configure_ffmpeg_logging = options.configure_ffmpeg_logging;
  auto decoder = HevcDecoder::create(decoding);
  if (!decoder) {
    const std::string why = options.who + ": opening the HEVC decoder: " +
                            decoder.status().message();
    return decoder.status().domain() == Status::Code::Unsupported
               ? Status::unsupported(why)
               : Status::io_error(why);
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

HevcColorDecoder::~HevcColorDecoder() { stop(); }

void HevcColorDecoder::push(std::shared_ptr<ob::FrameSet> pair) noexcept {
  if (pair == nullptr) return;
  std::uint64_t lost = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) return;
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
  options_.rgb_profile.reset();
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
      if (!decoder_->reset()) {
        lose();
        gate_.resync();
        return;
      }
      break;
    case ColorStreamGate::Admission::Decode:
      break;
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

  const auto color = pair->getColorFrame();
  // A frame allocated and copied per picture: 0.5 ms at 4K against the
  // decode's 24 (the 2026-09-28 decision).
  std::shared_ptr<ob::VideoFrame> rgb;
  if (options_.yuv) {
    rgb = i420_frame(picture);
  } else {
    const std::size_t row = 3u * picture.width;
    rgb = ob::FrameFactory::createVideoFrame(OB_FRAME_COLOR, OB_FORMAT_RGB,
                                             picture.width, picture.height,
                                             static_cast<std::uint32_t>(row));
    std::uint8_t* out = rgb->getData();
    for (std::uint32_t y = 0; y < picture.height; ++y) {
      std::memcpy(out + y * row, picture.plane[0] + y * picture.stride[0], row);
    }
  }
  ob::FrameHelper::setFrameDeviceTimestampUs(rgb, color->getTimeStampUs());
  rgb->setSystemTimestampUs(color->getSystemTimeStampUs());
  // Not on an I420 frame: the profile would restamp its format as RGB, and
  // the raw path takes its cameras from the profiles at open instead.
  if (options_.rgb_profile != nullptr && !options_.yuv) {
    rgb->setStreamProfile(options_.rgb_profile);
  }
  auto rebuilt = ob::FrameFactory::createFrameSet();
  rebuilt->pushFrame(pair->getDepthFrame());
  rebuilt->pushFrame(rgb);
  sink_(std::move(rebuilt));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
