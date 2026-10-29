// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "picture_frames.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace volumetric_kit::sensor::orbbec {
namespace {

// A device picture's frame's bytes: a tag and the picture's serial, plain
// bytes that any copy of the frame may carry without owning anything.
constexpr char kTag[8] = {'v', 'r', 'd', 'e', 'v', 'p', 'i', 'c'};
struct Payload {
  char tag[8];
  std::uint64_t serial;
};
// A variable-size format, so the frame is its bytes rather than a 0x0 image's
// pixels (the SDK refuses UNKNOWN for a colour frame), and one no stream
// here carries, so nothing decodes it either.
constexpr OBFormat kFormat = OB_FORMAT_H264;

// What a live frame keeps: its picture, and the camera's frame it was
// decoded from, when it keeps that.
struct Kept {
  DecodedPicture picture;
  std::shared_ptr<ob::Frame> source;
};

// The pictures whose frames are alive, each erased by its frame's deleter.
// Never destroyed, since a frame can outlive static destruction.
struct Registry {
  using Live = std::unordered_map<std::uint64_t, Kept>;
  std::mutex mutex;
  std::uint64_t next = 0;  // guarded by mutex
  Live live;               // guarded by mutex
};
Registry& registry() {
  static Registry* r = new Registry();
  return *r;
}

// The serial a frame from picture_frame names; empty for any other frame.
std::optional<std::uint64_t> serial_of(const ob::Frame& frame) {
  if (frame.getFormat() != kFormat || frame.getDataSize() != sizeof(Payload)) {
    return std::nullopt;
  }
  Payload p;
  std::memcpy(&p, frame.getData(), sizeof(p));
  if (std::memcmp(p.tag, kTag, sizeof(kTag)) != 0) return std::nullopt;
  return p.serial;
}

}  // namespace

std::shared_ptr<ob::Frame> picture_frame(const DecodedPicture& picture,
                                         std::shared_ptr<ob::Frame> source) {
  Registry& r = registry();
  std::unique_ptr<Payload> payload(new Payload{});
  std::memcpy(payload->tag, kTag, sizeof(kTag));
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    payload->serial = ++r.next;
  }
  auto frame = ob::FrameFactory::createFrameFromBuffer(
      OB_FRAME_COLOR, kFormat, reinterpret_cast<std::uint8_t*>(payload.get()),
      [](std::uint8_t* b) {
        const auto* p = reinterpret_cast<const Payload*>(b);
        // Let go after the lock: releasing the source frame calls the SDK.
        Registry::Live::node_type gone;
        {
          Registry& r = registry();
          std::lock_guard<std::mutex> lock(r.mutex);
          gone = r.live.extract(p->serial);
        }
        delete p;
      },
      static_cast<std::uint32_t>(sizeof(Payload)));
  const std::uint64_t serial = payload.release()->serial;  // the frame's now
  std::lock_guard<std::mutex> lock(r.mutex);
  r.live.emplace(serial, Kept{picture, std::move(source)});
  return frame;
}

std::optional<DecodedPicture> device_picture(const ob::Frame& frame) {
  const std::optional<std::uint64_t> serial = serial_of(frame);
  if (!serial) return std::nullopt;
  Registry& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto found = r.live.find(*serial);
  if (found == r.live.end()) return std::nullopt;
  return found->second.picture;
}

ColorBitstream color_bitstream(const ob::Frame& frame) {
  const std::optional<std::uint64_t> serial = serial_of(frame);
  if (!serial) return {};
  std::shared_ptr<ob::Frame> source;
  {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto found = r.live.find(*serial);
    if (found == r.live.end()) return {};
    source = found->second.source;
  }
  // Its entry, and with it the source, lives as long as `frame`.
  if (source == nullptr || source->getFormat() != OB_FORMAT_MJPG) return {};
  ColorBitstream out;
  out.codec = ColorBitstream::Codec::Jpeg;
  out.data = source->getData();
  out.size = source->getDataSize();
  return out;
}

std::shared_ptr<ob::FrameSet> rebuilt_pair(std::shared_ptr<ob::Frame> depth,
                                           const ob::Frame& source,
                                           std::shared_ptr<ob::Frame> decoded) {
  ob::FrameHelper::setFrameDeviceTimestampUs(decoded, source.getTimeStampUs());
  decoded->setSystemTimestampUs(source.getSystemTimeStampUs());
  auto pair = ob::FrameFactory::createFrameSet();
  pair->pushFrame(std::move(depth));
  pair->pushFrame(std::move(decoded));
  return pair;
}

}  // namespace volumetric_kit::sensor::orbbec
