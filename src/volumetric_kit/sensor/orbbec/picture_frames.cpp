// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "picture_frames.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace volumetric_kit::recon::sensor::orbbec {
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

// The pictures whose frames are alive, each erased by its frame's deleter.
// Never destroyed, since a frame can outlive static destruction.
struct Registry {
  std::mutex mutex;
  std::uint64_t next = 0;                                  // guarded by mutex
  std::unordered_map<std::uint64_t, DecodedPicture> live;  // guarded by mutex
};
Registry& registry() {
  static Registry* r = new Registry();
  return *r;
}

}  // namespace

std::shared_ptr<ob::Frame> picture_frame(const DecodedPicture& picture) {
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
        {
          Registry& r = registry();
          std::lock_guard<std::mutex> lock(r.mutex);
          r.live.erase(p->serial);
        }
        delete p;
      },
      static_cast<std::uint32_t>(sizeof(Payload)));
  const std::uint64_t serial = payload.release()->serial;  // the frame's now
  std::lock_guard<std::mutex> lock(r.mutex);
  r.live.emplace(serial, picture);
  return frame;
}

std::optional<DecodedPicture> device_picture(const ob::Frame& frame) {
  if (frame.getFormat() != kFormat || frame.getDataSize() != sizeof(Payload)) {
    return std::nullopt;
  }
  Payload p;
  std::memcpy(&p, frame.getData(), sizeof(p));
  if (std::memcmp(p.tag, kTag, sizeof(kTag)) != 0) return std::nullopt;
  Registry& r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  const auto found = r.live.find(p.serial);
  if (found == r.live.end()) return std::nullopt;
  return found->second;
}

void place_device_color(const DecodedPicture& picture, YuvImage* image) {
  image->chroma_location = picture.chroma_location;
  image->layout = picture.layout == VideoPixelLayout::Nv12 ? YuvLayout::Nv12
                                                           : YuvLayout::I420;
  if (picture.device != nullptr) {
    image->device = picture.device;
    for (int p = 0; p < 3; ++p) {
      image->offset[p] = picture.offset[p];
      image->stride[p] = picture.stride[p];
    }
    image->queue_family = kQueueFamilyExternal;  // CUDA wrote it
  } else {
    image->image[0] = picture.image[0];
    image->image[1] = picture.image[1];
  }
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

}  // namespace volumetric_kit::recon::sensor::orbbec
