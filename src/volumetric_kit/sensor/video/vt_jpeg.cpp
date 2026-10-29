// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "vt_jpeg.hpp"

#include <optional>
#include <string>
#include <utility>

#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::sensor::video {
namespace {

// A JPEG's frame header: its size, and whether it is baseline or extended
// 8-bit 4:2:0 in three components, the one layout taken here.
struct JpegFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  bool yuv420 = false;
};

// The frame header, read by walking the markers from SOI; empty for bytes
// that reach a scan or their end before one.
std::optional<JpegFrame> read_frame(const std::uint8_t* d, std::size_t n) {
  if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return std::nullopt;
  std::size_t i = 2;
  while (i + 4 <= n) {
    if (d[i] != 0xFF) return std::nullopt;
    const std::uint8_t marker = d[i + 1];
    if (marker == 0xFF) {  // a fill byte
      ++i;
      continue;
    }
    if (marker == 0xD9 || marker == 0xDA) return std::nullopt;
    const std::size_t length = (std::size_t{d[i + 2]} << 8) | d[i + 3];
    if (length < 2 || length > n - i - 2) return std::nullopt;
    // SOF0 to SOF15, less DHT, JPG and DAC, which share the range.
    if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 &&
        marker != 0xCC) {
      const std::uint8_t* s = d + i + 4;
      if (length < 8 || length < 8 + 3 * std::size_t{s[5]}) {
        return std::nullopt;
      }
      JpegFrame frame;
      frame.height = (std::uint32_t{s[1]} << 8) | s[2];
      frame.width = (std::uint32_t{s[3]} << 8) | s[4];
      frame.yuv420 = (marker == 0xC0 || marker == 0xC1) && s[0] == 8 &&
                     s[5] == 3 && s[7] == 0x22 && s[10] == 0x11 &&
                     s[13] == 0x11;
      return frame;
    }
    i += 2 + length;
  }
  return std::nullopt;
}

}  // namespace

core::Result<std::unique_ptr<VtJpeg>> VtJpeg::open(
    const core::Device& device, const char* who,
    const std::uint32_t* queue_families, std::uint32_t queue_family_count) {
  if (!VTIsHardwareDecodeSupported(kCMVideoCodecType_JPEG)) {
    return core::Status::unsupported(
        std::string(who) + ": VideoToolbox has no hardware JPEG decoder");
  }
  VKC_ASSIGN(PixelBufferImporter pictures,
             PixelBufferImporter::create(device, who, queue_families,
                                         queue_family_count));
  std::unique_ptr<VtJpeg> vt(new VtJpeg(std::move(pictures)));
  vt->who_ = who;
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  vt->max_extent_ = props.limits.maxImageDimension2D;
  return vt;
}

void VtJpeg::stop() noexcept {
  session_.reset();
  format_.reset();
  width_ = height_ = 0;
}

core::Status VtJpeg::start(std::uint32_t width, std::uint32_t height) {
  stop();
  CMVideoFormatDescriptionRef format = nullptr;
  const OSStatus described = CMVideoFormatDescriptionCreate(
      kCFAllocatorDefault, kCMVideoCodecType_JPEG,
      static_cast<std::int32_t>(width), static_cast<std::int32_t>(height),
      nullptr, &format);
  if (described != noErr) return vt_error(who_, "describing a JPEG", described);
  format_.reset(format);
  // Full range, as the JPEG's samples are.
  core::Result<SessionPtr> session =
      open_session(format_.get(), true, who_,
                   "a " + std::to_string(width) + "x" + std::to_string(height) +
                       " JPEG session");
  if (!session) {
    stop();
    return session.status();
  }
  session_ = std::move(session).value();
  width_ = width;
  height_ = height;
  return {};
}

core::Result<DecodedPicture> VtJpeg::decode(const std::uint8_t* data,
                                            std::size_t size) {
  const std::optional<JpegFrame> frame = read_frame(data, size);
  const auto fail = [this](const std::string& why) {
    return core::Status::io_error(std::string(who_) + ": " + why);
  };
  const auto refuse = [this](const std::string& why) {
    return core::Status::unsupported(std::string(who_) + ": " + why);
  };
  if (!frame) return fail("no JPEG frame header ahead of its scan");
  if (frame->width == 0 || frame->height == 0) return fail("a JPEG of no size");
  if (!frame->yuv420) {
    return refuse("the hardware takes baseline 8-bit 4:2:0 JPEGs only");
  }
  const std::string size_text =
      std::to_string(frame->width) + "x" + std::to_string(frame->height);
  // A plane past the extent cannot be a texture (Metal aborts on one: a
  // 16400-wide JPEG on an M4, whose extent is 16384).
  if (frame->width > max_extent_ || frame->height > max_extent_) {
    return refuse("a " + size_text + " JPEG is larger than the device's " +
                  std::to_string(max_extent_) + "-pixel images");
  }
  if (session_ == nullptr || frame->width != width_ ||
      frame->height != height_) {
    VKC_TRY(start(frame->width, frame->height));
  }

  core::Result<PixelBufferPtr> decoded =
      decode_sample(session_.get(), format_.get(), data, size, who_, "a JPEG");
  // A session that fails is started afresh for the next JPEG.
  if (!decoded) {
    stop();
    return decoded.status();
  }
  if (decoded.value() == nullptr) return fail("the JPEG does not decode");

  // The images hold the picture's buffer, ours let go on the way out.
  DecodedPicture out;
  VKC_TRY(pictures_.import(decoded.value().get(), frame->width, frame->height,
                           out.yuv));
  return out;
}

}  // namespace volumetric_kit::sensor::video
