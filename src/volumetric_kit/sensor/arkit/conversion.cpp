// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "conversion.hpp"

#include <sys/sysctl.h>
#include <time.h>

#include <TargetConditionals.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/sensor/apple/core_video_color.hpp"
#include "volumetric_kit/sensor/camera_conventions.hpp"
#include "volumetric_kit/sensor/frame/camera_params.hpp"
#include "volumetric_kit/sensor/frame/vector_types.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"
#include "volumetric_kit/sensor/video/frame_color.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::arkit {
namespace {

std::string size_text(std::size_t width, std::size_t height) {
  return std::to_string(width) + " x " + std::to_string(height);
}

// A buffer's base address locked for reading, for a scope; null where it does
// not lock or names no address.
class ReadLock {
 public:
  explicit ReadLock(CVPixelBufferRef pixels) : pixels_(pixels) {
    locked_ = CVPixelBufferLockBaseAddress(
                  pixels, kCVPixelBufferLock_ReadOnly) == kCVReturnSuccess;
  }
  ~ReadLock() {
    if (locked_) {
      CVPixelBufferUnlockBaseAddress(pixels_, kCVPixelBufferLock_ReadOnly);
    }
  }
  ReadLock(const ReadLock&) = delete;
  ReadLock& operator=(const ReadLock&) = delete;

  const std::uint8_t* base() const {
    return locked_ ? static_cast<const std::uint8_t*>(
                         CVPixelBufferGetBaseAddress(pixels_))
                   : nullptr;
  }

 private:
  CVPixelBufferRef pixels_;
  bool locked_ = false;
};

}  // namespace

core::Status copy_depth(CVPixelBufferRef depth, CVPixelBufferRef confidence,
                        std::uint32_t width, std::uint32_t height,
                        std::uint16_t* depth_out,
                        std::uint8_t* confidence_out) {
  if (depth == nullptr) return core::Status::invalid_argument("no depth map");
  if (CVPixelBufferGetPixelFormatType(depth) !=
      kCVPixelFormatType_DepthFloat32) {
    return core::Status::invalid_argument(
        "a depth map that is not DepthFloat32");
  }
  const std::size_t w = CVPixelBufferGetWidth(depth);
  const std::size_t h = CVPixelBufferGetHeight(depth);
  if (w != width || h != height) {
    return core::Status::invalid_argument("a depth map of " + size_text(w, h) +
                                          ", not the sensor's " +
                                          size_text(width, height));
  }
  if (confidence == nullptr) {
    return core::Status::invalid_argument("no confidence map");
  }
  if (CVPixelBufferGetPixelFormatType(confidence) !=
      kCVPixelFormatType_OneComponent8) {
    return core::Status::invalid_argument(
        "a confidence map that is not OneComponent8");
  }
  const std::size_t cw = CVPixelBufferGetWidth(confidence);
  const std::size_t ch = CVPixelBufferGetHeight(confidence);
  if (cw != width || ch != height) {
    return core::Status::invalid_argument(
        "a confidence map of " + size_text(cw, ch) + ", not the depth's " +
        size_text(width, height));
  }
  const ReadLock depth_lock(depth);
  const ReadLock confidence_lock(confidence);
  const std::uint8_t* src = depth_lock.base();
  const std::uint8_t* levels = confidence_lock.base();
  if (src == nullptr || levels == nullptr) {
    return core::Status::invalid_argument(
        "a depth or confidence map that does "
        "not lock");
  }
  // Each buffer's own row stride: Core Video pads rows, so a row is not
  // `width` samples apart.
  const std::size_t src_stride = CVPixelBufferGetBytesPerRow(depth);
  const std::size_t level_stride = CVPixelBufferGetBytesPerRow(confidence);
  for (std::uint32_t y = 0; y < height; ++y) {
    const auto* row =
        reinterpret_cast<const float*>(src + std::size_t{y} * src_stride);
    std::uint16_t* out = depth_out + std::size_t{y} * width;
    for (std::uint32_t x = 0; x < width; ++x) out[x] = arkit_depth_mm(row[x]);
    if (confidence_out != nullptr) {
      const std::uint8_t* level_row = levels + std::size_t{y} * level_stride;
      std::uint8_t* level_out = confidence_out + std::size_t{y} * width;
      for (std::uint32_t x = 0; x < width; ++x) level_out[x] = level_row[x];
    }
  }
  return {};
}

core::Status check_picture(CVPixelBufferRef picture, std::uint32_t width,
                           std::uint32_t height) {
  if (picture == nullptr) return core::Status::invalid_argument("no picture");
  const OSType format = CVPixelBufferGetPixelFormatType(picture);
  if (format != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange &&
      format != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
    return core::Status::invalid_argument("a picture that is not 8-bit NV12");
  }
  if (CVPixelBufferGetIOSurface(picture) == nullptr) {
    return core::Status::invalid_argument("a picture not on an IOSurface");
  }
  const std::size_t w = CVPixelBufferGetWidth(picture);
  const std::size_t h = CVPixelBufferGetHeight(picture);
  if (w != width || h != height) {
    return core::Status::invalid_argument("a picture of " + size_text(w, h) +
                                          ", not the sensor's " +
                                          size_text(width, height));
  }
  return {};
}

core::Result<PictureColor> picture_color(CVPixelBufferRef picture) {
  core::Result<ColorCodes> tagged = color_codes_of(picture);
  if (!tagged.ok()) {
    return core::Status::invalid_argument("the picture's colour tags: " +
                                          tagged.status().message());
  }
  ColorCodes codes = tagged.value();
  if (codes.matrix == 2) codes.matrix = 6;  // untagged: BT.601, ARKit's
  const std::optional<ColorEncoding> encoding =
      resolve_encoding(codes.transfer, codes.primaries);
  if (!encoding) {
    return core::Status::invalid_argument(
        "colour tags ColorEncoding cannot name: transfer " +
        std::to_string(codes.transfer) + ", primaries " +
        std::to_string(codes.primaries));
  }
  const YcbcrWeights weights = ycbcr_weights(resolve_matrix(
      codes.matrix,
      static_cast<std::uint32_t>(CVPixelBufferGetHeight(picture))));
  PictureColor out;
  out.kr = weights.kr;
  out.kb = weights.kb;
  out.full_range = codes.full_range;
  out.chroma_location = resolve_chroma_location(codes.chroma_location);
  out.encoding = *encoding;
  return out;
}

core::Result<camera::CameraModel> color_camera_of(
    const std::array<float, 9>& intrinsics, std::uint32_t width,
    std::uint32_t height) {
  camera::CameraModel model;
  model.size = {width, height};
  model.intrinsics.fx = intrinsics[0];
  model.intrinsics.fy = intrinsics[4];
  model.intrinsics.cx = intrinsics[6];
  model.intrinsics.cy = intrinsics[7];
  core::Status usable = camera::check_camera_model(model);
  if (!usable.ok()) {
    return core::Status::invalid_argument("ARKit's pinhole: " +
                                          usable.message());
  }
  return model;
}

core::Result<camera::CameraModel> depth_camera_of(
    const camera::CameraModel& color, std::uint32_t width, std::uint32_t height,
    float min_depth, float max_depth) {
  ColorCameraParams params{};
  params.fx = static_cast<float>(color.intrinsics.fx);
  params.fy = static_cast<float>(color.intrinsics.fy);
  params.cx = static_cast<float>(color.intrinsics.cx);
  params.cy = static_cast<float>(color.intrinsics.cy);
  params.width = color.size.width;
  params.height = color.size.height;
  params.cam_to_world = Mat4f(1.0f);
  VKC_ASSIGN(
      const DepthCameraParams depth,
      depth_from_registered_color(params, width, height, min_depth, max_depth));
  camera::CameraModel model;
  model.size = {depth.width, depth.height};
  model.intrinsics = {depth.fx, depth.fy, depth.cx, depth.cy};
  return model;
}

camera::Mat4d color_to_world_of(const std::array<float, 16>& transform) {
  Mat4f gl(1.0f);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) gl[c][r] = transform[std::size_t(c * 4 + r)];
  }
  return camera::Mat4d(cv_from_gl_camera(gl));
}

void ClockMap::start(std::int64_t system_ns, std::int64_t uptime_ns) noexcept {
  offset_ns_ = system_ns - uptime_ns;
}

void ClockMap::remeasure(std::int64_t system_ns,
                         std::int64_t uptime_ns) noexcept {
  const std::int64_t now = system_ns - uptime_ns;
  if (now > offset_ns_) offset_ns_ = now;
}

std::uint64_t ClockMap::to_system_ns(double uptime_s) const noexcept {
  // The uptime in whole ns first, then the offset in integers: a double this
  // far past the epoch steps by 256 ns.
  const std::int64_t ns = std::llround(uptime_s * 1e9) + offset_ns_;
  return ns > 0 ? static_cast<std::uint64_t>(ns) : 0;
}

std::uint64_t missed_periods(double before_s, double now_s,
                             std::uint32_t fps) noexcept {
  const double periods = std::round((now_s - before_s) * fps);
  return periods > 1.0 ? static_cast<std::uint64_t>(periods) - 1 : 0;
}

std::int64_t system_now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::int64_t uptime_now_ns() noexcept {
  // ARFrame.timestamp's clock: the host's uptime, which stops while the
  // device sleeps (mach_absolute_time).
  return static_cast<std::int64_t>(clock_gettime_nsec_np(CLOCK_UPTIME_RAW));
}

std::string device_model() {
#if TARGET_OS_IOS
  const char* key = "hw.machine";  // "iPad16,3"; hw.model is the board's
#else
  const char* key = "hw.model";  // "Mac16,5"; hw.machine is "arm64"
#endif
  std::size_t size = 0;
  if (sysctlbyname(key, nullptr, &size, nullptr, 0) != 0 || size == 0) {
    return "unknown";
  }
  std::vector<char> name(size);
  if (sysctlbyname(key, name.data(), &size, nullptr, 0) != 0) return "unknown";
  return std::string(name.data());
}

}  // namespace volumetric_kit::sensor::arkit
