// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The test pattern as ARKit's feed: the same Capture a camera's frame is,
// made here. Frame n's depth is testing::pattern_depth as float metres and
// its picture testing::test_pattern in one of 4 IOSurface '420f' buffers,
// tagged as ARKit tags its own; a tick finding all 4 held is skipped.

#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "feeds.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"

namespace volumetric_kit::sensor::arkit {
namespace {

// ARKit's depth size, at which the pattern's band is whole.
constexpr std::uint32_t kDepthWidth = 256;
constexpr std::uint32_t kDepthHeight = 192;
// The pictures in flight: a tick finding them all held is skipped.
constexpr int kPictures = 4;

// A Core Foundation object released with this.
template <typename T>
class Cf {
 public:
  Cf() = default;
  explicit Cf(T ref) : ref_(ref) {}
  ~Cf() {
    if (ref_ != nullptr) CFRelease(ref_);
  }
  Cf(const Cf&) = delete;
  Cf& operator=(const Cf&) = delete;
  T get() const { return ref_; }
  T* out() { return &ref_; }
  void reset(T ref) {
    if (ref_ != nullptr) CFRelease(ref_);
    ref_ = ref;
  }

 private:
  T ref_ = nullptr;
};

Cf<CFMutableDictionaryRef> dictionary() {
  return Cf<CFMutableDictionaryRef>(CFDictionaryCreateMutable(
      kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks));
}

void set_int(CFMutableDictionaryRef dict, CFStringRef key, int value) {
  const Cf<CFNumberRef> number(
      CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value));
  CFDictionarySetValue(dict, key, number.get());
}

class PatternFeed final : public Feed {
 public:
  PatternFeed(Driver& driver, const ArkitSensor::Options& options,
              dispatch_queue_t queue)
      : driver_(driver),
        width_(options.color_width),
        height_(options.color_height),
        fps_(options.fps),
        queue_(queue) {}
  ~PatternFeed() override { stop(); }
  PatternFeed(const PatternFeed&) = delete;
  PatternFeed& operator=(const PatternFeed&) = delete;

  // The pools and the depth buffers; OutOfMemory where Core Video makes none.
  core::Status make_buffers() {
    Cf<CFMutableDictionaryRef> pool_attrs = dictionary();
    Cf<CFMutableDictionaryRef> attrs = dictionary();
    Cf<CFMutableDictionaryRef> surface = dictionary();
    if (pool_attrs.get() == nullptr || attrs.get() == nullptr ||
        surface.get() == nullptr) {
      return core::Status::out_of_memory("ArkitSensor: the pattern's pools");
    }
    set_int(pool_attrs.get(), kCVPixelBufferPoolMinimumBufferCountKey,
            kPictures);
    set_int(attrs.get(), kCVPixelBufferPixelFormatTypeKey,
            static_cast<int>(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange));
    set_int(attrs.get(), kCVPixelBufferWidthKey, static_cast<int>(width_));
    set_int(attrs.get(), kCVPixelBufferHeightKey, static_cast<int>(height_));
    CFDictionarySetValue(attrs.get(), kCVPixelBufferIOSurfacePropertiesKey,
                         surface.get());
    CFDictionarySetValue(attrs.get(), kCVPixelBufferMetalCompatibilityKey,
                         kCFBooleanTrue);
    if (CVPixelBufferPoolCreate(kCFAllocatorDefault, pool_attrs.get(),
                                attrs.get(),
                                pictures_.out()) != kCVReturnSuccess) {
      return core::Status::out_of_memory("ArkitSensor: the pattern's pictures");
    }
    threshold_.reset(CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks));
    if (threshold_.get() == nullptr) {
      return core::Status::out_of_memory("ArkitSensor: the pattern's pictures");
    }
    set_int(threshold_.get(), kCVPixelBufferPoolAllocationThresholdKey,
            kPictures);
    if (CVPixelBufferCreate(kCFAllocatorDefault, kDepthWidth, kDepthHeight,
                            kCVPixelFormatType_DepthFloat32, nullptr,
                            depth_.out()) != kCVReturnSuccess ||
        CVPixelBufferCreate(kCFAllocatorDefault, kDepthWidth, kDepthHeight,
                            kCVPixelFormatType_OneComponent8, nullptr,
                            confidence_.out()) != kCVReturnSuccess) {
      return core::Status::out_of_memory("ArkitSensor: the pattern's depth");
    }
    return {};
  }

  // Frame 0, for the sensor's cameras, on the caller's thread: no timer runs
  // yet, so the feed's one thread is this one.
  void deliver_first() { make(uptime_now_ns() * 1e-9); }

  core::Status start() override {
    if (timer_ != nil) return {};
    t0_s_ = uptime_now_ns() * 1e-9;
    ticks_ = 0;
    timer_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
    if (timer_ == nil) {
      return core::Status::out_of_memory("ArkitSensor: the pattern's timer");
    }
    const auto period = static_cast<std::uint64_t>(1e9 / fps_);
    dispatch_source_set_timer(
        timer_,
        dispatch_time(DISPATCH_TIME_NOW, static_cast<std::int64_t>(period)),
        period, period / 20);
    // The source's own count: ticks the queue was too busy to run in one
    // handler count each, as periods the frames skip. The handler reads the
    // source through the feed, so it holds no reference to it: stop cancels
    // it, and no handler runs after.
    PatternFeed* self = this;
    dispatch_source_set_event_handler(timer_, ^{
      self->ticks_ += dispatch_source_get_data(self->timer_);
      self->make(self->t0_s_ + static_cast<double>(self->ticks_) / self->fps_);
    });
    dispatch_resume(timer_);
    return {};
  }

  void stop() noexcept override {
    if (timer_ == nil) return;
    dispatch_source_cancel(timer_);
    // A handler already running ends before this returns.
    dispatch_sync(queue_, ^{
                  });
    timer_ = nil;
  }

 private:
  // One frame at `timestamp_s`, numbered as the driver will number it; none
  // where all 4 pictures are held.
  void make(double timestamp_s) {
    CVPixelBufferRef picture = nullptr;
    if (CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
            kCFAllocatorDefault, pictures_.get(), threshold_.get(), &picture) !=
            kCVReturnSuccess ||
        picture == nullptr) {
      return;  // skipped: the next frame's gap counts it missed
    }
    const Cf<CVPixelBufferRef> held(picture);
    const auto index = static_cast<std::uint32_t>(driver_.next_sequence());
    if (!write_picture(picture, index) || !write_depth(index)) return;
    CVBufferSetAttachment(picture, kCVImageBufferYCbCrMatrixKey,
                          kCVImageBufferYCbCrMatrix_ITU_R_601_4,
                          kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(picture, kCVImageBufferTransferFunctionKey,
                          kCVImageBufferTransferFunction_ITU_R_709_2,
                          kCVAttachmentMode_ShouldPropagate);
    CVBufferSetAttachment(picture, kCVImageBufferColorPrimariesKey,
                          kCVImageBufferColorPrimaries_ITU_R_709_2,
                          kCVAttachmentMode_ShouldPropagate);
    Capture capture;
    capture.depth = depth_.get();
    capture.confidence = confidence_.get();
    capture.picture = picture;
    // A centred pinhole, column-major as ARCamera.intrinsics.
    const float f = 0.75f * static_cast<float>(width_);
    const float cx = 0.5f * static_cast<float>(width_ - 1);
    const float cy = 0.5f * static_cast<float>(height_ - 1);
    capture.intrinsics = {f, 0.0f, 0.0f, 0.0f, f, 0.0f, cx, cy, 1.0f};
    capture.transform = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    capture.timestamp_s = timestamp_s;
    capture.tracking = ArkitTracking::Normal;
    driver_.deliver(capture, static_cast<std::uint64_t>(system_now_ns()));
  }

  // Every byte of the picture: its background moves with the index.
  bool write_picture(CVPixelBufferRef picture, std::uint32_t index) const {
    const testing::HostNv12 p = testing::test_pattern(width_, height_, index);
    if (CVPixelBufferLockBaseAddress(picture, 0) != kCVReturnSuccess) {
      return false;
    }
    const std::uint8_t* planes[2] = {p.luma.data(), p.chroma.data()};
    const std::size_t row_bytes[2] = {width_,
                                      2 * std::size_t{(width_ + 1) / 2}};
    const std::uint32_t rows[2] = {height_, (height_ + 1) / 2};
    bool written = true;
    for (std::size_t k = 0; k < 2; ++k) {
      auto* base = static_cast<std::uint8_t*>(
          CVPixelBufferGetBaseAddressOfPlane(picture, k));
      if (base == nullptr) {
        written = false;
        break;
      }
      const std::size_t stride = CVPixelBufferGetBytesPerRowOfPlane(picture, k);
      for (std::uint32_t r = 0; r < rows[k]; ++r) {
        std::copy_n(planes[k] + r * row_bytes[k], row_bytes[k],
                    base + r * stride);
      }
    }
    CVPixelBufferUnlockBaseAddress(picture, 0);
    return written;
  }

  // pattern_depth as ARKit hands depth over: float metres, exact at 1 and
  // 2 m, and its levels.
  bool write_depth(std::uint32_t index) const {
    const testing::PatternDepth d =
        testing::pattern_depth(kDepthWidth, kDepthHeight, index);
    CVPixelBufferRef depth = depth_.get();
    CVPixelBufferRef levels = confidence_.get();
    if (CVPixelBufferLockBaseAddress(depth, 0) != kCVReturnSuccess) {
      return false;
    }
    if (CVPixelBufferLockBaseAddress(levels, 0) != kCVReturnSuccess) {
      CVPixelBufferUnlockBaseAddress(depth, 0);
      return false;
    }
    auto* depth_base =
        static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(depth));
    auto* level_base =
        static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(levels));
    const bool written = depth_base != nullptr && level_base != nullptr;
    if (written) {
      const std::size_t depth_stride = CVPixelBufferGetBytesPerRow(depth);
      const std::size_t level_stride = CVPixelBufferGetBytesPerRow(levels);
      for (std::uint32_t y = 0; y < kDepthHeight; ++y) {
        auto* row = reinterpret_cast<float*>(depth_base + y * depth_stride);
        std::uint8_t* level_row = level_base + y * level_stride;
        for (std::uint32_t x = 0; x < kDepthWidth; ++x) {
          const std::size_t at = std::size_t{y} * kDepthWidth + x;
          row[x] = static_cast<float>(d.depth[at]) / 1000.0f;
          level_row[x] = d.confidence[at];
        }
      }
    }
    CVPixelBufferUnlockBaseAddress(levels, 0);
    CVPixelBufferUnlockBaseAddress(depth, 0);
    return written;
  }

  Driver& driver_;
  const std::uint32_t width_;
  const std::uint32_t height_;
  const std::uint32_t fps_;
  dispatch_queue_t queue_;
  dispatch_source_t timer_ = nil;
  // The queue's alone, once the timer runs:
  double t0_s_ = 0.0;
  std::uint64_t ticks_ = 0;
  Cf<CVPixelBufferPoolRef> pictures_;
  Cf<CFMutableDictionaryRef> threshold_;
  Cf<CVPixelBufferRef> depth_;
  Cf<CVPixelBufferRef> confidence_;
};

}  // namespace

core::Result<std::unique_ptr<Feed>> open_pattern_feed(
    Driver& driver, const ArkitSensor::Options& options,
    dispatch_queue_t queue) {
  auto feed = std::make_unique<PatternFeed>(driver, options, queue);
  VKC_TRY(feed->make_buffers());
  feed->deliver_first();
  return std::unique_ptr<Feed>(std::move(feed));
}

}  // namespace volumetric_kit::sensor::arkit
