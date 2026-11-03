// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ArkitSensor's driver (arkit/driver.hpp, sensor_arkit_units) over a scripted
// feed of captures made here -- IOSurface pictures tagged as a camera tags
// them, ARKit's float depth and its levels -- as ARKit's feed and the pattern
// hand them over, on the device the pictures are imported on: the cameras
// from the first frame (info), and the open's refusals; the frame contract
// (rgbd_breach) over 120 frames with a withheld frame, refused pictures, a
// frame without the map, queue drops at depths 1 and 3 and drain oldest first;
// the counters; the sequences; a 3-period gap counted as 2 missed; each
// frame's colour from its tags, its depth and levels, its cameras, pose,
// exposure and noise; the picture's own buffer and how long it is held; the
// clock; a feeding thread; ARKit's failure; and the option checks.
//
// Skips where no Vulkan device is present, unless VKC_REQUIRE_VULKAN_DEVICE
// says this machine has one.

#include <CoreFoundation/CoreFoundation.h>
#include <CoreVideo/CoreVideo.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "driver.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/apple/pixel_buffer_importer.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"
#include "volumetric_kit/sensor/camera_conventions.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/testing/nv12_pixel_buffer.hpp"
#include "volumetric_kit/sensor/testing/rgbd_contract.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"
#include "volumetric_kit/sensor/video/frame_color.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace arkit = volumetric_kit::sensor::arkit;
namespace testing = volumetric_kit::sensor::testing;

namespace {

using Gpu = vkc::test::GpuContext;

constexpr std::uint32_t kW = 320;  // the pictures: small, as the driver
constexpr std::uint32_t kH = 240;  // takes any size its options name
constexpr std::uint32_t kDepthW = 256;
constexpr std::uint32_t kDepthH = 192;
constexpr double kPeriod = 1.0 / 60;

const sensor::ColorCodes kArkitTags{6, 1, 1, true, 0};  // 601, 709, full
const sensor::ColorCodes kHlg{9, 18, 9, true, 0};

// A buffer released with this.
struct Pixels {
  CVPixelBufferRef ref = nullptr;
  Pixels() = default;
  explicit Pixels(CVPixelBufferRef r) : ref(r) {}
  ~Pixels() {
    if (ref != nullptr) CVPixelBufferRelease(ref);
  }
  Pixels(const Pixels&) = delete;
  Pixels& operator=(const Pixels&) = delete;
  Pixels(Pixels&& o) noexcept : ref(o.ref) { o.ref = nullptr; }
  Pixels& operator=(Pixels&& o) noexcept {
    if (this != &o) {
      if (ref != nullptr) CVPixelBufferRelease(ref);
      ref = o.ref;
      o.ref = nullptr;
    }
    return *this;
  }
};

Pixels picture(const sensor::ColorCodes& codes, std::uint32_t width = kW,
               std::uint32_t height = kH, std::uint32_t index = 0) {
  vkc::Result<CVPixelBufferRef> made =
      testing::pixel_buffer(testing::test_pattern(width, height, index), codes);
  return Pixels(made.ok() ? made.value() : nullptr);
}

// ARKit's depth map and levels: pattern_depth as float metres, and its plane.
struct DepthMaps {
  Pixels depth;
  Pixels confidence;
  testing::PatternDepth pattern = testing::pattern_depth(kDepthW, kDepthH, 5);

  DepthMaps() {
    CVPixelBufferRef d = nullptr;
    CVPixelBufferRef c = nullptr;
    CVPixelBufferCreate(kCFAllocatorDefault, kDepthW, kDepthH,
                        kCVPixelFormatType_DepthFloat32, nullptr, &d);
    CVPixelBufferCreate(kCFAllocatorDefault, kDepthW, kDepthH,
                        kCVPixelFormatType_OneComponent8, nullptr, &c);
    depth = Pixels(d);
    confidence = Pixels(c);
    CVPixelBufferLockBaseAddress(d, 0);
    CVPixelBufferLockBaseAddress(c, 0);
    auto* db = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(d));
    auto* cb = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(c));
    for (std::uint32_t y = 0; y < kDepthH; ++y) {
      auto* row =
          reinterpret_cast<float*>(db + y * CVPixelBufferGetBytesPerRow(d));
      std::uint8_t* levels = cb + y * CVPixelBufferGetBytesPerRow(c);
      for (std::uint32_t x = 0; x < kDepthW; ++x) {
        const std::size_t at = std::size_t{y} * kDepthW + x;
        row[x] = static_cast<float>(pattern.depth[at]) / 1000.0f;
        levels[x] = pattern.confidence[at];
      }
    }
    CVPixelBufferUnlockBaseAddress(c, 0);
    CVPixelBufferUnlockBaseAddress(d, 0);
  }
};

// ARKit's pinhole for a kW x kH picture, column-major.
std::array<float, 9> intrinsics(float fx = 250.0f) {
  return {fx, 0.0f, 0.0f, 0.0f, 251.5f, 0.0f, 161.25f, 118.75f, 1.0f};
}

std::array<float, 16> identity() {
  return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

arkit::Capture capture(
    const DepthMaps& maps, CVPixelBufferRef pic, double t,
    sensor::ArkitTracking tracking = sensor::ArkitTracking::Normal) {
  arkit::Capture c;
  c.depth = maps.depth.ref;
  c.confidence = maps.confidence.ref;
  c.picture = pic;
  c.intrinsics = intrinsics();
  c.transform = identity();
  c.timestamp_s = t;
  c.tracking = tracking;
  return c;
}

sensor::ArkitSensor::Options options(const Gpu& gpu) {
  sensor::ArkitSensor::Options o;
  o.source = sensor::ArkitSource::Camera;
  o.id = "ipad";
  o.color_width = kW;
  o.color_height = kH;
  o.fps = 60;
  o.device = &gpu.device;
  return o;
}

// A driver opened on `maps` and `pic`'s first capture.
std::unique_ptr<arkit::Driver> opened(const sensor::ArkitSensor::Options& o,
                                      const DepthMaps& maps,
                                      CVPixelBufferRef pic) {
  vkc::Result<std::unique_ptr<arkit::Driver>> made =
      arkit::Driver::create(o, "ARKit, test");
  if (!made.ok()) return nullptr;
  std::unique_ptr<arkit::Driver> driver = std::move(made).value();
  driver->deliver(capture(maps, pic, 1.0), 1);
  if (!driver->wait_opened(std::chrono::milliseconds(0)).ok()) return nullptr;
  return driver;
}

std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(arkit::system_now_ns());
}

int test_info(const Gpu& gpu) {
  DepthMaps maps;
  const Pixels pic = picture(kArkitTags);
  CHECK(pic.ref != nullptr);
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, pic.ref);
  CHECK(driver != nullptr);
  const sensor::SensorInfo& info = driver->info();
  CHECK(info.id == "ipad");
  CHECK(info.model == "ARKit, test");
  CHECK(info.color && info.depth);
  CHECK(info.color->size.width == kW && info.color->size.height == kH);
  CHECK(info.color->intrinsics.fx == 250.0 &&
        info.color->intrinsics.fy == 251.5);
  CHECK(info.color->intrinsics.cx == 161.25 &&
        info.color->intrinsics.cy == 118.75);
  sensor::ColorCameraParams params{
      250.0f, 251.5f, 161.25f, 118.75f, kW, kH, sensor::Mat4f(1.0f)};
  const vkc::Result<sensor::DepthCameraParams> want =
      sensor::depth_from_registered_color(params, kDepthW, kDepthH, 0.15f,
                                          5.0f);
  CHECK(want.ok());
  CHECK(info.depth->size.width == kDepthW &&
        info.depth->size.height == kDepthH);
  CHECK(info.depth->intrinsics.fx == double{want->fx});
  CHECK(info.depth->intrinsics.cx == double{want->cx});
  CHECK(info.depth->intrinsics.cy == double{want->cy});
  CHECK(info.depth_to_color == sensor::camera::Mat4d(1.0));
  CHECK(info.role == sensor::SyncRole::FreeRun);
  CHECK(info.clock == sensor::ClockDomain::Host);
  CHECK(info.pose == sensor::PoseSource::Tracked);
  CHECK(info.fps == 60);
  CHECK(!info.color_controls);

  // Frames before the start are let go, uncounted.
  driver->deliver(capture(maps, pic.ref, 1.1), 2);
  CHECK(driver->begin().ok());
  const sensor::SensorStats s = driver->stats();
  CHECK(s.received == 0 && s.delivered == 0 && s.dropped == 0 && s.failed == 0);
  vkc::Result<std::optional<sensor::RgbdFrame>> none = driver->poll();
  CHECK(none.ok() && !none.value());
  return 0;
}

int test_open_refusals(const Gpu& gpu) {
  DepthMaps maps;
  // The first frame's picture is not the format's size.
  {
    const Pixels wrong = picture(kArkitTags, kW, kH - 16);
    auto made = arkit::Driver::create(options(gpu), "ARKit, test");
    CHECK(made.ok());
    (*made)->deliver(capture(maps, wrong.ref, 1.0), 1);
    const vkc::Status s = (*made)->wait_opened(std::chrono::milliseconds(0));
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message() ==
          "ArkitSensor: ARKit's first frame: a picture of 320 x 224, not the "
          "format's 320 x 240");
  }
  // Nothing with depth arrives: a frame without the map does not open it.
  {
    const Pixels pic = picture(kArkitTags);
    auto made = arkit::Driver::create(options(gpu), "ARKit, test");
    CHECK(made.ok());
    arkit::Capture no_map = capture(maps, pic.ref, 1.0);
    no_map.depth = nullptr;
    (*made)->deliver(no_map, 1);
    const vkc::Status s = (*made)->wait_opened(std::chrono::milliseconds(20));
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message() ==
          "ArkitSensor: no frame with depth from ARKit within 20 ms");
  }
  // The session failed first: its message.
  {
    auto made = arkit::Driver::create(options(gpu), "ARKit, test");
    CHECK(made.ok());
    (*made)->fail("Required sensor unavailable.");
    const vkc::Status s = (*made)->wait_opened(std::chrono::milliseconds(0));
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message() ==
          "ArkitSensor: ARKit's session failed: Required "
          "sensor unavailable.");
  }
  return 0;
}

// Every frame held, oldest first, appended to `frames`.
int take_all(arkit::Driver& driver, std::vector<sensor::RgbdFrame>& frames) {
  CHECK(driver.drain(&frames).ok());
  return 0;
}

int take_newest(arkit::Driver& driver, std::vector<sensor::RgbdFrame>& frames) {
  vkc::Result<std::optional<sensor::RgbdFrame>> got = driver.poll();
  CHECK(got.ok());
  if (got.value()) frames.push_back(std::move(*got.value()));
  return 0;
}

int test_contract(const Gpu& gpu) {
  DepthMaps maps;
  std::vector<Pixels> pics;
  for (std::uint32_t i = 0; i < 4; ++i) {
    pics.push_back(picture(kArkitTags, kW, kH, i));
    CHECK(pics.back().ref != nullptr);
  }
  const Pixels hlg = picture(kHlg);
  const Pixels small = picture(kArkitTags, kW, kH - 16);
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, pics[0].ref);
  CHECK(driver != nullptr);

  // Start 1, at depth 1: one poll each frame, but where two arrive first (the
  // older dropped); a withheld frame, two refused, one without the map, and a
  // 3-period gap.
  CHECK(driver->begin().ok());
  std::vector<sensor::RgbdFrame> frames;
  double t = 10.0;
  std::uint64_t captures = 0;
  const auto send = [&](CVPixelBufferRef pic,
                        sensor::ArkitTracking tracking =
                            sensor::ArkitTracking::Normal) {
    t += kPeriod;
    driver->deliver(capture(maps, pic, t, tracking), now_ns());
    ++captures;
  };
  for (int i = 0; i < 60; ++i) {
    switch (i) {
      case 10:
        send(pics[0].ref, sensor::ArkitTracking::Relocalizing);
        break;
      case 20:
        send(hlg.ref);
        break;
      case 21:
        send(small.ref);
        break;
      case 30: {
        arkit::Capture no_map = capture(maps, pics[1].ref, t + kPeriod);
        no_map.depth = nullptr;
        t += kPeriod;
        driver->deliver(no_map, now_ns());
        break;
      }
      case 40:
        t += 2 * kPeriod;  // two periods ARKit skipped
        send(pics[2].ref);
        break;
      case 50:
        send(pics[3].ref);  // dropped: a newer one arrives before the poll
        send(pics[0].ref);
        break;
      default:
        send(pics[std::size_t(i) % 4].ref);
    }
    if (take_newest(*driver, frames) != 0) return 1;
  }
  const sensor::SensorStats s1 = driver->stats();
  const sensor::ArkitStatus st1 = driver->status();
  CHECK(testing::rgbd_breach(driver->info(), frames, s1).empty());
  // 60 captures with the map (59 iterations, two at 50), and one without.
  CHECK(captures == 60);
  CHECK(s1.received == 60);
  CHECK(st1.withheld == 1 && st1.refused == 2 && st1.missed == 2);
  CHECK(st1.last_refusal ==
        "a picture of 320 x 224, not the sensor's 320 x 240");
  CHECK(s1.failed == st1.withheld + st1.refused);
  CHECK(s1.dropped == 1);
  CHECK(s1.delivered == frames.size());
  CHECK(s1.delivered + s1.dropped + s1.failed == s1.received);  // none held
  // Sequences: 0 up, a gap at each frame withheld (10), refused (20, 21) or
  // dropped (49), none for the frame without the map (between 29 and 30).
  std::vector<std::uint64_t> want;
  for (std::uint64_t n = 0; n < 60; ++n) {
    if (n != 10 && n != 20 && n != 21 && n != 49) want.push_back(n);
  }
  CHECK(frames.size() == want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    CHECK(frames[i].sequence == want[i]);
  }
  driver->end();

  // Start 2, at depth 3: five arrive, the two oldest dropped, the three
  // drained oldest first; sequences from 0 again.
  CHECK(driver->set_queue_depth(3).ok());
  CHECK(driver->begin().ok());
  CHECK(driver->set_queue_depth(2).domain() ==
        vkc::Status::Code::InvalidArgument);  // once started
  std::vector<sensor::RgbdFrame> second;
  for (int round = 0; round < 12; ++round) {
    for (int k = 0; k < 5; ++k) send(pics[std::size_t(k) % 4].ref);
    if (take_all(*driver, second) != 0) return 1;
  }
  const sensor::SensorStats s2 = driver->stats();
  CHECK(testing::rgbd_breach(driver->info(), second, s2).empty());
  CHECK(s2.received == 60 && s2.dropped == 24 && s2.delivered == 36);
  CHECK(s2.failed == 0 && driver->status().withheld == 0);
  CHECK(second.front().sequence == 2);
  for (std::size_t i = 0; i < second.size(); ++i) {
    CHECK(second[i].sequence == 5 * (i / 3) + 2 + i % 3);
  }
  driver->end();
  return 0;
}

int test_frame(const Gpu& gpu) {
  DepthMaps maps;
  const Pixels p601 = picture(kArkitTags);
  const Pixels p709 = picture({1, 1, 1, true, 0});
  const Pixels untagged = picture({2, 2, 2, true, 0});
  const Pixels video = picture({6, 1, 1, false, 0});
  const Pixels centred = picture(kArkitTags);
  CVBufferSetAttachment(centred.ref, kCVImageBufferChromaLocationTopFieldKey,
                        kCVImageBufferChromaLocation_Center,
                        kCVAttachmentMode_ShouldPropagate);
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, p601.ref);
  CHECK(driver != nullptr);
  CHECK(driver->begin().ok());

  // One frame of `pic`, at `fx` and posed by `transform`.
  const auto one = [&](CVPixelBufferRef pic, float fx = 250.0f,
                       std::array<float, 16> transform =
                           identity()) -> std::optional<sensor::RgbdFrame> {
    arkit::Capture c = capture(maps, pic, 20.0);
    c.intrinsics = intrinsics(fx);
    c.transform = transform;
    c.exposure = {1.0f / 120, 400.0f, -0.5f};
    driver->deliver(c, 123);
    vkc::Result<std::optional<sensor::RgbdFrame>> got = driver->poll();
    if (!got.ok() || !got.value()) return std::nullopt;
    return std::move(*got.value());
  };
  const sensor::YcbcrWeights w601 =
      sensor::ycbcr_weights(sensor::VideoColorMatrix::Bt601);
  const sensor::YcbcrWeights w709 =
      sensor::ycbcr_weights(sensor::VideoColorMatrix::Bt709);

  const std::optional<sensor::RgbdFrame> f = one(p601.ref);
  CHECK(f);
  // The picture: NV12's two plane images at its size, no copy.
  CHECK(f->color.layout == sensor::YuvLayout::Nv12);
  CHECK(f->color.image[0] != nullptr && f->color.image[1] != nullptr);
  CHECK(f->color.device == nullptr && f->color.host == nullptr);
  CHECK(f->color.width == kW && f->color.height == kH);
  CHECK(f->color.kr == w601.kr && f->color.kb == w601.kb);
  CHECK(f->color.full_range);
  CHECK(f->color.chroma_location == sensor::ChromaLocation::Left);
  CHECK(f->color_encoding == *sensor::resolve_encoding(1, 1));
  CHECK(sensor::pixel_buffer_of(f->color) == p601.ref);
  // The depth: millimetres, its levels as given, at the options' floor.
  CHECK(f->depth != nullptr && f->depth_confidence != nullptr);
  CHECK(f->min_confidence == 2);
  CHECK(f->metres_per_unit == 0.001f);
  CHECK(std::memcmp(f->depth, maps.pattern.depth.data(),
                    maps.pattern.depth.size() * 2) == 0);
  CHECK(std::memcmp(f->depth_confidence, maps.pattern.confidence.data(),
                    maps.pattern.confidence.size()) == 0);
  CHECK(testing::depth_index(f->depth, kDepthW, kDepthH) == 5u);
  CHECK(f->min_depth == 0.15f && f->max_depth == 5.0f);
  CHECK(f->depth_noise == sensor::kArkitDepthNoise);
  CHECK(f->color_exposure.duration_s == 1.0f / 120);
  CHECK(f->color_exposure.iso == 400.0f && f->color_exposure.bias_ev == -0.5f);
  CHECK(f->depth_to_color == sensor::camera::Mat4d(1.0));
  CHECK(f->arrival_ns == 123);
  CHECK(f->pixels != nullptr);

  {
    const std::optional<sensor::RgbdFrame> g = one(p709.ref);
    CHECK(g && g->color.kr == w709.kr && g->color.kb == w709.kb);
  }
  {
    const std::optional<sensor::RgbdFrame> g = one(untagged.ref);
    CHECK(g && g->color.kr == w601.kr && g->color.kb == w601.kb);
    CHECK(g->color_encoding == *sensor::resolve_encoding(2, 2));
  }
  {
    const std::optional<sensor::RgbdFrame> g = one(video.ref);
    CHECK(g && !g->color.full_range);
  }
  {
    const std::optional<sensor::RgbdFrame> g = one(centred.ref);
    CHECK(g && g->color.chroma_location == sensor::ChromaLocation::Center);
  }
  {
    // Each frame's own pinhole, which ARKit refines, and its depth camera
    // registered to it; its own pose, in this package's axes.
    const std::array<float, 16> moved = {1, 0, 0, 0, 0,    1,  0, 0,
                                         0, 0, 1, 0, 0.5f, -1, 2, 1};
    const std::optional<sensor::RgbdFrame> g = one(p601.ref, 252.0f, moved);
    CHECK(g && g->color_camera.intrinsics.fx == 252.0);
    CHECK(g->color_camera.size.width == kW);
    CHECK(std::fabs(g->depth_camera.intrinsics.fx - 252.0 * kDepthW / kW) <
          1e-4);
    CHECK(g->color_to_world == arkit::color_to_world_of(moved));
    CHECK(g->color_to_world[1][1] == -1.0 && g->color_to_world[2][2] == -1.0);
    CHECK(g->color_to_world[3][0] == 0.5 && g->color_to_world[3][1] == -1.0);
  }
  driver->end();

  // With a floor of 0: every sample kept, and no plane.
  sensor::ArkitSensor::Options all = options(gpu);
  all.min_confidence = 0;
  const std::unique_ptr<arkit::Driver> keeping = opened(all, maps, p601.ref);
  CHECK(keeping != nullptr && keeping->begin().ok());
  keeping->deliver(capture(maps, p601.ref, 30.0), 1);
  vkc::Result<std::optional<sensor::RgbdFrame>> kept = keeping->poll();
  CHECK(kept.ok() && kept.value());
  CHECK(kept.value()->depth_confidence == nullptr);
  CHECK(kept.value()->min_confidence == 0);
  CHECK(testing::rgbd_breach(keeping->info(), {*kept.value()}, keeping->stats())
            .empty());
  return 0;
}

int test_holds(const Gpu& gpu) {
  DepthMaps maps;
  const Pixels pic = picture(kArkitTags);
  std::vector<Pixels> others;
  for (std::uint32_t i = 0; i < 3; ++i) others.push_back(picture(kArkitTags));
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, pic.ref);
  CHECK(driver != nullptr);
  CHECK(driver->begin().ok());
  const CFIndex alone = CFGetRetainCount(pic.ref);
  driver->deliver(capture(maps, pic.ref, 40.0), 1);
  CHECK(CFGetRetainCount(pic.ref) == alone + 1);  // held in the queue
  std::optional<sensor::RgbdFrame> kept;
  {
    vkc::Result<std::optional<sensor::RgbdFrame>> got = driver->poll();
    CHECK(got.ok() && got.value());
    kept = std::move(*got.value());
  }
  CHECK(sensor::pixel_buffer_of(kept->color) == pic.ref);
  CHECK(CFGetRetainCount(pic.ref) == alone + 1);  // now the frame's images
  // Kept past three polls of other pictures: still held, still its own.
  double t = 40.0;
  for (const Pixels& other : others) {
    t += kPeriod;
    driver->deliver(capture(maps, other.ref, t), 2);
    vkc::Result<std::optional<sensor::RgbdFrame>> got = driver->poll();
    CHECK(got.ok() && got.value());
    CHECK(sensor::pixel_buffer_of(got.value()->color) == other.ref);
  }
  CHECK(CFGetRetainCount(pic.ref) == alone + 1);
  CHECK(sensor::pixel_buffer_of(kept->color) == pic.ref);
  // A copy holds it too; the last copy gone, the camera has it back.
  std::optional<sensor::RgbdFrame> copy = kept;
  kept.reset();
  CHECK(CFGetRetainCount(pic.ref) == alone + 1);
  copy.reset();
  CHECK(CFGetRetainCount(pic.ref) == alone);
  // A frame let go by a stop gives it back as well.
  driver->deliver(capture(maps, pic.ref, t + kPeriod), 3);
  CHECK(CFGetRetainCount(pic.ref) == alone + 1);
  driver->end();
  CHECK(CFGetRetainCount(pic.ref) == alone);
  return 0;
}

int test_clock(const Gpu& gpu) {
  DepthMaps maps;
  const Pixels pic = picture(kArkitTags);
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, pic.ref);
  CHECK(driver != nullptr);
  const std::int64_t before = arkit::system_now_ns() - arkit::uptime_now_ns();
  CHECK(driver->begin().ok());
  const std::int64_t after = arkit::system_now_ns() - arkit::uptime_now_ns();
  std::vector<sensor::RgbdFrame> frames;
  const double t0 = static_cast<double>(arkit::uptime_now_ns()) * 1e-9;
  for (int i = 0; i < 3; ++i) {
    driver->deliver(capture(maps, pic.ref, t0 + i * kPeriod), now_ns());
    if (take_newest(*driver, frames) != 0) return 1;
  }
  CHECK(frames.size() == 3);
  // Each ARKit time on the system clock by one offset, measured at the start.
  const auto offset = [&](std::size_t i) {
    return static_cast<std::int64_t>(frames[i].timestamp_ns) -
           std::llround((t0 + static_cast<double>(i) * kPeriod) * 1e9);
  };
  CHECK(offset(0) == offset(1) && offset(1) == offset(2));
  CHECK(std::llabs(offset(0) - (before + after) / 2) < 50'000'000);
  CHECK(frames[0].timestamp_ns < frames[1].timestamp_ns);
  // Its arrival on the system clock too, after its capture (within the
  // clocks' reads).
  CHECK(frames[0].arrival_ns + 1'000'000 >= frames[0].timestamp_ns);
  // An interruption: shown, and its end re-measures without going back.
  driver->interrupted(true);
  CHECK(driver->status().interrupted);
  driver->interrupted(false);
  CHECK(!driver->status().interrupted);
  driver->deliver(capture(maps, pic.ref, t0 + 3 * kPeriod), now_ns());
  if (take_newest(*driver, frames) != 0) return 1;
  CHECK(frames.size() == 4 && frames[3].timestamp_ns > frames[2].timestamp_ns);
  return 0;
}

// A feed on a thread of its own, as ARKit's queue is, and a consumer woken by
// the notification: every frame in order, none lost but as counted.
int test_threaded(const Gpu& gpu) {
  DepthMaps maps;
  std::vector<Pixels> pics;
  for (std::uint32_t i = 0; i < 4; ++i) pics.push_back(picture(kArkitTags));
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, pics[0].ref);
  CHECK(driver != nullptr);
  CHECK(driver->set_queue_depth(4).ok());
  std::mutex mutex;
  std::condition_variable woken;
  std::uint64_t notified = 0;
  CHECK(driver
            ->set_frame_notify([&] {
              std::lock_guard<std::mutex> lock(mutex);
              ++notified;
              woken.notify_one();
            })
            .ok());
  CHECK(driver->begin().ok());
  CHECK(driver->set_frame_notify(nullptr).domain() ==
        vkc::Status::Code::InvalidArgument);  // once started
  constexpr int kFrames = 200;
  std::atomic<bool> done{false};
  std::thread feed([&] {
    for (int i = 0; i < kFrames; ++i) {
      driver->deliver(
          capture(maps, pics[std::size_t(i) % 4].ref, 50.0 + i * kPeriod),
          now_ns());
      std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
    done = true;
  });
  testing::RgbdContract contract(driver->info());
  std::uint64_t taken = 0;
  // The consumer, returning 1 on a failed check, so the feed is joined
  // either way.
  const auto consume = [&]() -> int {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        woken.wait_for(lock, std::chrono::milliseconds(20),
                       [&] { return notified != seen; });
        seen = notified;
      }
      std::vector<sensor::RgbdFrame> frames;
      CHECK(driver->drain(&frames).ok());
      for (const sensor::RgbdFrame& f : frames) contract.take(f);
      taken += frames.size();
      const sensor::SensorStats s = driver->stats();  // as a read-out would
      CHECK(s.delivered + s.dropped + s.failed <= s.received);
      if (done && frames.empty() && s.received == kFrames &&
          s.delivered + s.dropped == kFrames) {
        return 0;
      }
    }
  };
  const int consumed = consume();
  feed.join();
  if (consumed != 0) return consumed;
  const sensor::SensorStats s = driver->stats();
  CHECK(contract.breach(s).empty());
  CHECK(s.received == kFrames && s.delivered == taken);
  CHECK(s.delivered + s.dropped == kFrames);
  {
    std::lock_guard<std::mutex> lock(mutex);
    CHECK(notified == kFrames);
  }
  driver->end();
  return 0;
}

int test_failure(const Gpu& gpu) {
  DepthMaps maps;
  const Pixels pic = picture(kArkitTags);
  const std::unique_ptr<arkit::Driver> driver =
      opened(options(gpu), maps, pic.ref);
  CHECK(driver != nullptr);
  std::atomic<int> notified{0};
  CHECK(driver->set_frame_notify([&] { ++notified; }).ok());
  CHECK(driver->begin().ok());
  driver->deliver(capture(maps, pic.ref, 60.0), 1);
  CHECK(notified == 1);
  CHECK(!driver->failed());
  driver->fail("The camera went away.");
  CHECK(notified == 2);
  CHECK(driver->failed());
  CHECK(driver->status().session_error == "The camera went away.");
  // The failure, not the frame held.
  vkc::Result<std::optional<sensor::RgbdFrame>> got = driver->poll();
  CHECK(!got.ok() && got.status().domain() == vkc::Status::Code::IoError);
  CHECK(got.status().message() ==
        "ArkitSensor: ARKit's session failed: The camera went away.");
  std::vector<sensor::RgbdFrame> frames;
  CHECK(driver->drain(&frames).domain() == vkc::Status::Code::IoError);
  driver->end();
  CHECK(driver->begin().domain() == vkc::Status::Code::IoError);
  return 0;
}

int test_options(const Gpu& gpu) {
  const auto refused = [](const sensor::ArkitSensor::Options& o) {
    return arkit::check_options(o).domain() ==
               vkc::Status::Code::InvalidArgument &&
           arkit::Driver::create(o, "x").status().domain() ==
               vkc::Status::Code::InvalidArgument;
  };
  CHECK(arkit::check_options(options(gpu)).ok());
  sensor::ArkitSensor::Options o = options(gpu);
  o.device = nullptr;
  CHECK(refused(o));
  CHECK(arkit::check_options(o).message() == "ArkitSensor: no device");
  for (int field = 0; field < 3; ++field) {
    o = options(gpu);
    (field == 0 ? o.color_width : field == 1 ? o.color_height : o.fps) = 0;
    CHECK(refused(o));
  }
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const auto& range : std::vector<std::pair<float, float>>{{0.0f, 5.0f},
                                                                {-0.1f, 5.0f},
                                                                {nan, 5.0f},
                                                                {0.2f, nan},
                                                                {0.2f, inf},
                                                                {1.0f, 1.0f},
                                                                {2.0f, 1.0f}}) {
    o = options(gpu);
    o.min_depth = range.first;
    o.max_depth = range.second;
    CHECK(refused(o));
  }
  o = options(gpu);
  o.min_confidence = 3;
  CHECK(refused(o));
  o = options(gpu);
  o.color_queue_family_count = sensor::kMaxColorQueueFamilies + 1;
  CHECK(refused(o));
  // The importer's own refusal of a family the device lacks.
  o = options(gpu);
  o.color_queue_families[0] = 99;
  o.color_queue_family_count = 1;
  CHECK(arkit::Driver::create(o, "x").status().domain() ==
        vkc::Status::Code::InvalidArgument);
  // Every family the device has, as many as fit: taken.
  o = options(gpu);
  o.color_queue_families[0] = gpu.device.queue_family();
  o.color_queue_family_count = 1;
  CHECK(arkit::Driver::create(o, "x").ok());
  return 0;
}

int gpu_main(Gpu& gpu) {
  if (!gpu.device.imports_metal_textures()) {
    return vkc::test::no_device_exit_code(
        "the device imports no Metal textures");
  }
  if (test_info(gpu) != 0) return 1;
  if (test_open_refusals(gpu) != 0) return 1;
  if (test_contract(gpu) != 0) return 1;
  if (test_frame(gpu) != 0) return 1;
  if (test_holds(gpu) != 0) return 1;
  if (test_clock(gpu) != 0) return 1;
  if (test_threaded(gpu) != 0) return 1;
  if (test_failure(gpu) != 0) return 1;
  if (test_options(gpu) != 0) return 1;
  std::puts("sensor_arkit_sensor: OK");
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
