// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ArkitSensor with its test pattern (ArkitSource::Pattern), through the
// public header alone, on the device: info; 300 frames taken through
// set_frame_notify, each one's depth band (depth_index) and picture band
// (pattern_index, its luma read back on the device's queue by
// PictureReadback) naming its sequence, the frame contract (rgbd_breach) and
// no failures; a tick finding all 4 pictures held is skipped and counted
// missed, leaving no gap in the sequences; sequences and counters from 0 at
// each start; the refusals of open; and moves.
//
// Skips where no Vulkan device is present, unless VKC_REQUIRE_VULKAN_DEVICE
// says this machine has one.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/arkit/arkit_sensor.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/picture_readback.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/testing/rgbd_contract.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace testing = volumetric_kit::sensor::testing;

namespace {

using Gpu = vkc::test::GpuContext;
using sensor::ArkitSensor;

constexpr std::uint64_t kBand = std::uint64_t{1} << testing::kIndexBits;

ArkitSensor::Options pattern(const Gpu& gpu) {
  ArkitSensor::Options o;
  o.source = sensor::ArkitSource::Pattern;
  o.device = &gpu.device;
  return o;
}

// Waits on a sensor's notification. The notification owns what it touches,
// as IRgbdSensor asks: it may run after the waker's scope has ended, until
// the sensor stops.
class Waker {
 public:
  std::function<void()> notify() {
    return [state = state_] {
      std::lock_guard<std::mutex> lock(state->mutex);
      ++state->count;
      state->woken.notify_one();
    };
  }
  // Until a notification past `seen`, or `timeout`; the count now.
  std::uint64_t wait(std::uint64_t seen, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->woken.wait_for(lock, timeout,
                           [&] { return state_->count != seen; });
    return state_->count;
  }
  std::uint64_t count() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->count;
  }

 private:
  struct State {
    std::mutex mutex;
    std::condition_variable woken;
    std::uint64_t count = 0;
  };
  std::shared_ptr<State> state_ = std::make_shared<State>();
};

int test_refusals(const Gpu& gpu) {
  ArkitSensor::Options o = pattern(gpu);
  o.source = sensor::ArkitSource::Camera;
  vkc::Result<ArkitSensor> camera = ArkitSensor::open(o);
  CHECK(!camera.ok());
  CHECK(camera.status().domain() == vkc::Status::Code::Unsupported);
  CHECK(camera.status().message() ==
        "ArkitSensor: ARKit's camera runs on iOS and iPadOS only; "
        "ArkitSource::Pattern runs here");
  o = pattern(gpu);
  o.device = nullptr;
  CHECK(ArkitSensor::open(o).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  o = pattern(gpu);
  o.color_queue_family_count = sensor::kMaxColorQueueFamilies + 1;
  CHECK(ArkitSensor::open(o).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  o = pattern(gpu);
  o.min_depth = 0.0f;
  CHECK(ArkitSensor::open(o).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  return 0;
}

int test_info(const Gpu& gpu) {
  vkc::Result<ArkitSensor> opened = ArkitSensor::open(pattern(gpu));
  CHECK(opened.ok());
  const sensor::SensorInfo& info = opened->info();
  CHECK(info.id == "arkit");
  CHECK(info.model.rfind("ARKit test pattern, ", 0) == 0);
  CHECK(info.model.size() > std::string("ARKit test pattern, ").size());
  CHECK(info.color && info.depth);
  CHECK(info.color->size.width == 1920 && info.color->size.height == 1440);
  CHECK(info.color->intrinsics.fx == 1440.0 &&
        info.color->intrinsics.fy == 1440.0);
  CHECK(info.color->intrinsics.cx == 959.5 &&
        info.color->intrinsics.cy == 719.5);
  CHECK(info.depth->size.width == 256 && info.depth->size.height == 192);
  CHECK(info.pose == sensor::PoseSource::Tracked);
  CHECK(info.clock == sensor::ClockDomain::Host);
  CHECK(info.role == sensor::SyncRole::FreeRun);
  CHECK(info.fps == 60);
  CHECK(opened->status().tracking == sensor::ArkitTracking::Normal);
  CHECK(!opened->exhausted());
  return 0;
}

// 300 frames, each carrying its sequence in its depth and its picture.
int test_frames(const Gpu& gpu) {
  vkc::Result<ArkitSensor> opened = ArkitSensor::open(pattern(gpu));
  CHECK(opened.ok());
  ArkitSensor camera = std::move(opened).value();
  vkc::Result<std::unique_ptr<sensor::PictureReadback>> readback =
      sensor::PictureReadback::create(gpu.device, gpu.allocator);
  CHECK(readback.ok());
  Waker waker;
  CHECK(camera.set_queue_depth(4).ok());
  CHECK(camera.set_frame_notify(waker.notify()).ok());
  CHECK(camera.start().ok());
  testing::RgbdContract contract(camera.info());
  std::uint64_t taken = 0;
  std::uint64_t seen = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (taken < 300) {
    CHECK(std::chrono::steady_clock::now() < deadline);
    seen = waker.wait(seen, std::chrono::milliseconds(200));
    std::vector<sensor::RgbdFrame> frames;
    CHECK(camera.drain(&frames).ok());
    for (sensor::RgbdFrame& frame : frames) {
      contract.take(frame);
      const std::uint64_t index = frame.sequence % kBand;
      CHECK(testing::depth_index(frame.depth, 256, 192) == index);
      CHECK(frame.depth_confidence != nullptr && frame.min_confidence == 2);
      CHECK(frame.depth_noise == sensor::kArkitDepthNoise);
      vkc::Result<sensor::PictureView> view =
          (*readback)->read(frame.color, sensor::PictureReadback::Planes::Luma);
      CHECK(view.ok());
      CHECK(view->width == 1920 && view->height == 1440 && view->full_range);
      CHECK(testing::pattern_index(view->luma, view->stride) ==
            static_cast<long>(index));
      ++taken;
    }
  }
  camera.stop();
  const sensor::SensorStats stats = camera.stats();
  CHECK(contract.breach(stats).empty());
  CHECK(stats.failed == 0);
  CHECK(stats.delivered == taken);
  CHECK(camera.status().withheld == 0 && camera.status().refused == 0);
  CHECK(!camera.exhausted());
  return 0;
}

// With all 4 pictures held, ticks are skipped: counted missed as the next
// frame arrives, and no gap in the sequences.
int test_skipped_ticks(const Gpu& gpu) {
  ArkitSensor::Options o = pattern(gpu);
  o.color_width = 320;
  o.color_height = 240;
  vkc::Result<ArkitSensor> opened = ArkitSensor::open(o);
  CHECK(opened.ok());
  ArkitSensor camera = std::move(opened).value();
  Waker waker;
  CHECK(camera.set_frame_notify(waker.notify()).ok());
  CHECK(camera.start().ok());
  std::vector<sensor::RgbdFrame> kept;
  std::uint64_t seen = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (kept.size() < 4) {
    CHECK(std::chrono::steady_clock::now() < deadline);
    seen = waker.wait(seen, std::chrono::milliseconds(200));
    vkc::Result<std::optional<sensor::RgbdFrame>> got = camera.poll();
    CHECK(got.ok());
    if (got.value()) kept.push_back(std::move(*got.value()));
  }
  // Every picture held: nothing more comes.
  const std::uint64_t received = camera.stats().received;
  const std::uint64_t missed = camera.status().missed;
  std::this_thread::sleep_for(std::chrono::milliseconds(150));  // 9 periods
  CHECK(camera.stats().received == received);
  const std::uint64_t last = kept.back().sequence;
  kept.clear();
  std::optional<sensor::RgbdFrame> next;
  while (!next) {
    CHECK(std::chrono::steady_clock::now() < deadline);
    seen = waker.wait(seen, std::chrono::milliseconds(200));
    vkc::Result<std::optional<sensor::RgbdFrame>> got = camera.poll();
    CHECK(got.ok());
    next = std::move(got.value());
  }
  CHECK(camera.status().missed >= missed + 3);
  // The 4 held were the last made, and the ticks skipped were never frames:
  // the next frame takes the next number.
  CHECK(last + 1 == received);
  CHECK(next->sequence == received);
  CHECK(camera.stats().failed == 0);
  return 0;
}

int test_restart(const Gpu& gpu) {
  ArkitSensor::Options o = pattern(gpu);
  o.color_width = 320;
  o.color_height = 240;
  vkc::Result<ArkitSensor> opened = ArkitSensor::open(o);
  CHECK(opened.ok());
  ArkitSensor camera = std::move(opened).value();
  Waker waker;
  CHECK(camera.set_frame_notify(waker.notify()).ok());
  CHECK(camera.set_queue_depth(64).ok());  // none dropped while draining
  for (int run = 0; run < 2; ++run) {
    CHECK(camera.start().ok());
    CHECK(camera.start().ok());  // idempotent
    CHECK(camera.set_queue_depth(2).domain() ==
          vkc::Status::Code::InvalidArgument);  // once started
    std::vector<sensor::RgbdFrame> frames;
    std::uint64_t seen = waker.count();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (frames.size() < 3) {  // under the 4 pictures, which they hold
      CHECK(std::chrono::steady_clock::now() < deadline);
      seen = waker.wait(seen, std::chrono::milliseconds(200));
      CHECK(camera.drain(&frames).ok());
    }
    // Sequences and counters from 0 at each start.
    CHECK(frames.front().sequence == 0);
    CHECK(camera.stats().dropped == 0);
    CHECK(testing::rgbd_breach(camera.info(), frames, camera.stats()).empty());
    camera.stop();
    camera.stop();                                     // idempotent
    const sensor::SensorStats after = camera.stats();  // kept through a stop
    CHECK(after.delivered == frames.size());
  }
  return 0;
}

int test_moves(const Gpu& gpu) {
  vkc::Result<ArkitSensor> opened = ArkitSensor::open(pattern(gpu));
  CHECK(opened.ok());
  ArkitSensor a = std::move(opened).value();
  ArkitSensor b(std::move(a));
  CHECK(b.info().color.has_value());
  // NOLINTBEGIN(bugprone-use-after-move): the moved-from contract
  CHECK(!a.info().color && a.info().id.empty());
  CHECK(a.exhausted());
  CHECK(a.start().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.poll().status().domain() == vkc::Status::Code::InvalidArgument);
  std::vector<sensor::RgbdFrame> frames;
  CHECK(a.drain(&frames).domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.set_queue_depth(2).domain() == vkc::Status::Code::InvalidArgument);
  CHECK(a.set_frame_notify(nullptr).domain() ==
        vkc::Status::Code::InvalidArgument);
  CHECK(a.stats().received == 0);
  CHECK(a.status().tracking == sensor::ArkitTracking::NotAvailable);
  a.stop();
  // NOLINTEND(bugprone-use-after-move)
  // Over a live, started sensor, and onto itself.
  vkc::Result<ArkitSensor> other = ArkitSensor::open(pattern(gpu));
  CHECK(other.ok());
  ArkitSensor c = std::move(other).value();
  CHECK(c.start().ok());
  c = std::move(b);
  ArkitSensor* alias = &c;
  c = std::move(*alias);
  CHECK(c.start().ok());
  c.stop();
  return 0;
}

int gpu_main(Gpu& gpu) {
  if (!gpu.device.imports_metal_textures()) {
    return vkc::test::no_device_exit_code(
        "the device imports no Metal textures");
  }
  if (test_refusals(gpu) != 0) return 1;
  if (test_info(gpu) != 0) return 1;
  if (test_frames(gpu) != 0) return 1;
  if (test_skipped_ticks(gpu) != 0) return 1;
  if (test_restart(gpu) != 0) return 1;
  if (test_moves(gpu) != 0) return 1;
  std::puts("sensor_arkit_pattern: OK");
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
