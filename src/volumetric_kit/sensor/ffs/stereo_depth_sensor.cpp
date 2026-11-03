// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/ffs/stereo_depth_sensor.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "../frame_queue.hpp"
#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor {
namespace {

constexpr const char* kWho = "StereoDepthSensor";

// How long the thread waits before asking again a camera that does not
// notify, when it had no new pair.
constexpr std::chrono::milliseconds kIdle{1};

// What wakes the thread: a stop, or the camera's notification, which may run
// on the camera's own thread even after the camera let it go, so it holds
// this.
struct Wake {
  std::mutex mutex;
  std::condition_variable cv;
  bool stopping = false;  // guarded by mutex
  bool pair = false;      // guarded by mutex: the camera holds a new pair
};

}  // namespace

struct StereoDepthSensor::Impl {
  // The camera goes last, and FFS, whose mappings read the rectifier's views,
  // before the rectifier.
  std::unique_ptr<IStereoSensor> camera;
  std::unique_ptr<StereoRectifier> rectifier;
  std::unique_ptr<FfsStereo> ffs;
  SensorInfo info;
  // Whether the camera calls the notification; else it is asked every kIdle.
  bool camera_notifies = false;
  std::shared_ptr<Wake> wake = std::make_shared<Wake>();

  // The caller's thread alone.
  std::thread worker;
  bool running = false;

  // The frames made, their drops and the thread's failure, which ends one
  // run and is forgotten at the next start; and whom each wakes.
  detail::FrameQueue<RgbdFrame> held;
  // Frames handed out, pairs refused here: each counted after what the
  // camera's counts last said of it.
  std::atomic<std::uint64_t> delivered{0};
  std::atomic<std::uint64_t> failed{0};
  std::mutex mutex;
  // The camera's own, as the thread last read them.
  SensorStats camera_counts;  // guarded by mutex

  void run();
};

void StereoDepthSensor::Impl::run() {
  for (;;) {
    {
      const std::lock_guard<std::mutex> lock(wake->mutex);
      if (wake->stopping) return;
      // A pair held from here on wakes the wait below.
      wake->pair = false;
    }
    core::Result<std::optional<StereoFrame>> polled = camera->poll();
    if (!polled) {
      held.fail(polled.status().with_context(kWho));
      return;
    }
    // On this thread, as the poll: the camera is called from one at a time.
    const SensorStats seen = camera->stats();
    {
      const std::lock_guard<std::mutex> lock(mutex);
      camera_counts = seen;
    }
    if (!polled->has_value()) {
      std::unique_lock<std::mutex> lock(wake->mutex);
      const auto woken = [this] { return wake->stopping || wake->pair; };
      if (camera_notifies) {
        wake->cv.wait(lock, woken);
      } else {
        wake->cv.wait_for(lock, kIdle, woken);
      }
      continue;
    }
    // The camera's frame goes once rectified, so its right picture returns to
    // the camera before FFS runs; the pair goes with this scope, so the
    // rectifier writes its views again.
    core::Result<RgbdFrame> made = [&]() -> core::Result<RgbdFrame> {
      VKC_ASSIGN(const RectifiedPair pair,
                 rectifier->rectify(std::exchange(**polled, {})));
      return ffs->depth(pair);
    }();
    if (!made) {
      failed.fetch_add(1, std::memory_order_release);
      // This pair; the next may do.
      if (made.status().domain() == core::Status::Code::InvalidArgument) {
        continue;
      }
      // The GPU failed, and the rectifier or FFS refuses every pair now.
      held.fail(made.status().with_context(kWho));
      return;
    }
    held.push(std::move(made).value());
  }
}

StereoDepthSensor::StereoDepthSensor(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

StereoDepthSensor::~StereoDepthSensor() { stop(); }

core::Result<std::unique_ptr<StereoDepthSensor>> StereoDepthSensor::create(
    const core::Device& device, core::Allocator& allocator,
    std::unique_ptr<IStereoSensor> camera, const Options& options) try {
  if (camera == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": no camera");
  }
  auto impl = std::make_unique<Impl>();
  impl->camera = std::move(camera);
  VKC_ASSIGN(impl->rectifier,
             StereoRectifier::create(device, allocator, impl->camera->info(),
                                     options.rectify));
  VKC_ASSIGN(impl->ffs, FfsStereo::create(device, allocator, *impl->rectifier,
                                          options.ffs));
  impl->info = impl->rectifier->depth_info();
  // The thread sleeps until the camera has a pair, where the camera says so.
  impl->camera_notifies =
      impl->camera
          ->set_frame_notify([wake = impl->wake] {
            {
              const std::lock_guard<std::mutex> lock(wake->mutex);
              wake->pair = true;
            }
            wake->cv.notify_one();
          })
          .ok();
  return std::unique_ptr<StereoDepthSensor>(
      new StereoDepthSensor(std::move(impl)));
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("StereoDepthSensor: out of host memory");
}

const SensorInfo& StereoDepthSensor::info() const noexcept {
  return impl_->info;
}

core::Status StereoDepthSensor::set_queue_depth(std::size_t frames) {
  if (frames == 0 || impl_->running) {
    return core::Status::invalid_argument(
        std::string(kWho) +
        ": set_queue_depth needs at least 1 frame, before start");
  }
  impl_->held.set_depth(frames);
  return {};
}

core::Status StereoDepthSensor::set_frame_notify(std::function<void()> notify) {
  if (impl_->running) {
    return core::Status::invalid_argument(std::string(kWho) +
                                          ": set_frame_notify once started");
  }
  impl_->held.set_notify(std::move(notify));
  return {};
}

core::Status StereoDepthSensor::start() {
  Impl& s = *impl_;
  if (s.running) return {};
  // Before the thread, so nothing races these.
  s.held.clear();
  s.held.reset_dropped();
  s.held.reset_failure();
  s.delivered.store(0, std::memory_order_relaxed);
  s.failed.store(0, std::memory_order_relaxed);
  {
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.camera_counts = {};
  }
  {
    const std::lock_guard<std::mutex> lock(s.wake->mutex);
    s.wake->stopping = false;
  }
  VKC_TRY(s.camera->start());
  try {
    s.worker = std::thread([&s] { s.run(); });
  } catch (const std::system_error& e) {
    s.camera->stop();
    return core::Status::out_of_memory(
        std::string(kWho) + ": cannot start its thread: " + e.what());
  }
  s.running = true;
  return {};
}

void StereoDepthSensor::stop() noexcept {
  Impl& s = *impl_;
  if (!s.running) return;
  {
    const std::lock_guard<std::mutex> lock(s.wake->mutex);
    s.wake->stopping = true;
  }
  s.wake->cv.notify_all();
  // The thread finishes the pair it is matching; only then is the camera
  // called from here.
  try {
    s.worker.join();
  } catch (const std::system_error&) {
    // Not joinable: it never ran.
  }
  s.camera->stop();
  // Let go uncounted, as any sensor's stop does: the counters' gap.
  s.held.clear();
  s.running = false;
}

core::Result<std::optional<RgbdFrame>> StereoDepthSensor::poll() {
  Impl& s = *impl_;
  VKC_ASSIGN(std::optional<RgbdFrame> frame, s.held.take_newest());
  if (!frame) return no_frame();
  s.delivered.fetch_add(1, std::memory_order_release);
  return some_frame(std::move(*frame));
}

core::Status StereoDepthSensor::drain(std::vector<RgbdFrame>* out) {
  VKC_CHECK(out != nullptr,
            "StereoDepthSensor::drain needs somewhere to put them");
  Impl& s = *impl_;
  const std::size_t before = out->size();
  VKC_TRY(s.held.take_all(out));
  s.delivered.fetch_add(out->size() - before, std::memory_order_release);
  return {};
}

SensorStats StereoDepthSensor::stats() const noexcept {
  const Impl& s = *impl_;
  // The camera's counts last: each of these was counted after the camera's
  // counts said of its pair, so they never sum past what it received.
  SensorStats out;
  out.delivered = s.delivered.load(std::memory_order_acquire);
  out.dropped = s.held.dropped();
  out.failed = s.failed.load(std::memory_order_acquire);
  const std::lock_guard<std::mutex> lock(s.mutex);
  out.received = s.camera_counts.received;
  out.dropped += s.camera_counts.dropped;
  out.failed += s.camera_counts.failed;
  return out;
}

}  // namespace volumetric_kit::sensor
