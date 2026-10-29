// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/ffs/stereo_depth_sensor.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor {
namespace {

constexpr const char* kWho = "StereoDepthSensor";

// How long the thread waits before asking the camera again when it had no new
// pair: IStereoSensor is polled and tells no one.
constexpr std::chrono::milliseconds kIdle{1};

}  // namespace

struct StereoDepthSensor::Impl {
  // The camera goes last, and FFS, whose mappings read the rectifier's views,
  // before the rectifier.
  std::unique_ptr<IStereoSensor> camera;
  std::unique_ptr<StereoRectifier> rectifier;
  std::unique_ptr<FfsStereo> ffs;
  SensorInfo info;

  // The caller's thread alone.
  std::thread worker;
  bool running = false;
  // Set before start, read by the worker.
  std::function<void()> notify;

  std::mutex mutex;
  std::condition_variable wake;  // the worker's stop
  bool stopping = false;         // guarded by mutex
  std::deque<RgbdFrame> held;    // guarded by mutex; oldest first
  std::size_t depth = 1;         // guarded by mutex
  core::Status failure;          // guarded by mutex
  SensorStats counts;            // guarded by mutex

  void run();
  void fail(core::Status status);
};

void StereoDepthSensor::Impl::fail(core::Status status) {
  {
    const std::lock_guard<std::mutex> lock(mutex);
    failure = std::move(status);
  }
  if (notify) notify();
}

void StereoDepthSensor::Impl::run() {
  for (;;) {
    {
      const std::lock_guard<std::mutex> lock(mutex);
      if (stopping) return;
    }
    core::Result<std::optional<StereoFrame>> polled = camera->poll();
    if (!polled) {
      fail(polled.status().with_context(kWho));
      return;
    }
    if (!polled->has_value()) {
      std::unique_lock<std::mutex> lock(mutex);
      wake.wait_for(lock, kIdle, [this] { return stopping; });
      continue;
    }
    {
      const std::lock_guard<std::mutex> lock(mutex);
      ++counts.received;
    }
    // The pair goes with this scope, so the rectifier writes its views again.
    core::Result<RgbdFrame> made = [&]() -> core::Result<RgbdFrame> {
      VKC_ASSIGN(const RectifiedPair pair, rectifier->rectify(**polled));
      return ffs->depth(pair);
    }();
    if (!made) {
      const bool refused =
          made.status().domain() == core::Status::Code::InvalidArgument;
      {
        const std::lock_guard<std::mutex> lock(mutex);
        ++counts.failed;
      }
      if (refused) continue;  // this pair; the next may do
      // The GPU failed, and the rectifier or FFS refuses every pair now.
      fail(made.status().with_context(kWho));
      return;
    }
    {
      const std::lock_guard<std::mutex> lock(mutex);
      held.push_back(std::move(made).value());
      while (held.size() > depth) {
        held.pop_front();
        ++counts.dropped;
      }
    }
    // After the lock, so the poll it wakes finds the frame.
    if (notify) notify();
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
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->depth = frames;
  return {};
}

core::Status StereoDepthSensor::set_frame_notify(std::function<void()> notify) {
  if (impl_->running) {
    return core::Status::invalid_argument(std::string(kWho) +
                                          ": set_frame_notify once started");
  }
  impl_->notify = std::move(notify);
  return {};
}

core::Status StereoDepthSensor::start() {
  Impl& s = *impl_;
  if (s.running) return {};
  {
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.held.clear();
    s.failure = {};
    s.counts = {};
    s.stopping = false;
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
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.stopping = true;
  }
  s.wake.notify_all();
  // The thread finishes the pair it is matching; only then is the camera
  // called from here.
  try {
    s.worker.join();
  } catch (const std::system_error&) {
    // Not joinable: it never ran.
  }
  s.camera->stop();
  {
    const std::lock_guard<std::mutex> lock(s.mutex);
    s.counts.dropped += s.held.size();
    s.held.clear();
  }
  s.running = false;
}

core::Result<std::optional<RgbdFrame>> StereoDepthSensor::poll() {
  Impl& s = *impl_;
  const std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.failure.ok()) return s.failure;
  if (s.held.empty()) return no_frame();
  s.counts.dropped += s.held.size() - 1;
  RgbdFrame frame = std::move(s.held.back());
  s.held.clear();
  ++s.counts.delivered;
  return some_frame(std::move(frame));
}

core::Status StereoDepthSensor::drain(std::vector<RgbdFrame>* out) {
  VKC_CHECK(out != nullptr,
            "StereoDepthSensor::drain needs somewhere to put them");
  Impl& s = *impl_;
  const std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.failure.ok()) return s.failure;
  for (RgbdFrame& frame : s.held) out->push_back(std::move(frame));
  s.counts.delivered += s.held.size();
  s.held.clear();
  return {};
}

SensorStats StereoDepthSensor::stats() const noexcept {
  const std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->counts;
}

}  // namespace volumetric_kit::sensor
