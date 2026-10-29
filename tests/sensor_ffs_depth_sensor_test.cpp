// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// StereoDepthSensor as any IRgbdSensor: a test camera (an IStereoSensor fed
// from here, not hardware) through the rectifier and FFS on the test model
// (tests/data/ffs, whose disparity is left less right on grey pairs), on the
// sensor's own thread. Checked: info is the rectifier's depth view; poll
// holds the newest frame, drain every frame oldest first; frame
// notifications, the pairs' times, the counters; a pair the rectifier
// refuses counted and passed over, a camera that fails reported; the camera
// polled on the thread alone and never called from two at once; stop and a
// restart; a frame kept while the next is made.
//
// It skips where CUDA does not reach the GPU, as sensor_ffs_stereo does.

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "decoder_test.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/ffs/stereo_depth_sensor.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

constexpr std::uint32_t kW = 64, kH = 32;
constexpr double kFocal = 40.0, kBaseline = 0.1;

// A pair whose left luma is `disparity` above the right one everywhere.
struct Pictures {
  std::vector<std::uint8_t> left, right;
};

sensor::StereoFrame pair_frame(std::uint8_t disparity, std::uint64_t sequence,
                               std::uint32_t width = kW) {
  auto p = std::make_shared<Pictures>();
  p->right.assign(std::size_t{width} * kH, 50);
  p->left.assign(p->right.size(), static_cast<std::uint8_t>(50 + disparity));
  sensor::StereoFrame f;
  for (auto [image, bytes] :
       {std::pair{&f.left, &p->left}, std::pair{&f.right, &p->right}}) {
    image->layout = sensor::YuvLayout::Luma;
    image->host = bytes->data();
    image->stride[0] = width;
    image->width = width;
    image->height = kH;
  }
  f.timestamp_ns = 1'000'000 * sequence;
  f.sequence = sequence;
  f.arrival_ns = 7 + sequence;
  f.pixels = p;
  return f;
}

// A stereo camera fed by the test: pairs pushed here come out of poll, once
// each. It records which thread calls it, and any two calls that overlap.
class TestCamera final : public sensor::IStereoSensor {
 public:
  struct Log {
    std::mutex mutex;
    std::deque<sensor::StereoFrame> pending;  // guarded by mutex
    std::optional<vkc::Status> fault;         // guarded by mutex
    std::atomic<int> in_call{0};
    std::atomic<int> overlaps{0};
    std::atomic<int> starts{0}, stops{0}, polls{0};
    std::atomic<bool> polled_off_main{true};
    std::thread::id main = std::this_thread::get_id();
    std::atomic<bool> started_on_main{true}, stopped_on_main{true};
  };

  explicit TestCamera(std::shared_ptr<Log> log) : log_(std::move(log)) {
    info_.calibration.left = info_.calibration.right = camera::CameraModel{
        {kW, kH}, {kFocal, kFocal, (kW - 1) / 2.0, (kH - 1) / 2.0}, {}};
    info_.calibration.left_to_right[3] = glm::dvec4(-kBaseline, 0.0, 0.0, 1.0);
  }

  const sensor::StereoInfo& info() const noexcept override { return info_; }

  vkc::Status start() override {
    const Call call(*log_);
    ++log_->starts;
    if (std::this_thread::get_id() != log_->main) log_->started_on_main = false;
    return {};
  }

  void stop() noexcept override {
    const Call call(*log_);
    ++log_->stops;
    if (std::this_thread::get_id() != log_->main) log_->stopped_on_main = false;
  }

  vkc::Result<std::optional<sensor::StereoFrame>> poll() override {
    const Call call(*log_);
    ++log_->polls;
    if (std::this_thread::get_id() == log_->main) log_->polled_off_main = false;
    const std::lock_guard<std::mutex> lock(log_->mutex);
    if (log_->fault) return *log_->fault;
    if (log_->pending.empty()) return std::optional<sensor::StereoFrame>{};
    sensor::StereoFrame f = std::move(log_->pending.front());
    log_->pending.pop_front();
    return std::optional<sensor::StereoFrame>{std::move(f)};
  }

  sensor::SensorStats stats() const noexcept override { return {}; }

 private:
  // Counts a call in, and finds another already in.
  struct Call {
    explicit Call(Log& log) : log_(log) {
      if (log_.in_call.fetch_add(1) != 0) ++log_.overlaps;
      std::this_thread::sleep_for(50us);
    }
    ~Call() { log_.in_call.fetch_sub(1); }
    Log& log_;
  };

  std::shared_ptr<Log> log_;
  sensor::StereoInfo info_;
};

// Frame notifications, counted, which a test waits on.
struct Notes {
  std::mutex mutex;
  std::condition_variable cv;
  int count = 0;

  std::function<void()> callback() {
    return [this] {
      {
        const std::lock_guard<std::mutex> lock(mutex);
        ++count;
      }
      cv.notify_all();
    };
  }
  bool wait_for(int n) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, 20s, [&] { return count >= n; });
  }
};

void push(TestCamera::Log& log, sensor::StereoFrame f) {
  const std::lock_guard<std::mutex> lock(log.mutex);
  log.pending.push_back(std::move(f));
}

// The frame's depth at its centre, read as a Vulkan reader reads it.
vkc::Result<std::uint16_t> centre_depth(vkc::test::GpuContext& gpu,
                                        const sensor::RgbdFrame& frame) {
  std::vector<std::uint16_t> out(std::size_t{kW} * kH);
  const vkc::Buffer& buffer = *frame.depth_device.buffer;
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  VKC_TRY(batch.acquire(buffer, VK_QUEUE_FAMILY_EXTERNAL));
  VKC_TRY(batch.readback(buffer, 0, out.size() * 2, out.data()));
  VKC_TRY(batch.release(buffer, VK_QUEUE_FAMILY_EXTERNAL));
  VKC_TRY(batch.submit());
  return out[std::size_t{kH / 2} * kW + kW / 2];
}

// Depth units for a disparity, as FfsStereo writes them (0.1 mm).
std::uint16_t units_for(int disparity) {
  return static_cast<std::uint16_t>(
      std::lround(kFocal * kBaseline / disparity / 1e-4));
}

sensor::StereoDepthSensor::Options options_for(const fs::path& cache) {
  sensor::StereoDepthSensor::Options o;
  o.rectify.output = sensor::StereoRectifier::Output::Luma8;
  o.ffs.model = fs::path(VS_FFS_DATA) / "difference_64x32.onnx";
  o.ffs.plan_cache = cache;
  o.ffs.min_depth = 0.05f;
  o.ffs.max_depth = 2.0f;
  return o;
}

int run(vkc::test::GpuContext& gpu, const fs::path& scratch) {
  auto log = std::make_shared<TestCamera::Log>();
  const sensor::StereoDepthSensor::Options options = options_for(scratch);

  CHECK(sensor::StereoDepthSensor::create(gpu.device, gpu.allocator, nullptr,
                                          options)
            .status()
            .domain() == vkc::Status::Code::InvalidArgument);
  auto made = sensor::StereoDepthSensor::create(
      gpu.device, gpu.allocator, std::make_unique<TestCamera>(log), options);
  if (!made) std::fprintf(stderr, "%s\n", made.status().message().c_str());
  CHECK(made.ok());
  sensor::IRgbdSensor& s = **made;

  // Its info: depth in the rectified left camera, the left one the colour.
  const sensor::SensorInfo& info = s.info();
  CHECK(info.depth && info.color);
  CHECK(info.depth->size.width == kW && info.depth->size.height == kH);
  CHECK(info.depth->intrinsics.fx == kFocal);
  CHECK(info.color->size.width == kW);
  std::puts("info: the rectifier's depth view");

  // Newest first: three pairs, one frame held, two dropped.
  Notes notes;
  CHECK(s.set_queue_depth(0).domain() == vkc::Status::Code::InvalidArgument);
  CHECK(s.set_frame_notify(notes.callback()).ok());
  CHECK(s.start().ok());
  CHECK(s.start().ok());  // idempotent
  CHECK(log->starts == 1);
  CHECK(s.set_frame_notify({}).domain() == vkc::Status::Code::InvalidArgument);
  CHECK(s.set_queue_depth(2).domain() == vkc::Status::Code::InvalidArgument);
  {
    auto none = s.poll();
    CHECK(none.ok() && !none->has_value());
  }
  for (std::uint64_t i = 1; i <= 3; ++i) {
    push(*log, pair_frame(static_cast<std::uint8_t>(10 + i), i));
    CHECK(notes.wait_for(static_cast<int>(i)));
  }
  auto newest = s.poll();
  CHECK(newest.ok() && newest->has_value());
  const sensor::RgbdFrame& frame = **newest;
  CHECK(frame.sequence == 3 && frame.timestamp_ns == 3'000'000 &&
        frame.arrival_ns == 10);
  auto centre = centre_depth(gpu, frame);
  CHECK(centre.ok());
  std::printf("  centre depth %u units (disparity 13: %u)\n", *centre,
              units_for(13));
  CHECK(*centre == units_for(13));
  sensor::SensorStats st = s.stats();
  CHECK(st.received == 3 && st.delivered == 1 && st.dropped == 2 &&
        st.failed == 0);
  std::puts("poll: the newest of three, its times the pair's; two dropped");

  // A frame kept while the next is made: both stand.
  push(*log, pair_frame(20, 4));
  CHECK(notes.wait_for(4));
  auto next = s.poll();
  CHECK(next.ok() && next->has_value());
  CHECK((*next)->depth_device.buffer != frame.depth_device.buffer);
  auto kept_centre = centre_depth(gpu, frame);
  auto next_centre = centre_depth(gpu, **next);
  CHECK(kept_centre.ok() && next_centre.ok());
  CHECK(*kept_centre == units_for(13) && *next_centre == units_for(20));
  std::puts("a frame kept while the next is made keeps its depth");

  // A pair the rectifier refuses (its pictures the wrong size) is counted and
  // passed over; the next one is matched.
  push(*log, pair_frame(10, 5, kW / 2));
  push(*log, pair_frame(16, 6));
  CHECK(notes.wait_for(5));
  auto after = s.poll();
  CHECK(after.ok() && after->has_value() && (*after)->sequence == 6);
  st = s.stats();
  CHECK(st.received == 6 && st.failed == 1);
  std::puts("a refused pair: counted failed, and the next one matched");

  // Stop: the thread, then the camera, on this thread; frames held dropped.
  push(*log, pair_frame(12, 7));
  CHECK(notes.wait_for(6));
  s.stop();
  s.stop();  // idempotent
  CHECK(log->stops == 1);
  st = s.stats();
  CHECK(st.dropped == 3);  // two at the first poll, one held at the stop
  {
    auto none = s.poll();
    CHECK(none.ok() && !none->has_value());
  }
  std::puts("stop: the frame held dropped");

  // A restart with a deeper queue: drain hands every frame, oldest first;
  // the counters start again.
  CHECK(s.set_queue_depth(3).ok());
  CHECK(s.start().ok());
  CHECK(s.stats().received == 0);
  for (std::uint64_t i = 8; i <= 10; ++i) {
    push(*log, pair_frame(static_cast<std::uint8_t>(i), i));
    CHECK(notes.wait_for(static_cast<int>(i) - 1));
  }
  std::vector<sensor::RgbdFrame> drained;
  CHECK(s.drain(&drained).ok());
  CHECK(drained.size() == 3);
  for (std::size_t i = 0; i < 3; ++i) CHECK(drained[i].sequence == 8 + i);
  st = s.stats();
  CHECK(st.received == 3 && st.delivered == 3 && st.dropped == 0);
  std::puts("drain: three frames, oldest first, after a restart");

  // The camera's failure is the sensor's, and a notification.
  {
    const std::lock_guard<std::mutex> lock(log->mutex);
    log->fault = vkc::Status::io_error("the cable is out");
  }
  CHECK(notes.wait_for(10));
  auto failed = s.poll();
  CHECK(!failed.ok());
  CHECK(failed.status().message().find("the cable is out") !=
        std::string::npos);
  std::vector<sensor::RgbdFrame> none;
  CHECK(!s.drain(&none).ok());
  s.stop();
  std::puts("a camera failure: reported by poll and drain, after a notice");

  // The camera was polled on the thread alone, started and stopped here, and
  // never called from two threads at once.
  CHECK(log->polls > 0 && log->polled_off_main);
  CHECK(log->started_on_main && log->stopped_on_main);
  CHECK(log->overlaps == 0);
  std::printf("camera: %d polls off this thread, no two calls at once\n",
              log->polls.load());
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  const vkc::Result<int> ordinal = sensor::cuda_ordinal_of(gpu.device);
  if (!ordinal) {
    if (ordinal.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", ordinal.status().message().c_str());
      return 1;
    }
    return vs_test::no_decoder(ordinal.status().message());
  }
  if (!gpu.device.exports_memory()) {
    return vs_test::no_decoder("the device does not export memory");
  }
  const vkc::test::Scratch scratch("sensor_ffs_depth_sensor_");
  CHECK(run(gpu, scratch.path) == 0);
  std::puts("sensor_ffs_depth_sensor: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
