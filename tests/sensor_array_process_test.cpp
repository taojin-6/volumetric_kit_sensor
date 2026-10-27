// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A sensor array's sets prepared on the GPU in one batch: each frame comes
// out as GpuFramePrep::prepare makes it alone, posed by the array's
// calibration with its sensor's depth offset, and an empty slot stays empty,
// recorded into a caller's batch the same; the prep options reach every
// pass of an array of any size; a depth offset in the prep options, a device
// or an allocator given without the other, an array without a device and a
// set of another size are refused. Skips where no device is.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "gpu_test.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;
namespace sensor = volumetric_kit::recon::sensor;

namespace {

constexpr std::uint32_t kW = 160, kH = 120;

// A recording of one frame per trigger, 100 ms apart on the host's clock,
// depth and I420 colour each sensor's own pattern: the depth held by the
// frame, the colour in a device buffer, as a decoder leaves it.
class Recording final : public sensor::IRgbdSensor {
 public:
  Recording(std::string id, int seed, sensor::SyncRole role) : seed_(seed) {
    info_.id = std::move(id);
    info_.role = role;
    info_.clock = sensor::ClockDomain::Host;
  }
  vkc::Status push(std::uint64_t sequence, const vkc::Device& device,
                   vkc::Allocator& allocator) {
    auto depth =
        std::make_shared<std::vector<std::uint16_t>>(std::size_t{kW} * kH);
    for (std::size_t i = 0; i < depth->size(); ++i) {
      (*depth)[i] = static_cast<std::uint16_t>(800 + 37 * seed_ + i % 211);
    }
    const std::size_t luma = std::size_t{kW} * kH;
    const std::size_t chroma = std::size_t{kW / 2} * (kH / 2);
    std::vector<std::uint8_t> planes(luma + 2 * chroma);
    for (std::size_t i = 0; i < luma; ++i) {
      planes[i] = static_cast<std::uint8_t>(30 * seed_ + i % 180);
    }
    std::fill_n(planes.begin() + luma, chroma, 110 + seed_);
    std::fill_n(planes.begin() + luma + chroma, chroma, 140 - seed_);
    VKC_ASSIGN(vkc::Buffer buffer,
               vr_test::upload_device_buffer(device, allocator, planes.data(),
                                             planes.size()));
    sensor::RgbdFrame f;
    f.depth = depth->data();
    f.depth_camera = {{kW, kH},
                      {150.0, 150.0, 79.5, 59.5},
                      {-0.2, 0.05, 0.001, -0.001, 0.0, 0.03, 0.0, 0.0}};
    f.min_depth = 0.1f;
    f.max_depth = 10.0f;
    f.color.device = std::make_shared<const vkc::Buffer>(std::move(buffer));
    f.color.offset[1] = luma;
    f.color.offset[2] = luma + chroma;
    f.color.stride[0] = kW;
    f.color.stride[1] = f.color.stride[2] = kW / 2;
    f.color.width = kW;
    f.color.height = kH;
    f.color_camera = {{kW, kH},
                      {140.0, 140.0, 79.5, 59.5},
                      {0.08, -0.1, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0}};
    f.depth_to_color[3] = glm::dvec4(-0.03, 0.0, 0.0, 1.0);
    f.timestamp_ns = (1 + sequence) * 100'000'000;
    f.sequence = sequence;
    f.pixels = depth;
    frames_.push_back(std::move(f));
    return {};
  }
  bool done = false;

  const sensor::SensorInfo& info() const noexcept override { return info_; }
  vkc::Status set_queue_depth(std::size_t) override { return {}; }
  vkc::Status start() override { return {}; }
  void stop() noexcept override {}
  vkc::Result<std::optional<sensor::RgbdFrame>> poll() override {
    return no_frame();
  }
  vkc::Status drain(std::vector<sensor::RgbdFrame>* out) override {
    for (auto& f : frames_) out->push_back(std::move(f));
    frames_.clear();
    return {};
  }
  bool exhausted() const noexcept override { return done && frames_.empty(); }
  sensor::SensorStats stats() const noexcept override { return {}; }

 private:
  sensor::SensorInfo info_;
  int seed_ = 0;
  std::vector<sensor::RgbdFrame> frames_;
};

camera::ArrayCalibration posed(const std::vector<std::string>& ids) {
  camera::ArrayCalibration c;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    camera::SensorCalibration s;
    s.id = ids[i];
    camera::Mat4d pose(camera::rotation_from_rodrigues({0.0, 0.3 * i, 0.0}));
    pose[3] = glm::dvec4(1.5 * i, 0.0, 0.0, 1.0);
    s.color_to_world = pose;
    c.sensors.push_back(s);
  }
  return c;
}

// A buffer's `count` values read back; empty if the copy failed.
template <typename T>
std::vector<T> read(vkc::Device& device, vkc::Allocator& allocator,
                    const vkc::Buffer& buffer, std::size_t count) {
  auto out = vr_test::read_back<T>(device, allocator, buffer, count);
  return out.ok() ? std::move(out).value() : std::vector<T>{};
}

int run(vkc::Device& device, vkc::Allocator& allocator) {
  const std::vector<std::string> ids = {"A", "B", "C"};
  std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
  std::vector<Recording*> recordings;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    auto r = std::make_unique<Recording>(
        ids[i], static_cast<int>(i),
        i == 0 ? sensor::SyncRole::Primary : sensor::SyncRole::Secondary);
    recordings.push_back(r.get());
    sensors.push_back(std::move(r));
  }
  sensor::SensorArray::Options o;
  o.calibration = posed(ids);
  // Each sensor's own depth offset: none, reads 4 mm too far, 2 mm too near.
  const double offsets[] = {0.0, 0.004, -0.002};
  for (std::size_t i = 0; i < ids.size(); ++i) {
    o.calibration.sensors[i].depth_offset_m = offsets[i];
  }
  o.device = &device;
  o.allocator = &allocator;
  auto opened = sensor::SensorArray::open(std::move(sensors), o);
  CHECK(opened.ok());
  sensor::SensorArray array = std::move(opened).value();
  CHECK(array.start().ok());

  // A pass alone per sensor, with that sensor's offset, and one with none.
  std::vector<sensor::GpuFramePrep> alone;
  for (const double offset : offsets) {
    sensor::GpuFramePrepConfig config;
    config.depth_offset_m = static_cast<float>(offset);
    auto made = sensor::GpuFramePrep::create(device, allocator, config);
    CHECK(made.ok());
    alone.push_back(std::move(made).value());
  }
  auto plain = sensor::GpuFramePrep::create(device, allocator);
  CHECK(plain.ok());
  const std::size_t n = std::size_t{kW} * kH;
  for (std::uint64_t seq = 0; seq < 3; ++seq) {
    for (std::size_t i = 0; i < recordings.size(); ++i) {
      if (seq == 1 && i == 2) continue;  // C's frame 1 never came
      CHECK(recordings[i]->push(seq, device, allocator).ok());
    }
    // A set missing C waits for it, then goes without it.
    auto set = array.poll_set();
    for (int ms = 0; ms < 200 && set.ok() && !set.value(); ++ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      set = array.poll_set();
    }
    CHECK(set.ok() && set.value() && set.value()->sequence == seq);
    auto prepared = array.process(*set.value());
    CHECK(prepared.ok());
    CHECK(prepared.value().sequence == seq);
    CHECK(prepared.value().frames.size() == ids.size());
    {
      // Recorded into a caller's batch: the same frames, once it has run.
      vkc::CommandBatch batch(device, allocator);
      auto recorded = array.record_process(batch, *set.value());
      CHECK(recorded.ok() && recorded->sequence == seq);
      CHECK(batch.submit().ok());
      for (std::size_t c = 0; c < ids.size(); ++c) {
        const auto& a = prepared.value().frames[c];
        const auto& b = recorded->frames[c];
        CHECK(a.has_value() == b.has_value());
        if (!a) continue;
        CHECK(read<float>(device, allocator, *b->depth, n) ==
              read<float>(device, allocator, *a->depth, n));
        CHECK(read<std::uint32_t>(device, allocator, *b->color, n) ==
              read<std::uint32_t>(device, allocator, *a->color, n));
      }
    }
    for (std::size_t c = 0; c < ids.size(); ++c) {
      const auto& frame = set.value()->frames[c];
      const auto& device_frame = prepared.value().frames[c];
      CHECK(frame.has_value() == device_frame.has_value());
      if (!frame) continue;
      // Posed by the calibration, depth through the sensor's extrinsic.
      const camera::Mat4d& color_to_world =
          *o.calibration.sensors[c].color_to_world;
      CHECK(device_frame->color_camera.cam_to_world ==
            vr::Mat4f(color_to_world));
      CHECK(device_frame->depth_camera.cam_to_world ==
            vr::Mat4f(color_to_world * frame->depth_to_color));
      // As GpuFramePrep makes it alone, with the sensor's own offset.
      auto one = alone[c].prepare(*frame);
      CHECK(one.ok());
      const auto depth =
          read<float>(device, allocator, *device_frame->depth, n);
      const auto color =
          read<std::uint32_t>(device, allocator, *device_frame->color, n);
      CHECK(depth.size() == n &&
            depth == read<float>(device, allocator, *one.value().depth, n));
      CHECK(color.size() == n &&
            color ==
                read<std::uint32_t>(device, allocator, *one.value().color, n));
      // And an offset is not a no-op: the depth differs from a pass's with
      // none exactly where the sensor has one.
      auto raw = plain.value().prepare(*frame);
      CHECK(raw.ok());
      CHECK((depth == read<float>(device, allocator, *raw.value().depth, n)) ==
            (offsets[c] == 0.0));
    }
  }
  for (Recording* r : recordings) r->done = true;
  CHECK(array.exhausted());

  // A set of another size is refused.
  sensor::FrameSet wrong;
  wrong.frames.resize(2);
  CHECK(array.process(wrong).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  {
    vkc::CommandBatch batch(device, allocator);
    CHECK(array.record_process(batch, wrong).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }
  // So is a device or an allocator without the other, and processing
  // without a device.
  for (const bool device_only : {true, false}) {
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> one;
    one.push_back(
        std::make_unique<Recording>("A", 0, sensor::SyncRole::Primary));
    sensor::SensorArray::Options bad = o;
    bad.calibration = {};
    if (device_only) {
      bad.allocator = nullptr;
    } else {
      bad.device = nullptr;
    }
    CHECK(sensor::SensorArray::open(std::move(one), bad).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }
  {
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> one;
    one.push_back(
        std::make_unique<Recording>("A", 0, sensor::SyncRole::Primary));
    sensor::SensorArray::Options plain;
    auto a = sensor::SensorArray::open(std::move(one), plain);
    CHECK(a.ok());
    sensor::FrameSet s;
    s.frames.resize(1);
    CHECK(a.value().process(s).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }
  return 0;
}

// The array's prep options reach every sensor's pass, whatever the array's
// size: arrays of 1, 3 and 5 sensors whose options name a second queue family
// for depth alone hand out every frame's depth CONCURRENT and its colour
// EXCLUSIVE (both EXCLUSIVE on a device with one family, where the pair
// collapses to the pass's own), and a depth count past the array or a depth
// family the device lacks is refused at open, as GpuFramePrep::create refuses
// it.
int test_prep_options(vkc::Device& device, vkc::Allocator& allocator) {
  const std::uint32_t own = device.queue_family();
  const std::optional<std::uint32_t> other =
      vr_test::other_queue_family(device);
  const VkSharingMode want =
      other ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
  for (const std::size_t count :
       {std::size_t{1}, std::size_t{3}, std::size_t{5}}) {
    std::vector<std::string> ids;
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> sensors;
    std::vector<Recording*> recordings;
    for (std::size_t i = 0; i < count; ++i) {
      ids.push_back(std::string(1, static_cast<char>('A' + i)));
      auto r = std::make_unique<Recording>(
          ids[i], static_cast<int>(i),
          i == 0 ? sensor::SyncRole::Primary : sensor::SyncRole::Secondary);
      recordings.push_back(r.get());
      sensors.push_back(std::move(r));
    }
    sensor::SensorArray::Options o;
    o.calibration = posed(ids);
    o.device = &device;
    o.allocator = &allocator;
    o.prep.depth_queue_families[0] = own;
    o.prep.depth_queue_families[1] = other.value_or(own);
    o.prep.depth_queue_family_count = 2;
    auto opened = sensor::SensorArray::open(std::move(sensors), o);
    CHECK(opened.ok());
    sensor::SensorArray array = std::move(opened).value();
    CHECK(array.start().ok());
    for (Recording* r : recordings) CHECK(r->push(0, device, allocator).ok());
    auto set = array.poll_set();
    for (int ms = 0; ms < 200 && set.ok() && !set.value(); ++ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      set = array.poll_set();
    }
    CHECK(set.ok() && set.value());
    auto prepared = array.process(*set.value());
    CHECK(prepared.ok() && prepared.value().frames.size() == count);
    for (const auto& frame : prepared.value().frames) {
      CHECK(frame.has_value() && frame->has_color());
      CHECK(frame->depth->sharing_mode() == want);
      CHECK(frame->color->sharing_mode() == VK_SHARING_MODE_EXCLUSIVE);
    }
  }
  sensor::SensorArray::Options too_many;
  too_many.device = &device;
  too_many.allocator = &allocator;
  too_many.prep.depth_queue_family_count =
      vkc::BufferDesc::kMaxQueueFamilies + 1;
  sensor::SensorArray::Options missing = too_many;
  missing.prep.depth_queue_families[0] = own;
  missing.prep.depth_queue_families[1] =
      static_cast<std::uint32_t>(device.caps().queue_families().size());
  missing.prep.depth_queue_family_count = 2;
  for (const sensor::SensorArray::Options* bad : {&too_many, &missing}) {
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> one;
    one.push_back(
        std::make_unique<Recording>("A", 0, sensor::SyncRole::Primary));
    const vkc::Status refused =
        sensor::SensorArray::open(std::move(one), *bad).status();
    CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
    CHECK(refused.message().find("GpuFramePrep::create, depth:") !=
          std::string::npos);
  }
  // A depth offset in the options, for every pass, is refused: each sensor's
  // comes from the calibration alone. With a device or without, and a NaN.
  for (const float offset : {0.004f, -0.001f, std::nanf("")}) {
    for (const bool with_device : {true, false}) {
      std::vector<std::unique_ptr<sensor::IRgbdSensor>> one;
      one.push_back(
          std::make_unique<Recording>("A", 0, sensor::SyncRole::Primary));
      sensor::SensorArray::Options set;
      set.calibration = posed({"A"});
      set.device = with_device ? &device : nullptr;
      set.allocator = with_device ? &allocator : nullptr;
      set.prep.depth_offset_m = offset;
      const vkc::Status refused =
          sensor::SensorArray::open(std::move(one), set).status();
      CHECK(refused.domain() == vkc::Status::Code::InvalidArgument);
      CHECK(refused.message().find("prep.depth_offset_m is set") !=
            std::string::npos);
    }
  }
  return 0;
}

int gpu_main(vr_test::GpuContext& gpu) {
  if (run(gpu.device, gpu.allocator) != 0) return 1;
  if (test_prep_options(gpu.device, gpu.allocator) != 0) return 1;
  std::printf("sensor_array_process: OK\n");
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
