// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A sensor array's sets prepared on the GPU in one batch: each frame comes
// out as GpuFramePrep::prepare makes it alone, posed by the array's
// calibration, and an empty slot stays empty; a device or an allocator given
// without the other, an array without a device and a set of another size are
// refused. Skips where no device is.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "buffer_readback.hpp"
#include "no_device.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "volumetric_kit/recon/sensor/array/sensor_array.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;
namespace sensor = volumetric_kit::recon::sensor;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr std::uint32_t kW = 160, kH = 120;

// A recording of one frame per sequence number, depth and I420 colour each
// sensor's own pattern: the depth held by the frame, the colour in a device
// buffer, as a decoder leaves it.
class Recording final : public sensor::IRgbdSensor {
 public:
  Recording(std::string id, int seed) : seed_(seed) {
    info_.id = std::move(id);
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
    f.timestamp_ns = 1000 + sequence;
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
    s.color_to_world =
        camera::Mat4d(camera::rotation_from_rodrigues({0.0, 0.3 * i, 0.0}));
    s.color_to_world[3] = glm::dvec4(1.5 * i, 0.0, 0.0, 1.0);
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
    auto r = std::make_unique<Recording>(ids[i], static_cast<int>(i));
    recordings.push_back(r.get());
    sensors.push_back(std::move(r));
  }
  sensor::SensorArray::Options o;
  o.sync = sensor::SyncMode::Sequence;
  o.calibration = posed(ids);
  o.device = &device;
  o.allocator = &allocator;
  auto opened = sensor::SensorArray::open(std::move(sensors), o);
  CHECK(opened.ok());
  sensor::SensorArray array = std::move(opened).value();
  CHECK(array.start().ok());

  auto alone = sensor::GpuFramePrep::create(device, allocator);
  CHECK(alone.ok());
  const std::size_t n = std::size_t{kW} * kH;
  for (std::uint64_t seq = 0; seq < 3; ++seq) {
    for (std::size_t i = 0; i < recordings.size(); ++i) {
      if (seq == 1 && i == 2) continue;  // C's frame 1 never came
      CHECK(recordings[i]->push(seq, device, allocator).ok());
    }
  }
  for (Recording* r : recordings) r->done = true;
  for (std::uint64_t seq = 0; seq < 3; ++seq) {
    auto set = array.poll_set();
    CHECK(set.ok() && set.value() && set.value()->sequence == seq);
    auto prepared = array.process(*set.value());
    CHECK(prepared.ok());
    CHECK(prepared.value().sequence == seq);
    CHECK(prepared.value().frames.size() == ids.size());
    for (std::size_t c = 0; c < ids.size(); ++c) {
      const auto& frame = set.value()->frames[c];
      const auto& device_frame = prepared.value().frames[c];
      CHECK(frame.has_value() == device_frame.has_value());
      if (!frame) continue;
      // Posed by the calibration, depth through the sensor's extrinsic.
      const vr::Mat4f pose(o.calibration.sensors[c].color_to_world);
      CHECK(device_frame->color_camera.cam_to_world == pose);
      CHECK(device_frame->depth_camera.cam_to_world ==
            vr::Mat4f(o.calibration.sensors[c].color_to_world *
                      frame->depth_to_color));
      // As GpuFramePrep makes it alone.
      auto one = alone.value().prepare(*frame);
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
    }
  }
  CHECK(array.exhausted());

  // A set of another size is refused.
  sensor::FrameSet wrong;
  wrong.frames.resize(2);
  CHECK(array.process(wrong).status().domain() ==
        vkc::Status::Code::InvalidArgument);
  // So is a device or an allocator without the other, and processing
  // without a device.
  for (const bool device_only : {true, false}) {
    std::vector<std::unique_ptr<sensor::IRgbdSensor>> one;
    one.push_back(std::make_unique<Recording>("A", 0));
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
    one.push_back(std::make_unique<Recording>("A", 0));
    sensor::SensorArray::Options plain;
    plain.sync = sensor::SyncMode::Sequence;
    auto a = sensor::SensorArray::open(std::move(one), plain);
    CHECK(a.ok());
    sensor::FrameSet s;
    s.frames.resize(1);
    CHECK(a.value().process(s).status().domain() ==
          vkc::Status::Code::InvalidArgument);
  }
  return 0;
}

}  // namespace

int main() {
  vkc::Result<vkc::Instance> instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  vkc::Result<vkc::PhysicalDeviceInfo> gpu =
      instance.value().select_physical_device(vr::device_requirements());
  if (!gpu) {
    return vr_test::no_device("no compute-capable device",
                              gpu.status().message());
  }
  vkc::Result<vkc::Device> device = vkc::Device::create(
      instance.value(), gpu.value(), vr::device_requirements());
  CHECK(device.ok());
  vkc::Result<vkc::Allocator> allocator =
      vkc::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  if (run(device.value(), allocator.value()) != 0) return 1;
  std::printf("sensor_array_process: OK\n");
  return 0;
}
