// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// After a failed fence wait the device may still run a batch, so what the
// batch reads goes with it (the core's CommandBatch::retain): its holder may
// drop it at once, the device keeps it, and it goes only once the device's
// destruction has waited for the work. Each reader here runs on a device of
// the case's own, its fence wait refused once (vkWaitForFences, interposed;
// the work itself runs, and a wait_idle tells the validation layer so):
//   - GpuFramePrep::prepare_batch: the frame's device depth and planes;
//     GpuFramePrep::record holds them in the caller's batch until it has run;
//   - LumaPass: the picture's buffer;
//   - PictureReadback: the picture's buffer, and a read after the failed one
//     reads right, into staging of its own;
//   - StereoRectifier: both views' buffers;
//   - the DeckLink upload: the two device pictures, out of their pool, and the
//     SDK's two frames, out of the card's reach, where the device offers
//     VK_EXT_external_memory_host.

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "ports.hpp"
#include "upload.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"
#include "volumetric_kit/sensor/prep/luma_pass.hpp"
#include "volumetric_kit/sensor/prep/picture_readback.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"

namespace dl = volumetric_kit::sensor::decklink;
namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

namespace {

// Atomic: the DeckLink upload waits on its own thread. The next wait is
// refused while refuse_wait is set, which that wait clears.
std::atomic<bool> refuse_wait{false};
std::atomic<int> refused{0};

}  // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL
vkWaitForFences(VkDevice device, std::uint32_t count, const VkFence* fences,
                VkBool32 wait_all, std::uint64_t timeout) {
  if (refuse_wait.exchange(false)) {
    ++refused;
    return VK_ERROR_UNKNOWN;
  }
  static const auto real = reinterpret_cast<PFN_vkWaitForFences>(
      dlsym(RTLD_NEXT, "vkWaitForFences"));
  if (real == nullptr) std::abort();
  return real(device, count, fences, wait_all, timeout);
}

namespace {

// A device and allocator of the case's own, destroyed to see what the device
// kept go: the allocator first, which its buffers outlive.
struct Own {
  std::optional<vkc::Device> device;
  std::optional<vkc::Allocator> allocator;

  int make(vkc::test::GpuContext& gpu) {
    vkc::Result<vkc::Device> d = vkc::Device::create(
        gpu.instance, gpu.physical, sensor::device_requirements());
    CHECK(d.ok());
    device.emplace(*std::move(d));
    vkc::Result<vkc::Allocator> a =
        vkc::Allocator::create(gpu.instance.handle(), *device);
    CHECK(a.ok());
    allocator.emplace(*std::move(a));
    return 0;
  }
  void destroy() {
    allocator.reset();
    device.reset();
  }
};

std::shared_ptr<const vkc::Buffer> device_buffer(vkc::Allocator& allocator,
                                                 VkDeviceSize bytes) {
  vkc::Result<vkc::Buffer> made = vkc::device_storage_buffer(allocator, bytes);
  if (!made.ok()) return nullptr;
  return std::make_shared<const vkc::Buffer>(*std::move(made));
}

// A 16 x 16 NV12 picture in a device buffer of its own.
sensor::YuvImage nv12(vkc::Allocator& allocator) {
  sensor::YuvImage p;
  p.layout = sensor::YuvLayout::Nv12;
  p.device = device_buffer(allocator, 384);
  p.offset[1] = 256;
  p.stride[0] = p.stride[1] = 16;
  p.width = p.height = 16;
  return p;
}

// A 16 x 16 frame with its depth and NV12 colour on the device.
sensor::RgbdFrame device_frame(vkc::Allocator& allocator) {
  sensor::RgbdFrame f;
  f.min_depth = 0.1f;
  f.max_depth = 4.0f;
  f.depth_camera.size = {16, 16};
  f.depth_camera.intrinsics = {16.0, 16.0, 8.0, 8.0};
  f.color_camera = f.depth_camera;
  f.depth_device.buffer = device_buffer(allocator, 512);
  f.color = nv12(allocator);
  return f;
}

// The wait is refused once: the call under test fails, having waited once.
template <typename Call>
bool fails_its_wait(Call&& call) {
  refused = 0;
  refuse_wait = true;
  const bool ok = call();
  refuse_wait = false;
  return !ok && refused == 1;
}

int prep(vkc::test::GpuContext& gpu) {
  // record: the caller's batch holds the frame's device depth and planes
  // until it has run.
  {
    auto made = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
    CHECK(made.ok());
    std::optional<sensor::RgbdFrame> frame = device_frame(gpu.allocator);
    const std::weak_ptr<const vkc::Buffer> depth = frame->depth_device.buffer;
    const std::weak_ptr<const vkc::Buffer> color = frame->color.device;
    vkc::CommandBatch batch(gpu.device, gpu.allocator);
    vkc::Result<sensor::DeviceFrame> out = made->record(batch, *frame);
    CHECK(out.ok());
    frame.reset();
    CHECK(!depth.expired() && !color.expired());
    CHECK(batch.submit().ok());
    CHECK(depth.expired() && color.expired());
  }
  // prepare_batch: a failed wait leaves them to the device.
  Own own;
  CHECK(own.make(gpu) == 0);
  std::weak_ptr<const vkc::Buffer> depth, color;
  {
    std::vector<sensor::GpuFramePrep> preps;
    auto made = sensor::GpuFramePrep::create(*own.device, *own.allocator);
    CHECK(made.ok());
    preps.push_back(*std::move(made));
    std::vector<std::optional<sensor::RgbdFrame>> frames(1);
    frames[0] = device_frame(*own.allocator);
    depth = frames[0]->depth_device.buffer;
    color = frames[0]->color.device;
    CHECK(fails_its_wait([&] {
      return sensor::GpuFramePrep::prepare_batch(preps, frames).ok();
    }));
    frames.clear();
    CHECK(own.device->wait_idle().ok());
    CHECK(!depth.expired() && !color.expired());
  }
  CHECK(!depth.expired() && !color.expired());
  own.destroy();
  CHECK(depth.expired() && color.expired());
  std::puts("prep: record holds the frame; a failed wait leaves it");
  return 0;
}

int luma(vkc::test::GpuContext& gpu) {
  Own own;
  CHECK(own.make(gpu) == 0);
  std::weak_ptr<const vkc::Buffer> picture;
  {
    auto pass = sensor::LumaPass::create(*own.device, *own.allocator);
    CHECK(pass.ok());
    std::optional<sensor::YuvImage> p = nv12(*own.allocator);
    picture = p->device;
    CHECK(fails_its_wait([&] { return (*pass)->run({&*p}).ok(); }));
    p.reset();
    CHECK(own.device->wait_idle().ok());
    CHECK(!picture.expired());
  }
  own.destroy();
  CHECK(picture.expired());
  std::puts("luma: a failed wait leaves the picture");
  return 0;
}

int readback(vkc::test::GpuContext& gpu) {
  Own own;
  CHECK(own.make(gpu) == 0);
  std::weak_ptr<const vkc::Buffer> picture;
  {
    auto made = sensor::PictureReadback::create(*own.device, *own.allocator);
    CHECK(made.ok());
    sensor::PictureReadback& readback = **made;
    std::optional<sensor::YuvImage> p = nv12(*own.allocator);
    picture = p->device;
    CHECK(fails_its_wait([&] {
      return readback.read(*p, sensor::PictureReadback::Planes::All).ok();
    }));
    p.reset();
    CHECK(own.device->wait_idle().ok());
    CHECK(!picture.expired());

    // The next read stages afresh, and reads what the picture holds.
    std::vector<std::uint8_t> bytes(384);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<std::uint8_t>(i * 7 + 3);
    }
    vkc::Result<vkc::Buffer> filled =
        vkc::device_storage_buffer(*own.allocator, bytes.size());
    CHECK(filled.ok());
    vkc::CommandBatch upload(*own.device, *own.allocator);
    CHECK(upload.upload(*filled, 0, bytes.data(), bytes.size()).ok());
    CHECK(upload.submit().ok());
    sensor::YuvImage next = nv12(*own.allocator);
    next.device = std::make_shared<const vkc::Buffer>(*std::move(filled));
    vkc::Result<sensor::PictureView> view =
        readback.read(next, sensor::PictureReadback::Planes::All);
    CHECK(view.ok());
    CHECK(std::memcmp(view->luma, bytes.data(), 256) == 0);
    CHECK(std::memcmp(view->chroma[0], bytes.data() + 256, 128) == 0);
  }
  own.destroy();
  CHECK(picture.expired());
  std::puts("readback: a failed wait leaves the picture; the next read reads");
  return 0;
}

int stereo(vkc::test::GpuContext& gpu) {
  Own own;
  CHECK(own.make(gpu) == 0);
  std::weak_ptr<const vkc::Buffer> left, right;
  {
    sensor::StereoInfo info;
    sensor::StereoCalibration& c = info.calibration;
    c.left.size = {16, 16};
    c.left.intrinsics = {12.8, 12.8, 8.0, 8.0};
    c.right = c.left;
    c.left_to_right[3] = glm::dvec4(-0.1, 0.0, 0.0, 1.0);
    auto made = sensor::StereoRectifier::create(
        *own.device, *own.allocator, info,
        {{16, 16}, 0.0f, sensor::StereoRectifier::Output::Luma8});
    CHECK(made.ok());
    std::optional<sensor::StereoFrame> frame;
    frame.emplace();
    for (sensor::YuvImage* view : {&frame->left, &frame->right}) {
      view->layout = sensor::YuvLayout::Luma;
      view->device = device_buffer(*own.allocator, 256);
      view->stride[0] = 16;
      view->width = view->height = 16;
    }
    left = frame->left.device;
    right = frame->right.device;
    CHECK(fails_its_wait([&] { return (*made)->rectify(*frame).ok(); }));
    frame.reset();
    CHECK(own.device->wait_idle().ok());
    CHECK(!left.expired() && !right.expired());
  }
  own.destroy();
  CHECK(left.expired() && right.expired());
  std::puts("stereo: a failed wait leaves both views");
  return 0;
}

int decklink(vkc::test::GpuContext& gpu) {
  Own own;
  CHECK(own.make(gpu) == 0);
  if (!own.device->extension_enabled(
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    std::puts("decklink: skipped, no VK_EXT_external_memory_host");
    own.destroy();
    return 0;
  }
  constexpr std::uint32_t kW = 96;
  constexpr std::uint32_t kH = 16;
  const std::size_t row = dl::v210_row_bytes(kW);
  const std::size_t bytes = row * kH;
  // The frames handed back, counted rather than watched by a weak_ptr: one
  // would keep the hold's deleter, and the pages it holds, past the device.
  const auto handed_back = std::make_shared<std::atomic<int>>(0);
  std::weak_ptr<dl::DevicePictures> pool;
  {
    std::array<std::shared_ptr<dl::PagePool>, 2> pages;
    for (std::shared_ptr<dl::PagePool>& p : pages) {
      auto made = dl::PagePool::create(*own.device, *own.allocator);
      CHECK(made.ok() && (*made)->set_size(bytes).ok());
      p = *std::move(made);
    }
    std::shared_ptr<dl::DevicePictures> pictures =
        dl::DevicePictures::create(*own.allocator, bytes);
    pool = pictures;
    dl::Uploader uploader(*own.device, *own.allocator, pages, pictures,
                          dl::Picture{kW, kH, row});
    pictures.reset();
    uploader.start([] {});
    // The card's two frames, each held as the SDK holds one.
    dl::Pair pair;
    dl::Arrival* sides[2] = {&pair.left, &pair.right};
    for (int i = 0; i < 2; ++i) {
      vkc::Result<std::uint8_t*> data = pages[i]->take();
      CHECK(data.ok() && *data != nullptr);
      std::memset(*data, 0x40 + i, bytes);
      sides[i]->input = i;
      sides[i]->bytes = *data;
      sides[i]->hold = std::shared_ptr<void>(
          *data, [p = pages[i], back = handed_back](void* q) {
            p->give_back(q);
            ++*back;
          });
    }
    refused = 0;
    refuse_wait = true;
    uploader.submit(std::move(pair));
    const auto end =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (uploader.fault().ok() && std::chrono::steady_clock::now() < end) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    refuse_wait = false;
    CHECK(!uploader.fault().ok() && refused == 1);
    uploader.stop();
    CHECK(*handed_back == 0);
    CHECK(pages[0]->in_use() == 1 && pages[1]->in_use() == 1);
    const std::shared_ptr<dl::DevicePictures> kept = pool.lock();
    CHECK(kept != nullptr && kept->in_use() == 2);
  }
  CHECK(own.device->wait_idle().ok());
  CHECK(*handed_back == 0 && !pool.expired());
  own.destroy();
  CHECK(*handed_back == 2 && pool.expired());
  std::puts("decklink: a failed wait leaves the pictures and the frames");
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  CHECK(prep(gpu) == 0);
  CHECK(luma(gpu) == 0);
  CHECK(readback(gpu) == 0);
  CHECK(stereo(gpu) == 0);
  CHECK(decklink(gpu) == 0);
  std::puts("sensor_failed_wait: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
