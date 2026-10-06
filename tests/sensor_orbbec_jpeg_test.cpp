// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's MJPEG colour decoder, with no camera: the committed
// JPEGs (tools/make_jpeg_fixtures.sh) wrapped in SDK frames, each paired with
// a depth frame, decoded on this machine's hardware onto the device. Every
// pair comes out with its depth and timestamps, its colour the pattern,
// BT.601 full range, carried in its frame and released with it. A pair
// missing a frame, an empty colour frame and a JPEG that does not decode each
// cost only themselves; a JPEG the hardware does not take stops the decoder;
// and a decoder slower than the camera skips pairs rather than falling behind.
//
// Where no device path opens the test skips; VR_TEST_HEVC_BACKEND, which CI
// sets on the legs that promise one, makes it fail instead.

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "device_picture_readback.hpp"
#include "jpeg_color.hpp"
#include "picture_frames.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

#include "no_device.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
namespace orbbec = volumetric_kit::recon::sensor::orbbec;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kHeight = 144;
constexpr std::uint64_t kStartUs = 1000000;
constexpr std::uint64_t kPeriodUs = 33333;  // 30 fps
constexpr const char* kJpeg = VR_JPEG_DATA "/patches_256x144.jpg";
constexpr const char* k422 = VR_JPEG_DATA "/patches_422_256x144.jpg";

// The device the decoders run on, set by main.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

// The fixture's pattern: patch p = (column + 3 * row) mod 8 over 32x72 luma
// patches; Y = 40 + 24p, U = 64 + 16 (3p mod 8), V = 64 + 16 (5p mod 8).
int patch(int column, int row) { return (column + 3 * row) % 8; }

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A pair as the SDK hands it over, dated in `slot`: MJPG colour holding
// `jpeg` (none when empty) and a depth frame.
std::shared_ptr<ob::FrameSet> pair(const std::vector<std::uint8_t>& jpeg,
                                   int slot, bool with_color = true,
                                   bool with_depth = true) {
  const std::uint64_t t =
      kStartUs + static_cast<std::uint64_t>(slot) * kPeriodUs;
  auto set = ob::FrameFactory::createFrameSet();
  if (with_depth) {
    auto depth =
        ob::FrameFactory::createVideoFrame(OB_FRAME_DEPTH, OB_FORMAT_Y16, 4, 4);
    ob::FrameHelper::setFrameDeviceTimestampUs(depth, t);
    set->pushFrame(depth);
  }
  if (with_color) {
    auto color =
        ob::FrameFactory::createFrame(OB_FRAME_COLOR, OB_FORMAT_MJPG,
                                      static_cast<std::uint32_t>(jpeg.size()));
    if (!jpeg.empty()) {
      color->updateData(jpeg.data(), static_cast<std::uint32_t>(jpeg.size()));
    }
    ob::FrameHelper::setFrameDeviceTimestampUs(color, t);
    set->pushFrame(color);
  }
  return set;
}

struct Run {
  std::vector<std::shared_ptr<ob::FrameSet>> out;
  std::uint64_t lost = 0;
  vkc::Status failure;
};

// Push `in`, wait for `expect` pairs out or the decoder to stop, and stop it.
// The queue holds all of `in`, so none is skipped.
Run run(const std::vector<std::shared_ptr<ob::FrameSet>>& in,
        std::size_t expect) {
  struct Collected {
    std::mutex mutex;
    std::vector<std::shared_ptr<ob::FrameSet>> sets;
  };
  auto collected = std::make_shared<Collected>();
  orbbec::JpegColorDecoder::Options options;
  options.depth = in.size();
  options.device = g_device;
  options.allocator = g_allocator;
  options.who = "test";
  auto decoder = orbbec::JpegColorDecoder::start(
      options, [collected](std::shared_ptr<ob::FrameSet> set) {
        std::lock_guard<std::mutex> lock(collected->mutex);
        collected->sets.push_back(std::move(set));
      });
  if (!decoder) {
    std::fprintf(stderr, "%s\n", decoder.status().message().c_str());
    std::exit(1);
  }
  for (const auto& p : in) decoder.value()->push(p);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    std::size_t n = 0;
    {
      std::lock_guard<std::mutex> lock(collected->mutex);
      n = collected->sets.size();
    }
    if (n >= expect || !decoder.value()->failure().ok() ||
        std::chrono::steady_clock::now() > deadline) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  // Let anything that should not come out have its chance to.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  Run r;
  r.failure = decoder.value()->failure();
  decoder.value()->stop();
  r.lost = decoder.value()->lost();
  std::lock_guard<std::mutex> lock(collected->mutex);
  r.out = collected->sets;
  return r;
}

// Y, U and V, packed, of a decoded pair's picture, read back through a batch.
struct Planes {
  std::vector<std::uint8_t> y, u, v;
};

Planes from_device(const sensor::DecodedPicture& p, vkc::Device& device,
                   vkc::Allocator& allocator) {
  std::vector<std::uint8_t> planes[3];
  vr_test::read_device_picture(p, device, allocator, planes);
  return {std::move(planes[0]), std::move(planes[1]), std::move(planes[2])};
}

// Each patch's centre holds the pattern's value, within the JPEG's loss.
int check_pattern(const Planes& p) {
  CHECK(p.y.size() == std::size_t{kWidth} * kHeight);
  CHECK(p.u.size() == std::size_t{kWidth / 2} * (kHeight / 2));
  CHECK(p.v.size() == p.u.size());
  const auto near = [](int got, int want) { return std::abs(got - want) <= 3; };
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 8; ++col) {
      const int k = patch(col, row);
      const std::size_t x = 32u * col + 16;
      const std::size_t y = 72u * row + 36;
      CHECK(near(p.y[y * kWidth + x], 40 + 24 * k));
      const std::size_t c = (y / 2) * (kWidth / 2) + x / 2;
      CHECK(near(p.u[c], 64 + 16 * ((3 * k) % 8)));
      CHECK(near(p.v[c], 64 + 16 * ((5 * k) % 8)));
    }
  }
  return 0;
}

// Each costs only itself: a pair without depth, one without colour, an empty
// colour frame, and bytes that are no JPEG.
int test_losses() {
  const std::vector<std::uint8_t> jpeg = read_file(kJpeg);
  const std::vector<std::uint8_t> garbage(64, 0x5a);
  CHECK(pair({}, 3)->getColorFrame()->getDataSize() == 0);
  const Run r = run(
      {pair(jpeg, 0), pair(jpeg, 1, true, false), pair(jpeg, 2, false, true),
       pair({}, 3), pair(garbage, 4), pair(jpeg, 5)},
      2);
  CHECK(r.out.size() == 2 && r.lost == 4);
  CHECK(r.out[1]->getColorFrame()->getTimeStampUs() ==
        kStartUs + 5 * kPeriodUs);
  return 0;
}

// A decoder slower than the camera skips pairs rather than falling behind:
// held in the sink with the queue one deep, each pair that arrives replaces
// the one waiting, counted dropped, not lost, and the newest is decoded next.
int test_skips_when_behind() {
  const std::vector<std::uint8_t> jpeg = read_file(kJpeg);
  struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool held = false, open = false;
    std::vector<std::shared_ptr<ob::FrameSet>> sets;
  };
  auto gate = std::make_shared<Gate>();
  const auto open = [gate] {
    {
      std::lock_guard<std::mutex> lock(gate->mutex);
      gate->open = true;
    }
    gate->changed.notify_all();
  };
  orbbec::JpegColorDecoder::Options options;
  options.depth = 1;
  options.device = g_device;
  options.allocator = g_allocator;
  options.who = "test";
  auto decoder = orbbec::JpegColorDecoder::start(
      options, [gate](std::shared_ptr<ob::FrameSet> set) {
        std::unique_lock<std::mutex> lock(gate->mutex);
        gate->sets.push_back(std::move(set));
        gate->held = true;
        gate->changed.notify_all();
        gate->changed.wait(lock, [&] { return gate->open; });
      });
  CHECK(decoder.ok());
  // Declared after the decoder, so a failed check opens the gate before the
  // decoder's destructor joins the thread it holds.
  struct Opener {
    std::function<void()> open;
    ~Opener() { open(); }
  } opener{open};
  decoder.value()->push(pair(jpeg, 0));
  {
    std::unique_lock<std::mutex> lock(gate->mutex);
    CHECK(gate->changed.wait_for(lock, std::chrono::seconds(10),
                                 [&] { return gate->held; }));
  }
  for (int f = 1; f < 5; ++f) decoder.value()->push(pair(jpeg, f));
  CHECK(decoder.value()->dropped() == 3);  // 1, 2 and 3, each replaced
  CHECK(decoder.value()->lost() == 0);
  open();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    {
      std::lock_guard<std::mutex> lock(gate->mutex);
      if (gate->sets.size() >= 2) break;
    }
    CHECK(std::chrono::steady_clock::now() < deadline);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  decoder.value()->stop();
  CHECK(decoder.value()->dropped() == 3 && decoder.value()->lost() == 0);
  CHECK(gate->sets.size() == 2);
  CHECK(gate->sets[1]->getColorFrame()->getTimeStampUs() ==
        kStartUs + 4 * kPeriodUs);
  return 0;
}

// Every pair out, in order, with its depth, dated as it came, its colour the
// pattern, on the device: BT.601 full range, chroma centred, carried in its
// frame and released with it.
int test_device() {
  const std::vector<std::uint8_t> jpeg = read_file(kJpeg);
  CHECK(!jpeg.empty());
  std::vector<std::shared_ptr<ob::FrameSet>> in;
  for (int f = 0; f < 5; ++f) in.push_back(pair(jpeg, f));
  Run r = run(in, 5);
  CHECK(r.out.size() == 5 && r.lost == 0 && r.failure.ok());
  std::shared_ptr<const void> held;
  for (int f = 0; f < 5; ++f) {
    const auto& set = *r.out[static_cast<std::size_t>(f)];
    const std::uint64_t t =
        kStartUs + static_cast<std::uint64_t>(f) * kPeriodUs;
    const auto depth = set.getDepthFrame();
    const auto color = set.getColorFrame();
    CHECK(depth != nullptr && color != nullptr);
    CHECK(depth->getTimeStampUs() == t && color->getTimeStampUs() == t);
    const std::optional<sensor::DecodedPicture> p =
        orbbec::device_picture(*color);
    CHECK(p.has_value());
    CHECK(p->width == kWidth && p->height == kHeight);
    CHECK(p->matrix == sensor::VideoColorMatrix::Bt601 && p->full_range);
    CHECK(p->chroma_location == sensor::ChromaLocation::Center);
    sensor::YuvImage image;
    orbbec::place_device_color(*p, &image);
    CHECK(image.chroma_location == sensor::ChromaLocation::Center);
    CHECK(check_pattern(from_device(*p, *g_device, *g_allocator)) == 0);
    if (p->device) {
      held = p->device;
    } else {
      held = p->image[0];
    }
  }
  CHECK(held.use_count() > 1);  // the frame still holds it
  r.out.clear();
  CHECK(held.use_count() == 1);  // and it went with the frame
  return 0;
}

// A decoder with no device to decode onto does not start.
int test_start_needs_device() {
  orbbec::JpegColorDecoder::Options options;
  options.who = "test";
  auto decoder = orbbec::JpegColorDecoder::start(
      options, [](std::shared_ptr<ob::FrameSet>) {});
  CHECK(decoder.status().domain() == vkc::Status::Code::Unsupported);
  return 0;
}

// A 4:2:2 JPEG, which the hardware does not take: the decoder stops for good,
// Unsupported, and hands nothing on.
int test_unsupported_jpeg() {
  const std::vector<std::uint8_t> jpeg = read_file(k422);
  CHECK(!jpeg.empty());
  const Run r = run({pair(jpeg, 0), pair(jpeg, 1)}, 1);
  CHECK(r.failure.domain() == vkc::Status::Code::Unsupported);
  CHECK(r.out.empty());
  return 0;
}

}  // namespace

int main() {
  // The SDK writes a log file into the working directory unless told not to.
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
  if (test_start_needs_device() != 0) return 1;

  auto instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  auto physical =
      instance.value().select_physical_device(vr::device_requirements());
  if (!physical) {
    return vr_test::no_device("no compute-capable device",
                              physical.status().message());
  }
  auto device = vkc::Device::create(instance.value(), physical.value(),
                                    vr::device_requirements());
  CHECK(device.ok());
  auto allocator =
      vkc::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  g_device = &device.value();
  g_allocator = &allocator.value();
  {
    sensor::JpegDecoder::Options options;
    options.device = g_device;
    options.allocator = g_allocator;
    auto probe = sensor::JpegDecoder::create(options);
    if (!probe) {
      if (probe.status().domain() != vkc::Status::Code::Unsupported) {
        std::fprintf(stderr, "FAIL: %s\n", probe.status().message().c_str());
        return 1;
      }
      return vr_test::no_decoder(probe.status().message());
    }
  }

  if (test_device() != 0) return 1;
  if (test_losses() != 0) return 1;
  if (test_skips_when_behind() != 0) return 1;
  if (test_unsupported_jpeg() != 0) return 1;
  std::puts("sensor_orbbec_jpeg: OK");
  return 0;
}
