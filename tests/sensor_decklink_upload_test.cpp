// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The DeckLink driver's GPU half, with no card: a capture that writes host
// pages as the card's DMA would (decklink::Capture), on the real device.
//   - the page pool: pages found by their address, made on demand, none past
//     the cap, reused once given back; refused (Unsupported, named) on a
//     device without VK_EXT_external_memory_host, where the test skips;
//   - the device pictures: reused once no frame holds them, none past the
//     cap;
//   - the driver over that capture: each picture's device bytes equal its
//     pages, at a small size and at 2160p; both SDK frames go back only
//     after the copy's fence (the pages are overwritten as they go back, as
//     the card's next picture would); the newest pair wins under a slow
//     poller; no device buffer is made after warm-up; the rectifier lets the
//     right picture go and the left stays with its frame; the sync contract
//     on what it handed out; a mode change fails poll, named; start fails
//     after lock_timeout naming the silent input or the cameras' offset, and
//     for a camera at another mode; a restart;
//   - with no calibration: raw pairs handed out, the info both cameras 0 x 0
//     and the pair the contract's, and the rectifier refusing the info,
//     named.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "driver.hpp"
#include "ports.hpp"
#include "stereo_contract.hpp"
#include "upload.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"

namespace dl = volumetric_kit::sensor::decklink;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace vkc = volumetric_kit::core;

namespace {

using std::chrono::milliseconds;

constexpr std::int64_t kMs = 1'000'000;
constexpr std::int64_t kPeriod = 33'333'333;

std::uint8_t pattern(std::size_t i, std::uint8_t seed) {
  return static_cast<std::uint8_t>((i * 7 + (i >> 8) * 13 + seed) & 0xFF);
}

// The card: pages taken as the SDK's allocator takes them, written as its
// DMA writes them, each frame delivered as its callback would; a frame
// handed back has its pages overwritten, as the card's next picture would.
struct Card {
  std::array<std::shared_ptr<dl::PagePool>, 2> pages;
  std::size_t bytes = 0;
  dl::CaptureSink* sink = nullptr;
  std::function<void(Card&)> on_start;
  std::shared_ptr<std::atomic<int>> handed_back =
      std::make_shared<std::atomic<int>>(0);

  bool deliver(int input, std::int64_t card_ns, std::uint8_t seed,
               bool no_signal = false) {
    if (sink == nullptr) return false;
    const std::shared_ptr<dl::PagePool>& pool = pages[input];
    if (pool->size() == 0 && !pool->set_size(bytes).ok()) return false;
    vkc::Result<std::uint8_t*> data = pool->take();
    if (!data.ok() || *data == nullptr) return false;  // the SDK drops it
    std::uint8_t* p = *data;
    for (std::size_t i = 0; i < bytes; ++i) p[i] = pattern(i, seed);
    dl::Arrival a;
    a.input = input;
    a.card_ns = card_ns;
    a.card_now_ns = card_ns + 3'600'000;
    a.host_now_ns = a.card_now_ns + 1'793'000'000'000'000'000;
    a.no_signal = no_signal;
    a.bytes = p;
    a.hold = std::shared_ptr<void>(
        p, [pool, n = bytes, back = handed_back](void* q) {
          std::memset(q, 0xEE, n);
          pool->give_back(q);
          ++*back;
        });
    sink->arrived(std::move(a));
    return true;
  }
  // A left and a right frame of pair k, the right 8.2 ms ahead.
  void pair(int k, std::int64_t right_apart = -8'200'000) {
    const std::int64_t t = 6'000 * 1000 * kMs + k * kPeriod;
    deliver(1, t + right_apart, static_cast<std::uint8_t>(2 * k + 1));
    deliver(0, t, static_cast<std::uint8_t>(2 * k));
  }
};

class FakeCapture final : public dl::Capture {
 public:
  explicit FakeCapture(Card& card) : card_(card) {}
  vkc::Status start(dl::CaptureSink& sink, milliseconds) override {
    card_.sink = &sink;
    if (card_.on_start) card_.on_start(card_);
    return {};
  }
  void stop() noexcept override { card_.sink = nullptr; }

 private:
  Card& card_;
};

// Counts the driver's wake-ups.
struct Woken {
  std::mutex mutex;
  std::condition_variable cv;
  int count = 0;
  std::function<void()> notify() {
    return [this] {
      {
        const std::lock_guard<std::mutex> lock(mutex);
        ++count;
      }
      cv.notify_all();
    };
  }
  int now() {
    const std::lock_guard<std::mutex> lock(mutex);
    return count;
  }
  bool wait_for(int n) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, std::chrono::seconds(10),
                       [&] { return count >= n; });
  }
};

std::optional<sensor::StereoFrame> poll_until(dl::Driver& driver) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < end) {
    vkc::Result<std::optional<sensor::StereoFrame>> f = driver.poll();
    if (!f.ok() || f->has_value()) return f.ok() ? *f : std::nullopt;
    std::this_thread::sleep_for(milliseconds(1));
  }
  return std::nullopt;
}

bool bytes_equal(vkc::test::Gpu& gpu, const vkc::Buffer& buffer,
                 std::size_t first, std::size_t count, std::uint8_t seed) {
  vkc::Result<std::vector<std::uint8_t>> got =
      vkc::test::read_back<std::uint8_t>(gpu.device, gpu.allocator, buffer,
                                         count, first);
  if (!got.ok()) return false;
  for (std::size_t i = 0; i < count; ++i) {
    if ((*got)[i] != pattern(first + i, seed)) {
      std::fprintf(stderr, "  byte %zu: 0x%02x, wrote 0x%02x\n", first + i,
                   (*got)[i], pattern(first + i, seed));
      return false;
    }
  }
  return true;
}

// A pinhole pair at w x h, 0.1 m apart along the left camera's +x.
sensor::StereoCalibration pinhole_pair(std::uint32_t w, std::uint32_t h) {
  sensor::StereoCalibration c;
  c.left.size = {w, h};
  c.left.intrinsics = {0.8 * w, 0.8 * w, 0.5 * w, 0.5 * h};
  c.right = c.left;
  c.left_to_right[3] = glm::dvec4(-0.1, 0.0, 0.0, 1.0);
  return c;
}

dl::Config config(std::uint32_t w, std::uint32_t h) {
  dl::Config c;
  c.left_id = "bm-1";
  c.right_id = "bm-2";
  c.left_input = "SDI 2";
  c.right_input = "SDI 1";
  c.width = w;
  c.height = h;
  c.calibration = pinhole_pair(w, h);
  c.lock_timeout = milliseconds(2000);
  return c;
}

vkc::Result<std::unique_ptr<dl::Driver>> make(vkc::test::Gpu& gpu,
                                              const dl::Config& c, Card& card) {
  card.bytes = dl::v210_row_bytes(c.width) * c.height;
  return dl::Driver::create(
      c, gpu.device, gpu.allocator,
      [&card](const std::array<std::shared_ptr<dl::PagePool>, 2>& pages)
          -> vkc::Result<std::unique_ptr<dl::Capture>> {
        card.pages = pages;
        return std::unique_ptr<dl::Capture>(new FakeCapture(card));
      });
}

int test_page_pool(vkc::test::Gpu& gpu) {
  auto made = dl::PagePool::create(gpu.device, gpu.allocator);
  CHECK(made.ok());
  dl::PagePool& pool = **made;
  CHECK(!pool.take().ok());  // no size yet
  CHECK(pool.set_size(10'000).ok());
  std::vector<std::uint8_t*> taken;
  for (std::size_t i = 0; i < dl::PagePool::kMaxBuffers; ++i) {
    vkc::Result<std::uint8_t*> p = pool.take();
    CHECK(p.ok() && *p != nullptr);
    taken.push_back(*p);
  }
  vkc::Result<std::uint8_t*> past = pool.take();
  CHECK(past.ok() && *past == nullptr);  // the SDK's E_OUTOFMEMORY
  CHECK(pool.made() == 8 && pool.in_use() == 8);
  for (std::uint8_t* p : taken) {
    const vkc::Buffer* buffer = pool.find(p);
    CHECK(buffer != nullptr && buffer->size() >= 10'000);
  }
  CHECK(pool.find(taken[0] + 16) == nullptr);
  CHECK(!pool.set_size(20'000).ok());  // pages made at another size
  pool.give_back(taken[3]);
  vkc::Result<std::uint8_t*> again = pool.take();
  CHECK(again.ok() && *again == taken[3]);
  CHECK(pool.made() == 8);

  std::shared_ptr<dl::DevicePictures> pictures =
      dl::DevicePictures::create(gpu.allocator, 4096, 2);
  {
    auto a = pictures->take();
    auto b = pictures->take();
    auto c = pictures->take();
    CHECK(a.ok() && *a != nullptr && b.ok() && *b != nullptr);
    CHECK(c.ok() && *c == nullptr);  // past the cap
    CHECK(pictures->in_use() == 2);
  }
  CHECK(pictures->in_use() == 0);
  auto d = pictures->take();
  CHECK(d.ok() && *d != nullptr && pictures->made() == 2);
  std::puts("pools: pages by address, the cap, reuse; device pictures too");
  return 0;
}

int test_driver(vkc::test::Gpu& gpu) {
  constexpr std::uint32_t kW = 96;
  constexpr std::uint32_t kH = 16;
  const dl::Config c = config(kW, kH);
  Card card;
  card.on_start = [](Card& k) { k.pair(0); };
  auto made = make(gpu, c, card);
  CHECK(made.ok());
  dl::Driver& driver = **made;
  const sensor::StereoInfo& info = driver.info();
  CHECK(info.id == "bm-1" && info.clock == sensor::ClockDomain::Host);
  CHECK(info.fps == 30 && info.max_skew_ns == c.max_skew_ns);
  CHECK(info.calibration.left.size.width == kW);
  CHECK(!info.color && !info.color_controls);
  Woken woken;
  CHECK(driver.set_frame_notify(woken.notify()).ok());
  CHECK(driver.start().ok());
  CHECK(!driver.set_frame_notify({}).ok());  // once started

  // The first pair: its pictures where the card wrote them, its SDK frames
  // back once copied.
  std::vector<sensor::StereoFrame> handed;
  const auto keep = [&handed](const sensor::StereoFrame& f) {
    sensor::StereoFrame light;
    light.timestamp_ns = f.timestamp_ns;
    light.skew_ns = f.skew_ns;
    light.sequence = f.sequence;
    handed.push_back(light);
  };
  std::optional<sensor::StereoFrame> f = poll_until(driver);
  CHECK(f.has_value());
  keep(*f);
  const std::size_t bytes = dl::v210_row_bytes(kW) * kH;
  for (const sensor::YuvImage* y : {&f->left, &f->right}) {
    CHECK(y->layout == sensor::YuvLayout::V210 && y->device != nullptr);
    CHECK(y->host == nullptr && y->offset[0] == 0 && y->stride[0] == 256);
    CHECK(y->width == kW && y->height == kH && !y->full_range);
    CHECK(y->queue_family == sensor::kQueueFamilyIgnored);
    CHECK(y->kr == 0.2126f && y->kb == 0.0722f);
    CHECK(y->chroma_location == sensor::ChromaLocation::Left);
  }
  CHECK(f->pixels == nullptr && f->right_pixels == nullptr);
  CHECK(f->skew_ns == -8'200'000 && f->sequence == 0);
  CHECK(f->color_encoding.transfer == sensor::ColorEncoding::Transfer::Bt709);
  // Handed back before this read, so the pages hold 0xEE now: the device
  // copy was made before they went back.
  CHECK(*card.handed_back == 2);
  CHECK(card.pages[0]->in_use() == 0 && card.pages[1]->in_use() == 0);
  CHECK(bytes_equal(gpu, *f->left.device, 0, bytes, 0));
  CHECK(bytes_equal(gpu, *f->right.device, 0, bytes, 1));
  f.reset();

  // A slow poller: four pairs published, each replacing the last unpolled;
  // the newest is taken.
  int woke = woken.now();
  for (int k = 1; k <= 4; ++k) {
    card.pair(k);
    CHECK(woken.wait_for(++woke));
  }
  f = poll_until(driver);
  CHECK(f && f->sequence == 4);
  keep(*f);
  CHECK(driver.stats().dropped == 3);
  CHECK(bytes_equal(gpu, *f->left.device, 0, bytes, 8));
  f.reset();

  // Warm: no device buffer made past the first few pairs.
  for (int k = 5; k < 15; ++k) {
    card.pair(k);
    CHECK(woken.wait_for(++woke));
    f = poll_until(driver);
    CHECK(f.has_value());
    keep(*f);
  }
  f.reset();
  const std::size_t warm = driver.pictures_made();
  for (int k = 15; k < 65; ++k) {
    card.pair(k);
    CHECK(woken.wait_for(++woke));
    f = poll_until(driver);
    CHECK(f.has_value());
    keep(*f);
  }
  std::printf(
      "  device pictures made %zu (after warm-up %zu); pages %zu, %zu\n",
      driver.pictures_made(), warm, driver.pages_made(0), driver.pages_made(1));
  CHECK(driver.pictures_made() == warm && warm <= 4);
  CHECK(driver.pages_made(0) <= 2 && driver.pages_made(1) <= 2);

  // The rectifier lets the right picture go; the left stays with the frame
  // made from the pair.
  auto rectifier = sensor::StereoRectifier::create(
      gpu.device, gpu.allocator, info,
      {{kW, kH}, 0.0f, sensor::StereoRectifier::Output::Luma8});
  if (!rectifier)
    std::fprintf(stderr, "%s\n", rectifier.status().message().c_str());
  CHECK(rectifier.ok());
  CHECK(driver.pictures_in_use() == 2);
  {
    vkc::Result<sensor::RectifiedPair> pair = (*rectifier)->rectify(*f);
    CHECK(pair.ok());
    f.reset();
    CHECK(driver.pictures_in_use() == 1);
    CHECK(pair->source.left.device != nullptr);
  }
  CHECK(driver.pictures_in_use() == 0);

  // The contract on everything handed out.
  sensor::SensorStats stats = driver.stats();
  std::printf(
      "  stats: received %llu delivered %llu dropped %llu failed %llu\n",
      static_cast<unsigned long long>(stats.received),
      static_cast<unsigned long long>(stats.delivered),
      static_cast<unsigned long long>(stats.dropped),
      static_cast<unsigned long long>(stats.failed));
  const std::string breach = stereo_contract::breach(
      info, stereo_contract::Opened::Calibrated, handed, stats);
  if (!breach.empty()) std::fprintf(stderr, "  %s\n", breach.c_str());
  CHECK(breach.empty());
  CHECK(stats.received == 65 && stats.failed == 0);

  // A camera changes mode: poll fails, naming the input and the mode.
  card.sink->mode_changed(0, "2160p24");
  vkc::Result<std::optional<sensor::StereoFrame>> changed = driver.poll();
  CHECK(!changed.ok());
  CHECK(changed.status().domain() == vkc::Status::Code::Unsupported);
  CHECK(changed.status().message().find("SDI 2 (bm-1)") != std::string::npos);
  CHECK(changed.status().message().find("sends 2160p24") != std::string::npos);
  std::printf("  %s\n", changed.status().message().c_str());

  // Stopped and started again: the counts restart, the mode fault clears.
  driver.stop();
  CHECK(driver.stats().received == 65);  // kept through the stop
  card.on_start = [](Card& k) { k.pair(100); };
  CHECK(driver.start().ok());
  f = poll_until(driver);
  CHECK(f && driver.stats().received == 1);
  f.reset();
  driver.stop();
  std::puts(
      "driver: the pages' bytes on the device, frames back after the "
      "fence, newest wins, warm reuse, the right let go, the contract, "
      "a mode change, a restart");
  return 0;
}

int test_start_failures(vkc::test::Gpu& gpu) {
  // Only the left camera sends: the right input is named.
  {
    dl::Config c = config(96, 16);
    c.lock_timeout = milliseconds(200);
    Card card;
    card.on_start = [](Card& k) {
      k.deliver(0, 0, 1);
      k.deliver(0, kPeriod, 2);
    };
    auto made = make(gpu, c, card);
    CHECK(made.ok());
    const vkc::Status s = (*made)->start();
    std::printf("  %s\n", s.message().c_str());
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message().find("no picture from SDI 1 (bm-2)") !=
          std::string::npos);
    CHECK(card.pages[0]->in_use() == 0);  // the held frame went back
  }
  // A genlock-sized bound on free-running cameras: the offset is named.
  {
    dl::Config c = config(96, 16);
    c.lock_timeout = milliseconds(200);
    c.max_skew_ns = 500'000;
    Card card;
    card.on_start = [](Card& k) {
      k.pair(0);
      k.pair(1);
    };
    auto made = make(gpu, c, card);
    CHECK(made.ok());
    const vkc::Status s = (*made)->start();
    std::printf("  %s\n", s.message().c_str());
    CHECK(s.domain() == vkc::Status::Code::IoError);
    CHECK(s.message().find("within max_skew_ns (500 us)") != std::string::npos);
    CHECK(s.message().find("-8.20 ms apart") != std::string::npos);
    CHECK((*made)->stats().failed >= 1);
  }
  // A camera at another mode as the inputs start.
  {
    Card card;
    card.on_start = [](Card& k) { k.sink->mode_changed(1, "2160p24"); };
    auto made = make(gpu, config(96, 16), card);
    CHECK(made.ok());
    const vkc::Status s = (*made)->start();
    CHECK(s.domain() == vkc::Status::Code::Unsupported);
    CHECK(s.message().find("SDI 1 (bm-2): the camera sends 2160p24") !=
          std::string::npos);
  }
  std::puts("start: the silent input, the cameras' offset, another mode");
  return 0;
}

// Cameras not calibrated yet (calib's lens mode): the driver starts and
// hands out raw pairs with their skew; its info says "no calibration",
// which the rectifier refuses, naming it.
int test_no_calibration(vkc::test::Gpu& gpu) {
  dl::Config c = config(96, 16);
  c.calibration.reset();
  Card card;
  card.on_start = [](Card& k) { k.pair(0); };
  auto made = make(gpu, c, card);
  CHECK(made.ok());
  dl::Driver& driver = **made;
  const sensor::StereoInfo& info = driver.info();
  CHECK(driver.start().ok());
  std::optional<sensor::StereoFrame> f = poll_until(driver);
  CHECK(f.has_value());
  // Both cameras exactly 0 x 0, and the pair the contract's.
  const std::string breach = stereo_contract::breach(
      info, stereo_contract::Opened::Uncalibrated, {*f}, driver.stats());
  if (!breach.empty()) std::fprintf(stderr, "  %s\n", breach.c_str());
  CHECK(breach.empty());
  CHECK(f->left.layout == sensor::YuvLayout::V210 && f->left.device);
  CHECK(f->right.layout == sensor::YuvLayout::V210 && f->right.device);
  CHECK(f->skew_ns == -8'200'000);
  const std::size_t bytes = dl::v210_row_bytes(96) * 16;
  CHECK(bytes_equal(gpu, *f->left.device, 0, bytes, 0));
  CHECK(bytes_equal(gpu, *f->right.device, 0, bytes, 1));
  auto refused = sensor::StereoRectifier::create(
      gpu.device, gpu.allocator, info,
      {{96, 16}, 0.0f, sensor::StereoRectifier::Output::Luma8});
  CHECK(!refused.ok());
  CHECK(refused.status().domain() == vkc::Status::Code::InvalidArgument);
  CHECK(refused.status().message().find("no calibration") != std::string::npos);
  f.reset();
  driver.stop();
  std::puts("no calibration: raw pairs handed out; the rectifier refuses it");
  return 0;
}

// One pair at the cameras' mode: 2160p v210, 22,118,400 bytes a picture.
int test_full_size(vkc::test::Gpu& gpu) {
  Card card;
  card.on_start = [](Card& k) { k.pair(0); };
  auto made = make(gpu, config(3840, 2160), card);
  CHECK(made.ok());
  CHECK((*made)->start().ok());
  std::optional<sensor::StereoFrame> f = poll_until(**made);
  CHECK(f.has_value());
  const std::size_t row = 10'240;
  CHECK(card.bytes == 22'118'400);
  for (const std::size_t y :
       {std::size_t{0}, std::size_t{1080}, std::size_t{2159}}) {
    CHECK(bytes_equal(gpu, *f->left.device, y * row, row, 0));
    CHECK(bytes_equal(gpu, *f->right.device, y * row, row, 1));
  }
  f.reset();
  (*made)->stop();
  std::puts("2160p: both pictures' rows on the device");
  return 0;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  if (!gpu.device.extension_enabled(
          VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    // The driver's answer on such a device, named; nothing else can run.
    auto refused = dl::PagePool::create(gpu.device, gpu.allocator);
    CHECK(!refused.ok());
    CHECK(refused.status().domain() == vkc::Status::Code::Unsupported);
    CHECK(refused.status().message().find("VK_EXT_external_memory_host") !=
          std::string::npos);
    std::printf("SKIP: %s does not offer VK_EXT_external_memory_host (%s)\n",
                gpu.physical.properties().deviceName,
                refused.status().message().c_str());
    return vkc::test::kSkipExitCode;
  }
  std::printf("device: %s\n", gpu.physical.properties().deviceName);
  if (test_page_pool(gpu) != 0 || test_driver(gpu) != 0 ||
      test_start_failures(gpu) != 0 || test_no_calibration(gpu) != 0 ||
      test_full_size(gpu) != 0) {
    return 1;
  }
  std::puts("sensor_decklink_upload: OK");
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
