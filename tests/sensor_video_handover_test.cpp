// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// NVIDIA's decoders hand their pictures over on the GPU (VS_WITH_CUDA): a
// picture comes out before CUDA's copy or decode into it has run, carrying the
// timeline value CUDA sets once it has (YuvImage::ready), and every reader
// here waits for that value on the device. To show it, the decoders' sources
// are compiled into this test over a CUDA driver table that can hold their
// stream: while a hold is armed, the next NVDEC copy, or the next JPEG
// decode's stream work, first waits on the GPU (cuStreamWaitValue32) for a
// flag the test raises. So held, receive and decode return with the value
// unset, and a batch that records prep and submits with ready_points, prepare,
// PictureReadback and LumaPass, each started before the flag is raised, read
// what the same picture reads unheld, to the bit. With CUDA three pictures
// behind, the next receive waits on the host for the oldest; a decoder
// destroyed while its stream is held waits for it, so the picture it handed
// out reads as unheld. A watchdog raises a flag the test has not after 10 s,
// so a decoder that waits on the host where it should not fails rather than
// hangs. The table also drops ready values' signals and reports a faulted
// stream, as a GPU fault leaves them: the error the decoder then returns comes
// back with every value it handed out set, so no reader waits for good.
//
// Where no device path opens the test skips, or fails where
// VS_TEST_HEVC_BACKEND promises one, as the decoder tests do; the JPEG part
// skips on a GPU without the hardware JPEG engine.

#include <cuda.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "cuda_driver.hpp"
#include "decoder_test.hpp"
#include "device_picture_readback.hpp"
#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/prep/gpu_frame_prep.hpp"
#include "volumetric_kit/sensor/prep/luma_pass.hpp"
#include "volumetric_kit/sensor/prep/picture_readback.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/testing/hevc_clip.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"
#include "volumetric_kit/sensor/video/hevc_decoder.hpp"
#include "volumetric_kit/sensor/video/jpeg_decoder.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace video = volumetric_kit::sensor::video;

namespace volumetric_kit::sensor::video {
// cuda_driver.cpp's loader, compiled for this test under this name
// (tests/CMakeLists.txt), which this test's cuda_driver() wraps.
const CudaDriver* vs_test_real_cuda_driver();
}  // namespace volumetric_kit::sensor::video

namespace {

using namespace std::chrono_literals;

// --- The held driver ---------------------------------------------------------

// What the next hold holds: nothing, the next NVDEC copy (cuMemcpy2DAsync,
// which takes the stream), or the next JPEG decode's stream work, queued once
// it makes its context current.
enum Armed : int { kNothing = 0, kCopy = 1, kJpeg = 2 };

struct Held {
  const video::CudaDriver* real = nullptr;
  decltype(&::cuStreamWaitValue32) wait_value = nullptr;
  decltype(&::cuStreamWriteValue32) write_value = nullptr;
  decltype(&::cuMemAlloc) mem_alloc = nullptr;
  std::atomic<int> armed{kNothing};
  std::atomic<std::uint32_t> number{
      0};                // the hold waits for the flag to reach it
  CUdeviceptr flag = 0;  // a word in device memory
  CUstream jpeg_stream = nullptr;
  std::atomic<CUstream> last_stream{nullptr};  // the last stream made
  std::atomic<CUresult> failed{CUDA_SUCCESS};  // a hold that did not queue
  // As a stream that failed leaves them: the next `drops` ready values'
  // signals dropped, never queued, and cuStreamQuery reporting a fault.
  std::atomic<int> drops{0};
  std::atomic<bool> faulted{false};
};

Held& held() {
  static Held h;
  return h;
}

CUresult hold_on(CUstream stream) {
  Held& h = held();
  const CUresult r =
      h.wait_value(stream, h.flag, h.number.load(), CU_STREAM_WAIT_VALUE_GEQ);
  if (r != CUDA_SUCCESS) h.failed.store(r);
  return r;
}

CUresult held_copy(const CUDA_MEMCPY2D* copy, CUstream stream) {
  int want = kCopy;
  if (held().armed.compare_exchange_strong(want, kNothing)) {
    const CUresult r = hold_on(stream);
    if (r != CUDA_SUCCESS) return r;
  }
  return held().real->cuMemcpy2DAsync(copy, stream);
}

CUresult held_push(CUcontext context) {
  const CUresult r = held().real->cuCtxPushCurrent(context);
  int want = kJpeg;
  if (r == CUDA_SUCCESS &&
      held().armed.compare_exchange_strong(want, kNothing)) {
    hold_on(held().jpeg_stream);
  }
  return r;
}

CUresult dropped_signal(const CUexternalSemaphore* semaphores,
                        const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* params,
                        unsigned int count, CUstream stream) {
  int left = held().drops.load();
  while (left > 0 && !held().drops.compare_exchange_weak(left, left - 1)) {
  }
  if (left > 0) return CUDA_SUCCESS;
  return held().real->cuSignalExternalSemaphoresAsync(semaphores, params, count,
                                                      stream);
}

CUresult faulted_query(CUstream stream) {
  if (held().faulted.load()) return CUDA_ERROR_ILLEGAL_ADDRESS;
  return held().real->cuStreamQuery(stream);
}

CUresult held_stream_create(CUstream* stream, unsigned int flags) {
  const CUresult r = held().real->cuStreamCreate(stream, flags);
  if (r == CUDA_SUCCESS) held().last_stream.store(*stream);
  return r;
}

}  // namespace

namespace volumetric_kit::sensor::video {

// The decoders' table: libcuda's, with the copy, the context push, the ready
// values' signal, the stream query and stream creation wrapped as above.
const CudaDriver* cuda_driver() {
  static const std::optional<CudaDriver> table =
      []() -> std::optional<CudaDriver> {
    const CudaDriver* real = vs_test_real_cuda_driver();
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (real == nullptr || lib == nullptr) return std::nullopt;
    Held& h = held();
    h.real = real;
    h.wait_value = reinterpret_cast<decltype(h.wait_value)>(
        dlsym(lib, VS_CUDA_STRING(cuStreamWaitValue32)));
    h.write_value = reinterpret_cast<decltype(h.write_value)>(
        dlsym(lib, VS_CUDA_STRING(cuStreamWriteValue32)));
    h.mem_alloc = reinterpret_cast<decltype(h.mem_alloc)>(
        dlsym(lib, VS_CUDA_STRING(cuMemAlloc)));
    if (h.wait_value == nullptr || h.write_value == nullptr ||
        h.mem_alloc == nullptr) {
      return std::nullopt;
    }
    CudaDriver t = *real;
    t.cuMemcpy2DAsync = &held_copy;
    t.cuCtxPushCurrent = &held_push;
    t.cuStreamCreate = &held_stream_create;
    t.cuSignalExternalSemaphoresAsync = &dropped_signal;
    t.cuStreamQuery = &faulted_query;
    return t;
  }();
  return table ? &*table : nullptr;
}

}  // namespace volumetric_kit::sensor::video

namespace {

// The flag's word and the stream that raises it, in the GPU's primary
// context: a stream of its own, so nothing the decoders queue holds it.
struct Flag {
  std::unique_ptr<sensor::CudaContext> context;
  CUstream raiser = nullptr;

  ~Flag() {
    const video::CudaDriver* cu = video::cuda_driver();
    if (context == nullptr) return;
    cu->cuCtxPushCurrent(context->handle());
    if (raiser != nullptr) {
      cu->cuStreamSynchronize(raiser);
      cu->cuStreamDestroy(raiser);
    }
    if (held().flag != 0) cu->cuMemFree(held().flag);
    held().flag = 0;
    CUcontext popped = nullptr;
    cu->cuCtxPopCurrent(&popped);
  }

  // The flag to @p value, and back once the raise has run.
  CUresult set(std::uint32_t value) const {
    const video::CudaDriver* cu = video::cuda_driver();
    CUresult r = cu->cuCtxPushCurrent(context->handle());
    if (r != CUDA_SUCCESS) return r;
    r = held().write_value(raiser, held().flag, value, 0);
    if (r == CUDA_SUCCESS) r = cu->cuStreamSynchronize(raiser);
    CUcontext popped = nullptr;
    cu->cuCtxPopCurrent(&popped);
    return r;
  }
};

vkc::Result<std::unique_ptr<Flag>> make_flag(const vkc::Device& device) {
  auto flag = std::make_unique<Flag>();
  VKC_ASSIGN(flag->context, sensor::CudaContext::open(device));
  const video::CudaDriver* cu = video::cuda_driver();
  CUresult r = cu->cuCtxPushCurrent(flag->context->handle());
  if (r == CUDA_SUCCESS) {
    r = cu->cuStreamCreate(&flag->raiser, CU_STREAM_NON_BLOCKING);
    if (r == CUDA_SUCCESS) r = held().mem_alloc(&held().flag, 4);
    CUcontext popped = nullptr;
    cu->cuCtxPopCurrent(&popped);
  }
  if (r == CUDA_SUCCESS) r = flag->set(0);
  if (r != CUDA_SUCCESS) {
    return video::cuda_error("held flag", r, "making the flag");
  }
  return flag;
}

// One hold, armed at construction: the flag stays below its number until
// raise(), or until `after` has passed, when the watchdog raises it.
class Hold {
 public:
  Hold(const Flag& flag, Armed what, std::chrono::milliseconds after)
      : flag_(flag), number_(held().number.load() + 1) {
    held().number.store(number_);
    held().failed.store(CUDA_SUCCESS);
    held().armed.store(what);
    watchdog_ = std::thread([this, after] {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, after, [this] { return now_; });
      raised_result_ = flag_.set(number_);
      raised_ = true;
      cv_.notify_all();
    });
  }
  ~Hold() {
    raise();
    watchdog_.join();
    held().armed.store(kNothing);
  }
  Hold(const Hold&) = delete;
  Hold& operator=(const Hold&) = delete;

  // Raise the flag now, if the watchdog has not; returns once it is up.
  void raise() {
    std::unique_lock<std::mutex> lock(mutex_);
    now_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return raised_; });
  }
  bool raised() {
    std::lock_guard<std::mutex> lock(mutex_);
    return raised_;
  }
  // Whether the hold reached the stream: armed, then taken by a copy or a
  // decode, and queued.
  static bool taken() {
    return held().armed.load() == kNothing &&
           held().failed.load() == CUDA_SUCCESS;
  }

 private:
  const Flag& flag_;
  std::uint32_t number_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool now_ = false;
  bool raised_ = false;
  CUresult raised_result_ = CUDA_SUCCESS;
  std::thread watchdog_;
};

// --- What the readers read ---------------------------------------------------

using Gpu = vkc::test::GpuContext;
using AccessUnits = std::vector<std::vector<std::uint8_t>>;

constexpr std::uint32_t kWidth = 256;
constexpr std::uint32_t kHeight = 144;
constexpr std::size_t kPixels = std::size_t{kWidth} * kHeight;
constexpr const char* kPatches = VS_HEVC_DATA "/patches_256x144.h265";
constexpr const char* kJpegFile = VS_JPEG_DATA "/patches_256x144.jpg";
constexpr const char* kRefused = VS_HEVC_DATA "/refused_256x144.h265";

// A frame of @p color's picture, through a lens, beside flat host depth.
struct Frames {
  sensor::camera::CameraModel camera;
  std::vector<std::uint16_t> depth = std::vector<std::uint16_t>(kPixels, 1500);

  Frames() {
    camera.size = {kWidth, kHeight};
    camera.intrinsics = {200.0f, 198.0f, 127.3f, 71.6f};
    camera.distortion = {-0.28f, 0.09f, 0.0012f, -0.0009f,
                         -0.01f, 0.05f, 0.0f,    0.0f};
  }
  sensor::RgbdFrame of(const sensor::YuvImage& color) const {
    sensor::RgbdFrame f;
    f.depth = depth.data();
    f.depth_camera = camera;
    f.min_depth = 0.1f;
    f.max_depth = 10.0f;
    f.color = color;
    f.color_camera = camera;
    return f;
  }
};

// What a picture reads as, unheld: prepare's words, and its planes read back.
struct Reads {
  std::vector<std::uint32_t> words;
  std::vector<std::uint8_t> planes[3];
};

std::vector<std::uint32_t> color_words(const Gpu& gpu,
                                       const sensor::DeviceFrame& prepared) {
  auto read = vkc::test::read_back<std::uint32_t>(gpu.device, gpu.allocator,
                                                  *prepared.color, kPixels);
  return read ? std::move(read).value() : std::vector<std::uint32_t>();
}

std::vector<std::uint32_t> prepared_words(const Gpu& gpu,
                                          sensor::GpuFramePrep& prep,
                                          const sensor::RgbdFrame& frame) {
  auto prepared = prep.prepare(frame);
  if (!prepared) {
    std::fprintf(stderr, "%s\n", prepared.status().message().c_str());
    return {};
  }
  return color_words(gpu, *prepared);
}

// Read after a host wait for the picture's ready value, so they hold whether
// or not the readers wait for it themselves.
Reads reads_of(const Gpu& gpu, sensor::GpuFramePrep& prep, const Frames& frames,
               const sensor::YuvImage& yuv) {
  Reads out;
  if (yuv.ready == nullptr || !yuv.ready->wait(yuv.ready_value).ok()) {
    return out;
  }
  out.words = prepared_words(gpu, prep, frames.of(yuv));
  sensor::YuvImage copy = yuv;
  vs_test::read_device_picture(copy, gpu.device, gpu.allocator, out.planes);
  return out;
}

// @p picture's luma read back, rows packed; empty if the read fails.
std::vector<std::uint8_t> luma_of(const Gpu& gpu, sensor::YuvImage& picture) {
  std::vector<std::uint8_t> out;
  auto readback = sensor::PictureReadback::create(gpu.device, gpu.allocator);
  if (!readback) return out;
  auto read = (*readback)->read(picture, sensor::PictureReadback::Planes::Luma);
  if (!read) {
    std::fprintf(stderr, "%s\n", read.status().message().c_str());
    return out;
  }
  for (std::uint32_t r = 0; r < read->height; ++r) {
    const std::uint8_t* line = read->luma + r * read->stride;
    out.insert(out.end(), line, line + read->width);
  }
  return out;
}

// The value @p yuv carries, not yet reached.
bool unset(const sensor::YuvImage& yuv) {
  if (yuv.ready == nullptr) return false;
  const vkc::Result<std::uint64_t> counter = yuv.ready->value();
  return counter.ok() && *counter < yuv.ready_value;
}

// A batch that records prep for @p frame and submits with its ready_points is
// still waiting on the device after 50 ms; once @p hold is raised it reads
// what @p want says.
int check_record(const Gpu& gpu, sensor::GpuFramePrep& prep,
                 const sensor::RgbdFrame& frame, Hold& hold,
                 const std::vector<std::uint32_t>& want) {
  vkc::CommandBatch batch(gpu.device, gpu.allocator);
  auto prepared = prep.record(batch, frame);
  CHECK(prepared.ok());
  const std::vector<vkc::TimelinePoint> waits = sensor::ready_points(frame);
  CHECK(waits.size() == 1 && waits[0].semaphore == frame.color.ready.get() &&
        waits[0].value == frame.color.ready_value);
  auto pending = batch.submit_async(waits, {});
  CHECK(pending.ok());
  std::this_thread::sleep_for(50ms);
  CHECK(!pending->ready());  // it waits for CUDA's copy on the device
  CHECK(!hold.raised());
  hold.raise();
  CHECK(pending->wait().ok());
  CHECK(color_words(gpu, *prepared) == want);
  return 0;
}

// --- NVDEC -------------------------------------------------------------------

// The patch clip, and each of its pictures as it reads unheld.
struct Clip {
  AccessUnits units;
  std::vector<Reads> want;
};

sensor::HevcDecoder::Options options_of(const Gpu& gpu) {
  sensor::HevcDecoder::Options options;
  options.device = &gpu.device;
  options.allocator = &gpu.allocator;
  return options;
}

double ms_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

int read_clip(const Gpu& gpu, sensor::GpuFramePrep& prep, const Frames& frames,
              Clip& clip) {
  clip.units = sensor::testing::access_units_of_file(kPatches);
  CHECK(!clip.units.empty());
  auto decoder = sensor::HevcDecoder::create(options_of(gpu));
  CHECK(decoder.ok());
  const AccessUnits& units = clip.units;
  for (std::size_t i = 0; i <= units.size(); ++i) {
    const bool end = i == units.size();
    CHECK(decoder
              ->send(end ? nullptr : units[i].data(), end ? 0 : units[i].size(),
                     static_cast<std::int64_t>(i))
              .ok());
    for (;;) {
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!picture.value()) break;
      clip.want.push_back(reads_of(gpu, prep, frames, picture.value()->yuv));
      CHECK(clip.want.back().words.size() == kPixels &&
            clip.want.back().planes[0].size() == kPixels);
    }
  }
  CHECK(clip.want.size() == units.size());
  // Each held picture reads otherwise than the one before it, whose images a
  // reader that did not wait would find.
  for (std::size_t k = 1; k < 4; ++k) {
    CHECK(clip.want[k].words != clip.want[k - 1].words &&
          clip.want[k].planes[0] != clip.want[k - 1].planes[0]);
  }
  return 0;
}

// The patch clip's pictures held. Picture 0 held for a recorded batch, 1 for
// prepare, 2 for PictureReadback and 3 for LumaPass, each started before the
// flag is raised; the rest unheld.
int test_hevc(const Gpu& gpu, const Flag& flag, sensor::GpuFramePrep& prep,
              const Frames& frames, const Clip& clip) {
  auto luma = sensor::LumaPass::create(gpu.device, gpu.allocator);
  CHECK(luma.ok());
  const AccessUnits& units = clip.units;
  const std::vector<Reads>& want = clip.want;
  auto decoder = sensor::HevcDecoder::create(options_of(gpu));
  CHECK(decoder.ok());
  std::size_t next = 0;  // the next picture to come out
  for (std::size_t i = 0; i <= units.size(); ++i) {
    const bool end = i == units.size();
    CHECK(decoder
              ->send(end ? nullptr : units[i].data(), end ? 0 : units[i].size(),
                     static_cast<std::int64_t>(i))
              .ok());
    for (;;) {
      const bool holds = next < 4;
      std::optional<Hold> hold;
      if (holds) hold.emplace(flag, kCopy, next == 0 ? 10000ms : 1000ms);
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!picture.value()) {
        if (hold) CHECK(held().armed.load() == kCopy);  // no copy yet
        break;
      }
      sensor::YuvImage& yuv = picture.value()->yuv;
      const Reads& w = want[next];
      if (!holds) {
        CHECK(prepared_words(gpu, prep, frames.of(yuv)) == w.words);
        ++next;
        continue;
      }
      // Out with its copy queued behind the flag, before it was raised.
      CHECK(Hold::taken());
      CHECK(!hold->raised());
      CHECK(unset(yuv));
      if (next == 0) {
        CHECK(check_record(gpu, prep, frames.of(yuv), *hold, w.words) == 0);
      } else if (next == 1) {
        CHECK(prepared_words(gpu, prep, frames.of(yuv)) == w.words);
        CHECK(hold->raised());  // prepare waited for it, on the device
      } else if (next == 2) {
        std::vector<std::uint8_t> planes[3];
        sensor::YuvImage copy = yuv;
        vs_test::read_device_picture(copy, gpu.device, gpu.allocator, planes);
        CHECK(hold->raised());
        for (int p = 0; p < 3; ++p) CHECK(planes[p] == w.planes[p]);
      } else {
        sensor::YuvImage copy = yuv;
        auto made = luma.value()->run({&copy});
        CHECK(made.ok() && made->size() == 1);
        CHECK(hold->raised());
        CHECK(luma_of(gpu, (*made)[0].full) == w.planes[0]);
      }
      const vkc::Result<std::uint64_t> counter = yuv.ready->value();
      CHECK(counter.ok() && *counter >= yuv.ready_value);
      ++next;
    }
  }
  CHECK(next == units.size());
  return 0;
}

// One hold before picture 0's copy holds every later copy on the stream:
// pictures 0 to 2 come out at once, and with three kept the decoder waits on
// the host for picture 0's value before it hands out another, until the
// watchdog raises the flag. All four then read as unheld.
int test_back_pressure(const Gpu& gpu, const Flag& flag,
                       sensor::GpuFramePrep& prep, const Frames& frames,
                       const Clip& clip) {
  constexpr auto kAfter = 2000ms;
  auto decoder = sensor::HevcDecoder::create(options_of(gpu));
  CHECK(decoder.ok());
  std::vector<sensor::DecodedPicture> got;
  std::vector<double> out_ms;  // when each came out, from the hold
  {
    Hold hold(flag, kCopy, kAfter);
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < clip.units.size() && got.size() < 4; ++i) {
      CHECK(decoder
                ->send(clip.units[i].data(), clip.units[i].size(),
                       static_cast<std::int64_t>(i))
                .ok());
      while (got.size() < 4) {
        auto picture = decoder->receive();
        CHECK(picture.ok());
        if (!picture.value()) break;
        out_ms.push_back(ms_since(start));
        got.push_back(std::move(*picture.value()));
      }
    }
    CHECK(got.size() == 4 && Hold::taken());
  }
  for (std::size_t k = 0; k < 3; ++k) CHECK(out_ms[k] < 1000.0);
  CHECK(out_ms[3] >= 1500.0);  // after the watchdog's raise
  for (std::size_t k = 0; k < 4; ++k) {
    CHECK(prepared_words(gpu, prep, frames.of(got[k].yuv)) ==
          clip.want[k].words);
  }
  return 0;
}

// The decoder destroyed while its stream is held: the destructor waits for
// the stream (until the watchdog raises the flag), so the picture it handed
// out reaches its value and reads as unheld, and no batch is left waiting.
int test_destroy_held(const Gpu& gpu, const Flag& flag,
                      sensor::GpuFramePrep& prep, const Frames& frames,
                      const Clip& clip) {
  std::optional<sensor::DecodedPicture> kept;
  {
    Hold hold(flag, kCopy, 1000ms);  // outlives the decoder
    auto decoder = sensor::HevcDecoder::create(options_of(gpu));
    CHECK(decoder.ok());
    for (std::size_t i = 0; i < clip.units.size() && !kept; ++i) {
      CHECK(decoder
                ->send(clip.units[i].data(), clip.units[i].size(),
                       static_cast<std::int64_t>(i))
                .ok());
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (picture.value()) kept = std::move(*picture.value());
    }
    CHECK(kept && Hold::taken() && unset(kept->yuv));
    const auto start = std::chrono::steady_clock::now();
    {
      sensor::HevcDecoder gone = std::move(decoder).value();
    }
    CHECK(ms_since(start) > 500.0);  // it waited for the raise
    CHECK(!unset(kept->yuv));
  }
  CHECK(prepared_words(gpu, prep, frames.of(kept->yuv)) == clip.want[0].words);
  return 0;
}

// --- A failed stream ---------------------------------------------------------

// Runs `act` after `after` unless stopped first: frees a decoder that waits
// for good, so the test fails rather than hangs.
class Watchdog {
 public:
  template <typename Act>
  Watchdog(std::chrono::milliseconds after, Act act)
      : thread_([this, after, act] {
          std::unique_lock<std::mutex> lock(mutex_);
          if (!cv_.wait_for(lock, after, [this] { return stopped_; })) {
            act();
            fired_ = true;
          }
        }) {}
  ~Watchdog() { stop(); }
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;

  // Whether it had run `act`.
  bool stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    return fired_;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_ = false;
  bool fired_ = false;
  std::thread thread_;  // last: it runs on the members above
};

// @p unit, a P picture's access unit, its slice naming picture parameter set
// 1, which the clip never sends.
std::vector<std::uint8_t> naming_no_pps(std::vector<std::uint8_t> unit) {
  for (std::size_t i = 0; i + 5 < unit.size(); ++i) {
    const bool slice = unit[i] == 0 && unit[i + 1] == 0 && unit[i + 2] == 1 &&
                       ((unit[i + 3] >> 1) & 0x3f) < 32;
    // After the two header bytes: first_slice_segment_in_pic_flag (1), then
    // slice_pic_parameter_set_id as ue(v), '1' for 0 and '010' for 1.
    if (slice && (unit[i + 5] & 0xc0) == 0xc0) {
      unit[i + 5] = static_cast<std::uint8_t>(0xa0 | (unit[i + 5] & 0x0f));
      return unit;
    }
  }
  return {};
}

// Whether this FFmpeg reports @p unit, sent after the patch clip's first
// picture is taken, as an error: FFmpeg 6.1 reports a slice naming a missing
// picture parameter set; FFmpeg 8 skips it.
bool reported(const Gpu& gpu, const Clip& clip,
              const std::vector<std::uint8_t>& unit) {
  auto decoder = sensor::HevcDecoder::create(options_of(gpu));
  if (!decoder ||
      !decoder->send(clip.units[0].data(), clip.units[0].size(), 0).ok()) {
    return false;
  }
  for (;;) {
    auto picture = decoder->receive();
    if (!picture.ok() || !picture.value()) break;
  }
  return !decoder->send(unit.data(), unit.size(), 1).ok();
}

// Picture 0's ready value dropped, as a stream that fails after queuing its
// signal drops it, then an access unit that fails: one naming a picture
// parameter set never sent (IoError, where FFmpeg reports it), or the refused
// clip's 4:0:0 grey (Unsupported). Sent after picture 0 is taken, send
// reports it; sent before, FFmpeg holds it until picture 0 is taken and
// receive reports it. Either way the value is set by the time the error
// returns, and the picture reads as unheld (its copy ran).
int test_error_settles(const Gpu& gpu, sensor::GpuFramePrep& prep,
                       const Frames& frames, const Clip& clip) {
  const AccessUnits refused = sensor::testing::access_units_of_file(kRefused);
  CHECK(refused.size() == 10);
  const struct {
    std::vector<std::uint8_t> unit;
    vkc::Status::Code code;
  } bad[] = {{naming_no_pps(clip.units[1]), vkc::Status::Code::IoError},
             {refused[8], vkc::Status::Code::Unsupported}};
  for (const auto& b : bad) {
    CHECK(!b.unit.empty());
    if (b.code == vkc::Status::Code::IoError && !reported(gpu, clip, b.unit)) {
      std::fprintf(stderr,
                   "this FFmpeg skips a slice naming a missing picture "
                   "parameter set; the decode-error cases skipped\n");
      continue;
    }
    for (const bool before : {false, true}) {
      auto decoder = sensor::HevcDecoder::create(options_of(gpu));
      CHECK(decoder.ok());
      held().drops.store(1);
      CHECK(decoder->send(clip.units[0].data(), clip.units[0].size(), 0).ok());
      if (before) CHECK(decoder->send(b.unit.data(), b.unit.size(), 1).ok());
      auto picture = decoder->receive();
      CHECK(picture.ok() && picture.value() && held().drops.load() == 0);
      const sensor::YuvImage yuv = picture.value()->yuv;
      CHECK(unset(yuv));
      vkc::Status failed;
      if (before) {
        failed = decoder->receive().status();
      } else {
        auto none = decoder->receive();
        CHECK(none.ok() && !none.value());
        failed = decoder->send(b.unit.data(), b.unit.size(), 1);
      }
      CHECK(failed.domain() == b.code);
      CHECK(!unset(yuv));
      CHECK(prepared_words(gpu, prep, frames.of(yuv)) == clip.want[0].words);
    }
  }
  return 0;
}

// Pictures 0 to 2's ready values dropped, and the stream then reporting a
// fault: with three kept, the next receive's wait for picture 0 sees the
// fault, sets every value handed out and returns CUDA's error. A watchdog
// sets them after 5 s, so a decoder that waits for good fails rather than
// hangs.
int test_wait_sees_fault(const Gpu& gpu, sensor::GpuFramePrep& prep,
                         const Frames& frames, const Clip& clip) {
  auto decoder = sensor::HevcDecoder::create(options_of(gpu));
  CHECK(decoder.ok());
  held().drops.store(3);
  std::vector<sensor::DecodedPicture> got;
  std::size_t sent = 0;
  while (got.size() < 3 && sent < clip.units.size()) {
    CHECK(decoder
              ->send(clip.units[sent].data(), clip.units[sent].size(),
                     static_cast<std::int64_t>(sent))
              .ok());
    ++sent;
    while (got.size() < 3) {  // a fourth receive would wait for picture 0
      auto picture = decoder->receive();
      CHECK(picture.ok());
      if (!picture.value()) break;
      got.push_back(std::move(*picture.value()));
    }
  }
  CHECK(got.size() == 3 && held().drops.load() == 0);
  for (const sensor::DecodedPicture& p : got) CHECK(unset(p.yuv));
  CHECK(decoder
            ->send(clip.units[sent].data(), clip.units[sent].size(),
                   static_cast<std::int64_t>(sent))
            .ok());
  held().faulted.store(true);
  const sensor::YuvImage last = got.back().yuv;
  Watchdog rescue(5000ms, [&last] {
    // The semaphore is not const: the decoder hands it out as such.
    auto& timeline = const_cast<vkc::TimelineSemaphore&>(*last.ready);
    static_cast<void>(timeline.signal(last.ready_value));
  });
  const vkc::Status failed = decoder->receive().status();
  const bool rescued = rescue.stop();
  held().faulted.store(false);
  CHECK(!rescued);
  CHECK(failed.domain() == vkc::Status::Code::Backend);
  for (std::size_t k = 0; k < got.size(); ++k) {
    CHECK(!unset(got[k].yuv));
    CHECK(prepared_words(gpu, prep, frames.of(got[k].yuv)) ==
          clip.want[k].words);
  }
  return 0;
}

// --- nvJPEG ------------------------------------------------------------------

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A JPEG decoded with the decoder's stream held: decode returns before the
// flag is raised, and a recorded batch, then PictureReadback (its bytes
// overwritten after decode), read what it reads unheld.
int test_jpeg(const Gpu& gpu, const Flag& flag) {
  sensor::JpegDecoder::Options options;
  options.device = &gpu.device;
  options.allocator = &gpu.allocator;
  auto decoder = sensor::JpegDecoder::create(options);
  if (!decoder) {
    if (decoder.status().message().find("no hardware JPEG engine") !=
        std::string::npos) {
      std::fprintf(stderr, "%s; nvJPEG's part skipped\n",
                   decoder.status().message().c_str());
      return 0;
    }
    std::fprintf(stderr, "%s\n", decoder.status().message().c_str());
    return 1;
  }
  held().jpeg_stream = held().last_stream.load();  // the decoder's own
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  CHECK(prep.ok());
  const Frames frames;
  const std::vector<std::uint8_t> bytes = read_file(kJpegFile);
  CHECK(!bytes.empty());

  Reads want;
  {
    auto picture = decoder->decode(bytes.data(), bytes.size());
    CHECK(picture.ok());
    want = reads_of(gpu, prep.value(), frames, picture->yuv);
    CHECK(want.words.size() == kPixels && want.planes[0].size() == kPixels);
  }
  {
    Hold hold(flag, kJpeg, 10000ms);
    auto picture = decoder->decode(bytes.data(), bytes.size());
    CHECK(picture.ok());
    CHECK(Hold::taken());
    CHECK(!hold.raised());
    CHECK(unset(picture->yuv));
    CHECK(check_record(gpu, prep.value(), frames.of(picture->yuv), hold,
                       want.words) == 0);
  }
  {
    // The bytes are read during decode only: overwritten once it returns,
    // with the stream still held, they decode the same.
    std::vector<std::uint8_t> scratch = bytes;
    Hold hold(flag, kJpeg, 1000ms);
    auto picture = decoder->decode(scratch.data(), scratch.size());
    std::fill(scratch.begin(), scratch.end(), std::uint8_t{0});
    CHECK(picture.ok());
    CHECK(Hold::taken());
    CHECK(!hold.raised());
    CHECK(unset(picture->yuv));
    std::vector<std::uint8_t> planes[3];
    vs_test::read_device_picture(picture->yuv, gpu.device, gpu.allocator,
                                 planes);
    CHECK(hold.raised());
    for (int p = 0; p < 3; ++p) CHECK(planes[p] == want.planes[p]);
  }
  return 0;
}

int gpu_main(Gpu& gpu) {
  sensor::HevcDecoder::Options options;
  options.device = &gpu.device;
  options.allocator = &gpu.allocator;
  if (auto probe = sensor::HevcDecoder::create(options); !probe) {
    if (probe.status().domain() != vkc::Status::Code::Unsupported) {
      std::fprintf(stderr, "FAIL: %s\n", probe.status().message().c_str());
      return 1;
    }
    return vs_test::no_decoder(probe.status().message());
  }
  auto flag = make_flag(gpu.device);
  if (!flag) {
    std::fprintf(stderr, "FAIL: %s\n", flag.status().message().c_str());
    return 1;
  }
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  if (!prep) {
    std::fprintf(stderr, "FAIL: %s\n", prep.status().message().c_str());
    return 1;
  }
  const Frames frames;
  Clip clip;
  if (read_clip(gpu, prep.value(), frames, clip) != 0) return 1;
  if (test_hevc(gpu, **flag, prep.value(), frames, clip) != 0) return 1;
  if (test_back_pressure(gpu, **flag, prep.value(), frames, clip) != 0) {
    return 1;
  }
  if (test_destroy_held(gpu, **flag, prep.value(), frames, clip) != 0) {
    return 1;
  }
  if (test_error_settles(gpu, prep.value(), frames, clip) != 0) return 1;
  if (test_wait_sees_fault(gpu, prep.value(), frames, clip) != 0) return 1;
  if (test_jpeg(gpu, **flag) != 0) return 1;
  std::puts("sensor_video_handover: OK");
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
