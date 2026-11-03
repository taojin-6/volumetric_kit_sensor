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
// what the same picture reads unheld, to the bit. A watchdog raises a flag the
// test has not after 10 s, so a decoder that waits on the host fails rather
// than hangs.
//
// Where no device path opens the test skips, or fails where
// VS_TEST_HEVC_BACKEND promises one, as the decoder tests do; the JPEG part
// skips on a GPU without the hardware JPEG engine.

#include <cuda.h>
#include <dlfcn.h>

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

CUresult held_stream_create(CUstream* stream, unsigned int flags) {
  const CUresult r = held().real->cuStreamCreate(stream, flags);
  if (r == CUDA_SUCCESS) held().last_stream.store(*stream);
  return r;
}

}  // namespace

namespace volumetric_kit::sensor::video {

// The decoders' table: libcuda's, with the copy, the context push and stream
// creation wrapped as above.
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

// The patch clip's pictures, unheld and held. Picture 0 held for a recorded
// batch, 1 for prepare, 2 for PictureReadback and 3 for LumaPass, each started
// before the flag is raised; the rest unheld.
int test_hevc(const Gpu& gpu, const Flag& flag) {
  auto prep = sensor::GpuFramePrep::create(gpu.device, gpu.allocator);
  CHECK(prep.ok());
  auto luma = sensor::LumaPass::create(gpu.device, gpu.allocator);
  CHECK(luma.ok());
  const Frames frames;
  const AccessUnits units = sensor::testing::access_units_of_file(kPatches);
  CHECK(!units.empty());
  sensor::HevcDecoder::Options options;
  options.device = &gpu.device;
  options.allocator = &gpu.allocator;

  // Unheld.
  std::vector<Reads> want;
  {
    auto decoder = sensor::HevcDecoder::create(options);
    CHECK(decoder.ok());
    for (std::size_t i = 0; i <= units.size(); ++i) {
      const bool end = i == units.size();
      CHECK(decoder
                ->send(end ? nullptr : units[i].data(),
                       end ? 0 : units[i].size(), static_cast<std::int64_t>(i))
                .ok());
      for (;;) {
        auto picture = decoder->receive();
        CHECK(picture.ok());
        if (!picture.value()) break;
        want.push_back(
            reads_of(gpu, prep.value(), frames, picture.value()->yuv));
        CHECK(want.back().words.size() == kPixels &&
              want.back().planes[0].size() == kPixels);
      }
    }
  }
  CHECK(want.size() == units.size());
  // Each held picture reads otherwise than the one before it, whose images a
  // reader that did not wait would find.
  for (std::size_t k = 1; k < 4; ++k) {
    CHECK(want[k].words != want[k - 1].words &&
          want[k].planes[0] != want[k - 1].planes[0]);
  }

  // Held.
  auto decoder = sensor::HevcDecoder::create(options);
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
        CHECK(prepared_words(gpu, prep.value(), frames.of(yuv)) == w.words);
        ++next;
        continue;
      }
      // Out with its copy queued behind the flag, before it was raised.
      CHECK(Hold::taken());
      CHECK(!hold->raised());
      CHECK(unset(yuv));
      if (next == 0) {
        CHECK(check_record(gpu, prep.value(), frames.of(yuv), *hold, w.words) ==
              0);
      } else if (next == 1) {
        CHECK(prepared_words(gpu, prep.value(), frames.of(yuv)) == w.words);
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

// --- nvJPEG ------------------------------------------------------------------

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A JPEG decoded with the decoder's stream held: decode returns before the
// flag is raised, and a recorded batch, then PictureReadback, read what it
// reads unheld.
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
    Hold hold(flag, kJpeg, 1000ms);
    auto picture = decoder->decode(bytes.data(), bytes.size());
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
  if (test_hevc(gpu, **flag) != 0) return 1;
  if (test_jpeg(gpu, **flag) != 0) return 1;
  std::puts("sensor_video_handover: OK");
  return 0;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
