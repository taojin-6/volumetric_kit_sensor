// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"

#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "ffmpeg.hpp"
#include "hw_backend.hpp"
#include "picture_converter.hpp"
#if VR_SENSOR_VIDEO_WITH_CUDA
#include <dlfcn.h>
#include <nvjpeg.h>

#include <vector>

#include "cuda_pictures.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#endif
#if defined(__APPLE__)
#include "vt_jpeg.hpp"
#endif

namespace volumetric_kit::recon::sensor {
namespace {

constexpr const char* kWho = "JpegDecoder";

// JFIF's matrix and range, which a JPEG that names none is coded in.
constexpr VideoColorDescription kJfif{VideoColorMatrix::Bt601, true};

#if VR_SENSOR_VIDEO_WITH_CUDA

// The nvJPEG entry points used here, loaded at run time as libcuda is.
#define VR_NVJPEG_FUNCTIONS(X)            \
  X(nvjpegCreateEx)                       \
  X(nvjpegDestroy)                        \
  X(nvjpegDecoderCreate)                  \
  X(nvjpegDecoderDestroy)                 \
  X(nvjpegDecoderStateCreate)             \
  X(nvjpegJpegStateDestroy)               \
  X(nvjpegBufferPinnedCreate)             \
  X(nvjpegBufferPinnedDestroy)            \
  X(nvjpegBufferDeviceCreate)             \
  X(nvjpegBufferDeviceDestroy)            \
  X(nvjpegStateAttachPinnedBuffer)        \
  X(nvjpegStateAttachDeviceBuffer)        \
  X(nvjpegJpegStreamCreate)               \
  X(nvjpegJpegStreamDestroy)              \
  X(nvjpegJpegStreamParse)                \
  X(nvjpegJpegStreamGetFrameDimensions)   \
  X(nvjpegJpegStreamGetComponentsNum)     \
  X(nvjpegJpegStreamGetSamplePrecision)   \
  X(nvjpegJpegStreamGetChromaSubsampling) \
  X(nvjpegDecodeParamsCreate)             \
  X(nvjpegDecodeParamsDestroy)            \
  X(nvjpegDecodeParamsSetOutputFormat)    \
  X(nvjpegDecoderJpegSupported)           \
  X(nvjpegDecodeJpegHost)                 \
  X(nvjpegDecodeJpegTransferToDevice)     \
  X(nvjpegDecodeJpegDevice)

struct Nvjpeg {
#define VR_NVJPEG_DECLARE(name) decltype(&::name) name = nullptr;
  VR_NVJPEG_FUNCTIONS(VR_NVJPEG_DECLARE)
#undef VR_NVJPEG_DECLARE
};

// libnvjpeg of the major version compiled against, loaded once and kept;
// null where it does not load or lacks an entry point.
const Nvjpeg* nvjpeg() {
  static const std::optional<Nvjpeg> loaded = []() -> std::optional<Nvjpeg> {
    void* lib = dlopen("libnvjpeg.so." VR_CUDA_STRING(NVJPEG_VER_MAJOR),
                       RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return std::nullopt;
    Nvjpeg n;
    bool ok = true;
#define VR_NVJPEG_LOAD(name)                                      \
  n.name = reinterpret_cast<decltype(n.name)>(dlsym(lib, #name)); \
  ok = ok && n.name != nullptr;
    VR_NVJPEG_FUNCTIONS(VR_NVJPEG_LOAD)
#undef VR_NVJPEG_LOAD
    if (!ok) {
      dlclose(lib);
      return std::nullopt;
    }
    return n;
  }();
  return loaded ? &*loaded : nullptr;
}

bool ok(nvjpegStatus_t s) { return s == NVJPEG_STATUS_SUCCESS; }

// nvJPEG's verdict on the JPEG rather than on the GPU: software may still
// decode it, and says what is wrong if not.
bool refused(nvjpegStatus_t s) {
  return s == NVJPEG_STATUS_BAD_JPEG || s == NVJPEG_STATUS_JPEG_NOT_SUPPORTED ||
         s == NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
}

// nvJPEG in the primary context of the Vulkan device's GPU, and everything it
// holds there, released together.
struct NvjpegDecoder {
  // Null where nvJPEG cannot use @p device. Its picture buffers are made
  // through @p allocator.
  static std::unique_ptr<NvjpegDecoder> open(const core::Device& device,
                                             core::Allocator& allocator);
  NvjpegDecoder() = default;
  ~NvjpegDecoder();
  NvjpegDecoder(const NvjpegDecoder&) = delete;
  NvjpegDecoder& operator=(const NvjpegDecoder&) = delete;

  // The JPEG on the device; empty for one nvJPEG refuses, which goes to
  // software instead. An error means the device path failed.
  core::Result<std::optional<DecodedPicture>> decode(const std::uint8_t* data,
                                                     std::size_t size);
  bool add_engine(nvjpegBackend_t nv);

  JpegDecodeBackend backend = JpegDecodeBackend::Software;
  CUdevice cuda_device = 0;
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  // A picture waits on an event made to block, not spin: CUDA's default wait
  // bills the whole wait as CPU.
  CUevent done = nullptr;
  nvjpegHandle_t handle = nullptr;
  nvjpegBufferPinned_t pinned = nullptr;
  nvjpegBufferDevice_t staging = nullptr;
  nvjpegJpegStream_t parsed = nullptr;
  nvjpegDecodeParams_t params = nullptr;
  // A decoder and its state per engine, the hardware's first: the cores take
  // what it refuses, a JPEG past 16384 pixels a side. Their states share the
  // staging buffers, since one JPEG decodes at a time.
  struct Engine {
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t state = nullptr;
  };
  std::vector<Engine> engines;
  std::unique_ptr<video::CudaPictures> pictures;
};

std::unique_ptr<NvjpegDecoder> NvjpegDecoder::open(const core::Device& device,
                                                   core::Allocator& allocator) {
  const video::CudaDriver* cu = video::cuda_driver();
  const Nvjpeg* n = nvjpeg();
  if (!device.exports_memory() || cu == nullptr || n == nullptr) return nullptr;
  const auto ordinal = video::cuda_ordinal_of(device, kWho);
  if (!ordinal) return nullptr;
  auto d = std::make_unique<NvjpegDecoder>();
  if (cu->cuDeviceGet(&d->cuda_device, ordinal.value()) != CUDA_SUCCESS ||
      cu->cuDevicePrimaryCtxRetain(&d->context, d->cuda_device) !=
          CUDA_SUCCESS) {
    d->context = nullptr;
    return nullptr;
  }
  const video::CudaContextScope scope(d->context);
  if (!scope.ok() ||
      cu->cuStreamCreate(&d->stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
      cu->cuEventCreate(&d->done,
                        CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING) !=
          CUDA_SUCCESS) {
    return nullptr;
  }
  // A handle on the GPU's hardware JPEG engine where it has one, else on its
  // cores.
  const bool hardware = ok(n->nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr,
                                             nullptr, 0, &d->handle));
  if (!hardware && !ok(n->nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr,
                                         nullptr, 0, &d->handle))) {
    d->handle = nullptr;
    return nullptr;
  }
  if (!ok(n->nvjpegBufferPinnedCreate(d->handle, nullptr, &d->pinned)) ||
      !ok(n->nvjpegBufferDeviceCreate(d->handle, nullptr, &d->staging)) ||
      !ok(n->nvjpegJpegStreamCreate(d->handle, &d->parsed)) ||
      !ok(n->nvjpegDecodeParamsCreate(d->handle, &d->params)) ||
      !ok(n->nvjpegDecodeParamsSetOutputFormat(d->params, NVJPEG_OUTPUT_YUV))) {
    return nullptr;
  }
  const bool engine = hardware && d->add_engine(NVJPEG_BACKEND_HARDWARE);
  if (!d->add_engine(NVJPEG_BACKEND_GPU_HYBRID) && !engine) return nullptr;
  auto pictures = video::CudaPictures::create(device, allocator, d->context,
                                              d->stream, kWho);
  if (!pictures) return nullptr;
  d->pictures = std::move(pictures).value();
  d->backend =
      engine ? JpegDecodeBackend::NvjpegHardware : JpegDecodeBackend::NvjpegGpu;
  return d;
}

bool NvjpegDecoder::add_engine(nvjpegBackend_t nv) {
  const Nvjpeg* n = nvjpeg();
  Engine e;
  if (!ok(n->nvjpegDecoderCreate(handle, nv, &e.decoder))) return false;
  if (!ok(n->nvjpegDecoderStateCreate(handle, e.decoder, &e.state)) ||
      !ok(n->nvjpegStateAttachPinnedBuffer(e.state, pinned)) ||
      !ok(n->nvjpegStateAttachDeviceBuffer(e.state, staging))) {
    if (e.state != nullptr) n->nvjpegJpegStateDestroy(e.state);
    n->nvjpegDecoderDestroy(e.decoder);
    return false;
  }
  engines.push_back(e);
  return true;
}

NvjpegDecoder::~NvjpegDecoder() {
  if (context == nullptr) return;
  const video::CudaDriver* cu = video::cuda_driver();
  const Nvjpeg* n = nvjpeg();
  {
    const video::CudaContextScope scope(context);
    pictures.reset();  // its imports go with the context
    for (const Engine& e : engines) {
      n->nvjpegJpegStateDestroy(e.state);
      n->nvjpegDecoderDestroy(e.decoder);
    }
    if (params != nullptr) n->nvjpegDecodeParamsDestroy(params);
    if (parsed != nullptr) n->nvjpegJpegStreamDestroy(parsed);
    if (staging != nullptr) n->nvjpegBufferDeviceDestroy(staging);
    if (pinned != nullptr) n->nvjpegBufferPinnedDestroy(pinned);
    if (handle != nullptr) n->nvjpegDestroy(handle);
    if (done != nullptr) cu->cuEventDestroy(done);
    if (stream != nullptr) cu->cuStreamDestroy(stream);
  }
  cu->cuDevicePrimaryCtxRelease(cuda_device);
}

core::Result<std::optional<DecodedPicture>> NvjpegDecoder::decode(
    const std::uint8_t* data, std::size_t size) {
  const video::CudaContextScope scope(context);
  if (!scope.ok()) {
    return core::Status::io_error(std::string(kWho) +
                                  ": making the CUDA context current");
  }
  // nvJPEG's last status: one that refuses the JPEG sends it to software,
  // any other fails the device path.
  nvjpegStatus_t status = NVJPEG_STATUS_SUCCESS;
  const auto step = [&status](nvjpegStatus_t s) { return ok(status = s); };
  const auto failed =
      [&status](
          const char* what) -> core::Result<std::optional<DecodedPicture>> {
    if (refused(status)) return std::optional<DecodedPicture>();
    return core::Status::io_error(std::string(kWho) + ": " + what +
                                  ": nvJPEG status " +
                                  std::to_string(static_cast<int>(status)));
  };

  // Only an 8-bit 4:2:0 JPEG, on the first engine that takes it.
  const Nvjpeg* n = nvjpeg();
  unsigned width = 0;
  unsigned height = 0;
  unsigned components = 0;
  unsigned precision = 0;
  nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
  if (!step(n->nvjpegJpegStreamParse(handle, data, size, 0, 0, parsed)) ||
      !step(n->nvjpegJpegStreamGetFrameDimensions(parsed, &width, &height)) ||
      !step(n->nvjpegJpegStreamGetComponentsNum(parsed, &components)) ||
      !step(n->nvjpegJpegStreamGetSamplePrecision(parsed, &precision)) ||
      !step(n->nvjpegJpegStreamGetChromaSubsampling(parsed, &subsampling))) {
    return failed("parsing a JPEG");
  }
  if (components != 3 || precision != 8 || subsampling != NVJPEG_CSS_420 ||
      width == 0 || height == 0) {
    return std::optional<DecodedPicture>();
  }
  const Engine* engine = nullptr;
  for (const Engine& e : engines) {
    int unsupported = -1;
    if (!step(n->nvjpegDecoderJpegSupported(e.decoder, parsed, params,
                                            &unsupported))) {
      return failed("parsing a JPEG");
    }
    if (unsupported == 0) {
      engine = &e;
      break;
    }
  }
  if (engine == nullptr) return std::optional<DecodedPicture>();

  // Y, then Cb and Cr at half size rounded up, each plane packed and
  // starting on 256 bytes.
  const std::uint64_t w = width;
  const std::uint64_t h = height;
  const std::uint64_t cw = (w + 1) / 2;
  const std::uint64_t ch = (h + 1) / 2;
  const std::uint64_t luma = video::round_up(w * h, 256);
  const std::uint64_t chroma = video::round_up(cw * ch, 256);
  const std::uint64_t at[3] = {0, luma, luma + chroma};
  const std::uint64_t pitch[3] = {w, cw, cw};
  VKC_ASSIGN(video::CudaPictures::Target target,
             pictures->take(luma + 2 * chroma));

  nvjpegImage_t out{};
  for (int p = 0; p < 3; ++p) {
    out.channel[p] = reinterpret_cast<unsigned char*>(target.pointer + at[p]);
    out.pitch[p] = static_cast<std::size_t>(pitch[p]);
  }
  const bool decoded =
      step(n->nvjpegDecodeJpegHost(handle, engine->decoder, engine->state,
                                   params, parsed)) &&
      step(n->nvjpegDecodeJpegTransferToDevice(
          handle, engine->decoder, engine->state, parsed, stream)) &&
      step(n->nvjpegDecodeJpegDevice(handle, engine->decoder, engine->state,
                                     &out, stream));
  // What was queued finishes either way, so the buffers are free for the
  // next JPEG, and a failing GPU fails here.
  const video::CudaDriver* cu = video::cuda_driver();
  CUresult r = cu->cuEventRecord(done, stream);
  if (r == CUDA_SUCCESS) r = cu->cuEventSynchronize(done);
  if (r != CUDA_SUCCESS) return video::cuda_error(kWho, r, "decoding a JPEG");
  if (!decoded) return failed("decoding a JPEG");

  DecodedPicture picture;
  picture.width = width;
  picture.height = height;
  picture.layout = VideoPixelLayout::Yuv420;
  picture.device = std::move(target.buffer);
  for (int p = 0; p < 3; ++p) {
    picture.offset[p] = at[p];
    picture.stride[p] = static_cast<std::size_t>(pitch[p]);
  }
  picture.matrix = kJfif.matrix;
  picture.full_range = kJfif.full_range;
  picture.chroma_location = ChromaLocation::Center;
  return std::optional<DecodedPicture>(std::move(picture));
}

#endif  // VR_SENSOR_VIDEO_WITH_CUDA

}  // namespace

const char* to_string(JpegDecodeBackend backend) noexcept {
  switch (backend) {
    case JpegDecodeBackend::NvjpegHardware:
      return "nvjpeg-hardware";
    case JpegDecodeBackend::NvjpegGpu:
      return "nvjpeg-gpu";
    case JpegDecodeBackend::VideoToolbox:
      return "videotoolbox";
    case JpegDecodeBackend::Software:
      break;
  }
  return "software";
}

struct JpegDecoder::Impl {
  // FFmpeg's decoder, for the JPEGs the device path does not take.
  video::CodecContextPtr codec;
  video::PacketPtr packet;
  video::FramePtr frame;
  video::PictureConverter converter{kWho};
  std::string label;  // as in Options
#if VR_SENSOR_VIDEO_WITH_CUDA
  // Null without a device nvJPEG can use, and once the device path fails.
  std::unique_ptr<NvjpegDecoder> gpu;
#endif
#if defined(__APPLE__)
  // Null without a device VideoToolbox's pictures can reach, and once the
  // device path fails.
  std::unique_ptr<video::VtJpeg> vt;
#endif

  // Whether a JPEG can decode onto the device.
  bool device_path() const noexcept {
#if VR_SENSOR_VIDEO_WITH_CUDA
    if (gpu != nullptr) return true;
#endif
#if defined(__APPLE__)
    if (vt != nullptr) return true;
#endif
    return false;
  }

  core::Status open_software();
  core::Result<DecodedPicture> decode_software(const std::uint8_t* data,
                                               std::size_t size);
};

core::Status JpegDecoder::Impl::open_software() {
  const AVCodec* found = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  if (found == nullptr) {
    return core::Status::io_error(std::string(kWho) +
                                  ": this FFmpeg has no JPEG decoder");
  }
  codec.reset(avcodec_alloc_context3(found));
  packet.reset(av_packet_alloc());
  frame.reset(av_frame_alloc());
  if (codec == nullptr || packet == nullptr || frame == nullptr) {
    return video::ffmpeg_alloc_error(kWho, "opening the decoder");
  }
  codec->thread_count = 1;  // one picture a call, none held back
  const int err = avcodec_open2(codec.get(), found, nullptr);
  if (err < 0) return video::ffmpeg_error(kWho, "opening the decoder", err);
  return {};
}

core::Result<DecodedPicture> JpegDecoder::Impl::decode_software(
    const std::uint8_t* data, std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return core::Status::invalid_argument(std::string(kWho) +
                                          ": a JPEG past 2 GiB");
  }
  av_frame_unref(frame.get());
  // Not reference-counted, so FFmpeg copies it, padded as it needs.
  packet->data = const_cast<std::uint8_t*>(data);
  packet->size = static_cast<int>(size);
  int err = avcodec_send_packet(codec.get(), packet.get());
  packet->data = nullptr;
  packet->size = 0;
  if (err >= 0) err = avcodec_receive_frame(codec.get(), frame.get());
  if (err < 0) return video::ffmpeg_error(kWho, "decoding a JPEG", err);
  // JPEG's centred samples also apply when 4:2:2 is resampled to 4:2:0.
  frame->chroma_location = AVCHROMA_LOC_CENTER;
  return converter.convert(*frame, VideoPixelLayout::Yuv420, kJfif);
}

core::Result<JpegDecoder> JpegDecoder::create(const Options& options) {
  if (options.configure_ffmpeg_logging) av_log_set_level(AV_LOG_ERROR);
  auto impl = std::make_unique<Impl>();
  VKC_TRY(impl->open_software());
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (options.device != nullptr && options.allocator != nullptr) {
    impl->gpu = NvjpegDecoder::open(*options.device, *options.allocator);
  }
#endif
#if defined(__APPLE__)
  if (options.device != nullptr) {
    impl->vt = video::VtJpeg::open(*options.device, kWho);
  }
#endif
  impl->label = options.label;
  if (options.device != nullptr && !impl->device_path()) {
    std::string why = "no device path opened";
#if VR_SENSOR_VIDEO_WITH_CUDA
    if (options.allocator == nullptr) {
      why += " (nvJPEG needs JpegDecoder::Options::allocator)";
    }
#endif
    video::warn_host_pictures(kWho, impl->label, why);
  }
  return JpegDecoder(std::move(impl));
}

JpegDecoder::JpegDecoder(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
JpegDecoder::JpegDecoder(JpegDecoder&& other) noexcept = default;
JpegDecoder& JpegDecoder::operator=(JpegDecoder&& other) noexcept = default;
JpegDecoder::~JpegDecoder() = default;

core::Result<DecodedPicture> JpegDecoder::decode(const std::uint8_t* data,
                                                 std::size_t size) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": moved from");
  }
  if (data == nullptr || size == 0) {
    return core::Status::invalid_argument(std::string(kWho) + ": no JPEG");
  }
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (impl_->gpu != nullptr) {
    auto on_device = impl_->gpu->decode(data, size);
    if (on_device && on_device.value()) return std::move(*on_device.value());
    // A device path that fails, out of memory or refused by CUDA, is let go
    // with all it holds on the GPU: this JPEG and every later one decode in
    // software, which is said once.
    if (!on_device) {
      video::warn_host_pictures(
          kWho, impl_->label,
          "the device path failed (" + on_device.status().message() + ")");
      impl_->gpu.reset();
    }
  }
#endif
#if defined(__APPLE__)
  if (impl_->vt != nullptr) {
    auto on_device = impl_->vt->decode(data, size);
    if (on_device && on_device.value()) return std::move(*on_device.value());
    // As for nvJPEG.
    if (!on_device) {
      video::warn_host_pictures(
          kWho, impl_->label,
          "the device path failed (" + on_device.status().message() + ")");
      impl_->vt.reset();
    }
  }
#endif
  return impl_->decode_software(data, size);
}

JpegDecodeBackend JpegDecoder::backend() const noexcept {
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (impl_ != nullptr && impl_->gpu != nullptr) return impl_->gpu->backend;
#endif
#if defined(__APPLE__)
  if (impl_ != nullptr && impl_->vt != nullptr) {
    return JpegDecodeBackend::VideoToolbox;
  }
#endif
  return JpegDecodeBackend::Software;
}

}  // namespace volumetric_kit::recon::sensor
