// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/video/jpeg_decoder.hpp"

#include <string>
#include <utility>

#if VS_VIDEO_WITH_CUDA
#include <dlfcn.h>

#include <optional>

#include "cuda_pictures.hpp"
#include "cuda_timeline.hpp"
#include "nvjpeg.hpp"
#elif defined(__APPLE__)
#include "vt_jpeg.hpp"
#else
#error "JpegDecoder decodes on nvJPEG (VS_WITH_CUDA) or VideoToolbox only"
#endif

namespace volumetric_kit::sensor {
namespace {

constexpr const char* kWho = "JpegDecoder";

core::Status unsupported(const std::string& why) {
  return core::Status::unsupported(std::string(kWho) + ": " + why);
}

// How a JPEG is coded, as JFIF defines it: BT.601 full range, its chroma
// centred.
void describe_jfif(YuvImage& yuv) noexcept {
  const YcbcrWeights weights = ycbcr_weights(VideoColorMatrix::Bt601);
  yuv.kr = weights.kr;
  yuv.kb = weights.kb;
  yuv.full_range = true;
  yuv.chroma_location = ChromaLocation::Center;
}

#if VS_VIDEO_WITH_CUDA

// The nvJPEG entry points used here, loaded at run time as libcuda is.
#define VS_NVJPEG_FUNCTIONS(X)            \
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
#define VS_NVJPEG_DECLARE(name) decltype(&::name) name = nullptr;
  VS_NVJPEG_FUNCTIONS(VS_NVJPEG_DECLARE)
#undef VS_NVJPEG_DECLARE
};

// libnvjpeg of the major version compiled against, loaded once and kept;
// null where it does not load or lacks an entry point.
const Nvjpeg* nvjpeg() {
  static const std::optional<Nvjpeg> loaded = []() -> std::optional<Nvjpeg> {
    void* lib = dlopen("libnvjpeg.so." VS_CUDA_STRING(NVJPEG_VER_MAJOR),
                       RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return std::nullopt;
    Nvjpeg n;
    bool ok = true;
#define VS_NVJPEG_LOAD(name)                                      \
  n.name = reinterpret_cast<decltype(n.name)>(dlsym(lib, #name)); \
  ok = ok && n.name != nullptr;
    VS_NVJPEG_FUNCTIONS(VS_NVJPEG_LOAD)
#undef VS_NVJPEG_LOAD
    if (!ok) {
      dlclose(lib);
      return std::nullopt;
    }
    return n;
  }();
  return loaded ? &*loaded : nullptr;
}

bool ok(nvjpegStatus_t s) { return s == NVJPEG_STATUS_SUCCESS; }

// nvJPEG's verdict that the bytes are corrupt.
bool refused(nvjpegStatus_t s) {
  return s == NVJPEG_STATUS_BAD_JPEG || s == NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
}

// nvJPEG on the GPU's hardware JPEG engine, on a stream of its own in the
// primary context of the Vulkan device's GPU, and everything it holds there,
// released together. Only the engine: nvJPEG's other back ends decode the
// entropy on the host (4.2 ms of CPU a 4K frame for GPU_HYBRID; the
// 2026-09-28 decoded-frame decision).
struct NvjpegDecoder {
  // Its picture buffers are made through @p allocator.
  static core::Result<std::unique_ptr<NvjpegDecoder>> open(
      const core::Device& device, core::Allocator& allocator);
  NvjpegDecoder() = default;
  ~NvjpegDecoder();
  NvjpegDecoder(const NvjpegDecoder&) = delete;
  NvjpegDecoder& operator=(const NvjpegDecoder&) = delete;

  core::Result<DecodedPicture> decode(const std::uint8_t* data,
                                      std::size_t size);

  std::unique_ptr<video::CudaSession> cuda;  // let go last
  // Set on the stream after each JPEG's work, and waited on at the next
  // decode, before nvJPEG's pinned, device and state buffers are rewritten:
  // the last JPEG's work is long done by then, so the wait seldom blocks. Made
  // to block, not spin: CUDA's default wait bills the whole wait as CPU.
  CUevent done = nullptr;
  bool queued = false;  // `done` recorded and not yet waited on
  // The pictures' ready values, which CUDA sets after each decode; after the
  // session, whose context and stream it uses.
  std::unique_ptr<video::CudaTimeline> timeline;
  nvjpegHandle_t handle = nullptr;
  nvjpegBufferPinned_t pinned = nullptr;
  nvjpegBufferDevice_t staging = nullptr;
  nvjpegJpegStream_t parsed = nullptr;
  nvjpegDecodeParams_t params = nullptr;
  nvjpegJpegDecoder_t decoder = nullptr;
  nvjpegJpegState_t state = nullptr;
  std::unique_ptr<video::CudaPictures> pictures;
};

core::Result<std::unique_ptr<NvjpegDecoder>> NvjpegDecoder::open(
    const core::Device& device, core::Allocator& allocator) {
  const video::CudaDriver* cu = video::cuda_driver();
  const Nvjpeg* n = nvjpeg();
  if (cu == nullptr) return unsupported("libcuda does not load");
  if (n == nullptr) return unsupported("libnvjpeg does not load");
  auto d = std::make_unique<NvjpegDecoder>();
  VKC_ASSIGN(d->cuda, video::CudaSession::open(device, kWho));
  const video::CudaContextScope scope(d->cuda->context->handle());
  if (!scope.ok()) {
    return video::cuda_error(kWho, scope.result(),
                             "making the CUDA context current");
  }
  const CUresult r = cu->cuEventCreate(
      &d->done, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING);
  if (r != CUDA_SUCCESS) {
    d->done = nullptr;
    return video::cuda_error(kWho, r, "opening nvJPEG");
  }
  const auto no_engine = [](nvjpegStatus_t status) {
    return unsupported("the GPU has no hardware JPEG engine (nvJPEG status " +
                       std::to_string(static_cast<int>(status)) + ")");
  };
  nvjpegStatus_t s = n->nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr,
                                       nullptr, 0, &d->handle);
  if (!ok(s)) {
    d->handle = nullptr;
    return no_engine(s);
  }
  s = n->nvjpegDecoderCreate(d->handle, NVJPEG_BACKEND_HARDWARE, &d->decoder);
  if (!ok(s)) {
    d->decoder = nullptr;
    return no_engine(s);
  }
  if (!ok(s = n->nvjpegBufferPinnedCreate(d->handle, nullptr, &d->pinned)) ||
      !ok(s = n->nvjpegBufferDeviceCreate(d->handle, nullptr, &d->staging)) ||
      !ok(s = n->nvjpegJpegStreamCreate(d->handle, &d->parsed)) ||
      !ok(s = n->nvjpegDecodeParamsCreate(d->handle, &d->params)) ||
      !ok(s = n->nvjpegDecodeParamsSetOutputFormat(d->params,
                                                   NVJPEG_OUTPUT_YUV)) ||
      !ok(s = n->nvjpegDecoderStateCreate(d->handle, d->decoder, &d->state)) ||
      !ok(s = n->nvjpegStateAttachPinnedBuffer(d->state, d->pinned)) ||
      !ok(s = n->nvjpegStateAttachDeviceBuffer(d->state, d->staging))) {
    return video::nvjpeg_error(kWho, s, "opening nvJPEG");
  }
  VKC_ASSIGN(d->timeline,
             video::CudaTimeline::create(device, *d->cuda->context,
                                         d->cuda->stream->handle(), kWho));
  VKC_ASSIGN(d->pictures,
             video::CudaPictures::create(device, allocator, *d->cuda->context,
                                         *d->timeline, kWho));
  return d;
}

NvjpegDecoder::~NvjpegDecoder() {
  if (cuda == nullptr) return;
  const video::CudaDriver* cu = video::cuda_driver();
  const Nvjpeg* n = nvjpeg();
  const video::CudaContextScope scope(cuda->context->handle());
  // The stream's work done first: it writes the pictures' buffers and reads
  // nvJPEG's.
  timeline.reset();
  pictures.reset();  // its imports go with the context
  if (state != nullptr) n->nvjpegJpegStateDestroy(state);
  if (decoder != nullptr) n->nvjpegDecoderDestroy(decoder);
  if (params != nullptr) n->nvjpegDecodeParamsDestroy(params);
  if (parsed != nullptr) n->nvjpegJpegStreamDestroy(parsed);
  if (staging != nullptr) n->nvjpegBufferDeviceDestroy(staging);
  if (pinned != nullptr) n->nvjpegBufferPinnedDestroy(pinned);
  if (handle != nullptr) n->nvjpegDestroy(handle);
  if (done != nullptr) cu->cuEventDestroy(done);
}

core::Result<DecodedPicture> NvjpegDecoder::decode(const std::uint8_t* data,
                                                   std::size_t size) {
  const video::CudaContextScope scope(cuda->context->handle());
  if (!scope.ok()) {
    return video::cuda_error(kWho, scope.result(),
                             "making the CUDA context current");
  }
  const video::CudaDriver* cu = video::cuda_driver();
  // The last JPEG's queued work, which reads nvJPEG's buffers, done before
  // they are rewritten; a failing GPU fails here.
  if (queued) {
    queued = false;
    const CUresult r = cu->cuEventSynchronize(done);
    if (r != CUDA_SUCCESS) {
      timeline->settle();
      return video::cuda_error(kWho, r, "decoding a JPEG");
    }
  }
  // nvJPEG's last status: corrupt bytes lose this JPEG, one it does not take
  // is Unsupported, any other fails the device path.
  nvjpegStatus_t status = NVJPEG_STATUS_SUCCESS;
  const auto step = [&status](nvjpegStatus_t s) { return ok(status = s); };
  const auto failed = [&status](const char* what) -> core::Status {
    if (status == NVJPEG_STATUS_JPEG_NOT_SUPPORTED) {
      return core::Status::unsupported(std::string(kWho) + ": " + what +
                                       ": nvJPEG does not take this JPEG");
    }
    if (refused(status)) {
      return core::Status::io_error(std::string(kWho) + ": " + what +
                                    ": the bytes do not decode (nvJPEG "
                                    "status " +
                                    std::to_string(static_cast<int>(status)) +
                                    ")");
    }
    return video::nvjpeg_error(kWho, status, what);
  };

  // Only an 8-bit 4:2:0 JPEG the engine takes.
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
  if (width == 0 || height == 0) {
    return core::Status::io_error(std::string(kWho) + ": a JPEG of no size");
  }
  if (components != 3 || precision != 8 || subsampling != NVJPEG_CSS_420) {
    return unsupported("the hardware takes baseline 8-bit 4:2:0 JPEGs only");
  }
  int refuses = -1;
  if (!step(n->nvjpegDecoderJpegSupported(decoder, parsed, params, &refuses))) {
    return failed("parsing a JPEG");
  }
  if (refuses != 0) {
    return unsupported("the hardware JPEG engine does not take a " +
                       std::to_string(width) + "x" + std::to_string(height) +
                       " JPEG");
  }

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
      step(n->nvjpegDecodeJpegHost(handle, decoder, state, params, parsed)) &&
      step(n->nvjpegDecodeJpegTransferToDevice(handle, decoder, state, parsed,
                                               cuda->stream->handle())) &&
      step(n->nvjpegDecodeJpegDevice(handle, decoder, state, &out,
                                     cuda->stream->handle()));
  // The picture's ready value after its decode, and the event after both,
  // which the next decode waits on; what was queued runs either way.
  DecodedPicture picture;
  const core::Status marked =
      decoded ? pictures->mark(target, picture.yuv) : core::Status();
  const CUresult r = cu->cuEventRecord(done, cuda->stream->handle());
  if (!marked.ok() || r != CUDA_SUCCESS) {
    timeline->settle();
    return marked.ok() ? video::cuda_error(kWho, r, "decoding a JPEG") : marked;
  }
  queued = true;
  if (!decoded) return failed("decoding a JPEG");

  picture.yuv.width = width;
  picture.yuv.height = height;
  picture.yuv.layout = YuvLayout::I420;
  video::place_cuda_buffer(std::move(target.buffer), picture.yuv);
  for (int p = 0; p < 3; ++p) {
    picture.yuv.offset[p] = at[p];
    picture.yuv.stride[p] = static_cast<std::size_t>(pitch[p]);
  }
  return picture;
}

#endif  // VS_VIDEO_WITH_CUDA

}  // namespace

struct JpegDecoder::Impl {
#if VS_VIDEO_WITH_CUDA
  std::unique_ptr<NvjpegDecoder> gpu;
#else
  std::unique_ptr<video::VtJpeg> vt;
#endif
};

core::Result<JpegDecoder> JpegDecoder::create(const Options& options) {
  if (options.device == nullptr) {
    return unsupported("no device to decode onto (Options::device)");
  }
  auto impl = std::make_unique<Impl>();
#if VS_VIDEO_WITH_CUDA
  if (options.allocator == nullptr) {
    return unsupported(
        "no allocator to make nvJPEG's picture buffers through "
        "(Options::allocator)");
  }
  // TODO(sensor): Options::queue_families for nvJPEG's buffers, which a
  // reader on any family takes over from outside Vulkan for now.
  VKC_ASSIGN(impl->gpu,
             NvjpegDecoder::open(*options.device, *options.allocator));
#else
  VKC_ASSIGN(impl->vt,
             video::VtJpeg::open(*options.device, kWho, options.queue_families,
                                 options.queue_family_count));
#endif
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
#if VS_VIDEO_WITH_CUDA
  VKC_ASSIGN(DecodedPicture picture, impl_->gpu->decode(data, size));
#else
  VKC_ASSIGN(DecodedPicture picture, impl_->vt->decode(data, size));
#endif
  describe_jfif(picture.yuv);
  return picture;
}

}  // namespace volumetric_kit::sensor
