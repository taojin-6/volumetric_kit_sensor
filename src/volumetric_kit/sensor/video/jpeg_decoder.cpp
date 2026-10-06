// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"

#include <string>
#include <utility>

#if VR_SENSOR_VIDEO_WITH_CUDA
#include <dlfcn.h>
#include <nvjpeg.h>

#include <optional>

#include "cuda_pictures.hpp"
#elif defined(__APPLE__)
#include "vt_jpeg.hpp"
#else
#error "JpegDecoder decodes on nvJPEG (VR_WITH_CUDA) or VideoToolbox only"
#endif

namespace volumetric_kit::recon::sensor {
namespace {

constexpr const char* kWho = "JpegDecoder";

core::Status unsupported(const std::string& why) {
  return core::Status::unsupported(std::string(kWho) + ": " + why);
}

#if VR_SENSOR_VIDEO_WITH_CUDA

// JFIF's matrix and range, which a JPEG that names none is coded in.
constexpr VideoColorDescription kJfif{VideoColorMatrix::Bt601, true};

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

// nvJPEG's verdict that the bytes are corrupt.
bool refused(nvjpegStatus_t s) {
  return s == NVJPEG_STATUS_BAD_JPEG || s == NVJPEG_STATUS_INCOMPLETE_BITSTREAM;
}

core::Status nvjpeg_error(nvjpegStatus_t s, const std::string& what) {
  return core::Status::backend_error(static_cast<std::int64_t>(s),
                                     std::string(kWho) + ": " + what +
                                         ": nvJPEG status " +
                                         std::to_string(static_cast<int>(s)));
}

// nvJPEG on the GPU's hardware JPEG engine, in the primary context of the
// Vulkan device's GPU, and everything it holds there, released together. Only
// the engine: nvJPEG's other back ends decode the entropy on the host (4.2 ms
// of CPU a 4K frame for GPU_HYBRID; the 2026-09-28 decoded-frame decision).
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
  VKC_ASSIGN(const int ordinal, video::cuda_ordinal_of(device, kWho));
  auto d = std::make_unique<NvjpegDecoder>();
  CUresult r = cu->cuDeviceGet(&d->cuda_device, ordinal);
  if (r == CUDA_SUCCESS) {
    r = cu->cuDevicePrimaryCtxRetain(&d->context, d->cuda_device);
  }
  if (r != CUDA_SUCCESS) {
    d->context = nullptr;
    return video::cuda_error(kWho, r, "retaining the GPU's CUDA context");
  }
  const video::CudaContextScope scope(d->context);
  if (!scope.ok()) {
    return video::cuda_error(kWho, scope.result(),
                             "making the CUDA context current");
  }
  r = cu->cuStreamCreate(&d->stream, CU_STREAM_NON_BLOCKING);
  if (r != CUDA_SUCCESS) {
    d->stream = nullptr;
    return video::cuda_error(kWho, r, "opening nvJPEG");
  }
  r = cu->cuEventCreate(&d->done,
                        CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING);
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
    return nvjpeg_error(s, "opening nvJPEG");
  }
  VKC_ASSIGN(d->pictures, video::CudaPictures::create(
                              device, allocator, d->context, d->stream, kWho));
  return d;
}

NvjpegDecoder::~NvjpegDecoder() {
  if (context == nullptr) return;
  const video::CudaDriver* cu = video::cuda_driver();
  const Nvjpeg* n = nvjpeg();
  {
    const video::CudaContextScope scope(context);
    pictures.reset();  // its imports go with the context
    if (state != nullptr) n->nvjpegJpegStateDestroy(state);
    if (decoder != nullptr) n->nvjpegDecoderDestroy(decoder);
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

core::Result<DecodedPicture> NvjpegDecoder::decode(const std::uint8_t* data,
                                                   std::size_t size) {
  const video::CudaContextScope scope(context);
  if (!scope.ok()) {
    return video::cuda_error(kWho, scope.result(),
                             "making the CUDA context current");
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
    return nvjpeg_error(status, what);
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
                                               stream)) &&
      step(n->nvjpegDecodeJpegDevice(handle, decoder, state, &out, stream));
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
  return picture;
}

#endif  // VR_SENSOR_VIDEO_WITH_CUDA

}  // namespace

struct JpegDecoder::Impl {
#if VR_SENSOR_VIDEO_WITH_CUDA
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
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (options.allocator == nullptr) {
    return unsupported(
        "no allocator to make nvJPEG's picture buffers through "
        "(Options::allocator)");
  }
  VKC_ASSIGN(impl->gpu,
             NvjpegDecoder::open(*options.device, *options.allocator));
#else
  VKC_ASSIGN(impl->vt, video::VtJpeg::open(*options.device, kWho));
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
#if VR_SENSOR_VIDEO_WITH_CUDA
  return impl_->gpu->decode(data, size);
#else
  return impl_->vt->decode(data, size);
#endif
}

}  // namespace volumetric_kit::recon::sensor
