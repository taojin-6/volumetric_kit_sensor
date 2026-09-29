// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/jpeg_decoder.hpp"

#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "ffmpeg.hpp"
#include "picture_converter.hpp"
#if VR_SENSOR_VIDEO_WITH_CUDA
#include <dlfcn.h>
#include <nvjpeg.h>

#include "cuda_pictures.hpp"
#include "volumetric_kit/recon/core/device.hpp"
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

// libnvjpeg, loaded once and kept; null where it does not load or lacks an
// entry point.
const Nvjpeg* nvjpeg() {
  static const std::optional<Nvjpeg> loaded = []() -> std::optional<Nvjpeg> {
    void* lib = dlopen("libnvjpeg.so.13", RTLD_NOW | RTLD_LOCAL);
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

#endif  // VR_SENSOR_VIDEO_WITH_CUDA

}  // namespace

const char* to_string(JpegDecodeBackend backend) noexcept {
  switch (backend) {
    case JpegDecodeBackend::NvjpegHardware:
      return "nvjpeg-hardware";
    case JpegDecodeBackend::NvjpegGpu:
      return "nvjpeg-gpu";
    case JpegDecodeBackend::Software:
      break;
  }
  return "software";
}

struct JpegDecoder::Impl {
  JpegDecodeBackend backend = JpegDecodeBackend::Software;
  // FFmpeg's decoder, for the JPEGs the device path does not take.
  video::CodecContextPtr codec;
  video::PacketPtr packet;
  video::FramePtr frame;
  video::PictureConverter converter{kWho};

  Status open_software();
  Result<DecodedPicture> decode_software(const std::uint8_t* data,
                                         std::size_t size);

#if VR_SENSOR_VIDEO_WITH_CUDA
  // nvJPEG in the primary context of the Vulkan device's GPU. A picture waits
  // on an event made to block, not spin: CUDA's default wait bills the whole
  // wait as CPU.
  CUdevice cuda_device = 0;
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUevent done = nullptr;
  nvjpegHandle_t handle = nullptr;
  nvjpegJpegDecoder_t decoder = nullptr;
  nvjpegJpegState_t state = nullptr;
  nvjpegBufferPinned_t pinned = nullptr;
  nvjpegBufferDevice_t staging = nullptr;
  nvjpegJpegStream_t parsed = nullptr;
  nvjpegDecodeParams_t params = nullptr;
  std::unique_ptr<video::CudaPictures> pictures;

  ~Impl();
  // Open nvJPEG on @p device's GPU, its hardware engine first; a device it
  // cannot use leaves the decoder in software.
  void open_nvjpeg(const Device& device);
  bool open_backend(const Device& device, nvjpegBackend_t nv);
  void close_backend();
  // The JPEG on the device; empty for one the device path does not take,
  // which goes to software instead. An error means the device path failed.
  Result<std::optional<DecodedPicture>> decode_device(const std::uint8_t* data,
                                                      std::size_t size);
#endif
};

Status JpegDecoder::Impl::open_software() {
  const AVCodec* found = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  if (found == nullptr) {
    return Status::io_error(std::string(kWho) +
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

Result<DecodedPicture> JpegDecoder::Impl::decode_software(
    const std::uint8_t* data, std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Status::invalid_argument(std::string(kWho) + ": a JPEG past 2 GiB");
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
  return converter.convert(*frame, VideoPixelLayout::Yuv420, kJfif);
}

#if VR_SENSOR_VIDEO_WITH_CUDA

JpegDecoder::Impl::~Impl() {
  if (context == nullptr) return;
  const video::CudaDriver* cu = video::cuda_driver();
  {
    const video::CudaContextScope scope(context);
    close_backend();
    if (done != nullptr) cu->cuEventDestroy(done);
    if (stream != nullptr) cu->cuStreamDestroy(stream);
    pictures.reset();  // its imports go with the context
  }
  cu->cuDevicePrimaryCtxRelease(cuda_device);
}

void JpegDecoder::Impl::open_nvjpeg(const Device& device) {
  const video::CudaDriver* cu = video::cuda_driver();
  if (!device.exports_memory() || cu == nullptr || nvjpeg() == nullptr) return;
  const auto ordinal = video::cuda_ordinal_of(device, kWho);
  if (!ordinal ||
      cu->cuDeviceGet(&cuda_device, ordinal.value()) != CUDA_SUCCESS ||
      cu->cuDevicePrimaryCtxRetain(&context, cuda_device) != CUDA_SUCCESS) {
    context = nullptr;
    return;
  }
  const video::CudaContextScope scope(context);
  if (!scope.ok() ||
      cu->cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
      cu->cuEventCreate(&done, CU_EVENT_BLOCKING_SYNC |
                                   CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
    return;
  }
  // The hardware engine where the GPU has one, else the GPU's cores.
  if (open_backend(device, NVJPEG_BACKEND_HARDWARE)) {
    backend = JpegDecodeBackend::NvjpegHardware;
  } else if (open_backend(device, NVJPEG_BACKEND_GPU_HYBRID)) {
    backend = JpegDecodeBackend::NvjpegGpu;
  }
}

bool JpegDecoder::Impl::open_backend(const Device& device, nvjpegBackend_t nv) {
  const Nvjpeg* n = nvjpeg();
  const auto ok = [](nvjpegStatus_t s) { return s == NVJPEG_STATUS_SUCCESS; };
  if (ok(n->nvjpegCreateEx(nv, nullptr, nullptr, 0, &handle)) &&
      ok(n->nvjpegDecoderCreate(handle, nv, &decoder)) &&
      ok(n->nvjpegDecoderStateCreate(handle, decoder, &state)) &&
      ok(n->nvjpegBufferPinnedCreate(handle, nullptr, &pinned)) &&
      ok(n->nvjpegBufferDeviceCreate(handle, nullptr, &staging)) &&
      ok(n->nvjpegStateAttachPinnedBuffer(state, pinned)) &&
      ok(n->nvjpegStateAttachDeviceBuffer(state, staging)) &&
      ok(n->nvjpegJpegStreamCreate(handle, &parsed)) &&
      ok(n->nvjpegDecodeParamsCreate(handle, &params)) &&
      ok(n->nvjpegDecodeParamsSetOutputFormat(params, NVJPEG_OUTPUT_YUV))) {
    auto made = video::CudaPictures::create(device, context, stream, kWho);
    if (made) {
      pictures = std::move(made).value();
      return true;
    }
  }
  close_backend();
  return false;
}

void JpegDecoder::Impl::close_backend() {
  const Nvjpeg* n = nvjpeg();
  if (params != nullptr) n->nvjpegDecodeParamsDestroy(params);
  if (parsed != nullptr) n->nvjpegJpegStreamDestroy(parsed);
  if (state != nullptr) n->nvjpegJpegStateDestroy(state);
  if (staging != nullptr) n->nvjpegBufferDeviceDestroy(staging);
  if (pinned != nullptr) n->nvjpegBufferPinnedDestroy(pinned);
  if (decoder != nullptr) n->nvjpegDecoderDestroy(decoder);
  if (handle != nullptr) n->nvjpegDestroy(handle);
  params = nullptr;
  parsed = nullptr;
  state = nullptr;
  staging = nullptr;
  pinned = nullptr;
  decoder = nullptr;
  handle = nullptr;
}

Result<std::optional<DecodedPicture>> JpegDecoder::Impl::decode_device(
    const std::uint8_t* data, std::size_t size) {
  const video::CudaContextScope scope(context);
  if (!scope.ok()) {
    return Status::io_error(std::string(kWho) +
                            ": making the CUDA context current");
  }
  // Only an 8-bit 4:2:0 JPEG this back end decodes; anything else, a JPEG
  // nvJPEG cannot parse included, goes to software, which says what is wrong.
  const Nvjpeg* n = nvjpeg();
  const auto ok = [](nvjpegStatus_t s) { return s == NVJPEG_STATUS_SUCCESS; };
  unsigned width = 0;
  unsigned height = 0;
  unsigned components = 0;
  unsigned precision = 0;
  nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
  int unsupported = -1;
  if (!ok(n->nvjpegJpegStreamParse(handle, data, size, 0, 0, parsed)) ||
      !ok(n->nvjpegJpegStreamGetFrameDimensions(parsed, &width, &height)) ||
      !ok(n->nvjpegJpegStreamGetComponentsNum(parsed, &components)) ||
      !ok(n->nvjpegJpegStreamGetSamplePrecision(parsed, &precision)) ||
      !ok(n->nvjpegJpegStreamGetChromaSubsampling(parsed, &subsampling)) ||
      !ok(n->nvjpegDecoderJpegSupported(decoder, parsed, params,
                                        &unsupported)) ||
      unsupported != 0 || components != 3 || precision != 8 ||
      subsampling != NVJPEG_CSS_420 || width == 0 || height == 0) {
    return std::optional<DecodedPicture>();
  }
  // Y, then Cb and Cr at half size rounded up, each plane packed and
  // starting on 256 bytes.
  const std::uint64_t w = width;
  const std::uint64_t h = height;
  const std::uint64_t cw = (w + 1) / 2;
  const std::uint64_t ch = (h + 1) / 2;
  const auto round_up = [](std::uint64_t v) { return (v + 255) / 256 * 256; };
  const std::uint64_t at[3] = {0, round_up(w * h),
                               round_up(w * h) + round_up(cw * ch)};
  const std::uint64_t pitch[3] = {w, cw, cw};
  VR_ASSIGN(video::CudaPictures::Target target,
            pictures->take(at[2] + round_up(cw * ch)));

  nvjpegImage_t out{};
  for (int p = 0; p < 3; ++p) {
    out.channel[p] = reinterpret_cast<unsigned char*>(target.pointer + at[p]);
    out.pitch[p] = static_cast<std::size_t>(pitch[p]);
  }
  // A JPEG that parses and then does not decode goes to software, like one
  // that does not parse; only CUDA failing lets the device path go.
  if (!ok(n->nvjpegDecodeJpegHost(handle, decoder, state, params, parsed)) ||
      !ok(n->nvjpegDecodeJpegTransferToDevice(handle, decoder, state, parsed,
                                              stream)) ||
      !ok(n->nvjpegDecodeJpegDevice(handle, decoder, state, &out, stream))) {
    return std::optional<DecodedPicture>();
  }
  const video::CudaDriver* cu = video::cuda_driver();
  CUresult r = cu->cuEventRecord(done, stream);
  if (r == CUDA_SUCCESS) r = cu->cuEventSynchronize(done);
  if (r != CUDA_SUCCESS) return video::cuda_error(kWho, r, "decoding a JPEG");

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
  return std::optional<DecodedPicture>(std::move(picture));
}

#endif  // VR_SENSOR_VIDEO_WITH_CUDA

Result<JpegDecoder> JpegDecoder::create(const Options& options) {
  if (options.configure_ffmpeg_logging) av_log_set_level(AV_LOG_ERROR);
  auto impl = std::make_unique<Impl>();
  VR_TRY(impl->open_software());
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (options.device != nullptr) impl->open_nvjpeg(*options.device);
#endif
  return JpegDecoder(std::move(impl));
}

JpegDecoder::JpegDecoder(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
JpegDecoder::JpegDecoder(JpegDecoder&& other) noexcept = default;
JpegDecoder& JpegDecoder::operator=(JpegDecoder&& other) noexcept = default;
JpegDecoder::~JpegDecoder() = default;

Result<DecodedPicture> JpegDecoder::decode(const std::uint8_t* data,
                                           std::size_t size) {
  if (impl_ == nullptr) {
    return Status::invalid_argument(std::string(kWho) + ": moved from");
  }
  if (data == nullptr || size == 0) {
    return Status::invalid_argument(std::string(kWho) + ": no JPEG");
  }
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (impl_->pictures != nullptr) {
    auto on_device = impl_->decode_device(data, size);
    if (on_device && on_device.value()) return std::move(*on_device.value());
    // A device path that fails, out of memory or refused by CUDA, is let
    // go: this JPEG and every later one decode in software.
    if (!on_device) {
      impl_->pictures.reset();
      impl_->backend = JpegDecodeBackend::Software;
    }
  }
#endif
  return impl_->decode_software(data, size);
}

JpegDecodeBackend JpegDecoder::backend() const noexcept {
  return impl_ != nullptr ? impl_->backend : JpegDecodeBackend::Software;
}

}  // namespace volumetric_kit::recon::sensor
