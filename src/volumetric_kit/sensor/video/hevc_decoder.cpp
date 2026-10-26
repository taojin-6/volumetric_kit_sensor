// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"

#include <string>
#include <utility>

#if VR_SENSOR_VIDEO_WITH_CUDA
#include <climits>
#include <cstring>

#include "cuda_pictures.hpp"
#include "ffmpeg.hpp"
#include "frame_color.hpp"
extern "C" {
#include <libavutil/hwcontext_cuda.h>
}
#elif defined(__APPLE__)
#include "vt_hevc.hpp"
#else
#error "HevcDecoder decodes on NVDEC (VR_WITH_CUDA) or VideoToolbox only"
#endif

namespace volumetric_kit::recon::sensor {
namespace {

constexpr const char* kWho = "HevcDecoder";

core::Status unsupported(const std::string& why) {
  return core::Status::unsupported(std::string(kWho) + ": " + why);
}

#if VR_SENSOR_VIDEO_WITH_CUDA
constexpr AVHWDeviceType kDeviceType = AV_HWDEVICE_TYPE_CUDA;
constexpr AVPixelFormat kDeviceFormat = AV_PIX_FMT_CUDA;
constexpr const char* kPath = "NVDEC";

// FFmpeg's device context's free: the context and stream it was given.
void release_cuda_session(AVHWDeviceContext* device) {
  delete static_cast<video::CudaSession*>(device->user_opaque);
}

// Whether this FFmpeg decodes HEVC on the build's hardware path.
bool has_device_path(const AVCodec* codec) {
  for (int i = 0;; ++i) {
    const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
    if (config == nullptr) return false;
    if ((config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0 &&
        config->device_type == kDeviceType &&
        config->pix_fmt == kDeviceFormat) {
      return true;
    }
  }
}
#endif

}  // namespace

#if VR_SENSOR_VIDEO_WITH_CUDA
struct HevcDecoder::Impl {
  std::optional<VideoColorDescription> unlabelled_color;  // as in Options
  bool ended = false;
  std::string refusal;  // why the hardware gave up on the stream, if it did
  video::BufferRef hw_device;
  video::CodecContextPtr codec;
  video::PacketPtr packet;
  video::FramePtr decoded;
  std::unique_ptr<video::CudaPlanePictures> pictures;

  // @p frame handed over on the device; Unsupported for one the device path
  // cannot take.
  core::Result<DecodedPicture> picture(const AVFrame& frame);

  // FFmpeg's get_format, asked at each new SPS, and again without the
  // hardware format if the hardware fails to start: the hardware format, or
  // the refusal, for a stream it cannot decode or hand out.
  static AVPixelFormat pick_format(AVCodecContext* context,
                                   const AVPixelFormat* formats) {
    auto* impl = static_cast<Impl*>(context->opaque);
    // A stream whose VUI says full range is YUVJ420P.
    if (context->sw_pix_fmt != AV_PIX_FMT_YUV420P &&
        context->sw_pix_fmt != AV_PIX_FMT_YUVJ420P) {
      const char* name = av_get_pix_fmt_name(context->sw_pix_fmt);
      impl->refusal = std::string(kWho) +
                      ": the device path takes 8-bit 4:2:0 only, not " +
                      (name != nullptr ? name : "this stream's format");
      return AV_PIX_FMT_NONE;
    }
    for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
      if (*f == kDeviceFormat) return *f;
    }
    impl->refusal =
        std::string(kWho) + ": " + kPath + " cannot decode this stream";
    return AV_PIX_FMT_NONE;
  }
};

// Copied device to device into two plane images Vulkan samples. FFmpeg has
// taken the right and bottom crop off a hardware picture's size, and left the
// left and top to whoever reads it.
core::Result<DecodedPicture> HevcDecoder::Impl::picture(const AVFrame& frame) {
  const auto* frames = frame.hw_frames_ctx != nullptr
                           ? reinterpret_cast<const AVHWFramesContext*>(
                                 frame.hw_frames_ctx->data)
                           : nullptr;
  if (frame.format != AV_PIX_FMT_CUDA || frames == nullptr ||
      frames->sw_format != AV_PIX_FMT_NV12) {
    return unsupported(
        "NVDEC handed out a picture that is not NV12 on the GPU");
  }
  if ((frame.crop_left & 1) != 0 || (frame.crop_top & 1) != 0) {
    return unsupported(
        "a picture cropped by an odd count of columns or rows at the left or "
        "top cannot be handed out as NV12");
  }
  const std::size_t left = frame.crop_left;
  const std::size_t top = frame.crop_top;
  const auto width = static_cast<std::uint32_t>(
      frame.width - static_cast<int>(frame.crop_left + frame.crop_right));
  const auto height = static_cast<std::uint32_t>(
      frame.height - static_cast<int>(frame.crop_top + frame.crop_bottom));
  const auto luma_pitch = static_cast<std::size_t>(frame.linesize[0]);
  const auto chroma_pitch = static_cast<std::size_t>(frame.linesize[1]);
  const CUdeviceptr luma =
      reinterpret_cast<CUdeviceptr>(frame.data[0]) + top * luma_pitch + left;
  const CUdeviceptr chroma = reinterpret_cast<CUdeviceptr>(frame.data[1]) +
                             top / 2 * chroma_pitch + left;
  DecodedPicture out;
  VKC_TRY(pictures->copy(luma, luma_pitch, chroma, chroma_pitch, width, height,
                         out.yuv));
  video::describe_color(video::color_codes(frame), unlabelled_color, out);
  return out;
}
#else
struct HevcDecoder::Impl {
  std::unique_ptr<video::VtHevc> vt;
};
#endif

core::Result<HevcDecoder> HevcDecoder::create(const Options& options) {
  if (options.device == nullptr) {
    return unsupported("no device to decode onto (Options::device)");
  }
  auto impl = std::make_unique<Impl>();
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (options.allocator == nullptr) {
    return unsupported(
        "no allocator to make NVDEC's picture images through "
        "(Options::allocator)");
  }
  if (options.configure_ffmpeg_logging) av_log_set_level(AV_LOG_ERROR);
  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
  if (codec == nullptr || !has_device_path(codec)) {
    return unsupported(std::string("this FFmpeg has no ") + kPath +
                       " HEVC decoder");
  }
  impl->unlabelled_color = options.unlabelled_color;
  impl->codec.reset(avcodec_alloc_context3(codec));
  impl->packet.reset(av_packet_alloc());
  impl->decoded.reset(av_frame_alloc());
  if (impl->codec == nullptr || impl->packet == nullptr ||
      impl->decoded == nullptr) {
    return video::ffmpeg_alloc_error(kWho, "opening the decoder");
  }

  // The hardware device: the primary context of the Vulkan device's GPU and a
  // stream of this decoder's own, which FFmpeg's device holds and lets go
  // with its last reference.
  VKC_ASSIGN(std::unique_ptr<video::CudaSession> cuda,
             video::CudaSession::open(*options.device, kWho));
  const CudaContext& cuda_context = *cuda->context;
  AVBufferRef* hw = av_hwdevice_ctx_alloc(kDeviceType);
  if (hw == nullptr) return video::ffmpeg_alloc_error(kWho, "opening NVDEC");
  impl->hw_device.reset(hw);
  auto* hw_context = reinterpret_cast<AVHWDeviceContext*>(hw->data);
  auto* hw_cuda = static_cast<AVCUDADeviceContext*>(hw_context->hwctx);
  hw_cuda->cuda_ctx = cuda->context->handle();
  hw_cuda->stream = cuda->stream->handle();
  hw_context->user_opaque = cuda.release();
  hw_context->free = &release_cuda_session;
  int err = av_hwdevice_ctx_init(hw);
  if (err < 0) {
    return unsupported(std::string(kPath) +
                       ": no device opens: " + video::ffmpeg_message(err));
  }
  VKC_ASSIGN(
      impl->pictures,
      video::CudaPlanePictures::create(
          *options.device, *options.allocator, cuda_context, hw_cuda->stream,
          options.queue_families, options.queue_family_count, kWho));

  AVCodecContext* context = impl->codec.get();
  context->hw_device_ctx = av_buffer_ref(hw);
  if (context->hw_device_ctx == nullptr) {
    return video::ffmpeg_alloc_error(kWho, "sharing the device");
  }
  context->opaque = impl.get();
  context->get_format = &Impl::pick_format;
  // Parameter-only packets are parsed by send(), before it returns. Hardware
  // decoding needs no frame threads holding those packets for a later call.
  context->thread_count = 1;
  err = avcodec_open2(context, codec, nullptr);
  if (err < 0) return video::ffmpeg_error(kWho, "opening the decoder", err);
#else
  VKC_ASSIGN(impl->vt, video::VtHevc::open(
                           *options.device, kWho, options.unlabelled_color,
                           options.queue_families, options.queue_family_count));
#endif
  return HevcDecoder(std::move(impl));
}

HevcDecoder::HevcDecoder(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
HevcDecoder::HevcDecoder(HevcDecoder&& other) noexcept = default;
HevcDecoder& HevcDecoder::operator=(HevcDecoder&& other) noexcept = default;
HevcDecoder::~HevcDecoder() = default;

core::Status HevcDecoder::send(const std::uint8_t* data, std::size_t size,
                               std::int64_t pts) {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": moved from");
  }
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (!impl_->refusal.empty()) return core::Status::unsupported(impl_->refusal);
  if (impl_->ended) {
    return core::Status::invalid_argument(
        std::string(kWho) + ": data sent after the end of the stream");
  }
  AVCodecContext* context = impl_->codec.get();
  if (size == 0) {
    impl_->ended = true;
    const int err = avcodec_send_packet(context, nullptr);
    if (err < 0) return video::ffmpeg_error(kWho, "ending the stream", err);
    return {};
  }
  if (data == nullptr || size > static_cast<std::size_t>(INT_MAX)) {
    return core::Status::invalid_argument(
        std::string(kWho) + ": an access unit needs data and under 2 GiB");
  }
  AVPacket* packet = impl_->packet.get();
  av_packet_unref(packet);
  int err = av_new_packet(packet, static_cast<int>(size));
  if (err < 0) return video::ffmpeg_error(kWho, "allocating a packet", err);
  std::memcpy(packet->data, data, size);
  packet->pts = pts;
  err = avcodec_send_packet(context, packet);
  if (!impl_->refusal.empty()) return core::Status::unsupported(impl_->refusal);
  if (err == AVERROR(EAGAIN)) {
    return core::Status::invalid_argument(
        std::string(kWho) + ": take the pictures waiting before sending more");
  }
  if (err < 0) return video::ffmpeg_error(kWho, "decoding", err);
  return {};
#else
  return impl_->vt->send(data, size, pts);
#endif
}

core::Result<std::optional<DecodedPicture>> HevcDecoder::receive() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": moved from");
  }
#if VR_SENSOR_VIDEO_WITH_CUDA
  AVFrame* decoded = impl_->decoded.get();
  av_frame_unref(decoded);
  const int err = avcodec_receive_frame(impl_->codec.get(), decoded);
  if (err < 0) {
    // Pictures decoded before a refusal still come out; then the refusal.
    if (!impl_->refusal.empty())
      return core::Status::unsupported(impl_->refusal);
    if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
      return std::optional<DecodedPicture>();
    }
    return video::ffmpeg_error(kWho, "decoding", err);
  }
  core::Result<DecodedPicture> picture = impl_->picture(*decoded);
  if (!picture) {
    // A picture the device path cannot take: the next ones would be the same.
    if (picture.status().domain() == core::Status::Code::Unsupported) {
      impl_->refusal = picture.status().message();
    }
    return picture.status();
  }
  picture.value().pts = decoded->pts != AV_NOPTS_VALUE
                            ? decoded->pts
                            : decoded->best_effort_timestamp;
  return std::optional<DecodedPicture>(std::move(picture).value());
#else
  return impl_->vt->receive();
#endif
}

core::Status HevcDecoder::reset() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": moved from");
  }
#if VR_SENSOR_VIDEO_WITH_CUDA
  // As for a seek: FFmpeg drops what it holds, leaves draining, and takes the
  // next key frame as the first, skipping the pictures that lead it.
  avcodec_flush_buffers(impl_->codec.get());
  impl_->ended = false;
#else
  impl_->vt->reset();
#endif
  return {};
}

}  // namespace volumetric_kit::recon::sensor
