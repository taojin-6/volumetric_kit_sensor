// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"

#include <climits>
#include <cstring>
#include <string>
#include <utility>

#include "ffmpeg.hpp"
#include "frame_color.hpp"
#if VR_SENSOR_VIDEO_WITH_CUDA
#include "cuda_pictures.hpp"
extern "C" {
#include <libavutil/hwcontext_cuda.h>
}
#elif defined(__APPLE__)
#include <vector>

#include "vt_pictures.hpp"
#else
#error "HevcDecoder decodes on NVDEC (VR_WITH_CUDA) or VideoToolbox only"
#endif

namespace volumetric_kit::recon::sensor {
namespace {

constexpr const char* kWho = "HevcDecoder";

#if VR_SENSOR_VIDEO_WITH_CUDA
constexpr AVHWDeviceType kDeviceType = AV_HWDEVICE_TYPE_CUDA;
constexpr AVPixelFormat kDeviceFormat = AV_PIX_FMT_CUDA;
constexpr const char* kPath = "NVDEC";
#else
constexpr AVHWDeviceType kDeviceType = AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
constexpr AVPixelFormat kDeviceFormat = AV_PIX_FMT_VIDEOTOOLBOX;
constexpr const char* kPath = "VideoToolbox";

// A NAL unit's payload bits, emulation prevention removed, read MSB first. A
// read past the end reads zeros and sets overrun().
class BitReader {
 public:
  BitReader(const std::uint8_t* payload, std::size_t size) {
    int zeros = 0;
    for (std::size_t i = 0; i < size; ++i) {
      if (zeros >= 2 && payload[i] == 3) {  // the 03 of 00 00 03
        zeros = 0;
        continue;
      }
      zeros = payload[i] == 0 ? zeros + 1 : 0;
      rbsp_.push_back(payload[i]);
    }
  }
  std::uint32_t bits(int n) {
    std::uint32_t value = 0;
    for (; n > 0; --n) {
      if (pos_ >= 8 * rbsp_.size()) {
        overrun_ = true;
        return 0;
      }
      value = (value << 1) | ((rbsp_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1u);
      ++pos_;
    }
    return value;
  }
  void skip(std::size_t n) { pos_ += n; }
  std::uint32_t ue() {  // Exp-Golomb
    int zeros = 0;
    while (bits(1) == 0) {
      if (overrun_ || ++zeros > 31) {
        overrun_ = true;
        return 0;
      }
    }
    return static_cast<std::uint32_t>((std::uint64_t{1} << zeros) - 1 +
                                      bits(zeros));
  }
  bool overrun() const noexcept { return overrun_ || pos_ > 8 * rbsp_.size(); }

 private:
  std::vector<std::uint8_t> rbsp_;
  std::size_t pos_ = 0;
  bool overrun_ = false;
};

// Whether an SPS (the NAL unit, header included) has a conformance window
// with a left or top offset (H.265 7.3.2.2). One that cannot be read counts
// as having none: FFmpeg rejects it too.
bool sps_crops_left_or_top(const std::uint8_t* nal, std::size_t size) {
  BitReader r(nal + 2, size - 2);
  r.skip(4);  // sps_video_parameter_set_id
  const std::uint32_t sub_layers = r.bits(3);
  r.skip(1);   // sps_temporal_id_nesting_flag
  r.skip(96);  // profile_tier_level: the general profile, tier and level
  bool profile[8] = {};
  bool level[8] = {};
  for (std::uint32_t i = 0; i < sub_layers; ++i) {
    profile[i] = r.bits(1) != 0;
    level[i] = r.bits(1) != 0;
  }
  if (sub_layers > 0) r.skip(2 * (8 - sub_layers));
  for (std::uint32_t i = 0; i < sub_layers; ++i) {
    if (profile[i]) r.skip(88);
    if (level[i]) r.skip(8);
  }
  r.ue();                            // sps_seq_parameter_set_id
  if (r.ue() == 3) r.skip(1);        // chroma_format_idc; separate_colour_plane
  r.ue();                            // pic_width_in_luma_samples
  r.ue();                            // pic_height_in_luma_samples
  if (r.bits(1) == 0) return false;  // conformance_window_flag
  const std::uint32_t left = r.ue();
  r.ue();  // right
  const std::uint32_t top = r.ue();
  return !r.overrun() && (left != 0 || top != 0);
}

bool is_cropping_sps(const std::uint8_t* data, std::size_t begin,
                     std::size_t end) {
  return begin + 2 <= end && ((data[begin] >> 1) & 0x3f) == 33 &&  // SPS_NUT
         sps_crops_left_or_top(data + begin, end - begin);
}

// The end of the first SPS whose prefix says it crops the left or top, or 0.
// FFmpeg must still validate the whole SPS before the stream is refused.
// Only NAL units ahead of the first slice are read: a cropping SPS must not
// reach VideoToolbox, which would decode the wrong region of the picture.
std::size_t cropping_sps_end(const std::uint8_t* data, std::size_t size) {
  std::size_t nal = size;  // where the NAL unit being scanned starts
  for (std::size_t i = 0; i + 3 <= size; ++i) {
    if (data[i] != 0 || data[i + 1] != 0 || data[i + 2] != 1) continue;
    if (is_cropping_sps(data, nal, i)) return i;
    nal = i + 3;
    if (nal < size && ((data[nal] >> 1) & 0x3f) < 32) return 0;  // VCL
    i += 2;
  }
  return is_cropping_sps(data, nal, size) ? size : 0;
}
#endif

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

core::Status unsupported(const std::string& why) {
  return core::Status::unsupported(std::string(kWho) + ": " + why);
}

}  // namespace

struct HevcDecoder::Impl {
  std::optional<VideoColorDescription> unlabelled_color;  // as in Options
  bool ended = false;
  std::string refusal;  // why the hardware gave up on the stream, if it did
  video::BufferRef hw_device;
  video::CodecContextPtr codec;
  video::PacketPtr packet;
  video::FramePtr decoded;
#if VR_SENSOR_VIDEO_WITH_CUDA
  std::unique_ptr<video::CudaPictures> pictures;
#else
  std::unique_ptr<video::VtPictures> pictures;
#endif

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

#if VR_SENSOR_VIDEO_WITH_CUDA
// Copied device to device into a buffer Vulkan reads. FFmpeg has taken the
// right and bottom crop off a hardware picture's size, and left the left and
// top to whoever reads it.
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
                         out));
  video::describe_color(frame, unlabelled_color, out);
  return out;
}
#else
// Its planes as images. VideoToolbox has cropped the right and bottom, and a
// stream cropped at the left or top is refused before it reaches it.
core::Result<DecodedPicture> HevcDecoder::Impl::picture(const AVFrame& frame) {
  if (frame.format != AV_PIX_FMT_VIDEOTOOLBOX) {
    return unsupported("VideoToolbox handed out a picture not on the GPU");
  }
  if (frame.crop_left != 0 || frame.crop_top != 0) {
    return unsupported("VideoToolbox cannot crop the left or top of a picture");
  }
  DecodedPicture out;
  VKC_TRY(pictures->import(
      reinterpret_cast<CVPixelBufferRef>(frame.data[3]),
      static_cast<std::uint32_t>(frame.width -
                                 static_cast<int>(frame.crop_right)),
      static_cast<std::uint32_t>(frame.height -
                                 static_cast<int>(frame.crop_bottom)),
      out));
  video::describe_color(frame, unlabelled_color, out);
  return out;
}
#endif

core::Result<HevcDecoder> HevcDecoder::create(const Options& options) {
  if (options.device == nullptr) {
    return unsupported("no device to decode onto (Options::device)");
  }
#if VR_SENSOR_VIDEO_WITH_CUDA
  if (options.allocator == nullptr) {
    return unsupported(
        "no allocator to make NVDEC's picture buffers through "
        "(Options::allocator)");
  }
#endif
  if (options.configure_ffmpeg_logging) av_log_set_level(AV_LOG_ERROR);
  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
  if (codec == nullptr || !has_device_path(codec)) {
    return unsupported(std::string("this FFmpeg has no ") + kPath +
                       " HEVC decoder");
  }
  auto impl = std::make_unique<Impl>();
  impl->unlabelled_color = options.unlabelled_color;
  impl->codec.reset(avcodec_alloc_context3(codec));
  impl->packet.reset(av_packet_alloc());
  impl->decoded.reset(av_frame_alloc());
  if (impl->codec == nullptr || impl->packet == nullptr ||
      impl->decoded == nullptr) {
    return video::ffmpeg_alloc_error(kWho, "opening the decoder");
  }

  // The hardware device: on CUDA, the CUDA device that is the Vulkan
  // device's GPU.
  AVBufferRef* hw = nullptr;
#if VR_SENSOR_VIDEO_WITH_CUDA
  VKC_ASSIGN(const int ordinal, video::cuda_ordinal_of(*options.device, kWho));
  const std::string name = std::to_string(ordinal);
  int err = av_hwdevice_ctx_create(&hw, kDeviceType, name.c_str(), nullptr, 0);
#else
  VKC_ASSIGN(impl->pictures, video::VtPictures::create(*options.device, kWho));
  int err = av_hwdevice_ctx_create(&hw, kDeviceType, nullptr, nullptr, 0);
#endif
  if (err < 0) {
    return unsupported(std::string(kPath) +
                       ": no device opens: " + video::ffmpeg_message(err));
  }
  impl->hw_device.reset(hw);
#if VR_SENSOR_VIDEO_WITH_CUDA
  const auto* hw_context = reinterpret_cast<const AVHWDeviceContext*>(hw->data);
  const auto* cuda = static_cast<const AVCUDADeviceContext*>(hw_context->hwctx);
  VKC_ASSIGN(impl->pictures,
             video::CudaPictures::create(*options.device, *options.allocator,
                                         cuda->cuda_ctx, cuda->stream, kWho));
#endif

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
  std::size_t crop_end = 0;
#if !VR_SENSOR_VIDEO_WITH_CUDA
  crop_end = cropping_sps_end(data, size);
#endif
  // Submit only through the suspect SPS: no slices can reach VideoToolbox
  // before the crop is refused, and no later NAL can obscure whether this
  // SPS parsed. Its preceding VPS is needed for that validation.
  const std::size_t submitted = crop_end != 0 ? crop_end : size;
  AVPacket* packet = impl_->packet.get();
  av_packet_unref(packet);
  int err = av_new_packet(packet, static_cast<int>(submitted));
  if (err < 0) return video::ffmpeg_error(kWho, "allocating a packet", err);
  std::memcpy(packet->data, data, submitted);
  packet->pts = pts;
  // FFmpeg normally logs and skips a malformed parameter set. Require its
  // error here so a truncated SPS cannot become a permanent crop refusal.
  const int recognition = context->err_recognition;
  if (crop_end != 0) context->err_recognition |= AV_EF_EXPLODE;
  err = avcodec_send_packet(context, packet);
  context->err_recognition = recognition;
  if (!impl_->refusal.empty()) return core::Status::unsupported(impl_->refusal);
  if (err == AVERROR(EAGAIN)) {
    return core::Status::invalid_argument(
        std::string(kWho) + ": take the pictures waiting before sending more");
  }
  if (err < 0) return video::ffmpeg_error(kWho, "decoding", err);
  if (crop_end != 0) {
    impl_->refusal = std::string(kWho) +
                     ": VideoToolbox cannot crop the left or top of a picture";
    return core::Status::unsupported(impl_->refusal);
  }
  return {};
}

core::Result<std::optional<DecodedPicture>> HevcDecoder::receive() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": moved from");
  }
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
}

core::Status HevcDecoder::reset() {
  if (impl_ == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) + ": moved from");
  }
  // As for a seek: FFmpeg drops what it holds, leaves draining, and takes the
  // next key frame as the first, skipping the pictures that lead it.
  avcodec_flush_buffers(impl_->codec.get());
  impl_->ended = false;
  return {};
}

}  // namespace volumetric_kit::recon::sensor
