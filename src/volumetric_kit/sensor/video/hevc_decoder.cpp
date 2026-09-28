// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <utility>

#include "ffmpeg.hpp"
#include "hw_backend.hpp"
#include "picture_converter.hpp"

namespace volumetric_kit::recon::sensor {
namespace {

constexpr const char* kWho = "HevcDecoder";

constexpr std::uint8_t kProbeClip[] = {
#include "probe_clip.inc"
};

std::string backend_list(const std::vector<VideoDecodeBackend>& backends) {
  std::string list = "software";
  for (const VideoDecodeBackend b : backends) {
    list += ", ";
    list += to_string(b);
  }
  return list;
}

}  // namespace

struct HevcDecoder::Impl {
  VideoDecodeBackend backend = VideoDecodeBackend::Software;
  bool may_fall_back = false;  // Auto: software if the hardware refuses
  AVPixelFormat hw_format = AV_PIX_FMT_NONE;
  VideoPixelLayout layout = VideoPixelLayout::Rgb24;
  bool ended = false;
  video::BufferRef device;
  video::CodecContextPtr codec;
  video::PacketPtr packet;
  video::FramePtr decoded;      // as the codec hands it out, maybe on a GPU
  video::FramePtr transferred;  // a hardware frame copied to the host
  video::PictureConverter converter;

  static Result<std::unique_ptr<Impl>> open(VideoDecodeBackend backend,
                                            bool may_fall_back,
                                            VideoPixelLayout layout,
                                            int threads);

  // FFmpeg's get_format: the hardware format while it is offered. When it is
  // not (the hardware cannot decode this stream), a named back end fails and
  // Auto moves to software.
  static AVPixelFormat pick_format(AVCodecContext* context,
                                   const AVPixelFormat* formats) {
    auto* impl = static_cast<Impl*>(context->opaque);
    for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
      if (*f == impl->hw_format) return *f;
    }
    if (!impl->may_fall_back) return AV_PIX_FMT_NONE;
    for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
      const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(*f);
      if (desc != nullptr && (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) == 0) {
        impl->backend = VideoDecodeBackend::Software;
        impl->hw_format = AV_PIX_FMT_NONE;
        return *f;
      }
    }
    return AV_PIX_FMT_NONE;
  }
};

Result<std::unique_ptr<HevcDecoder::Impl>> HevcDecoder::Impl::open(
    VideoDecodeBackend backend, bool may_fall_back, VideoPixelLayout layout,
    int threads) {
  const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
  if (codec == nullptr) {
    return Status::unsupported(std::string(kWho) +
                               ": this FFmpeg has no HEVC decoder");
  }
  auto impl = std::make_unique<Impl>();
  impl->backend = backend;
  impl->may_fall_back = may_fall_back;
  impl->layout = layout;
  impl->codec.reset(avcodec_alloc_context3(codec));
  impl->packet.reset(av_packet_alloc());
  impl->decoded.reset(av_frame_alloc());
  impl->transferred.reset(av_frame_alloc());
  if (impl->codec == nullptr || impl->packet == nullptr ||
      impl->decoded == nullptr || impl->transferred == nullptr) {
    return video::ffmpeg_alloc_error(kWho, "opening the decoder");
  }

  AVCodecContext* context = impl->codec.get();
  if (backend == VideoDecodeBackend::Software) {
    context->thread_count = threads;
  } else {
    impl->hw_format =
        video::hardware_pixel_format(codec, video::device_type(backend));
    if (impl->hw_format == AV_PIX_FMT_NONE) {
      return Status::unsupported(std::string(kWho) + ": this FFmpeg has no " +
                                 to_string(backend) + " HEVC decoder");
    }
    VR_ASSIGN(impl->device, video::open_hardware_device(backend));
    context->hw_device_ctx = av_buffer_ref(impl->device.get());
    if (context->hw_device_ctx == nullptr) {
      return video::ffmpeg_alloc_error(kWho, "sharing the device");
    }
    context->opaque = impl.get();
    context->get_format = &Impl::pick_format;
    context->thread_count = 1;
  }
  const int err = avcodec_open2(context, codec, nullptr);
  if (err < 0) return video::ffmpeg_error(kWho, "opening the decoder", err);
  return impl;
}

std::vector<VideoDecodeBackend> HevcDecoder::hardware_backends() {
  static const std::vector<VideoDecodeBackend> kDecoding = [] {
    std::vector<VideoDecodeBackend> found;
    for (const VideoDecodeBackend backend : video::platform_hardware_order()) {
      auto opened = Impl::open(backend, /*may_fall_back=*/false,
                               VideoPixelLayout::Yuv420, 1);
      if (!opened) continue;
      HevcDecoder decoder(std::move(opened).value());
      bool decoded = decoder.send(kProbeClip, sizeof(kProbeClip), 0).ok() &&
                     decoder.send(nullptr, 0, 0).ok();
      bool pictured = false;
      while (decoded) {
        auto picture = decoder.receive();
        if (!picture)
          decoded = false;
        else if (!picture.value())
          break;
        else
          pictured = true;
      }
      if (decoded && pictured) found.push_back(backend);
    }
    return found;
  }();
  return kDecoding;
}

Result<HevcDecoder> HevcDecoder::create(const Options& options) {
  if (options.threads < 0) {
    return Status::invalid_argument(std::string(kWho) +
                                    ": threads must be 0 or more");
  }
  if (options.configure_ffmpeg_logging) av_log_set_level(AV_LOG_ERROR);

  VideoDecodeBackend backend = options.backend;
  if (backend != VideoDecodeBackend::Software) {
    const std::vector<VideoDecodeBackend> hardware = hardware_backends();
    if (backend == VideoDecodeBackend::Auto) {
      if (!hardware.empty()) {
        auto opened = Impl::open(hardware.front(), /*may_fall_back=*/true,
                                 options.layout, options.threads);
        if (opened) return HevcDecoder(std::move(opened).value());
      }
      backend = VideoDecodeBackend::Software;
    } else if (std::find(hardware.begin(), hardware.end(), backend) ==
               hardware.end()) {
      return Status::unsupported(
          std::string(kWho) + ": " + to_string(backend) +
          " does not decode HEVC here; available: " + backend_list(hardware));
    }
  }
  VR_ASSIGN(auto impl, Impl::open(backend, /*may_fall_back=*/false,
                                  options.layout, options.threads));
  return HevcDecoder(std::move(impl));
}

HevcDecoder::HevcDecoder(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
HevcDecoder::HevcDecoder(HevcDecoder&& other) noexcept = default;
HevcDecoder& HevcDecoder::operator=(HevcDecoder&& other) noexcept = default;
HevcDecoder::~HevcDecoder() = default;

VideoDecodeBackend HevcDecoder::backend() const noexcept {
  return impl_ != nullptr ? impl_->backend : VideoDecodeBackend::Auto;
}

Status HevcDecoder::send(const std::uint8_t* data, std::size_t size,
                         std::int64_t pts) {
  if (impl_ == nullptr) {
    return Status::invalid_argument(std::string(kWho) + ": moved from");
  }
  if (impl_->ended) {
    return Status::invalid_argument(std::string(kWho) +
                                    ": data sent after the end of the stream");
  }
  AVCodecContext* context = impl_->codec.get();
  if (size == 0) {
    impl_->ended = true;
    const int err = avcodec_send_packet(context, nullptr);
    if (err < 0) return video::ffmpeg_error(kWho, "ending the stream", err);
    return {};
  }
  if (data == nullptr || size > static_cast<std::size_t>(INT_MAX)) {
    return Status::invalid_argument(
        std::string(kWho) + ": an access unit needs data and under 2 GiB");
  }
  AVPacket* packet = impl_->packet.get();
  av_packet_unref(packet);
  int err = av_new_packet(packet, static_cast<int>(size));
  if (err < 0) return video::ffmpeg_error(kWho, "allocating a packet", err);
  std::memcpy(packet->data, data, size);
  packet->pts = pts;
  err = avcodec_send_packet(context, packet);
  if (err == AVERROR(EAGAIN)) {
    return Status::invalid_argument(
        std::string(kWho) + ": take the pictures waiting before sending more");
  }
  if (err < 0) return video::ffmpeg_error(kWho, "decoding", err);
  return {};
}

Result<std::optional<DecodedPicture>> HevcDecoder::receive() {
  if (impl_ == nullptr) {
    return Status::invalid_argument(std::string(kWho) + ": moved from");
  }
  AVFrame* decoded = impl_->decoded.get();
  AVFrame* transferred = impl_->transferred.get();
  av_frame_unref(decoded);
  av_frame_unref(transferred);
  int err = avcodec_receive_frame(impl_->codec.get(), decoded);
  if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
    return std::optional<DecodedPicture>();
  }
  if (err < 0) return video::ffmpeg_error(kWho, "decoding", err);

  const AVFrame* host = decoded;
  if (impl_->hw_format != AV_PIX_FMT_NONE &&
      decoded->format == impl_->hw_format) {
    // TODO(sensor): hand a CUDA or VideoToolbox frame to the GPU module
    // without this copy through host memory.
    err = av_hwframe_transfer_data(transferred, decoded, 0);
    if (err >= 0) err = av_frame_copy_props(transferred, decoded);
    if (err < 0) {
      return video::ffmpeg_error(kWho, "copying the picture to the host", err);
    }
    host = transferred;
  }
  VR_ASSIGN(DecodedPicture picture,
            impl_->converter.convert(*host, impl_->layout));
  picture.pts = decoded->pts != AV_NOPTS_VALUE ? decoded->pts
                                               : decoded->best_effort_timestamp;
  return std::optional<DecodedPicture>(picture);
}

}  // namespace volumetric_kit::recon::sensor
