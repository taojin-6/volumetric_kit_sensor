// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/video/hevc_decoder.hpp"

#include <climits>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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
// as having one, so hardware that cannot crop stays off a stream nothing
// vouches for.
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
  r.ue();                      // sps_seq_parameter_set_id
  if (r.ue() == 3) r.skip(1);  // chroma_format_idc; separate_colour_plane
  r.ue();                      // pic_width_in_luma_samples
  r.ue();                      // pic_height_in_luma_samples
  if (r.bits(1) == 0) return r.overrun();  // conformance_window_flag
  const std::uint32_t left = r.ue();
  r.ue();  // right
  const std::uint32_t top = r.ue();
  return r.overrun() || left != 0 || top != 0;
}

bool is_cropping_sps(const std::uint8_t* data, std::size_t begin,
                     std::size_t end) {
  return begin + 2 <= end && ((data[begin] >> 1) & 0x3f) == 33 &&  // SPS_NUT
         sps_crops_left_or_top(data + begin, end - begin);
}

// Whether an Annex B access unit carries an SPS that crops the left or top.
// Only the NAL units ahead of its first slice are read: an SPS is sent before
// the slices that activate it, and the slices are nearly all of the bytes.
bool crops_left_or_top(const std::uint8_t* data, std::size_t size) {
  std::size_t nal = size;  // where the NAL unit being scanned starts
  for (std::size_t i = 0; i + 3 <= size; ++i) {
    if (data[i] != 0 || data[i + 1] != 0 || data[i + 2] != 1) continue;
    if (is_cropping_sps(data, nal, i)) return true;
    nal = i + 3;
    if (nal < size && ((data[nal] >> 1) & 0x3f) < 32) return false;  // VCL
    i += 2;
  }
  return is_cropping_sps(data, nal, size);
}

}  // namespace

struct HevcDecoder::Impl {
  VideoDecodeBackend backend = VideoDecodeBackend::Software;
  bool may_fall_back = false;  // Auto: software if the hardware refuses
  AVPixelFormat hw_format = AV_PIX_FMT_NONE;
  VideoPixelLayout layout = VideoPixelLayout::Rgb24;
  bool ended = false;
  bool left_top_crop = false;  // an SPS so far crops the left or top
  std::string refusal;         // why the named back end gave up, if it did
  video::BufferRef device;
  video::CodecContextPtr codec;
  video::PacketPtr packet;
  video::FramePtr decoded;      // as the codec hands it out, maybe on a GPU
  video::FramePtr transferred;  // a hardware frame copied to the host
  video::PictureConverter converter{kWho};

  static Result<std::unique_ptr<Impl>> open(VideoDecodeBackend backend,
                                            bool may_fall_back,
                                            VideoPixelLayout layout,
                                            int threads);

  // Whether @p backend decodes HEVC here, found once per process: asked of
  // the platform where it can be, else by decoding the probe clip.
  static bool decodes(VideoDecodeBackend backend);

  Status copy_to_host(const AVFrame& picture);

  // FFmpeg's get_format, asked at each new SPS: the hardware format while it
  // is offered and the stream is one the hardware can crop. When not, a named
  // back end fails and Auto moves to software.
  static AVPixelFormat pick_format(AVCodecContext* context,
                                   const AVPixelFormat* formats) {
    auto* impl = static_cast<Impl*>(context->opaque);
    const bool crops =
        !impl->left_top_crop || video::crops_left_and_top(impl->backend);
    for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
      if (crops && *f == impl->hw_format) return *f;
    }
    if (!impl->may_fall_back) {
      impl->refusal = std::string(kWho) + ": " + to_string(impl->backend) +
                      (crops ? " cannot decode this stream"
                             : " cannot crop the left or top of a picture");
      return AV_PIX_FMT_NONE;
    }
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
  // Crop the left edge exactly: FFmpeg otherwise keeps up to 64 columns of
  // it to keep the planes aligned.
  context->flags |= AV_CODEC_FLAG_UNALIGNED;
  context->thread_count = threads;
  if (backend != VideoDecodeBackend::Software) {
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
    // Slice threads only: frame threads would hold pictures back on the
    // hardware too. The hardware ignores them; they are for a stream Auto
    // moves to software.
    context->thread_type = FF_THREAD_SLICE;
  }
  const int err = avcodec_open2(context, codec, nullptr);
  if (err < 0) return video::ffmpeg_error(kWho, "opening the decoder", err);
  return impl;
}

bool HevcDecoder::Impl::decodes(VideoDecodeBackend backend) {
  static std::mutex mutex;
  static std::map<VideoDecodeBackend, bool> known;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto found = known.find(backend);
  if (found != known.end()) return found->second;

  // A back end that is not there says so at ERROR ("Cannot load
  // libcuda.so.1"); here that is the answer, not a fault.
  const int level = av_log_get_level();
  av_log_set_level(AV_LOG_QUIET);
  bool answer = false;
  auto opened =
      open(backend, /*may_fall_back=*/false, VideoPixelLayout::Yuv420, 1);
  if (opened) {
    // Opening answers for this FFmpeg. Where the platform can be asked, it
    // answers for the hardware; elsewhere only decoding the clip can.
    if (const auto asked = video::hardware_decodes(backend, AV_CODEC_ID_HEVC)) {
      answer = *asked;
    } else {
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
      answer = pictured && decoded;
    }
  }
  av_log_set_level(level);
  known.emplace(backend, answer);
  return answer;
}

// FFmpeg crops a hardware picture at the right and bottom only, so the copy
// is cropped here. It is allocated afresh each time, since keeping one buffer
// measured no faster (the 2026-09-27 decoder decision).
Status HevcDecoder::Impl::copy_to_host(const AVFrame& picture) {
  AVFrame* copy = transferred.get();
  int err = av_hwframe_transfer_data(copy, &picture, 0);
  if (err >= 0) err = av_frame_copy_props(copy, &picture);
  if (err >= 0) err = av_frame_apply_cropping(copy, AV_FRAME_CROP_UNALIGNED);
  if (err < 0) {
    return video::ffmpeg_error(kWho, "copying the picture to the host", err);
  }
  return {};
}

std::vector<VideoDecodeBackend> HevcDecoder::hardware_backends() {
  std::vector<VideoDecodeBackend> found;
  for (const VideoDecodeBackend backend : video::platform_hardware_order()) {
    if (Impl::decodes(backend)) found.push_back(backend);
  }
  return found;
}

Result<HevcDecoder> HevcDecoder::create(const Options& options) {
  if (options.threads < 0) {
    return Status::invalid_argument(std::string(kWho) +
                                    ": threads must be 0 or more");
  }
  if (options.configure_ffmpeg_logging) av_log_set_level(AV_LOG_ERROR);

  VideoDecodeBackend backend = options.backend;
  if (backend == VideoDecodeBackend::Auto) {
    // In order, stopping at the first that decodes and opens, so the back
    // ends behind it are never probed.
    for (const VideoDecodeBackend b : video::platform_hardware_order()) {
      if (!Impl::decodes(b)) continue;
      auto opened = Impl::open(b, /*may_fall_back=*/true, options.layout,
                               options.threads);
      if (opened) return HevcDecoder(std::move(opened).value());
    }
    backend = VideoDecodeBackend::Software;
  } else if (backend != VideoDecodeBackend::Software &&
             !Impl::decodes(backend)) {
    return Status::unsupported(std::string(kWho) + ": " + to_string(backend) +
                               " does not decode HEVC here; available: " +
                               backend_list(hardware_backends()));
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
  if (!impl_->refusal.empty()) return Status::unsupported(impl_->refusal);
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
  // Read before FFmpeg activates the SPS, which is when pick_format asks.
  if (impl_->backend != VideoDecodeBackend::Software &&
      !video::crops_left_and_top(impl_->backend) && !impl_->left_top_crop) {
    impl_->left_top_crop = crops_left_or_top(data, size);
  }
  AVPacket* packet = impl_->packet.get();
  av_packet_unref(packet);
  int err = av_new_packet(packet, static_cast<int>(size));
  if (err < 0) return video::ffmpeg_error(kWho, "allocating a packet", err);
  std::memcpy(packet->data, data, size);
  packet->pts = pts;
  err = avcodec_send_packet(context, packet);
  if (!impl_->refusal.empty()) return Status::unsupported(impl_->refusal);
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
  av_frame_unref(decoded);
  av_frame_unref(impl_->transferred.get());
  const int err = avcodec_receive_frame(impl_->codec.get(), decoded);
  if (err < 0) {
    // Pictures decoded before a refusal still come out; then the refusal.
    if (!impl_->refusal.empty()) return Status::unsupported(impl_->refusal);
    if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) {
      return std::optional<DecodedPicture>();
    }
    return video::ffmpeg_error(kWho, "decoding", err);
  }

  // Asked of the picture, not the back end: pictures decoded on the GPU are
  // still waiting after Auto moves the stream to software.
  const AVFrame* host = decoded;
  if (decoded->hw_frames_ctx != nullptr) {
    // TODO(sensor): hand a CUDA or VideoToolbox frame to the GPU module
    // without this copy through host memory.
    VR_TRY(impl_->copy_to_host(*decoded));
    host = impl_->transferred.get();
  }
  VR_ASSIGN(DecodedPicture picture,
            impl_->converter.convert(*host, impl_->layout));
  picture.pts = decoded->pts != AV_NOPTS_VALUE ? decoded->pts
                                               : decoded->best_effort_timestamp;
  return std::optional<DecodedPicture>(picture);
}

}  // namespace volumetric_kit::recon::sensor
