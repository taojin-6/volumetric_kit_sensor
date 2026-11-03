// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HevcEncoder's NVIDIA backend (hevc_encode_session.hpp): NVENC through
// FFmpeg's hevc_nvenc, reading targets the device writes. A target is one of
// kTargets exported Vulkan buffers, each mapped into CUDA once, which FFmpeg
// registers with NVENC once, by its pointer: a picture crosses to NVENC with
// no copy. The session codes on HevcEncoder's coding thread, which has waited
// on the host for the picture's writer: NVENC on Linux codes synchronously,
// and with `delay` 0 FFmpeg's send waits for the whole picture.

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cuda_driver.hpp"
#include "cuda_pictures.hpp"
#include "ffmpeg.hpp"
#include "hevc_encode_session.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
}

namespace volumetric_kit::sensor::video {
namespace {

constexpr const char* kWho = "HevcEncoder";
constexpr const char* kPath = "NVENC";

// Targets a session holds: one coding, one the device writes, and one more
// for a target taken while the one before is still held (HevcEncoder::target
// keeps it until the new one is handed out).
constexpr std::size_t kTargets = HevcEncoder::kInFlight + 1;

// From this many rows a picture is split into 3 horizontal strips across the
// GPU's encoders, where it has them; below, the driver chooses.
constexpr std::uint32_t kStripRows = 4096;

std::string size_text(std::uint32_t width, std::uint32_t height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

core::Status unsupported(const std::string& why) {
  return core::Status::unsupported(std::string(kWho) + ": " + why);
}

core::Status io_error(const std::string& why) {
  return core::Status::io_error(std::string(kWho) + ": " + why);
}

// What an open that failed says. NVENC refuses a session past its limit, a
// busy engine, memory it lacks and a parameter this GPU does not take alike,
// with EINVAL or ENOMEM.
core::Status open_failure(const std::string& what, int err) {
  if (err == AVERROR(EINVAL) || err == AVERROR(ENOMEM)) {
    return unsupported(
        "NVENC is busy, at its session limit, out of memory, or does not take "
        "a parameter of " +
        what + " on this GPU: it did not open (" + ffmpeg_message(err) +
        "); try again once another session closes");
  }
  if (err == AVERROR(ENOSYS) || err == AVERROR(ENOENT) || err == -1) {
    return unsupported(
        "no hardware NVENC here opens " + what +
        " (NVIDIA's driver with its encode library, on a GPU with NVENC): " +
        ffmpeg_message(err));
  }
  return ffmpeg_error(kWho, "opening " + what, err);
}

// The rows of a target, in bytes: the width rounded up to the GPU's texture
// alignment, as FFmpeg pads its own CUDA frames, whose pitch NVENC registers.
core::Result<std::uint32_t> target_stride(const core::Device& device,
                                          std::uint32_t width) {
  VKC_ASSIGN(const int ordinal, cuda_ordinal_of(device));
  const CudaDriver* cu = cuda_driver();
  CUdevice gpu = 0;
  CUresult r = cu->cuDeviceGet(&gpu, ordinal);
  if (r != CUDA_SUCCESS) return cuda_error(kWho, r, "finding the GPU");
  int alignment = 0;
  r = cu->cuDeviceGetAttribute(&alignment,
                               CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, gpu);
  if (r != CUDA_SUCCESS) {
    return cuda_error(kWho, r, "reading the GPU's texture alignment");
  }
  if (alignment <= 0) return io_error("the GPU states no texture alignment");
  return static_cast<std::uint32_t>(round_up(width, std::uint64_t(alignment)));
}

// hevc_nvenc's options for @p o: one configuration, every QP set from the
// options (the P-picture's, the B-picture's, which no picture uses, and the
// IDR's), so FFmpeg chooses no QP of its own.
std::vector<std::pair<const char*, std::string>> codec_options(
    const HevcEncoder::Options& o, bool strips) {
  const std::string qp = std::to_string(o.qp);
  std::vector<std::pair<const char*, std::string>> options = {
      {"preset", "p" + std::to_string(o.effort)},
      {"tune", "ll"},
      {"profile", "main"},
      {"rc", "constqp"},
      {"init_qpP", qp},
      {"init_qpB", qp},
      {"init_qpI", std::to_string(o.key_qp.value_or(o.qp))},
      {"forced-idr", "1"},
      {"no-scenecut", "1"},
      {"rc-lookahead", "0"},
      {"delay", "0"},
      {"zerolatency", "1"},
      {"multipass", "disabled"},
  };
  if (strips) options.emplace_back("split_encode_mode", "3");
  return options;
}

}  // namespace

const bool HevcEncodeSession::kWritesSiting = false;

// One target: an exported buffer and CUDA's view of it, which goes first.
struct NvencTarget {
  NvencTarget() = default;
  NvencTarget(const NvencTarget&) = delete;
  NvencTarget& operator=(const NvencTarget&) = delete;

  // FFmpeg's frame of it let go: NVENC has read it.
  static void release(void* self, std::uint8_t* /*data*/) noexcept {
    static_cast<NvencTarget*>(self)->in_ffmpeg.store(
        false, std::memory_order_release);
  }

  std::shared_ptr<core::Buffer> buffer;  // a target's YuvImage::device
  std::unique_ptr<CudaBufferMapping> mapping;
  std::uint64_t handed = 0;  // the hand-out that took it last
  std::atomic<bool> in_ffmpeg{false};
};

namespace {

// A session's targets and FFmpeg's CUDA device and frames, which pass from a
// lost session to the one that replaces it: a batch may still be writing a
// target. The targets go first, CUDA's context, which FFmpeg's device holds,
// last.
struct NvencTargets {
  static core::Result<std::unique_ptr<NvencTargets>> create(
      const core::Device& device, core::Allocator& allocator,
      std::uint32_t width, std::uint32_t height);

  FfmpegCudaDevice cuda;
  BufferRef frames;  // AV_PIX_FMT_CUDA frames of NV12 at the session's size
  std::uint32_t stride = 0;
  std::uint64_t bytes = 0;
  std::array<std::shared_ptr<NvencTarget>, kTargets> slots;
  std::uint64_t handed = 0;  // hand-outs so far
};

core::Result<std::unique_ptr<NvencTargets>> NvencTargets::create(
    const core::Device& device, core::Allocator& allocator,
    std::uint32_t width, std::uint32_t height) {
  if (!device.exports_memory()) {
    return unsupported(
        "the device exports no memory (VK_KHR_external_memory_fd), through "
        "which NVENC reads its targets");
  }
  auto t = std::make_unique<NvencTargets>();
  VKC_ASSIGN(t->cuda, FfmpegCudaDevice::open(device, kWho, kPath));
  VKC_ASSIGN(t->stride, target_stride(device, width));
  t->bytes = std::uint64_t{t->stride} * height * 3 / 2;
  t->frames.reset(av_hwframe_ctx_alloc(t->cuda.device.get()));
  if (t->frames == nullptr) {
    return ffmpeg_alloc_error(kWho, "making NVENC's frames");
  }
  auto* frames = reinterpret_cast<AVHWFramesContext*>(t->frames->data);
  frames->format = AV_PIX_FMT_CUDA;
  frames->sw_format = AV_PIX_FMT_NV12;
  frames->width = static_cast<int>(width);
  frames->height = static_cast<int>(height);
  frames->initial_pool_size = 0;  // its frames are the targets, not a pool's
  const int err = av_hwframe_ctx_init(t->frames.get());
  if (err < 0) return ffmpeg_error(kWho, "making NVENC's frames", err);
  const CudaContext& context = *t->cuda.cuda->context;
  for (std::shared_ptr<NvencTarget>& slot : t->slots) {
    VKC_ASSIGN(core::ExportedBuffer exported,
               core::create_exported_buffer(device, allocator, t->bytes));
    device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                           core::debug_object_handle(exported.buffer.handle()),
                           "sensor.encoder_target");
    // Each owner held from the moment it exists, CUDA's view after the
    // buffer, so a failure lets the view go first.
    core::Result<std::unique_ptr<CudaBufferMapping>> mapping =
        CudaBufferMapping::map(context, exported);
    if (!mapping) return mapping.status().with_context(kWho);
    slot = std::make_shared<NvencTarget>();
    slot->buffer = std::make_shared<core::Buffer>(std::move(exported.buffer));
    slot->mapping = std::move(mapping).value();
  }
  return t;
}

}  // namespace

struct HevcEncodeSession::Impl {
  HevcEncoder::Options options;
  std::unique_ptr<NvencTargets> targets;
  FramePtr frame;
  PacketPtr packet;
  // Last, so it goes first, before the frames and device it refers to; null
  // once lost, or closed by the session that replaces it.
  CodecContextPtr codec;
  std::int64_t pts = 0;
};

HevcEncodeSession::HevcEncodeSession(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
HevcEncodeSession::~HevcEncodeSession() = default;

core::Result<std::unique_ptr<HevcEncodeSession>> HevcEncodeSession::open(
    const HevcEncoder::Options& options, HevcEncoder::Facts& facts,
    HevcEncodeSession* previous) try {
  if (options.device == nullptr || options.allocator == nullptr) {
    return core::Status::invalid_argument(
        std::string(kWho) +
        ": NVENC codes the targets a device writes: it needs "
        "Options::device and Options::allocator");
  }
  const std::string what = "a low-latency hevc_nvenc session of " +
                           size_text(options.width, options.height);
  quiet_ffmpeg_log();
  const AVCodec* nvenc = avcodec_find_encoder_by_name("hevc_nvenc");
  if (nvenc == nullptr) return unsupported("this FFmpeg has no hevc_nvenc");

  auto impl = std::make_unique<Impl>();
  Impl& s = *impl;
  s.options = options;
  s.frame.reset(av_frame_alloc());
  s.packet.reset(av_packet_alloc());
  s.codec.reset(avcodec_alloc_context3(nvenc));
  if (s.frame == nullptr || s.packet == nullptr || s.codec == nullptr) {
    return ffmpeg_alloc_error(kWho, "opening NVENC");
  }
  const bool strips = options.height >= kStripRows;
  if (strips && av_opt_find(s.codec->priv_data, "split_encode_mode", nullptr,
                            0, 0) == nullptr) {
    return unsupported(
        "a picture of " + std::to_string(kStripRows) +
        " rows or more is coded in strips, which this FFmpeg cannot ask "
        "NVENC for: FFmpeg 7.1 or later, built with nv-codec-headers 12.1 or "
        "later, has them");
  }

  // A lost session's hardware closes before this one opens; its targets
  // pass to this one only once it has.
  Impl* before = previous != nullptr ? previous->impl_.get() : nullptr;
  if (before != nullptr) before->codec.reset();
  std::unique_ptr<NvencTargets> made;
  NvencTargets* targets =
      before != nullptr ? before->targets.get() : nullptr;
  if (targets == nullptr) {
    VKC_ASSIGN(made, NvencTargets::create(*options.device, *options.allocator,
                                          options.width, options.height));
    targets = made.get();
  }

  AVCodecContext* codec = s.codec.get();
  codec->width = static_cast<int>(options.width);
  codec->height = static_cast<int>(options.height);
  codec->pix_fmt = AV_PIX_FMT_CUDA;
  codec->sw_pix_fmt = AV_PIX_FMT_NV12;
  codec->hw_frames_ctx = av_buffer_ref(targets->frames.get());
  if (codec->hw_frames_ctx == nullptr) {
    return ffmpeg_alloc_error(kWho, "opening NVENC");
  }
  codec->time_base = AVRational{1, 30};
  codec->framerate = AVRational{30, 1};
  codec->max_b_frames = 0;
  codec->gop_size = INT_MAX;  // keys only when asked
  set_color(*codec, options.color);
  const auto applied = codec_options(options, strips);
  for (const auto& [name, value] : applied) {
    const int err = av_opt_set(codec->priv_data, name, value.c_str(), 0);
    if (err < 0) {
      return ffmpeg_error(kWho,
                          std::string("setting hevc_nvenc's ") + name +
                              " to " + value,
                          err);
    }
  }
  const int err = avcodec_open2(codec, nvenc, nullptr);
  if (err < 0) return open_failure(what, err);

  HevcEncoder::Facts read;
  read.encoder = std::string("hevc_nvenc on ") +
                 options.device->caps().properties().deviceName;
  read.color = options.color;
  read.effort = options.effort;
  read.strips = strips ? 3 : 0;
  read.settings = "hevc_nvenc " + size_text(options.width, options.height) +
                  ":";
  for (const auto& [name, value] : applied) {
    read.settings.append(" ").append(name).append("=").append(value);
  }
  read.settings.append(" max_b_frames=")
      .append(std::to_string(codec->max_b_frames))
      .append(" gop_size=")
      .append(std::to_string(codec->gop_size))
      .append(" colorspace=")
      .append(std::to_string(codec->colorspace))
      .append(" color_trc=")
      .append(std::to_string(codec->color_trc))
      .append(" color_primaries=")
      .append(std::to_string(codec->color_primaries))
      .append(codec->color_range == AVCOL_RANGE_JPEG ? " full range"
                                                     : " video range");
  s.targets = before != nullptr ? std::move(before->targets) : std::move(made);
  facts = std::move(read);
  return std::unique_ptr<HevcEncodeSession>(
      new HevcEncodeSession(std::move(impl)));
} catch (...) {
  return core::exception_status("HevcEncoder::create",
                                core::Status::Code::OutOfMemory);
}

core::Result<bool> HevcEncodeSession::encode(
    EncodePicture& picture, bool key, bool drop,
    std::vector<std::uint8_t>& access_unit, bool& idr) noexcept try {
  Impl& s = *impl_;
  access_unit.clear();
  idr = false;
  if (s.codec == nullptr) return io_error("the NVENC session is lost");
  if (picture.target == nullptr) {
    return core::Status::invalid_argument(std::string(kWho) +
                                          ": NVENC codes targets only");
  }
  NvencTarget& t = *picture.target;
  const NvencTargets& all = *s.targets;
  AVFrame* frame = s.frame.get();
  av_frame_unref(frame);
  auto* const data = reinterpret_cast<std::uint8_t*>(t.mapping->pointer());
  frame->buf[0] = av_buffer_create(data, all.bytes, &NvencTarget::release, &t,
                                   AV_BUFFER_FLAG_READONLY);
  if (frame->buf[0] == nullptr) {
    return ffmpeg_alloc_error(kWho, "a picture's frame");
  }
  t.in_ffmpeg.store(true, std::memory_order_relaxed);
  frame->hw_frames_ctx = av_buffer_ref(all.frames.get());
  if (frame->hw_frames_ctx == nullptr) {
    av_frame_unref(frame);
    return ffmpeg_alloc_error(kWho, "a picture's frame");
  }
  const std::uint64_t luma = std::uint64_t{all.stride} * s.options.height;
  frame->data[0] = data;
  frame->data[1] = data + luma;
  frame->linesize[0] = static_cast<int>(all.stride);
  frame->linesize[1] = static_cast<int>(all.stride);
  frame->format = AV_PIX_FMT_CUDA;
  frame->width = static_cast<int>(s.options.width);
  frame->height = static_cast<int>(s.options.height);
  frame->pts = s.pts++;
  frame->pict_type = key ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
  int err = avcodec_send_frame(s.codec.get(), frame);
  av_frame_unref(frame);
  if (err < 0) return ffmpeg_error(kWho, "sending a picture", err);
  AVPacket* packet = s.packet.get();
  err = avcodec_receive_packet(s.codec.get(), packet);
  if (err == AVERROR(EAGAIN)) return io_error("NVENC gave no packet");
  if (err < 0) return ffmpeg_error(kWho, "receiving a picture", err);
  idr = (packet->flags & AV_PKT_FLAG_KEY) != 0;
  if (!drop) access_unit.assign(packet->data, packet->data + packet->size);
  av_packet_unref(packet);
  err = avcodec_receive_packet(s.codec.get(), packet);
  if (err != AVERROR(EAGAIN)) {
    av_packet_unref(packet);
    access_unit.clear();
    return err < 0 ? ffmpeg_error(kWho, "receiving a picture", err)
                   : io_error("NVENC gave two access units for one picture");
  }
  if (t.in_ffmpeg.load(std::memory_order_acquire)) {
    access_unit.clear();
    return io_error("NVENC still holds the picture after its access unit");
  }
  if (key && !idr) {
    access_unit.clear();
    return io_error("asked for an IDR, NVENC coded a P-picture");
  }
  if (drop) return false;
  if (access_unit.empty()) return io_error("NVENC gave an empty access unit");
  return true;
} catch (...) {
  return core::exception_status("HevcEncoder: coding a picture",
                                core::Status::Code::OutOfMemory);
}

core::Status HevcEncodeSession::target(core::CommandBatch& batch,
                                       EncodePicture& picture, YuvImage& out,
                                       bool& gone) try {
  Impl& s = *impl_;
  if (s.codec == nullptr) {
    gone = true;
    return io_error("the NVENC session is lost");
  }
  NvencTargets& all = *s.targets;
  // The free target handed out longest ago, so one given up comes back only
  // after the others. Free: no picture holds it, as the coding thread lets
  // go of one once NVENC has read it; the fence orders this batch's writes
  // after that read.
  std::shared_ptr<NvencTarget>* pick = nullptr;
  for (std::shared_ptr<NvencTarget>& slot : all.slots) {
    if (slot.use_count() == 1 &&
        (pick == nullptr || slot->handed < (*pick)->handed)) {
      pick = &slot;
    }
  }
  if (pick == nullptr) return io_error("every target is held");
  std::atomic_thread_fence(std::memory_order_acquire);
  const core::Buffer& buffer = *(*pick)->buffer;
  // Vulkan writes, then NVENC reads through CUDA: the batch takes the buffer
  // over from outside Vulkan before its writes and hands it back after them
  // (video/cuda_interop.hpp).
  VKC_TRY(batch.acquire(buffer, VK_QUEUE_FAMILY_EXTERNAL));
  VKC_TRY(batch.release(buffer, VK_QUEUE_FAMILY_EXTERNAL));
  YuvImage planes;
  planes.layout = YuvLayout::Nv12;
  planes.width = s.options.width;
  planes.height = s.options.height;
  planes.device = (*pick)->buffer;
  planes.offset[0] = 0;
  planes.offset[1] = std::uint64_t{all.stride} * s.options.height;
  planes.stride[0] = all.stride;
  planes.stride[1] = all.stride;
  (*pick)->handed = ++all.handed;
  picture.target = *pick;
  out = std::move(planes);
  return {};
} catch (...) {
  return core::exception_status("HevcEncoder::target",
                                core::Status::Code::OutOfMemory);
}

void HevcEncodeSession::lose() noexcept { impl_->codec.reset(); }

}  // namespace volumetric_kit::sensor::video
