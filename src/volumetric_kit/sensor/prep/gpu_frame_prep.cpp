// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "undistort_color_comp.spv.hpp"
#include "undistort_depth_comp.spv.hpp"
#include "volumetric_kit/recon/core/command_batch.hpp"
#include "volumetric_kit/recon/core/compute_util.hpp"

namespace volumetric_kit::recon::sensor {
namespace {

// Both kernels' local_size_x.
constexpr std::uint32_t kLocalSize = 256;

// shaders/lens.glsl's LensCamera, under scalar layout.
struct LensParams {
  float fx, fy, cx, cy;
  std::uint32_t width, height;
  float k1, k2, p1, p2, k3, k4, k5, k6;
};
static_assert(sizeof(LensParams) == 56, "LensParams must match lens.glsl");

struct DepthParams {
  LensParams cam;
  float metres_per_unit;
};
static_assert(sizeof(DepthParams) == 60, "DepthParams layout drift");

struct ColorParams {
  LensParams cam;
  std::uint32_t y_offset;
  std::uint32_t cb_offset;
  std::uint32_t cr_offset;
  std::uint32_t y_stride;
  std::uint32_t c_stride;
  std::uint32_t c_step;
  float kr;
  float kb;
  std::uint32_t full_range;
};
static_assert(sizeof(ColorParams) == 92, "ColorParams layout drift");

LensParams lens_params(const LensCamera& c) noexcept {
  const LensDistortion& d = c.lens;
  return LensParams{c.fx, c.fy, c.cx, c.cy, c.width, c.height, d.k1,
                    d.k2, d.p1, d.p2, d.k3, d.k4,    d.k5,     d.k6};
}

bool finite(float v) noexcept { return std::isfinite(v); }

// A camera the pass can undistort: a non-empty image, positive finite focal
// lengths, and a finite principal point and lens.
Status check_camera(const char* what, const LensCamera& c) {
  const LensDistortion& d = c.lens;
  if (c.width == 0 || c.height == 0) {
    return Status::invalid_argument(std::string("GpuFramePrep: the ") + what +
                                    " camera has an empty image");
  }
  if (!(finite(c.fx) && c.fx > 0.0f && finite(c.fy) && c.fy > 0.0f &&
        finite(c.cx) && finite(c.cy))) {
    return Status::invalid_argument(std::string("GpuFramePrep: the ") + what +
                                    " camera's intrinsics are not finite and "
                                    "positive");
  }
  for (const float k : {d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6}) {
    if (!finite(k)) {
      return Status::invalid_argument(std::string("GpuFramePrep: the ") + what +
                                      " camera's lens is not finite");
    }
  }
  return {};
}

// A buffer of at least `bytes`, kept when it is big enough: a device-local
// input, filled through the pass's batch so the kernels never read the raw
// frame across the bus, or the host-visible staging the frame is written
// into. The staging is kept too: a batch stages through a buffer of its own
// per call, and four passes doing that at once with 4K frames had VMA
// allocate and free a block for every set.
Status ensure_buffer(const Device& device, Allocator& allocator, Buffer& buffer,
                     VkDeviceSize bytes, bool staging, const char* name) {
  if (buffer.valid() && buffer.size() >= bytes) return {};
  buffer = Buffer();
  if (staging) {
    VR_ASSIGN(buffer,
              storage_buffer(allocator, bytes, HostAccess::SequentialWrite,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
  } else {
    VR_ASSIGN(buffer, device_storage_buffer(allocator, bytes));
  }
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         debug_object_handle(buffer.handle()), name);
  return {};
}

VkDeviceSize round_up4(VkDeviceSize bytes) noexcept {
  return (bytes + 3) & ~VkDeviceSize{3};
}

// A frame's depth half, checked, and the bytes it moves.
struct DepthLayout {
  std::uint32_t pixels = 0;
  VkDeviceSize in_bytes = 0;   // the raw samples, rounded up to a word
  VkDeviceSize out_bytes = 0;  // the metres
};

// A frame's colour half, checked: where the kernel finds each plane in the
// buffer it binds, and the bytes it moves. Host planes are staged packed
// tightly, each starting on a word; device planes are read where they are.
struct ColorLayout {
  std::uint32_t pixels = 0;
  std::uint32_t ch = 0;  // chroma rows
  bool on_device = false;
  VkDeviceSize y_offset = 0, cb_offset = 0, cr_offset = 0;
  VkDeviceSize y_stride = 0, c_stride = 0, c_step = 1;
  VkDeviceSize in_bytes = 0;    // staged: none for device planes
  VkDeviceSize bind_bytes = 0;  // the range bound, from byte 0
  VkDeviceSize out_bytes = 0;
};

// Whether `rows` rows of `row_bytes`, `stride` apart from `offset`, fit in
// `size` bytes, worked out without overflowing 64 bits. `stride` is at least
// `row_bytes`, which is at least 1.
bool plane_fits(std::uint64_t offset, std::uint64_t stride, std::uint64_t rows,
                std::uint64_t row_bytes, std::uint64_t size) noexcept {
  if (offset > size || row_bytes > size - offset) return false;
  return rows <= 1 || rows - 1 <= (size - offset - row_bytes) / stride;
}

Result<DepthLayout> check_depth(const RawFrame& frame, std::uint64_t max_pixels,
                                VkDeviceSize max_range) {
  const LensCamera& cam = frame.depth_camera;
  if (frame.depth == nullptr) {
    return Status::invalid_argument("GpuFramePrep: the frame has no depth");
  }
  VR_TRY(check_camera("depth", cam));
  if (!(finite(frame.metres_per_unit) && frame.metres_per_unit > 0.0f)) {
    return Status::invalid_argument(
        "GpuFramePrep: metres_per_unit is not finite and positive");
  }
  // 0 is the pass's "no return", where the sensor had none and where the lens
  // maps outside the image, so a range reaching it would fuse those pixels
  // as a surface at the camera; and a range left at RawFrame's zeros would
  // quietly fuse nothing.
  if (!(finite(frame.min_depth) && finite(frame.max_depth) &&
        frame.min_depth > 0.0f && frame.min_depth < frame.max_depth)) {
    return Status::invalid_argument(
        "GpuFramePrep: the depth range [" + std::to_string(frame.min_depth) +
        ", " + std::to_string(frame.max_depth) +
        "] m must be finite, with 0 < min_depth < max_depth");
  }
  const std::uint64_t pixels = std::uint64_t{cam.width} * cam.height;
  if (pixels > max_pixels) {
    return Status::invalid_argument(
        "GpuFramePrep: the depth image is past a single dispatch");
  }
  DepthLayout out;
  out.pixels = static_cast<std::uint32_t>(pixels);
  out.in_bytes = round_up4(pixels * sizeof(std::uint16_t));
  out.out_bytes = pixels * sizeof(float);
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the depth buffer",
                                    out.out_bytes, max_range));
  return out;
}

Result<ColorLayout> check_color(const RawFrame& frame, std::uint64_t max_pixels,
                                VkDeviceSize max_range) {
  const LensCamera& cam = frame.color_camera;
  const YuvImage& image = frame.color;
  if (!is_canonical(frame.color_encoding)) {
    // TODO(sensor): the other transfers and primaries, through the curve
    // and matrix sensor::to_canonical uses on the host.
    return Status::unsupported(
        "GpuFramePrep: converts colour from the canonical encoding only (sRGB "
        "or BT.709 transfer, BT.709 primaries)");
  }
  VR_TRY(check_camera("colour", cam));
  if (image.width != cam.width || image.height != cam.height) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour picture is " + std::to_string(image.width) +
        "x" + std::to_string(image.height) + ", its camera " +
        std::to_string(cam.width) + "x" + std::to_string(cam.height));
  }
  if (!(finite(image.kr) && finite(image.kb) && image.kr > 0.0f &&
        image.kb > 0.0f && image.kr + image.kb < 1.0f)) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour matrix's kr and kb must be positive and sum "
        "below 1");
  }
  const std::uint64_t pixels = std::uint64_t{cam.width} * cam.height;
  if (pixels > max_pixels) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour image is past a single dispatch");
  }
  const bool nv12 = image.layout == YuvLayout::Nv12;
  const bool on_device = image.device != nullptr;
  if (on_device == (image.plane[0] != nullptr)) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour planes must be on the host or on the device, "
        "not both");
  }
  const int planes = nv12 ? 2 : 3;
  const std::uint64_t cw = (std::uint64_t{image.width} + 1) / 2;
  const std::uint64_t ch = (std::uint64_t{image.height} + 1) / 2;
  const std::uint64_t c_row = nv12 ? 2 * cw : cw;
  const std::uint64_t rows[3] = {image.height, ch, ch};
  const std::uint64_t row_bytes[3] = {image.width, c_row, c_row};
  for (int p = 0; p < planes; ++p) {
    if ((!on_device && image.plane[p] == nullptr) ||
        image.stride[p] < row_bytes[p]) {
      return Status::invalid_argument(
          std::string("GpuFramePrep: the colour picture needs ") +
          (nv12 ? "two planes" : "three planes") +
          ", each row at least its width");
    }
  }

  ColorLayout out;
  out.pixels = static_cast<std::uint32_t>(pixels);
  out.ch = static_cast<std::uint32_t>(ch);
  out.on_device = on_device;
  out.c_step = nv12 ? 2 : 1;
  if (!on_device) {
    const std::uint64_t chroma_bytes = c_row * ch;
    out.y_stride = image.width;
    out.c_stride = c_row;
    out.cb_offset = round_up4(pixels);
    out.cr_offset =
        nv12 ? out.cb_offset + 1 : out.cb_offset + round_up4(chroma_bytes);
    out.in_bytes =
        (nv12 ? out.cb_offset : out.cr_offset) + round_up4(chroma_bytes);
    out.bind_bytes = out.in_bytes;
  } else {
    const Buffer& buffer = *image.device;
    if ((buffer.usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0) {
      return Status::invalid_argument(
          "GpuFramePrep: the colour planes' buffer is not a storage buffer");
    }
    // One chroma stride, as the kernel reads both chroma planes with one.
    if (!nv12 && image.stride[1] != image.stride[2]) {
      return Status::invalid_argument(
          "GpuFramePrep: the device's Cb and Cr rows must be as long");
    }
    std::uint64_t end = 0;
    for (int p = 0; p < planes; ++p) {
      if (!plane_fits(image.offset[p], image.stride[p], rows[p], row_bytes[p],
                      buffer.size())) {
        return Status::invalid_argument(
            "GpuFramePrep: a colour plane runs past its buffer");
      }
      end = std::max(end, image.offset[p] + (rows[p] - 1) * image.stride[p] +
                              row_bytes[p]);
    }
    // The kernel reads whole words, so the last one must be in the buffer.
    out.bind_bytes = round_up4(end);
    if (out.bind_bytes > buffer.size()) {
      return Status::invalid_argument(
          "GpuFramePrep: the colour planes' last word runs past their buffer");
    }
    out.y_offset = image.offset[0];
    out.cb_offset = image.offset[1];
    out.cr_offset = nv12 ? image.offset[1] + 1 : image.offset[2];
    out.y_stride = image.stride[0];
    out.c_stride = image.stride[1];
  }
  out.out_bytes = pixels * sizeof(std::uint32_t);
  // The kernel addresses the planes in 32-bit bytes.
  if (out.bind_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour picture is past 4 GiB");
  }
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                    out.out_bytes, max_range));
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the colour planes",
                                    out.bind_bytes, max_range));
  return out;
}

}  // namespace

Result<GpuFramePrep> GpuFramePrep::create(Device& device,
                                          Allocator& allocator) {
  GpuFramePrep prep;
  prep.device_ = &device;
  prep.allocator_ = &allocator;

  // Two storage bindings each, input then output, and the kernel's params as
  // push constants.
  VkPushConstantRange depth_push{};
  depth_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  depth_push.size = sizeof(DepthParams);
  VkPushConstantRange color_push = depth_push;
  color_push.size = sizeof(ColorParams);
  KernelSetBuilder kb(device);
  VR_TRY(kb.add(prep.depth_kernel_, "undistort_depth",
                vr_undistort_depth_comp_spv, vr_undistort_depth_comp_spv_size,
                2, &depth_push));
  VR_TRY(kb.add(prep.color_kernel_, "undistort_color",
                vr_undistort_color_comp_spv, vr_undistort_color_comp_spv_size,
                2, &color_push));
  VR_ASSIGN(prep.pool_, kb.build());

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  prep.max_workgroup_count_x_ = props.limits.maxComputeWorkGroupCount[0];
  prep.max_storage_buffer_range_ = props.limits.maxStorageBufferRange;
  VR_ASSIGN(prep.gpu_timer_, GpuTimer::create(device));
  return prep;
}

Result<DeviceFrame> GpuFramePrep::prepare(const RawFrame& frame,
                                          StageMetrics* metrics) {
  GpuStageScope stage(metrics, gpu_timer_, "frame prep");
  if (!valid()) {
    return Status::invalid_argument("GpuFramePrep: moved-from pass");
  }
  // Both halves checked before either is uploaded, so a refused frame costs
  // no work and leaves every buffer as it was.
  const std::uint64_t max_pixels =
      std::uint64_t{max_workgroup_count_x_} * kLocalSize;
  VR_ASSIGN(const DepthLayout depth,
            check_depth(frame, max_pixels, max_storage_buffer_range_));
  ColorLayout color;
  if (frame.has_color()) {
    VR_ASSIGN(color, check_color(frame, max_pixels, max_storage_buffer_range_));
  }

  VR_TRY(ensure_buffer(*device_, *allocator_, depth_in_, depth.in_bytes, false,
                       "sensor.raw_depth"));
  VR_TRY(ensure_output(depth_out_, depth.out_bytes, "sensor.depth_frame"));
  if (frame.has_color()) {
    if (!color.on_device) {
      VR_TRY(ensure_buffer(*device_, *allocator_, color_in_, color.in_bytes,
                           false, "sensor.raw_color"));
    }
    VR_TRY(ensure_output(color_out_, color.out_bytes, "sensor.color_frame"));
  }

  VR_TRY(ensure_buffer(*device_, *allocator_, staging_,
                       depth.in_bytes + color.in_bytes, true,
                       "sensor.raw_staging"));

  // The frame staged as the inputs lay it out: depth, then any host planes
  // packed tightly whatever the decoder's strides. A tight plane is one
  // memcpy: row by row, it cost the 5090 0.2 ms a 4K frame. Device planes
  // are read where they are.
  auto* staged = static_cast<std::uint8_t*>(staging_.mapped());
  const VkDeviceSize depth_bytes =
      VkDeviceSize{depth.pixels} * sizeof(std::uint16_t);
  std::memcpy(staged, frame.depth, static_cast<std::size_t>(depth_bytes));
  if (frame.has_color() && !color.on_device) {
    const YuvImage& image = frame.color;
    const std::size_t widths[3] = {image.width, color.c_stride, color.c_stride};
    const std::uint32_t heights[3] = {image.height, color.ch, color.ch};
    const VkDeviceSize offsets[3] = {0, color.cb_offset, color.cr_offset};
    std::uint8_t* dst = staged + depth.in_bytes;
    const int planes = image.layout == YuvLayout::Nv12 ? 2 : 3;
    for (int p = 0; p < planes; ++p) {
      if (image.stride[p] == widths[p]) {
        std::memcpy(dst + offsets[p], image.plane[p],
                    std::size_t{widths[p]} * heights[p]);
        continue;
      }
      for (std::uint32_t row = 0; row < heights[p]; ++row) {
        std::memcpy(dst + offsets[p] + std::size_t{row} * widths[p],
                    image.plane[p] + std::size_t{row} * image.stride[p],
                    widths[p]);
      }
    }
  }

  // One batch: both copies up, then both passes, one submit a frame. The
  // copies are timed with the passes, so the row's device half counts moving
  // the frame too.
  CommandBatch batch(*device_, *allocator_);
  VR_TRY(batch.copy(staging_, 0, depth_in_, 0, depth_bytes, &stage));
  if (frame.has_color() && !color.on_device) {
    VR_TRY(batch.copy(staging_, depth.in_bytes, color_in_, 0, color.in_bytes,
                      &stage));
  }

  depth_kernel_.set.write_storage_buffer(0, depth_in_.handle(), 0,
                                         depth.in_bytes);
  depth_kernel_.set.write_storage_buffer(1, depth_out_->handle(), 0,
                                         depth.out_bytes);
  const DepthParams depth_params{lens_params(frame.depth_camera),
                                 frame.metres_per_unit};
  VR_TRY(batch.dispatch(depth_kernel_, &depth_params, sizeof(depth_params),
                        group_count(depth.pixels, kLocalSize),
                        max_workgroup_count_x_, &stage));
  if (frame.has_color()) {
    const YuvImage& image = frame.color;
    color_kernel_.set.write_storage_buffer(
        0, color.on_device ? image.device->handle() : color_in_.handle(), 0,
        color.bind_bytes);
    color_kernel_.set.write_storage_buffer(1, color_out_->handle(), 0,
                                           color.out_bytes);
    const auto u32 = [](VkDeviceSize v) {
      return static_cast<std::uint32_t>(v);
    };
    const ColorParams color_params{lens_params(frame.color_camera),
                                   u32(color.y_offset),
                                   u32(color.cb_offset),
                                   u32(color.cr_offset),
                                   u32(color.y_stride),
                                   u32(color.c_stride),
                                   u32(color.c_step),
                                   image.kr,
                                   image.kb,
                                   image.full_range ? 1u : 0u};
    VR_TRY(batch.dispatch(color_kernel_, &color_params, sizeof(color_params),
                          group_count(color.pixels, kLocalSize),
                          max_workgroup_count_x_, &stage));
  }
  const Status submitted = batch.submit();
  if (!submitted.ok()) {
    // A failed wait may leave the copy running, so the staging is let go
    // rather than rewritten or freed, as the batch lets go of its own.
    static_cast<void>(new Buffer(std::move(staging_)));
    return submitted;
  }

  const LensCamera& d = frame.depth_camera;
  DeviceFrame out;
  out.depth = depth_out_;
  out.depth_camera.fx = d.fx;
  out.depth_camera.fy = d.fy;
  out.depth_camera.cx = d.cx;
  out.depth_camera.cy = d.cy;
  out.depth_camera.min_depth = frame.min_depth;
  out.depth_camera.max_depth = frame.max_depth;
  out.depth_camera.width = d.width;
  out.depth_camera.height = d.height;
  out.depth_camera.cam_to_world = frame.depth_cam_to_world;
  if (frame.has_color()) {
    const LensCamera& c = frame.color_camera;
    out.color = color_out_;
    out.color_camera = ColorCameraParams{
        c.fx, c.fy, c.cx, c.cy, c.width, c.height, frame.color_cam_to_world};
    out.color_encoding = frame.color_encoding;
  }
  out.timestamp_ns = frame.timestamp_ns;
  return out;
}

Status GpuFramePrep::ensure_output(std::shared_ptr<Buffer>& buffer,
                                   VkDeviceSize bytes, const char* name) {
  // Reused only when this pass holds the last reference: a DeviceFrame kept
  // past this call keeps its contents, and this frame goes to a new buffer,
  // which measured no slower than reusing one (the residency decision's step
  // 5b), so there is no ring.
  if (buffer != nullptr && buffer.use_count() == 1 && buffer->size() >= bytes) {
    return {};
  }
  // Device-local: only the kernels touch it, and on a discrete GPU the
  // fusion kernels' reads would otherwise cross the bus.
  VR_ASSIGN(Buffer created, device_storage_buffer(*allocator_, bytes));
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           debug_object_handle(created.handle()), name);
  buffer = std::make_shared<Buffer>(std::move(created));
  return {};
}

Result<std::vector<std::optional<DeviceFrame>>> prepare_set(
    std::vector<GpuFramePrep>& preps,
    const std::vector<std::optional<RawFrame>>& frames) {
  if (preps.size() < frames.size()) {
    return Status::invalid_argument(
        "prepare_set: " + std::to_string(frames.size()) + " frames for " +
        std::to_string(preps.size()) + " passes");
  }
  std::vector<std::optional<DeviceFrame>> out(frames.size());
  std::vector<Status> status(frames.size());
  // Never throws: an exception leaving a thread would end the process.
  const auto run = [&](std::size_t i) noexcept {
    try {
      Result<DeviceFrame> prepared = preps[i].prepare(*frames[i]);
      if (prepared.ok()) {
        out[i] = std::move(prepared).value();
      } else {
        status[i] = prepared.status();
      }
    } catch (const std::bad_alloc&) {
      status[i] = Status::out_of_memory("prepare_set: out of host memory");
    }
  };
  // Joined on every way out, a throw included, so no thread outlives what it
  // writes to.
  struct Workers {
    std::vector<std::thread> threads;
    ~Workers() {
      for (std::thread& t : threads) t.join();
    }
  };
  {
    Workers workers;
    workers.threads.reserve(frames.size());
    // Every frame but the last on a thread of its own, the last on this one.
    // A thread that cannot be started runs its frame here instead.
    std::optional<std::size_t> last;
    for (std::size_t i = 0; i < frames.size(); ++i) {
      if (!frames[i]) continue;
      if (last) {
        try {
          workers.threads.emplace_back(run, *last);
        } catch (const std::system_error&) {
          run(*last);
        }
      }
      last = i;
    }
    if (last) run(*last);
  }
  for (const Status& s : status) {
    if (!s.ok()) return s;
  }
  return out;
}

}  // namespace volumetric_kit::recon::sensor
