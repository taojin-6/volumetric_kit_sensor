// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
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
  std::uint32_t cb_offset;
  std::uint32_t cr_offset;
  float kr;
  float kb;
  std::uint32_t full_range;
};
static_assert(sizeof(ColorParams) == 76, "ColorParams layout drift");

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

// An input of at least `bytes`, kept when it is big enough. Device-local, and
// filled through the pass's batch, so the kernels never read the raw frame
// across the bus.
Status ensure_input(const Device& device, Allocator& allocator, Buffer& buffer,
                    VkDeviceSize bytes, const char* name) {
  if (buffer.valid() && buffer.size() >= bytes) return {};
  buffer = Buffer();
  VR_ASSIGN(buffer, device_storage_buffer(allocator, bytes));
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

// A frame's colour half, checked: the three planes packed tightly, each
// starting on a word.
struct ColorLayout {
  std::uint32_t pixels = 0;
  std::uint32_t cw = 0, ch = 0;  // the chroma planes' size
  VkDeviceSize cb_offset = 0, cr_offset = 0;
  VkDeviceSize in_bytes = 0;
  VkDeviceSize out_bytes = 0;
};

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
  ColorLayout out;
  out.cw = (image.width + 1) / 2;
  out.ch = (image.height + 1) / 2;
  if (image.plane[1] == nullptr || image.plane[2] == nullptr ||
      image.stride[0] < image.width || image.stride[1] < out.cw ||
      image.stride[2] < out.cw) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour picture needs three planes, each row at "
        "least its width");
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
  out.pixels = static_cast<std::uint32_t>(pixels);
  const std::uint64_t chroma_bytes = std::uint64_t{out.cw} * out.ch;
  out.cb_offset = round_up4(pixels);
  out.cr_offset = out.cb_offset + round_up4(chroma_bytes);
  out.in_bytes = out.cr_offset + round_up4(chroma_bytes);
  out.out_bytes = pixels * sizeof(std::uint32_t);
  if (out.in_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour picture is past 4 GiB");
  }
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                    out.out_bytes, max_range));
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the colour planes",
                                    out.in_bytes, max_range));
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

  VR_TRY(ensure_input(*device_, *allocator_, depth_in_, depth.in_bytes,
                      "sensor.raw_depth"));
  VR_TRY(ensure_output(depth_out_, depth.out_bytes, "sensor.depth_frame"));
  if (frame.has_color()) {
    VR_TRY(ensure_input(*device_, *allocator_, color_in_, color.in_bytes,
                        "sensor.raw_color"));
    VR_TRY(ensure_output(color_out_, color.out_bytes, "sensor.color_frame"));
  }

  // One batch: both uploads, then both passes, one submit a frame. The
  // uploads are timed with the passes, so the row's device half counts
  // moving the frame too.
  CommandBatch batch(*device_, *allocator_);
  VR_TRY(batch.upload(depth_in_, 0, frame.depth,
                      VkDeviceSize{depth.pixels} * sizeof(std::uint16_t),
                      &stage));
  if (frame.has_color()) {
    // The three planes packed tightly into one staging buffer, whatever the
    // decoder's strides, and copied up as one. A tight plane is one memcpy:
    // row by row, it cost the 5090 0.2 ms a 4K frame.
    const YuvImage& image = frame.color;
    const std::uint32_t widths[3] = {image.width, color.cw, color.cw};
    const std::uint32_t heights[3] = {image.height, color.ch, color.ch};
    const VkDeviceSize offsets[3] = {0, color.cb_offset, color.cr_offset};
    VR_ASSIGN(void* staging,
              batch.reserve_upload(color_in_, 0, color.in_bytes, &stage));
    auto* dst = static_cast<std::uint8_t*>(staging);
    for (int p = 0; p < 3; ++p) {
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
    color_kernel_.set.write_storage_buffer(0, color_in_.handle(), 0,
                                           color.in_bytes);
    color_kernel_.set.write_storage_buffer(1, color_out_->handle(), 0,
                                           color.out_bytes);
    const YuvImage& image = frame.color;
    const ColorParams color_params{lens_params(frame.color_camera),
                                   static_cast<std::uint32_t>(color.cb_offset),
                                   static_cast<std::uint32_t>(color.cr_offset),
                                   image.kr,
                                   image.kb,
                                   image.full_range ? 1u : 0u};
    VR_TRY(batch.dispatch(color_kernel_, &color_params, sizeof(color_params),
                          group_count(color.pixels, kLocalSize),
                          max_workgroup_count_x_, &stage));
  }
  VR_TRY(batch.submit());

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
  // past this call keeps its contents, and this frame goes to a new buffer.
  // TODO(sensor): the outputs on a ring (the residency decision's step 5), so
  // a frame kept past the next costs no allocation.
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

}  // namespace volumetric_kit::recon::sensor
