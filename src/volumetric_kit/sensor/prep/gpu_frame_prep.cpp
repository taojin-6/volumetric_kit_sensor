// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>

#include "undistort_color_comp.spv.hpp"
#include "undistort_depth_comp.spv.hpp"
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

// A buffer of at least `bytes`, kept when it is big enough already.
Status ensure(Allocator& allocator, Buffer& buffer, VkDeviceSize bytes,
              HostAccess access) {
  if (buffer.valid() && buffer.size() >= bytes) return {};
  buffer = Buffer();
  VR_ASSIGN(buffer, storage_buffer(allocator, bytes, access));
  return {};
}

VkDeviceSize round_up4(VkDeviceSize bytes) noexcept {
  return (bytes + 3) & ~VkDeviceSize{3};
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
  if (frame.has_color() && !is_canonical(frame.color_encoding)) {
    // TODO(sensor): the other transfers and primaries, through the curve
    // and matrix sensor::to_canonical uses on the host.
    return Status::unsupported(
        "GpuFramePrep: converts colour from the canonical encoding only (sRGB "
        "or BT.709 transfer, BT.709 primaries)");
  }
  VR_TRY(prepare_depth(frame, stage));
  if (frame.has_color()) VR_TRY(prepare_color(frame, stage));

  const LensCamera& d = frame.depth_camera;
  DeviceFrame out;
  out.depth = &depth_out_;
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
    out.color = &color_out_;
    out.color_camera = ColorCameraParams{
        c.fx, c.fy, c.cx, c.cy, c.width, c.height, frame.color_cam_to_world};
    out.color_encoding = frame.color_encoding;
  }
  out.timestamp_ns = frame.timestamp_ns;
  return out;
}

Status GpuFramePrep::prepare_depth(const RawFrame& frame,
                                   GpuStageScope& stage) {
  const LensCamera& cam = frame.depth_camera;
  if (frame.depth == nullptr) {
    return Status::invalid_argument("GpuFramePrep: the frame has no depth");
  }
  VR_TRY(check_camera("depth", cam));
  if (!(finite(frame.metres_per_unit) && frame.metres_per_unit > 0.0f)) {
    return Status::invalid_argument(
        "GpuFramePrep: metres_per_unit is not finite and positive");
  }
  const std::uint64_t pixels = std::uint64_t{cam.width} * cam.height;
  if (pixels > std::uint64_t{max_workgroup_count_x_} * kLocalSize) {
    return Status::invalid_argument(
        "GpuFramePrep: the depth image is past a single dispatch");
  }
  const VkDeviceSize raw_bytes = round_up4(pixels * sizeof(std::uint16_t));
  const VkDeviceSize out_bytes = pixels * sizeof(float);
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the depth buffer", out_bytes,
                                    max_storage_buffer_range_));
  VR_TRY(
      ensure(*allocator_, depth_in_, raw_bytes, HostAccess::SequentialWrite));
  VR_TRY(ensure(*allocator_, depth_out_, out_bytes, HostAccess::Random));
  std::memcpy(depth_in_.mapped(), frame.depth,
              static_cast<std::size_t>(pixels) * sizeof(std::uint16_t));

  depth_kernel_.set.write_storage_buffer(0, depth_in_.handle(), 0,
                                         VK_WHOLE_SIZE);
  depth_kernel_.set.write_storage_buffer(1, depth_out_.handle(), 0,
                                         VK_WHOLE_SIZE);
  const DepthParams params{lens_params(cam), frame.metres_per_unit};
  return dispatch(*device_, depth_kernel_, &params, sizeof(params),
                  group_count(static_cast<std::uint32_t>(pixels), kLocalSize),
                  max_workgroup_count_x_, &stage);
}

Status GpuFramePrep::prepare_color(const RawFrame& frame,
                                   GpuStageScope& stage) {
  const LensCamera& cam = frame.color_camera;
  const YuvImage& image = frame.color;
  VR_TRY(check_camera("colour", cam));
  if (image.width != cam.width || image.height != cam.height) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour picture is " + std::to_string(image.width) +
        "x" + std::to_string(image.height) + ", its camera " +
        std::to_string(cam.width) + "x" + std::to_string(cam.height));
  }
  const std::uint32_t cw = (image.width + 1) / 2;
  const std::uint32_t ch = (image.height + 1) / 2;
  if (image.plane[1] == nullptr || image.plane[2] == nullptr ||
      image.stride[0] < image.width || image.stride[1] < cw ||
      image.stride[2] < cw) {
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
  if (pixels > std::uint64_t{max_workgroup_count_x_} * kLocalSize) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour image is past a single dispatch");
  }
  const std::uint64_t luma_bytes = pixels;
  const std::uint64_t chroma_bytes = std::uint64_t{cw} * ch;
  const VkDeviceSize cb_offset = round_up4(luma_bytes);
  const VkDeviceSize cr_offset = cb_offset + round_up4(chroma_bytes);
  const VkDeviceSize in_bytes = cr_offset + round_up4(chroma_bytes);
  const VkDeviceSize out_bytes = pixels * sizeof(std::uint32_t);
  if (in_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return Status::invalid_argument(
        "GpuFramePrep: the colour picture is past 4 GiB");
  }
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                    out_bytes, max_storage_buffer_range_));
  VR_TRY(check_storage_buffer_range("GpuFramePrep: the colour planes", in_bytes,
                                    max_storage_buffer_range_));
  VR_TRY(ensure(*allocator_, color_in_, in_bytes, HostAccess::SequentialWrite));
  VR_TRY(ensure(*allocator_, color_out_, out_bytes, HostAccess::Random));

  // Rows packed tightly, whatever the decoder's strides.
  auto* dst = static_cast<std::uint8_t*>(color_in_.mapped());
  const std::uint32_t widths[3] = {image.width, cw, cw};
  const std::uint32_t heights[3] = {image.height, ch, ch};
  const VkDeviceSize offsets[3] = {0, cb_offset, cr_offset};
  for (int p = 0; p < 3; ++p) {
    for (std::uint32_t row = 0; row < heights[p]; ++row) {
      std::memcpy(dst + offsets[p] + std::size_t{row} * widths[p],
                  image.plane[p] + std::size_t{row} * image.stride[p],
                  widths[p]);
    }
  }

  color_kernel_.set.write_storage_buffer(0, color_in_.handle(), 0,
                                         VK_WHOLE_SIZE);
  color_kernel_.set.write_storage_buffer(1, color_out_.handle(), 0,
                                         VK_WHOLE_SIZE);
  const ColorParams params{lens_params(cam),
                           static_cast<std::uint32_t>(cb_offset),
                           static_cast<std::uint32_t>(cr_offset),
                           image.kr,
                           image.kb,
                           image.full_range ? 1u : 0u};
  return dispatch(*device_, color_kernel_, &params, sizeof(params),
                  group_count(static_cast<std::uint32_t>(pixels), kLocalSize),
                  max_workgroup_count_x_, &stage);
}

}  // namespace volumetric_kit::recon::sensor
