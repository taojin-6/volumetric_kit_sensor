// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/sensor/utils/gpu_frame_prep.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "undistort_color_comp.spv.hpp"
#include "undistort_depth_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"
#include "yuv_planes.hpp"

namespace volumetric_kit::recon::sensor {
namespace {

// Both kernels' local_size_x.
constexpr std::uint32_t kLocalSize = 256;

// The most commands record_inputs and record_passes time for one frame: the
// slot's copy, two plane-image copies and the two passes.
constexpr std::uint32_t kSpansPerFrame = 5;

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
  std::uint32_t within_color;  // mask by OverlapParams
};
static_assert(sizeof(DepthParams) == 64, "DepthParams layout drift");

// undistort_depth.comp's Overlap block: the colour camera, and the rigid
// transform from the depth camera's frame into the colour camera's
// (column-major, as the shader's mat4).
struct OverlapParams {
  LensParams color;
  Mat4f depth_to_color;
};
static_assert(sizeof(OverlapParams) == 56 + 64, "OverlapParams layout drift");
static_assert(offsetof(OverlapParams, depth_to_color) == 56,
              "OverlapParams layout drift");

struct ColorParams {
  LensParams cam;
  std::uint32_t y_offset;
  std::uint32_t cb_offset;
  std::uint32_t cr_offset;
  std::uint32_t y_stride;
  std::uint32_t cb_stride;
  std::uint32_t cr_stride;
  std::uint32_t c_step;
  float kr;
  float kb;
  std::uint32_t full_range;
  float chroma_x;
  float chroma_y;
  std::uint32_t packed_rgb;  // packed R'G'B' words, not Y'CbCr planes
};
static_assert(sizeof(ColorParams) == 108, "ColorParams layout drift");
static_assert(offsetof(ColorParams, chroma_x) == 96,
              "ColorParams layout drift");
static_assert(offsetof(ColorParams, packed_rgb) == 104,
              "ColorParams layout drift");

// rgbd_frame.hpp spells Vulkan's special queue families without Vulkan.
static_assert(kQueueFamilyIgnored == VK_QUEUE_FAMILY_IGNORED,
              "kQueueFamilyIgnored must be VK_QUEUE_FAMILY_IGNORED");
static_assert(kQueueFamilyExternal == VK_QUEUE_FAMILY_EXTERNAL,
              "kQueueFamilyExternal must be VK_QUEUE_FAMILY_EXTERNAL");

// The model as the shaders read it: narrowed to float once, here.
LensParams lens_params(const camera::CameraModel& c) noexcept {
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  const auto f = [](double v) { return static_cast<float>(v); };
  return LensParams{f(k.fx),       f(k.fy), f(k.cx), f(k.cy), c.size.width,
                    c.size.height, f(d.k1), f(d.k2), f(d.p1), f(d.p2),
                    f(d.k3),       f(d.k4), f(d.k5), f(d.k6)};
}

bool finite(float v) noexcept { return std::isfinite(v); }

// A camera the pass can undistort: camera::check_camera_model's checks, and
// every value within float's range, as the shaders read it -- compared before
// narrowing, which past that range is undefined -- with focal lengths that
// stay positive once narrowed.
core::Status check_camera(const char* what, const camera::CameraModel& c) {
  const core::Status valid = camera::check_camera_model(c);
  if (!valid.ok()) {
    return core::Status::invalid_argument(std::string("GpuFramePrep: the ") +
                                          what + " " + valid.message());
  }
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  for (const double v : {k.fx, k.fy, k.cx, k.cy, d.k1, d.k2, d.p1, d.p2, d.k3,
                         d.k4, d.k5, d.k6}) {
    if (std::fabs(v) > std::numeric_limits<float>::max()) {
      return core::Status::invalid_argument(std::string("GpuFramePrep: the ") +
                                            what +
                                            " camera is past float's range");
    }
  }
  const LensParams p = lens_params(c);
  if (!(p.fx > 0.0f && p.fy > 0.0f)) {
    return core::Status::invalid_argument(
        std::string("GpuFramePrep: the ") + what +
        " camera's focal lengths narrow to 0 in float");
  }
  return {};
}

// A pose the pass can apply; `what` names it in the error.
core::Status check_pose(const char* what, const camera::Mat4d& pose) {
  const core::Status rigid = camera::check_rigid(pose);
  if (!rigid.ok()) {
    return core::Status::invalid_argument(std::string("GpuFramePrep: ") + what +
                                          ": " + rigid.message());
  }
  return {};
}

VkDeviceSize round_up4(VkDeviceSize bytes) noexcept {
  return (bytes + 3) & ~VkDeviceSize{3};
}

// `bytes` rounded up to a multiple of `alignment`, a power of two.
VkDeviceSize round_up(VkDeviceSize bytes, VkDeviceSize alignment) noexcept {
  return (bytes + alignment - 1) & ~(alignment - 1);
}

// A frame's depth half, checked, and the bytes it moves.
struct DepthLayout {
  std::uint32_t pixels = 0;
  VkDeviceSize in_bytes = 0;   // the raw samples, rounded up to a word
  VkDeviceSize out_bytes = 0;  // the metres
};

// A frame's colour half, checked: the range of the planes' buffer the kernel
// binds, where it finds each plane in that range, and the bytes it moves.
// Plane images are copied into the pass's input packed tightly, each plane
// starting on a word, and bound from byte 0, as are packed host words; planes
// in a buffer are read where they are, bound from the first.
struct ColorLayout {
  std::uint32_t pixels = 0;
  std::uint32_t ch = 0;  // chroma rows
  // Each plane's first byte, from bind_offset, and its row stride; chroma
  // samples c_step bytes apart.
  VkDeviceSize y_offset = 0, cb_offset = 0, cr_offset = 0;
  VkDeviceSize y_stride = 0, cb_stride = 0, cr_stride = 0, c_step = 1;
  VkDeviceSize bind_offset = 0, bind_bytes = 0;
  VkDeviceSize out_bytes = 0;
};

core::Result<DepthLayout> check_depth(const RgbdFrame& frame,
                                      std::uint64_t max_pixels,
                                      VkDeviceSize max_range) {
  const camera::CameraModel& cam = frame.depth_camera;
  if (frame.depth == nullptr) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the frame has no depth");
  }
  VKC_TRY(check_camera("depth", cam));
  if (!(finite(frame.metres_per_unit) && frame.metres_per_unit > 0.0f)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: metres_per_unit is not finite and positive");
  }
  // 0 is the pass's "no return", where the sensor had none and where the lens
  // maps outside the image, so a range reaching it would fuse those pixels
  // as a surface at the camera; and a range left at RgbdFrame's zeros would
  // quietly fuse nothing.
  if (!(finite(frame.min_depth) && finite(frame.max_depth) &&
        frame.min_depth > 0.0f && frame.min_depth < frame.max_depth)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth range [" + std::to_string(frame.min_depth) +
        ", " + std::to_string(frame.max_depth) +
        "] m must be finite, with 0 < min_depth < max_depth");
  }
  VKC_TRY(check_pose("color_to_world", frame.color_to_world));
  VKC_TRY(check_pose("depth_to_color", frame.depth_to_color));
  const std::uint64_t pixels = std::uint64_t{cam.size.width} * cam.size.height;
  if (pixels > max_pixels) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the depth image is past a single dispatch");
  }
  DepthLayout out;
  out.pixels = static_cast<std::uint32_t>(pixels);
  out.in_bytes = round_up4(pixels * sizeof(std::uint16_t));
  out.out_bytes = pixels * sizeof(float);
  VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the depth buffer",
                                           out.out_bytes, max_range));
  return out;
}

core::Result<ColorLayout> check_color(const RgbdFrame& frame,
                                      std::uint64_t max_pixels,
                                      VkDeviceSize max_range,
                                      VkDeviceSize offset_alignment) {
  const camera::ImageSize& cam = frame.color_camera.size;
  const YuvImage& image = frame.color;
  if (!is_canonical(frame.color_encoding)) {
    // TODO(sensor): the other transfers and primaries, through the curve
    // and matrix sensor::to_canonical uses on the host.
    return core::Status::unsupported(
        "GpuFramePrep: converts colour from the canonical encoding only (sRGB "
        "or BT.709 transfer, BT.709 primaries)");
  }
  VKC_TRY(check_camera("colour", frame.color_camera));
  const std::uint64_t pixels = std::uint64_t{cam.width} * cam.height;
  if (pixels > max_pixels) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour image is past a single dispatch");
  }
  const bool on_device = image.device != nullptr;
  const bool as_images = image.image[0] != nullptr || image.image[1] != nullptr;
  ColorLayout out;
  out.pixels = static_cast<std::uint32_t>(pixels);
  out.out_bytes = pixels * sizeof(std::uint32_t);
  if (frame.color_packed != nullptr) {
    if (on_device || as_images) {
      return core::Status::invalid_argument(
          "GpuFramePrep: the colour must be packed words or Y'CbCr planes, "
          "not both");
    }
    // Staged whole, a word a pixel, and bound from byte 0 of the input.
    out.bind_bytes = out.out_bytes;
    VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                             out.out_bytes, max_range));
    return out;
  }
  if (static_cast<unsigned>(image.chroma_location) >
      static_cast<unsigned>(ChromaLocation::Bottom)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: unknown chroma location");
  }
  if (image.width != cam.width || image.height != cam.height) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour picture is " + std::to_string(image.width) +
        "x" + std::to_string(image.height) + ", its camera " +
        std::to_string(cam.width) + "x" + std::to_string(cam.height));
  }
  if (!(finite(image.kr) && finite(image.kb) && image.kr > 0.0f &&
        image.kb > 0.0f && image.kr + image.kb < 1.0f)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour matrix's kr and kb must be positive and sum "
        "below 1");
  }
  const bool nv12 = image.layout == YuvLayout::Nv12;
  if (on_device && !as_images) {
    // Empty or without storage usage, before its size is weighed.
    VKC_TRY(core::StorageInput(*image.device)
                .check("GpuFramePrep: the colour planes' buffer", 0));
  }
  VKC_ASSIGN(const detail::YuvPlanes planes,
             detail::check_yuv_planes(image, "GpuFramePrep"));
  const std::uint64_t c_row = planes.row_bytes[1];
  const std::uint64_t ch = planes.rows[1];

  out.ch = static_cast<std::uint32_t>(ch);
  out.c_step = nv12 ? 2 : 1;
  if (as_images) {
    // NV12, so Cr sits one byte after Cb.
    out.y_stride = image.width;
    out.cb_stride = out.cr_stride = c_row;
    out.cb_offset = round_up4(pixels);
    out.cr_offset = out.cb_offset + 1;
    out.bind_bytes = out.cb_offset + round_up4(c_row * ch);
  } else {
    const core::StorageInput bound(*image.device);
    const std::uint64_t first =
        *std::min_element(planes.begin, planes.begin + planes.count);
    const std::uint64_t last =
        *std::max_element(planes.end, planes.end + planes.count);
    // The kernel reads whole words, so the last one must be in the buffer.
    VKC_TRY(bound.check("GpuFramePrep: the colour planes' buffer",
                        round_up4(last)));
    // Bound from the first plane, rounded down to the alignment a binding's
    // offset needs (and to a word, which the kernel reads), so the limits
    // below weigh the picture rather than where it sits in its buffer: a
    // decoder's ring slot far into one buffer, say.
    const VkDeviceSize align = std::max<VkDeviceSize>(offset_alignment, 4);
    out.bind_offset = first / align * align;
    out.bind_bytes = round_up4(last) - out.bind_offset;
    out.y_offset = image.offset[0] - out.bind_offset;
    out.cb_offset = image.offset[1] - out.bind_offset;
    out.cr_offset =
        (nv12 ? image.offset[1] + 1 : image.offset[2]) - out.bind_offset;
    out.y_stride = image.stride[0];
    out.cb_stride = image.stride[1];
    out.cr_stride = nv12 ? image.stride[1] : image.stride[2];
  }
  // The kernel addresses the planes in 32-bit bytes from the binding.
  if (out.bind_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour planes span more than 4 GiB");
  }
  VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the colour buffer",
                                           out.out_bytes, max_range));
  VKC_TRY(core::check_storage_buffer_range("GpuFramePrep: the colour planes",
                                           out.bind_bytes, max_range));
  return out;
}

// An output's queue families: no more than the config's array holds, each one
// the device has.
core::Status check_queue_families(const core::Device& device,
                                  const std::uint32_t* families,
                                  std::uint32_t count, const char* who) {
  VKC_TRY(core::check_queue_family_count(count, who));
  const std::size_t device_families = device.caps().queue_families().size();
  for (std::uint32_t i = 0; i < count; ++i) {
    if (families[i] >= device_families) {
      return core::Status::invalid_argument(
          std::string(who) + ": queue family " + std::to_string(families[i]) +
          ", which the device does not have (" +
          std::to_string(device_families) + " families)");
    }
  }
  return {};
}

}  // namespace

core::Result<GpuFramePrep> GpuFramePrep::create(
    core::Device& device, core::Allocator& allocator,
    const GpuFramePrepConfig& config) {
  VKC_TRY(check_device_requirements(device, "GpuFramePrep::create"));
  // Here rather than at the first output, where it would surface as a buffer
  // failure on the first frame, naming neither the pass nor the output -- or,
  // for colour, not until a frame carries some.
  VKC_TRY(check_queue_families(device, config.color_queue_families,
                               config.color_queue_family_count,
                               "GpuFramePrep::create, colour"));
  VKC_TRY(check_queue_families(device, config.depth_queue_families,
                               config.depth_queue_family_count,
                               "GpuFramePrep::create, depth"));
  GpuFramePrep prep;
  prep.device_ = &device;
  prep.allocator_ = &allocator;
  prep.config_ = config;

  // Two storage bindings each, input then output, and the kernel's params as
  // push constants.
  VkPushConstantRange depth_push{};
  depth_push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  depth_push.size = sizeof(DepthParams);
  VkPushConstantRange color_push = depth_push;
  color_push.size = sizeof(ColorParams);
  core::KernelSetBuilder kb(device);
  // Depth: raw in, depth out, and the overlap it masks by.
  VKC_TRY(kb.add(prep.depth_kernel_, "undistort_depth",
                 vr_undistort_depth_comp_spv, vr_undistort_depth_comp_spv_size,
                 3, &depth_push));
  VKC_TRY(kb.add(prep.color_kernel_, "undistort_color",
                 vr_undistort_color_comp_spv, vr_undistort_color_comp_spv_size,
                 2, &color_push));
  VKC_ASSIGN(prep.pool_, kb.build());

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  prep.max_workgroup_count_x_ = props.limits.maxComputeWorkGroupCount[0];
  prep.max_storage_buffer_range_ = props.limits.maxStorageBufferRange;
  prep.min_storage_buffer_offset_alignment_ =
      props.limits.minStorageBufferOffsetAlignment;
  VKC_ASSIGN(prep.gpu_timer_, core::GpuTimer::create(device));
  // Always bound, so the set is complete whether or not the mask runs; the
  // kernel reads it only when it does.
  VKC_ASSIGN(prep.overlap_,
             core::device_storage_buffer(allocator, sizeof(OverlapParams)));
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(prep.overlap_.handle()),
                         "sensor.depth_overlap");
  prep.depth_kernel_.set.write_storage_buffer(2, prep.overlap_.handle(), 0,
                                              VK_WHOLE_SIZE);
  return prep;
}

// A checked frame: its two halves' layouts, where its colour comes from, and
// its bytes in the slot.
struct GpuFramePrep::Layout {
  DepthLayout depth;
  ColorLayout color;
  // NV12 plane images, copied into color_in_; packed host words, copied into
  // the slot after the depth; else colour is read in place.
  bool from_images = false;
  bool from_host = false;
  // The slot holds the depth from byte 0, and packed colour from this
  // offset, at minStorageBufferOffsetAlignment; its bytes used.
  VkDeviceSize color_at = 0;
  VkDeviceSize slot_bytes = 0;
};

// An output the fusion tiers read, and its lease: held while a DeviceFrame
// shares the buffer. The frame's deleter clears it with release order and
// ensure_output reads it with acquire order, so whatever a holder did with
// the buffer on any thread happens before the pass writes it again.
struct GpuFramePrep::Output {
  core::Buffer buffer;
  std::atomic<bool> held{false};
};

core::Result<GpuFramePrep::Layout> GpuFramePrep::check(
    const RgbdFrame& frame) const {
  if (!valid()) {
    return core::Status::invalid_argument("GpuFramePrep: moved-from pass");
  }
  const std::uint64_t max_pixels =
      std::uint64_t{max_workgroup_count_x_} * kLocalSize;
  Layout layout;
  VKC_ASSIGN(layout.depth,
             check_depth(frame, max_pixels, max_storage_buffer_range_));
  if (frame.has_color()) {
    VKC_ASSIGN(layout.color,
               check_color(frame, max_pixels, max_storage_buffer_range_,
                           min_storage_buffer_offset_alignment_));
  }
  layout.from_images =
      frame.color.image[0] != nullptr || frame.color.image[1] != nullptr;
  layout.from_host = frame.color_packed != nullptr;
  // Device planes come from a family the batch can take them over from, so
  // the acquire refuses nothing once recording starts.
  if (frame.color.device != nullptr) {
    const std::uint32_t family = frame.color.queue_family;
    if (family != kQueueFamilyIgnored && family != kQueueFamilyExternal &&
        family >= device_->caps().queue_families().size()) {
      return core::Status::invalid_argument(
          "GpuFramePrep: the colour planes come from queue family " +
          std::to_string(family) + ", which the device does not have");
    }
  }
  layout.slot_bytes = layout.depth.in_bytes;
  if (layout.from_host) {
    layout.color_at = round_up(
        layout.depth.in_bytes,
        std::max<VkDeviceSize>(min_storage_buffer_offset_alignment_, 4));
    layout.slot_bytes = layout.color_at + layout.color.bind_bytes;
  }
  return layout;
}

core::Result<core::HostSlot> GpuFramePrep::stage(const RgbdFrame& frame,
                                                 const Layout& layout) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  // One slot: the frame is copied in as it is recorded, so the batch of the
  // pass's last record, which binds the slot, has resolved by then (its
  // kernels' sets are rewritten too). A record before it has refuses here,
  // before ensure_buffers can free a buffer that batch still uses.
  if (inputs_ == nullptr) {
    core::HostRingDesc desc;
    desc.slot_bytes = layout.slot_bytes;
    desc.slots = 1;
    desc.name = "sensor.raw_frame";
    VKC_ASSIGN(inputs_, core::HostRing::create(*device_, *allocator_, desc));
  }
  VKC_ASSIGN(core::HostSlot slot, inputs_->claim(layout.slot_bytes));
  // The depth, then any packed colour at its aligned offset.
  std::memcpy(slot.data(), frame.depth,
              std::size_t{depth.pixels} * sizeof(std::uint16_t));
  if (layout.from_host) {
    std::memcpy(slot.data() + layout.color_at, frame.color_packed,
                static_cast<std::size_t>(color.bind_bytes));
  }
  return slot;
}

core::Status GpuFramePrep::ensure_buffers(const RgbdFrame& frame,
                                          const Layout& layout) {
  VKC_TRY(ensure_output(depth_out_, layout.depth.out_bytes,
                        "sensor.depth_frame", config_.depth_queue_families,
                        config_.depth_queue_family_count));
  if (!frame.has_color()) return {};
  if (layout.from_images) {
    VKC_TRY(core::ensure_device_scratch(
        *device_, *allocator_, color_in_, layout.color.bind_bytes,
        max_storage_buffer_range_, "sensor.raw_color"));
  }
  return ensure_output(color_out_, layout.color.out_bytes, "sensor.color_frame",
                       config_.color_queue_families,
                       config_.color_queue_family_count);
}

core::Status GpuFramePrep::record_inputs(
    core::CommandBatch& batch, const RgbdFrame& frame, const Layout& layout,
    const core::HostSlot& slot, core::GpuStageScope* stage, VkBuffer* input) {
  const ColorLayout& color = layout.color;
  const YuvImage& image = frame.color;
  // Device planes taken over from the family that wrote them.
  if (image.device != nullptr) {
    VKC_TRY(batch.acquire(*image.device, image.queue_family));
  }
  // The slot's used range: bound in place, or one copy into its mirror, which
  // joins the open tile, the frame prep's.
  VKC_ASSIGN(*input, batch.input(slot, 0, layout.slot_bytes));
  if (layout.from_images) {
    VKC_TRY(batch.copy(*image.image[0], image.width, image.height, color_in_, 0,
                       stage));
    VKC_TRY(batch.copy(*image.image[1], (image.width + 1) / 2, color.ch,
                       color_in_, color.cb_offset, stage));
  }
  // The overlap mask, for a frame with colour when the config asks: the
  // colour camera and the depth-to-colour transform, written inline.
  if (config_.depth_within_color && frame.has_color()) {
    const OverlapParams overlap{lens_params(frame.color_camera),
                                Mat4f(frame.depth_to_color)};
    VKC_TRY(batch.upload(overlap_, 0, &overlap, sizeof(overlap)));
  }
  return {};
}

core::Status GpuFramePrep::record_passes(core::CommandBatch& batch,
                                         const RgbdFrame& frame,
                                         const Layout& layout, VkBuffer input,
                                         core::GpuStageScope* stage) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  const YuvImage& image = frame.color;
  depth_kernel_.set.write_storage_buffer(0, input, 0, depth.in_bytes);
  depth_kernel_.set.write_storage_buffer(1, depth_out_->buffer.handle(), 0,
                                         depth.out_bytes);
  const bool within_color = config_.depth_within_color && frame.has_color();
  const DepthParams depth_params{lens_params(frame.depth_camera),
                                 frame.metres_per_unit, within_color ? 1u : 0u};
  VKC_TRY(batch.dispatch(depth_kernel_, &depth_params, sizeof(depth_params),
                         core::group_count(depth.pixels, kLocalSize),
                         max_workgroup_count_x_, stage));
  if (frame.has_color()) {
    // Packed words where the slot holds them, plane images where they were
    // copied, planes in a buffer where the decoder left them.
    const VkBuffer planes = layout.from_host     ? input
                            : layout.from_images ? color_in_.handle()
                                                 : image.device->handle();
    const VkDeviceSize at =
        layout.from_host ? layout.color_at : color.bind_offset;
    color_kernel_.set.write_storage_buffer(0, planes, at, color.bind_bytes);
    color_kernel_.set.write_storage_buffer(1, color_out_->buffer.handle(), 0,
                                           color.out_bytes);
    const auto u32 = [](VkDeviceSize v) {
      return static_cast<std::uint32_t>(v);
    };
    const auto chroma = chroma_offset(image.chroma_location);
    const ColorParams color_params{lens_params(frame.color_camera),
                                   u32(color.y_offset),
                                   u32(color.cb_offset),
                                   u32(color.cr_offset),
                                   u32(color.y_stride),
                                   u32(color.cb_stride),
                                   u32(color.cr_stride),
                                   u32(color.c_step),
                                   image.kr,
                                   image.kb,
                                   image.full_range ? 1u : 0u,
                                   chroma[0],
                                   chroma[1],
                                   layout.from_host ? 1u : 0u};
    VKC_TRY(batch.dispatch(color_kernel_, &color_params, sizeof(color_params),
                           core::group_count(color.pixels, kLocalSize),
                           max_workgroup_count_x_, stage));
  }
  return {};
}

void GpuFramePrep::abandon(const RgbdFrame& frame) {
  // The device may still run the copies and the kernels, so the device
  // planes, buffer or images, which the caller may drop, and a decoder reuse,
  // as soon as the call returns, are let go rather than freed. The slot stays
  // with the batch, which the device keeps.
  if (frame.color.device != nullptr) {
    static_cast<void>(
        new std::shared_ptr<const core::Buffer>(frame.color.device));
  }
  for (const std::shared_ptr<const core::Image>& plane : frame.color.image) {
    if (plane != nullptr) {
      static_cast<void>(new std::shared_ptr<const core::Image>(plane));
    }
  }
}

DeviceFrame GpuFramePrep::finish(const RgbdFrame& frame) const {
  // An output's buffer for the frame, its lease held until every copy goes.
  const auto lease = [](const std::shared_ptr<Output>& output) {
    output->held.store(true, std::memory_order_relaxed);
    return std::shared_ptr<const core::Buffer>(
        &output->buffer, [keep = output](const core::Buffer*) {
          keep->held.store(false, std::memory_order_release);
        });
  };
  // The pinhole cameras of the undistorted images, narrowed as the passes
  // read them, each posed: the depth camera through the sensor's extrinsic.
  const LensParams d = lens_params(frame.depth_camera);
  DeviceFrame out;
  out.depth = lease(depth_out_);
  out.depth_camera.fx = d.fx;
  out.depth_camera.fy = d.fy;
  out.depth_camera.cx = d.cx;
  out.depth_camera.cy = d.cy;
  out.depth_camera.min_depth = frame.min_depth;
  out.depth_camera.max_depth = frame.max_depth;
  out.depth_camera.width = d.width;
  out.depth_camera.height = d.height;
  out.depth_camera.cam_to_world =
      Mat4f(frame.color_to_world * frame.depth_to_color);
  if (frame.has_color()) {
    const LensParams c = lens_params(frame.color_camera);
    out.color = lease(color_out_);
    out.color_camera = ColorCameraParams{
        c.fx, c.fy, c.cx, c.cy, c.width, c.height, Mat4f(frame.color_to_world)};
    out.color_encoding = frame.color_encoding;
  }
  out.timestamp_ns = frame.timestamp_ns;
  return out;
}

namespace {

// A set's frames as pointers, a missing one null; refused with fewer passes
// than frames, naming `op`.
template <typename Pass>
core::Result<std::vector<const RgbdFrame*>> listed_frames(
    const char* op, const std::vector<Pass>& preps,
    const std::vector<std::optional<RgbdFrame>>& frames) {
  if (preps.size() < frames.size()) {
    return core::Status::invalid_argument(
        std::string("GpuFramePrep::") + op + ": " +
        std::to_string(frames.size()) + " frames for " +
        std::to_string(preps.size()) + " passes");
  }
  std::vector<const RgbdFrame*> listed(frames.size(), nullptr);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames[i]) listed[i] = &*frames[i];
  }
  return listed;
}

}  // namespace

core::Result<DeviceFrame> GpuFramePrep::record(
    core::CommandBatch& batch, const RgbdFrame& frame,
    core::StageMetrics* metrics) try {
  // A set of one, through record_set's path.
  const RgbdFrame* const frames[] = {&frame};
  Layout layout;
  VKC_ASSIGN(GpuFramePrep* const first, check_frames(this, frames, &layout, 1));
  VKC_TRY(gpu_timer_.reserve(*device_, kSpansPerFrame));
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  std::optional<DeviceFrame> out;
  VKC_TRY(record_frames(batch, this, frames, &layout, &out, 1, &stage));
  return std::move(*out);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("GpuFramePrep: out of host memory");
}

core::Result<std::vector<std::optional<DeviceFrame>>> GpuFramePrep::record_set(
    core::CommandBatch& batch, std::vector<GpuFramePrep>& preps,
    const std::vector<std::optional<RgbdFrame>>& frames,
    core::StageMetrics* metrics) try {
  VKC_ASSIGN(const std::vector<const RgbdFrame*> listed,
             listed_frames("record_set", preps, frames));
  std::vector<Layout> layouts(frames.size());
  std::vector<std::optional<DeviceFrame>> out(frames.size());
  VKC_ASSIGN(
      GpuFramePrep* const first,
      check_frames(preps.data(), listed.data(), layouts.data(), frames.size()));
  if (first == nullptr) return out;
  // The set's timed commands all in one window, which grows with the array.
  VKC_TRY(first->gpu_timer_.reserve(
      *first->device_,
      kSpansPerFrame * static_cast<std::uint32_t>(frames.size())));
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  VKC_TRY(record_frames(batch, preps.data(), listed.data(), layouts.data(),
                        out.data(), frames.size(), &stage));
  return out;
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("GpuFramePrep: out of host memory");
}

core::Result<DeviceFrame> GpuFramePrep::prepare(const RgbdFrame& frame,
                                                core::StageMetrics* metrics) {
  // A set of one, through prepare_batch's path.
  const RgbdFrame* const frames[] = {&frame};
  Layout layout;
  std::optional<DeviceFrame> out;
  VKC_TRY(prepare_frames(this, frames, &layout, &out, 1, metrics));
  return std::move(*out);
}

core::Result<std::vector<std::optional<DeviceFrame>>>
GpuFramePrep::prepare_batch(std::vector<GpuFramePrep>& preps,
                            const std::vector<std::optional<RgbdFrame>>& frames,
                            core::StageMetrics* metrics) {
  VKC_ASSIGN(const std::vector<const RgbdFrame*> listed,
             listed_frames("prepare_batch", preps, frames));
  std::vector<Layout> layouts(frames.size());
  std::vector<std::optional<DeviceFrame>> out(frames.size());
  VKC_TRY(prepare_frames(preps.data(), listed.data(), layouts.data(),
                         out.data(), frames.size(), metrics));
  return out;
}

core::Result<GpuFramePrep*> GpuFramePrep::check_frames(
    GpuFramePrep* preps, const RgbdFrame* const* frames, Layout* layouts,
    std::size_t count) {
  // Every frame checked before any is recorded, so a refused set costs no
  // work and times no row; and every pass on the first one's device, which
  // runs the batch.
  GpuFramePrep* first = nullptr;
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_ASSIGN(layouts[i], preps[i].check(*frames[i]));
    if (first == nullptr) {
      first = &preps[i];
    } else if (preps[i].device_ != first->device_) {
      return core::Status::invalid_argument(
          "GpuFramePrep::prepare_batch: the passes are on different devices");
    }
  }
  return first;
}

core::Status GpuFramePrep::record_frames(core::CommandBatch& batch,
                                         GpuFramePrep* preps,
                                         const RgbdFrame* const* frames,
                                         const Layout* layouts,
                                         std::optional<DeviceFrame>* out,
                                         std::size_t count,
                                         core::GpuStageScope* stage) {
  // Every frame's slot claimed and filled before anything is recorded or
  // resized, so a slot held refuses with the batch and every pass as they
  // were; a slot claimed before a refusal goes back free. A claimed slot
  // says its pass's last batch has resolved, so then its buffers may grow.
  std::vector<core::HostSlot> slots(count);
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_ASSIGN(slots[i], preps[i].stage(*frames[i], layouts[i]));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_TRY(preps[i].ensure_buffers(*frames[i], layouts[i]));
  }
  // Every camera's inputs, then every camera's passes: the copies write
  // buffers of their own, so they run with no barrier between them.
  std::vector<VkBuffer> inputs(count, VK_NULL_HANDLE);
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_TRY(preps[i].record_inputs(batch, *frames[i], layouts[i], slots[i],
                                   stage, &inputs[i]));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_TRY(preps[i].record_passes(batch, *frames[i], layouts[i], inputs[i],
                                   stage));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) out[i] = preps[i].finish(*frames[i]);
  }
  return {};
}

core::Status GpuFramePrep::prepare_frames(GpuFramePrep* preps,
                                          const RgbdFrame* const* frames,
                                          Layout* layouts,
                                          std::optional<DeviceFrame>* out,
                                          std::size_t count,
                                          core::StageMetrics* metrics) try {
  VKC_ASSIGN(GpuFramePrep* const first,
             check_frames(preps, frames, layouts, count));
  if (first == nullptr) return {};
  VKC_TRY(first->gpu_timer_.reserve(
      *first->device_, kSpansPerFrame * static_cast<std::uint32_t>(count)));
  // The row spans the wait too, and takes the batch's device half after it.
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  core::CommandBatch batch(*first->device_, *first->allocator_);
  VKC_TRY(record_frames(batch, preps, frames, layouts, out, count, &stage));
  core::Result<core::PendingBatch> pending = batch.submit_async({}, {});
  const core::Status ran = pending.ok() ? pending->wait() : pending.status();
  if (!ran.ok()) {
    // A refused submit ran nothing; a failed wait may leave the device
    // running the batch, which reads the planes.
    if (pending.ok() && pending->in_flight()) {
      for (std::size_t i = 0; i < count; ++i) {
        if (frames[i] != nullptr) abandon(*frames[i]);
      }
    }
    for (std::size_t i = 0; i < count; ++i) out[i].reset();
    return ran;
  }
  if (metrics != nullptr) metrics->merge_device(pending->device_rows());
  return {};
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("GpuFramePrep: out of host memory");
}

core::Status GpuFramePrep::ensure_output(std::shared_ptr<Output>& output,
                                         VkDeviceSize bytes, const char* name,
                                         const std::uint32_t* families,
                                         std::uint32_t family_count) {
  // Reused only once no DeviceFrame holds its lease: a frame kept past this
  // call keeps its contents, and this frame goes to a new buffer, which
  // measured no slower than reusing one (the residency decision's step 5b),
  // so there is no ring.
  if (output != nullptr && !output->held.load(std::memory_order_acquire) &&
      output->buffer.size() >= bytes) {
    return {};
  }
  // Device-local: only the kernels touch it, and on a discrete GPU the
  // fusion kernels' reads would otherwise cross the bus. Shared with the
  // families the config names for it, for a consumer on another queue, with
  // no usage beyond the pass's storage and transfer usage. The host's half
  // first, so running out of host memory leaves no buffer made.
  auto made = std::make_shared<Output>();
  VKC_ASSIGN(made->buffer, core::device_storage_buffer(*allocator_, bytes, 0,
                                                       families, family_count));
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           core::debug_object_handle(made->buffer.handle()),
                           name);
  output = std::move(made);
  return {};
}

}  // namespace volumetric_kit::recon::sensor
