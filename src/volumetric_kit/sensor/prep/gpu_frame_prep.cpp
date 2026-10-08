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
#include <string>
#include <utility>

#include "undistort_color_comp.spv.hpp"
#include "undistort_depth_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace volumetric_kit::recon::sensor {
namespace {

// Both kernels' local_size_x.
constexpr std::uint32_t kLocalSize = 256;

// The most commands record_uploads and record_passes time for one frame: the
// depth copy and two plane-image copies (or one packed-colour copy), and the
// two passes.
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

// A buffer of at least `bytes`, kept when it is big enough: a device-local
// input, filled through the pass's batch so the kernels never read the raw
// frame across the bus, or the system-memory staging the frame is written
// into. The staging is kept too: a batch stages through a buffer of its own
// per call, and four passes doing that at once with 4K frames had VMA
// allocate and free a block for every set.
core::Status ensure_buffer(const core::Device& device,
                           core::Allocator& allocator, core::Buffer& buffer,
                           VkDeviceSize bytes, bool staging, const char* name) {
  if (buffer.valid() && buffer.size() >= bytes) return {};
  buffer = core::Buffer();
  if (staging) {
    VKC_ASSIGN(buffer,
               allocator.create_buffer({bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                        core::MemoryUsage::Staging,
                                        core::HostAccess::SequentialWrite}));
  } else {
    VKC_ASSIGN(buffer, core::device_storage_buffer(allocator, bytes));
  }
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(buffer.handle()), name);
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

// Whether `rows` rows of `row_bytes`, `stride` apart from `offset`, fit in
// `size` bytes, worked out without overflowing 64 bits. `stride` is at least
// `row_bytes`, which is at least 1.
bool plane_fits(std::uint64_t offset, std::uint64_t stride, std::uint64_t rows,
                std::uint64_t row_bytes, std::uint64_t size) noexcept {
  if (offset > size || row_bytes > size - offset) return false;
  return rows <= 1 || rows - 1 <= (size - offset - row_bytes) / stride;
}

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

// NV12's two planes as images the pass can copy: R8 luma and R8G8 chroma,
// each at least the picture's size, as CommandBatch::copy checks them too.
core::Status check_images(const YuvImage& image) {
  if (image.layout != YuvLayout::Nv12 || image.image[0] == nullptr ||
      image.image[1] == nullptr) {
    return core::Status::invalid_argument(
        "GpuFramePrep: colour planes as images are NV12's two");
  }
  const core::Image& y = *image.image[0];
  const core::Image& c = *image.image[1];
  const std::uint32_t cw = image.width / 2 + image.width % 2;
  const std::uint32_t ch = image.height / 2 + image.height % 2;
  if (y.format() != VK_FORMAT_R8_UNORM || c.format() != VK_FORMAT_R8G8_UNORM ||
      y.width() < image.width || y.height() < image.height || c.width() < cw ||
      c.height() < ch) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour images must be R8 luma and R8G8 chroma, at "
        "least the picture's size");
  }
  if ((y.usage() & c.usage() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour images need TRANSFER_SRC usage");
  }
  const auto copyable = [](const core::Image& i) {
    return i.layout() == VK_IMAGE_LAYOUT_GENERAL ||
           i.layout() == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  };
  if (!copyable(y) || !copyable(c)) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour images must be in GENERAL or "
        "TRANSFER_SRC_OPTIMAL");
  }
  return {};
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
  if (on_device && as_images) {
    return core::Status::invalid_argument(
        "GpuFramePrep: the colour planes must be in a buffer or images, not "
        "both");
  }
  if (as_images) VKC_TRY(check_images(image));
  const int planes = nv12 ? 2 : 3;
  const std::uint64_t cw = (std::uint64_t{image.width} + 1) / 2;
  const std::uint64_t ch = (std::uint64_t{image.height} + 1) / 2;
  const std::uint64_t c_row = nv12 ? 2 * cw : cw;
  const std::uint64_t rows[3] = {image.height, ch, ch};
  const std::uint64_t row_bytes[3] = {image.width, c_row, c_row};
  for (int p = 0; p < planes && on_device; ++p) {
    if (image.stride[p] < row_bytes[p]) {
      return core::Status::invalid_argument(
          "GpuFramePrep: a colour plane's stride is shorter than its rows");
    }
  }

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
    const core::Buffer& buffer = *image.device;
    const core::StorageInput bound(buffer);
    // Empty or without storage usage, before its size is weighed.
    VKC_TRY(bound.check("GpuFramePrep: the colour planes' buffer", 0));
    std::uint64_t begin[3] = {}, end[3] = {};
    for (int p = 0; p < planes; ++p) {
      if (!plane_fits(image.offset[p], image.stride[p], rows[p], row_bytes[p],
                      buffer.size())) {
        return core::Status::invalid_argument(
            "GpuFramePrep: a colour plane runs past its buffer");
      }
      begin[p] = image.offset[p];
      end[p] = image.offset[p] + (rows[p] - 1) * image.stride[p] + row_bytes[p];
    }
    // Planes sharing bytes would read one as another, as I420 does with its
    // offsets left at zero.
    for (int p = 0; p < planes; ++p) {
      for (int q = p + 1; q < planes; ++q) {
        if (begin[p] < end[q] && begin[q] < end[p]) {
          return core::Status::invalid_argument(
              "GpuFramePrep: the colour planes overlap");
        }
      }
    }
    const std::uint64_t first = *std::min_element(begin, begin + planes);
    const std::uint64_t last = *std::max_element(end, end + planes);
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

}  // namespace

core::Result<GpuFramePrep> GpuFramePrep::create(
    core::Device& device, core::Allocator& allocator,
    const GpuFramePrepConfig& config) {
  VKC_TRY(check_device_requirements(device, "GpuFramePrep::create"));
  // Here rather than at the first output, where it would surface as a buffer
  // failure on the first frame.
  VKC_TRY(core::check_queue_family_count(config.color_queue_family_count,
                                         "GpuFramePrep::create"));
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

// A checked frame: its two halves' layouts, and where its colour comes from.
struct GpuFramePrep::Layout {
  DepthLayout depth;
  ColorLayout color;
  // NV12 plane images, or packed host words staged after the depth, both
  // copied into color_in_; else colour is read in place.
  bool from_images = false;
  bool from_host = false;
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
  return layout;
}

core::Status GpuFramePrep::acquire(core::CommandBatch& batch,
                                   const RgbdFrame& frame) {
  // Device planes taken over from the family that wrote them, first, so a
  // family the device lacks is refused before any work.
  if (frame.color.device != nullptr) {
    VKC_TRY(batch.acquire(*frame.color.device, frame.color.queue_family));
  }
  return {};
}

core::Status GpuFramePrep::stage_host(const RgbdFrame& frame,
                                      const Layout& layout) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  VKC_TRY(ensure_buffer(*device_, *allocator_, depth_in_, depth.in_bytes, false,
                        "sensor.raw_depth"));
  VKC_TRY(ensure_output(depth_out_, depth.out_bytes, "sensor.depth_frame",
                        /*color=*/false));
  if (frame.has_color()) {
    if (layout.from_images || layout.from_host) {
      VKC_TRY(ensure_buffer(*device_, *allocator_, color_in_, color.bind_bytes,
                            false, "sensor.raw_color"));
    }
    VKC_TRY(ensure_output(color_out_, color.out_bytes, "sensor.color_frame",
                          /*color=*/true));
  }

  // The depth, then any packed colour from the word after it.
  const VkDeviceSize color_bytes = layout.from_host ? color.bind_bytes : 0;
  VKC_TRY(ensure_buffer(*device_, *allocator_, staging_,
                        depth.in_bytes + color_bytes, true,
                        "sensor.raw_staging"));
  auto* staged = static_cast<std::uint8_t*>(staging_.mapped());
  std::memcpy(staged, frame.depth,
              std::size_t{depth.pixels} * sizeof(std::uint16_t));
  if (layout.from_host) {
    std::memcpy(staged + depth.in_bytes, frame.color_packed,
                static_cast<std::size_t>(color_bytes));
  }
  return {};
}

core::Status GpuFramePrep::record_uploads(core::CommandBatch& batch,
                                          const RgbdFrame& frame,
                                          const Layout& layout,
                                          core::GpuStageScope* stage) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  const YuvImage& image = frame.color;
  // The copies are timed with the passes, so the row's device half counts
  // moving the frame too.
  const VkDeviceSize depth_bytes =
      VkDeviceSize{depth.pixels} * sizeof(std::uint16_t);
  VKC_TRY(batch.copy(staging_, 0, depth_in_, 0, depth_bytes, stage));
  if (layout.from_host) {
    VKC_TRY(batch.copy(staging_, depth.in_bytes, color_in_, 0, color.bind_bytes,
                       stage));
  }
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
                                         const Layout& layout,
                                         core::GpuStageScope* stage) {
  const DepthLayout& depth = layout.depth;
  const ColorLayout& color = layout.color;
  const YuvImage& image = frame.color;
  depth_kernel_.set.write_storage_buffer(0, depth_in_.handle(), 0,
                                         depth.in_bytes);
  depth_kernel_.set.write_storage_buffer(1, depth_out_->handle(), 0,
                                         depth.out_bytes);
  const bool within_color = config_.depth_within_color && frame.has_color();
  const DepthParams depth_params{lens_params(frame.depth_camera),
                                 frame.metres_per_unit, within_color ? 1u : 0u};
  VKC_TRY(batch.dispatch(depth_kernel_, &depth_params, sizeof(depth_params),
                         core::group_count(depth.pixels, kLocalSize),
                         max_workgroup_count_x_, stage));
  if (frame.has_color()) {
    const bool in_input = layout.from_images || layout.from_host;
    color_kernel_.set.write_storage_buffer(
        0, in_input ? color_in_.handle() : image.device->handle(),
        color.bind_offset, color.bind_bytes);
    color_kernel_.set.write_storage_buffer(1, color_out_->handle(), 0,
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

void GpuFramePrep::abandon(const RgbdFrame& frame, const Layout& layout) {
  // A failed wait may leave the copy and the kernels running, so the staging
  // is let go rather than rewritten or freed, as the batch lets go of its
  // own; and so are the device planes, buffer or images, which the caller may
  // drop, and a decoder reuse, as soon as the call returns.
  static_cast<void>(new core::Buffer(std::move(staging_)));
  if (frame.color.device != nullptr) {
    static_cast<void>(
        new std::shared_ptr<const core::Buffer>(frame.color.device));
  }
  if (layout.from_images) {
    static_cast<void>(
        new std::shared_ptr<const core::Image>(frame.color.image[0]));
    static_cast<void>(
        new std::shared_ptr<const core::Image>(frame.color.image[1]));
  }
}

DeviceFrame GpuFramePrep::finish(const RgbdFrame& frame) const {
  // The pinhole cameras of the undistorted images, narrowed as the passes
  // read them, each posed: the depth camera through the sensor's extrinsic.
  const LensParams d = lens_params(frame.depth_camera);
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
  out.depth_camera.cam_to_world =
      Mat4f(frame.color_to_world * frame.depth_to_color);
  if (frame.has_color()) {
    const LensParams c = lens_params(frame.color_camera);
    out.color = color_out_;
    out.color_camera = ColorCameraParams{
        c.fx, c.fy, c.cx, c.cy, c.width, c.height, Mat4f(frame.color_to_world)};
    out.color_encoding = frame.color_encoding;
  }
  out.timestamp_ns = frame.timestamp_ns;
  return out;
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
  if (preps.size() < frames.size()) {
    return core::Status::invalid_argument(
        "GpuFramePrep::prepare_batch: " + std::to_string(frames.size()) +
        " frames for " + std::to_string(preps.size()) + " passes");
  }
  std::vector<const RgbdFrame*> listed(frames.size(), nullptr);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames[i]) listed[i] = &*frames[i];
  }
  std::vector<Layout> layouts(frames.size());
  std::vector<std::optional<DeviceFrame>> out(frames.size());
  VKC_TRY(prepare_frames(preps.data(), listed.data(), layouts.data(),
                         out.data(), frames.size(), metrics));
  return out;
}

core::Status GpuFramePrep::prepare_frames(GpuFramePrep* preps,
                                          const RgbdFrame* const* frames,
                                          Layout* layouts,
                                          std::optional<DeviceFrame>* out,
                                          std::size_t count,
                                          core::StageMetrics* metrics) {
  // Every frame checked before any is recorded, so a refused set costs no
  // work and times no row; and every pass on the first one's device, which
  // runs the batch.
  GpuFramePrep* first = nullptr;
  std::uint32_t present = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] == nullptr) continue;
    VKC_ASSIGN(layouts[i], preps[i].check(*frames[i]));
    if (first == nullptr) {
      first = &preps[i];
    } else if (preps[i].device_ != first->device_) {
      return core::Status::invalid_argument(
          "GpuFramePrep::prepare_batch: the passes are on different devices");
    }
    ++present;
  }
  if (first == nullptr) return {};

  // The set's timed commands all in one window, which grows with the array.
  VKC_TRY(first->gpu_timer_.reserve(*first->device_, kSpansPerFrame * present));
  core::GpuStageScope stage(metrics, first->gpu_timer_, "frame prep");
  core::CommandBatch batch(*first->device_, *first->allocator_);
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) VKC_TRY(preps[i].acquire(batch, *frames[i]));
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) {
      VKC_TRY(preps[i].stage_host(*frames[i], layouts[i]));
    }
  }
  // Every camera's uploads, then every camera's passes: the uploads write
  // buffers of their own, so they run with no barrier between them.
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) {
      VKC_TRY(preps[i].record_uploads(batch, *frames[i], layouts[i], &stage));
    }
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) {
      VKC_TRY(preps[i].record_passes(batch, *frames[i], layouts[i], &stage));
    }
  }
  // TODO: submit through the core's submit_async once the pipelined stages
  // land (DESIGN.md's Next work, step 4): the set then waits on the previous
  // stage's timeline rather than on the host, and a failure abandons the
  // passes only while its PendingBatch::in_flight() says the device may still
  // run them, where every failure abandons them now.
  const core::Status submitted = batch.submit();
  if (!submitted.ok()) {
    for (std::size_t i = 0; i < count; ++i) {
      if (frames[i] != nullptr) preps[i].abandon(*frames[i], layouts[i]);
    }
    return submitted;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (frames[i] != nullptr) out[i] = preps[i].finish(*frames[i]);
  }
  return {};
}

core::Status GpuFramePrep::ensure_output(std::shared_ptr<core::Buffer>& buffer,
                                         VkDeviceSize bytes, const char* name,
                                         bool color) {
  // Reused only when this pass holds the last reference: a DeviceFrame kept
  // past this call keeps its contents, and this frame goes to a new buffer,
  // which measured no slower than reusing one (the residency decision's step
  // 5b), so there is no ring.
  if (buffer != nullptr && buffer.use_count() == 1 && buffer->size() >= bytes) {
    // use_count() is a relaxed load. The fence orders this thread's next
    // write after whatever the thread that dropped the last other reference
    // did first, such as a renderer's fence wait on its copy from it.
    std::atomic_thread_fence(std::memory_order_acquire);
    return {};
  }
  // Device-local: only the kernels touch it, and on a discrete GPU the
  // fusion kernels' reads would otherwise cross the bus. The colour is shared
  // with the families the config names, for a consumer on another queue.
  VKC_ASSIGN(
      core::Buffer created,
      core::device_storage_buffer(
          *allocator_, bytes, 0, color ? config_.color_queue_families : nullptr,
          color ? config_.color_queue_family_count : 0));
  device_->set_object_name(VK_OBJECT_TYPE_BUFFER,
                           core::debug_object_handle(created.handle()), name);
  buffer = std::make_shared<core::Buffer>(std::move(created));
  return {};
}

}  // namespace volumetric_kit::recon::sensor
