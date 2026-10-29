// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include "../prep/color_pass.hpp"
#include "rectify_comp.spv.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace volumetric_kit::sensor {
namespace {

using camera::CameraModel;
using camera::Mat3d;
using camera::Mat4d;
using camera::Vec2d;
using camera::Vec3d;

constexpr const char* kWho = "StereoRectifier";

// rectify.comp's local_size_x (undistort_color.glsl's).
constexpr std::uint32_t kLocalSize = 256;

// The commands a frame times: the two passes.
constexpr std::uint32_t kSpans = 2;

// rectify.comp's View block, under scalar layout: a ray of the rectified
// camera in the source camera, the rectified camera and size, and the output.
struct ViewParams {
  float rect_to_source[9];  // column-major, as GLSL's mat3
  float fx, fy, cx, cy;
  std::uint32_t width, height;
  std::uint32_t luma_out;  // Output::Luma8: four pixels a word
};
static_assert(sizeof(ViewParams) == 64, "ViewParams must match rectify.comp");

core::Status refuse(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

VkDeviceSize round_up(VkDeviceSize bytes, VkDeviceSize alignment) noexcept {
  return (bytes + alignment - 1) / alignment * alignment;
}

// ---------------------------------------------------------------------------
// The rectification, on the host: OpenCV's stereoRectify (calib3d's
// stereo_geom.cpp, CALIB_ZERO_DISPARITY), step for step, in double.

// OpenCV's undistortPoints for one pixel of `c`, with its default criteria:
// normalized, five fixed-point iterations of the lens's inverse, then through
// `rr` (P's 3x3 times R) as a homography when given.
Vec2d undistort_point(const CameraModel& c, Vec2d pixel, const Mat3d* rr) {
  const camera::PinholeIntrinsics& k = c.intrinsics;
  const camera::RationalDistortion& d = c.distortion;
  const double x0 = (pixel.x - k.cx) / k.fx;
  const double y0 = (pixel.y - k.cy) / k.fy;
  double x = x0;
  double y = y0;
  for (int j = 0; j < 5; ++j) {
    const double r2 = x * x + y * y;
    const double icdist = (1 + ((d.k6 * r2 + d.k5) * r2 + d.k4) * r2) /
                          (1 + ((d.k3 * r2 + d.k2) * r2 + d.k1) * r2);
    if (icdist < 0) {  // OpenCV's regression_14583: past the lens's fold
      x = x0;
      y = y0;
      break;
    }
    const double dx = 2 * d.p1 * x * y + d.p2 * (r2 + 2 * x * x);
    const double dy = d.p1 * (r2 + 2 * y * y) + 2 * d.p2 * x * y;
    x = (x0 - dx) * icdist;
    y = (y0 - dy) * icdist;
  }
  if (rr == nullptr) return Vec2d(x, y);
  const Vec3d h = *rr * Vec3d(x, y, 1.0);
  return Vec2d(h.x / h.z, h.y / h.z);
}

// A rectangle, OpenCV's Rect_<double>.
struct Rect {
  double x = 0, y = 0, width = 0, height = 0;
};

// OpenCV's getUndistortRectangles: the border of a 9 x 9 grid over the image,
// undistorted and taken through `r` and `p` (a pinhole of focal length `f`
// and principal point `cc`), and the rectangles inside and around it.
void undistort_rectangles(const CameraModel& c, const Mat3d& r, double f,
                          Vec2d cc, Rect* inner, Rect* outer) {
  constexpr int kN = 9;
  Mat3d p(0.0);
  p[0][0] = f;
  p[1][1] = f;
  p[2][0] = cc.x;
  p[2][1] = cc.y;
  p[2][2] = 1.0;
  const Mat3d rr = p * r;
  const double step_x = (c.size.width - 1) / static_cast<double>(kN - 1);
  const double step_y = (c.size.height - 1) / static_cast<double>(kN - 1);
  double ix0 = -std::numeric_limits<float>::max();
  double ix1 = std::numeric_limits<float>::max();
  double iy0 = -std::numeric_limits<float>::max();
  double iy1 = std::numeric_limits<float>::max();
  double ox0 = std::numeric_limits<float>::max();
  double ox1 = -std::numeric_limits<float>::max();
  double oy0 = std::numeric_limits<float>::max();
  double oy1 = -std::numeric_limits<float>::max();
  for (int y = 0; y < kN; ++y) {
    for (int x = 0; x < kN; ++x) {
      if (x != 0 && x != kN - 1 && y != 0 && y != kN - 1) continue;
      const Vec2d q = undistort_point(c, Vec2d(x * step_x, y * step_y), &rr);
      ox0 = std::min(ox0, q.x);
      ox1 = std::max(ox1, q.x);
      oy0 = std::min(oy0, q.y);
      oy1 = std::max(oy1, q.y);
      if (x == 0) ix0 = std::max(ix0, q.x);
      if (x == kN - 1) ix1 = std::min(ix1, q.x);
      if (y == 0) iy0 = std::max(iy0, q.y);
      if (y == kN - 1) iy1 = std::min(iy1, q.y);
    }
  }
  *inner = Rect{ix0, iy0, ix1 - ix0, iy1 - iy0};
  *outer = Rect{ox0, oy0, ox1 - ox0, oy1 - oy0};
}

Mat4d rotation4(const Mat3d& r) {
  Mat4d m(r);
  m[3][3] = 1.0;
  return m;
}

// The rectification: the rig, and each view's rotation from its source camera
// into the rectified one (OpenCV's R1 and R2).
struct Rectification {
  StereoRig rig;
  Mat3d source_to_rect[2] = {Mat3d(1.0), Mat3d(1.0)};
};

// Whether the pair is rectified already: no rotation, no lens, one camera
// model, the right camera along +x alone, each within round-off.
bool already_rectified(const StereoCalibration& c) {
  constexpr double kEps = 1e-9;
  const Mat3d r(c.left_to_right);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (std::fabs(r[i][j] - (i == j ? 1.0 : 0.0)) > kEps) return false;
    }
  }
  const auto no_lens = [](const camera::RationalDistortion& d) {
    return d.k1 == 0 && d.k2 == 0 && d.p1 == 0 && d.p2 == 0 && d.k3 == 0 &&
           d.k4 == 0 && d.k5 == 0 && d.k6 == 0;
  };
  const camera::PinholeIntrinsics& a = c.left.intrinsics;
  const camera::PinholeIntrinsics& b = c.right.intrinsics;
  const Vec3d t(c.left_to_right[3]);
  return no_lens(c.left.distortion) && no_lens(c.right.distortion) &&
         std::fabs(a.fx - b.fx) <= kEps && std::fabs(a.fy - b.fy) <= kEps &&
         std::fabs(a.cx - b.cx) <= kEps && std::fabs(a.cy - b.cy) <= kEps &&
         std::fabs(t.y) <= kEps && std::fabs(t.z) <= kEps && t.x < 0;
}

core::Result<Rectification> rectify_pair(const StereoCalibration& c,
                                         camera::ImageSize size, double alpha) {
  const camera::ImageSize image = c.left.size;
  if (size.width == 0) size = image;
  const double nx = image.width;
  const double ny = image.height;
  Rectification out;
  out.rig.camera.size = size;

  if (already_rectified(c)) {
    // Resampled only: the camera scaled, edge to edge (pixel centres at
    // integers, so an image spans [-0.5, w - 0.5]).
    const double sx = size.width / nx;
    const double sy = size.height / ny;
    const camera::PinholeIntrinsics& k = c.left.intrinsics;
    out.rig.camera.intrinsics = {k.fx * sx, k.fy * sy, (k.cx + 0.5) * sx - 0.5,
                                 (k.cy + 0.5) * sy - 0.5};
    out.rig.baseline = -c.left_to_right[3][0];
    return out;
  }

  // The relative rotation split in half: each camera turned half-way.
  const Mat3d r(c.left_to_right);
  const Vec3d translation(c.left_to_right[3]);
  const Mat3d r_r = camera::rotation_from_rodrigues(
      -0.5 * camera::rodrigues_from_rotation(r));
  Vec3d t = r_r * translation;
  const int idx = std::fabs(t.x) > std::fabs(t.y) ? 0 : 1;
  const double cc = t[idx];
  const double nt = glm::length(t);
  if (!(nt > 0.0)) return refuse("the two cameras are at one place");
  if (idx != 0) {
    // TODO(sensor): a vertical pair, which OpenCV rectifies along y.
    return refuse(
        "the pair is vertical: the right camera must be along the "
        "left one's x axis");
  }
  if (cc > 0) {
    return refuse("the right camera is on the left one's -x side: swap them");
  }
  // Then the baseline turned onto -x of the right camera's offset, about the
  // axis that takes it there.
  Vec3d uu(0.0);
  uu[idx] = cc > 0 ? 1 : -1;
  Vec3d ww = glm::cross(t, uu);
  const double nw = glm::length(ww);
  if (nw > 0.0) ww *= std::acos(std::fabs(cc) / nt) / nw;
  const Mat3d w_r = camera::rotation_from_rodrigues(ww);
  const Mat3d r1 = w_r * glm::transpose(r_r);
  const Mat3d r2 = w_r * r_r;
  t = r2 * translation;

  // One focal length for both views: the vertical ones, scaled to the
  // rectified height.
  const double ratio_y = static_cast<double>(size.height) / ny / 2;
  double fc = (c.left.intrinsics.fy + c.right.intrinsics.fy) * ratio_y;
  // Each principal point: where the image's corners, undistorted and turned,
  // centre.
  Vec2d cc_new[2];
  for (int k = 0; k < 2; ++k) {
    const CameraModel& cam = k == 0 ? c.left : c.right;
    const Mat3d& rk = k == 0 ? r1 : r2;
    Vec2d sum(0.0);
    for (int i = 0; i < 4; ++i) {
      const Vec2d corner((i % 2) * (nx - 1), (i < 2 ? 0 : 1) * (ny - 1));
      const Vec2d n = undistort_point(cam, corner, nullptr);
      const Vec3d q = rk * Vec3d(n.x, n.y, 1.0);
      sum += Vec2d(fc * q.x / q.z, fc * q.y / q.z);
    }
    cc_new[k] = Vec2d((nx - 1) / 2, (ny - 1) / 2) - sum / 4.0;
  }
  // CALIB_ZERO_DISPARITY: one principal point for both.
  const Vec2d c0 = (cc_new[0] + cc_new[1]) * 0.5;

  // The scale alpha chooses: s0 shows only pixels both views see, s1 every
  // source pixel.
  Rect inner[2], outer[2];
  undistort_rectangles(c.left, r1, fc, c0, &inner[0], &outer[0]);
  undistort_rectangles(c.right, r2, fc, c0, &inner[1], &outer[1]);
  const double cx = size.width * c0.x / nx;
  const double cy = size.height * c0.y / ny;
  const double w1 = size.width - 1.0;
  const double h1 = size.height - 1.0;
  double s0 = -std::numeric_limits<double>::max();
  double s1 = std::numeric_limits<double>::max();
  for (int k = 0; k < 2; ++k) {
    const Rect& i = inner[k];
    const Rect& o = outer[k];
    s0 = std::max({s0, cx / (c0.x - i.x), cy / (c0.y - i.y),
                   (w1 - cx) / (i.x + i.width - c0.x),
                   (h1 - cy) / (i.y + i.height - c0.y)});
    s1 = std::min({s1, cx / (c0.x - o.x), cy / (c0.y - o.y),
                   (w1 - cx) / (o.x + o.width - c0.x),
                   (h1 - cy) / (o.y + o.height - c0.y)});
  }
  const double s = s0 * (1 - alpha) + s1 * alpha;
  fc *= s;
  if (!(std::isfinite(fc) && fc > 0 && std::isfinite(cx) &&
        std::isfinite(cy))) {
    return refuse(
        "the rectification is degenerate: the lenses fold or the "
        "cameras look too far apart");
  }
  out.rig.camera.intrinsics = {fc, fc, cx, cy};
  out.rig.baseline = -t.x;
  out.rig.rectified_to_left = rotation4(glm::transpose(r1));
  out.source_to_rect[0] = r1;
  out.source_to_rect[1] = r2;
  return out;
}

// The kernel's view of one rectification: the rectified ray into the source
// camera (the transpose of `source_to_rect`).
ViewParams view_params(const Mat3d& source_to_rect, const CameraModel& rect,
                       bool luma_out) {
  ViewParams v{};
  const Mat3d m = glm::transpose(source_to_rect);
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      v.rect_to_source[col * 3 + row] = static_cast<float>(m[col][row]);
    }
  }
  v.fx = static_cast<float>(rect.intrinsics.fx);
  v.fy = static_cast<float>(rect.intrinsics.fy);
  v.cx = static_cast<float>(rect.intrinsics.cx);
  v.cy = static_cast<float>(rect.intrinsics.cy);
  v.width = rect.size.width;
  v.height = rect.size.height;
  v.luma_out = luma_out ? 1u : 0u;
  return v;
}

core::Status check_pose(const char* what, const Mat4d& pose) {
  const core::Status rigid = camera::check_rigid(pose);
  if (!rigid.ok()) return refuse(std::string(what) + ": " + rigid.message());
  return {};
}

// A view's buffer and its lease: held while a pair shares the buffer. The
// pair's deleter clears it with release order and ensure_output reads it with
// acquire order, so whatever a holder did with the buffer happens before the
// pass writes it again.
struct ViewBuffer {
  core::ExportedBuffer view;
  std::atomic<bool> held{false};
  // Released to VK_QUEUE_FAMILY_EXTERNAL by the last pass that wrote it, so
  // the next one takes it back first.
  bool external = false;
};

}  // namespace

struct StereoRectifier::Impl {
  const core::Device* device = nullptr;
  core::Allocator* allocator = nullptr;
  StereoInfo info;
  Options options;
  Rectification rectification;
  SensorInfo depth_info;

  std::uint32_t max_workgroup_count_x = 0;
  VkDeviceSize max_storage_buffer_range = 0;
  VkDeviceSize offset_alignment = 4;
  // What a view's pass writes: its bytes, and the words its threads write.
  VkDeviceSize out_bytes = 0;
  std::uint32_t out_items = 0;
  // Whether the views are on exported memory, chosen at create.
  bool exported = false;

  // The left view binds the kernel's own set, the right one right_set[0].
  core::ComputeKernel kernel;
  core::DescriptorPool pool;
  core::KernelSets right_set;
  // Both views' ViewParams, at offsets a binding takes.
  core::Buffer views;
  core::GpuTimer timer;
  // Host pictures copied in, left then right; made at the first that has one.
  std::unique_ptr<core::HostRing> inputs;
  std::shared_ptr<ViewBuffer> out[2];
  // A submit or wait that failed: the device may still run that pass.
  core::Status failed;

  core::Status ensure_output(int view);
  core::Result<RectifiedPair> rectify(const StereoFrame& frame,
                                      core::StageMetrics* metrics);
};

StereoRectifier::StereoRectifier(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

StereoRectifier::~StereoRectifier() = default;

const StereoRig& StereoRectifier::rig() const noexcept {
  return impl_->rectification.rig;
}

const SensorInfo& StereoRectifier::depth_info() const noexcept {
  return impl_->depth_info;
}

core::Result<std::unique_ptr<StereoRectifier>> StereoRectifier::create(
    const core::Device& device, core::Allocator& allocator,
    const StereoInfo& info, const Options& options) try {
  VKC_TRY(check_device_requirements(device, "StereoRectifier::create"));
  const StereoCalibration& calibration = info.calibration;
  VKC_TRY(detail::check_camera(kWho, "left", calibration.left));
  VKC_TRY(detail::check_camera(kWho, "right", calibration.right));
  if (calibration.left.size.width != calibration.right.size.width ||
      calibration.left.size.height != calibration.right.size.height) {
    return refuse("the two cameras' sizes differ");
  }
  VKC_TRY(check_pose("left_to_right", calibration.left_to_right));
  if (info.color) VKC_TRY(detail::check_camera(kWho, "colour", *info.color));
  VKC_TRY(check_pose("left_to_color", info.left_to_color));
  if (!(std::isfinite(options.alpha) && options.alpha >= 0.0f &&
        options.alpha <= 1.0f)) {
    return refuse("alpha must be in [0, 1]");
  }
  if ((options.size.width == 0) != (options.size.height == 0)) {
    return refuse("the rectified size has one side 0");
  }
  if (static_cast<unsigned>(options.output) >
      static_cast<unsigned>(Output::Rgba8)) {
    return refuse("unknown output");
  }

  auto impl = std::make_unique<Impl>();
  impl->device = &device;
  impl->allocator = &allocator;
  impl->info = info;
  impl->options = options;
  VKC_ASSIGN(impl->rectification,
             rectify_pair(calibration, options.size, options.alpha));
  const StereoRig& rig = impl->rectification.rig;
  VKC_TRY(detail::check_camera(kWho, "rectified", rig.camera));

  // Depth in the rectified left camera; colour from the third camera, or the
  // left one's picture through its own lens.
  impl->depth_info.depth = rig.camera;
  impl->depth_info.color = info.color ? *info.color : calibration.left;
  impl->depth_info.depth_to_color =
      (info.color ? info.left_to_color : Mat4d(1.0)) * rig.rectified_to_left;

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(device.physical_device(), &props);
  impl->max_workgroup_count_x = props.limits.maxComputeWorkGroupCount[0];
  impl->max_storage_buffer_range = props.limits.maxStorageBufferRange;
  impl->offset_alignment =
      std::max<VkDeviceSize>(props.limits.minStorageBufferOffsetAlignment, 4);

  const bool luma = options.output == Output::Luma8;
  const std::uint64_t pixels =
      std::uint64_t{rig.camera.size.width} * rig.camera.size.height;
  const std::uint64_t items = luma ? (pixels + 3) / 4 : pixels;
  // The kernel counts pixels, and luma's first of four, in 32 bits.
  if (items > std::uint64_t{impl->max_workgroup_count_x} * kLocalSize ||
      pixels > std::numeric_limits<std::uint32_t>::max() - 3) {
    return refuse("the rectified size is past a single dispatch");
  }
  impl->out_items = static_cast<std::uint32_t>(items);
  impl->out_bytes = items * sizeof(std::uint32_t);
  VKC_TRY(core::check_storage_buffer_range("StereoRectifier: a view",
                                           impl->out_bytes,
                                           impl->max_storage_buffer_range));

  // Planes in, the view out, and its rectification; the picture's coding as
  // the colour pass's push constants.
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.size = sizeof(detail::ColorParams);
  core::KernelSetBuilder builder(device);
  VKC_TRY(builder.add(impl->kernel, "stereo_rectify", vs_rectify_comp_spv,
                      vs_rectify_comp_spv_size, 3, &push));
  VKC_ASSIGN(impl->pool, builder.build());
  VKC_TRY(impl->right_set.reserve(device, impl->kernel, 1));
  VKC_ASSIGN(impl->timer, core::GpuTimer::create(device));

  // Both views' rectification, fixed from here: uploaded once.
  const VkDeviceSize stride =
      round_up(sizeof(ViewParams), impl->offset_alignment);
  VKC_ASSIGN(impl->views, core::device_storage_buffer(
                              allocator, stride + sizeof(ViewParams)));
  device.set_object_name(VK_OBJECT_TYPE_BUFFER,
                         core::debug_object_handle(impl->views.handle()),
                         "sensor.stereo_views");
  {
    core::CommandBatch batch(device, allocator);
    for (int v = 0; v < 2; ++v) {
      const ViewParams params =
          view_params(impl->rectification.source_to_rect[v], rig.camera, luma);
      VKC_TRY(batch.upload(impl->views, v * stride, &params, sizeof(params)));
    }
    VKC_TRY(batch.submit());
  }
  impl->kernel.set.write_storage_buffer(2, impl->views.handle(), 0,
                                        sizeof(ViewParams));
  impl->right_set[0].write_storage_buffer(2, impl->views.handle(), stride,
                                          sizeof(ViewParams));

  // The views' buffers, exported where the device can: tried once, here.
  impl->exported = device.exports_memory();
  for (int v = 0; v < 2; ++v) {
    const core::Status made = impl->ensure_output(v);
    if (v == 0 && impl->exported &&
        made.domain() == core::Status::Code::Unsupported) {
      impl->exported = false;
      VKC_TRY(impl->ensure_output(v));
    } else {
      VKC_TRY(made);
    }
  }
  return std::unique_ptr<StereoRectifier>(new StereoRectifier(std::move(impl)));
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("StereoRectifier: out of host memory");
}

core::Status StereoRectifier::Impl::ensure_output(int view) {
  std::shared_ptr<ViewBuffer>& output = out[view];
  // Reused only once no pair holds its lease, as GpuFramePrep's outputs are.
  if (output != nullptr && !output->held.load(std::memory_order_acquire)) {
    return {};
  }
  auto made = std::make_shared<ViewBuffer>();
  if (exported) {
    VKC_ASSIGN(made->view,
               core::create_exported_buffer(*device, *allocator, out_bytes));
  } else {
    VKC_ASSIGN(made->view.buffer,
               core::device_storage_buffer(*allocator, out_bytes));
  }
  device->set_object_name(
      VK_OBJECT_TYPE_BUFFER,
      core::debug_object_handle(made->view.buffer.handle()),
      view == 0 ? "sensor.stereo_left" : "sensor.stereo_right");
  // The set names the buffer the next pass writes.
  const core::DescriptorSet& set = view == 0 ? kernel.set : right_set[0];
  set.write_storage_buffer(1, made->view.buffer.handle(), 0, out_bytes);
  output = std::move(made);
  return {};
}

namespace {

// One view's picture, checked: its planes as the pass binds them, and where
// its host bytes go in the slot.
struct Picture {
  detail::PlaneBinding binding;
  bool on_host = false;
  VkDeviceSize slot_at = 0;  // the slot's byte its planes start at
};

core::Result<Picture> check_picture(const core::Device& device,
                                    const YuvImage& image,
                                    const CameraModel& source, const char* what,
                                    VkDeviceSize max_range,
                                    VkDeviceSize offset_alignment) {
  if (image.image[0] != nullptr || image.image[1] != nullptr) {
    // TODO(sensor): NV12 plane images, sampled where they are, as
    // GpuFramePrep's image entry does, when a stereo source decodes on the
    // GPU.
    return refuse(std::string("the ") + what +
                  " picture's planes are images: it reads a buffer or the "
                  "host");
  }
  Picture out;
  out.on_host = image.host != nullptr;
  if (image.device == nullptr && !out.on_host) {
    return refuse(std::string("the ") + what + " picture has no planes");
  }
  VKC_ASSIGN(out.binding, detail::bind_planes(image, source.size, kWho, what,
                                              max_range, offset_alignment));
  if (image.device != nullptr) {
    const std::uint32_t family = image.queue_family;
    const bool special =
        family == kQueueFamilyIgnored || family == kQueueFamilyExternal;
    if (!special && family >= device.caps().queue_families().size()) {
      return refuse(
          std::string("the ") + what + " picture comes from queue family " +
          std::to_string(family) + ", which the device does not have");
    }
  }
  return out;
}

// After a batch that failed and may still run: let go of the device planes
// it may read, which the caller may drop as soon as the call returns.
void abandon(const StereoFrame& frame) {
  for (const YuvImage* image : {&frame.left, &frame.right}) {
    if (image->device != nullptr) {
      static_cast<void>(new std::shared_ptr<const core::Buffer>(image->device));
    }
  }
}

}  // namespace

core::Result<RectifiedPair> StereoRectifier::rectify(
    const StereoFrame& frame, core::StageMetrics* metrics) try {
  return impl_->rectify(frame, metrics);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("StereoRectifier: out of host memory");
}

core::Result<RectifiedPair> StereoRectifier::Impl::rectify(
    const StereoFrame& frame, core::StageMetrics* metrics) {
  if (!failed.ok()) {
    return failed.with_context(
        "StereoRectifier: an earlier pass failed and may still run");
  }
  const StereoCalibration& calibration = info.calibration;
  const YuvImage* const images[2] = {&frame.left, &frame.right};
  Picture pictures[2];
  VKC_ASSIGN(pictures[0],
             check_picture(*device, frame.left, calibration.left, "left",
                           max_storage_buffer_range, offset_alignment));
  VKC_ASSIGN(pictures[1],
             check_picture(*device, frame.right, calibration.right, "right",
                           max_storage_buffer_range, offset_alignment));

  // Host pictures copied into the one slot, each from a binding's alignment;
  // the last batch, which bound it, has resolved.
  VkDeviceSize slot_bytes = 0;
  for (Picture& p : pictures) {
    if (!p.on_host) continue;
    p.slot_at = round_up(slot_bytes, offset_alignment);
    slot_bytes = p.slot_at + p.binding.bind_bytes;
  }
  core::HostSlot slot;
  if (slot_bytes != 0) {
    if (inputs == nullptr) {
      core::HostRingDesc desc;
      desc.slot_bytes = slot_bytes;
      desc.slots = 1;
      desc.name = "sensor.stereo_pictures";
      desc.host_access = core::HostAccess::SequentialWrite;
      VKC_ASSIGN(inputs, core::HostRing::create(*device, *allocator, desc));
    }
    VKC_ASSIGN(slot, inputs->claim(slot_bytes));
    for (int v = 0; v < 2; ++v) {
      if (!pictures[v].on_host) continue;
      std::memcpy(slot.data() + pictures[v].slot_at,
                  images[v]->host + pictures[v].binding.bind_offset,
                  static_cast<std::size_t>(pictures[v].binding.copy_bytes));
    }
  }
  for (int v = 0; v < 2; ++v) VKC_TRY(ensure_output(v));

  if (metrics != nullptr) VKC_TRY(timer.reserve(*device, kSpans));
  core::GpuStageScope stage(metrics, timer, "stereo rectify");
  core::CommandBatch batch(*device, *allocator);
  for (int v = 0; v < 2; ++v) {
    if (images[v]->device != nullptr) {
      VKC_TRY(batch.acquire(*images[v]->device, images[v]->queue_family));
    }
    if (out[v]->external) {
      VKC_TRY(batch.acquire(out[v]->view.buffer, VK_QUEUE_FAMILY_EXTERNAL));
    }
  }
  VkBuffer input = VK_NULL_HANDLE;
  if (slot_bytes != 0) {
    VKC_ASSIGN(input, batch.input(slot, 0, slot_bytes));
  }
  for (int v = 0; v < 2; ++v) {
    const Picture& p = pictures[v];
    const core::DescriptorSet& set = v == 0 ? kernel.set : right_set[0];
    if (p.on_host) {
      set.write_storage_buffer(0, input, p.slot_at, p.binding.bind_bytes);
    } else {
      set.write_storage_buffer(0, images[v]->device->handle(),
                               p.binding.bind_offset, p.binding.bind_bytes);
    }
    const CameraModel& source = v == 0 ? calibration.left : calibration.right;
    const detail::ColorParams params = detail::color_params(
        detail::lens_params(source), *images[v], p.binding, false);
    VKC_TRY(batch.dispatch(kernel, set, &params, sizeof(params),
                           core::group_count(out_items, kLocalSize),
                           max_workgroup_count_x, &stage));
  }
  if (exported) {
    for (const std::shared_ptr<ViewBuffer>& o : out) {
      VKC_TRY(batch.release(o->view.buffer, VK_QUEUE_FAMILY_EXTERNAL));
    }
  }

  core::Result<core::PendingBatch> pending = batch.submit_async({}, {});
  const core::Status ran = pending.ok() ? pending->wait() : pending.status();
  if (!ran.ok()) {
    // The pass may still run: the frame's device planes are let go rather
    // than freed, and no later pass rewrites what it binds.
    abandon(frame);
    failed = ran;
    return ran;
  }
  if (metrics != nullptr) metrics->merge_device(pending->device_rows());

  RectifiedPair pair;
  const auto lease = [](const std::shared_ptr<ViewBuffer>& output) {
    output->held.store(true, std::memory_order_relaxed);
    return std::shared_ptr<core::ExportedBuffer>(
        &output->view, [keep = output](core::ExportedBuffer*) {
          keep->held.store(false, std::memory_order_release);
        });
  };
  for (const std::shared_ptr<ViewBuffer>& o : out) o->external = exported;
  pair.left = lease(out[0]);
  pair.right = lease(out[1]);
  pair.queue_family = exported ? kQueueFamilyExternal : kQueueFamilyIgnored;
  pair.rig = rectification.rig;
  pair.source = frame;
  // Taken over by the pass's family.
  for (YuvImage* image : {&pair.source.left, &pair.source.right}) {
    if (image->device != nullptr) image->queue_family = kQueueFamilyIgnored;
  }
  return pair;
}

}  // namespace volumetric_kit::sensor
