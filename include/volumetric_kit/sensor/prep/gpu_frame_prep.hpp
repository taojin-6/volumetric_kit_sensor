// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/utils/gpu_frame_prep.hpp
/// @brief A captured frame undistorted and converted on the GPU, left there
///        for fusion.
///
/// The host path a driver runs on each frame -- undistorting colour,
/// converting Y'CbCr to R'G'B', undistorting depth -- is the cost that grows
/// with the picture: at 4K it is most of a frame's ~55 ms of CPU. This does
/// the same on the device and hands the fusion tiers buffers they read in
/// place (`VoxelHashMap::allocate_from_depth` and `TsdfIntegrator::integrate`
/// take a `Buffer`). Driver-neutral: it takes a @ref RgbdFrame, whoever made
/// it.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/gpu_timer.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/rgbd_frame.hpp"
#include "volumetric_kit/recon/sensor/utils/export.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief One frame on the device, pinhole and R'G'B', in the shapes the
///        fusion tiers read.
///
/// Depth and colour keep their own cameras. Pass @ref depth to
/// `allocate_from_depth` and `integrate` with @ref depth_camera, and @ref color
/// as `ColorFrame::buffer` with @ref color_camera and
/// `ColorFrame::coverage_in_alpha` set.
///
/// The frame holds its buffers, which are `device_storage_buffer`s, so a
/// `CommandBatch` can copy or read them back. One kept past the next
/// @ref GpuFramePrep::prepare keeps its contents, and that call writes to new
/// buffers instead; drop a frame once it is fused and the pass reuses them.
/// The `Allocator` must outlive the frame, as it must the pass.
struct DeviceFrame {
  /// Row-major depth in metres, `depth_camera.width * height` floats; 0 where
  /// the sensor had no return or the lens maps outside its image.
  std::shared_ptr<const core::Buffer> depth;
  /// Row-major colour, `color_camera.width * height` words: R, G and B in the
  /// low three bytes, and the pixel's coverage in the high one -- 0xFF where
  /// the lens maps inside the captured picture, 0 (and black) where it maps
  /// outside, which `ColorFrame::coverage_in_alpha` has fusion skip. Null
  /// when the frame has no colour.
  std::shared_ptr<const core::Buffer> color;
  DepthCameraParams depth_camera{};  ///< The undistorted depth camera.
  ColorCameraParams color_camera{};  ///< The undistorted colour camera.
  ColorEncoding color_encoding{};    ///< What @ref color is encoded as.
  std::uint64_t timestamp_ns = 0;    ///< The raw frame's.

  /// @return `true` if this frame carries colour.
  bool has_color() const noexcept { return color != nullptr; }
};

/// @brief Options for @ref GpuFramePrep::create.
struct GpuFramePrepConfig {
  /// @brief Queue families that will access the frames' colour buffers,
  ///        `Device`Frame::color.
  ///
  /// Left empty (the default) the buffer is `VK_SHARING_MODE_EXCLUSIVE` to
  /// the pass's own family, which is right for a recon-only consumer and what
  /// the pass always made. A consumer on another queue family -- a renderer
  /// copying the colour into its atlas -- must name both, as
  /// `mesh::MarchingCubesConfig::queue_families` explains for the mesh:
  /// reading an EXCLUSIVE buffer from a family that does not own it is
  /// undefined, and on Apple, where Metal has no ownership to violate, it is
  /// undefined in the way that appears to work. Duplicates collapse, so both
  /// indices may be passed unconditionally. `Device`Frame::depth stays
  /// EXCLUSIVE whatever this says, since only recon's tiers read it.
  ///
  /// @see BufferDesc::queue_families, which this is copied into. Held by value,
  ///      since the pass re-reads it whenever it makes an output.
  std::uint32_t color_queue_families[core::BufferDesc::kMaxQueueFamilies] = {};
  /// Entries in @ref color_queue_families; more than `kMaxQueueFamilies` is
  /// refused by @ref GpuFramePrep::create.
  std::uint32_t color_queue_family_count = 0;

  /// @brief Keep depth only where the colour camera recorded it.
  ///
  /// A depth sensor's field of view and its colour camera's rarely match: a
  /// Femto Mega's depth covers about 65 degrees vertically and its 16:9
  /// colour about 51. Depth past the colour's view fuses into surfaces no
  /// colour camera sees, which carry no colour and cannot be textured. With
  /// this set, a frame with colour has every depth pixel zeroed -- the pass's
  /// "no return" -- whose point, moved into the colour camera by the two
  /// poses, is behind it or lands on a pixel the colour pass marks uncovered
  /// (`Device`Frame::color's coverage byte, black where the lens saw
  /// nothing). What survives is the two cameras' overlap. The test is the
  /// colour camera's view, not its line of sight: a point inside the view
  /// that a nearer surface hides from the colour camera is kept. A frame
  /// without colour keeps all its depth. Off (the default) keeps all of it
  /// always, as the pass always did.
  bool depth_within_color = false;
};

/// @brief Undistorts a @ref RgbdFrame's depth and colour, and converts its
///        colour to R'G'B', in two compute passes.
///
/// Undistorting keeps each camera's intrinsics and drops its lens: every pixel
/// of the pinhole image is sampled from where `camera::distort_rational` puts
/// it in the captured one. Colour is sampled bilinearly, luma and chroma each
/// at its own siting (@ref YuvImage::chroma_location), and converted by the
/// picture's matrix and range; depth is
/// sampled at the nearest pixel, so an edge never blends a foreground and a
/// background depth into a point between them.
///
/// Colour comes as I420 or NV12, on the device as a hardware decoder left it:
/// planes in a buffer, as NVDEC's and nvJPEG's pictures arrive, are read
/// where they are, after the pass takes them over from the queue family that
/// wrote them (`YuvImage::queue_family`); NV12's planes as images, as
/// VideoToolbox's picture arrives, are copied into the pass's input on the
/// device in the same batch. A dataset decoded on the host hands its colour
/// over as packed R'G'B' words instead (@ref RgbdFrame::color_packed), which
/// go up with the depth: each copied into the pass's staging, and up in its
/// batch. Nothing else goes up from the host.
///
/// Every check on the frame is made before anything is uploaded, so a refused
/// frame leaves the pass and the frames it handed out as they were.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object; it stores references to them. Not
///          thread-safe.
class VR_SENSOR_UTILS_API GpuFramePrep {
 public:
  /// @brief Build the two passes on @p device.
  /// @param device     The device the frames are prepared on (must outlive
  ///                   this object).
  /// @param allocator  The allocator its buffers come from (must outlive
  ///                   this object and every frame it hands out).
  /// @param config     Which queue families read the colour output (see
  ///                   @ref GpuFramePrepConfig).
  /// @return The pass; `Status::Code::InvalidArgument` for a
  ///         `color_queue_family_count` past `BufferDesc::kMaxQueueFamilies`;
  ///         or a non-OK `Status` if a pipeline, descriptor object or
  ///         buffer fails to build.
  static core::Result<GpuFramePrep> create(
      core::Device& device, core::Allocator& allocator,
      const GpuFramePrepConfig& config = {});

  ~GpuFramePrep() = default;
  GpuFramePrep(GpuFramePrep&&) noexcept = default;
  GpuFramePrep& operator=(GpuFramePrep&&) noexcept = default;
  GpuFramePrep(const GpuFramePrep&) = delete;
  GpuFramePrep& operator=(const GpuFramePrep&) = delete;

  /// @brief Upload @p frame's depth and any host colour, undistort and
  ///        convert the frame, and hand the result over as buffers on the
  ///        device.
  /// @param frame    The frame; read during the call only. Device planes must
  ///                 be on this pass's device, their writer finished.
  /// @param metrics  Optional `StageMetrics` collecting a `"frame prep"`
  ///                 row: the upload and both passes on the host, and on the
  ///                 device the frame's copies up and the two dispatches.
  ///                 `nullptr` measures nothing.
  /// @return The frame, which holds its buffers; @ref
  ///         Status::Code::InvalidArgument for a moved-from pass, a frame
  ///         without depth, a depth range that is not finite with
  ///         `0 < min_depth < max_depth` (0 being the pass's "no return"),
  ///         a `color_to_world` or `depth_to_color` that is not rigid, a
  ///         camera or picture that is empty, not finite or disagrees with
  ///         its image, an unknown chroma location, colour both as packed
  ///         words and as planes, colour planes both in a buffer and as
  ///         images, a plane stride shorter than its rows,
  ///         planes that overlap, lie outside their buffer or are in one
  ///         that is empty or without storage usage, plane images
  ///         that are not NV12's R8 and R8G8 planes at least the picture's
  ///         size with `TRANSFER_SRC` usage, in a layout a copy reads, a
  ///         `queue_family` the device lacks, or an image past a single
  ///         dispatch (16.7 M pixels);
  ///         `Status::Code::Unsupported` for a colour encoding
  ///         @ref is_canonical refuses; `Status::Code::OutOfMemory` when the
  ///         device or the host has no memory for a buffer; otherwise a
  ///         buffer or dispatch failure.
  core::Result<DeviceFrame> prepare(const RgbdFrame& frame,
                                    core::StageMetrics* metrics = nullptr);

  /// @brief @ref prepare several cameras' frames in one batch: every frame's
  ///        uploads and passes recorded into one command buffer, submitted
  ///        once and waited on once.
  ///
  /// The array's way (`SensorArray::process`): one submit and one wait for
  /// the set rather than one a camera. Every frame is checked before any is
  /// recorded, so a refused set leaves every pass as it was.
  /// @param preps    One pass per camera, all made with one `Device`;
  ///                 `preps[i]` prepares `frames[i]`.
  /// @param frames   One entry per camera, an empty one skipped.
  /// @param metrics  Optional: one `"frame prep"` row for the whole batch,
  ///                 timed by the first pass used.
  /// @return One `DeviceFrame` per present frame, empty where the frame was;
  ///         `Status::Code::InvalidArgument` for fewer passes than frames or
  ///         passes made with different `Device`s; what @ref prepare returns
  ///         for a frame it refuses; otherwise a buffer or submit failure.
  static core::Result<std::vector<std::optional<DeviceFrame>>> prepare_batch(
      std::vector<GpuFramePrep>& preps,
      const std::vector<std::optional<RgbdFrame>>& frames,
      core::StageMetrics* metrics = nullptr);

  /// @return `true` if this owns its pipelines (`false` when moved-from).
  bool valid() const noexcept { return depth_kernel_.valid(); }

 private:
  GpuFramePrep() = default;

  // A checked frame's layout (gpu_frame_prep.cpp).
  struct Layout;
  // The one path prepare and prepare_batch take: `preps[i]` prepares
  // `*frames[i]` into `out[i]`, a null frame skipped, with `layouts[i]` its
  // scratch, for `count` frames.
  static core::Status prepare_frames(GpuFramePrep* preps,
                                     const RgbdFrame* const* frames,
                                     Layout* layouts,
                                     std::optional<DeviceFrame>* out,
                                     std::size_t count,
                                     core::StageMetrics* metrics);
  // The whole frame checked, before anything is uploaded.
  core::Result<Layout> check(const RgbdFrame& frame) const;
  // Device planes taken over from their writer's family, into `batch`.
  core::Status acquire(core::CommandBatch& batch, const RgbdFrame& frame);
  // The buffers made big enough, and the depth and any packed colour staged.
  core::Status stage_host(const RgbdFrame& frame, const Layout& layout);
  // The copies up and the overlap mask's parameters, recorded into `batch`.
  core::Status record_uploads(core::CommandBatch& batch, const RgbdFrame& frame,
                              const Layout& layout, core::GpuStageScope* stage);
  // Both passes, recorded into `batch` after the uploads; the outputs hold
  // the result once the batch is submitted.
  core::Status record_passes(core::CommandBatch& batch, const RgbdFrame& frame,
                             const Layout& layout, core::GpuStageScope* stage);
  // After a submit that failed: let go of what the GPU may still read.
  void abandon(const RgbdFrame& frame, const Layout& layout);
  // The frame handed out after a submit that succeeded.
  DeviceFrame finish(const RgbdFrame& frame) const;

  // An output of at least `bytes`: the one held, when no DeviceFrame still
  // holds it too and it is big enough, else a new one, shared with the
  // config's colour families when `color`; OutOfMemory, without throwing,
  // when the host has no memory to hold a new one.
  core::Status ensure_output(std::shared_ptr<core::Buffer>& buffer,
                             VkDeviceSize bytes, const char* name, bool color);

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  // Re-read by ensure_output whenever an output is made.
  GpuFramePrepConfig config_;

  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;
  VkDeviceSize min_storage_buffer_offset_alignment_ = 0;

  core::ComputeKernel depth_kernel_;
  core::ComputeKernel color_kernel_;
  core::DescriptorPool pool_;
  core::GpuTimer gpu_timer_;

  // The raw inputs, device-local and filled through the pass's batch, grown
  // to the largest frame seen and kept; colour already on the device is read
  // where it is instead.
  core::Buffer depth_in_;
  core::Buffer color_in_;
  // The depth and any packed colour on the host side, host-visible and kept
  // like the inputs.
  core::Buffer staging_;
  // The colour camera and the depth-to-colour transform the depth pass masks
  // by (GpuFramePrepConfig::depth_within_color), written inline each frame
  // that uses them; bound once, at create.
  core::Buffer overlap_;
  // The outputs the fusion tiers read, device-local, shared with the
  // DeviceFrames handed out; reused only once no frame holds them.
  std::shared_ptr<core::Buffer> depth_out_;
  std::shared_ptr<core::Buffer> color_out_;
};

}  // namespace volumetric_kit::recon::sensor
