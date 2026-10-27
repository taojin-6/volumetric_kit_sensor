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
#include "volumetric_kit/core/vulkan/host_ring.hpp"
#include "volumetric_kit/core/vulkan/unique_handle.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
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
  /// Row-major depth in metres, less the pass's
  /// @ref GpuFramePrepConfig::depth_offset_m, `depth_camera.width * height`
  /// floats; 0 where the sensor had no return, the offset leaves none, or
  /// the lens maps outside its image.
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
  ///        @ref DeviceFrame::color.
  ///
  /// Left empty (the default) the buffer is `VK_SHARING_MODE_EXCLUSIVE` to
  /// the pass's own family, which is right for a recon-only consumer and what
  /// the pass always made. A consumer on another queue family -- a renderer
  /// copying the colour into its atlas -- must name both, as
  /// `mesh::MarchingCubesConfig::queue_families` explains for the mesh:
  /// reading an EXCLUSIVE buffer from a family that does not own it is
  /// undefined, and on Apple, where Metal has no ownership to violate, it is
  /// undefined in the way that appears to work. Duplicates collapse, so both
  /// indices may be passed unconditionally.
  ///
  /// @see BufferDesc::queue_families, which this is copied into. Held by value,
  ///      since the pass re-reads it whenever it makes an output.
  std::uint32_t color_queue_families[core::BufferDesc::kMaxQueueFamilies] = {};
  /// Entries in @ref color_queue_families; more than `kMaxQueueFamilies`, or
  /// an entry naming a family the device does not have, is refused by
  /// @ref GpuFramePrep::create.
  std::uint32_t color_queue_family_count = 0;

  /// @brief Queue families that will access the frames' depth buffers,
  ///        @ref DeviceFrame::depth, as @ref color_queue_families does for
  ///        colour and independently of it.
  ///
  /// Left empty (the default) depth is EXCLUSIVE to the pass's own family,
  /// as recon's tiers read it. A consumer on another family -- a renderer
  /// drawing points from the depth in place -- names both. The buffer has no
  /// usage beyond the pass's storage and transfer usage (no vertex usage); a
  /// consumer reads it as a storage buffer.
  std::uint32_t depth_queue_families[core::BufferDesc::kMaxQueueFamilies] = {};
  /// Entries in @ref depth_queue_families; refused by
  /// @ref GpuFramePrep::create as for @ref color_queue_family_count.
  std::uint32_t depth_queue_family_count = 0;

  /// @brief Keep depth only where the colour camera recorded it.
  ///
  /// A depth sensor's field of view and its colour camera's rarely match: a
  /// Femto Mega's depth covers about 65 degrees vertically and its 16:9
  /// colour about 51. Depth past the colour's view fuses into surfaces no
  /// colour camera sees, which carry no colour and cannot be textured. With
  /// this set, a frame with colour has every depth pixel zeroed -- the pass's
  /// "no return" -- whose point, moved into the colour camera by the two
  /// poses, is behind it or lands on a pixel the colour pass marks uncovered
  /// (@ref DeviceFrame::color's coverage byte, black where the lens saw
  /// nothing). What survives is the two cameras' overlap. The test is the
  /// colour camera's view, not its line of sight: a point inside the view
  /// that a nearer surface hides from the colour camera is kept. A frame
  /// without colour keeps all its depth. Off (the default) keeps all of it
  /// always, as the pass always did.
  bool depth_within_color = false;

  /// @brief Metres the sensor's depth reads too far (+) or too near (-),
  ///        subtracted from every return.
  ///
  /// The sensor's own, from its calibration
  /// (`camera::SensorCalibration::depth_offset_m`): `SensorArray` sets it
  /// per sensor, and refuses one in `SensorArray::Options::prep`. A return
  /// the offset takes to 0 or below is no return (0), and 0 stays no return
  /// whatever the sign, so a negative offset never makes a hole a surface.
  /// It is subtracted before @ref depth_within_color's test, which sees the
  /// corrected depth. 0, the default, leaves depth as the sensor reported
  /// it. Finite, or @ref GpuFramePrep::create refuses it.
  float depth_offset_m = 0.0f;
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
/// planes in a buffer, as nvJPEG's pictures arrive, are read where they are,
/// and NV12's planes as images, as VideoToolbox's and NVDEC's pictures
/// arrive, are sampled where they are, with no copy, either after the pass
/// takes them over from the queue family that wrote them
/// (`YuvImage::queue_family`). The two read the same samples, so give the
/// same frame. A frame's depth, and colour a dataset decoded on the host
/// (@ref RgbdFrame::color_packed), are copied into the pass's
/// persistent slot (`core::HostRing`), which its batch binds: in place where
/// the device's mapped memory is cached (Apple, lavapipe), and through one
/// copy of the used range elsewhere (a discrete GPU). Nothing else goes up
/// from the host, and nothing is allocated per frame.
///
/// Every check on the frame is made before anything is recorded, so a refused
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
  /// @param config     Which queue families read the colour and the depth
  ///                   outputs, and how depth is corrected and masked (see
  ///                   @ref GpuFramePrepConfig).
  /// @return The pass; `Status::Code::InvalidArgument` for a
  ///         `color_queue_family_count` or a `depth_queue_family_count` past
  ///         `BufferDesc::kMaxQueueFamilies`, a colour or depth queue
  ///         family @p device does not have, or a `depth_offset_m` that is
  ///         not finite; or a non-OK `Status` if a
  ///         pipeline, descriptor object or buffer fails to build.
  static core::Result<GpuFramePrep> create(
      core::Device& device, core::Allocator& allocator,
      const GpuFramePrepConfig& config = {});

  ~GpuFramePrep() = default;
  GpuFramePrep(GpuFramePrep&&) noexcept = default;
  GpuFramePrep& operator=(GpuFramePrep&&) noexcept = default;
  GpuFramePrep(const GpuFramePrep&) = delete;
  GpuFramePrep& operator=(const GpuFramePrep&) = delete;

  /// @brief Record @ref prepare's work for @p frame into @p batch: the frame
  ///        copied into one of the pass's slots, any device planes taken over,
  ///        and both passes, with nothing submitted.
  ///
  /// The frame is checked before anything is recorded, so a refusal leaves
  /// @p batch and the pass as they were. The frame handed back holds its
  /// buffers, which hold the prepared frame once @p batch has run.
  /// @param batch    Records the work; the caller submits it.
  /// @param frame    The frame. Its depth and host colour are copied here.
  ///                 Device planes are read when @p batch runs, so keep the
  ///                 frame until the batch has resolved, and after a failed
  ///                 wait for as long as `PendingBatch::in_flight` says.
  /// @param metrics  Optional: a `"frame prep"` row, its host half now and its
  ///                 device half from `PendingBatch::device_rows`.
  /// @return The frame; with nothing recorded, @ref prepare's refusals, or
  ///         `Status::Code::InvalidArgument` while the batch of the pass's
  ///         last record has not resolved (its slot is held); or, after the
  ///         first commands, so the caller drops @p batch, a buffer failure.
  /// @pre The batch of this pass's last record has resolved: it binds the
  ///      pass's slot and its kernels' sets, which this rewrites.
  core::Result<DeviceFrame> record(core::CommandBatch& batch,
                                   const RgbdFrame& frame,
                                   core::StageMetrics* metrics = nullptr);

  /// @brief @ref record several cameras' frames into @p batch: every frame's
  ///        copies, then every frame's passes, so the copies run with no
  ///        barrier between them.
  ///
  /// What `SensorArray::record_process` records. Every frame is checked
  /// before any is recorded.
  /// @param batch    As @ref record.
  /// @param preps    One pass per camera, all made with @p batch's `Device`;
  ///                 `preps[i]` records `frames[i]`.
  /// @param frames   One entry per camera, an empty one skipped; kept as
  ///                 @ref record says.
  /// @param metrics  Optional: one `"frame prep"` row for the set, timed by
  ///                 the first pass used.
  /// @return One `DeviceFrame` per present frame, empty where the frame was;
  ///         @ref prepare_batch's refusals; or as @ref record.
  static core::Result<std::vector<std::optional<DeviceFrame>>> record_set(
      core::CommandBatch& batch, std::vector<GpuFramePrep>& preps,
      const std::vector<std::optional<RgbdFrame>>& frames,
      core::StageMetrics* metrics = nullptr);

  /// @brief Undistort and convert @p frame, and hand the result over as
  ///        buffers on the device: @ref record, one submit and its wait.
  /// @param frame    The frame; read during the call only. Device planes must
  ///                 be on this pass's device, their writer finished.
  /// @param metrics  Optional `StageMetrics` collecting a `"frame prep"`
  ///                 row: the copy into the slot and both passes on the host,
  ///                 and on the device the frame's copy up and the two
  ///                 dispatches. `nullptr` measures nothing.
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
  ///         size, 2D, of one layer and one sample, with `SAMPLED` usage
  ///         and a view, in `GENERAL` or `SHADER_READ_ONLY_OPTIMAL`, a
  ///         `queue_family` the device lacks, `EXCLUSIVE` images from
  ///         another of its families, or an image past a single
  ///         dispatch (16.7 M pixels);
  ///         `Status::Code::Unsupported` for a colour encoding
  ///         @ref is_canonical refuses; `Status::Code::OutOfMemory` for a
  ///         host allocation failure, with previously returned frames intact;
  ///         otherwise a buffer or dispatch failure.
  core::Result<DeviceFrame> prepare(const RgbdFrame& frame,
                                    core::StageMetrics* metrics = nullptr);

  /// @brief @ref prepare several cameras' frames in one batch:
  ///        @ref record_set, one submit and its wait.
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

  // A checked frame's layout, and an output with its lease
  // (gpu_frame_prep.cpp).
  struct Layout;
  struct Output;
  // Every frame checked, `layouts[i]` for `*frames[i]`, a null frame skipped,
  // and every pass on one device: the first pass used, or null for no frame.
  static core::Result<GpuFramePrep*> check_frames(
      GpuFramePrep* preps, const RgbdFrame* const* frames, Layout* layouts,
      std::size_t count);
  // The one path record, record_set, prepare and prepare_batch take: the
  // checked frames recorded into `batch`, `preps[i]` recording `*frames[i]`
  // into `out[i]`, timed under `stage`.
  static core::Status record_frames(core::CommandBatch& batch,
                                    GpuFramePrep* preps,
                                    const RgbdFrame* const* frames,
                                    const Layout* layouts,
                                    std::optional<DeviceFrame>* out,
                                    std::size_t count,
                                    core::GpuStageScope* stage);
  // prepare's and prepare_batch's: record_frames, one submit and its wait.
  static core::Status prepare_frames(GpuFramePrep* preps,
                                     const RgbdFrame* const* frames,
                                     Layout* layouts,
                                     std::optional<DeviceFrame>* out,
                                     std::size_t count,
                                     core::StageMetrics* metrics);
  // The whole frame checked, before anything is recorded.
  core::Result<Layout> check(const RgbdFrame& frame) const;
  // A slot claimed and filled with the depth and any packed colour; nothing
  // recorded and no buffer changed.
  core::Result<core::HostSlot> stage(const RgbdFrame& frame,
                                     const Layout& layout);
  // Once the frame's slot is claimed, so the pass's last batch has resolved:
  // the outputs made big enough.
  core::Status ensure_buffers(const RgbdFrame& frame, const Layout& layout);
  // Device planes taken over from their writer's family, the slot bound (one
  // copy where it is copied) and the overlap mask's parameters written, into
  // `batch`; `*input` is the buffer the passes bind.
  core::Status record_inputs(core::CommandBatch& batch, const RgbdFrame& frame,
                             const Layout& layout, const core::HostSlot& slot,
                             VkBuffer* input);
  // Both passes, recorded into `batch` after the inputs; the outputs hold
  // the result once the batch has run.
  core::Status record_passes(core::CommandBatch& batch, const RgbdFrame& frame,
                             const Layout& layout, VkBuffer input,
                             core::GpuStageScope* stage);
  // After a batch that failed and may still run: let go of the device planes
  // it may read.
  static void abandon(const RgbdFrame& frame);
  // The frame handed out, its outputs leased to it.
  DeviceFrame finish(const RgbdFrame& frame) const;

  // An output of at least `bytes`: the one held, when no DeviceFrame holds
  // its lease and it is big enough, else a new one, shared with the
  // `family_count` families at `families` (the config's for that output).
  core::Status ensure_output(std::shared_ptr<Output>& output,
                             VkDeviceSize bytes, const char* name,
                             const std::uint32_t* families,
                             std::uint32_t family_count);

  // Borrowed (must outlive this).
  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  // Re-read by ensure_output whenever an output is made.
  GpuFramePrepConfig config_;

  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;
  VkDeviceSize min_storage_buffer_offset_alignment_ = 0;

  // Outlives the image kernel's set, which names it.
  core::UniqueHandle<VkSampler, vkDestroySampler> sampler_;
  core::ComputeKernel depth_kernel_;
  core::ComputeKernel color_kernel_;
  core::DescriptorPool pool_;
  // The colour pass over NV12's plane images, sampled in place: its layout
  // built by hand (two combined image samplers and the output), its set from
  // a pool of its own.
  core::ComputeKernel color_images_kernel_;
  core::DescriptorPool images_pool_;
  core::GpuTimer gpu_timer_;

  // Each frame's depth, and colour a dataset decoded on the host, copied into
  // its one slot, the colour range at minStorageBufferOffsetAlignment; made
  // at the first record, grown as a larger frame asks. The host only writes
  // the slot (core::HostAccess::SequentialWrite), so where the ring is
  // copied, as on a discrete GPU, it is write-combined staging.
  std::unique_ptr<core::HostRing> inputs_;
  // The colour camera and the depth-to-colour transform the depth pass masks
  // by (GpuFramePrepConfig::depth_within_color), written inline each frame
  // that uses them; bound once, at create.
  core::Buffer overlap_;
  // The outputs the fusion tiers read, device-local, leased to the
  // DeviceFrames handed out; reused only once no frame holds the lease.
  std::shared_ptr<Output> depth_out_;
  std::shared_ptr<Output> color_out_;
};

}  // namespace volumetric_kit::recon::sensor
