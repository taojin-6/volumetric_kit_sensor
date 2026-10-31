// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file prep/gpu_frame_prep.hpp
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
#include <utility>
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
#include "volumetric_kit/sensor/frame/camera_params.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/frame/device_frame.hpp"
#include "volumetric_kit/sensor/prep/export.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"

namespace volumetric_kit::sensor {

namespace detail {
class DepthOutlierPass;
}  // namespace detail

/// @brief The radius outlier filter @ref GpuFramePrep runs on depth after
///        undistorting it (@ref GpuFramePrepConfig::depth_outliers).
///
/// A return is kept where at least @ref min_points returns, itself included,
/// lie strictly within `radius_px * z / fx` metres of its point, that is
/// @ref radius_px pixel footprints at its own depth `z`, and is 0, the pass's
/// "no return", elsewhere. The points are the undistorted image's pinhole
/// points, `((u - cx) z / fx, (v - cy) z / fy, z)`, so the radius spans the
/// same pixels at every depth. A surface holds about
/// `pi radius_px^2 cos(theta)` returns within it, `theta` its slant from
/// facing the camera: `201 cos(theta)` at the defaults, so a surface is kept
/// to about 84 degrees of slant. The curtain of flying pixels a stereo
/// matcher or a ToF sensor leaves between a foreground and a background lies
/// along the rays and holds about `2 radius_px + 1`, 17, so it goes, as does
/// any structure under about `min_points / (2 radius_px)` pixels wide (a
/// cable one pixel across); an outline's rim, half a disc, stays.
///
/// The defaults are the FFS study's (2026-10-30, D455 stereo at 1280x720 and
/// 960x544): 77-83% of the flying-pixel proxy removed, 0.3% of smooth
/// surface nearer than 5 m lost. It adds holes, never fills them, and sees
/// one camera alone.
struct DepthOutlierFilter {
  /// Whether depth is filtered; `false`, the default, keeps every return.
  bool enabled = false;
  /// The radius, in pixel footprints at each return's own depth. Finite, in
  /// [1, @ref kMaxRadiusPx], or, with @ref enabled, @ref GpuFramePrep::create
  /// refuses it.
  float radius_px = 8.0f;
  /// The fewest returns within the radius, the return itself included, for
  /// it to be kept; at least 2, or, with @ref enabled,
  /// @ref GpuFramePrep::create refuses it.
  std::uint32_t min_points = 20;

  /// The largest @ref radius_px: a removed return scans a window of about
  /// `(2 radius_px + 1)^2` pixels, 65x65 at this.
  static constexpr float kMaxRadiusPx = 32.0f;
};

/// @brief Options for @ref GpuFramePrep::create.
struct GpuFramePrepConfig {
  /// @brief Queue families that will access the frames' colour buffers,
  ///        @ref DeviceFrame::color.
  ///
  /// Left empty (the default) the buffer is `VK_SHARING_MODE_EXCLUSIVE` to
  /// the pass's own family, which is right for a consumer on that family and
  /// what the pass always made. A consumer on another queue family -- a
  /// renderer copying the colour into its atlas -- must name both, as
  /// `vgrid::mesh::MarchingCubesConfig::queue_families` explains for the mesh:
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
  /// as a fusion on that family reads it. A consumer on another family -- a
  /// renderer drawing points from the depth in place -- names both. The buffer
  /// has no usage beyond the pass's storage and transfer usage (no vertex
  /// usage); a consumer reads it as a storage buffer.
  std::uint32_t depth_queue_families[core::BufferDesc::kMaxQueueFamilies] = {};
  /// Entries in @ref depth_queue_families; refused by
  /// @ref GpuFramePrep::create as for @ref color_queue_family_count.
  std::uint32_t depth_queue_family_count = 0;

  /// @brief Metres the sensor's depth reads too far (+) or too near (-),
  ///        subtracted from every return.
  ///
  /// The sensor's own, from its calibration
  /// (`camera::SensorCalibration::depth_offset_m`): `SensorArray` sets it
  /// per sensor, and refuses one in `SensorArray::Options::prep`. A return
  /// the offset takes to 0 or below is no return (0), and 0 stays no return
  /// whatever the sign, so a negative offset never makes a hole a surface.
  /// It is subtracted from each raw pixel before the depth mix, so a return
  /// it takes to 0 is a hole the mix skips, and before the colour-view test
  /// (@ref GpuFramePrep), which sees the corrected depth. 0, the
  /// default, leaves depth as the sensor reported it. Finite, or
  /// @ref GpuFramePrep::create refuses it.
  float depth_offset_m = 0.0f;

  /// @brief Returns removed where too few others lie near them in 3D, after
  ///        undistortion and the colour-view test (@ref DepthOutlierFilter).
  ///
  /// Off by default. On, the depth pass writes into a buffer of the pass's
  /// own, which the filter reads into @ref DeviceFrame::depth: one more
  /// dispatch and one more depth-sized buffer per pass.
  DepthOutlierFilter depth_outliers;
};

/// @brief Undistorts a @ref RgbdFrame's depth and colour, and converts its
///        colour to R'G'B', in two compute passes, and optionally
///        removes depth outliers in a third.
///
/// Undistorting keeps each camera's intrinsics and drops its lens: every pixel
/// of the pinhole image is sampled from where `camera::distort_rational` puts
/// it in the captured one. Colour is sampled bilinearly, luma and chroma each
/// at its own siting (@ref YuvImage::chroma_location), and converted by the
/// picture's matrix and range. Depth is the bilinear mix of the four raw
/// pixels around that position where all four hold a return within 4 cm of
/// each other, so a slope comes out smooth rather than in half-pixel
/// terraces, and the nearest raw pixel elsewhere, so an edge never blends a
/// foreground and a background depth into a point between them, nor a hole
/// into a return.
///
/// Colour comes as I420, NV12, YUYV or luma alone (`YuvLayout::Luma`, a mono
/// camera's, which comes out grey), on the device as a decoder left it:
/// planes in a buffer, as nvJPEG's pictures arrive, are read where they are,
/// and NV12's planes as images, as VideoToolbox's and NVDEC's pictures
/// arrive, are sampled where they are, with no copy, either after the pass
/// takes them over from the queue family that wrote them
/// (`YuvImage::queue_family`). The two read the same samples, so give the
/// same frame. Or it comes on the host: planes as a camera's SDK hands them
/// over (`YuvImage::host`; a UVC camera's YUYV), converted on the device like
/// a buffer's, or words a dataset decoded (@ref RgbdFrame::color_packed).
///
/// Depth comes from the host, as a sensor's SDK hands it over
/// (@ref RgbdFrame::depth), or from the device, as a GPU stage such as stereo
/// matching wrote it (@ref RgbdFrame::depth_device): read where it is, at any
/// even byte of its buffer, after the pass takes it over from its writer's
/// queue family, with no copy. A buffer is taken over once a batch, however
/// many depths and pictures of its frames it holds -- a frame's depth and
/// planes, or every camera's depth -- so they must all name the family that
/// wrote it.
///
/// A frame with colour keeps only what both its cameras see, so each pixel
/// handed out is in both views: a depth sensor's field of view and its colour
/// camera's rarely match, and the overlap is each sensor's own, from its
/// intrinsics and extrinsic. A view test turns a pixel's direction into the
/// other camera by the extrinsic's rotation alone.
/// - Colour is a 0 word, coverage and all, where its direction lies outside
///   the depth camera's view: on no pixel the depth pass covers.
/// - Depth, wherever it came from, is zeroed, the pass's "no return", where
///   its direction lies outside the colour camera's view, or where its point,
///   moved into the colour camera by the two poses, is behind it or lands on
///   a colour pixel that is not kept, so kept depth always lands on colour
///   with its coverage byte set.
///
/// Both test a camera's view, not its line of sight: a point inside the view
/// that a nearer surface hides from it is kept. A frame without colour keeps
/// all its depth (every frame has depth).
///
/// A frame's host depth and host colour are copied into the pass's persistent
/// slot (`core::HostRing`), which its batch binds: in place where the device's
/// mapped memory is cached (Apple, lavapipe), and through one copy of the used
/// range elsewhere (a discrete GPU). Nothing else goes up from the host, and
/// nothing is allocated per frame.
///
/// Every check on the frame is made before anything is recorded, so a refused
/// frame leaves the pass and the frames it handed out as they were.
///
/// @warning The `Device` and `Allocator` passed to @ref create must
///          outlive this object; it stores references to them. Not
///          thread-safe.
class VS_PREP_API GpuFramePrep {
 public:
  /// @brief Build the passes on @p device.
  /// @param device     The device the frames are prepared on (must outlive
  ///                   this object).
  /// @param allocator  The allocator its buffers come from (must outlive
  ///                   this object and every frame it hands out).
  /// @param config     Which queue families read the colour and the depth
  ///                   outputs, and how depth is corrected and filtered
  ///                   (see @ref GpuFramePrepConfig).
  /// @return The pass; `Status::Code::InvalidArgument` for a
  ///         `color_queue_family_count` or a `depth_queue_family_count` past
  ///         `BufferDesc::kMaxQueueFamilies`, a colour or depth queue
  ///         family @p device does not have, a `depth_offset_m` that is
  ///         not finite, or `depth_outliers` enabled with a `radius_px` that
  ///         is not finite or lies outside [1, `kMaxRadiusPx`], or a
  ///         `min_points` below 2; or a non-OK `Status` if a
  ///         pipeline, descriptor object or buffer fails to build.
  static core::Result<GpuFramePrep> create(
      core::Device& device, core::Allocator& allocator,
      const GpuFramePrepConfig& config = {});

  ~GpuFramePrep();
  GpuFramePrep(GpuFramePrep&&) noexcept;
  GpuFramePrep& operator=(GpuFramePrep&&) noexcept;
  GpuFramePrep(const GpuFramePrep&) = delete;
  GpuFramePrep& operator=(const GpuFramePrep&) = delete;

  /// @brief Record @ref prepare's work for @p frame into @p batch: its host
  ///        depth and colour copied into the pass's slot, any device depth
  ///        and planes taken over, and the passes, with nothing submitted.
  ///
  /// The frame is checked before anything is recorded, so a refusal leaves
  /// @p batch and the pass as they were. The frame handed back holds its
  /// buffers, which hold the prepared frame once @p batch has run.
  /// @param batch    Records the work; the caller submits it.
  /// @param frame    The frame. Its host depth and colour are copied here.
  ///                 Device depth and planes are read when @p batch runs, so
  ///                 keep the frame until the batch has resolved, and after a
  ///                 failed wait for as long as `PendingBatch::in_flight`
  ///                 says.
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
  /// @param frame    The frame; read during the call only. Device depth and
  ///                 planes must be on this pass's device, their writer
  ///                 finished.
  /// @param metrics  Optional `StageMetrics` collecting a `"frame prep"`
  ///                 row: the copy into the slot and the passes on the host,
  ///                 and on the device the frame's copy up and the two
  ///                 dispatches (three with the outlier filter). `nullptr`
  ///                 measures nothing.
  /// @return The frame, which holds its buffers; @ref
  ///         Status::Code::InvalidArgument for a moved-from pass, a frame
  ///         without depth or with depth both on the host and on the device,
  ///         device depth at an odd offset, past its buffer (to the word
  ///         holding its last sample) or in one that is empty, without
  ///         storage usage or not device-local, a depth range that is not
  ///         finite with
  ///         `0 < min_depth < max_depth` (0 being the pass's "no return"),
  ///         a `color_to_world` or `depth_to_color` that is not rigid, a
  ///         camera or picture that is empty, not finite or disagrees with
  ///         its image, an unknown chroma location or colour layout, colour
  ///         both as packed words and as planes, colour planes in more than
  ///         one of a buffer, the host and images, YUYV of odd width, a
  ///         plane stride shorter than its rows, host planes whose bytes run
  ///         past 64 bits or span more than 4 GiB,
  ///         planes that overlap, lie outside their buffer or are in one
  ///         that is empty or without storage usage, plane images
  ///         that are not NV12's R8 and R8G8 planes at least the picture's
  ///         size, 2D, of one layer and one sample, with `SAMPLED` usage
  ///         and a view, in `GENERAL` or `SHADER_READ_ONLY_OPTIMAL`, a
  ///         `queue_family` the device lacks, depth and planes in one buffer
  ///         that name different writers' families, `EXCLUSIVE` images from
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
  ///         `Status::Code::InvalidArgument` for fewer passes than frames,
  ///         passes made with different `Device`s, or frames whose depth or
  ///         planes share a buffer and name different writers' families;
  ///         what @ref prepare returns for a frame it refuses; otherwise a
  ///         buffer or submit failure.
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
  // A buffer the frames read on the device, and the queue family the batch
  // takes it over from.
  using Acquire = std::pair<const core::Buffer*, std::uint32_t>;
  // Every frame checked, `layouts[i]` for `*frames[i]`, a null frame skipped,
  // and every pass on one device: the first pass used, or null for no frame.
  // `*acquires` gets each distinct buffer the frames read on the device that
  // needs taking over, once, however many depths or pictures name it.
  static core::Result<GpuFramePrep*> check_frames(
      GpuFramePrep* preps, const RgbdFrame* const* frames, Layout* layouts,
      std::size_t count, std::vector<Acquire>* acquires);
  // The one path record, record_set, prepare and prepare_batch take: the
  // checked frames' buffers taken over (`acquires`), and the frames recorded
  // into `batch`, `preps[i]` recording `*frames[i]` into `out[i]`, timed
  // under `stage`.
  static core::Status record_frames(
      core::CommandBatch& batch, GpuFramePrep* preps,
      const RgbdFrame* const* frames, const Layout* layouts,
      const std::vector<Acquire>& acquires, std::optional<DeviceFrame>* out,
      std::size_t count, core::GpuStageScope* stage);
  // prepare's and prepare_batch's: record_frames, one submit and its wait.
  static core::Status prepare_frames(GpuFramePrep* preps,
                                     const RgbdFrame* const* frames,
                                     Layout* layouts,
                                     std::optional<DeviceFrame>* out,
                                     std::size_t count,
                                     core::StageMetrics* metrics);
  // The whole frame checked, before anything is recorded.
  core::Result<Layout> check(const RgbdFrame& frame) const;
  // A slot claimed and filled with any host depth and colour; nothing
  // recorded and no buffer changed.
  core::Result<core::HostSlot> stage(const RgbdFrame& frame,
                                     const Layout& layout);
  // Once the frame's slot is claimed, so the pass's last batch has resolved:
  // the outputs made big enough.
  core::Status ensure_buffers(const RgbdFrame& frame, const Layout& layout);
  // Plane images taken over from their writer's family, the slot bound (one
  // copy where it is copied) and the overlap mask's parameters written, into
  // `batch`; `*input` is the slot's buffer the passes bind. Buffers are taken
  // over by record_frames, once each.
  core::Status record_inputs(core::CommandBatch& batch, const RgbdFrame& frame,
                             const Layout& layout, const core::HostSlot& slot,
                             VkBuffer* input);
  // The passes, recorded into `batch` after the inputs; the outputs hold
  // the result once the batch has run.
  core::Status record_passes(core::CommandBatch& batch, const RgbdFrame& frame,
                             const Layout& layout, VkBuffer input,
                             core::GpuStageScope* stage);
  // After a batch that failed and may still run: let go of the device depth
  // and planes it may read.
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

  // Each frame's depth and colour on the host, copied into its one slot, the
  // colour range at minStorageBufferOffsetAlignment, and a word for a frame
  // with neither, which its batch binds all the same; made at the first
  // record, grown as a larger frame asks. The host only writes
  // the slot (core::HostAccess::SequentialWrite), so where the ring is
  // copied, as on a discrete GPU, it is write-combined staging.
  std::unique_ptr<core::HostRing> inputs_;
  // The two cameras and the transform between them, which both passes mask
  // a frame with colour by, written inline each such frame; bound once, at
  // create.
  core::Buffer overlap_;
  // The outputs the fusion tiers read, device-local, leased to the
  // DeviceFrames handed out; reused only once no frame holds the lease.
  std::shared_ptr<Output> depth_out_;
  std::shared_ptr<Output> color_out_;
  // With the outlier filter on: its pass, made at create, and the undistorted
  // depth it reads, device-local, made at the first record and grown as a
  // larger frame asks once the pass's last batch has resolved.
  std::unique_ptr<detail::DepthOutlierPass> outliers_;
  core::Buffer undistorted_;
};

}  // namespace volumetric_kit::sensor
