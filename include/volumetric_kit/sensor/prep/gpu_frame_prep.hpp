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
/// take a `Buffer`). Driver-neutral: it takes a @ref RawFrame, whoever made
/// it.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/buffer.hpp"
#include "volumetric_kit/recon/core/camera_params.hpp"
#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/core/compute_kernel.hpp"
#include "volumetric_kit/recon/core/descriptor.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/gpu_timer.hpp"
#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/core/stage_metrics.hpp"
#include "volumetric_kit/recon/sensor/raw_frame.hpp"
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
/// The @ref Allocator must outlive the frame, as it must the pass.
struct DeviceFrame {
  /// Row-major depth in metres, `depth_camera.width * height` floats; 0 where
  /// the sensor had no return or the lens maps outside its image.
  std::shared_ptr<const Buffer> depth;
  /// Row-major colour, `color_camera.width * height` words: R, G and B in the
  /// low three bytes, and the pixel's coverage in the high one -- 0xFF where
  /// the lens maps inside the captured picture, 0 (and black) where it maps
  /// outside, which `ColorFrame::coverage_in_alpha` has fusion skip. Null
  /// when the frame has no colour.
  std::shared_ptr<const Buffer> color;
  DepthCameraParams depth_camera{};  ///< The undistorted depth camera.
  ColorCameraParams color_camera{};  ///< The undistorted colour camera.
  ColorEncoding color_encoding{};    ///< What @ref color is encoded as.
  std::uint64_t timestamp_ns = 0;    ///< The raw frame's.

  /// @return `true` if this frame carries colour.
  bool has_color() const noexcept { return color != nullptr; }
};

/// @brief Undistorts a @ref RawFrame's depth and colour, and converts its
///        colour to R'G'B', in two compute passes.
///
/// Undistorting keeps each camera's intrinsics and drops its lens: every pixel
/// of the pinhole image is sampled from where @ref distort_normalized puts it
/// in the captured one. Colour is sampled bilinearly, luma and chroma each at
/// its own siting, and converted by the picture's matrix and range; depth is
/// sampled at the nearest pixel, so an edge never blends a foreground and a
/// background depth into a point between them.
///
/// Every check on the frame is made before anything is uploaded, so a refused
/// frame leaves the pass and the frames it handed out as they were.
///
/// @warning The @ref Device and @ref Allocator passed to @ref create must
///          outlive this object; it stores references to them. Not
///          thread-safe.
class VR_SENSOR_UTILS_API GpuFramePrep {
 public:
  /// @return The pass; or a non-OK @ref Status if a pipeline, descriptor
  ///         object or buffer fails to build.
  static Result<GpuFramePrep> create(Device& device, Allocator& allocator);

  ~GpuFramePrep() = default;
  GpuFramePrep(GpuFramePrep&&) noexcept = default;
  GpuFramePrep& operator=(GpuFramePrep&&) noexcept = default;
  GpuFramePrep(const GpuFramePrep&) = delete;
  GpuFramePrep& operator=(const GpuFramePrep&) = delete;

  /// @brief Upload @p frame, undistort and convert it, and hand the result
  ///        over as buffers on the device.
  /// @param frame    The frame; read during the call only.
  /// @param metrics  Optional @ref StageMetrics collecting a `"frame prep"`
  ///                 row: the upload and both passes on the host, and on the
  ///                 device the frame's copy up and the two dispatches.
  ///                 `nullptr` measures nothing.
  /// @return The frame, which holds its buffers; @ref
  ///         Status::Code::InvalidArgument for a moved-from pass, a frame
  ///         without depth, a depth range that is not finite with
  ///         `0 < min_depth < max_depth` (0 being the pass's "no return"), a
  ///         camera or picture that is empty, not finite or disagrees with
  ///         its image, or an image past a single dispatch (16.7 M pixels);
  ///         @ref Status::Code::Unsupported for a colour encoding
  ///         @ref is_canonical refuses; otherwise a buffer or dispatch
  ///         failure.
  Result<DeviceFrame> prepare(const RawFrame& frame,
                              StageMetrics* metrics = nullptr);

  /// @return `true` if this owns its pipelines (`false` when moved-from).
  bool valid() const noexcept { return depth_kernel_.valid(); }

 private:
  GpuFramePrep() = default;

  // An output of at least `bytes`: the one held, when no DeviceFrame still
  // holds it too and it is big enough, else a new one.
  Status ensure_output(std::shared_ptr<Buffer>& buffer, VkDeviceSize bytes,
                       const char* name);

  // Borrowed (must outlive this).
  Device* device_ = nullptr;
  Allocator* allocator_ = nullptr;

  std::uint32_t max_workgroup_count_x_ = 0;
  VkDeviceSize max_storage_buffer_range_ = 0;

  ComputeKernel depth_kernel_;
  ComputeKernel color_kernel_;
  DescriptorPool pool_;
  GpuTimer gpu_timer_;

  // The raw inputs, device-local and filled through the pass's batch, grown
  // to the largest frame seen and kept.
  // TODO(sensor): zero-copy inputs from a hardware decoder's frames.
  Buffer depth_in_;
  Buffer color_in_;
  // The frame on the host side, host-visible and kept like the inputs, so
  // passes on several threads never allocate staging at once.
  Buffer staging_;
  // The outputs the fusion tiers read, device-local, shared with the
  // DeviceFrames handed out; reused only once no frame holds them.
  std::shared_ptr<Buffer> depth_out_;
  std::shared_ptr<Buffer> color_out_;
};

/// @brief Prepare several cameras' frames at once, each on its own thread
///        with its own pass: they do not depend on one another, so their
///        uploads, submits and waits overlap. The @ref Device must be shared
///        by the passes, and may be.
/// @param preps   One pass per camera; `preps[i]` prepares `frames[i]`, and
///                each is used by one thread only for the call.
/// @param frames  One entry per camera, an empty one skipped -- a rig's set
///                (`OrbbecRig::poll_raw_set`).
/// @return One @ref DeviceFrame per present frame, empty where the frame
///         was; @ref Status::Code::InvalidArgument for fewer passes than
///         frames; otherwise the lowest camera's failure, once every thread
///         has finished.
VR_SENSOR_UTILS_API Result<std::vector<std::optional<DeviceFrame>>> prepare_set(
    std::vector<GpuFramePrep>& preps,
    const std::vector<std::optional<RawFrame>>& frames);

}  // namespace volumetric_kit::recon::sensor
