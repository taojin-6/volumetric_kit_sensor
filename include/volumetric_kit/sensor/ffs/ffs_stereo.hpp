// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file ffs/ffs_stereo.hpp
/// @brief Depth from a rectified stereo pair by Fast-FoundationStereo (FFS), on
///        an NVIDIA GPU: TensorRT runs the network, CUDA kernels make its input
///        and turn its disparity into depth, written straight into a Vulkan
///        buffer that `GpuFramePrep` reads in place.
///
/// Built with VS_WITH_FFS (Linux, CUDA and TensorRT) only. No CUDA or
/// TensorRT type reaches this header.

#include <filesystem>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/ffs/export.hpp"
#include "volumetric_kit/sensor/rgbd_frame.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"

namespace volumetric_kit::sensor {

/// @brief Fast-FoundationStereo on any rectified pair: depth in the rectified
///        left camera, on the device.
///
/// The model is an ONNX export of FFS (`tools/ffs/export_onnx.py` makes it,
/// once per checkpoint and size): inputs `left_image` and `right_image`, float
/// `1 x 3 x H x W`, R'G'B' less ImageNet's mean over its deviation; output
/// `disparity`, float `1 x 1 x H x W`, in pixels. Its size is the rig's
/// rounded up to a multiple of 32, which FFS needs: the views are padded by
/// repeating their last row and column, and the padding cut off again.
///
/// @ref create builds the model into a TensorRT plan, which takes minutes, and
/// keeps it in @ref Options::plan_cache, named for the GPU, the TensorRT
/// version, the model's bytes and its size; a later create on the same GPU
/// loads it in seconds. A plan TensorRT will not load is built again; one
/// that cannot be written runs unkept, with a warning.
///
/// Each @ref depth reads the rectifier's views in place, through CUDA
/// (`CudaBufferMapping`, each buffer mapped once, by the first matcher that
/// reads it, which takes its descriptor: one rectifier feeds one matcher);
/// a CUDA kernel makes the
/// network's input from them (luma repeated in all three channels, or the
/// R'G'B' of `Rgba8`), TensorRT runs, and a second kernel turns the disparity
/// `d` at pixel `x` into depth `fx * baseline / d`, 16-bit, written into a
/// Vulkan buffer of its own (`RgbdFrame::depth_device`, taken over from
/// @ref kQueueFamilyExternal). A disparity that is not above 0, that points
/// off the right view (`x - d < 0`), or whose depth is outside
/// [@ref Options::min_depth, @ref Options::max_depth] is "no return" (0); so
/// is, in `Rgba8` views, a pixel the rectifier did not see (coverage 0) in
/// the left view, or matched to one in the right view. `Luma8` views carry
/// no coverage: a border the rectifier leaves -- alpha above 0, or alpha 1 on
/// a pair rectified already whose aspect the size changes -- is matched as
/// black picture, so take `Rgba8` or alpha 0 there. The host only orders the
/// work and waits for it, asleep; nothing is allocated a frame while no frame
/// is kept past the next.
///
/// Up to eight views stay mapped, the least recently read let go first, and
/// a view read again after its mapping went is refused. A view the rectifier
/// replaced -- each frame a pair is kept past the next -- keeps its mapping,
/// and its memory, until then.
///
/// @code
/// VKC_ASSIGN(auto rectifier, StereoRectifier::create(
///     device, allocator, camera.info(),
///     {{960, 544}, 0.0f, StereoRectifier::Output::Luma8}));
/// FfsStereo::Options options{model, cache};
/// options.disparity_sigma = 0.04f;  // this camera's, as measured
/// VKC_ASSIGN(auto ffs, FfsStereo::create(device, allocator, *rectifier,
///                                        options));
/// VKC_ASSIGN(auto prep, GpuFramePrep::create(device, allocator));
/// VKC_ASSIGN(RectifiedPair pair, rectifier->rectify(frame));
/// VKC_ASSIGN(RgbdFrame rgbd, ffs->depth(pair));
/// VKC_ASSIGN(const DeviceFrame prepared, prep->prepare(rgbd));
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object and every frame it hands out. Not thread-safe.
class VS_FFS_API FfsStereo final {
 public:
  /// @brief The model, where its plans are kept, the depths kept, and the
  ///        disparity's noise.
  struct Options {
    /// The FFS ONNX (`tools/ffs/export_onnx.py`).
    std::filesystem::path model;
    /// The directory built plans are kept in, made if missing; not empty.
    std::filesystem::path plan_cache;
    /// Nearer depth is "no return" (metres); above 0.
    float min_depth = 0.3f;
    /// Farther depth is "no return" (metres); above @ref min_depth. The
    /// depth's unit is the larger of 0.1 mm and `max_depth / 65535`.
    float max_depth = 6.0f;
    /// The disparity's frame-to-frame standard deviation, in pixels at the
    /// rig's size, which every frame's noise is made from
    /// (`RgbdFrame::depth_noise`: @ref DepthNoise::stereo with the rig's fx
    /// and baseline). It is the camera's and the model's, measured on the
    /// camera, so FFS has no default: 0, the default, leaves the frames'
    /// noise unknown. A D455's infrared pair with the 23-36-37 checkpoint at
    /// 1280 x 720, its projector off, measured 0.04 px. Finite and at least
    /// 0.
    float disparity_sigma = 0.0f;
  };

  /// @brief Build or load the plan for @p rectifier's views.
  /// @param device     The device the views and the depth are on (must
  ///                   outlive this object and its frames); one that exports
  ///                   memory, on an NVIDIA GPU.
  /// @param allocator  Where the depth buffers come from (must outlive this
  ///                   object and its frames).
  /// @param rectifier  The rectifier whose pairs come in: its rig (camera,
  ///                   baseline) and its depth view (`depth_info()`: the
  ///                   colour camera and where the depth camera sits from
  ///                   it), read here.
  /// @param options    The model, the plan cache, the depth range and the
  ///                   disparity's noise.
  /// @return The matcher; `Status::Code::InvalidArgument` for a depth range
  ///         that is not `0 < min_depth < max_depth`, a `disparity_sigma`
  ///         that is not finite and at least 0, no plan cache, a model
  ///         that TensorRT cannot parse, or whose inputs and output are not
  ///         those above at the rig's size rounded up to 32;
  ///         `Status::Code::NotFound` for no model file;
  ///         `Status::Code::IoError` for a plan cache that cannot be made,
  ///         before any build; `Status::Code::Unsupported` where CUDA does
  ///         not reach the device's GPU or the device does not export memory;
  ///         or a CUDA, TensorRT, file or buffer failure.
  static core::Result<std::unique_ptr<FfsStereo>> create(
      const core::Device& device, core::Allocator& allocator,
      const StereoRectifier& rectifier, const Options& options);

  ~FfsStereo();
  FfsStereo(const FfsStereo&) = delete;
  FfsStereo& operator=(const FfsStereo&) = delete;
  FfsStereo(FfsStereo&&) = delete;
  FfsStereo& operator=(FfsStereo&&) = delete;

  /// @brief Depth for @p pair: one run of the network, and its wait.
  ///
  /// The frame's depth is `RgbdFrame::depth_device`, in the rectified left
  /// camera (`depth_camera`, the rig's), its writer finished and its queue
  /// family @ref kQueueFamilyExternal. Its colour is the pair's third
  /// camera's picture, or else the left one's, and its colour camera and
  /// `depth_to_color` the rectifier's `depth_info()`'s; its depth noise
  /// @ref DepthNoise::stereo of @ref Options::disparity_sigma and the rig's
  /// fx and baseline; its colour encoding, pose and times are the pair's; it
  /// holds the pair's pictures.
  /// @param pair     A pair from the rectifier @ref create was given, its
  ///                 views exported (@ref kQueueFamilyExternal).
  /// @param metrics  Optional: `"ffs input"`, `"ffs network"` and `"ffs
  ///                 depth"` rows of device time, and a `"ffs"` row of host
  ///                 time, the wait included.
  /// @return The frame; `Status::Code::InvalidArgument` for a pair of another
  ///         size, views that are not exported or not `Luma8` or `Rgba8` at
  ///         that size, or a view imported by something else or whose
  ///         mapping was let go; or a CUDA,
  ///         TensorRT or buffer failure, after which every pair is refused,
  ///         as the GPU may still be writing.
  core::Result<RgbdFrame> depth(const RectifiedPair& pair,
                                core::StageMetrics* metrics = nullptr);

 private:
  struct Impl;
  explicit FfsStereo(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
