// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stereo/stereo_rectifier.hpp
/// @brief A calibrated stereo pair undistorted and rectified on the GPU: the
///        row-aligned pinhole pair a stereo matcher reads.
///
/// Any stereo source's pair (@ref StereoFrame) comes out as two views of one
/// pinhole camera, the right one a baseline along the left one's +x, so a
/// point lies on the same row in both. The rectification is computed once,
/// from the calibration; each frame is one compute pass a view, on any GPU
/// the package runs on (Vulkan, MoltenVK on Apple). The views stay on the
/// device, in buffers CUDA can import where the device exports memory, for a
/// matcher such as Fast-FoundationStereo to read in place.

#include <cstdint>
#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/base/stage_metrics.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/stereo/export.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @brief The rectified pair: both views are @ref camera, the right one
///        @ref baseline metres along the left one's +x.
///
/// A point at depth `z` in the rectified left camera shows in the right view
/// `camera.intrinsics.fx * baseline / z` pixels to the left, on the same row:
/// what a matcher's disparity turns back into depth.
struct StereoRig {
  /// Both views' camera: pinhole (no lens), at the rectified size.
  camera::CameraModel camera;
  double baseline = 0.0;  ///< Metres; above 0.
  /// The rectified left camera's frame to the source left camera's: a
  /// rotation, so depth from the pair is posed through it.
  camera::Mat4d rectified_to_left = camera::Mat4d(1.0);
};

/// @brief Both views of one frame, rectified on the GPU.
///
/// Each view is `rig.camera.size` pixels of @ref
/// StereoRectifier::Options::output, rows packed: `Luma8` a byte a pixel, four
/// to a word, the first in the low byte; `Rgba8` a word a pixel, `R | G << 8 |
/// B << 16 | coverage << 24`, R'G'B' as the source pictures encoded it. A pixel
/// whose ray misses its source picture, or passes its lens's fold (where the
/// lens model stops carrying rays outward and turns back), is 0, coverage
/// included. The pass has finished when the pair is handed out.
///
/// The buffers are device-local storage buffers, the rectifier's own, one a
/// view, leased to the pair. **Use each pair in turn -- rectify, read, drop
/// -- and every pair has the same two buffers**: nothing is allocated a
/// frame, and an importer maps each buffer once. A buffer is written again
/// only once no pair holds it, so a pair kept past the next keeps its
/// contents, and the next gets a new buffer -- a new allocation, and a new
/// import for CUDA -- each frame that happens. **Drop the last copy of a pair
/// only once every read of its views has completed** -- CUDA's work
/// synchronized, the Vulkan batch waited: the next @ref
/// StereoRectifier::rectify may write them as soon as it is gone.
///
/// Where the device exports memory (`core::Device::exports_memory`), each
/// buffer is on memory of its own, its descriptor (`core::ExportedBuffer::fd`)
/// open until an importer takes it. CUDA maps a view through
/// `CudaBufferMapping::map` when its descriptor is open, a buffer not yet
/// imported, and reuses its mapping otherwise; an address seen before may be
/// a new buffer, so the descriptor, not the address, tells. The views are
/// handed out released to `VK_QUEUE_FAMILY_EXTERNAL` (@ref queue_family): a
/// reader outside Vulkan reads them as they are, and a Vulkan reader takes
/// them over from there (`CommandBatch::acquire`) and need not hand them back
/// -- the pass writes every word, so it takes them without a transfer.
/// Elsewhere the descriptor is closed and the buffers stay with the device's
/// queue family. Only `CudaBufferMapping::map` may change a view, taking its
/// descriptor: moving or resetting its `buffer` would leave the next pass
/// writing freed memory.
struct RectifiedPair {
  std::shared_ptr<core::ExportedBuffer> left;   ///< The left view.
  std::shared_ptr<core::ExportedBuffer> right;  ///< The right view.
  /// The queue family the views are owned by, as `YuvImage::queue_family`:
  /// @ref kQueueFamilyExternal for exported buffers, else
  /// @ref kQueueFamilyIgnored.
  std::uint32_t queue_family = kQueueFamilyIgnored;
  StereoRig rig;  ///< The views' camera and baseline.
  /// The frame they came from: colour and times pass through, and its
  /// pictures stay held. A picture in a buffer the pass took over -- the left
  /// and right ones, and a colour picture in one of their buffers -- reads
  /// @ref kQueueFamilyIgnored here, so a later stage reads this frame, not
  /// the one passed to @ref StereoRectifier::rectify.
  StereoFrame source;
};

/// @brief Undistorts and rectifies a calibrated stereo pair on the GPU,
///        resampled to one size.
///
/// The rectification is computed once, at @ref create, on the host:
/// Bouguet's method, as OpenCV's `stereoRectify` computes it with
/// `CALIB_ZERO_DISPARITY` -- the relative rotation split between the two
/// cameras, then the baseline turned onto +x; one pinhole camera for both
/// views, its focal length the two cameras' vertical one scaled to the
/// rectified height, and @ref Options::alpha choosing between only valid
/// pixels and every source pixel. A pair that is rectified already -- no
/// rotation between the cameras, no lens, one camera model, the right camera
/// along +x alone, as a RealSense's infrared pair -- is only resampled: its
/// camera scaled to the size by one factor, so its pixels keep their shape,
/// about the centres; alpha chooses between filling the view, the picture
/// cropped (0), and fitting the whole picture, the view bordered (1), which
/// differ only where the size changes the aspect ratio.
///
/// Each frame is one compute pass a view: each rectified pixel's ray turned
/// into the source camera, through its lens, and the source picture sampled
/// there bilinearly and converted as `GpuFramePrep` converts colour (the
/// same shader code). The pictures come in any layout the package reads --
/// luma alone, YUYV, NV12, I420 -- on the host, copied into one persistent
/// slot (`core::HostRing`) as `GpuFramePrep` stages host planes, or in a
/// buffer on the device, taken over from `YuvImage::queue_family`, once a
/// buffer. Nothing is allocated a frame while each pair is dropped before the
/// next is rectified (@ref RectifiedPair).
///
/// @code
/// VKC_ASSIGN(auto rectifier,
///            StereoRectifier::create(device, allocator, camera.info(),
///                                    {{960, 544}, 0.0f,
///                                     StereoRectifier::Output::Luma8}));
/// VKC_ASSIGN(RectifiedPair pair, rectifier->rectify(frame));
/// // pair.left->buffer and pair.right->buffer hold 960 x 544 luma bytes.
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object and every pair it hands out. Not thread-safe.
class VS_STEREO_API StereoRectifier final {
 public:
  /// @brief What each view is written as.
  enum class Output : std::uint8_t {
    Luma8,  ///< Y' alone, full range, a byte a pixel: a mono matcher's input.
    Rgba8,  ///< R'G'B' and the coverage byte, a word a pixel.
  };

  /// @brief How @ref create rectifies.
  struct Options {
    /// The rectified size; {0, 0}: the left camera's.
    camera::ImageSize size{};
    /// 0: only pixels both views see; 1: every source pixel (OpenCV's
    /// meaning); between, between. In [0, 1]. A pair rectified already: 0
    /// crops its picture to the view's shape, 1 fits all of it.
    float alpha = 0.0f;
    Output output = Output::Rgba8;  ///< What each view is written as.
  };

  /// @brief Compute the rectification of @p info's pair and build its pass on
  ///        @p device.
  /// @param device     The device the pairs are rectified on (must outlive
  ///                   this object and its pairs).
  /// @param allocator  Where the views' buffers come from (must outlive this
  ///                   object and its pairs).
  /// @param info       The pair's calibration, and any colour camera.
  /// @param options    The rectified size, alpha and output.
  /// @return The rectifier; `Status::Code::InvalidArgument` for a camera
  ///         that `camera::check_camera_model` refuses or that is past
  ///         float's range, cameras of different sizes, a `left_to_right` or
  ///         `left_to_color` that is not rigid, cameras at one place, a pair
  ///         whose right camera is not along the left one's +x after the
  ///         rotation is split (a vertical pair, or the two swapped), an
  ///         alpha outside [0, 1], a size with one side 0, a view past a
  ///         single dispatch or a storage binding, or a rectification that
  ///         comes out degenerate; or a device that lacks what the package's
  ///         kernels need, or a pipeline or buffer failure.
  static core::Result<std::unique_ptr<StereoRectifier>> create(
      const core::Device& device, core::Allocator& allocator,
      const StereoInfo& info, const Options& options);

  ~StereoRectifier();
  StereoRectifier(const StereoRectifier&) = delete;
  StereoRectifier& operator=(const StereoRectifier&) = delete;
  StereoRectifier(StereoRectifier&&) = delete;
  StereoRectifier& operator=(StereoRectifier&&) = delete;

  /// @return The rectified pair's camera, baseline and rotation.
  const StereoRig& rig() const noexcept;

  /// @brief The sensor a matcher's depth makes, for whoever prepares it: depth
  ///        in the rectified left camera (`SensorInfo::depth`, the rig's
  ///        camera), the colour camera -- the third one, or else the left
  ///        camera itself, lens included, whose picture is the colour -- and
  ///        the depth camera's frame to the colour camera's
  ///        (`left_to_color * rectified_to_left`).
  /// @return The sensor's cameras; its id, model and rates are the source's
  ///         to fill.
  const SensorInfo& depth_info() const noexcept;

  /// @brief Rectify both views of @p frame: one submit, both passes, and its
  ///        wait.
  /// @param frame    The pair. Host planes are copied here; device planes,
  ///                 their writer finished, are read before this returns,
  ///                 each buffer taken over once: pictures in one buffer (a
  ///                 side-by-side frame, a colour picture beside them) name
  ///                 the family that wrote it. @p frame keeps the families it
  ///                 names; hand on @ref RectifiedPair::source instead,
  ///                 whose pictures name what the pass left them in.
  /// @param metrics  Optional: a `"stereo rectify"` row, the staging and both
  ///                 passes on the host and on the device.
  /// @return The pair; `Status::Code::InvalidArgument` for a view with no
  ///         planes, planes as images (a buffer or the host only), a picture
  ///         that `GpuFramePrep` would refuse as a frame's colour (its size
  ///         not its camera's, an unknown layout or chroma location, a bad
  ///         matrix, YUYV of odd width, strides, offsets, overlap or a buffer
  ///         it cannot bind), a `queue_family` the device lacks, or pictures
  ///         in one buffer that name different writers' queue families;
  ///         `Status::Code::OutOfMemory` for a host allocation failure; or a
  ///         buffer or submit failure, after which the rectifier refuses
  ///         every frame, as the device may still run the pass.
  core::Result<RectifiedPair> rectify(const StereoFrame& frame,
                                      core::StageMetrics* metrics = nullptr);

 private:
  struct Impl;
  explicit StereoRectifier(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
