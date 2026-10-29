// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file ffs/stereo_depth_sensor.hpp
/// @brief Any calibrated stereo camera as an RGB-D sensor: its pairs
///        rectified on the GPU and matched by Fast-FoundationStereo, on a
///        thread of its own.
///
/// Built with VS_WITH_FFS (Linux, CUDA and TensorRT) only.

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/ffs/export.hpp"
#include "volumetric_kit/sensor/ffs/ffs_stereo.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"
#include "volumetric_kit/sensor/stereo/stereo_rectifier.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor {

/// @brief Depth from a stereo camera (@ref IStereoSensor): rectification
///        (@ref StereoRectifier) and FFS (@ref FfsStereo), as an
///        @ref IRgbdSensor, so everything downstream -- `GpuFramePrep`, a
///        `SensorArray` -- sees one more depth sensor.
///
/// A thread of its own takes each new pair from the camera, rectifies it and
/// runs FFS, then holds the frame for @ref poll or @ref drain and calls the
/// @ref set_frame_notify callback: @ref poll stays "the newest frame held",
/// a frame one FFS run (28 ms at 960 x 544 on an RTX 4090) after its pair.
/// Its timestamps are the pair's capture times. Its @ref info is the
/// rectifier's `depth_info()`: depth in the rectified left camera, and the
/// colour camera.
///
/// The camera is called from one thread at a time, never two at once:
/// @ref start and @ref stop call its `start` and `stop` on the caller's thread,
/// the thread its `poll` while it runs. The thread's Vulkan submits share the
/// device's queue with the caller's (`core::Device` serialises them).
///
/// @code
/// VKC_ASSIGN(auto camera, RealSenseStereo::open({}));  // any IStereoSensor
/// VKC_ASSIGN(std::unique_ptr<IRgbdSensor> sensor,
///            StereoDepthSensor::create(
///                device, allocator, std::move(camera),
///                {{{960, 544}, 0.0f, StereoRectifier::Output::Luma8},
///                 {model, cache}}));
/// VKC_ASSIGN(auto prep, GpuFramePrep::create(device, allocator,
///                                            sensor->info()));
/// VKC_TRY(sensor->start());
/// // poll, prep->prepare(*frame), fuse, as for any IRgbdSensor
/// @endcode
///
/// @warning The `Device` and `Allocator` passed to @ref create must outlive
///          this object and every frame it hands out. Not thread-safe, as
///          @ref IRgbdSensor: create, start, poll, drain and stop from one
///          thread.
class VS_FFS_API StereoDepthSensor final : public IRgbdSensor {
 public:
  /// @brief How the pairs are rectified and matched.
  struct Options {
    /// The rectified size, alpha and views (`Luma8` for a mono pair).
    StereoRectifier::Options rectify;
    FfsStereo::Options ffs;  ///< The model, its plan cache, the depth range.
  };

  /// @brief Make the rectifier and FFS for @p camera's pair (building FFS's
  ///        plan if it is not cached: minutes). Does not start.
  /// @param device     The device (must outlive this object and its frames).
  /// @param allocator  Where the views and depth come from (must outlive this
  ///                   object and its frames).
  /// @param camera     The stereo camera, opened; not null.
  /// @param options    The rectification and FFS.
  /// @return The sensor; or why @ref StereoRectifier::create or
  ///         @ref FfsStereo::create refused.
  static core::Result<std::unique_ptr<StereoDepthSensor>> create(
      const core::Device& device, core::Allocator& allocator,
      std::unique_ptr<IStereoSensor> camera, const Options& options);

  ~StereoDepthSensor() override;
  StereoDepthSensor(const StereoDepthSensor&) = delete;
  StereoDepthSensor& operator=(const StereoDepthSensor&) = delete;
  StereoDepthSensor(StereoDepthSensor&&) = delete;
  StereoDepthSensor& operator=(StereoDepthSensor&&) = delete;

  /// @return The rectifier's `depth_info()`: depth in the rectified left
  ///         camera, the colour camera, and between them.
  const SensorInfo& info() const noexcept override;

  core::Status set_queue_depth(std::size_t frames) override;

  /// @brief Start the camera, then the thread.
  /// @return OK once running; or why the camera did not start.
  core::Status start() override;

  /// @brief Stop the thread, after the pair it is matching, then the camera;
  ///        the frames held are dropped.
  void stop() noexcept override;

  /// @return The newest frame held; or, once the camera's poll or the GPU
  ///         has failed, that failure, until the next @ref start.
  core::Result<std::optional<RgbdFrame>> poll() override;

  /// @return OK with every frame held appended, oldest first; or the
  ///         failure, as @ref poll.
  core::Status drain(std::vector<RgbdFrame>* out) override;

  /// @return Since the last @ref start: `received` pairs taken from the
  ///         camera; `failed` pairs the rectifier or FFS refused; frames
  ///         `delivered` and `dropped` as any sensor counts them.
  SensorStats stats() const noexcept override;

  /// @brief Called on the thread once a frame is held, and on a failure.
  /// @return OK; `Status::Code::InvalidArgument` once started.
  core::Status set_frame_notify(std::function<void()> notify) override;

 private:
  struct Impl;
  explicit StereoDepthSensor(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
