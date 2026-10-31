// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file color_fit.hpp
/// @brief The rig's colour responses (`camera::ColorResponse`) fitted from
///        posed, prepared sets of a static scene, on the GPU: what the
///        `color_fit` tool runs (DESIGN.md, "colour response fit").
///
/// An internal static library: no export macro, not installed. Its one user
/// is the tool; its test reaches the same entry points.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/descriptor.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/color_response.hpp"
#include "volumetric_kit/sensor/frame/device_frame.hpp"
#include "volumetric_kit/sensor/rgbd_sensor.hpp"

namespace volumetric_kit::sensor::color_fit {

/// @brief One kept cross-view sample, as the kernels store it (scalar
///        layout, `shaders/color_fit.glsl`'s `Sample`).
struct Sample {
  float a[3];  ///< A's colour at sigma 8, linear.
  float b[3];  ///< B's colour at the same point, linear.
  float qa;    ///< A's pixel's squared pinhole radius, uncapped.
  float qb;    ///< B's.
  /// `a | b << 8 | held << 16`: the cameras, in the array's order, and the
  /// hold-out half (0 trains, 1 is held out).
  std::uint32_t cams;
};
static_assert(sizeof(Sample) == 36, "Sample layout drift");

/// @brief How much a fit takes.
struct Options {
  /// The sets @ref ColorFit::add takes; each keeps about
  /// `samples / sets` samples.
  std::uint32_t sets = 8;
  /// The samples kept in all, by a hashed ratio per set; the 2026-10 study
  /// fitted 1 M of 8.56 M.
  std::uint64_t samples = 1'000'000;
  /// A camera with fewer kept pixels (as A or as B) is refused.
  std::uint64_t min_camera_samples = 5'000;
  /// Two cameras with fewer samples between them, both ways, are not joined
  /// in the overlap graph, which must join every camera.
  std::uint64_t min_pair_samples = 1'000;
};

/// @brief An ordered pair's samples and RMS residuals, in 8-bit levels.
struct PairResidual {
  std::uint32_t a = 0;                 ///< A's index in the array.
  std::uint32_t b = 0;                 ///< B's.
  std::uint64_t train_samples = 0;     ///< In half 0.
  std::uint64_t held_out_samples = 0;  ///< In half 1.
  double identity = 0.0;               ///< Held out, with no response.
  double train = 0.0;                  ///< The fit, on half 0.
  double held_out = 0.0;               ///< The fit, on half 1.
};

/// @brief A camera's fitted response.
struct Camera {
  std::string id;                  ///< Its sensor's.
  camera::ColorResponse response;  ///< Gauged (G2).
  std::uint64_t samples = 0;       ///< Its kept pixels, as A or as B.
  /// The share of its covered pixels, on the last set, that no channel held
  /// at 255 and the response takes to 255.
  double pushed_to_white = 0.0;
};

/// @brief What @ref ColorFit::solve found: RMS in 8-bit sRGB levels at
///        sigma 8.
struct Report {
  std::vector<Camera> cameras;         ///< In the array's order.
  std::vector<PairResidual> pairs;     ///< Every ordered pair.
  double rms_identity = 0.0;           ///< Held out, with no response.
  double rms_fit = 0.0;                ///< The fit, on half 0.
  double rms_held_out = 0.0;           ///< The fit, on half 1.
  std::uint64_t train_samples = 0;     ///< Half 0's.
  std::uint64_t held_out_samples = 0;  ///< Half 1's.
  int iterations = 0;                  ///< Levenberg-Marquardt steps taken.
  int evaluations = 0;                 ///< Passes over the samples on the GPU.
  double solve_ms = 0.0;               ///< The solve's wall time.
  /// The input with every camera's response; the file's other fields as
  /// they were.
  camera::ArrayCalibration calibration;
};

/// @brief The samples' RMS residuals under given responses.
struct Residuals {
  double train = 0.0;                  ///< Half 0.
  double held_out = 0.0;               ///< Half 1.
  std::uint64_t train_samples = 0;     ///< Half 0's.
  std::uint64_t held_out_samples = 0;  ///< Half 1's.
  /// Per ordered pair, `identity` unset.
  std::vector<PairResidual> pairs;
};

/// @brief The Huber IRLS normal equations at given responses, on half 0.
///
/// The free parameters, in order: camera 0's a and b, then for each other
/// camera its three log-gains, a and b (camera 0's log-gains are pinned).
/// The step `delta` solves `(h + lambda diag(h)) delta = -g`.
struct System {
  std::size_t n = 0;      ///< The parameters: 5 per camera less 3.
  std::vector<double> h;  ///< n x n, row-major: sum of w J J^T.
  std::vector<double> g;  ///< n: sum of w J r.
  double loss = 0.0;      ///< The Huber loss.
};

/// @brief The parameters' Levenberg-Marquardt step for @p system, by
///        Cholesky in double.
/// @return The step; `Status::Code::InvalidArgument` where the damped matrix
///         is not positive definite.
core::Result<std::vector<double>> step(const System& system, double lambda);

/// @return @p report as the tool prints it: the RMS overall and per ordered
///         pair, then each camera's samples, gains, correction at r 0 and at
///         r_max in stops, r_max and the share pushed to 255.
std::string format_report(const Report& report);

/// @brief The fit: samples a set's overlaps on the GPU as it arrives, and
///        solves for every camera's response once the sets are in.
///
/// Every per-pixel step runs on the GPU: blurring each camera's colour at
/// sigma 2 and 8, sampling each ordered pair from A's depth pixels and gating
/// the samples, keeping about `Options::samples` of them by a hashed ratio,
/// the radius histogram, the normal equations, the residuals and the gauge's
/// moments. The host solves the small system.
///
/// @warning The `Device` and `Allocator` must outlive it. Not thread-safe.
class ColorFit {
 public:
  /// @brief Build the kernels.
  /// @param sensors  The array's, in its order: two or more, distinct ids,
  ///                 one `model`, each with colour and depth.
  /// @return The fit; `Status::Code::InvalidArgument` naming the sensors
  ///         refused (fewer than two, a repeated id, models that differ,
  ///         one without colour or depth) or options of no sets or samples;
  ///         or a kernel's failure.
  static core::Result<std::unique_ptr<ColorFit>> create(
      core::Device& device, core::Allocator& allocator,
      std::vector<SensorInfo> sensors, const Options& options = {});

  ~ColorFit();
  ColorFit(const ColorFit&) = delete;
  ColorFit& operator=(const ColorFit&) = delete;
  ColorFit(ColorFit&&) = delete;
  ColorFit& operator=(ColorFit&&) = delete;

  /// @brief Sample @p set on the GPU: one submit, one wait.
  /// @param set  Prepared with identity responses, posed; held only during
  ///             the call.
  /// @return OK; `Status::Code::InvalidArgument` for a set of another size,
  ///         a frame without colour or depth, frames of another size than
  ///         the first set's, or a set past `Options::sets`, with nothing
  ///         sampled; or a buffer or submit failure.
  core::Status add(const DeviceFrameSet& set);

  /// @brief Fit, gauge and check the responses.
  ///
  /// Each camera's r_max is the 99th percentile of its kept pixels' radii.
  /// The Huber (3 levels) Gauss-Newton IRLS with Levenberg-Marquardt damping
  /// fits half 0 with camera 0's log-gains pinned; then the G2 gauge scales
  /// every gain, per channel, so the mean of gain F over both pixels of every
  /// kept sample is 1; half 1 is held out.
  /// @param calibration  The array's, which names every sensor and records
  ///                     the `color_controls` the sets were taken at.
  /// @return The report, its calibration the input with the responses; or
  ///         `Status::Code::InvalidArgument`, writing nothing, for a
  ///         calibration without `color_controls` or without a sensor, no
  ///         set added, a camera with fewer than `min_camera_samples` kept
  ///         pixels or an overlap graph that leaves cameras apart (named),
  ///         a fitted F that is not above 0 up to r_max, or a held-out RMS
  ///         not below the identity's; or a buffer or submit failure.
  core::Result<Report> solve(const camera::ArrayCalibration& calibration);

  /// @brief The kept samples' residuals under @p responses, one per camera.
  core::Result<Residuals> residuals(
      const std::vector<camera::ColorResponse>& responses);

  /// @brief The normal equations at @p responses, one per camera.
  core::Result<System> system(
      const std::vector<camera::ColorResponse>& responses);

  /// @return The kept samples, read back, in their buffer's order.
  core::Result<std::vector<Sample>> samples();

 private:
  ColorFit() = default;

  struct Segment {
    std::uint64_t first = 0;
    std::uint64_t count = 0;
    std::uint32_t pair = 0;  // its ordered pair's index
  };
  struct Rows;  // the per-pair sums of one evaluation (color_fit.cpp)

  core::Status ensure_set_buffers(const DeviceFrameSet& set);
  core::Status ensure_chunks();
  core::Result<Rows> evaluate(
      const std::vector<camera::ColorResponse>& responses);
  core::Result<std::vector<double>> pushed(
      const std::vector<camera::ColorResponse>& responses);

  core::Device* device_ = nullptr;
  core::Allocator* allocator_ = nullptr;
  std::vector<SensorInfo> sensors_;
  Options options_;
  std::uint32_t max_groups_ = 0;
  VkDeviceSize max_range_ = 0;

  core::ComputeKernel blur_;
  core::ComputeKernel sample_;
  core::ComputeKernel normal_equations_;
  core::ComputeKernel pushed_;
  core::DescriptorPool pool_;

  // The set being sampled: its cameras, depth and words, and the blurred
  // colour (sigma 2 then sigma 8) with the blur's rows.
  std::vector<std::uint32_t> depth_at_, color_at_;
  std::vector<std::uint32_t> depth_size_, color_size_;  // w | h << 16
  std::uint32_t depth_total_ = 0, color_total_ = 0, color_max_ = 0;
  core::Buffer cameras_, depths_, words_, rows_, blurred_;
  // Across sets: the counters, the samples and the radius histogram.
  core::Buffer counters_, samples_, histogram_;
  std::uint32_t capacity_ = 0;
  std::uint32_t sets_ = 0;
  std::uint64_t kept_ = 0;
  std::vector<Segment> segments_;

  // The solve's: the parameters, the chunks and pairs, the partial and
  // final rows, and the white counts.
  core::Buffer params_, chunks_, pairs_, partials_, rows_out_, counts_;
  std::uint32_t chunk_count_ = 0;
  bool chunks_ready_ = false;
};

}  // namespace volumetric_kit::sensor::color_fit
