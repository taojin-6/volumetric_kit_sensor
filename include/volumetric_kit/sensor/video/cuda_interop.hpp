// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/video/cuda_interop.hpp
/// @brief CUDA on the GPU of a recon device: its primary context, a stream,
///        and a buffer the device exported, mapped into CUDA.
///
/// Built with VR_WITH_CUDA (Linux) only, and includes cuda.h. libcuda is
/// loaded at run time, so a program built with it starts where no NVIDIA
/// driver is: there @ref cuda_ordinal_of and @ref CudaContext::open return
/// `Status::Code::Unsupported`. @ref HevcDecoder (NVDEC) and
/// @ref JpegDecoder (nvJPEG) hand their pictures over through these.

#include <cuda.h>

#include <memory>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/recon/core/fwd.hpp"
#include "volumetric_kit/recon/sensor/video/export.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief The CUDA device that is @p device's GPU: Vulkan's `deviceUUID`
///        matched against each CUDA device's.
/// @param device  The device whose GPU is looked for.
/// @return The CUDA ordinal of @p device's GPU; `Status::Code::Unsupported`
///         where libcuda does not load or start, or no CUDA device is that
///         GPU; or `Status::Code::Backend` if CUDA fails to count its
///         devices.
VR_SENSOR_VIDEO_API core::Result<int> cuda_ordinal_of(
    const core::Device& device);

/// @brief The primary CUDA context of a device's GPU, retained while this
///        lives.
///
/// Every holder on a GPU shares the one context, as CUDA's runtime does. It
/// is never left current on a thread: each call that needs it makes it
/// current for its own scope and puts back the context before, and a caller
/// does the same with @ref handle (`cuCtxPushCurrent`, `cuCtxPopCurrent`).
class VR_SENSOR_VIDEO_API CudaContext {
 public:
  /// @brief Retain the primary context of @p device's GPU.
  /// @param device  The device whose GPU the context is on.
  /// @return The context; `Status::Code::Unsupported` as
  ///         @ref cuda_ordinal_of; or `Status::Code::Backend` if CUDA fails.
  static core::Result<std::unique_ptr<CudaContext>> open(
      const core::Device& device);

  ~CudaContext();
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  /// @return The context, retained until this is destroyed.
  CUcontext handle() const noexcept { return context_; }

 private:
  CudaContext() = default;

  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

/// @brief A CUDA stream of its own on a context: its work queues apart from
///        every other stream's, and never waits on the legacy default
///        stream.
class VR_SENSOR_VIDEO_API CudaStream {
 public:
  /// @brief Create a stream on @p context.
  /// @param context  The context; it must outlive the stream.
  /// @return The stream; or `Status::Code::Backend` if CUDA fails.
  static core::Result<std::unique_ptr<CudaStream>> open(
      const CudaContext& context);

  ~CudaStream();
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  /// @return The stream, destroyed with this.
  CUstream handle() const noexcept { return stream_; }

 private:
  explicit CudaStream(CUcontext context) noexcept : context_(context) {}

  CUcontext context_;
  CUstream stream_ = nullptr;
};

/// @brief A buffer made with `core::create_exported_buffer`, mapped into
///        CUDA.
///
/// CUDA writes, Vulkan reads: once the CUDA work that writes the buffer has
/// finished (its stream or an event synchronized), a Vulkan reader acquires
/// the buffer from `VK_QUEUE_FAMILY_EXTERNAL` before it reads.
///
/// @code
/// VKC_ASSIGN(core::ExportedBuffer exported,
///            core::create_exported_buffer(device, allocator, bytes));
/// VKC_ASSIGN(auto mapping, CudaBufferMapping::map(*context, exported));
/// // ... CUDA writes mapping->pointer() on a stream, which is synchronized
/// core::CommandBatch batch(device, allocator);
/// VKC_TRY(batch.acquire(exported.buffer, VK_QUEUE_FAMILY_EXTERNAL));
/// // ... the batch reads exported.buffer
/// VKC_TRY(batch.submit());
/// @endcode
class VR_SENSOR_VIDEO_API CudaBufferMapping {
 public:
  /// @brief Import @p exported's memory into @p context and map its buffer.
  ///
  /// Takes @p exported's file descriptor, which CUDA then owns; a failed
  /// import leaves it with @p exported. The buffer stays the caller's.
  /// @param context   The context CUDA reaches the buffer in; it must
  ///                  outlive the mapping.
  /// @param exported  The buffer and its descriptor; its buffer must outlive
  ///                  the mapping.
  /// @return The mapping; `Status::Code::InvalidArgument` when the descriptor
  ///         was taken, or there is no buffer; or `Status::Code::Backend` if
  ///         CUDA fails.
  static core::Result<std::unique_ptr<CudaBufferMapping>> map(
      const CudaContext& context, core::ExportedBuffer& exported);

  ~CudaBufferMapping();
  CudaBufferMapping(const CudaBufferMapping&) = delete;
  CudaBufferMapping& operator=(const CudaBufferMapping&) = delete;

  /// @return Where CUDA sees the buffer: its first byte, for the whole of
  ///         its size.
  CUdeviceptr pointer() const noexcept { return pointer_; }

 private:
  explicit CudaBufferMapping(CUcontext context) noexcept : context_(context) {}

  CUcontext context_;
  CUexternalMemory memory_ = nullptr;
  CUdeviceptr pointer_ = 0;
};

}  // namespace volumetric_kit::recon::sensor
