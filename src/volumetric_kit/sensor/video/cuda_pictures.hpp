// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// NVDEC's pictures, handed over on the device (VR_WITH_CUDA): each is copied,
// device to device, into a Vulkan buffer CUDA has imported, so the kernels
// read it where it is and it never crosses the bus.

#include <cuda.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "volumetric_kit/recon/core/result.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon {
class Buffer;
class Device;
}  // namespace volumetric_kit::recon

namespace volumetric_kit::recon::sensor::video {

// The CUDA device that is @p device's GPU: Vulkan's deviceUUID matched
// against each CUDA device's. NotFound when no CUDA device is.
Result<int> cuda_ordinal_of(const Device& device);

// A ring of exported Vulkan buffers on one device, each imported into CUDA
// once, in the decoder's CUDA context. A picture takes one that no earlier
// picture still holds, or a new one; the ring only grows.
class CudaPictures {
 public:
  static Result<std::unique_ptr<CudaPictures>> create(const Device& device,
                                                      CUcontext context,
                                                      CUstream stream);
  ~CudaPictures();
  CudaPictures(const CudaPictures&) = delete;
  CudaPictures& operator=(const CudaPictures&) = delete;

  // Copy an NV12 picture on the device, Y at `luma` and the chroma at
  // `chroma` with their pitches, into a buffer, rows packed; the copy has
  // finished when this returns. Fills @p out's size, layout and planes.
  Status copy(CUdeviceptr luma, std::size_t luma_pitch, CUdeviceptr chroma,
              std::size_t chroma_pitch, std::uint32_t width,
              std::uint32_t height, DecodedPicture& out);

 private:
  struct Slot {
    std::shared_ptr<Buffer> buffer;
    CUexternalMemory memory = nullptr;
    CUdeviceptr pointer = 0;
    std::uint64_t bytes = 0;
  };
  CudaPictures(const Device& device, CUcontext context, CUstream stream)
      : device_(&device), context_(context), stream_(stream) {}
  Result<Slot*> slot(std::uint64_t bytes);

  const Device* device_;
  CUcontext context_;
  CUstream stream_;
  std::vector<Slot> slots_;
};

}  // namespace volumetric_kit::recon::sensor::video
