// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "cuda_timeline.hpp"

#include <string>
#include <utility>

#include "cuda_driver.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"

namespace volumetric_kit::sensor::video {

core::Result<std::unique_ptr<CudaTimeline>> CudaTimeline::create(
    const core::Device& device, const CudaContext& context, CUstream stream,
    const char* who) {
  if (!device.exports_semaphores()) {
    return core::Status::unsupported(std::string(who) +
                                     ": the device does not export semaphores "
                                     "(VK_KHR_external_semaphore_fd)");
  }
  core::Result<core::ExportedTimeline> made =
      core::create_exported_timeline(device);
  if (!made) return made.status().with_context(who);
  core::ExportedTimeline exported = std::move(made).value();
  std::unique_ptr<CudaTimeline> t(
      new CudaTimeline(context.handle(), stream, who));
  t->semaphore_ =
      std::make_shared<core::TimelineSemaphore>(std::move(exported.semaphore));

  const CudaContextScope scope(t->context_);
  if (!scope.ok()) {
    return cuda_error(who, scope.result(), "making the CUDA context current");
  }
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC handle{};
  handle.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD;
  handle.handle.fd = exported.fd.get();
  const CUresult r =
      cuda_driver()->cuImportExternalSemaphore(&t->imported_, &handle);
  if (r != CUDA_SUCCESS) {
    t->imported_ = nullptr;  // the descriptor stays with `exported`
    return cuda_error(who, r, "importing the pictures' timeline");
  }
  exported.fd.release();  // CUDA owns it now
  return t;
}

CudaTimeline::~CudaTimeline() {
  if (imported_ == nullptr) return;
  settle();
  const CudaContextScope scope(context_);
  if (scope.ok()) cuda_driver()->cuDestroyExternalSemaphore(imported_);
}

core::Status CudaTimeline::mark(YuvImage& out) {
  const CudaContextScope scope(context_);
  if (!scope.ok()) {
    return cuda_error(who_, scope.result(), "making the CUDA context current");
  }
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal{};
  signal.params.fence.value = last_ + 1;
  const CUresult r = cuda_driver()->cuSignalExternalSemaphoresAsync(
      &imported_, &signal, 1, stream_);
  if (r != CUDA_SUCCESS) {
    return cuda_error(who_, r, "queuing a picture's ready value");
  }
  ++last_;
  out.ready = semaphore_;
  out.ready_value = last_;
  return {};
}

bool CudaTimeline::reached(std::uint64_t value) const {
  const core::Result<std::uint64_t> counter = semaphore_->value();
  return counter.ok() && *counter >= value;
}

core::Status CudaTimeline::wait(std::uint64_t value) const {
  return semaphore_->wait(value).with_context(who_);
}

void CudaTimeline::settle() noexcept {
  const CudaContextScope scope(context_);
  // Without the stream waited for, CUDA might still set a value below the
  // host's: nothing is set.
  if (!scope.ok()) return;
  // Returns once the stream's work has run or failed, either way for good.
  static_cast<void>(cuda_driver()->cuStreamSynchronize(stream_));
  if (!reached(last_)) static_cast<void>(semaphore_->signal(last_));
}

}  // namespace volumetric_kit::sensor::video
