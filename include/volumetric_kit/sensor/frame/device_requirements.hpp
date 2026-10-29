// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file frame/device_requirements.hpp
/// @brief What the sensor package's kernels and decoders require of a device.
///
/// The device is volumetric_kit_core's (`core/vulkan/device.hpp`). A caller
/// that runs the sensor package alone passes @ref device_requirements to
/// `Instance::select_physical_device` and `Device::create`; one that also
/// runs another package's kernels on the same device merges the two lists
/// with the core's `merge` first.
///
/// @code
/// VKC_ASSIGN(const core::DeviceRequirements reqs,
///            core::merge(sensor::device_requirements(),
///                        recon::device_requirements()));
/// VKC_ASSIGN(core::PhysicalDeviceInfo gpu,
///            instance.select_physical_device(reqs));
/// VKC_ASSIGN(core::Device device, core::Device::create(instance, gpu, reqs));
/// @endcode

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"

namespace volumetric_kit::sensor {

/// @brief What the sensor package's kernels and decoders require of a device.
///
/// A compute queue with timeline semaphores (the core's defaults) and
/// `scalarBlockLayout`, the buffer ABI the frame prep's kernels declare
/// (`layout(scalar)`). Two extensions are optional and enabled where offered:
/// `VK_KHR_external_memory_fd`, through which a hardware decoder's picture
/// lands in the package's buffers without a host trip
/// (`volumetric_kit/core/vulkan/external_memory.hpp`), and
/// `VK_EXT_metal_objects`, through which a VideoToolbox picture does on
/// Apple. They are named here, not left to the device's creator, so a merged
/// bootstrap that never hears of them does not silently send every decoded
/// picture through the host.
///
/// @return The requirements; a caller may add to them before creating.
inline core::DeviceRequirements device_requirements() {
  core::DeviceRequirements reqs;
  reqs.scalar_block_layout = true;
  reqs.optional_extensions = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                              "VK_EXT_metal_objects"};
  return reqs;
}

/// @brief Refuse a device that did not enable what the package's kernels
///        need.
///
/// Called first by every `create` here that builds kernels: on a device
/// without `scalarBlockLayout` the kernels would build and run without an
/// error, reading their buffers at the wrong offsets. Optional extensions are
/// not required.
///
/// @param device  The device the kernels are for.
/// @param who     The caller, prefixed to a refusal.
/// @return OK; or `Device::check_enabled`'s `Status::Code::Unsupported`
///         naming what @ref device_requirements needs and the device did not
///         enable.
inline core::Status check_device_requirements(const core::Device& device,
                                              const char* who) {
  return device.check_enabled(device_requirements()).with_context(who);
}

}  // namespace volumetric_kit::sensor
