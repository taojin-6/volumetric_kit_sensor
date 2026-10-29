# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_frame: the frame contract -----------------------------------------
# DeviceFrame and DeviceFrameSet, the posed, undistorted RGB-D frame on the
# device that fusion, texturing and a sender read, in the vocabulary they are
# written in: the float pinhole cameras (camera_params.hpp) with their GLSL
# mirror (shaders/pinhole.glsl, exported), the colour encoding (color_space.hpp)
# and the package's GLM aliases; and what the package requires of a device
# (device_requirements.hpp). Header-only, over the core's vulkan tier: a
# consumer of frames links this tier alone, whoever produced them.

# The core's device serializes submits with std::mutex and the sensor tiers run
# capture threads, so everything above needs the platform threading runtime.
# Linked explicitly rather than through the compiler's implicit pthread, so a
# static archive resolves on toolchains that require it (GCC/libstdc++ with
# --as-needed, Android, static links).
find_package(Threads REQUIRED)
find_package(glm CONFIG REQUIRED)
# PUBLIC through the core's vulkan tier: the headers reach Vulkan through the
# core's core/vulkan/vulkan.hpp. MoltenVK is the loader and driver on Apple.
find_package(Vulkan REQUIRED)

add_library(sensor_frame INTERFACE)
add_library(volumetric_kit::sensor_frame ALIAS sensor_frame)

target_compile_features(sensor_frame INTERFACE cxx_std_17)
target_link_libraries(
  sensor_frame INTERFACE volumetric_kit::core_base volumetric_kit::core_vulkan
                         Threads::Threads glm::glm Vulkan::Vulkan)
target_include_directories(
  sensor_frame INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

# The GLSL the package exports, for a consumer's kernels to #include as the
# sensor's own do ("volumetric_kit/sensor/frame/shaders/pinhole.glsl"), through
# the core's INCLUDE_TARGETS volumetric_kit::sensor_frame. The other tiers'
# shaders read the package's src/ (vs_embed_shaders) and stay private.
vkc_export_glsl(sensor_frame PREFIX volumetric_kit/sensor/frame/shaders FILES
                shaders/pinhole.glsl)
