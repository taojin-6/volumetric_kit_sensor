# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Orchestrator for the package's tiers. Each subdirectory owns its own target
# and build settings, and a tier links only the tiers it builds on:
# sensor_camera (core_base and GLM, no Vulkan), sensor_image_io and
# sensor_video_syntax (core_base) on the core alone; sensor_frame (core_vulkan),
# the frame contract recon's libraries link; sensor_interface on camera and
# frame; sensor_prep and then sensor_array on the interface; sensor_apple and
# sensor_video on frame; sensor_orbbec on the interface and the decoders.
add_subdirectory(volumetric_kit/sensor/camera)
set(_vs_tiers sensor_camera)
if(NOT VS_CAMERA_ONLY)
  add_subdirectory(volumetric_kit/sensor/frame)
  add_subdirectory(volumetric_kit/sensor/io)
  # The sensor interface, and the tiers above it (prep, array, apple, video,
  # orbbec), which its directory adds.
  add_subdirectory(volumetric_kit/sensor)
  list(
    APPEND
    _vs_tiers
    sensor_frame
    sensor_image_io
    sensor_interface
    sensor_prep
    sensor_array
    sensor_video_syntax)
  foreach(_vs_optional sensor_apple sensor_video sensor_orbbec)
    if(TARGET ${_vs_optional})
      list(APPEND _vs_tiers ${_vs_optional})
    endif()
  endforeach()
endif()

# --- sensor: umbrella alias --------------------------------------------------
# Consumers link `volumetric_kit::sensor` and get whatever tiers are built. It
# carries the package's version, which a consumer's vkc_require_sibling reads.
add_library(sensor INTERFACE)
add_library(volumetric_kit::sensor ALIAS sensor)
set_target_properties(sensor PROPERTIES VKC_PACKAGE_VERSION
                                        "${PROJECT_VERSION}")
target_link_libraries(sensor INTERFACE ${_vs_tiers})
