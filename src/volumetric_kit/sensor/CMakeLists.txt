# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor: capture contract -----------------------------------------
# The ICameraCapture interface, the CapturedFrame view the fusion tiers already
# consume, and the camera-convention conversions. Deliberately carries no driver
# implementations: one ships here only if this repo can build *and* test it (the
# 2026-08-02 decision), so a platform-bound source such as ARKit lives with the
# application that can.
#
# Depends on recon_tsdf for ColorCameraParams and (transitively) recon_volume
# for DepthCameraParams -- both to its left, so the strict tier rule holds. No
# Vulkan of its own: this tier is host-side types and arithmetic, which is why
# its tests run everywhere rather than only where a device exists.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor)
add_library(volumetric_kit::recon_sensor ALIAS recon_sensor)

target_sources(recon_sensor PRIVATE camera_conventions.cpp)

# Per-tier export header (VR_SENSOR_API / VR_SENSOR_STATIC_DEFINE), isolated so
# the macros never collide with another tier's on one include path.
generate_export_header(
  recon_sensor
  BASE_NAME
  VR_SENSOR
  EXPORT_MACRO_NAME
  VR_SENSOR_API
  EXPORT_FILE_NAME
  "${_vr_sensor_generated_inc}/volumetric_kit/recon/sensor/export.hpp")

set_target_properties(
  recon_sensor
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_type recon_sensor TYPE)
if(_vr_sensor_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor PUBLIC VR_SENSOR_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor PUBLIC cxx_std_17)

# PUBLIC: the sensor public headers expose tsdf types (ColorCameraParams /
# ColorFrame) and volume types (DepthCameraParams), so consumers compile against
# recon_tsdf + recon_volume (and, transitively, recon_core) too.
target_link_libraries(recon_sensor PUBLIC recon_tsdf recon_volume)

target_include_directories(
  recon_sensor
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_sensor)

# Install the target into the shared export set + this tier's headers (the
# sensor/ subtree) + its generated export header. Mirrors recon_texture so a
# partial build never installs a header for a tier it did not build.
if(VR_INSTALL)
  install(
    TARGETS recon_sensor
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/sensor"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_sensor_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
