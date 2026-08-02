# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor: capture contract -----------------------------------------
# The ICameraCapture interface, the CapturedFrame view the fusion tiers already
# consume, and the camera-convention conversions. Deliberately carries no driver
# implementations: one ships here only if this repo can build *and* test it (the
# 2026-08-02 decision), so a platform-bound source such as ARKit lives with the
# application that can.
#
# Depends on recon_core alone -- the whole tier is host types and arithmetic
# over the core math vocabulary, so it needs neither the volume nor the tsdf
# tier: DepthCameraParams and ColorCameraParams live in core/camera_params.hpp
# (they are posed pinhole cameras, not voxels or fusion). That is what makes
# this tier's contract implementable from outside: an ARKit driver in
# volumetric_kit_ios is Objective-C++, and it now compiles and links against
# core alone rather than preprocessing the Vulkan surface to fill in an
# intrinsics struct. It is also why the tests need no device.
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

# PUBLIC: the sensor public headers expose core types only (DepthCameraParams /
# ColorCameraParams / Result), so a consumer compiles against recon_core and
# nothing else.
target_link_libraries(recon_sensor PUBLIC recon_core)

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
