# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_utils: GPU pre-processing ---------------------------------
# A captured frame undistorted and converted by compute kernels, left on the
# device for fusion. A target of its own because it carries Vulkan and shaders,
# which recon_sensor -- what an out-of-tree driver compiles against -- does not
# (the 2026-09-26 decision). Driver-neutral: it takes a sensor::RgbdFrame.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_utils_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_utils)
add_library(volumetric_kit::recon_sensor_utils ALIAS recon_sensor_utils)

target_sources(recon_sensor_utils PRIVATE gpu_frame_prep.cpp)

generate_export_header(
  recon_sensor_utils
  BASE_NAME
  VR_SENSOR_UTILS
  EXPORT_MACRO_NAME
  VR_SENSOR_UTILS_API
  EXPORT_FILE_NAME
  "${_vr_sensor_utils_generated_inc}/volumetric_kit/recon/sensor/utils/export.hpp"
)

set_target_properties(
  recon_sensor_utils
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_utils_type recon_sensor_utils TYPE)
if(_vr_sensor_utils_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_utils
                             PUBLIC VR_SENSOR_UTILS_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_utils PUBLIC cxx_std_17)

# PUBLIC: its header takes an RgbdFrame (recon_sensor) and hands out core
# Buffers.
target_link_libraries(recon_sensor_utils PUBLIC recon_sensor recon_core)

target_include_directories(
  recon_sensor_utils
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_utils_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

include(vr_embed)
vr_embed_shaders(
  recon_sensor_utils SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/undistort_depth.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/undistort_color.comp")

vr_target_warnings(recon_sensor_utils)

# The target, its headers (recon_sensor's install leaves them out) and the
# generated export header.
if(VR_INSTALL)
  install(
    TARGETS recon_sensor_utils
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/sensor/utils"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon/sensor"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_sensor_utils_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
