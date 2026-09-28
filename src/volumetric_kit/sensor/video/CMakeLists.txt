# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_video: the video decoder ----------------------------------
# An HEVC decoder over FFmpeg and its hardware back ends, built only with
# VR_WITH_FFMPEG (see cmake/vr_ffmpeg.cmake). Camera-neutral, so it links
# recon_core alone and a capture driver links it. FFmpeg is PRIVATE and no
# FFmpeg type reaches the public headers; ffmpeg.hpp is the one include.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_video_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_video)
add_library(volumetric_kit::recon_sensor_video ALIAS recon_sensor_video)

target_sources(
  recon_sensor_video PRIVATE ffmpeg.cpp hevc_decoder.cpp hw_backend.cpp
                             picture_converter.cpp video_backend.cpp)

generate_export_header(
  recon_sensor_video
  BASE_NAME
  VR_SENSOR_VIDEO
  EXPORT_MACRO_NAME
  VR_SENSOR_VIDEO_API
  EXPORT_FILE_NAME
  "${_vr_sensor_video_generated_inc}/volumetric_kit/recon/sensor/video/export.hpp"
)

set_target_properties(
  recon_sensor_video
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_video_type recon_sensor_video TYPE)
if(_vr_sensor_video_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_video
                             PUBLIC VR_SENSOR_VIDEO_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_video PUBLIC cxx_std_17)

target_link_libraries(
  recon_sensor_video
  PUBLIC recon_core
  PRIVATE PkgConfig::VR_FFMPEG)
# VideoToolbox answers whether its hardware decodes HEVC (hw_backend.cpp).
if(APPLE)
  target_link_libraries(recon_sensor_video PRIVATE "-framework VideoToolbox")
endif()

target_include_directories(
  recon_sensor_video
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_video_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_sensor_video)

# The target, its headers (recon_sensor's install leaves them out) and the
# generated export header.
if(VR_INSTALL)
  install(
    TARGETS recon_sensor_video
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/sensor/video"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon/sensor"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_sensor_video_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
