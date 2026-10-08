# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_apple: Apple's pictures on the device ---------------------
# PixelBufferImporter: a CVPixelBuffer's NV12 planes, on their IOSurface, as
# Vulkan images through Metal textures (Objective-C++, with ARC). Built on every
# Apple platform and with no FFmpeg, since it uses none: the video decoders'
# VideoToolbox pictures come through it, and so can a camera's on iOS, such as
# ARKit's. Links recon_core alone, as the decoders do; the YuvImage it fills is
# a header of recon_sensor's (sensor/yuv_image.hpp) that names no camera.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_apple_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_apple)
add_library(volumetric_kit::recon_sensor_apple ALIAS recon_sensor_apple)

target_sources(recon_sensor_apple PRIVATE pixel_buffer_importer.mm)
set_source_files_properties(pixel_buffer_importer.mm PROPERTIES COMPILE_OPTIONS
                                                                -fobjc-arc)

generate_export_header(
  recon_sensor_apple
  BASE_NAME
  VR_SENSOR_APPLE
  EXPORT_MACRO_NAME
  VR_SENSOR_APPLE_API
  EXPORT_FILE_NAME
  "${_vr_sensor_apple_generated_inc}/volumetric_kit/recon/sensor/apple/export.hpp"
)

set_target_properties(
  recon_sensor_apple
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             OBJCXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_apple_type recon_sensor_apple TYPE)
if(_vr_sensor_apple_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_apple
                             PUBLIC VR_SENSOR_APPLE_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_apple PUBLIC cxx_std_17)

# PUBLIC: the header takes a core Device and a CVPixelBufferRef.
target_link_libraries(
  recon_sensor_apple
  PUBLIC recon_core "-framework CoreVideo"
  PRIVATE "-framework IOSurface" "-framework Metal" "-framework Foundation")

target_include_directories(
  recon_sensor_apple
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_apple_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_sensor_apple)

# The target, its headers (recon_sensor's install leaves them out) and the
# generated export header.
if(VR_INSTALL)
  install(
    TARGETS recon_sensor_apple
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/sensor/apple"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon/sensor"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_sensor_apple_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
