# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_apple: Apple's pictures on the device ----------------------------
# PixelBufferImporter: a CVPixelBuffer's NV12 planes, on their IOSurface, as
# Vulkan images through Metal textures (Objective-C++, with ARC). Built on every
# Apple platform and with no FFmpeg, since it uses none: the video decoders'
# VideoToolbox pictures come through it, and so can a camera's on iOS, such as
# ARKit's. Links sensor_frame alone; the YuvImage it fills is a header of
# sensor_interface's (sensor/yuv_image.hpp) that names no camera.
include(GenerateExportHeader)

set(_vs_sensor_apple_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_apple)
add_library(volumetric_kit::sensor_apple ALIAS sensor_apple)

target_sources(sensor_apple PRIVATE pixel_buffer_importer.mm)
set_source_files_properties(pixel_buffer_importer.mm PROPERTIES COMPILE_OPTIONS
                                                                -fobjc-arc)

generate_export_header(
  sensor_apple
  BASE_NAME
  VS_APPLE
  EXPORT_MACRO_NAME
  VS_APPLE_API
  EXPORT_FILE_NAME
  "${_vs_sensor_apple_generated_inc}/volumetric_kit/sensor/apple/export.hpp")

set_target_properties(
  sensor_apple
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             OBJCXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_apple_type sensor_apple TYPE)
if(_vs_sensor_apple_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_apple PUBLIC VS_APPLE_STATIC_DEFINE)
endif()

target_compile_features(sensor_apple PUBLIC cxx_std_17)

# PUBLIC: the header takes a core Device and a CVPixelBufferRef.
target_link_libraries(
  sensor_apple
  PUBLIC sensor_frame "-framework CoreVideo"
  PRIVATE "-framework IOSurface" "-framework Metal" "-framework Foundation")

target_include_directories(
  sensor_apple PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                      "$<BUILD_INTERFACE:${_vs_sensor_apple_generated_inc}>")

vs_target_warnings(sensor_apple)
