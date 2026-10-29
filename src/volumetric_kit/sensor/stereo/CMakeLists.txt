# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_stereo: calibrated stereo pairs, rectified on the GPU ------------
# What a stereo camera driver implements (IStereoSensor) and hands out
# (StereoFrame), and the StereoRectifier: the pair undistorted and rectified by
# a compute pass a view, for a stereo matcher. No SDK; Vulkan on every platform.
# The rectifier samples and converts through the colour pass's own shader code
# and checks (prep's color_pass.hpp), so it links sensor_prep, privately: its
# headers need only the interface and the core's Vulkan tier.
include(GenerateExportHeader)

set(_vs_stereo_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_stereo)
add_library(volumetric_kit::sensor_stereo ALIAS sensor_stereo)

target_sources(sensor_stereo PRIVATE stereo_rectifier.cpp stereo_sensor.cpp)

generate_export_header(
  sensor_stereo
  BASE_NAME
  VS_STEREO
  EXPORT_MACRO_NAME
  VS_STEREO_API
  EXPORT_FILE_NAME
  "${_vs_stereo_generated_inc}/volumetric_kit/sensor/stereo/export.hpp")

set_target_properties(
  sensor_stereo
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_stereo_type sensor_stereo TYPE)
if(_vs_stereo_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_stereo PUBLIC VS_STEREO_STATIC_DEFINE)
endif()

target_compile_features(sensor_stereo PUBLIC cxx_std_17)

# PUBLIC: its headers take the interface's SensorStats and SensorInfo, the frame
# tier's pictures and the core's buffers. PRIVATE: the colour pass's code it
# shares.
target_link_libraries(
  sensor_stereo
  PUBLIC sensor_interface sensor_frame
  PRIVATE sensor_prep)

target_include_directories(
  sensor_stereo PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                       "$<BUILD_INTERFACE:${_vs_stereo_generated_inc}>")

include(vs_embed)
vs_embed_shaders(sensor_stereo SHADERS
                 "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rectify.comp")

vs_target_warnings(sensor_stereo)
