# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_prep: GPU pre-processing and readback ---------------------
# A captured frame undistorted and converted by compute kernels, left on the
# device for fusion; and a set's colour pictures copied to the host. A target of
# its own because it carries Vulkan and shaders, which sensor_interface -- what
# an out-of-tree driver compiles against -- does not (the 2026-09-26 decision).
# Driver-neutral: it takes a sensor::RgbdFrame.
include(GenerateExportHeader)

set(_vs_sensor_utils_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_prep)
add_library(volumetric_kit::sensor_prep ALIAS sensor_prep)

target_sources(sensor_prep PRIVATE gpu_frame_prep.cpp picture_readback.cpp
                                   yuv_planes.cpp)

generate_export_header(
  sensor_prep
  BASE_NAME
  VS_PREP
  EXPORT_MACRO_NAME
  VS_PREP_API
  EXPORT_FILE_NAME
  "${_vs_sensor_utils_generated_inc}/volumetric_kit/sensor/prep/export.hpp")

set_target_properties(
  sensor_prep
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_utils_type sensor_prep TYPE)
if(_vs_sensor_utils_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_prep PUBLIC VS_PREP_STATIC_DEFINE)
endif()

target_compile_features(sensor_prep PUBLIC cxx_std_17)

# PUBLIC: its header takes an RgbdFrame (sensor_interface) and hands out core
# Buffers.
target_link_libraries(sensor_prep PUBLIC sensor_interface sensor_frame)

target_include_directories(
  sensor_prep PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                     "$<BUILD_INTERFACE:${_vs_sensor_utils_generated_inc}>")

include(vs_embed)
vs_embed_shaders(
  sensor_prep
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/undistort_depth.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/undistort_color.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/undistort_color_images.comp")

vs_target_warnings(sensor_prep)
