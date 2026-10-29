# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_realsense: the RealSense (D400) driver ---------------------------
# RealSenseSensor, an IRgbdSensor over librealsense, built only with
# VS_WITH_REALSENSE (see cmake/vs_realsense.cmake). Its own target so
# sensor_interface stays free of the SDK (the 2026-09-26 decision). The SDK is
# PRIVATE and no SDK type reaches the public header, but linking still needs its
# library.
include(GenerateExportHeader)

set(_vs_sensor_realsense_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_realsense)
add_library(volumetric_kit::sensor_realsense ALIAS sensor_realsense)

target_sources(sensor_realsense PRIVATE realsense_sensor.cpp conversion.cpp)

generate_export_header(
  sensor_realsense
  BASE_NAME
  VS_REALSENSE
  EXPORT_MACRO_NAME
  VS_REALSENSE_API
  EXPORT_FILE_NAME
  "${_vs_sensor_realsense_generated_inc}/volumetric_kit/sensor/realsense/export.hpp"
)

set_target_properties(
  sensor_realsense
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_realsense_type sensor_realsense TYPE)
if(_vs_sensor_realsense_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_realsense PUBLIC VS_REALSENSE_STATIC_DEFINE)
endif()

target_compile_features(sensor_realsense PUBLIC cxx_std_17)

# PUBLIC sensor_interface: the sensor *is* an IRgbdSensor, and its header hands
# out the sensor tier's frame. The SDK privately, compiled with the exceptions
# it throws (vs_link_realsense_sdk).
target_link_libraries(sensor_realsense PUBLIC sensor_interface)
vs_link_realsense_sdk(sensor_realsense)

target_include_directories(
  sensor_realsense
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vs_sensor_realsense_generated_inc}>")

vs_target_warnings(sensor_realsense)
