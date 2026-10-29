# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_orbbec: the Orbbec (Femto Mega) driver ---------------------
# OrbbecSensor, an IRgbdSensor over the Orbbec SDK (a rig is several of them in
# a SensorArray), built only with VS_WITH_ORBBEC (see cmake/vs_orbbec.cmake).
# Its own target so sensor_interface -- the interface an out-of-tree driver
# compiles against -- stays free of the SDK (the 2026-09-26 decision). The SDK
# is PRIVATE and no SDK type reaches the public header, but linking still needs
# its library.
include(GenerateExportHeader)

set(_vs_sensor_orbbec_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_orbbec)
add_library(volumetric_kit::sensor_orbbec ALIAS sensor_orbbec)

target_sources(
  sensor_orbbec
  PRIVATE orbbec_sensor.cpp
          orbbec_sync_config.cpp
          camera_stream.cpp
          frame_conversion.cpp
          hevc_color.cpp
          jpeg_color.cpp
          picture_frames.cpp)

generate_export_header(
  sensor_orbbec
  BASE_NAME
  VS_ORBBEC
  EXPORT_MACRO_NAME
  VS_ORBBEC_API
  EXPORT_FILE_NAME
  "${_vs_sensor_orbbec_generated_inc}/volumetric_kit/sensor/orbbec/export.hpp")

set_target_properties(
  sensor_orbbec
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_orbbec_type sensor_orbbec TYPE)
if(_vs_sensor_orbbec_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_orbbec PUBLIC VS_ORBBEC_STATIC_DEFINE)
endif()

target_compile_features(sensor_orbbec PUBLIC cxx_std_17)

# PUBLIC sensor_interface: the sensor *is* an IRgbdSensor, and its header hands
# out the sensor tier's frame. PRIVATE sensor_video, which decodes the cameras'
# colour: no decoder type reaches the public header. The SDK privately (see
# above), compiled with the exceptions it throws (vs_link_orbbec_sdk).
target_link_libraries(
  sensor_orbbec
  PUBLIC sensor_interface
  PRIVATE sensor_video)
vs_link_orbbec_sdk(sensor_orbbec)

target_include_directories(
  sensor_orbbec PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                       "$<BUILD_INTERFACE:${_vs_sensor_orbbec_generated_inc}>")

# nlohmann/json parses the sync configuration, privately (see sensor_camera).
target_include_directories(sensor_orbbec SYSTEM
                           PRIVATE "${VS_JSON_INCLUDE_DIR}")

vs_target_warnings(sensor_orbbec)
