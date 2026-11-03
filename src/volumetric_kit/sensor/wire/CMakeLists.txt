# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GenerateExportHeader)

set(_vs_sensor_wire_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

# --- sensor_wire: an RGB-D sensor's frames as bytes --------------------------
# What a sensor is (SensorInfo), each frame's depth half and its colour half
# (RgbdFrame) as the bytes a recording or a live link carries, and those bytes
# read back in place (wire/rgbd_wire.hpp). Host code with no codec and no
# device: it copies and reads no depth sample and no picture, so it builds
# wherever the interface does, iOS included. The colour codes a record states
# are sensor_video_syntax's ColorCodes.
add_library(sensor_wire)
add_library(volumetric_kit::sensor_wire ALIAS sensor_wire)

target_sources(sensor_wire PRIVATE rgbd_wire.cpp)

generate_export_header(
  sensor_wire
  BASE_NAME
  VS_WIRE
  EXPORT_MACRO_NAME
  VS_WIRE_API
  EXPORT_FILE_NAME
  "${_vs_sensor_wire_generated_inc}/volumetric_kit/sensor/wire/export.hpp")

set_target_properties(
  sensor_wire
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_wire_type sensor_wire TYPE)
if(_vs_sensor_wire_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_wire PUBLIC VS_WIRE_STATIC_DEFINE)
endif()

target_compile_features(sensor_wire PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary even when a consuming
# application disables exceptions. Private: no public header reaches it.
target_compile_options(sensor_wire PRIVATE -fexceptions)

# PUBLIC: the header takes and returns the interface's SensorInfo and RgbdFrame,
# and the H.273 codes of video/color_codes.hpp.
target_link_libraries(sensor_wire PUBLIC sensor_interface sensor_video_syntax)

target_include_directories(
  sensor_wire PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                     "$<BUILD_INTERFACE:${_vs_sensor_wire_generated_inc}>")

vs_target_warnings(sensor_wire)
