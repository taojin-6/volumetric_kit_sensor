# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_array: several sensors read as one -------------------------
# The vendor-neutral array of IRgbdSensors: start order from their rig roles,
# frames grouped into one set per trigger, each posed by the array's
# calibration, each set prepared on the GPU in one batch (the 2026-10-06
# decisions), and a run's sets waited for and cut to its length (RigSets). A
# target of its own above the drivers, which never link it.
include(GenerateExportHeader)

set(_vs_sensor_array_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_array)
add_library(volumetric_kit::sensor_array ALIAS sensor_array)

# The trigger grouping is internal: no export, no public header.
target_sources(sensor_array PRIVATE rig_sets.cpp sensor_array.cpp
                                    trigger_grouper.cpp)

generate_export_header(
  sensor_array
  BASE_NAME
  VS_ARRAY
  EXPORT_MACRO_NAME
  VS_ARRAY_API
  EXPORT_FILE_NAME
  "${_vs_sensor_array_generated_inc}/volumetric_kit/sensor/array/export.hpp")

set_target_properties(
  sensor_array
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_array_type sensor_array TYPE)
if(_vs_sensor_array_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_array PUBLIC VS_ARRAY_STATIC_DEFINE)
endif()

target_compile_features(sensor_array PUBLIC cxx_std_17)

# PUBLIC: its header takes IRgbdSensors and the camera tier's calibration, and
# hands out the GPU pass's DeviceFrames.
target_link_libraries(sensor_array PUBLIC sensor_interface sensor_camera
                                          sensor_prep)

target_include_directories(
  sensor_array PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                      "$<BUILD_INTERFACE:${_vs_sensor_array_generated_inc}>")

vs_target_warnings(sensor_array)
