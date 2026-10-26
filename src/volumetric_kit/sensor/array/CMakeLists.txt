# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_array: several sensors read as one -------------------------
# The vendor-neutral array of IRgbdSensors: start order from their rig roles,
# frames grouped into one set per trigger, each posed by the array's
# calibration, each set prepared on the GPU in one batch (the 2026-10-06
# decisions), and a run's sets waited for and cut to its length (RigSets). A
# target of its own above the drivers, which never link it.
include(GenerateExportHeader)

set(_vr_sensor_array_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_array)
add_library(volumetric_kit::recon_sensor_array ALIAS recon_sensor_array)

# The trigger grouping is internal: no export, no public header.
target_sources(recon_sensor_array PRIVATE rig_sets.cpp sensor_array.cpp
                                          trigger_grouper.cpp)

generate_export_header(
  recon_sensor_array
  BASE_NAME
  VR_SENSOR_ARRAY
  EXPORT_MACRO_NAME
  VR_SENSOR_ARRAY_API
  EXPORT_FILE_NAME
  "${_vr_sensor_array_generated_inc}/volumetric_kit/recon/sensor/array/export.hpp"
)

set_target_properties(
  recon_sensor_array
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_array_type recon_sensor_array TYPE)
if(_vr_sensor_array_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_array
                             PUBLIC VR_SENSOR_ARRAY_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_array PUBLIC cxx_std_17)

# PUBLIC: its header takes IRgbdSensors and the camera tier's calibration, and
# hands out the GPU pass's DeviceFrames.
target_link_libraries(recon_sensor_array PUBLIC recon_sensor recon_camera
                                                recon_sensor_utils)

target_include_directories(
  recon_sensor_array
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_array_generated_inc}>")

vr_target_warnings(recon_sensor_array)
