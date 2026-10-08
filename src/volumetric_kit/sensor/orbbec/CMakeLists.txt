# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_orbbec: the Orbbec (Femto Mega) driver ---------------------
# OrbbecSensor, an IRgbdSensor over the Orbbec SDK (a rig is several of them in
# a SensorArray), built only with VR_WITH_ORBBEC (see cmake/vr_orbbec.cmake).
# Its own target so recon_sensor -- the interface an out-of-tree driver compiles
# against -- stays free of the SDK (the 2026-09-26 decision). The SDK is PRIVATE
# and no SDK type reaches the public header, but linking still needs its
# library, so the package config re-finds the SDK.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_orbbec_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_orbbec)
add_library(volumetric_kit::recon_sensor_orbbec ALIAS recon_sensor_orbbec)

target_sources(
  recon_sensor_orbbec
  PRIVATE orbbec_sensor.cpp
          orbbec_sync_config.cpp
          camera_stream.cpp
          frame_conversion.cpp
          hevc_color.cpp
          jpeg_color.cpp
          picture_frames.cpp)

generate_export_header(
  recon_sensor_orbbec
  BASE_NAME
  VR_SENSOR_ORBBEC
  EXPORT_MACRO_NAME
  VR_SENSOR_ORBBEC_API
  EXPORT_FILE_NAME
  "${_vr_sensor_orbbec_generated_inc}/volumetric_kit/recon/sensor/orbbec/export.hpp"
)

set_target_properties(
  recon_sensor_orbbec
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_orbbec_type recon_sensor_orbbec TYPE)
if(_vr_sensor_orbbec_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_orbbec
                             PUBLIC VR_SENSOR_ORBBEC_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_orbbec PUBLIC cxx_std_17)

# PUBLIC recon_sensor: the sensor *is* an IRgbdSensor, and its header hands out
# the sensor tier's frame. PRIVATE recon_sensor_video, which decodes the
# cameras' colour: no decoder type reaches the public header. The SDK privately
# (see above), compiled with the exceptions it throws (vr_link_orbbec_sdk).
target_link_libraries(
  recon_sensor_orbbec
  PUBLIC recon_sensor
  PRIVATE recon_sensor_video)
vr_link_orbbec_sdk(recon_sensor_orbbec)

target_include_directories(
  recon_sensor_orbbec
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_orbbec_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

# nlohmann/json parses the sync configuration, privately (see recon_sensor).
target_include_directories(recon_sensor_orbbec SYSTEM
                           PRIVATE "${VR_JSON_INCLUDE_DIR}")

vr_target_warnings(recon_sensor_orbbec)

# The target, its headers (recon_sensor's install leaves them out) and the
# generated export header.
if(VR_INSTALL)
  install(
    TARGETS recon_sensor_orbbec
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/sensor/orbbec"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon/sensor"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_sensor_orbbec_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
