# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_orbbec: the Orbbec (Femto Mega) capture driver -------------
# An ICameraCapture over the Orbbec SDK, built only with VR_WITH_ORBBEC (see
# cmake/vr_orbbec.cmake). Its own target so recon_sensor -- the contract an
# out-of-tree driver compiles against -- stays free of the SDK (the 2026-09-26
# decision). The SDK is PRIVATE and no SDK type reaches the public header, but
# linking still needs its library, so the package config re-finds the SDK.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_orbbec_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_orbbec)
add_library(volumetric_kit::recon_sensor_orbbec ALIAS recon_sensor_orbbec)

target_sources(recon_sensor_orbbec PRIVATE orbbec_capture.cpp
                                           frame_conversion.cpp)

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

# PUBLIC recon_sensor: the capture *is* an ICameraCapture, and its header hands
# out the contract's frame. PRIVATE ob::OrbbecSDK: see above.
target_link_libraries(
  recon_sensor_orbbec
  PUBLIC recon_sensor
  PRIVATE ob::OrbbecSDK)

target_include_directories(
  recon_sensor_orbbec
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_orbbec_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

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
