# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_decklink: two DeckLink SDI inputs as one stereo camera -----------
# Three targets, so everything but the SDK's calls builds and is tested on every
# leg, with no SDK:
#
# * sensor_decklink_units (static, internal): the connectors and modes, the
#   pairing and its clock map, the skew report's numbers, the page pool, the
#   device pictures, the upload thread, and the driver over any capture
#   (driver.hpp). Built on every build that asks for it (the tests do).
# * sensor_decklink_capture (static, internal, VS_WITH_DECKLINK): capture.cpp,
#   the one SDK user, and the SDK's dispatch file. The driver and
#   decklink_skew_report share it.
# * sensor_decklink (VS_WITH_DECKLINK): DecklinkStereo, the public class. The
#   SDK is PRIVATE and no SDK type reaches its header.
include(GenerateExportHeader)

add_library(sensor_decklink_units STATIC EXCLUDE_FROM_ALL
            driver.cpp pairing.cpp ports.cpp skew_stats.cpp upload.cpp)
add_library(volumetric_kit::sensor_decklink_units ALIAS sensor_decklink_units)
set_target_properties(
  sensor_decklink_units
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)
target_compile_features(sensor_decklink_units PUBLIC cxx_std_17)
# PUBLIC: the units hand out StereoFrames of device buffers.
target_link_libraries(sensor_decklink_units PUBLIC sensor_stereo
                                                   volumetric_kit::core_vulkan)
target_include_directories(
  sensor_decklink_units PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
vs_target_warnings(sensor_decklink_units)

if(NOT VS_WITH_DECKLINK)
  return()
endif()

add_library(sensor_decklink_capture STATIC capture.cpp)
set_target_properties(
  sensor_decklink_capture
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)
target_link_libraries(sensor_decklink_capture PUBLIC sensor_decklink_units)
vs_link_decklink_sdk(sensor_decklink_capture)
vs_target_warnings(sensor_decklink_capture)

set(_vs_decklink_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_decklink)
add_library(volumetric_kit::sensor_decklink ALIAS sensor_decklink)
target_sources(sensor_decklink PRIVATE decklink_stereo.cpp)

generate_export_header(
  sensor_decklink
  BASE_NAME
  VS_DECKLINK
  EXPORT_MACRO_NAME
  VS_DECKLINK_API
  EXPORT_FILE_NAME
  "${_vs_decklink_generated_inc}/volumetric_kit/sensor/decklink/export.hpp")

set_target_properties(
  sensor_decklink
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_decklink_type sensor_decklink TYPE)
if(_vs_decklink_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_decklink PUBLIC VS_DECKLINK_STATIC_DEFINE)
endif()

target_compile_features(sensor_decklink PUBLIC cxx_std_17)

# PUBLIC sensor_stereo: the camera *is* an IStereoSensor. The units and the
# capture privately (static, so a static build's consumer links them too).
target_link_libraries(
  sensor_decklink
  PUBLIC sensor_stereo
  PRIVATE sensor_decklink_units sensor_decklink_capture)

target_include_directories(
  sensor_decklink PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                         "$<BUILD_INTERFACE:${_vs_decklink_generated_inc}>")

vs_target_warnings(sensor_decklink)
