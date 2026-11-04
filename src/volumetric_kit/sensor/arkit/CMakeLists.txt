# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_arkit: ARKit as an IRgbdSensor ------------------------------------
# ArkitSensor on every Apple build, with the system's frameworks alone: ARKit's
# camera on iOS and iPadOS (arkit_feed.mm, which refuses elsewhere), and its
# test pattern everywhere (pattern_feed.mm), which the Mac's tests run. Two
# targets, as DeckLink's driver is split over any capture:
#
# * sensor_arkit_units (static, internal): the host arithmetic on an ARFrame's
#   parts (conversion.cpp), the driver over any feed (driver.cpp): the queue,
#   the depth copies, the counters, the import at poll, and what ARKit's feed
#   does to its session at each event (session_state.cpp). Plain C++ on Core
#   Video, no ARKit, so the tests run it over a scripted feed.
# * sensor_arkit: ArkitSensor, its two feeds and its queue (Objective-C++, with
#   ARC). ARKit and AVFoundation link only on iOS; no ARKit type reaches the
#   public header.
include(GenerateExportHeader)

set(_vs_arkit_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_arkit)
add_library(volumetric_kit::sensor_arkit ALIAS sensor_arkit)

generate_export_header(
  sensor_arkit
  BASE_NAME
  VS_ARKIT
  EXPORT_MACRO_NAME
  VS_ARKIT_API
  EXPORT_FILE_NAME
  "${_vs_arkit_generated_inc}/volumetric_kit/sensor/arkit/export.hpp")

get_target_property(_vs_arkit_type sensor_arkit TYPE)

add_library(sensor_arkit_units STATIC conversion.cpp driver.cpp
                                      session_state.cpp)
add_library(volumetric_kit::sensor_arkit_units ALIAS sensor_arkit_units)
set_target_properties(
  sensor_arkit_units
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)
target_compile_features(sensor_arkit_units PUBLIC cxx_std_17)
# The units include the public header (its enums and constants), whose export
# macro is the target's; a static build defines it away for both.
if(_vs_arkit_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_arkit_units PUBLIC VS_ARKIT_STATIC_DEFINE)
endif()
# PUBLIC: the driver hands out the interface's frames, imports with
# sensor_apple, and reads tags with sensor_video's frame_color.
target_link_libraries(
  sensor_arkit_units PUBLIC sensor_interface sensor_apple sensor_video
                            "-framework CoreVideo")
target_include_directories(
  sensor_arkit_units
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vs_arkit_generated_inc}>"
         "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
vs_target_warnings(sensor_arkit_units)

target_sources(sensor_arkit PRIVATE arkit_feed.mm arkit_sensor.mm
                                    pattern_feed.mm)
set_source_files_properties(arkit_feed.mm arkit_sensor.mm pattern_feed.mm
                            PROPERTIES COMPILE_OPTIONS -fobjc-arc)

set_target_properties(
  sensor_arkit
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             OBJCXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

if(_vs_arkit_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_arkit PUBLIC VS_ARKIT_STATIC_DEFINE)
endif()

target_compile_features(sensor_arkit PUBLIC cxx_std_17)

# PUBLIC sensor_interface: the sensor *is* an IRgbdSensor. The units (and
# through them sensor_apple and sensor_video) privately: no importer or decoder
# type reaches the public header. ARKit and AVFoundation on iOS alone.
target_link_libraries(
  sensor_arkit
  PUBLIC sensor_interface
  PRIVATE sensor_arkit_units "-framework CoreVideo" "-framework Foundation")
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  target_link_libraries(sensor_arkit PRIVATE "-framework ARKit"
                                             "-framework AVFoundation")
endif()

target_include_directories(
  sensor_arkit PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                      "$<BUILD_INTERFACE:${_vs_arkit_generated_inc}>")

vs_target_warnings(sensor_arkit)
