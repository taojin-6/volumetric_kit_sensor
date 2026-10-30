# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_microphone: an OS audio input through PortAudio -----------------
# Microphone, an IAudioSensor over PortAudio (Core Audio on macOS, ALSA on
# Linux), built only with VS_WITH_PORTAUDIO (see cmake/vs_portaudio.cmake). Its
# own target so sensor_interface stays free of the library, which it links
# PRIVATE. capture.cpp, the chunks' stamping and queue, holds no PortAudio, so
# the tests compile it themselves in every build.
include(GenerateExportHeader)

set(_vs_sensor_microphone_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_microphone)
add_library(volumetric_kit::sensor_microphone ALIAS sensor_microphone)

target_sources(sensor_microphone PRIVATE capture.cpp microphone.cpp)

# Named after the target, not the directory, as io's is (DESIGN.md, "Names").
generate_export_header(
  sensor_microphone
  BASE_NAME
  VS_MICROPHONE
  EXPORT_MACRO_NAME
  VS_MICROPHONE_API
  EXPORT_FILE_NAME
  "${_vs_sensor_microphone_generated_inc}/volumetric_kit/sensor/audio/microphone_export.hpp"
)

set_target_properties(
  sensor_microphone
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_microphone_type sensor_microphone TYPE)
if(_vs_sensor_microphone_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_microphone
                             PUBLIC VS_MICROPHONE_STATIC_DEFINE)
endif()

target_compile_features(sensor_microphone PUBLIC cxx_std_17)

# PUBLIC sensor_interface: the source *is* an IAudioSensor.
target_link_libraries(
  sensor_microphone
  PUBLIC sensor_interface
  PRIVATE PkgConfig::VS_PORTAUDIO)

target_include_directories(
  sensor_microphone
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vs_sensor_microphone_generated_inc}>")

vs_target_warnings(sensor_microphone)
