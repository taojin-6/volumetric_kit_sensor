# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_interface: the sensor interface ----------------------------------
# IRgbdSensor, the RgbdFrame it hands out, and the camera- and colour-convention
# conversions. The calibration file is the camera tier's
# (camera/array_calibration.hpp). Carries no driver itself: one ships in this
# package only if it can build *and* test it (the 2026-08-02 decision), and the
# one that does, Orbbec, is its own target (orbbec/, below).
#
# Depends on sensor_camera and sensor_frame alone -- the whole tier is host
# types and arithmetic. That is what makes the interface implementable from
# outside: an ARKit driver in volumetric_kit_ios is Objective-C++, and it
# compiles and links against the camera and frame tiers rather than
# preprocessing the Vulkan surface to fill in an intrinsics struct. It is also
# why the tests need no device.
include(GenerateExportHeader)

set(_vs_interface_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_interface)
add_library(volumetric_kit::sensor_interface ALIAS sensor_interface)

target_sources(sensor_interface PRIVATE camera_conventions.cpp
                                        color_conventions.cpp rgbd_sensor.cpp)

# Per-tier export header (VS_INTERFACE_API / VS_INTERFACE_STATIC_DEFINE),
# isolated so the macros never collide with another tier's on one include path.
generate_export_header(
  sensor_interface
  BASE_NAME
  VS_INTERFACE
  EXPORT_MACRO_NAME
  VS_INTERFACE_API
  EXPORT_FILE_NAME
  "${_vs_interface_generated_inc}/volumetric_kit/sensor/export.hpp")

set_target_properties(
  sensor_interface
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_interface_type sensor_interface TYPE)
if(_vs_interface_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_interface PUBLIC VS_INTERFACE_STATIC_DEFINE)
endif()

target_compile_features(sensor_interface PUBLIC cxx_std_17)

# PUBLIC: the headers expose the frame tier's cameras and colour encoding, the
# core's Result, and the camera tier's model and projection, which the lens
# delegates to.
target_link_libraries(sensor_interface PUBLIC sensor_camera sensor_frame)

target_include_directories(
  sensor_interface PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                          "$<BUILD_INTERFACE:${_vs_interface_generated_inc}>")

vs_target_warnings(sensor_interface)

# The GPU pre-processing (Vulkan and shaders, so a target of its own); see
# prep/CMakeLists.txt.
add_subdirectory(prep)

# Several sensors read as one, above every driver; see array/CMakeLists.txt.
add_subdirectory(array)

# Apple's pictures on the device; see apple/CMakeLists.txt. Ahead of the video
# decoders, whose VideoToolbox pictures come through it.
if(APPLE)
  add_subdirectory(apple)
endif()

# The H.265 syntax read on the host, on every build, and the video decoders, on
# every Apple platform and elsewhere with FFmpeg; see video/CMakeLists.txt.
# Ahead of the Orbbec driver, which decodes its cameras' colour with them.
add_subdirectory(video)

# The Orbbec driver; see orbbec/CMakeLists.txt.
if(VS_WITH_ORBBEC)
  add_subdirectory(orbbec)
endif()
