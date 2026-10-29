# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_image_io: host image decoding and PNG encoding --------------------
# On the family core's base tier alone, with no Vulkan and no other tier, so a
# capture app writes its images without the GPU stack. stb's implementation is
# private, compiled here at internal linkage: no public header or target, and no
# clash with another library that defines STB_IMAGE_IMPLEMENTATION.
include(GenerateExportHeader)

set(_vs_io_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_image_io image_io.cpp stb_impl.cpp)
add_library(volumetric_kit::sensor_image_io ALIAS sensor_image_io)
generate_export_header(
  sensor_image_io
  BASE_NAME
  VS_IMAGE_IO
  EXPORT_MACRO_NAME
  VS_IMAGE_IO_API
  EXPORT_FILE_NAME
  "${_vs_io_generated_inc}/volumetric_kit/sensor/io/image_io_export.hpp")
set_target_properties(
  sensor_image_io
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)
get_target_property(_vs_image_io_type sensor_image_io TYPE)
if(_vs_image_io_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_image_io PUBLIC VS_IMAGE_IO_STATIC_DEFINE)
endif()
target_compile_features(sensor_image_io PUBLIC cxx_std_17)
# Catch backend / allocation exceptions at the Status boundary even when a
# consuming application globally disables exceptions. Keep this private.
target_compile_options(sensor_image_io PRIVATE -fexceptions)
# PUBLIC: the header returns the core's Status/Result.
target_link_libraries(sensor_image_io PUBLIC volumetric_kit::core_base)
target_include_directories(
  sensor_image_io PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                         "$<BUILD_INTERFACE:${_vs_io_generated_inc}>")
if(NOT VS_STB_INCLUDE_DIR)
  message(FATAL_ERROR "sensor_image_io: VS_STB_INCLUDE_DIR is unset -- "
                      "third_party/ must be configured before src/.")
endif()
target_include_directories(sensor_image_io SYSTEM
                           PRIVATE "${VS_STB_INCLUDE_DIR}")
vs_target_warnings(sensor_image_io)

# Only the vendored implementation TU opts out of first-party warnings.
set_source_files_properties(stb_impl.cpp PROPERTIES COMPILE_OPTIONS -w)
