# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_camera: the family's camera vocabulary ---------------------------
# The camera model (image size, intrinsics, OpenCV's rational lens), the lens as
# a scalar template, rigid transforms, and the sensor array's calibration file
# (the 2026-10-06 decisions); and the package's version, which a camera-only
# build has too.
#
# It links volumetric_kit_core's base tier and GLM and nothing else -- no Vulkan
# and no other sensor tier -- so a driver can report its factory calibration in
# it without a GPU API. VS_CAMERA_ONLY builds it alone.
include(GenerateExportHeader)

# The canonical glm::glm target: every glm packaging provides it, where the
# newer glm::glm-header-only does not exist on older ones (Ubuntu's libglm-dev).
# An imported target is visible only in the directory that finds it.
find_package(glm CONFIG REQUIRED)

set(_vs_camera_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_camera)
add_library(volumetric_kit::sensor_camera ALIAS sensor_camera)

target_sources(sensor_camera PRIVATE array_calibration.cpp camera_model.cpp
                                     geometry.cpp version.cpp)

generate_export_header(
  sensor_camera
  BASE_NAME
  VS_CAMERA
  EXPORT_MACRO_NAME
  VS_CAMERA_API
  EXPORT_FILE_NAME
  "${_vs_camera_generated_inc}/volumetric_kit/sensor/camera/export.hpp")
configure_file(
  "${PROJECT_SOURCE_DIR}/include/volumetric_kit/sensor/version.hpp.in"
  "${_vs_camera_generated_inc}/volumetric_kit/sensor/version.hpp" @ONLY)

set_target_properties(
  sensor_camera
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_camera_type sensor_camera TYPE)
if(_vs_camera_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_camera PUBLIC VS_CAMERA_STATIC_DEFINE)
endif()

target_compile_features(sensor_camera PUBLIC cxx_std_17)

# PUBLIC: the headers return the core's Status/Result and are written in GLM's
# double types.
target_link_libraries(sensor_camera PUBLIC volumetric_kit::core_base glm::glm)

# nlohmann/json parses the calibration file: a PRIVATE SYSTEM include, so it
# reaches no public header, no link line and no consumer.
if(NOT VS_JSON_INCLUDE_DIR)
  message(FATAL_ERROR "sensor_camera: VS_JSON_INCLUDE_DIR is unset -- "
                      "third_party/ must be configured before src/.")
endif()
target_include_directories(sensor_camera SYSTEM
                           PRIVATE "${VS_JSON_INCLUDE_DIR}")

target_include_directories(
  sensor_camera PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                       "$<BUILD_INTERFACE:${_vs_camera_generated_inc}>")

vs_target_warnings(sensor_camera)
