# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_camera: the family's camera vocabulary ----------------------------
# The camera model (image size, intrinsics, OpenCV's rational lens), the lens as
# a scalar template, rigid transforms, and the sensor array's calibration file
# (the 2026-10-06 decisions).
#
# It links volumetric_kit_core's base tier and GLM and nothing else -- no Vulkan
# and no recon_core -- so a driver can report its factory calibration in it
# without a GPU API. The sensor tier builds on it.
include(GenerateExportHeader)
include(GNUInstallDirs)

# The canonical glm::glm target, as recon_core finds it (see that file for why);
# an imported target is visible only in the directory that finds it.
find_package(glm CONFIG REQUIRED)

set(_vr_camera_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_camera)
add_library(volumetric_kit::recon_camera ALIAS recon_camera)

target_sources(recon_camera PRIVATE array_calibration.cpp camera_model.cpp
                                    geometry.cpp)

generate_export_header(
  recon_camera
  BASE_NAME
  VR_CAMERA
  EXPORT_MACRO_NAME
  VR_CAMERA_API
  EXPORT_FILE_NAME
  "${_vr_camera_generated_inc}/volumetric_kit/recon/camera/export.hpp")

set_target_properties(
  recon_camera
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_camera_type recon_camera TYPE)
if(_vr_camera_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_camera PUBLIC VR_CAMERA_STATIC_DEFINE)
endif()

target_compile_features(recon_camera PUBLIC cxx_std_17)

# PUBLIC: the headers return the core's Status/Result and are written in GLM's
# double types.
target_link_libraries(recon_camera PUBLIC volumetric_kit::core_base glm::glm)

# nlohmann/json parses the calibration file: a PRIVATE SYSTEM include, so it
# reaches no public header, no link line and no consumer.
if(NOT VR_JSON_INCLUDE_DIR)
  message(FATAL_ERROR "recon_camera: VR_JSON_INCLUDE_DIR is unset -- "
                      "third_party/ must be configured before src/.")
endif()
target_include_directories(recon_camera SYSTEM PRIVATE "${VR_JSON_INCLUDE_DIR}")

target_include_directories(
  recon_camera
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_camera_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_camera)

if(VR_INSTALL)
  install(
    TARGETS recon_camera
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/camera"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_camera_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
