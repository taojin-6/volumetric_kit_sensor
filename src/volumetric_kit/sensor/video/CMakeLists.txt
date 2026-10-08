# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- recon_sensor_video: the video decoders ---------------------------------
# HEVC and JPEG decoders on the GPU's hardware, handing out pictures on the
# device only, built only with VR_WITH_FFMPEG (see cmake/vr_ffmpeg.cmake):
# VideoToolbox on Apple, NVDEC and nvJPEG through CUDA elsewhere.
# Camera-neutral: it links recon_core and, on Apple, recon_sensor_apple, and no
# camera tier, so a capture driver links it. FFmpeg is PRIVATE and no FFmpeg
# type reaches the public headers; ffmpeg.hpp is the one include.
include(GenerateExportHeader)
include(GNUInstallDirs)

set(_vr_sensor_video_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(recon_sensor_video)
add_library(volumetric_kit::recon_sensor_video ALIAS recon_sensor_video)

target_sources(recon_sensor_video PRIVATE ffmpeg.cpp frame_color.cpp
                                          hevc_decoder.cpp jpeg_decoder.cpp)

generate_export_header(
  recon_sensor_video
  BASE_NAME
  VR_SENSOR_VIDEO
  EXPORT_MACRO_NAME
  VR_SENSOR_VIDEO_API
  EXPORT_FILE_NAME
  "${_vr_sensor_video_generated_inc}/volumetric_kit/recon/sensor/video/export.hpp"
)

set_target_properties(
  recon_sensor_video
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             OBJCXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_video_type recon_sensor_video TYPE)
if(_vr_sensor_video_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_video
                             PUBLIC VR_SENSOR_VIDEO_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_video PUBLIC cxx_std_17)

target_link_libraries(
  recon_sensor_video
  PUBLIC recon_core
  PRIVATE PkgConfig::VR_FFMPEG)
# NVDEC's pictures copied, and nvJPEG's decoded, into Vulkan buffers CUDA has
# imported, without a trip through the host (cuda_pictures.cpp). Headers only:
# libcuda and libnvjpeg are loaded at run time, so a program built with CUDA
# starts where no NVIDIA driver is.
if(VR_WITH_CUDA)
  target_sources(recon_sensor_video PRIVATE cuda_pictures.cpp)
  target_include_directories(recon_sensor_video SYSTEM
                             PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(recon_sensor_video PRIVATE ${CMAKE_DL_LIBS})
  target_compile_definitions(recon_sensor_video
                             PRIVATE VR_SENSOR_VIDEO_WITH_CUDA=1)
endif()
# VideoToolbox decodes JPEGs (vt_jpeg.cpp), and its pictures reach Vulkan
# through recon_sensor_apple's PixelBufferImporter.
if(APPLE)
  target_sources(recon_sensor_video PRIVATE vt_jpeg.cpp)
  target_link_libraries(
    recon_sensor_video PRIVATE recon_sensor_apple "-framework VideoToolbox"
                               "-framework CoreMedia" "-framework CoreVideo")
endif()

target_include_directories(
  recon_sensor_video
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_video_generated_inc}>"
         "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

vr_target_warnings(recon_sensor_video)

# The target, its headers (recon_sensor's install leaves them out) and the
# generated export header.
if(VR_INSTALL)
  install(
    TARGETS recon_sensor_video
    EXPORT volumetric_kit_reconTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/volumetric_kit/recon/sensor/video"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/volumetric_kit/recon/sensor"
    FILES_MATCHING
    PATTERN "*.hpp")
  install(
    DIRECTORY "${_vr_sensor_video_generated_inc}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.hpp")
endif()
