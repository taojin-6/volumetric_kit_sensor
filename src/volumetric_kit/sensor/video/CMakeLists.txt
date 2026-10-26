# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GenerateExportHeader)

set(_vr_sensor_video_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

# --- recon_sensor_video_syntax: H.265 syntax, read on the host ---------------
# An Annex B access unit's NAL units, the parameter sets and the start of a
# slice header (hevc_syntax.hpp), with the H.273 colour codes an SPS declares
# (color_codes.hpp). Pure host code on the family core's base tier, with no
# codec, no GPU and no FFmpeg, so it builds on every platform, where the
# decoders below do not. VideoToolbox's HEVC path reads its streams with it.
add_library(recon_sensor_video_syntax)
add_library(volumetric_kit::recon_sensor_video_syntax ALIAS
            recon_sensor_video_syntax)

target_sources(recon_sensor_video_syntax PRIVATE hevc_syntax.cpp)

generate_export_header(
  recon_sensor_video_syntax
  BASE_NAME
  VR_SENSOR_VIDEO_SYNTAX
  EXPORT_MACRO_NAME
  VR_SENSOR_VIDEO_SYNTAX_API
  EXPORT_FILE_NAME
  "${_vr_sensor_video_generated_inc}/volumetric_kit/recon/sensor/video/syntax_export.hpp"
)

set_target_properties(
  recon_sensor_video_syntax
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vr_sensor_video_syntax_type recon_sensor_video_syntax TYPE)
if(_vr_sensor_video_syntax_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(recon_sensor_video_syntax
                             PUBLIC VR_SENSOR_VIDEO_SYNTAX_STATIC_DEFINE)
endif()

target_compile_features(recon_sensor_video_syntax PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary even when a consuming
# application disables exceptions. Private: no public header reaches it.
target_compile_options(recon_sensor_video_syntax PRIVATE -fexceptions)

# PUBLIC: the header returns the core's Status/Result.
target_link_libraries(recon_sensor_video_syntax
                      PUBLIC volumetric_kit::core_base)

target_include_directories(
  recon_sensor_video_syntax
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_video_generated_inc}>")

vr_target_warnings(recon_sensor_video_syntax)

# The decoders, on every Apple platform and elsewhere with FFmpeg.
if(NOT APPLE AND NOT VR_WITH_FFMPEG)
  return()
endif()

# --- recon_sensor_video: the video decoders ---------------------------------
# HEVC and JPEG decoders on the GPU's hardware, handing out pictures on the
# device only: VideoToolbox's on every Apple platform, with no FFmpeg, and NVDEC
# and nvJPEG through CUDA elsewhere, built only with VR_WITH_FFMPEG (see
# cmake/vr_ffmpeg.cmake). Camera-neutral: it links recon_core and, on Apple,
# recon_sensor_apple, and no camera tier, so a capture driver links it. No
# FFmpeg or platform type reaches the public headers; ffmpeg.hpp is FFmpeg's one
# include and videotoolbox.hpp the one home of VideoToolbox's sessions.
add_library(recon_sensor_video)
add_library(volumetric_kit::recon_sensor_video ALIAS recon_sensor_video)

target_sources(recon_sensor_video PRIVATE frame_color.cpp hevc_decoder.cpp
                                          jpeg_decoder.cpp)

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

# PRIVATE: the colour codes (frame_color.hpp) and, on Apple, the H.265 syntax
# are read inside; no public header of the decoders includes them.
target_link_libraries(
  recon_sensor_video
  PUBLIC recon_core
  PRIVATE recon_sensor_video_syntax)
if(APPLE)
  # VideoToolbox decodes on the hardware directly (vt_hevc.cpp, vt_jpeg.cpp,
  # their sessions in videotoolbox.cpp), and its pictures reach Vulkan through
  # recon_sensor_apple's PixelBufferImporter. The H.265 syntax it is handed
  # slices by is recon_sensor_video_syntax's.
  target_sources(recon_sensor_video PRIVATE videotoolbox.cpp vt_hevc.cpp
                                            vt_jpeg.cpp)
  target_link_libraries(
    recon_sensor_video
    PRIVATE recon_sensor_apple "-framework VideoToolbox" "-framework CoreMedia"
            "-framework CoreVideo" "-framework CoreFoundation")
else()
  # NVDEC's pictures copied into Vulkan plane images, and nvJPEG's decoded into
  # Vulkan buffers, that CUDA has imported, without a trip through the host
  # (cuda_pictures.cpp), through the public CUDA interop (cuda_interop.hpp,
  # which includes cuda.h, so the toolkit's headers are PUBLIC). FFmpeg is
  # PRIVATE. CUDA's headers only: libcuda and libnvjpeg are loaded at run time
  # (cuda_driver.cpp), so a program built with CUDA starts where no NVIDIA
  # driver is.
  target_sources(recon_sensor_video PRIVATE ffmpeg.cpp cuda_driver.cpp
                                            cuda_interop.cpp cuda_pictures.cpp)
  target_include_directories(recon_sensor_video SYSTEM
                             PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(recon_sensor_video PRIVATE PkgConfig::VR_FFMPEG
                                                   ${CMAKE_DL_LIBS})
  target_compile_definitions(recon_sensor_video
                             PRIVATE VR_SENSOR_VIDEO_WITH_CUDA=1)
endif()

target_include_directories(
  recon_sensor_video
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vr_sensor_video_generated_inc}>")

vr_target_warnings(recon_sensor_video)
