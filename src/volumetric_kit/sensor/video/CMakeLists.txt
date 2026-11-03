# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

include(GenerateExportHeader)

set(_vs_sensor_video_generated_inc
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

# --- sensor_video_syntax: H.265 syntax, read on the host ---------------
# An Annex B access unit's NAL units and its first slice's type, the parameter
# sets and the start of a slice header (hevc_syntax.hpp), with the H.273 colour
# codes an SPS declares (color_codes.hpp), and the gate that says which units
# can be decoded after a loss (key_gate.hpp). Pure host code on the family
# core's base tier, with no codec, no GPU and no FFmpeg, so it builds on every
# platform, where the decoders below do not. VideoToolbox's HEVC path reads its
# streams with it.
add_library(sensor_video_syntax)
add_library(volumetric_kit::sensor_video_syntax ALIAS sensor_video_syntax)

target_sources(sensor_video_syntax PRIVATE hevc_slices.cpp hevc_syntax.cpp
                                           key_gate.cpp)

generate_export_header(
  sensor_video_syntax
  BASE_NAME
  VS_VIDEO_SYNTAX
  EXPORT_MACRO_NAME
  VS_VIDEO_SYNTAX_API
  EXPORT_FILE_NAME
  "${_vs_sensor_video_generated_inc}/volumetric_kit/sensor/video/syntax_export.hpp"
)

set_target_properties(
  sensor_video_syntax
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_video_syntax_type sensor_video_syntax TYPE)
if(_vs_sensor_video_syntax_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_video_syntax
                             PUBLIC VS_VIDEO_SYNTAX_STATIC_DEFINE)
endif()

target_compile_features(sensor_video_syntax PUBLIC cxx_std_17)
# Catch allocation failures at the Status boundary even when a consuming
# application disables exceptions. Private: no public header reaches it.
target_compile_options(sensor_video_syntax PRIVATE -fexceptions)

# PUBLIC: the header returns the core's Status/Result.
target_link_libraries(sensor_video_syntax PUBLIC volumetric_kit::core_base)

target_include_directories(
  sensor_video_syntax
  PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${_vs_sensor_video_generated_inc}>")

vs_target_warnings(sensor_video_syntax)

# The coders, on every Apple platform and elsewhere with FFmpeg.
if(NOT APPLE AND NOT VS_WITH_FFMPEG)
  return()
endif()

# --- sensor_video: the video coders ------------------------------------------
# HEVC and JPEG decoders on the GPU's hardware, handing out pictures on the
# device only, and the HEVC encoder: VideoToolbox's on every Apple platform,
# with no FFmpeg, and NVDEC, nvJPEG and NVENC through CUDA elsewhere, built
# only with VS_WITH_FFMPEG (see cmake/vs_ffmpeg.cmake). Camera-neutral: it links
# sensor_frame and, on Apple, sensor_apple, and no camera tier, so a capture
# driver links it. No FFmpeg type
# reaches the public headers, and a platform type only Apple's encoder entry
# (video/hevc_encoder_apple.hpp, CVPixelBufferRef); ffmpeg.hpp is FFmpeg's one
# include and videotoolbox.hpp the one home of VideoToolbox's sessions.
# hevc_encoder.cpp holds the encoder's rules and coding thread for every
# backend, its session the backend's (hevc_encode_session.hpp).
add_library(sensor_video)
add_library(volumetric_kit::sensor_video ALIAS sensor_video)

target_sources(sensor_video PRIVATE frame_color.cpp hevc_decoder.cpp
                                    hevc_encoder.cpp jpeg_decoder.cpp)

generate_export_header(
  sensor_video
  BASE_NAME
  VS_VIDEO
  EXPORT_MACRO_NAME
  VS_VIDEO_API
  EXPORT_FILE_NAME
  "${_vs_sensor_video_generated_inc}/volumetric_kit/sensor/video/export.hpp")

set_target_properties(
  sensor_video
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             OBJCXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)

get_target_property(_vs_sensor_video_type sensor_video TYPE)
if(_vs_sensor_video_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_video PUBLIC VS_VIDEO_STATIC_DEFINE)
endif()

target_compile_features(sensor_video PUBLIC cxx_std_17)

# PUBLIC: video/frame_color.hpp resolves the H.273 codes a stream declares
# (ColorCodes), which sensor_video_syntax reads off its SPS (hevc::Sps::color);
# on Apple the decoder reads its streams with that syntax too.
target_link_libraries(sensor_video PUBLIC sensor_frame sensor_video_syntax)
if(APPLE)
  # VideoToolbox decodes on the hardware directly (vt_hevc.cpp, vt_jpeg.cpp,
  # their sessions in videotoolbox.cpp), and its pictures reach Vulkan through
  # sensor_apple's PixelBufferImporter. The H.265 syntax it is handed slices by
  # is sensor_video_syntax's. It encodes on the hardware too
  # (vt_hevc_encode.cpp), a CVPixelBuffer sent as it is
  # (hevc_encoder_apple.cpp), whose header makes CoreVideo PUBLIC.
  target_sources(
    sensor_video PRIVATE hevc_encoder_apple.cpp videotoolbox.cpp vt_hevc.cpp
                         vt_hevc_encode.cpp vt_jpeg.cpp)
  target_link_libraries(sensor_video PUBLIC "-framework CoreVideo")
  target_link_libraries(
    sensor_video PRIVATE sensor_apple "-framework VideoToolbox"
                         "-framework CoreMedia" "-framework CoreFoundation")
else()
  # NVDEC's pictures copied into Vulkan plane images, and nvJPEG's decoded into
  # Vulkan buffers, that CUDA has imported, without a trip through the host
  # (cuda_pictures.cpp), through the public CUDA interop (cuda_interop.hpp,
  # which includes cuda.h, so the toolkit's headers are PUBLIC), and handed out
  # before CUDA's work has run, with the value of an exported timeline CUDA
  # signals after it (cuda_timeline.cpp). NVENC reads the encoder's targets,
  # exported buffers the device writes, through CUDA too (nvenc_session.cpp).
  # FFmpeg is PRIVATE. CUDA's headers only: libcuda and libnvjpeg are loaded at
  # run time (cuda_driver.cpp), so a program built with CUDA starts where no
  # NVIDIA driver is.
  target_sources(
    sensor_video PRIVATE ffmpeg.cpp cuda_driver.cpp cuda_interop.cpp
                         cuda_pictures.cpp cuda_timeline.cpp nvenc_session.cpp)
  target_include_directories(sensor_video SYSTEM
                             PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(sensor_video PRIVATE PkgConfig::VS_FFMPEG
                                             ${CMAKE_DL_LIBS})
  target_compile_definitions(sensor_video PRIVATE VS_VIDEO_WITH_CUDA=1)
endif()

target_include_directories(
  sensor_video PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                      "$<BUILD_INTERFACE:${_vs_sensor_video_generated_inc}>")

vs_target_warnings(sensor_video)

# --- sensor_video_testing: the encoder rules, for any runner -----------------
# hevc_encoder_breach (testing/hevc_conformance.hpp): the rules every backend's
# HevcEncoder keeps, checked with sensor's syntax readers and decoder, for the
# package's tests and an application's on a device; and Nv12Writer
# (testing/nv12_writer.hpp), the compute kernels that write a test picture into
# an encoder's target as a producer's kernel would: plane images
# (shaders/write_nv12.comp) or a buffer's rows (shaders/write_nv12_buffer.comp),
# embedded. Static, with no test framework and no core test support, so it
# builds wherever sensor_video does; the test pattern is
# testing/nv12_pattern.hpp's.
add_library(sensor_video_testing STATIC hevc_conformance.cpp nv12_writer.cpp)
add_library(volumetric_kit::sensor_video_testing ALIAS sensor_video_testing)
include(vs_embed)
vs_embed_shaders(
  sensor_video_testing SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/write_nv12.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/write_nv12_buffer.comp")

generate_export_header(
  sensor_video_testing
  BASE_NAME
  VS_VIDEO_TESTING
  EXPORT_MACRO_NAME
  VS_VIDEO_TESTING_API
  EXPORT_FILE_NAME
  "${_vs_sensor_video_generated_inc}/volumetric_kit/sensor/video/testing_export.hpp"
)
target_compile_definitions(sensor_video_testing
                           PUBLIC VS_VIDEO_TESTING_STATIC_DEFINE)
set_target_properties(
  sensor_video_testing
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON)
target_compile_features(sensor_video_testing PUBLIC cxx_std_17)
target_compile_options(sensor_video_testing PRIVATE -fexceptions)
target_link_libraries(sensor_video_testing PUBLIC sensor_video)
if(APPLE)
  target_link_libraries(
    sensor_video_testing PRIVATE "-framework CoreVideo"
                                 "-framework CoreFoundation")
endif()
vs_target_warnings(sensor_video_testing)
