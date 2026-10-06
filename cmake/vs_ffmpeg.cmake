# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# FFmpeg (libavcodec, libavutil), for the video decoders: a prerequisite, found
# and never fetched, like the Orbbec SDK. Included from the root CMakeLists only
# when VR_WITH_FFMPEG is ON; defines the imported target PkgConfig::VR_FFMPEG.
#
# Install it from the platform: `brew install ffmpeg` on macOS, `apt install
# libavcodec-dev libavutil-dev pkg-config` on Ubuntu. The floor is Ubuntu
# 24.04's FFmpeg 6.1 (libavcodec 60.31), the oldest CI builds.

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(
    FATAL_ERROR
      "VR_WITH_FFMPEG is ON, but FFmpeg is found through the host's "
      "pkg-config, which has no ${CMAKE_SYSTEM_NAME} build. Configure this "
      "build with -DVR_WITH_FFMPEG=OFF.")
endif()
# The decoders hand out pictures on the device only: VideoToolbox's on Apple,
# NVDEC's and nvJPEG's through CUDA elsewhere (the 2026-10-06 device-only
# decoder decision).
if(NOT APPLE AND NOT VR_WITH_CUDA)
  message(
    FATAL_ERROR
      "VR_WITH_FFMPEG needs VR_WITH_CUDA here: off Apple, the decoders hand "
      "their pictures out on the device only through CUDA "
      "(cmake/vr_cuda.cmake).")
endif()

find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(VR_FFMPEG IMPORTED_TARGET libavcodec>=60.31
                    libavutil>=58.29)
endif()
if(NOT VR_FFMPEG_FOUND)
  message(
    FATAL_ERROR
      "VR_WITH_FFMPEG is ON but FFmpeg >= 6.1 (libavcodec, libavutil) was not "
      "found through pkg-config. Install it (macOS: brew install ffmpeg; "
      "Ubuntu: apt install libavcodec-dev libavutil-dev pkg-config), or point "
      "PKG_CONFIG_PATH at an FFmpeg build.")
endif()

message(STATUS "FFmpeg: libavcodec ${VR_FFMPEG_libavcodec_VERSION}")

# What the package config asks a consumer's pkg-config for: this FFmpeg's major
# versions, no older than its minors. A static recon_sensor_video is code
# compiled against these headers, which a consumer's libraries of another major
# (or an older minor) lay out differently: that corrupts memory at run time
# rather than failing to link.
set(VR_FFMPEG_CONSUMER_MODULES "")
foreach(_vr_module libavcodec libavutil)
  set(_vr_version "${VR_FFMPEG_${_vr_module}_VERSION}")
  string(REGEX MATCH "^[0-9]+" _vr_major "${_vr_version}")
  math(EXPR _vr_next "${_vr_major} + 1")
  list(APPEND VR_FFMPEG_CONSUMER_MODULES "${_vr_module}>=${_vr_version}"
       "${_vr_module}<${_vr_next}")
endforeach()
