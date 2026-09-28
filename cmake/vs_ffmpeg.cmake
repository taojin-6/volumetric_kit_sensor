# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# FFmpeg (libavcodec, libavutil, libswscale), for the video decoder: a
# prerequisite, found and never fetched, like the Orbbec SDK. Included from the
# root CMakeLists only when VR_WITH_FFMPEG is ON; defines the imported target
# PkgConfig::VR_FFMPEG.
#
# Install it from the platform: `brew install ffmpeg` on macOS, `apt install
# libavcodec-dev libavutil-dev libswscale-dev pkg-config` on Ubuntu. The floor
# is Ubuntu 22.04's FFmpeg 4.4 (libavcodec 58.134), the oldest CI builds.

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(
    FATAL_ERROR
      "VR_WITH_FFMPEG is ON, but FFmpeg is found through the host's "
      "pkg-config, which has no ${CMAKE_SYSTEM_NAME} build. Configure this "
      "build with -DVR_WITH_FFMPEG=OFF.")
endif()

find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(VR_FFMPEG IMPORTED_TARGET libavcodec>=58.134
                    libavutil>=56.70 libswscale>=5.9)
endif()
if(NOT VR_FFMPEG_FOUND)
  message(
    FATAL_ERROR
      "VR_WITH_FFMPEG is ON but FFmpeg >= 4.4 (libavcodec, libavutil, "
      "libswscale) was not found through pkg-config. Install it (macOS: brew "
      "install ffmpeg; Ubuntu: apt install libavcodec-dev libavutil-dev "
      "libswscale-dev pkg-config), or point PKG_CONFIG_PATH at an FFmpeg "
      "build.")
endif()

message(STATUS "FFmpeg: libavcodec ${VR_FFMPEG_libavcodec_VERSION}")
