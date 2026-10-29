# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# FFmpeg (libavcodec, libavutil), for the video decoders off Apple, where NVDEC
# decodes through it: a prerequisite, found and never fetched, like the Orbbec
# SDK. Included from the root CMakeLists only when VS_WITH_FFMPEG is ON; defines
# the imported target PkgConfig::VS_FFMPEG.
#
# Install it from the platform: `apt install libavcodec-dev libavutil-dev
# pkg-config` on Ubuntu. The floor is Ubuntu 24.04's FFmpeg 6.1 (libavcodec
# 60.31), the oldest CI builds.

# Apple's decoders use VideoToolbox directly and are built with no FFmpeg (the
# 2026-10-25 decision), so the option has nothing to do there.
if(APPLE)
  message(
    FATAL_ERROR
      "VS_WITH_FFMPEG is ON, but on Apple the video decoders use VideoToolbox "
      "directly and need no FFmpeg. Configure this build with "
      "-DVS_WITH_FFMPEG=OFF.")
endif()
# The decoders hand out pictures on the device only: NVDEC's and nvJPEG's
# through CUDA (the 2026-10-06 device-only decoder decision).
if(NOT VS_WITH_CUDA)
  message(
    FATAL_ERROR
      "VS_WITH_FFMPEG needs VS_WITH_CUDA: off Apple, the decoders hand their "
      "pictures out on the device only through CUDA (cmake/vs_cuda.cmake).")
endif()

find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(VS_FFMPEG IMPORTED_TARGET libavcodec>=60.31
                    libavutil>=58.29)
endif()
if(NOT VS_FFMPEG_FOUND)
  message(
    FATAL_ERROR
      "VS_WITH_FFMPEG is ON but FFmpeg >= 6.1 (libavcodec, libavutil) was not "
      "found through pkg-config. Install it (Ubuntu: apt install "
      "libavcodec-dev libavutil-dev pkg-config), or point PKG_CONFIG_PATH at "
      "an FFmpeg build.")
endif()

message(STATUS "FFmpeg: libavcodec ${VS_FFMPEG_libavcodec_VERSION}")
