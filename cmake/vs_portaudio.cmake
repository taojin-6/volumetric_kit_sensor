# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# PortAudio, for the microphone driver (sensor_microphone): a prerequisite,
# found through pkg-config and never fetched (the 2026-09-24 decision). Included
# from the root CMakeLists only when VS_WITH_PORTAUDIO is ON; defines the
# imported target PkgConfig::VS_PORTAUDIO.
#
# Installed once from the platform's packages:
#
# ~~~
# macOS:         brew install portaudio            (19.7.0)
# Ubuntu 24/26:  apt install portaudio19-dev pkg-config  (19.6.0 on 24.04)
# ~~~
#
# Every release since 2011 calls itself "19" to pkg-config, so no floor is
# checked: 19.6.0 is the oldest CI builds.

# PortAudio has no iOS backend.
if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(
    FATAL_ERROR
      "VS_WITH_PORTAUDIO is ON, but PortAudio has no ${CMAKE_SYSTEM_NAME} "
      "build. Configure this build with -DVS_WITH_PORTAUDIO=OFF.")
endif()

find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(VS_PORTAUDIO IMPORTED_TARGET portaudio-2.0)
endif()
if(NOT VS_PORTAUDIO_FOUND)
  message(
    FATAL_ERROR
      "VS_WITH_PORTAUDIO is ON but PortAudio (portaudio-2.0) was not found "
      "through pkg-config. Install it (macOS: brew install portaudio; Ubuntu: "
      "apt install portaudio19-dev pkg-config), or point PKG_CONFIG_PATH at a "
      "PortAudio build.")
endif()

message(STATUS "PortAudio: ${VS_PORTAUDIO_LINK_LIBRARIES}")
