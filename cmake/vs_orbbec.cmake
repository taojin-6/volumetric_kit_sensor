# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The Orbbec SDK, for the Orbbec (Femto Mega) capture driver: a prerequisite,
# found and never fetched (the 2026-09-24 decision). Included from the root
# CMakeLists only when VR_WITH_ORBBEC is ON; defines ob::OrbbecSDK,
# VR_ORBBEC_SDK_VERSION and its _MAJOR / _MINOR / _PATCH parts.
#
# The SDK is a prebuilt binary, so it is installed once and every repo in the
# family that talks to a camera links that one copy -- by convention
# <workspace>/third_party/OrbbecSDK_v<version>, outside every repo. Point the
# build at it with the SDK root, either per build tree or, through the
# environment, once for every repo:
#
# ~~~
# cmake -B build -DVR_WITH_ORBBEC=ON -DOrbbecSDK_ROOT=<sdk>
# export OrbbecSDK_ROOT=<sdk>
# ~~~
#
# Two quirks of the SDK's own package files, handled here:
#
# * OrbbecSDKConfig.cmake sits in <sdk>/lib, which none of find_package's
#   standard layouts search under a prefix -- hence PATH_SUFFIXES lib.
# * The version file is named OrbbecSDKVersion.cmake, not
#   OrbbecSDKConfigVersion.cmake, so find_package never reads it and a versioned
#   `find_package(OrbbecSDK 2.9.3)` rejects even a matching SDK. The version is
#   read out of that file instead, and checked here.

# The oldest SDK this repo builds against. Raise it with the SDK the family
# installs, and keep it in lockstep with any sibling that finds the same copy
# and with the SDK CI installs (.github/workflows/_build.yml).
set(VR_ORBBEC_SDK_MIN_VERSION 2.9.3)

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(
    FATAL_ERROR
      "VR_WITH_ORBBEC is ON, but the Orbbec SDK ships no ${CMAKE_SYSTEM_NAME} "
      "build (macOS is its only Apple platform). Configure this build with "
      "-DVR_WITH_ORBBEC=OFF.")
endif()

# A named root is authoritative. find_package caches the directory it found in
# OrbbecSDK_DIR and keeps it for the life of the build tree, so without this,
# re-pointing OrbbecSDK_ROOT -- at an upgrade, or past the version error below
# -- would be ignored without a word.
if(OrbbecSDK_ROOT OR DEFINED ENV{OrbbecSDK_ROOT})
  unset(OrbbecSDK_DIR CACHE)
endif()

find_package(OrbbecSDK CONFIG PATH_SUFFIXES lib)
if(NOT OrbbecSDK_FOUND)
  message(
    FATAL_ERROR
      "VR_WITH_ORBBEC is ON but the Orbbec SDK was not found. Install the SDK "
      "(>= ${VR_ORBBEC_SDK_MIN_VERSION}, "
      "https://github.com/orbbec/OrbbecSDK_v2/releases) and point the build "
      "at its root with -DOrbbecSDK_ROOT=<sdk> or the OrbbecSDK_ROOT "
      "environment variable.")
endif()

file(STRINGS "${OrbbecSDK_DIR}/OrbbecSDKVersion.cmake" _vr_orbbec_version_line
     REGEX "^set\\(PACKAGE_VERSION \"[0-9.]+\"\\)")
if(NOT _vr_orbbec_version_line MATCHES "\"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"")
  message(
    FATAL_ERROR
      "Could not read the Orbbec SDK version from "
      "${OrbbecSDK_DIR}/OrbbecSDKVersion.cmake; the SDK's package layout may "
      "have changed.")
endif()
set(VR_ORBBEC_SDK_VERSION_MAJOR ${CMAKE_MATCH_1})
set(VR_ORBBEC_SDK_VERSION_MINOR ${CMAKE_MATCH_2})
set(VR_ORBBEC_SDK_VERSION_PATCH ${CMAKE_MATCH_3})
set(VR_ORBBEC_SDK_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")

# The floor and the same major version -- the rule the SDK's own version file
# encodes, had find_package been able to read it.
string(REGEX MATCH "^[0-9]+" _vr_orbbec_min_major
             "${VR_ORBBEC_SDK_MIN_VERSION}")
if(VR_ORBBEC_SDK_VERSION VERSION_LESS VR_ORBBEC_SDK_MIN_VERSION
   OR NOT VR_ORBBEC_SDK_VERSION_MAJOR EQUAL _vr_orbbec_min_major)
  message(
    FATAL_ERROR
      "Orbbec SDK ${VR_ORBBEC_SDK_VERSION} at ${OrbbecSDK_DIR} is not one this "
      "repo builds against: it requires ${VR_ORBBEC_SDK_MIN_VERSION} or a "
      "newer ${_vr_orbbec_min_major}.x release. Point OrbbecSDK_ROOT at such "
      "an SDK; the next configure finds it.")
endif()

message(STATUS "Orbbec SDK ${VR_ORBBEC_SDK_VERSION}: ${OrbbecSDK_DIR}")
