# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The Orbbec SDK, for the Orbbec (Femto Mega) capture driver: a prerequisite,
# found and never fetched (the 2026-09-24 decision). Included from the root
# CMakeLists only when VR_WITH_ORBBEC is ON; defines ob::OrbbecSDK and
# VR_ORBBEC_SDK_VERSION.
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
# (-DOrbbecSDK_DIR=<sdk>/lib also works: that is the directory holding the SDK's
# OrbbecSDKConfig.cmake.)
#
# Two quirks of the SDK's own package files, both handled here rather than left
# for each caller to rediscover:
#
# * OrbbecSDKConfig.cmake sits in <sdk>/lib, which none of find_package's
#   standard layouts search under a prefix -- hence PATH_SUFFIXES lib, without
#   which OrbbecSDK_ROOT and CMAKE_PREFIX_PATH both miss it.
# * The version file is named OrbbecSDKVersion.cmake, not
#   OrbbecSDKConfigVersion.cmake, so find_package never reads it and a versioned
#   `find_package(OrbbecSDK 2.9.3)` rejects even a matching SDK. The floor is
#   enforced by reading PACKAGE_VERSION out of that file instead.

# The oldest SDK this repo builds against. Raise it with the SDK the family
# installs, and keep it in lockstep with any sibling that finds the same copy.
set(VR_ORBBEC_SDK_MIN_VERSION 2.9.3)

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
     REGEX "^set\\(PACKAGE_VERSION \"[0-9]+\\.[0-9]+\\.[0-9]+\"\\)")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" VR_ORBBEC_SDK_VERSION
             "${_vr_orbbec_version_line}")
if(NOT VR_ORBBEC_SDK_VERSION)
  message(
    FATAL_ERROR
      "Could not read the Orbbec SDK version from "
      "${OrbbecSDK_DIR}/OrbbecSDKVersion.cmake; the SDK's package layout may "
      "have changed.")
endif()
if(VR_ORBBEC_SDK_VERSION VERSION_LESS VR_ORBBEC_SDK_MIN_VERSION)
  message(
    FATAL_ERROR
      "Orbbec SDK ${VR_ORBBEC_SDK_VERSION} at ${OrbbecSDK_DIR} is older than "
      "the ${VR_ORBBEC_SDK_MIN_VERSION} this repo requires.")
endif()

message(STATUS "Orbbec SDK ${VR_ORBBEC_SDK_VERSION}: ${OrbbecSDK_DIR}")
