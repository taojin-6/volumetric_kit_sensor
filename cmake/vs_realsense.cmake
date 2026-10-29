# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# librealsense, for the RealSense (D400) capture driver: a prerequisite, found
# and never fetched (the 2026-09-24 decision). Included from the root CMakeLists
# only when VS_WITH_REALSENSE is ON; defines realsense2::realsense2 (the SDK's
# own package), VS_REALSENSE_SDK_VERSION and its _MAJOR / _MINOR / _PATCH parts,
# and vs_link_realsense_sdk.
#
# Installed once from the platform's packages, which carry the SDK's CMake
# package (no source build):
#
# ~~~
# macOS:         brew install librealsense
# Ubuntu 24/26:  RealSense's apt repository, librealsense2-dev
#                (https://github.com/realsenseai/librealsense,
#                 doc/distribution_linux.md); CI's install-librealsense action
# ~~~

# The oldest SDK this package builds against. Keep it in lockstep with the SDK
# CI installs (.github/actions/install-librealsense and the macOS formula).
set(VS_REALSENSE_SDK_MIN_VERSION 2.59.1)

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(
    FATAL_ERROR
      "VS_WITH_REALSENSE is ON, but librealsense has no ${CMAKE_SYSTEM_NAME} "
      "build. Configure this build with -DVS_WITH_REALSENSE=OFF.")
endif()

# The floor through find_package (the SDK's version file accepts any newer
# release), and the same major here, which that file does not check.
find_package(realsense2 ${VS_REALSENSE_SDK_MIN_VERSION} CONFIG)
if(NOT realsense2_FOUND)
  message(
    FATAL_ERROR
      "VS_WITH_REALSENSE is ON but librealsense >= "
      "${VS_REALSENSE_SDK_MIN_VERSION} was not found. Install it (`brew install "
      "librealsense` on macOS, librealsense2-dev from RealSense's apt "
      "repository on Ubuntu) or point the build at its package with "
      "-Drealsense2_DIR=<dir>.")
endif()
if(NOT realsense2_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  message(
    FATAL_ERROR
      "Could not read librealsense's version (\"${realsense2_VERSION}\") from "
      "${realsense2_DIR}; its package layout may have changed.")
endif()
set(VS_REALSENSE_SDK_VERSION_MAJOR ${CMAKE_MATCH_1})
set(VS_REALSENSE_SDK_VERSION_MINOR ${CMAKE_MATCH_2})
set(VS_REALSENSE_SDK_VERSION_PATCH ${CMAKE_MATCH_3})
set(VS_REALSENSE_SDK_VERSION
    "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
string(REGEX MATCH "^[0-9]+" _vs_realsense_min_major
             "${VS_REALSENSE_SDK_MIN_VERSION}")
if(NOT VS_REALSENSE_SDK_VERSION_MAJOR EQUAL _vs_realsense_min_major)
  message(
    FATAL_ERROR
      "librealsense ${VS_REALSENSE_SDK_VERSION} at ${realsense2_DIR} is not one "
      "the sensor package builds against: it requires a "
      "${_vs_realsense_min_major}.x release from "
      "${VS_REALSENSE_SDK_MIN_VERSION}.")
endif()

message(STATUS "librealsense ${VS_REALSENSE_SDK_VERSION}: ${realsense2_DIR}")

# vs_link_realsense_sdk(<target>): links the SDK, privately, and compiles the
# target with exceptions. The SDK reports every failure by throwing rs2::error,
# so code that calls it needs exceptions to catch one, even where
# CMAKE_CXX_FLAGS turns them off; the driver turns each into a Status, and none
# crosses its API.
function(vs_link_realsense_sdk target)
  target_link_libraries(${target} PRIVATE realsense2::realsense2)
  target_compile_options(${target} PRIVATE -fexceptions)
endfunction()
