# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

cmake_minimum_required(VERSION 3.21)

project(
  volumetric_kit_sensor
  VERSION 0.1.0
  DESCRIPTION
    "Cameras, capture and the device-frame contract for the volumetric_kit family"
  HOMEPAGE_URL "https://github.com/taojin-6/volumetric_kit_sensor"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Options -----------------------------------------------------------------
# PROJECT_IS_TOP_LEVEL is ON only when this package is the root project, not
# added by recon or an application through add_subdirectory/FetchContent; the
# tests and -Werror default to it, so a consumer's newer compiler shows it our
# new warnings rather than failing its build. The package has no install rules:
# a consumer fetches it by commit.
#
# VS_CAMERA_ONLY builds sensor_camera alone, with no Vulkan (calib's headless
# solver is the consumer it exists for). Every other tier and every driver need
# the rest.
option(VS_CAMERA_ONLY "Build only the camera tier, with no Vulkan" OFF)
option(VS_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(VS_WARNINGS_AS_ERRORS "Treat first-party compiler warnings as errors"
       ${PROJECT_IS_TOP_LEVEL})
# OFF by default: the Orbbec SDK is a prerequisite you install, not a vendored
# dependency, and it ships no iOS build -- the package cross-compiles to iOS
# unchanged. See cmake/vs_orbbec.cmake for how the build finds it.
option(VS_WITH_ORBBEC
       "Build the Orbbec-SDK-backed driver (requires an installed Orbbec SDK)"
       OFF)
# OFF by default: FFmpeg is a prerequisite you install (cmake/vs_ffmpeg.cmake),
# and Apple takes none: its decoders, built always, use VideoToolbox directly.
option(
  VS_WITH_FFMPEG
  "Build the hardware video decoders off Apple (requires FFmpeg and VS_WITH_CUDA)"
  OFF)
# OFF by default too: off Apple, the video decoders decode on NVIDIA's hardware
# through CUDA, so VS_WITH_FFMPEG needs it there (cmake/vs_cuda.cmake).
option(
  VS_WITH_CUDA
  "Decode video on NVIDIA's hardware, onto the device (requires the CUDA toolkit)"
  OFF)

# Semicolon list of sanitizers (e.g. "address;undefined"); empty = off. Applied
# globally -- see cmake/vs_sanitizers.cmake for why.
set(VS_SANITIZE
    ""
    CACHE STRING "Sanitizers to enable, e.g. address;undefined (empty = off)")

if(VS_CAMERA_ONLY)
  foreach(_vs_option VS_WITH_ORBBEC VS_WITH_FFMPEG VS_WITH_CUDA)
    if(${_vs_option})
      message(FATAL_ERROR "VS_CAMERA_ONLY builds the camera tier alone: "
                          "turn ${_vs_option} OFF")
    endif()
  endforeach()
elseif(APPLE)
  # Objective-C++ for the files that reach Metal (apple's
  # pixel_buffer_importer.mm and a test's helper), at the C++ standard the rest
  # compiles at.
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()

include(vs_warnings)
include(vs_sanitizers)

# --- Dependencies + library --------------------------------------------------
# The Vulkan loader (linked at runtime) + headers, from the system SDK (MoltenVK
# on Apple), reached through the core's core/vulkan/vulkan.hpp.
if(NOT VS_CAMERA_ONLY)
  find_package(Vulkan REQUIRED)
endif()

# The Orbbec SDK (ob::OrbbecSDK), FFmpeg and the CUDA toolkit: installed once,
# outside the repo, and never fetched (the 2026-09-24 decision).
if(VS_WITH_ORBBEC)
  include(vs_orbbec)
endif()
if(VS_WITH_FFMPEG)
  include(vs_ffmpeg)
endif()
if(VS_WITH_CUDA)
  include(vs_cuda)
endif()

# The core this package builds with (third_party/ checks the core that arrives):
# 0.29.0, whose vulkan/fwd.hpp the public headers include, with 0.27.0's
# vkc_export_glsl and 0.28.0's test support.
set(VS_CORE_MIN_VERSION 0.29.0)
add_subdirectory(third_party)

add_subdirectory(src)

if(VS_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
