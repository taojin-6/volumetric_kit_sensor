# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# TensorRT and nvcc, for sensor_ffs: Fast-FoundationStereo on NVIDIA's GPUs
# (DECISIONS.md, the 2026-10-29 FFS entry). A prerequisite, found and never
# fetched, like the CUDA toolkit; included from the root CMakeLists only when
# VS_WITH_FFS is ON, which needs VS_WITH_CUDA (cmake/vs_cuda.cmake found the
# toolkit, and nvcc with it).
#
# TensorRT 11 or newer: its headers and libnvinfer and libnvonnxparser. On
# Ubuntu, from NVIDIA's CUDA repository: apt install libnvinfer-dev
# libnvonnxparsers-dev. A tarball's: point TensorRT_ROOT at it.

if(NOT VS_WITH_CUDA)
  message(FATAL_ERROR "VS_WITH_FFS needs VS_WITH_CUDA: FFS reads and writes "
                      "Vulkan memory through CUDA.")
endif()

find_path(
  VS_TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES include)
find_library(
  VS_NVINFER_LIBRARY nvinfer
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64)
find_library(
  VS_NVONNXPARSER_LIBRARY nvonnxparser
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64)
if(NOT VS_TENSORRT_INCLUDE_DIR
   OR NOT VS_NVINFER_LIBRARY
   OR NOT VS_NVONNXPARSER_LIBRARY)
  message(
    FATAL_ERROR
      "VS_WITH_FFS needs TensorRT 11 or newer (NvInfer.h, libnvinfer, "
      "libnvonnxparser): apt install libnvinfer-dev libnvonnxparsers-dev from "
      "NVIDIA's CUDA repository, or point TensorRT_ROOT at a TensorRT tree.")
endif()

# The version: TensorRT 11 names its parts TRT_*_ENTERPRISE, 10 NV_TENSORRT_*.
file(
  STRINGS "${VS_TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _vs_trt_lines
  REGEX
    "^#define (TRT_(MAJOR|MINOR|PATCH)_ENTERPRISE|NV_TENSORRT_(MAJOR|MINOR|PATCH)) [0-9]+"
)
foreach(_vs_part MAJOR MINOR PATCH)
  set(_vs_trt_${_vs_part} "")
  foreach(_vs_line IN LISTS _vs_trt_lines)
    if(_vs_line MATCHES "_${_vs_part}(_ENTERPRISE)? ([0-9]+)")
      set(_vs_trt_${_vs_part} "${CMAKE_MATCH_2}")
    endif()
  endforeach()
endforeach()
set(VS_TENSORRT_VERSION "${_vs_trt_MAJOR}.${_vs_trt_MINOR}.${_vs_trt_PATCH}")
if(VS_TENSORRT_VERSION VERSION_LESS 11)
  message(FATAL_ERROR "VS_WITH_FFS needs TensorRT 11 or newer; found "
                      "${VS_TENSORRT_VERSION} in ${VS_TENSORRT_INCLUDE_DIR}")
endif()
message(STATUS "TensorRT: ${VS_TENSORRT_VERSION} (${VS_NVINFER_LIBRARY})")

add_library(VS::TensorRT INTERFACE IMPORTED)
set_target_properties(
  VS::TensorRT
  PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${VS_TENSORRT_INCLUDE_DIR}"
             INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${VS_TENSORRT_INCLUDE_DIR}"
             INTERFACE_LINK_LIBRARIES
             "${VS_NVINFER_LIBRARY};${VS_NVONNXPARSER_LIBRARY}")

# The kernels: nvcc, the toolkit's, for the major architectures (sm_89 runs
# sm_80's) unless the build names its own.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()
enable_language(CUDA)
