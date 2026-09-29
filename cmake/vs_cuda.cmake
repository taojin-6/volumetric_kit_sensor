# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The CUDA toolkit, for NVIDIA's hardware decoders to leave their pictures on
# the device (the 2026-09-28 decoded-frame decision): NVDEC's through the driver
# API, which copies a picture into a Vulkan buffer CUDA has imported. A
# prerequisite, found and never fetched, like FFmpeg; included from the root
# CMakeLists only when VR_WITH_CUDA is ON, which needs VR_WITH_FFMPEG.
#
# recon's code is host C++ that calls CUDA and compiles none, but CMake's
# FindCUDAToolkit locates the toolkit through nvcc. On Ubuntu, from NVIDIA's
# repository: apt install cuda-nvcc-13-4 cuda-cudart-dev-13-4
# cuda-driver-dev-13-4 \ libnvjpeg-dev-13-4 and point CUDAToolkit_ROOT at
# /usr/local/cuda-13.4 if nvcc is not on PATH.

if(NOT VR_WITH_FFMPEG)
  message(FATAL_ERROR "VR_WITH_CUDA needs VR_WITH_FFMPEG: NVDEC decodes "
                      "through FFmpeg.")
endif()
find_package(CUDAToolkit 13 REQUIRED)
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
