# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# --- sensor_ffs: stereo depth by Fast-FoundationStereo (VS_WITH_FFS) ----------
# FfsStereo, the network on TensorRT with its input and depth made by two CUDA
# kernels (ffs_kernels.cu), and StereoDepthSensor, any stereo camera as an RGB-D
# sensor on a thread of its own. NVIDIA only: it reads the rectifier's views and
# writes depth through the video tier's CUDA interop (cuda_interop.hpp), on the
# primary context its CudaContext retains, which the internal driver loader's
# CudaContextScope makes current (../video/cuda_driver.hpp). No CUDA or TensorRT
# type reaches its headers. TensorRT and the CUDA runtime are PRIVATE:
# libnvinfer and libnvonnxparser are shared libraries a program loads; cudart is
# static and loads libcuda at run time, as the decoders do.
include(GenerateExportHeader)

set(_vs_ffs_generated_inc "${CMAKE_CURRENT_BINARY_DIR}/generated/include")

add_library(sensor_ffs)
add_library(volumetric_kit::sensor_ffs ALIAS sensor_ffs)

target_sources(sensor_ffs PRIVATE ffs_stereo.cpp ffs_kernels.cu
                                  stereo_depth_sensor.cpp)

generate_export_header(
  sensor_ffs
  BASE_NAME
  VS_FFS
  EXPORT_MACRO_NAME
  VS_FFS_API
  EXPORT_FILE_NAME
  "${_vs_ffs_generated_inc}/volumetric_kit/sensor/ffs/export.hpp")

set_target_properties(
  sensor_ffs
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             CUDA_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             CUDA_EXTENSIONS OFF
             # Linked below, so a consumer's C++ link has it too.
             CUDA_RUNTIME_LIBRARY None)

get_target_property(_vs_ffs_type sensor_ffs TYPE)
if(_vs_ffs_type STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(sensor_ffs PUBLIC VS_FFS_STATIC_DEFINE)
endif()

target_compile_features(sensor_ffs PUBLIC cxx_std_17)
target_compile_options(sensor_ffs
                       PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fexceptions>)

# PUBLIC: its headers take the stereo tier's rectifier and pairs and the
# interface's IRgbdSensor. PRIVATE: CUDA through the video tier, TensorRT.
target_link_libraries(
  sensor_ffs
  PUBLIC sensor_stereo sensor_interface
  PRIVATE sensor_video VS::TensorRT CUDA::cudart_static ${CMAKE_DL_LIBS})

target_include_directories(
  sensor_ffs PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
                    "$<BUILD_INTERFACE:${_vs_ffs_generated_inc}>")

vs_target_warnings(sensor_ffs)
