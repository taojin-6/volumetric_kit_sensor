# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# Every test is a main() of its own on the core's test support: CHECK,
# VKC_TEST_SANITIZED and the scratch directory (core_test_base), in camera-only
# builds too. A GPU test also links core_test_policy, and one that fails or
# measures allocations core_test_allocation_failure.
link_libraries(volumetric_kit::core_test_base)

# The package boundary (package_boundary.cmake): the package is on the core
# alone, so no file under sensor/ includes a header or GLSL file of recon or of
# grid, and it builds where neither is. A scan of the sources, so camera-only
# builds run it too.
add_test(NAME sensor_package_boundary
         COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" -P
                 "${CMAKE_CURRENT_SOURCE_DIR}/package_boundary.cmake")

# camera tier: the camera model (its checks, the lens template against OpenCV's
# numbers), its geometry (rigid checks, Rodrigues both ways), and the sensor
# array's calibration file (versions 2 and 1 read, the refusals, written and
# read back, its document parsed independently with nlohmann/json). Host-only,
# so they always run.
add_executable(sensor_camera_model_test camera_model_test.cpp)
target_link_libraries(sensor_camera_model_test
                      PRIVATE volumetric_kit::sensor_camera)
vs_target_warnings(sensor_camera_model_test)
add_test(NAME sensor_camera_model COMMAND sensor_camera_model_test)

add_executable(sensor_camera_geometry_test camera_geometry_test.cpp)
target_link_libraries(sensor_camera_geometry_test
                      PRIVATE volumetric_kit::sensor_camera)
vs_target_warnings(sensor_camera_geometry_test)
add_test(NAME sensor_camera_geometry COMMAND sensor_camera_geometry_test)

add_executable(sensor_camera_array_calibration_test
               camera_array_calibration_test.cpp)
target_link_libraries(sensor_camera_array_calibration_test
                      PRIVATE volumetric_kit::sensor_camera)
target_include_directories(sensor_camera_array_calibration_test SYSTEM
                           PRIVATE "${VS_JSON_INCLUDE_DIR}")
vs_target_warnings(sensor_camera_array_calibration_test)
add_test(NAME sensor_camera_array_calibration
         COMMAND sensor_camera_array_calibration_test)

# The frame prep's same-K depth crop against a folding lens: a forward scan of
# distort_rational that finds a strong barrel lens's fold, run on a stand-in for
# a Femto Mega's NFOV depth lens (not an SDK report). It needs sensor_camera
# alone, so it runs in camera-only builds too.
add_executable(sensor_depth_lens_fold_test sensor_depth_lens_fold_test.cpp)
target_link_libraries(sensor_depth_lens_fold_test
                      PRIVATE volumetric_kit::sensor_camera)
vs_target_warnings(sensor_depth_lens_fold_test)
add_test(NAME sensor_depth_lens_fold COMMAND sensor_depth_lens_fold_test)

# A project that adds the package (subproject/) takes each dependency this build
# fetched as a local source, so it needs no network.
file(
  GLOB _vs_fetched
  LIST_DIRECTORIES true
  "${FETCHCONTENT_BASE_DIR}/*-src")
set(_vs_fetched_sources)
foreach(_vs_source IN LISTS _vs_fetched)
  get_filename_component(_vs_name "${_vs_source}" NAME)
  string(REGEX REPLACE "-src$" "" _vs_name "${_vs_name}")
  string(TOUPPER "${_vs_name}" _vs_name)
  # This package itself, where a project added it, is the source under test.
  if(NOT _vs_name STREQUAL "VOLUMETRIC_KIT_SENSOR")
    list(APPEND _vs_fetched_sources
         "-DFETCHCONTENT_SOURCE_DIR_${_vs_name}=${_vs_source}")
  endif()
endforeach()
# A dependency this build took from a local source instead, as given.
get_cmake_property(_vs_cache CACHE_VARIABLES)
foreach(_vs_var IN LISTS _vs_cache)
  if(_vs_var MATCHES "^FETCHCONTENT_SOURCE_DIR_"
     AND NOT _vs_var STREQUAL "FETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_SENSOR")
    if(NOT "${${_vs_var}}" STREQUAL "")
      list(APPEND _vs_fetched_sources "-D${_vs_var}=${${_vs_var}}")
    endif()
  endif()
endforeach()

# The camera tier is all VS_CAMERA_ONLY builds. Only they check the package
# camera-only (subproject/camera_only.cmake): with find_package(Vulkan)
# disabled, the options that need the rest are refused, and a consumer builds
# against the fetched source. The camera-only CI job, which has no Vulkan at
# all, is its home; it is Release whatever the build's type.
if(VS_CAMERA_ONLY)
  add_test(
    NAME sensor_subproject_camera_only
    COMMAND
      "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
      "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/subproject_camera_only"
      "-DGENERATOR=${CMAKE_GENERATOR}" "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
      ${_vs_fetched_sources} -P
      "${CMAKE_CURRENT_SOURCE_DIR}/subproject/camera_only.cmake")
  set_tests_properties(sensor_package_boundary sensor_subproject_camera_only
                       PROPERTIES TIMEOUT 300)
  return()
endif()

# The package's own pieces: the version CMake wires through, the GLM aliases,
# and what it asks of a device -- scalar block layout, and the decoders'
# extensions as optional, which recon's list leaves to this one.
add_executable(sensor_smoke_test smoke_test.cpp)
target_link_libraries(sensor_smoke_test PRIVATE volumetric_kit::sensor_camera
                                                volumetric_kit::sensor_frame)
vs_target_warnings(sensor_smoke_test)
add_test(NAME sensor_smoke COMMAND sensor_smoke_test)

# The decoder tests' skips (decoder_test.hpp), as VS_TEST_HEVC_BACKEND says. The
# GPU tests run on the core's harness (core/testing/gpu_test.hpp), which the
# core tests.
add_executable(sensor_decoder_test_harness_test decoder_test_harness_test.cpp)
target_link_libraries(
  sensor_decoder_test_harness_test PRIVATE volumetric_kit::sensor_frame
                                           volumetric_kit::core_test_policy)
vs_target_warnings(sensor_decoder_test_harness_test)
add_test(NAME sensor_decoder_test_harness
         COMMAND sensor_decoder_test_harness_test)

# GPU pre-processing (sensor_prep) on synthetic frames: undistortion in the
# right direction, the colour matrix, a host reference of the sampling, the
# refusals, the outputs' sharing, read on another queue family. Fusing its
# output is recon's (recon_tsdf_prepared_frame). Skips where no device is
# present.
add_executable(sensor_gpu_frame_prep_test sensor_gpu_frame_prep_test.cpp)
target_link_libraries(
  sensor_gpu_frame_prep_test PRIVATE volumetric_kit::sensor_prep
                                     volumetric_kit::core_test_policy)
vs_target_warnings(sensor_gpu_frame_prep_test)
add_test(NAME sensor_gpu_frame_prep COMMAND sensor_gpu_frame_prep_test)

# Allocation failure while staging is an OutOfMemory result, and a retained
# frame survives it. Separate executable for its replacement operator new.
add_executable(sensor_gpu_frame_prep_oom_test
               sensor_gpu_frame_prep_oom_test.cpp)
target_link_libraries(
  sensor_gpu_frame_prep_oom_test
  PRIVATE volumetric_kit::core_test_policy volumetric_kit::sensor_prep
          volumetric_kit::core_test_allocation_failure)
vs_target_warnings(sensor_gpu_frame_prep_oom_test)
add_test(NAME sensor_gpu_frame_prep_oom COMMAND sensor_gpu_frame_prep_oom_test)

# Submission-loss lifetime fault injection needs the Unix dynamic loader.
if(UNIX)
  add_executable(sensor_gpu_frame_prep_submit_test
                 sensor_gpu_frame_prep_submit_test.cpp)
  target_link_libraries(
    sensor_gpu_frame_prep_submit_test
    PRIVATE volumetric_kit::core_test_policy volumetric_kit::sensor_prep
            ${CMAKE_DL_LIBS})
  vs_target_warnings(sensor_gpu_frame_prep_submit_test)
  add_test(NAME sensor_gpu_frame_prep_submit
           COMMAND sensor_gpu_frame_prep_submit_test)
endif()

# A set's pictures read back to the host (sensor_prep's PictureReadback) on
# synthetic NV12 images and NV12 and I420 buffers: exact bytes at odd sizes,
# offsets and strides, every queue family a buffer comes from, the host memory
# grown and reused, the refusals and the stage row. Skips with no device.
add_executable(sensor_picture_readback_test sensor_picture_readback_test.cpp)
target_link_libraries(
  sensor_picture_readback_test PRIVATE volumetric_kit::sensor_prep
                                       volumetric_kit::core_test_policy)
vs_target_warnings(sensor_picture_readback_test)
add_test(NAME sensor_picture_readback COMMAND sensor_picture_readback_test)

# Allocation failure during creation (including the timer) or a read is an
# OutOfMemory result, and retry succeeds. Separate allocation overrides.
add_executable(sensor_picture_readback_oom_test
               sensor_picture_readback_oom_test.cpp)
target_link_libraries(
  sensor_picture_readback_oom_test
  PRIVATE volumetric_kit::sensor_prep volumetric_kit::core_test_policy
          volumetric_kit::core_test_allocation_failure)
vs_target_warnings(sensor_picture_readback_oom_test)
add_test(NAME sensor_picture_readback_oom
         COMMAND sensor_picture_readback_oom_test)

# Color-space vocabulary: the exact piecewise sRGB curve, the sensor boundary's
# packed helpers, and the primaries matrices. Pure host math over core, so it
# always runs -- no device, no platform. recon's tsdf holds the voxel colour
# format to this curve (recon_tsdf_voxel_color_space).
add_executable(sensor_frame_color_space_test frame_color_space_test.cpp)
target_link_libraries(sensor_frame_color_space_test
                      PRIVATE volumetric_kit::sensor_frame)
vs_target_warnings(sensor_frame_color_space_test)
add_test(NAME sensor_frame_color_space COMMAND sensor_frame_color_space_test)

# frame/shaders/pinhole.glsl, the GLSL pinhole the package exports, built
# through INCLUDE_TARGETS volumetric_kit::sensor_frame as a consumer's kernels
# are, against the host's arithmetic in double (shaders/pinhole_cases.comp,
# std430 as a consumer's kernel): written-out edges in exact arithmetic, and a
# grid around five cameras. Skips where no device is present.
add_executable(sensor_frame_pinhole_gpu_test frame_pinhole_gpu_test.cpp)
include(vs_embed)
vs_embed_shaders(
  sensor_frame_pinhole_gpu_test INCLUDE_TARGETS volumetric_kit::sensor_frame
  SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pinhole_cases.comp")
target_link_libraries(
  sensor_frame_pinhole_gpu_test
  PRIVATE volumetric_kit::sensor_frame volumetric_kit::sensor_camera
          volumetric_kit::core_test_policy)
vs_target_warnings(sensor_frame_pinhole_gpu_test)
add_test(NAME sensor_frame_pinhole_gpu COMMAND sensor_frame_pinhole_gpu_test)

# The sensor interface: the camera-convention conversions -- the OpenGL/ARKit ->
# CV pose change of basis and the registered-depth intrinsics rescale. These are
# the two things a capture integration gets *silently* wrong (a flipped axis or
# a half-pixel bias misaligns the reconstruction without ever erroring), and
# they are pure host math, so they always run rather than being gated on a
# device or on the platform whose driver produced the numbers.
add_executable(sensor_conventions_test sensor_conventions_test.cpp)
target_link_libraries(sensor_conventions_test
                      PRIVATE volumetric_kit::sensor_interface)
vs_target_warnings(sensor_conventions_test)
add_test(NAME sensor_conventions COMMAND sensor_conventions_test)

# The sensor array's trigger grouper, driven by synthetic streams -- drops, a
# silent sensor, a slow poll, a clock off by more than the tolerance, a late
# frame, the start-up frames. Host-only. Internal and unexported, so the test
# compiles its source itself, as the Orbbec tests do.
set(_vs_array_src ${PROJECT_SOURCE_DIR}/src/volumetric_kit/sensor/array)
add_executable(sensor_trigger_grouper_test sensor_trigger_grouper_test.cpp
                                           ${_vs_array_src}/trigger_grouper.cpp)
target_link_libraries(sensor_trigger_grouper_test
                      PRIVATE volumetric_kit::sensor_interface)
target_include_directories(sensor_trigger_grouper_test PRIVATE ${_vs_array_src})
vs_target_warnings(sensor_trigger_grouper_test)
add_test(NAME sensor_trigger_grouper COMMAND sensor_trigger_grouper_test)

# The sensor array over scripted fake sensors -- the arrays it refuses, its
# start order, trigger grouping, poses from the calibration. Host-only.
add_executable(sensor_array_test sensor_array_test.cpp)
target_link_libraries(sensor_array_test PRIVATE volumetric_kit::sensor_array)
vs_target_warnings(sensor_array_test)
add_test(NAME sensor_array COMMAND sensor_array_test)

# A run's sets over pushed fake sensors -- each waited for, counted and cut to
# the run's length, its pictures let go, each secondary's skew kept, the silence
# limit from the last set and a lone sync secondary named, and the array's
# refusal passed on. Host-only.
add_executable(sensor_array_rig_sets_test sensor_array_rig_sets_test.cpp)
target_link_libraries(sensor_array_rig_sets_test
                      PRIVATE volumetric_kit::sensor_array)
vs_target_warnings(sensor_array_rig_sets_test)
add_test(NAME sensor_array_rig_sets COMMAND sensor_array_rig_sets_test)

# A sensor array's sets prepared on the GPU in one batch, each frame as
# GpuFramePrep makes it alone and posed by the calibration. Skips without a
# device.
add_executable(sensor_array_process_test sensor_array_process_test.cpp)
target_link_libraries(
  sensor_array_process_test PRIVATE volumetric_kit::sensor_array
                                    volumetric_kit::core_test_policy)
vs_target_warnings(sensor_array_process_test)
add_test(NAME sensor_array_process COMMAND sensor_array_process_test)

# RGB-D sensor defaults, frame-return helpers, copied-frame ownership and stable
# enum names. Host-only.
add_executable(sensor_rgbd_sensor_test sensor_rgbd_sensor_test.cpp)
target_link_libraries(sensor_rgbd_sensor_test
                      PRIVATE volumetric_kit::sensor_interface)
vs_target_warnings(sensor_rgbd_sensor_test)
add_test(NAME sensor_rgbd_sensor COMMAND sensor_rgbd_sensor_test)

# Apple's pixel buffers imported as images (sensor_apple, any Apple build, with
# or without FFmpeg): buffers made here, read back in place, prepared as a
# frame's colour, what the import allocates in Metal (metal_allocation.mm), and
# the refusals. Skips without a device.
if(APPLE)
  add_executable(
    sensor_apple_pixel_buffer_importer_test
    sensor_apple_pixel_buffer_importer_test.cpp metal_allocation.mm)
  set_source_files_properties(metal_allocation.mm PROPERTIES COMPILE_OPTIONS
                                                             -fobjc-arc)
  target_link_libraries(
    sensor_apple_pixel_buffer_importer_test
    PRIVATE volumetric_kit::core_test_policy
            volumetric_kit::sensor_apple
            volumetric_kit::sensor_prep
            "-framework CoreFoundation"
            "-framework CoreVideo"
            "-framework IOSurface"
            "-framework Metal"
            "-framework Foundation")
  vs_target_warnings(sensor_apple_pixel_buffer_importer_test)
  add_test(NAME sensor_apple_pixel_buffer_importer
           COMMAND sensor_apple_pixel_buffer_importer_test)
endif()

# The video decoders' host pieces, with no codec and no GPU, so on every leg:
# the colour H.273 codes describe, compiling the internal source itself, as the
# Orbbec conversion test does; and the public H.265 syntax (sensor_video_syntax)
# on the committed clips.
set(_vs_video_src ${PROJECT_SOURCE_DIR}/src/volumetric_kit/sensor/video)
add_executable(sensor_video_frame_color_test sensor_video_frame_color_test.cpp
                                             ${_vs_video_src}/frame_color.cpp)
target_link_libraries(sensor_video_frame_color_test
                      PRIVATE volumetric_kit::sensor_interface)
target_include_directories(sensor_video_frame_color_test
                           PRIVATE ${_vs_video_src})
vs_target_warnings(sensor_video_frame_color_test)
add_test(NAME sensor_video_frame_color COMMAND sensor_video_frame_color_test)

add_executable(sensor_video_hevc_syntax_test sensor_video_hevc_syntax_test.cpp)
target_link_libraries(
  sensor_video_hevc_syntax_test
  PRIVATE volumetric_kit::sensor_video_syntax
          volumetric_kit::core_test_allocation_failure)
target_compile_definitions(
  sensor_video_hevc_syntax_test
  PRIVATE VS_HEVC_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/hevc")
vs_target_warnings(sensor_video_hevc_syntax_test)
add_test(NAME sensor_video_hevc_syntax COMMAND sensor_video_hevc_syntax_test)

# The video decoders (on Apple, and elsewhere with VS_WITH_FFMPEG). The FFmpeg
# helper test maps error codes and a frame's colour fields, compiling ffmpeg.cpp
# itself. The decoder tests decode the committed clips and JPEGs on this
# machine's hardware, onto the device; on a GPU with no device path they skip,
# unless VS_TEST_HEVC_BACKEND says this leg has one, where they fail.
if(APPLE OR VS_WITH_FFMPEG)
  if(VS_WITH_FFMPEG)
    add_executable(sensor_video_ffmpeg_test sensor_video_ffmpeg_test.cpp
                                            ${_vs_video_src}/ffmpeg.cpp)
    target_link_libraries(
      sensor_video_ffmpeg_test PRIVATE volumetric_kit::sensor_interface
                                       PkgConfig::VS_FFMPEG)
    target_include_directories(sensor_video_ffmpeg_test
                               PRIVATE ${_vs_video_src})
    vs_target_warnings(sensor_video_ffmpeg_test)
    add_test(NAME sensor_video_ffmpeg COMMAND sensor_video_ffmpeg_test)
  endif()

  add_executable(sensor_video_hevc_test sensor_video_hevc_test.cpp)
  target_link_libraries(
    sensor_video_hevc_test
    PRIVATE volumetric_kit::sensor_video volumetric_kit::sensor_prep
            volumetric_kit::core_test_policy)
  target_compile_definitions(
    sensor_video_hevc_test
    PRIVATE VS_HEVC_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/hevc")
  if(VS_WITH_CUDA)
    # It asks libcuda which primary contexts the decoders hold.
    target_include_directories(sensor_video_hevc_test SYSTEM
                               PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(sensor_video_hevc_test PRIVATE ${CMAKE_DL_LIBS})
  endif()
  vs_target_warnings(sensor_video_hevc_test)
  add_test(NAME sensor_video_hevc COMMAND sensor_video_hevc_test)

  add_executable(sensor_video_jpeg_test sensor_video_jpeg_test.cpp)
  target_link_libraries(
    sensor_video_jpeg_test
    PRIVATE volumetric_kit::sensor_video volumetric_kit::sensor_prep
            volumetric_kit::core_test_policy)
  target_compile_definitions(
    sensor_video_jpeg_test
    PRIVATE VS_JPEG_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/jpeg")
  vs_target_warnings(sensor_video_jpeg_test)
  add_test(NAME sensor_video_jpeg COMMAND sensor_video_jpeg_test)

  if(VS_WITH_CUDA)
    # The public CUDA interop on this machine's GPU, which skips as the decoder
    # tests do; it calls libcuda through a loader of its own.
    add_executable(sensor_video_cuda_interop_test
                   sensor_video_cuda_interop_test.cpp)
    target_link_libraries(
      sensor_video_cuda_interop_test
      PRIVATE volumetric_kit::sensor_video volumetric_kit::core_test_policy
              ${CMAKE_DL_LIBS})
    vs_target_warnings(sensor_video_cuda_interop_test)
    add_test(NAME sensor_video_cuda_interop
             COMMAND sensor_video_cuda_interop_test)

    # The Status of a failed driver or nvJPEG call. It compiles the driver
    # source itself, which loads libcuda for the error's name and opens no
    # device.
    add_executable(
      sensor_video_cuda_error_test sensor_video_cuda_error_test.cpp
                                   ${_vs_video_src}/cuda_driver.cpp)
    target_link_libraries(
      sensor_video_cuda_error_test
      PRIVATE volumetric_kit::sensor_frame volumetric_kit::core_test_policy
              ${CMAKE_DL_LIBS})
    target_include_directories(sensor_video_cuda_error_test
                               PRIVATE ${_vs_video_src})
    target_include_directories(sensor_video_cuda_error_test SYSTEM
                               PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    vs_target_warnings(sensor_video_cuda_error_test)
    add_test(NAME sensor_video_cuda_error COMMAND sensor_video_cuda_error_test)
  endif()

  if(APPLE)
    # Compile the public JPEG decoder here so the executable's native-call
    # replacements intercept it in both static and shared library builds.
    add_executable(
      sensor_video_vt_jpeg_error_test
      sensor_video_vt_jpeg_error_test.cpp ${_vs_video_src}/jpeg_decoder.cpp
      ${_vs_video_src}/vt_jpeg.cpp ${_vs_video_src}/videotoolbox.cpp)
    target_link_libraries(
      sensor_video_vt_jpeg_error_test
      PRIVATE volumetric_kit::sensor_video volumetric_kit::sensor_apple
              "-framework VideoToolbox" "-framework CoreMedia"
              "-framework CoreVideo" volumetric_kit::core_test_policy)
    target_compile_definitions(
      sensor_video_vt_jpeg_error_test
      PRIVATE VS_JPEG_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/jpeg")
    vs_target_warnings(sensor_video_vt_jpeg_error_test)
    add_test(NAME sensor_video_vt_jpeg_error
             COMMAND sensor_video_vt_jpeg_error_test)
  endif()
endif()

# The worker each Orbbec colour decoder decodes on (decode_worker.hpp), driven
# by a fake decode. It names no SDK type, so it is tested without the driver
# too, on every leg, the sanitizers' included; with exceptions, as the driver is
# compiled, since it catches what a decode throws.
set(_vs_orbbec_src ${PROJECT_SOURCE_DIR}/src/volumetric_kit/sensor/orbbec)
add_executable(sensor_orbbec_decode_worker_test
               sensor_orbbec_decode_worker_test.cpp)
target_link_libraries(sensor_orbbec_decode_worker_test
                      PRIVATE volumetric_kit::core_base)
target_include_directories(sensor_orbbec_decode_worker_test
                           PRIVATE ${_vs_orbbec_src})
target_compile_options(sensor_orbbec_decode_worker_test PRIVATE -fexceptions)
vs_target_warnings(sensor_orbbec_decode_worker_test)
add_test(NAME sensor_orbbec_decode_worker
         COMMAND sensor_orbbec_decode_worker_test)

# The Orbbec SDK prerequisite (VS_WITH_ORBBEC only): the copy the build found is
# the one that loads -- runtime version == configured version, since the SDK is
# linked in place from an install outside the repo. It opens no SDK context.
if(VS_WITH_ORBBEC)
  add_executable(sensor_orbbec_sdk_smoke_test orbbec_sdk_smoke_test.cpp)
  vs_link_orbbec_sdk(sensor_orbbec_sdk_smoke_test)
  target_compile_definitions(
    sensor_orbbec_sdk_smoke_test
    PRIVATE VS_ORBBEC_SDK_VERSION_MAJOR=${VS_ORBBEC_SDK_VERSION_MAJOR}
            VS_ORBBEC_SDK_VERSION_MINOR=${VS_ORBBEC_SDK_VERSION_MINOR}
            VS_ORBBEC_SDK_VERSION_PATCH=${VS_ORBBEC_SDK_VERSION_PATCH})
  vs_target_warnings(sensor_orbbec_sdk_smoke_test)
  add_test(NAME sensor_orbbec_sdk_smoke COMMAND sensor_orbbec_sdk_smoke_test)

  # The Orbbec driver's arithmetic, with no camera. frame_conversion.hpp is
  # internal and names SDK types, so the test reaches it by include path; its
  # functions are not exported, so the test compiles their source itself rather
  # than linking a shared build's hidden copies.
  add_executable(
    sensor_orbbec_conversion_test sensor_orbbec_conversion_test.cpp
                                  ${_vs_orbbec_src}/frame_conversion.cpp)
  target_link_libraries(sensor_orbbec_conversion_test
                        PRIVATE volumetric_kit::sensor_orbbec)
  vs_link_orbbec_sdk(sensor_orbbec_conversion_test)
  target_include_directories(sensor_orbbec_conversion_test
                             PRIVATE ${_vs_orbbec_src})
  vs_target_warnings(sensor_orbbec_conversion_test)
  add_test(NAME sensor_orbbec_conversion COMMAND sensor_orbbec_conversion_test)

  # The rig's sync configuration: the SDK's file layout, its refusals, and the
  # comparison OrbbecSensor::open makes against what the camera reads back. No
  # camera; compiles frame_conversion.cpp itself, as the conversion test does.
  add_executable(
    sensor_orbbec_sync_config_test sensor_orbbec_sync_config_test.cpp
                                   ${_vs_orbbec_src}/frame_conversion.cpp)
  target_link_libraries(sensor_orbbec_sync_config_test
                        PRIVATE volumetric_kit::sensor_orbbec)
  vs_link_orbbec_sdk(sensor_orbbec_sync_config_test)
  target_include_directories(sensor_orbbec_sync_config_test
                             PRIVATE ${_vs_orbbec_src})
  vs_target_warnings(sensor_orbbec_sync_config_test)
  add_test(NAME sensor_orbbec_sync_config
           COMMAND sensor_orbbec_sync_config_test)

  # Public open() against a camera stub (orbbec_camera_stub.cpp): SDK
  # normalization after a sync write must be refused. Compile the sensor here so
  # the stub also works in shared builds; no real camera or SDK context is
  # opened.
  add_executable(
    sensor_orbbec_sync_apply_test
    sensor_orbbec_sync_apply_test.cpp orbbec_camera_stub.cpp
    ${_vs_orbbec_src}/orbbec_sensor.cpp ${_vs_orbbec_src}/frame_conversion.cpp)
  target_link_libraries(sensor_orbbec_sync_apply_test
                        PRIVATE volumetric_kit::sensor_orbbec)
  vs_link_orbbec_sdk(sensor_orbbec_sync_apply_test)
  target_include_directories(sensor_orbbec_sync_apply_test
                             PRIVATE ${_vs_orbbec_src})
  vs_target_warnings(sensor_orbbec_sync_apply_test)
  add_test(NAME sensor_orbbec_sync_apply COMMAND sensor_orbbec_sync_apply_test)

  # The colour controls open() writes and reads back, against the same camera
  # stub, whose properties go through the driver's own sequence
  # (frame_conversion.cpp's write_color_controls).
  add_executable(
    sensor_orbbec_color_controls_test
    sensor_orbbec_color_controls_test.cpp orbbec_camera_stub.cpp
    ${_vs_orbbec_src}/orbbec_sensor.cpp ${_vs_orbbec_src}/frame_conversion.cpp)
  target_link_libraries(sensor_orbbec_color_controls_test
                        PRIVATE volumetric_kit::sensor_orbbec)
  vs_link_orbbec_sdk(sensor_orbbec_color_controls_test)
  target_include_directories(sensor_orbbec_color_controls_test
                             PRIVATE ${_vs_orbbec_src})
  vs_target_warnings(sensor_orbbec_color_controls_test)
  add_test(NAME sensor_orbbec_color_controls
           COMMAND sensor_orbbec_color_controls_test)

  # A rig opened from its sync file (open_orbbec_sensors) against the same
  # camera stub, with the fixture files in data/orbbec_sync.
  add_executable(
    sensor_orbbec_open_sensors_test
    sensor_orbbec_open_sensors_test.cpp orbbec_camera_stub.cpp
    ${_vs_orbbec_src}/orbbec_sensor.cpp ${_vs_orbbec_src}/frame_conversion.cpp)
  target_link_libraries(sensor_orbbec_open_sensors_test
                        PRIVATE volumetric_kit::sensor_orbbec)
  vs_link_orbbec_sdk(sensor_orbbec_open_sensors_test)
  target_include_directories(sensor_orbbec_open_sensors_test
                             PRIVATE ${_vs_orbbec_src})
  target_compile_definitions(
    sensor_orbbec_open_sensors_test
    PRIVATE VS_SYNC_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/data/orbbec_sync")
  vs_target_warnings(sensor_orbbec_open_sensors_test)
  add_test(NAME sensor_orbbec_open_sensors
           COMMAND sensor_orbbec_open_sensors_test)

  # The driver's H.265 colour decoder, with no camera: the committed patch clip
  # wrapped in SDK frames. Internal and unexported, so the test compiles its
  # source itself.
  add_executable(
    sensor_orbbec_hevc_test
    sensor_orbbec_hevc_test.cpp ${_vs_orbbec_src}/hevc_color.cpp
    ${_vs_orbbec_src}/picture_frames.cpp)
  target_link_libraries(
    sensor_orbbec_hevc_test
    PRIVATE volumetric_kit::sensor_orbbec volumetric_kit::sensor_video
            volumetric_kit::sensor_prep volumetric_kit::core_test_policy)
  vs_link_orbbec_sdk(sensor_orbbec_hevc_test)
  target_include_directories(sensor_orbbec_hevc_test PRIVATE ${_vs_orbbec_src})
  target_compile_definitions(
    sensor_orbbec_hevc_test
    PRIVATE VS_HEVC_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/hevc")
  vs_target_warnings(sensor_orbbec_hevc_test)
  add_test(NAME sensor_orbbec_hevc COMMAND sensor_orbbec_hevc_test)

  # The MJPEG colour decoder, with no camera: the committed JPEG wrapped in SDK
  # frames.
  add_executable(
    sensor_orbbec_jpeg_test
    sensor_orbbec_jpeg_test.cpp ${_vs_orbbec_src}/jpeg_color.cpp
    ${_vs_orbbec_src}/picture_frames.cpp)
  target_link_libraries(
    sensor_orbbec_jpeg_test
    PRIVATE volumetric_kit::sensor_orbbec volumetric_kit::sensor_video
            volumetric_kit::sensor_prep volumetric_kit::core_test_policy)
  vs_link_orbbec_sdk(sensor_orbbec_jpeg_test)
  target_include_directories(sensor_orbbec_jpeg_test PRIVATE ${_vs_orbbec_src})
  target_compile_definitions(
    sensor_orbbec_jpeg_test
    PRIVATE VS_JPEG_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data/jpeg")
  vs_target_warnings(sensor_orbbec_jpeg_test)
  add_test(NAME sensor_orbbec_jpeg COMMAND sensor_orbbec_jpeg_test)
endif()

# Host image I/O, testable without creating a Vulkan device, on the family
# core's base tier alone, with no Vulkan (subproject/ checks its links).
add_executable(sensor_io_image_test io_image_test.cpp)
target_link_libraries(
  sensor_io_image_test PRIVATE volumetric_kit::sensor_image_io
                               volumetric_kit::core_test_allocation_failure)
vs_target_warnings(sensor_io_image_test)
add_test(NAME sensor_io_image COMMAND sensor_io_image_test)

# The tests whose code runs threads -- the Orbbec decode workers -- which CI
# builds and runs again under ThreadSanitizer (.github/workflows/_build.yml):
# one target to build them.
add_custom_target(sensor_tsan_tests)
add_dependencies(sensor_tsan_tests sensor_orbbec_decode_worker_test)
if(VS_WITH_ORBBEC)
  add_dependencies(sensor_tsan_tests sensor_orbbec_jpeg_test
                   sensor_orbbec_hevc_test sensor_orbbec_open_sensors_test)
endif()

# What a project that adds the package gets (subproject/): its configure, from a
# clean tree, with each dependency this build fetched as a local source, so it
# needs no network. The defaults hold, image I/O links the core's base tier
# alone, the project's vkc_require_sibling accepts this package and refuses it
# under a floor above its version, its kernel builds on the package's exported
# GLSL, and the package refuses a core the project made available first that is
# older than VS_CORE_MIN_VERSION or lacks the vulkan tier. The old core is this
# one stamped 0.0.1, standing in for one below a later floor.
#
# TODO(sensor): this harness -- the fetched sources above, the subproject test
# and message helpers and subproject/'s version stamp -- copies recon's, which
# grid, gfx and calib repeat too; move it into the core's test support with
# theirs, as one function each sibling calls with its package and floor (the
# task recon's tests/CMakeLists.txt names).
set(_vs_subproject "${CMAKE_CURRENT_BINARY_DIR}/subproject")
add_test(NAME sensor_subproject_clean COMMAND "${CMAKE_COMMAND}" -E rm -rf
                                              "${_vs_subproject}")
set_tests_properties(sensor_subproject_clean PROPERTIES FIXTURES_SETUP
                                                        vs_subproject)
# Configures subproject/ into <dir> under _vs_subproject with the given -D args.
function(_vs_add_subproject_test name dir)
  add_test(
    NAME ${name}
    COMMAND
      "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_SOURCE_DIR}/subproject" -B
      "${_vs_subproject}/${dir}" -G "${CMAKE_GENERATOR}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DVS_SOURCE_DIR=${PROJECT_SOURCE_DIR}" ${_vs_fetched_sources} ${ARGN})
  set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED vs_subproject)
endfunction()
# Sets <test>'s PASS_REGULAR_EXPRESSION to <message>, a space in which matches
# however CMake wrapped the error's lines.
function(_vs_expect_message test message)
  string(REPLACE " " "[ \n]+" message "${message}")
  set_tests_properties(${test} PROPERTIES PASS_REGULAR_EXPRESSION "${message}")
endfunction()

_vs_add_subproject_test(sensor_subproject_defaults defaults
                        "-DVS_REQUIRE_SENSOR=${PROJECT_VERSION}")
set_tests_properties(sensor_subproject_defaults
                     PROPERTIES FIXTURES_SETUP vs_subproject_defaults)
math(EXPR _vs_next_minor "${PROJECT_VERSION_MINOR} + 1")
set(_vs_newer "${PROJECT_VERSION_MAJOR}.${_vs_next_minor}.0")
_vs_add_subproject_test(sensor_subproject_old_sensor old_sensor
                        "-DVS_REQUIRE_SENSOR=${_vs_newer}")
_vs_expect_message(
  sensor_subproject_old_sensor
  "vs_subproject_consumer needs volumetric_kit_sensor ${_vs_newer} or newer, but the sensor at .+ is ${PROJECT_VERSION}: pin v${_vs_newer} or newer"
)
# The project compiles a kernel that includes the package's exported GLSL
# through the root sensor_frame names, which holds exactly what it exports.
add_test(NAME sensor_subproject_glsl
         COMMAND "${CMAKE_COMMAND}" --build "${_vs_subproject}/defaults"
                 --target vs_glsl_consumer)
set_tests_properties(sensor_subproject_glsl PROPERTIES FIXTURES_REQUIRED
                                                       vs_subproject_defaults)
_vs_add_subproject_test(sensor_subproject_old_core old_core -DVS_CORE_FIRST=ON
                        -DVKC_WITH_VULKAN=ON -DVS_CONSUMER_CORE_VERSION=0.0.1)
_vs_expect_message(
  sensor_subproject_old_core
  "volumetric_kit_sensor needs volumetric_kit_core ${VS_CORE_MIN_VERSION} or newer, but the core at .+ is 0.0.1: pin v${VS_CORE_MIN_VERSION} or newer"
)
_vs_add_subproject_test(sensor_subproject_core_without_vulkan no_vulkan
                        -DVS_CORE_FIRST=ON -DVKC_WITH_VULKAN=OFF)
_vs_expect_message(
  sensor_subproject_core_without_vulkan
  "volumetric_kit_sensor needs volumetric_kit_core's vulkan tier, which the core at .+ was built without: set VKC_WITH_VULKAN ON before"
)
# On Apple the decoders use VideoToolbox directly, so VS_WITH_FFMPEG is refused
# with the fix (cmake/vs_ffmpeg.cmake).
if(APPLE)
  _vs_add_subproject_test(sensor_subproject_apple_ffmpeg apple_ffmpeg
                          -DVS_WITH_FFMPEG=ON)
  _vs_expect_message(
    sensor_subproject_apple_ffmpeg
    "on Apple the video decoders use VideoToolbox directly and need no FFmpeg. Configure this build with -DVS_WITH_FFMPEG=OFF."
  )
endif()

# Start the slowest test first under `ctest --parallel`: without a COST, CTest
# runs tests in the order they were added. Waits out RigSets' 10-s silence
# limit, under any build.
set_tests_properties(sensor_array_rig_sets PROPERTIES COST 11)

# A test skips by exiting with the core's test::kSkipExitCode (gpu_test.hpp). A
# hang (a fence never signalled, a deadlocked thread) fails in 300 s rather than
# holding a shared runner to its job's limit.
get_property(
  _vs_tests
  DIRECTORY
  PROPERTY TESTS)
set_tests_properties(${_vs_tests} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
