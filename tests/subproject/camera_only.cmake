# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The package's camera-only configure (VS_CAMERA_ONLY), with
# find_package(Vulkan) disabled throughout: run by the
# sensor_subproject_camera_only test (tests/CMakeLists.txt), in BINARY_DIR.
#
# 1. Each option that needs the rest of the package is refused with
#    VS_CAMERA_ONLY.
# 2. The consumer (CMakeLists.txt here) adds the package's source from SOURCE_DIR
#    camera-only, checks that nothing reaches Vulkan and builds against the
#    umbrella, which reaches sensor_camera alone.
#
# GENERATOR and CXX_COMPILER repeat the calling build's; the build is Release.
# Every FETCHCONTENT_SOURCE_DIR_<NAME> given is passed on, so it needs no
# network.
foreach(_var SOURCE_DIR BINARY_DIR GENERATOR CXX_COMPILER)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "camera_only.cmake: ${_var} is not set")
  endif()
endforeach()

set(_fetch "-DVS_SOURCE_DIR=${SOURCE_DIR}" -DVS_CAMERA_ONLY=ON)
get_cmake_property(_vars VARIABLES)
foreach(_var IN LISTS _vars)
  if(_var MATCHES "^FETCHCONTENT_SOURCE_DIR_")
    list(APPEND _fetch "-D${_var}=${${_var}}")
  endif()
endforeach()

# Runs `cmake <ARGN>`, into _result and _output.
function(_vs_cmake)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _output)
  set(_result
      "${_result}"
      PARENT_SCOPE)
  set(_output
      "${_output}"
      PARENT_SCOPE)
endfunction()

# Configures the consumer into BINARY_DIR/<dir>, with Vulkan hidden and ARGN,
# into _result and _output.
macro(_vs_configure dir)
  _vs_cmake(
    -S
    "${CMAKE_CURRENT_LIST_DIR}"
    -B
    "${BINARY_DIR}/${dir}"
    -G
    "${GENERATOR}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=ON
    -DVS_EXPECT_CAMERA_ONLY=ON
    ${ARGN})
endmacro()

# Fails unless the last command succeeded.
function(_vs_succeeded what)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${what} failed:\n${_output}")
  endif()
endfunction()

file(REMOVE_RECURSE "${BINARY_DIR}")

foreach(_option VS_WITH_ORBBEC VS_WITH_FFMPEG VS_WITH_CUDA)
  _vs_configure(refuse_${_option} ${_fetch} "-D${_option}=ON")
  # CMake wraps a message's lines: match it with each run of blanks as one.
  string(REGEX REPLACE "[ \n]+" " " _flat "${_output}")
  set(_refusal
      "VS_CAMERA_ONLY builds the camera tier alone: turn ${_option} OFF")
  if(_result EQUAL 0 OR NOT _flat MATCHES "${_refusal}")
    message(FATAL_ERROR "VS_CAMERA_ONLY with ${_option} ON was not refused "
                        "with '${_refusal}':\n${_output}")
  endif()
endforeach()

_vs_configure(fetched ${_fetch})
_vs_succeeded("Configuring the package's source camera-only")
_vs_cmake(--build "${BINARY_DIR}/fetched" --config Release)
_vs_succeeded("Building against the package's source")
