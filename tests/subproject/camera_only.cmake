# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# recon's camera-only configure (VR_CAMERA_ONLY), with find_package(Vulkan)
# disabled throughout: run by the recon_subproject_camera_only test
# (tests/CMakeLists.txt), in BINARY_DIR.
#
# 1. Each option that needs the rest of recon is refused with VR_CAMERA_ONLY.
# 2. The consumer (CMakeLists.txt here) fetches recon's source from SOURCE_DIR
#    camera-only, checks that nothing reaches Vulkan and builds against the
#    umbrella, which reaches recon_camera alone; recon's install from that build
#    goes to BINARY_DIR/prefix.
# 3. The consumer finds that installed package and builds against it again.
#
# GENERATOR and CXX_COMPILER repeat the calling build's; each build and the
# install are Release, so a multi-config generator builds what it installs.
# Every FETCHCONTENT_SOURCE_DIR_<NAME> given is passed on, so it needs no
# network.
foreach(_var SOURCE_DIR BINARY_DIR GENERATOR CXX_COMPILER)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "camera_only.cmake: ${_var} is not set")
  endif()
endforeach()

set(_fetch "-DVR_SOURCE_DIR=${SOURCE_DIR}" -DVR_CAMERA_ONLY=ON)
get_cmake_property(_vars VARIABLES)
foreach(_var IN LISTS _vars)
  if(_var MATCHES "^FETCHCONTENT_SOURCE_DIR_")
    list(APPEND _fetch "-D${_var}=${${_var}}")
  endif()
endforeach()

# Runs `cmake <ARGN>`, into _result and _output.
function(_vr_cmake)
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
macro(_vr_configure dir)
  _vr_cmake(
    -S
    "${CMAKE_CURRENT_LIST_DIR}"
    -B
    "${BINARY_DIR}/${dir}"
    -G
    "${GENERATOR}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=ON
    -DVR_EXPECT_CAMERA_ONLY=ON
    ${ARGN})
endmacro()

# Fails unless the last command succeeded.
function(_vr_succeeded what)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${what} failed:\n${_output}")
  endif()
endfunction()

file(REMOVE_RECURSE "${BINARY_DIR}")

foreach(_option VR_BUILD_EXAMPLES VR_WITH_ORBBEC VR_WITH_FFMPEG VR_WITH_CUDA
                VR_WITH_ASSIMP)
  _vr_configure(refuse_${_option} ${_fetch} "-D${_option}=ON")
  # CMake wraps a message's lines: match it with each run of blanks as one.
  string(REGEX REPLACE "[ \n]+" " " _flat "${_output}")
  set(_refusal
      "VR_CAMERA_ONLY builds the camera tier alone: turn ${_option} OFF")
  if(_result EQUAL 0 OR NOT _flat MATCHES "${_refusal}")
    message(FATAL_ERROR "VR_CAMERA_ONLY with ${_option} ON was not refused "
                        "with '${_refusal}':\n${_output}")
  endif()
endforeach()

set(_prefix "${BINARY_DIR}/prefix")
_vr_configure(fetched ${_fetch} -DVR_INSTALL=ON
              "-DCMAKE_INSTALL_PREFIX=${_prefix}")
_vr_succeeded("Configuring recon's source camera-only")
_vr_cmake(--build "${BINARY_DIR}/fetched" --config Release)
_vr_succeeded("Building against recon's source")
_vr_cmake(--install "${BINARY_DIR}/fetched" --config Release)
_vr_succeeded("Installing recon")

_vr_configure(installed "-DCMAKE_PREFIX_PATH=${_prefix}")
_vr_succeeded("Configuring against the installed recon")
_vr_cmake(--build "${BINARY_DIR}/installed" --config Release)
_vr_succeeded("Building against the installed recon")
