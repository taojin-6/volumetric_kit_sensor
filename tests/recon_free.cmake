# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The sensor package is on the core alone: no file under it includes recon
# ("volumetric_kit/recon/..."), so it builds and tests where recon is not, and
# leaves recon's repository whole. Run as `cmake -DSOURCE_DIR=<package root> -P
# recon_free.cmake`; fails naming every such include.
foreach(_vs_dir include src tests)
  if(NOT IS_DIRECTORY "${SOURCE_DIR}/${_vs_dir}")
    message(FATAL_ERROR "no ${SOURCE_DIR}/${_vs_dir}: the check reads nothing")
  endif()
endforeach()
file(GLOB_RECURSE _vs_all "${SOURCE_DIR}/*")
set(_vs_found "")
set(_vs_count 0)
foreach(_vs_file IN LISTS _vs_all)
  file(RELATIVE_PATH _vs_rel "${SOURCE_DIR}" "${_vs_file}")
  # The fixtures are binary; the build trees a configure may leave are not the
  # package's.
  if(_vs_rel MATCHES "^tests/data/|^build[^/]*/|^\\.git/")
    continue()
  endif()
  math(EXPR _vs_count "${_vs_count} + 1")
  file(STRINGS "${_vs_file}" _vs_lines
       REGEX "#[ \t]*include[ \t]*[\"<]volumetric_kit/recon/")
  foreach(_vs_line IN LISTS _vs_lines)
    string(APPEND _vs_found "\n  ${_vs_rel}: ${_vs_line}")
  endforeach()
endforeach()
if(_vs_count EQUAL 0)
  message(
    FATAL_ERROR "no file read under ${SOURCE_DIR}: the check reads nothing")
endif()
if(_vs_found)
  message(FATAL_ERROR "the sensor package includes recon:${_vs_found}")
endif()
message(STATUS "the sensor package includes no recon header: ${_vs_count} "
               "files read")
