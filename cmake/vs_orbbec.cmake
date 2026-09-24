# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The Orbbec SDK, for the Orbbec (Femto Mega) capture driver: a prerequisite,
# found and never fetched (the 2026-09-24 decision). Included from the root
# CMakeLists only when VR_WITH_ORBBEC is ON; defines ob::OrbbecSDK,
# VR_ORBBEC_SDK_VERSION and its _MAJOR / _MINOR / _PATCH parts.
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
# OrbbecSDKConfig.cmake, and naming it pins that one SDK.)
#
# Two quirks of the SDK's own package files, both handled here rather than left
# for each caller to rediscover:
#
# * OrbbecSDKConfig.cmake sits in <sdk>/lib, which none of find_package's
#   standard layouts search under a prefix -- hence PATH_SUFFIXES lib, without
#   which OrbbecSDK_ROOT and CMAKE_PREFIX_PATH both miss it.
# * The version file is named OrbbecSDKVersion.cmake, not
#   OrbbecSDKConfigVersion.cmake, so find_package never reads it: a versioned
#   `find_package(OrbbecSDK 2.9.3)` rejects even a matching SDK, and an
#   unversioned one takes the first SDK on the search path whatever its version.
#   So the search here does what a versioned find_package would have done --
#   walks the same search path, asks each candidate's version file for its
#   verdict on VR_ORBBEC_SDK_MIN_VERSION, and skips the ones it rejects -- and
#   only then loads the one it chose. A rejected SDK's config must never load:
#   ob::OrbbecSDK, once imported, cannot be taken back.
#
# The search re-runs on every configure. find_package caches the directory it
# found in OrbbecSDK_DIR and would otherwise keep it for the life of the build
# tree, so re-pointing OrbbecSDK_ROOT at an upgraded SDK -- or at a newer one
# after an "incompatible" error -- would be ignored without a word. Only when
# the search finds nothing and nothing names an SDK any more (the environment
# variable the last configure saw is simply gone) is the last choice kept. A
# value in OrbbecSDK_DIR the search did not write is the caller's own choice and
# is taken as given.

# The oldest SDK this repo builds against. Raise it with the SDK the family
# installs, and keep it in lockstep with any sibling that finds the same copy
# and with the SDK CI installs (.github/workflows/_build.yml).
set(VR_ORBBEC_SDK_MIN_VERSION 2.9.3)

# The OFF-by-default reason, checked: macOS is the SDK's only Apple platform.
if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(
    FATAL_ERROR
      "VR_WITH_ORBBEC is ON, but the Orbbec SDK ships no ${CMAKE_SYSTEM_NAME} "
      "build (macOS is its only Apple platform). Configure this build with "
      "-DVR_WITH_ORBBEC=OFF.")
endif()

# Asks the SDK package in `dir` whether it satisfies VR_ORBBEC_SDK_MIN_VERSION,
# exactly as find_package asks a version file: the request goes in as
# PACKAGE_FIND_VERSION*, the verdict comes back as PACKAGE_VERSION_COMPATIBLE /
# _UNSUITABLE. That file carries the SDK's own rules -- the same major version
# and no older, and the same pointer width -- so they need no copy here. Sets
# `out_version` to the version it reports, and `out_rejection` to why the SDK
# cannot be used, empty if it can.
function(_vr_orbbec_judge dir out_version out_rejection)
  set(${out_version}
      ""
      PARENT_SCOPE)
  if(NOT EXISTS "${dir}/OrbbecSDKVersion.cmake")
    set(${out_rejection}
        "no OrbbecSDKVersion.cmake (the SDK's package layout may have changed)"
        PARENT_SCOPE)
    return()
  endif()

  # Unset first: this scope inherits the caller's, where a previous candidate's
  # file may have left them, and _UNSUITABLE is only ever set, never cleared.
  unset(PACKAGE_VERSION)
  unset(PACKAGE_VERSION_COMPATIBLE)
  unset(PACKAGE_VERSION_EXACT)
  unset(PACKAGE_VERSION_UNSUITABLE)
  set(PACKAGE_FIND_NAME OrbbecSDK)
  set(PACKAGE_FIND_VERSION ${VR_ORBBEC_SDK_MIN_VERSION})
  string(REPLACE "." ";" _parts "${VR_ORBBEC_SDK_MIN_VERSION}")
  list(GET _parts 0 PACKAGE_FIND_VERSION_MAJOR)
  list(GET _parts 1 PACKAGE_FIND_VERSION_MINOR)
  list(GET _parts 2 PACKAGE_FIND_VERSION_PATCH)
  set(PACKAGE_FIND_VERSION_TWEAK 0)
  set(PACKAGE_FIND_VERSION_COUNT 3)
  include("${dir}/OrbbecSDKVersion.cmake")

  set(${out_version}
      "${PACKAGE_VERSION}"
      PARENT_SCOPE)
  if(PACKAGE_VERSION_UNSUITABLE)
    set(_why "built for a different pointer width")
  elseif(NOT PACKAGE_VERSION_COMPATIBLE)
    if(PACKAGE_VERSION VERSION_LESS VR_ORBBEC_SDK_MIN_VERSION)
      set(_why "older than ${VR_ORBBEC_SDK_MIN_VERSION}")
    else()
      set(_why "a different major version")
    endif()
  elseif(NOT EXISTS "${dir}/OrbbecSDKConfig.cmake")
    set(_why "no OrbbecSDKConfig.cmake beside its version file")
  else()
    set(${out_rejection}
        ""
        PARENT_SCOPE)
    return()
  endif()
  set(${out_rejection}
      "SDK ${PACKAGE_VERSION}, ${_why}"
      PARENT_SCOPE)
endfunction()

# Walks find_package's search path for SDK packages -- pointed at the version
# file, so a candidate's config is not loaded while it is judged -- and sets
# `out_dir` / `out_version` to the first one _vr_orbbec_judge accepts, both
# empty if none. Each rejected candidate is ignored for the rest of the walk and
# reported in `out_rejected`, one "<dir>: <rejection>" per entry.
function(_vr_orbbec_search out_dir out_version out_rejected)
  set(${out_dir}
      ""
      PARENT_SCOPE)
  set(${out_version}
      ""
      PARENT_SCOPE)
  set(_rejected "")
  while(TRUE)
    unset(OrbbecSDK_DIR CACHE)
    find_package(
      OrbbecSDK
      QUIET
      CONFIG
      CONFIGS
      OrbbecSDKVersion.cmake
      PATH_SUFFIXES
      lib)
    if(NOT OrbbecSDK_FOUND)
      break()
    endif()
    _vr_orbbec_judge("${OrbbecSDK_DIR}" _version _rejection)
    if(NOT _rejection)
      set(${out_dir}
          "${OrbbecSDK_DIR}"
          PARENT_SCOPE)
      set(${out_version}
          "${_version}"
          PARENT_SCOPE)
      break()
    endif()
    list(APPEND _rejected "${OrbbecSDK_DIR}: ${_rejection}")
    # Local to this function, so the caller's own CMAKE_IGNORE_PATH is where the
    # walk starts and is left as it found it.
    list(APPEND CMAKE_IGNORE_PATH "${OrbbecSDK_DIR}")
  endwhile()
  set(${out_rejected}
      "${_rejected}"
      PARENT_SCOPE)
endfunction()

string(REGEX MATCH "^[0-9]+" _vr_orbbec_min_major
             "${VR_ORBBEC_SDK_MIN_VERSION}")
set(_vr_orbbec_wanted
    "${VR_ORBBEC_SDK_MIN_VERSION} or a newer ${_vr_orbbec_min_major}.x release")

if(OrbbecSDK_DIR AND NOT OrbbecSDK_DIR STREQUAL
                     "${_VR_ORBBEC_SDK_SEARCHED_DIR}")
  # Named by the caller (-DOrbbecSDK_DIR=<sdk>/lib): that SDK or none.
  set(_vr_orbbec_dir "${OrbbecSDK_DIR}")
  _vr_orbbec_judge("${_vr_orbbec_dir}" _vr_orbbec_version _vr_orbbec_rejection)
  if(_vr_orbbec_rejection)
    message(
      FATAL_ERROR
        "OrbbecSDK_DIR is set to ${_vr_orbbec_dir}, which this repo cannot "
        "use: ${_vr_orbbec_rejection}. It requires ${_vr_orbbec_wanted}. Point "
        "OrbbecSDK_DIR at the lib/ of such an SDK, or drop it "
        "(-UOrbbecSDK_DIR) to search OrbbecSDK_ROOT instead.")
  endif()
  set(_VR_ORBBEC_SDK_SEARCHED_DIR
      ""
      CACHE INTERNAL "")
else()
  # Cleared before the search, so a failed one leaves no stale claim behind: the
  # cache is written even when the configure stops.
  set(_vr_orbbec_previous "${_VR_ORBBEC_SDK_SEARCHED_DIR}")
  set(_VR_ORBBEC_SDK_SEARCHED_DIR
      ""
      CACHE INTERNAL "")
  _vr_orbbec_search(_vr_orbbec_dir _vr_orbbec_version _vr_orbbec_rejected)
  if(NOT _vr_orbbec_dir
     AND NOT _vr_orbbec_rejected
     AND _vr_orbbec_previous
     AND NOT OrbbecSDK_ROOT
     AND "$ENV{OrbbecSDK_ROOT}" STREQUAL "")
    # Nothing on the search path, and nothing names an SDK any more: the
    # environment the last configure saw is gone -- a one-off `OrbbecSDK_ROOT=
    # <sdk> cmake ...`, or an IDE that never read the shell profile. Keep the
    # SDK that configure chose, as any cached find result would be kept, if it
    # still passes. A root that is named and holds nothing usable is an error.
    _vr_orbbec_judge("${_vr_orbbec_previous}" _vr_orbbec_version
                     _vr_orbbec_rejection)
    if(NOT _vr_orbbec_rejection)
      set(_vr_orbbec_dir "${_vr_orbbec_previous}")
      message(STATUS "Orbbec SDK: nothing names one now; keeping the last "
                     "configure's")
    endif()
  endif()
  if(NOT _vr_orbbec_dir)
    set(_vr_orbbec_seen "")
    if(OrbbecSDK_ROOT)
      string(APPEND _vr_orbbec_seen
             "\n  searched OrbbecSDK_ROOT=${OrbbecSDK_ROOT}")
    endif()
    if(NOT "$ENV{OrbbecSDK_ROOT}" STREQUAL "")
      string(APPEND _vr_orbbec_seen
             "\n  searched ENV{OrbbecSDK_ROOT}=$ENV{OrbbecSDK_ROOT}")
    endif()
    foreach(_vr_orbbec_skipped IN LISTS _vr_orbbec_rejected)
      string(APPEND _vr_orbbec_seen "\n  skipped ${_vr_orbbec_skipped}")
    endforeach()
    message(
      FATAL_ERROR
        "VR_WITH_ORBBEC is ON but no usable Orbbec SDK was found."
        "${_vr_orbbec_seen}\n"
        "Install the SDK (${_vr_orbbec_wanted}, "
        "https://github.com/orbbec/OrbbecSDK_v2/releases) and point the build "
        "at its root with -DOrbbecSDK_ROOT=<sdk> or the OrbbecSDK_ROOT "
        "environment variable; the next configure searches again.")
  endif()
  foreach(_vr_orbbec_skipped IN LISTS _vr_orbbec_rejected)
    message(STATUS "Orbbec SDK skipped ${_vr_orbbec_skipped}")
  endforeach()
  set(OrbbecSDK_DIR
      "${_vr_orbbec_dir}"
      CACHE PATH "The directory containing OrbbecSDKConfig.cmake." FORCE)
  set(_VR_ORBBEC_SDK_SEARCHED_DIR
      "${_vr_orbbec_dir}"
      CACHE INTERNAL "")
endif()

if(NOT _vr_orbbec_version MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
  message(
    FATAL_ERROR
      "Could not read the Orbbec SDK version \"${_vr_orbbec_version}\" from "
      "${_vr_orbbec_dir}/OrbbecSDKVersion.cmake; the SDK's package layout may "
      "have changed.")
endif()
set(VR_ORBBEC_SDK_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
set(VR_ORBBEC_SDK_VERSION_MAJOR ${CMAKE_MATCH_1})
set(VR_ORBBEC_SDK_VERSION_MINOR ${CMAKE_MATCH_2})
set(VR_ORBBEC_SDK_VERSION_PATCH ${CMAKE_MATCH_3})

# Only now, with the SDK chosen, does its config load and import ob::OrbbecSDK.
find_package(OrbbecSDK CONFIG REQUIRED NO_DEFAULT_PATH PATHS
             "${_vr_orbbec_dir}")

message(STATUS "Orbbec SDK ${VR_ORBBEC_SDK_VERSION}: ${_vr_orbbec_dir}")
