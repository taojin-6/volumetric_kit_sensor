# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# The DeckLink SDK, for the DeckLink stereo driver (sensor_decklink): its
# headers and DeckLinkAPIDispatch.cpp, a prerequisite unpacked once outside the
# repo and never fetched (the 2026-09-24 decision). Included from the root
# CMakeLists only when VS_WITH_DECKLINK is ON; defines VS_DECKLINK_SDK_VERSION
# and vs_link_decklink_sdk.
#
# Download "Desktop Video SDK" 16.0 or newer from Blackmagic's support site,
# unpack it, and point DeckLinkSDK_ROOT at its Linux directory, the one holding
# include/DeckLinkAPI.h. Desktop Video itself (the driver and libDeckLinkAPI.so)
# is needed only to capture: the dispatch file loads the library at run time, so
# nothing links it.
#
# TODO(sensor): the SDK headers in the CI image, once the owner decides on
# their licence; until then no CI leg builds VS_WITH_DECKLINK.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES
                                             "^(x86_64|AMD64)$")
  message(
    FATAL_ERROR
      "VS_WITH_DECKLINK is ON, but the DeckLink driver builds on Linux x86-64 "
      "only. Configure this build with -DVS_WITH_DECKLINK=OFF.")
endif()

set(DeckLinkSDK_ROOT
    "$ENV{DeckLinkSDK_ROOT}"
    CACHE PATH "The DeckLink SDK's Linux directory (include/DeckLinkAPI.h)")
find_path(
  VS_DECKLINK_INCLUDE_DIR DeckLinkAPI.h
  PATHS "${DeckLinkSDK_ROOT}/include"
  NO_DEFAULT_PATH)
if(NOT VS_DECKLINK_INCLUDE_DIR
   OR NOT EXISTS "${VS_DECKLINK_INCLUDE_DIR}/DeckLinkAPIDispatch.cpp"
   OR NOT EXISTS "${VS_DECKLINK_INCLUDE_DIR}/DeckLinkAPIVersion.h")
  unset(VS_DECKLINK_INCLUDE_DIR CACHE)
  message(
    FATAL_ERROR
      "VS_WITH_DECKLINK is ON but no DeckLink SDK was found at "
      "DeckLinkSDK_ROOT=\"${DeckLinkSDK_ROOT}\": download the Desktop Video SDK "
      "(16.0 or newer) from Blackmagic, unpack it, and set -DDeckLinkSDK_ROOT=<its "
      "Linux directory>, the one holding include/DeckLinkAPI.h.")
endif()

file(STRINGS "${VS_DECKLINK_INCLUDE_DIR}/DeckLinkAPIVersion.h" _vs_decklink_line
     REGEX "BLACKMAGIC_DECKLINK_API_VERSION_STRING")
if(NOT _vs_decklink_line MATCHES "\"([0-9]+)\\.([0-9]+)")
  message(FATAL_ERROR "Could not read the DeckLink SDK's version from "
                      "${VS_DECKLINK_INCLUDE_DIR}/DeckLinkAPIVersion.h.")
endif()
set(VS_DECKLINK_SDK_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
if(CMAKE_MATCH_1 LESS 16)
  message(
    FATAL_ERROR
      "The DeckLink SDK at ${DeckLinkSDK_ROOT} is ${VS_DECKLINK_SDK_VERSION}; "
      "the driver builds against 16.0 or newer.")
endif()
message(
  STATUS "DeckLink SDK ${VS_DECKLINK_SDK_VERSION}: ${VS_DECKLINK_INCLUDE_DIR}")

find_package(Threads REQUIRED)

# vs_link_decklink_sdk(<target>): the SDK's headers, as system headers, and its
# dispatch file compiled into the target, which loads Desktop Video's library at
# run time (dlopen). Its warnings are the SDK's, so it is compiled without ours.
function(vs_link_decklink_sdk target)
  set(_vs_dispatch "${VS_DECKLINK_INCLUDE_DIR}/DeckLinkAPIDispatch.cpp")
  target_sources(${target} PRIVATE "${_vs_dispatch}")
  set_source_files_properties("${_vs_dispatch}" TARGET_DIRECTORY ${target}
                              PROPERTIES COMPILE_OPTIONS "-w")
  target_include_directories(${target} SYSTEM
                             PRIVATE "${VS_DECKLINK_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
endfunction()
