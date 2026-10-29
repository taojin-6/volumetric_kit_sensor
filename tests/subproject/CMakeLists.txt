# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# A stand-in for a project that adds the sensor package's source from
# VS_SOURCE_DIR, as the sensor_subproject_* tests configure it: the package is
# then not the top-level project, so its tests and -Werror must default off, and
# the consumer builds the library tiers alone, under its own compiler's
# warnings. With all of the package, image I/O links the core's base tier alone,
# and sensor_subproject_glsl builds a kernel on the package's exported GLSL.
#
# With -DVS_CORE_FIRST=ON it makes the core available first, as an application
# that also fetches calib or gfx does, with VKC_WITH_VULKAN as given, so the
# package's pin yields to it; -DVS_CONSUMER_CORE_VERSION=<version> stamps that
# version on the core, standing in for an older one. -DVS_REQUIRE_SENSOR=
# <version> is the consumer's floor on the package.
cmake_minimum_required(VERSION 3.21)
project(vs_subproject_consumer LANGUAGES CXX)

if(VS_CORE_FIRST)
  include(FetchContent)
  set(_core_source "${FETCHCONTENT_SOURCE_DIR_VOLUMETRIC_KIT_CORE}")
  FetchContent_Declare(volumetric_kit_core SOURCE_DIR "${_core_source}")
  FetchContent_MakeAvailable(volumetric_kit_core)
  if(VS_CONSUMER_CORE_VERSION)
    get_target_property(_core volumetric_kit::core_base ALIASED_TARGET)
    set_target_properties(${_core} PROPERTIES VKC_CORE_VERSION
                                              ${VS_CONSUMER_CORE_VERSION})
  endif()
endif()
add_subdirectory("${VS_SOURCE_DIR}" volumetric_kit_sensor)
foreach(_option VS_BUILD_TESTS VS_WARNINGS_AS_ERRORS)
  if(${_option})
    message(FATAL_ERROR "${_option} must default OFF for a non-top-level "
                        "sensor package")
  endif()
endforeach()

# With -DVS_REQUIRE_SENSOR=<version>, the consumer needs a package that new, and
# checks it as a sibling does, with the core's vkc_require_sibling.
if(VS_REQUIRE_SENSOR)
  vkc_require_sibling(volumetric_kit::sensor ${VS_REQUIRE_SENSOR})
endif()

# A camera-only package (VS_CAMERA_ONLY) makes no core vulkan tier and no other
# tier, and fetches no dependency of the other tiers. That it finds no Vulkan is
# the caller's CMAKE_DISABLE_FIND_PACKAGE_Vulkan, as Vulkan's target would be
# local to the package's directory, not visible here.
if(VS_EXPECT_CAMERA_ONLY)
  foreach(_target volumetric_kit::core_vulkan volumetric_kit::sensor_frame
                  volumetric_kit::sensor_interface)
    if(TARGET ${_target})
      message(FATAL_ERROR "a camera-only sensor package made ${_target}")
    endif()
  endforeach()
  include(FetchContent)
  FetchContent_GetProperties(stb)
  if(stb_POPULATED)
    message(FATAL_ERROR "a camera-only sensor package fetched stb")
  endif()
endif()

# The umbrella reaches sensor_camera, so it must name only tiers that exist.
add_executable(vs_camera_consumer camera_consumer.cpp)
target_link_libraries(vs_camera_consumer PRIVATE volumetric_kit::sensor)

if(NOT TARGET volumetric_kit::sensor_frame)
  return()
endif()

# Image I/O, which a capture app links to write its images, links the core's
# base tier alone: no Vulkan and no other tier.
get_target_property(_image_io volumetric_kit::sensor_image_io ALIASED_TARGET)
get_target_property(_image_io_links ${_image_io} LINK_LIBRARIES)
if(NOT _image_io_links STREQUAL "volumetric_kit::core_base")
  message(FATAL_ERROR "sensor_image_io links ${_image_io_links}, "
                      "not volumetric_kit::core_base alone")
endif()

# What a project whose kernels include the package's exported GLSL does: compile
# through the core's INCLUDE_TARGETS, with vkc_compile_shaders, a kernel that
# includes frame/shaders/pinhole.glsl (volumetric_kit::sensor_frame;
# ../shaders/pinhole_cases.comp). The root holds exactly the GLSL sensor_frame
# exports: none of the package's own.
get_target_property(_roots volumetric_kit::sensor_frame VKC_SHADER_INCLUDE_DIRS)
set(_held "")
foreach(_root IN LISTS _roots)
  file(
    GLOB_RECURSE _files
    LIST_DIRECTORIES false
    RELATIVE "${_root}"
    "${_root}/*")
  list(APPEND _held ${_files})
endforeach()
if(NOT _held STREQUAL "volumetric_kit/sensor/frame/shaders/pinhole.glsl")
  message(FATAL_ERROR "sensor_frame's GLSL include root holds [${_held}], "
                      "not exactly what it exports, [pinhole.glsl]")
endif()
add_custom_target(vs_glsl_consumer)
vkc_compile_shaders(
  vs_glsl_consumer INCLUDE_TARGETS volumetric_kit::sensor_frame SHADERS
  "${CMAKE_CURRENT_LIST_DIR}/../shaders/pinhole_cases.comp")
