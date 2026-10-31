# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# color_fit: the rig's colour responses fitted from posed, prepared sets
# (DESIGN.md, "colour response fit"). An internal static library, built with the
# tests on every leg so its GPU test runs there, and its hand-run tool, which
# opens the Orbbec rig, built with VS_WITH_ORBBEC; no test runs the tool.
# Neither is installed, and the library has no public header.
add_library(sensor_color_fit STATIC color_fit.cpp)
target_include_directories(sensor_color_fit
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sensor_color_fit PUBLIC volumetric_kit::sensor_interface
                                              volumetric_kit::sensor_frame)
include(vs_embed)
vs_embed_shaders(
  sensor_color_fit
  SYMBOL_PREFIX
  vs_color_fit_
  INCLUDE_TARGETS
  volumetric_kit::sensor_frame
  SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sample.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/normal_equations.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/pushed.comp")
vs_target_warnings(sensor_color_fit)

if(VS_WITH_ORBBEC)
  add_executable(sensor_color_fit_tool main.cpp)
  set_target_properties(sensor_color_fit_tool PROPERTIES OUTPUT_NAME color_fit)
  target_link_libraries(
    sensor_color_fit_tool PRIVATE sensor_color_fit volumetric_kit::sensor_array
                                  volumetric_kit::sensor_orbbec)
  vs_target_warnings(sensor_color_fit_tool)
endif()
