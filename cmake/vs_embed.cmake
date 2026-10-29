# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# vs_embed_shaders(<target> [SYMBOL_PREFIX <prefix>] [INCLUDE_TARGETS
# <target>...] SHADERS <file>...)
#
# The core's vkc_embed_shaders with spirv-val's --scalar-block-layout, the
# buffer ABI the kernels declare, and the vs_ symbol prefix unless another is
# given: undistort_depth.comp embeds as undistort_depth_comp.spv.hpp, declaring
# `vs_undistort_depth_comp_spv[]` and `vs_undistort_depth_comp_spv_size`. The
# shaders include through the package's src/, so a cross-tier include reads like
# its C++ header path ("volumetric_kit/sensor/frame/shaders/pinhole.glsl"); with
# INCLUDE_TARGETS, through those targets' exported GLSL alone instead, as a
# project that adds the package does.
function(vs_embed_shaders target)
  cmake_parse_arguments(ARG "" "SYMBOL_PREFIX" "INCLUDE_TARGETS;SHADERS"
                        ${ARGN})
  if(NOT ARG_SYMBOL_PREFIX)
    set(ARG_SYMBOL_PREFIX vs_)
  endif()
  if(ARG_INCLUDE_TARGETS)
    set(_includes INCLUDE_TARGETS ${ARG_INCLUDE_TARGETS})
  else()
    set(_includes INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")
  endif()
  vkc_embed_shaders(
    ${target}
    SYMBOL_PREFIX
    ${ARG_SYMBOL_PREFIX}
    ${_includes}
    SPIRV_VAL_ARGS
    --scalar-block-layout
    ${ARG_UNPARSED_ARGUMENTS}
    SHADERS
    ${ARG_SHADERS})
endfunction()
