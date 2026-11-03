# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# vt_pattern_clip: makes the committed VideoToolbox-coded clip
# (tools/make_hevc_fixtures.sh). Built with the tests on Apple, so CI compiles
# it; no test runs it, since its output is committed.
add_executable(sensor_vt_pattern_clip vt_pattern_clip.cpp)
set_target_properties(sensor_vt_pattern_clip PROPERTIES OUTPUT_NAME
                                                        vt_pattern_clip)
target_link_libraries(sensor_vt_pattern_clip
                      PRIVATE volumetric_kit::sensor_video)
vs_target_warnings(sensor_vt_pattern_clip)
