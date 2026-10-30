# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# mic_check: a microphone checked by hand (README, "Microphone driver"). Built
# with the tests when VS_WITH_PORTAUDIO is ON, so CI compiles it; no test runs
# it, as it opens a mic.
add_executable(sensor_mic_check mic_check.cpp)
set_target_properties(sensor_mic_check PROPERTIES OUTPUT_NAME mic_check)
target_link_libraries(sensor_mic_check
                      PRIVATE volumetric_kit::sensor_microphone)
vs_target_warnings(sensor_mic_check)
