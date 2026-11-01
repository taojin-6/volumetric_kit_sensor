# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tao Jin

# decklink_skew_report: a DeckLink stereo pair's timing by hand (README,
# "DeckLink stereo driver"), through the driver's capture unit and skew numbers.
# Built with VS_WITH_DECKLINK; no test runs it, as it opens the cameras.
add_executable(sensor_decklink_skew_report decklink_skew_report.cpp)
set_target_properties(sensor_decklink_skew_report
                      PROPERTIES OUTPUT_NAME decklink_skew_report)
target_link_libraries(sensor_decklink_skew_report
                      PRIVATE sensor_decklink_capture)
vs_target_warnings(sensor_decklink_skew_report)
