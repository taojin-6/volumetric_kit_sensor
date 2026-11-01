// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "ports.hpp"

namespace volumetric_kit::sensor::decklink {

int sub_device_of(std::string_view input) {
  if (input == "SDI 1") return 0;
  if (input == "SDI 3") return 1;
  if (input == "SDI 2") return 2;
  if (input == "SDI 4") return 3;
  return -1;
}

std::optional<Mode> mode_of(std::uint32_t width, std::uint32_t height,
                            std::uint32_t fps) {
  if (width != 3840 || height != 2160) return std::nullopt;
  switch (fps) {
    case 24:
      return Mode{0x346B3234, width, height, fps, "2160p24"};
    case 25:
      return Mode{0x346B3235, width, height, fps, "2160p25"};
    case 30:
      return Mode{0x346B3330, width, height, fps, "2160p30"};
    default:
      return std::nullopt;
  }
}

std::size_t v210_row_bytes(std::uint32_t width) {
  return (static_cast<std::size_t>(width) + 47) / 48 * 128;
}

std::uint64_t period_ns(std::uint32_t fps) {
  return fps == 0 ? 0 : (std::uint64_t{1'000'000'000} + fps / 2) / fps;
}

std::string mode_fix(std::string_view wanted, std::uint32_t fps,
                     std::string_view detected) {
  return "the camera sends " + std::string(detected) + ", not " +
         std::string(wanted) + ": set it to " + std::to_string(fps) +
         " fps (its menu, or USB PTP property 0xd006 = " +
         std::to_string(fps * 100) + ") and open again";
}

}  // namespace volumetric_kit::sensor::decklink
