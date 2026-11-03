// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the DeckLink 8K Pro's connectors and the modes
// the stereo driver opens, as the SDK numbers them, with no SDK type. The
// SDK's smoke test holds every code here to the SDK's own enum.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace volumetric_kit::sensor::decklink {

// The connectors, as on the card's bracket, and their sub-devices under the
// profile "four sub-devices half-duplex" (SDK manual 2.4.11): SDI 1 = 0,
// SDI 3 = 1, SDI 2 = 2, SDI 4 = 3. -1 for a name that is none of them.
int sub_device_of(std::string_view input);

// The profile that gives each connector its own sub-device ('4dhd',
// bmdProfileFourSubDevicesHalfDuplex); the card ships in another.
inline constexpr std::uint32_t kFourSubDevicesHalfDuplex = 0x34646864;

// v210, the 10-bit 4:2:2 format captured ('v210', bmdFormat10BitYUV).
inline constexpr std::uint32_t kV210 = 0x76323130;

// A mode the driver opens: its BMDDisplayMode code, size, rate and name.
struct Mode {
  std::uint32_t code = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t fps = 0;
  const char* name = "";
};

// 2160p 24, 25 or 30 ('4k24', '4k25', '4k30'): the rates the cameras send
// and the card captures at UHD; empty for any other.
std::optional<Mode> mode_of(std::uint32_t width, std::uint32_t height,
                            std::uint32_t fps);

// The SDK's bytes a v210 row: whole 48-pixel blocks of 128 bytes, so 10,240
// at 3840 (the YuvLayout::V210 floor, ceil(width / 6) * 16, rounded up).
std::size_t v210_row_bytes(std::uint32_t width);

// A frame's period at `fps`, nanoseconds, rounded: 33,333,333 at 30.
std::uint64_t period_ns(std::uint32_t fps);

// What to do about a camera that sends `detected` where the mode named
// `wanted`, at `fps`, was opened: both named, and the camera's own setting
// to change.
std::string mode_fix(std::string_view wanted, std::uint32_t fps,
                     std::string_view detected);

}  // namespace volumetric_kit::sensor::decklink
