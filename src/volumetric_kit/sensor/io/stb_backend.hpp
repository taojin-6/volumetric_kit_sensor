// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Private bridge to TU-local stb implementations. No stb symbols or mutable
// settings are shared with applications that bring their own copy of stb.

#include <cstdint>

namespace volumetric_kit::recon::io::detail::stb {

bool info(const char* path, int* width, int* height, int* channels);
bool is_16_bit(const char* path);
std::uint8_t* load8(const char* path, int* width, int* height, int* channels,
                    int requested_channels);
std::uint16_t* load16(const char* path, int* width, int* height, int* channels,
                      int requested_channels);
void free_image(void* pixels) noexcept;

using WriteCallback = void (*)(void* context, void* data, int size);
bool write_png(WriteCallback callback, void* context, int width, int height,
               int channels, const void* pixels, int stride_bytes);

}  // namespace volumetric_kit::recon::io::detail::stb
