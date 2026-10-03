// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Private bridge to TU-local stb implementations. No stb symbols or mutable
// settings are shared with applications that bring their own copy of stb.

#include <cstdint>

namespace volumetric_kit::recon::io::detail::stb {

// Decoders read an encoded file already in memory, so inspection and decoding
// see the same bytes.
bool info(const std::uint8_t* data, int size, int* width, int* height,
          int* channels);
bool is_16_bit(const std::uint8_t* data, int size);
std::uint8_t* load8(const std::uint8_t* data, int size, int* width, int* height,
                    int* channels, int requested_channels);
std::uint16_t* load16(const std::uint8_t* data, int size, int* width,
                      int* height, int* channels, int requested_channels);
// The calling thread's most recent decode failure.
const char* failure_reason();
void free_image(void* pixels) noexcept;

using WriteCallback = void (*)(void* context, void* data, int size);
bool write_png(WriteCallback callback, void* context, int width, int height,
               int channels, const void* pixels, int stride_bytes);

}  // namespace volumetric_kit::recon::io::detail::stb
