// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The single translation unit that compiles stb image decoding and writing.
// Kept apart from image_io.cpp behind a private bridge so third-party code can
// be built with warnings disabled. Its symbols and settings are TU-local,
// including when recon_io is static and its consumer uses another stb copy.

#include "stb_backend.hpp"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace volumetric_kit::recon::io::detail::stb {

bool info(const std::uint8_t* data, int size, int* width, int* height,
          int* channels) {
  return stbi_info_from_memory(data, size, width, height, channels) != 0;
}

bool is_16_bit(const std::uint8_t* data, int size) {
  return stbi_is_16_bit_from_memory(data, size) != 0;
}

std::uint8_t* load8(const std::uint8_t* data, int size, int* width, int* height,
                    int* channels, int requested_channels) {
  return stbi_load_from_memory(data, size, width, height, channels,
                               requested_channels);
}

std::uint16_t* load16(const std::uint8_t* data, int size, int* width,
                      int* height, int* channels, int requested_channels) {
  return stbi_load_16_from_memory(data, size, width, height, channels,
                                  requested_channels);
}

const char* failure_reason() { return stbi_failure_reason(); }

void free_image(void* pixels) noexcept { stbi_image_free(pixels); }

bool write_png(WriteCallback callback, void* context, int width, int height,
               int channels, const void* pixels, int stride_bytes) {
  return stbi_write_png_to_func(callback, context, width, height, channels,
                                pixels, stride_bytes) != 0;
}

}  // namespace volumetric_kit::recon::io::detail::stb
