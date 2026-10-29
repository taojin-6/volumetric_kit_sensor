// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The single translation unit that compiles stb image decoding and writing.
// Kept apart from image_io.cpp behind a private bridge so third-party code can
// be built with warnings disabled. Its symbols and settings are TU-local,
// including when sensor_image_io is static and its consumer uses another stb
// copy.

#include "stb_backend.hpp"
#include "volumetric_kit/core/base/check.hpp"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

// stb_image_write checks a failed realloc of its compressed stream only with
// STBIW_ASSERT, then writes through the old or null pointer: with assert
// compiled out, that is undefined. Fail fast in every build instead. Its
// fixed-size buffers' allocation failures still return 0 (the last of them
// leaks the stream).
#define STBIW_ASSERT(x) \
  VKC_CHECK(x, "stb_image_write: allocation or internal check failed")
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace volumetric_kit::sensor::io::detail::stb {

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

std::uint32_t png_crc(const std::uint8_t* data, int size) {
  // stb's chunk CRC; it only reads the bytes.
  return stbiw__crc32(const_cast<unsigned char*>(data), size);
}

}  // namespace volumetric_kit::sensor::io::detail::stb
