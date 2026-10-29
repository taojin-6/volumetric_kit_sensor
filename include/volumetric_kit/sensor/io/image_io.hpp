// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file io/image_io.hpp
/// @brief Host image decoding and PNG encoding, without backend types.
///
/// PNG encoding returns an allocation failure as a non-OK Status, except one
/// while its compressor grows its output, which aborts the process.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/io/image_io_export.hpp"

namespace volumetric_kit::recon::io {

/// @brief Decode a JPEG or PNG to encoded RGB packed as `R | G<<8 | B<<16`.
///
/// Rows are top to bottom, pixels left to right, and the high byte is zero.
/// Grayscale is expanded to RGB; alpha is discarded without premultiplication.
/// Encoded values are preserved (16-bit PNG samples are reduced to 8 bits).
/// This does not apply ICC profiles, transfer curves, or primary conversion:
/// callers must establish the source encoding before reconstruction.
/// @param path Image file path, without embedded NUL characters.
/// @param expected_w Expected positive width; a mismatch is an error.
/// @param expected_h Expected positive height; a mismatch is an error.
/// @return Packed pixels, or non-OK Status for invalid dimensions, a decode
///         failure, or an allocation failure.
VR_IMAGE_IO_API core::Result<std::vector<std::uint32_t>> load_color_packed(
    const std::string& path, std::uint32_t expected_w,
    std::uint32_t expected_h);

/// @brief Decode a genuine 16-bit, single-channel grayscale PNG to its
///        samples as stored.
///
/// Eight-bit, color, and grayscale-plus-alpha images (including a tRNS
/// transparency key) are rejected. No gamma conversion or scale is applied:
/// what a sample means, such as its units per metre, is the caller's to know.
/// Rows are top to bottom, pixels left to right.
/// @param path Depth PNG path, without embedded NUL characters.
/// @param expected_w Expected positive width; a mismatch is an error.
/// @param expected_h Expected positive height; a mismatch is an error.
/// @return The raw samples, or non-OK Status on invalid input, decode failure,
///         or allocation failure.
VR_IMAGE_IO_API core::Result<std::vector<std::uint16_t>> load_depth_u16(
    const std::string& path, std::uint32_t expected_w,
    std::uint32_t expected_h);

/// @brief Write tightly packed, encoded RGBA8 rows to a PNG file.
///
/// Bytes are preserved without flipping rows, premultiplication, or color
/// conversion. No color profile is attached. The caller retains ownership;
/// no extra caller-side copy of a mapped readback buffer is needed. Encoding
/// allocates temporary host storage. The output is opened only after encoding
/// succeeds, so invalid input or an encoder allocation failure leaves an
/// existing file intact; an I/O failure can leave a partial file.
/// @param path Output path, without embedded NUL characters; replaced on
///        success.
/// @param pixels Pointer to top-left-origin RGBA bytes, valid for this call.
/// @param byte_count Available bytes; must cover `width * height * 4`.
/// @param width Positive width supported by the encoder's integer bounds.
/// @param height Positive height supported by the encoder's integer bounds.
/// @return OK after closing the file, or non-OK Status for invalid input,
///         allocation failure, or encoding/write failure.
VR_IMAGE_IO_API core::Status write_png_rgba8(const std::string& path,
                                             const std::uint8_t* pixels,
                                             std::size_t byte_count,
                                             std::uint32_t width,
                                             std::uint32_t height);

/// @brief Encode tightly packed 16-bit samples as a 16-bit grayscale PNG in
///        memory.
///
/// The samples are stored as given, big-endian as PNG requires, with no
/// gamma, scale or color profile, so @ref load_depth_u16 returns them
/// exactly. Calls share no state and may run concurrently.
/// @param samples Pointer to top-left-origin row-major samples, valid for
///        this call.
/// @param sample_count Available samples; must cover `width * height`.
/// @param width Positive width supported by the encoder's integer bounds.
/// @param height Positive height supported by the encoder's integer bounds.
/// @return The PNG file's bytes, or non-OK Status for invalid input or
///         allocation failure.
VR_IMAGE_IO_API core::Result<std::vector<std::uint8_t>> encode_png_u16(
    const std::uint16_t* samples, std::size_t sample_count, std::uint32_t width,
    std::uint32_t height);

/// @brief Write tightly packed 16-bit samples to a 16-bit grayscale PNG file.
///
/// Writes the bytes of @ref encode_png_u16. The output is opened only after
/// encoding succeeds, so invalid input or an allocation failure leaves an
/// existing file intact; an I/O failure can leave a partial file.
/// @param path Output path, without embedded NUL characters; replaced on
///        success.
/// @param samples Pointer to top-left-origin row-major samples, valid for
///        this call.
/// @param sample_count Available samples; must cover `width * height`.
/// @param width Positive width supported by the encoder's integer bounds.
/// @param height Positive height supported by the encoder's integer bounds.
/// @return OK after closing the file, or non-OK Status for invalid input,
///         allocation failure, or write failure.
VR_IMAGE_IO_API core::Status write_png_u16(const std::string& path,
                                           const std::uint16_t* samples,
                                           std::size_t sample_count,
                                           std::uint32_t width,
                                           std::uint32_t height);

}  // namespace volumetric_kit::recon::io
