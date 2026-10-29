// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/hevc_syntax.hpp
/// @brief H.265 syntax read on the host: an Annex B access unit's NAL units,
///        the parameter sets, and the start of a slice header.
///
/// What @ref HevcDecoder reads itself on Apple, where it hands VideoToolbox
/// the slices: the parameter sets' ids and what they say about the pictures
/// (size, crop, format, the VUI's colour, the reorder depth), and each
/// picture's slice header as far as its picture order count. Pure host code,
/// with no codec, no GPU and no FFmpeg, so its target,
/// `volumetric_kit::sensor_video_syntax`, builds on every platform.
/// Every parser reads through @ref hevc::BitReader.
/// @ref hevc::split_annex_b and the parsers return a host allocation failure
/// as `Status::Code::OutOfMemory`. Clause and table numbers are ITU-T H.265's.

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"
#include "volumetric_kit/sensor/video/syntax_export.hpp"

namespace volumetric_kit::sensor::hevc {

/// @name NAL unit types (Table 7-1)
/// @{
inline constexpr int kRaslN = 8;      ///< RASL_N.
inline constexpr int kRaslR = 9;      ///< RASL_R.
inline constexpr int kIdrWRadl = 19;  ///< IDR_W_RADL.
inline constexpr int kIdrNLp = 20;    ///< IDR_N_LP.
inline constexpr int kCraNut = 21;    ///< CRA_NUT.
inline constexpr int kVps = 32;       ///< VPS_NUT: a video parameter set.
inline constexpr int kSps = 33;       ///< SPS_NUT: a sequence parameter set.
inline constexpr int kPps = 34;       ///< PPS_NUT: a picture parameter set.
inline constexpr int kEos = 36;       ///< EOS_NUT: an end of sequence.
/// @}

/// @return Whether NAL unit type @p type is an IRAP picture's (16 to 23): a
///         BLA, an IDR or a CRA, where decoding can start.
constexpr bool is_irap(int type) noexcept { return type >= 16 && type <= 23; }

/// @return Whether NAL unit type @p type is an IDR picture's.
constexpr bool is_idr(int type) noexcept {
  return type == kIdrWRadl || type == kIdrNLp;
}

/// @return Whether NAL unit type @p type is a RASL picture's: a leading
///         picture predicted from before its CRA, which a decoder that
///         starts at that CRA skips.
constexpr bool is_rasl(int type) noexcept {
  return type == kRaslN || type == kRaslR;
}

/// @brief One NAL unit of an Annex B stream, as @ref split_annex_b finds it.
///
/// Borrowed: @ref data points into the caller's bytes, which must outlive it.
struct Nal {
  /// The unit's first byte, its header's: the start code and the zeros after
  /// the unit are left off, its emulation prevention bytes left in.
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;  ///< The bytes at @ref data, the header's two included.
  int type = 0;          ///< nal_unit_type.
  int layer_id = 0;      ///< nuh_layer_id.
  int temporal_id = 0;   ///< TemporalId: nuh_temporal_id_plus1 less one.
};

/// @brief Find the NAL units of Annex B bytes (B.2), such as an access unit.
/// @param data The bytes: NAL units, each after a start code (`00 00 01`);
///             the zeros between units, a four-byte start code's first among
///             them, are dropped, and bytes before the first start code are
///             not read.
/// @param size The bytes at @p data.
/// @param out  Cleared, then given each NAL unit in order. Its capacity is
///             kept, so a caller that reuses it allocates only for an access
///             unit of more units than before.
/// @return OK; `Status::Code::IoError` if there is no start code, or one is
///         not followed by a NAL unit header that is valid (two bytes, the
///         forbidden bit clear and nuh_temporal_id_plus1 not 0), @p out then
///         holding the units before it; or `Status::Code::InvalidArgument`
///         for a null @p data with a size.
VS_VIDEO_SYNTAX_API core::Status split_annex_b(const std::uint8_t* data,
                                               std::size_t size,
                                               std::vector<Nal>& out);

/// @brief Reads a NAL unit's payload as H.265's descriptors read it (7.2):
///        its bits most significant first, from the byte after the two-byte
///        header, each emulation prevention byte (the 03 of `00 00 03`)
///        skipped as it comes, so what it reads is the RBSP.
///
/// Every parser here is built on it. A read past the end reads zeros and
/// sets @ref overrun, so a parser can read a structure through and check
/// once. A copy reads on from where the original was.
class VS_VIDEO_SYNTAX_API BitReader {
 public:
  /// @brief A reader of @p nal's payload; a unit of no more than its header
  ///        has none.
  /// @param nal Borrowed: its bytes must outlive the reader.
  explicit BitReader(const Nal& nal) noexcept;

  /// @brief Read u(n), @p n bits as an unsigned number.
  /// @param n 0 to 32; anything else is a programmer error (`VKC_CHECK`).
  /// @return The bits, the first read the most significant.
  std::uint32_t bits(int n);

  /// @brief Read u(1) as a flag.
  /// @return Whether the bit is 1.
  bool flag();

  /// @brief Read past @p n bits; none if @p n is not positive.
  void skip(int n);

  /// @brief Read ue(v), an unsigned Exp-Golomb code (9.2).
  /// @return The value, 0 to 2^32 - 2; 0, with @ref overrun set, for a code
  ///         of more than 31 leading zeros, which no H.265 field takes.
  std::uint32_t ue();

  /// @brief Read se(v), a signed Exp-Golomb code (9.2.2): the ue(v) value k
  ///        as (k + 1) / 2 when k is odd, else as -(k / 2).
  /// @return The value, -(2^31 - 1) to 2^31 - 1.
  std::int32_t se();

  /// @return Whether a read went past the payload's end, reading zeros, or
  ///         @ref ue met a code too long.
  bool overrun() const noexcept { return overrun_; }

  /// @brief Read the rest of the payload: rbsp_trailing_bits (7.3.2.11), if
  ///        it is a one and then zeros to the end.
  /// @return Whether it is; false too if a read had gone past the end.
  bool at_trailing_bits();

 private:
  std::uint32_t bit();
  bool load();

  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t pos_ = 0;
  int zeros_ = 0;  // zero bytes just read, for the next 03
  std::uint8_t byte_ = 0;
  int left_ = 0;  // bits of byte_ not yet read
  bool overrun_ = false;
};

/// @brief Read a VPS's id: vps_video_parameter_set_id, its first four bits.
/// @param vps A video parameter set (@ref kVps).
/// @return The id, 0 to 15; `Status::Code::IoError` if @p vps has no
///         payload; or `Status::Code::InvalidArgument` if it is not a VPS.
VS_VIDEO_SYNTAX_API core::Result<std::uint32_t> vps_id(const Nal& vps);

/// @brief What a sequence parameter set (7.3.2.2) says about its pictures.
struct Sps {
  std::uint32_t id = 0;      ///< sps_seq_parameter_set_id, 0 to 15.
  std::uint32_t vps_id = 0;  ///< sps_video_parameter_set_id: the VPS it names.
  /// chroma_format_idc: 0 (4:0:0), 1 (4:2:0), 2 (4:2:2) or 3 (4:4:4).
  std::uint32_t chroma_format_idc = 1;
  /// separate_colour_plane_flag: 4:4:4 coded as three monochrome planes.
  bool separate_colour_plane = false;
  std::uint32_t coded_width = 0;   ///< pic_width_in_luma_samples.
  std::uint32_t coded_height = 0;  ///< pic_height_in_luma_samples.
  /// @name The conformance window: the luma samples cropped off each edge.
  /// @{
  std::uint32_t crop_left = 0;    ///< Off the left.
  std::uint32_t crop_right = 0;   ///< Off the right.
  std::uint32_t crop_top = 0;     ///< Off the top.
  std::uint32_t crop_bottom = 0;  ///< Off the bottom.
  /// @}
  std::uint32_t bit_depth_luma = 8;    ///< BitDepthY, 8 to 16.
  std::uint32_t bit_depth_chroma = 8;  ///< BitDepthC, 8 to 16.
  /// The bits of a slice's slice_pic_order_cnt_lsb, 4 to 16.
  std::uint32_t log2_max_poc_lsb = 4;
  /// sps_max_num_reorder_pics of the highest sub-layer: how many pictures
  /// may precede one in decoding order and follow it in output order.
  std::uint32_t max_num_reorder = 0;
  /// The colour the VUI declares; the defaults where it declares none or
  /// there is no VUI.
  ColorCodes color;

  /// @return The picture's width after the crop.
  std::uint32_t width() const noexcept {
    return coded_width - crop_left - crop_right;
  }
  /// @return The picture's height after the crop.
  std::uint32_t height() const noexcept {
    return coded_height - crop_top - crop_bottom;
  }
};

/// @brief Read a sequence parameter set whole, through its
///        rbsp_trailing_bits; an SPS with a 3D, screen-content or later
///        extension is taken as it is past that extension's flag, whose
///        syntax is not walked.
/// @param sps A sequence parameter set (@ref kSps).
/// @return The SPS; `Status::Code::IoError` if it does not parse: it is cut
///         short, a value is out of its range (an id past 15, a size of 0 or
///         past 16888, a window that leaves no picture, a bit depth past 16,
///         a reorder depth not below its buffer, ...), or anything but its
///         trailing bits follows it; or `Status::Code::InvalidArgument` if
///         @p sps is not an SPS.
VS_VIDEO_SYNTAX_API core::Result<Sps> parse_sps(const Nal& sps);

/// @brief The leading fields of a picture parameter set (7.3.2.3): what the
///        start of a slice header needs.
struct Pps {
  std::uint32_t id = 0;      ///< pps_pic_parameter_set_id, 0 to 63.
  std::uint32_t sps_id = 0;  ///< pps_seq_parameter_set_id: the SPS it names.
  /// output_flag_present_flag: whether a slice header carries
  /// pic_output_flag.
  bool output_flag_present = false;
  /// num_extra_slice_header_bits, 0 to 7.
  std::uint32_t num_extra_slice_header_bits = 0;
};

/// @brief Read a picture parameter set's leading fields.
/// @param pps A picture parameter set (@ref kPps).
/// @return Its fields; `Status::Code::IoError` if they are cut short or an
///         id is out of its range; or `Status::Code::InvalidArgument` if
///         @p pps is not a PPS.
VS_VIDEO_SYNTAX_API core::Result<Pps> parse_pps(const Nal& pps);

/// @brief The start of a slice segment header (7.3.6.1), as far as the
///        picture order count.
struct SliceStart {
  /// first_slice_segment_in_pic_flag: the segment starts its picture.
  bool first_slice_segment = false;
  /// no_output_of_prior_pics_flag, which an IRAP picture's header carries.
  bool no_output_of_prior_pics = false;
  std::uint32_t pps_id = 0;  ///< slice_pic_parameter_set_id.
  /// pic_output_flag; true where the PPS says the header carries none.
  bool pic_output = true;
  /// slice_pic_order_cnt_lsb; 0 for an IDR picture, which carries none.
  std::uint32_t poc_lsb = 0;
};

/// @brief Read the start of @p slice's header, with the parameter sets it
///        names. Only the first segment of a picture is read past its PPS
///        id: a later one's fields after @ref SliceStart::pps_id keep their
///        defaults.
/// @param slice A VCL NAL unit (a type below 32).
/// @param pps   The picture parameter sets, by id.
/// @param sps   The sequence parameter sets, by id.
/// @return The fields; `Status::Code::IoError` if the header is cut short,
///         its slice_type is past 2, or it names a PPS not in @p pps or one
///         whose SPS is not in @p sps; or `Status::Code::InvalidArgument` if
///         @p slice is not a VCL NAL unit.
VS_VIDEO_SYNTAX_API core::Result<SliceStart> parse_slice_start(
    const Nal& slice, const std::map<std::uint32_t, Pps>& pps,
    const std::map<std::uint32_t, Sps>& sps);

}  // namespace volumetric_kit::sensor::hevc
