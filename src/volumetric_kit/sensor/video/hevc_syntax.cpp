// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/video/hevc_syntax.hpp"

#include <array>
#include <string>

#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

// The public calls return a host allocation failure (a list grown, a
// refusal's message) as OutOfMemory through core::exception_status: no
// exception leaves them.

namespace volumetric_kit::sensor::hevc {
namespace {

// @p what (a parameter set, a slice header) that does not parse, and why.
core::Status malformed(const char* what, const char* why) {
  return core::Status::io_error(std::string(what) +
                                " that does not parse: " + why);
}

// @p call handed a NAL unit that is not @p want.
core::Status wrong_type(const char* call, const char* want, const Nal& nal) {
  return core::Status::invalid_argument(std::string(call) + ": NAL unit type " +
                                        std::to_string(nal.type) + " is not " +
                                        want);
}

// profile_tier_level(1, max_sub_layers_minus1) (7.3.3).
void skip_profile_tier_level(BitReader& r,
                             std::uint32_t max_sub_layers_minus1) {
  r.skip(96);  // the general profile, tier and level
  bool profile[8] = {};
  bool level[8] = {};
  for (std::uint32_t i = 0; i < max_sub_layers_minus1; ++i) {
    profile[i] = r.flag();
    level[i] = r.flag();
  }
  if (max_sub_layers_minus1 > 0) {
    r.skip(2 * (8 - static_cast<int>(max_sub_layers_minus1)));
  }
  for (std::uint32_t i = 0; i < max_sub_layers_minus1; ++i) {
    if (profile[i]) r.skip(88);
    if (level[i]) r.skip(8);
  }
}

// scaling_list_data() (7.3.4).
bool skip_scaling_list_data(BitReader& r) {
  for (int size_id = 0; size_id < 4; ++size_id) {
    for (int matrix_id = 0; matrix_id < 6; matrix_id += size_id == 3 ? 3 : 1) {
      if (!r.flag()) {  // scaling_list_pred_mode_flag
        if (r.ue() > static_cast<std::uint32_t>(matrix_id)) return false;
        continue;
      }
      const int coefficients = size_id == 0 ? 16 : 64;
      if (size_id > 1) r.se();  // scaling_list_dc_coef_minus8
      for (int i = 0; i < coefficients; ++i) r.se();
    }
  }
  return !r.overrun();
}

// The delta POCs of one short-term reference picture set, as 7.4.8 derives
// them: the negative ones nearest first, then the positive ones.
struct ShortTermSet {
  int negative = 0;
  int positive = 0;
  std::array<std::int32_t, 16> s0{};
  std::array<std::int32_t, 16> s1{};
  int count() const noexcept { return negative + positive; }
};

// st_ref_pic_set(index) in an SPS (7.3.7), its deltas derived (7-61, 7-62)
// from the sets before it, which a set predicted from another needs.
bool parse_short_term_set(BitReader& r, std::uint32_t index,
                          std::vector<ShortTermSet>& sets) {
  ShortTermSet set;
  if (index != 0 && r.flag()) {  // inter_ref_pic_set_prediction_flag
    const bool sign = r.flag();
    const std::uint32_t abs_delta = r.ue() + 1;
    if (abs_delta > (1u << 15)) return false;
    const std::int32_t delta_rps = sign ? -static_cast<std::int32_t>(abs_delta)
                                        : static_cast<std::int32_t>(abs_delta);
    const ShortTermSet& ref = sets[index - 1];
    bool use[17] = {};
    for (int j = 0; j <= ref.count(); ++j) {
      const bool used_by_curr = r.flag();
      use[j] = used_by_curr || r.flag();  // use_delta_flag, 1 if absent
    }
    int i = 0;
    for (int j = ref.positive - 1; j >= 0; --j) {
      const std::int32_t d = ref.s1[static_cast<std::size_t>(j)] + delta_rps;
      if (d < 0 && use[ref.negative + j]) {
        if (i >= 16) return false;
        set.s0[static_cast<std::size_t>(i++)] = d;
      }
    }
    if (delta_rps < 0 && use[ref.count()]) {
      if (i >= 16) return false;
      set.s0[static_cast<std::size_t>(i++)] = delta_rps;
    }
    for (int j = 0; j < ref.negative; ++j) {
      const std::int32_t d = ref.s0[static_cast<std::size_t>(j)] + delta_rps;
      if (d < 0 && use[j]) {
        if (i >= 16) return false;
        set.s0[static_cast<std::size_t>(i++)] = d;
      }
    }
    set.negative = i;
    i = 0;
    for (int j = ref.negative - 1; j >= 0; --j) {
      const std::int32_t d = ref.s0[static_cast<std::size_t>(j)] + delta_rps;
      if (d > 0 && use[j]) {
        if (i >= 16) return false;
        set.s1[static_cast<std::size_t>(i++)] = d;
      }
    }
    if (delta_rps > 0 && use[ref.count()]) {
      if (i >= 16) return false;
      set.s1[static_cast<std::size_t>(i++)] = delta_rps;
    }
    for (int j = 0; j < ref.positive; ++j) {
      const std::int32_t d = ref.s1[static_cast<std::size_t>(j)] + delta_rps;
      if (d > 0 && use[ref.negative + j]) {
        if (i >= 16) return false;
        set.s1[static_cast<std::size_t>(i++)] = d;
      }
    }
    set.positive = i;
  } else {
    const std::uint32_t negative = r.ue();
    const std::uint32_t positive = r.ue();
    if (negative > 16 || positive > 16 - negative) return false;
    set.negative = static_cast<int>(negative);
    set.positive = static_cast<int>(positive);
    std::int32_t poc = 0;
    for (int i = 0; i < set.negative; ++i) {
      poc -= static_cast<std::int32_t>(r.ue() % (1u << 16)) + 1;
      set.s0[static_cast<std::size_t>(i)] = poc;
      r.skip(1);  // used_by_curr_pic_s0_flag
    }
    poc = 0;
    for (int i = 0; i < set.positive; ++i) {
      poc += static_cast<std::int32_t>(r.ue() % (1u << 16)) + 1;
      set.s1[static_cast<std::size_t>(i)] = poc;
      r.skip(1);  // used_by_curr_pic_s1_flag
    }
  }
  if (set.count() > 16 || r.overrun()) return false;
  sets.push_back(set);
  return true;
}

// sub_layer_hrd_parameters() (E.2.3).
void skip_sub_layer_hrd(BitReader& r, std::uint32_t cpb_count,
                        bool sub_pic_params) {
  for (std::uint32_t i = 0; i <= cpb_count; ++i) {
    r.ue();  // bit_rate_value_minus1
    r.ue();  // cpb_size_value_minus1
    if (sub_pic_params) {
      r.ue();  // cpb_size_du_value_minus1
      r.ue();  // bit_rate_du_value_minus1
    }
    r.skip(1);  // cbr_flag
  }
}

// hrd_parameters(1, max_sub_layers_minus1) (E.2.2).
bool skip_hrd_parameters(BitReader& r, std::uint32_t max_sub_layers_minus1) {
  const bool nal = r.flag();
  const bool vcl = r.flag();
  bool sub_pic_params = false;
  if (nal || vcl) {
    sub_pic_params = r.flag();
    if (sub_pic_params) r.skip(8 + 5 + 1 + 5);
    r.skip(4 + 4);  // bit_rate_scale, cpb_size_scale
    if (sub_pic_params) r.skip(4);
    r.skip(5 + 5 + 5);
  }
  for (std::uint32_t i = 0; i <= max_sub_layers_minus1; ++i) {
    const bool fixed_general = r.flag();
    const bool fixed_within_cvs = fixed_general || r.flag();
    bool low_delay = false;
    if (fixed_within_cvs) {
      r.ue();  // elemental_duration_in_tc_minus1
    } else {
      low_delay = r.flag();
    }
    std::uint32_t cpb_count = 0;
    if (!low_delay) {
      cpb_count = r.ue();
      if (cpb_count > 31) return false;
    }
    if (nal) skip_sub_layer_hrd(r, cpb_count, sub_pic_params);
    if (vcl) skip_sub_layer_hrd(r, cpb_count, sub_pic_params);
  }
  return !r.overrun();
}

// vui_parameters() (E.2.1): the colour into @p color, the rest read past.
bool parse_vui(BitReader& r, std::uint32_t max_sub_layers_minus1,
               ColorCodes& color) {
  if (r.flag()) {                      // aspect_ratio_info_present_flag
    if (r.bits(8) == 255) r.skip(32);  // EXTENDED_SAR: sar_width, sar_height
  }
  if (r.flag()) r.skip(1);  // overscan_info_present_flag
  if (r.flag()) {           // video_signal_type_present_flag
    r.skip(3);              // video_format
    color.full_range = r.flag();
    if (r.flag()) {  // colour_description_present_flag
      color.primaries = static_cast<int>(r.bits(8));
      color.transfer = static_cast<int>(r.bits(8));
      color.matrix = static_cast<int>(r.bits(8));
    }
  }
  if (r.flag()) {  // chroma_loc_info_present_flag
    const std::uint32_t top = r.ue();
    if (r.ue() > 5 || top > 5) return false;
    color.chroma_location = static_cast<int>(top);
  }
  r.skip(3);       // neutral_chroma_indication, field_seq, frame_field_info
  if (r.flag()) {  // default_display_window_flag
    for (int i = 0; i < 4; ++i) r.ue();
  }
  if (r.flag()) {          // vui_timing_info_present_flag
    r.skip(64);            // num_units_in_tick, time_scale
    if (r.flag()) r.ue();  // vui_num_ticks_poc_diff_one_minus1
    if (r.flag() && !skip_hrd_parameters(r, max_sub_layers_minus1)) {
      return false;
    }
  }
  if (r.flag()) {  // bitstream_restriction_flag
    r.skip(3);
    for (int i = 0; i < 5; ++i) r.ue();
  }
  return !r.overrun();
}

}  // namespace

core::Status split_annex_b(const std::uint8_t* data, std::size_t size,
                           std::vector<Nal>& out) try {
  out.clear();
  if (data == nullptr && size != 0) {
    return core::Status::invalid_argument(
        "split_annex_b: null data with a size");
  }
  std::size_t begin = 0;  // where the NAL unit being found starts, if any
  bool in_nal = false;
  const auto close = [&](std::size_t end) {
    while (end > begin && data[end - 1] == 0) --end;  // trailing zeros
    if (end - begin < 2) return false;
    Nal nal;
    nal.data = data + begin;
    nal.size = end - begin;
    if ((data[begin] & 0x80) != 0) return false;  // forbidden_zero_bit
    nal.type = (data[begin] >> 1) & 0x3f;
    nal.layer_id = ((data[begin] & 1) << 5) | (data[begin + 1] >> 3);
    const int temporal_id_plus1 = data[begin + 1] & 7;
    if (temporal_id_plus1 == 0) return false;
    nal.temporal_id = temporal_id_plus1 - 1;
    out.push_back(nal);
    return true;
  };
  const auto bad_header = [] {
    return core::Status::io_error(
        "Annex B bytes with a NAL unit header that is not valid");
  };
  // A byte above 1 starts no start code at itself or the two before it, so
  // the scan steps past all three.
  for (std::size_t i = 0; i + 3 <= size;) {
    if (data[i + 2] > 1) {
      i += 3;
    } else if (data[i + 2] == 1 && data[i] == 0 && data[i + 1] == 0) {
      if (in_nal && !close(i)) return bad_header();
      begin = i + 3;
      in_nal = true;
      i += 3;
    } else {
      ++i;
    }
  }
  if (!in_nal) {
    return core::Status::io_error("Annex B bytes with no start code");
  }
  if (!close(size)) return bad_header();
  return {};
} catch (...) {
  return core::exception_status("split_annex_b", core::Status::Code::IoError);
}

BitReader::BitReader(const Nal& nal) noexcept
    : data_(nal.size > 2 ? nal.data + 2 : nullptr),
      size_(nal.size > 2 ? nal.size - 2 : 0) {}

std::uint32_t BitReader::bit() {
  if (left_ == 0 && !load()) {
    overrun_ = true;
    return 0;
  }
  --left_;
  return (byte_ >> left_) & 1u;
}

bool BitReader::load() {
  if (pos_ >= size_) return false;
  std::uint8_t b = data_[pos_++];
  if (zeros_ >= 2 && b == 3) {  // the 03 of 00 00 03
    zeros_ = 0;
    if (pos_ >= size_) return false;
    b = data_[pos_++];
  }
  zeros_ = b == 0 ? zeros_ + 1 : 0;
  byte_ = b;
  left_ = 8;
  return true;
}

std::uint32_t BitReader::bits(int n) {
  VKC_CHECK(n >= 0 && n <= 32, "BitReader::bits reads 0 to 32 bits");
  std::uint32_t value = 0;
  for (; n > 0; --n) value = (value << 1) | bit();
  return value;
}

bool BitReader::flag() { return bit() != 0; }

void BitReader::skip(int n) {
  for (; n > 0; --n) bit();
}

std::uint32_t BitReader::ue() {
  int zeros = 0;
  while (bit() == 0) {
    if (overrun_ || ++zeros > 31) {
      overrun_ = true;
      return 0;
    }
  }
  return static_cast<std::uint32_t>((std::uint64_t{1} << zeros) - 1 +
                                    bits(zeros));
}

std::int32_t BitReader::se() {
  const std::uint32_t k = ue();
  return (k & 1u) != 0 ? static_cast<std::int32_t>((k + 1) / 2)
                       : -static_cast<std::int32_t>(k / 2);
}

bool BitReader::at_trailing_bits() {
  if (bit() != 1 || overrun_) return false;
  while (left_ > 0 || load()) {
    if (bit() != 0) return false;
  }
  return true;
}

core::Result<std::uint32_t> vps_id(const Nal& vps) try {
  if (vps.type != kVps) return wrong_type("vps_id", "a VPS", vps);
  BitReader r(vps);
  const std::uint32_t id = r.bits(4);
  if (r.overrun()) return malformed("a VPS", "cut short");
  return id;
} catch (...) {
  return core::exception_status("vps_id", core::Status::Code::IoError);
}

core::Result<Sps> parse_sps(const Nal& nal) try {
  if (nal.type != kSps) return wrong_type("parse_sps", "an SPS", nal);
  BitReader r(nal);
  // Why the SPS is refused: cut short, if a read went past its end, since
  // the values read there are zeros.
  const auto bad = [&r](const char* why) {
    return malformed("an SPS", r.overrun() ? "cut short" : why);
  };
  Sps sps;
  sps.vps_id = r.bits(4);
  const std::uint32_t max_sub_layers_minus1 = r.bits(3);
  if (max_sub_layers_minus1 > 6) return bad("more than 7 sub-layers");
  r.skip(1);  // sps_temporal_id_nesting_flag
  skip_profile_tier_level(r, max_sub_layers_minus1);
  sps.id = r.ue();
  sps.chroma_format_idc = r.ue();
  if (sps.id > 15) return bad("an id past 15");
  if (sps.chroma_format_idc > 3) return bad("a chroma format past 3");
  if (sps.chroma_format_idc == 3) sps.separate_colour_plane = r.flag();
  sps.coded_width = r.ue();
  sps.coded_height = r.ue();
  if (sps.coded_width == 0 || sps.coded_height == 0 ||
      sps.coded_width > 16888 || sps.coded_height > 16888) {
    return bad("a picture size of 0 or past 16888");
  }
  if (r.flag()) {  // conformance_window_flag, offsets in chroma samples
    const bool chroma =
        sps.chroma_format_idc != 0 && !sps.separate_colour_plane;
    const std::uint32_t sub_width =
        chroma && sps.chroma_format_idc != 3 ? 2 : 1;
    const std::uint32_t sub_height =
        chroma && sps.chroma_format_idc == 1 ? 2 : 1;
    const std::uint32_t left = r.ue();
    const std::uint32_t right = r.ue();
    const std::uint32_t top = r.ue();
    const std::uint32_t bottom = r.ue();
    if (left > sps.coded_width || right > sps.coded_width ||
        top > sps.coded_height || bottom > sps.coded_height) {
      return bad("a conformance window that leaves no picture");
    }
    sps.crop_left = sub_width * left;
    sps.crop_right = sub_width * right;
    sps.crop_top = sub_height * top;
    sps.crop_bottom = sub_height * bottom;
    if (sps.crop_left + sps.crop_right >= sps.coded_width ||
        sps.crop_top + sps.crop_bottom >= sps.coded_height) {
      return bad("a conformance window that leaves no picture");
    }
  }
  sps.bit_depth_luma = r.ue() + 8;
  sps.bit_depth_chroma = r.ue() + 8;
  const std::uint32_t log2_max_poc_lsb = r.ue() + 4;
  if (sps.bit_depth_luma > 16 || sps.bit_depth_chroma > 16) {
    return bad("a bit depth past 16");
  }
  if (log2_max_poc_lsb > 16) return bad("a POC LSB of more than 16 bits");
  sps.log2_max_poc_lsb = log2_max_poc_lsb;
  const bool ordering_per_sub_layer = r.flag();
  for (std::uint32_t i = ordering_per_sub_layer ? 0 : max_sub_layers_minus1;
       i <= max_sub_layers_minus1; ++i) {
    const std::uint32_t buffering = r.ue() + 1;
    sps.max_num_reorder = r.ue();
    r.ue();  // sps_max_latency_increase_plus1
    if (buffering > 16 || sps.max_num_reorder >= buffering) {
      return bad("a buffer past 16 pictures, or a reorder depth not below it");
    }
  }
  for (int i = 0; i < 6; ++i) r.ue();  // block sizes, transform depths
  if (r.flag() && r.flag() && !skip_scaling_list_data(r)) {
    return bad("its scaling lists");  // scaling_list_enabled, sps_..._data
  }
  r.skip(2);       // amp_enabled_flag, sample_adaptive_offset_enabled_flag
  if (r.flag()) {  // pcm_enabled_flag
    r.skip(8);
    r.ue();
    r.ue();
    r.skip(1);
  }
  const std::uint32_t short_term_sets = r.ue();
  if (short_term_sets > 64) return bad("more than 64 short-term sets");
  std::vector<ShortTermSet> sets;
  for (std::uint32_t i = 0; i < short_term_sets; ++i) {
    if (!parse_short_term_set(r, i, sets)) return bad("a short-term set");
  }
  if (r.flag()) {  // long_term_ref_pics_present_flag
    const std::uint32_t long_term = r.ue();
    if (long_term > 32) return bad("more than 32 long-term pictures");
    for (std::uint32_t i = 0; i < long_term; ++i) {
      r.skip(static_cast<int>(log2_max_poc_lsb) + 1);
    }
  }
  r.skip(2);  // sps_temporal_mvp_enabled, strong_intra_smoothing_enabled
  if (r.flag() && !parse_vui(r, max_sub_layers_minus1, sps.color)) {
    return bad("its VUI");
  }
  if (r.flag()) {  // sps_extension_present_flag
    const bool range = r.flag();
    const bool multilayer = r.flag();
    const bool three_d = r.flag();
    const bool scc = r.flag();
    const bool more = r.bits(4) != 0;
    if (range) r.skip(9);
    if (multilayer) r.skip(1);
    // Extensions this reader does not walk: the rest is taken as it is.
    if (three_d || scc || more) {
      if (r.overrun()) return bad("cut short");
      return sps;
    }
  }
  if (r.overrun() || !r.at_trailing_bits()) {
    return bad("more than its trailing bits after it");
  }
  return sps;
} catch (...) {
  return core::exception_status("parse_sps", core::Status::Code::IoError);
}

core::Result<Pps> parse_pps(const Nal& nal) try {
  if (nal.type != kPps) return wrong_type("parse_pps", "a PPS", nal);
  BitReader r(nal);
  Pps pps;
  pps.id = r.ue();
  pps.sps_id = r.ue();
  r.skip(1);  // dependent_slice_segments_enabled_flag
  pps.output_flag_present = r.flag();
  pps.num_extra_slice_header_bits = r.bits(3);
  if (r.overrun()) return malformed("a PPS", "cut short");
  if (pps.id > 63) return malformed("a PPS", "an id past 63");
  if (pps.sps_id > 15) return malformed("a PPS", "an SPS id past 15");
  return pps;
} catch (...) {
  return core::exception_status("parse_pps", core::Status::Code::IoError);
}

core::Result<SliceStart> parse_slice_start(
    const Nal& nal, const std::map<std::uint32_t, Pps>& pps,
    const std::map<std::uint32_t, Sps>& sps) try {
  if (nal.type < 0 || nal.type > 31) {
    return wrong_type("parse_slice_start", "a VCL NAL unit", nal);
  }
  BitReader r(nal);
  SliceStart slice;
  slice.first_slice_segment = r.flag();
  if (is_irap(nal.type)) slice.no_output_of_prior_pics = r.flag();
  slice.pps_id = r.ue();
  if (r.overrun()) return malformed("a slice header", "cut short");
  const auto p = pps.find(slice.pps_id);
  if (p == pps.end()) {
    return core::Status::io_error("a slice header naming PPS " +
                                  std::to_string(slice.pps_id) +
                                  ", which is not given");
  }
  const auto s = sps.find(p->second.sps_id);
  if (s == sps.end()) {
    return core::Status::io_error("a slice header whose PPS " +
                                  std::to_string(slice.pps_id) + " names SPS " +
                                  std::to_string(p->second.sps_id) +
                                  ", which is not given");
  }
  if (!slice.first_slice_segment) return slice;
  r.skip(static_cast<int>(p->second.num_extra_slice_header_bits));
  const std::uint32_t slice_type = r.ue();
  if (p->second.output_flag_present) slice.pic_output = r.flag();
  if (s->second.separate_colour_plane) r.skip(2);  // colour_plane_id
  if (!is_idr(nal.type)) {
    slice.poc_lsb = r.bits(static_cast<int>(s->second.log2_max_poc_lsb));
  }
  if (r.overrun()) return malformed("a slice header", "cut short");
  if (slice_type > 2) return malformed("a slice header", "a slice type past 2");
  return slice;
} catch (...) {
  return core::exception_status("parse_slice_start",
                                core::Status::Code::IoError);
}

}  // namespace volumetric_kit::sensor::hevc
