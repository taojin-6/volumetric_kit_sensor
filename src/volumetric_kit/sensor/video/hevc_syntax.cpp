// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "hevc_syntax.hpp"

#include <array>

namespace volumetric_kit::recon::sensor::video::hevc {
namespace {

// A NAL unit's payload bits, MSB first, emulation prevention bytes skipped as
// they are read. A read past the end reads zeros and sets overrun().
class BitReader {
 public:
  BitReader(const std::uint8_t* payload, std::size_t size)
      : data_(payload), size_(size) {}

  std::uint32_t bit() {
    if (left_ == 0 && !load()) {
      overrun_ = true;
      return 0;
    }
    --left_;
    return (byte_ >> left_) & 1u;
  }
  std::uint32_t bits(int n) {
    std::uint32_t value = 0;
    for (; n > 0; --n) value = (value << 1) | bit();
    return value;
  }
  void skip(int n) {
    for (; n > 0; --n) bit();
  }
  bool flag() { return bit() != 0; }
  std::uint32_t ue() {  // Exp-Golomb, at most 2^32 - 2
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
  std::int32_t se() {
    const std::uint32_t k = ue();
    return (k & 1u) != 0 ? static_cast<std::int32_t>((k + 1) / 2)
                         : -static_cast<std::int32_t>(k / 2);
  }
  bool overrun() const noexcept { return overrun_; }
  // Whether what is left is rbsp_trailing_bits: a one, then zeros only.
  bool at_trailing_bits() {
    if (bit() != 1 || overrun_) return false;
    while (left_ > 0 || load()) {
      if (bit() != 0) return false;
    }
    return true;
  }

 private:
  bool load() {
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

  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
  int zeros_ = 0;
  std::uint8_t byte_ = 0;
  int left_ = 0;
  bool overrun_ = false;
};

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

bool split_annex_b(const std::uint8_t* data, std::size_t size,
                   std::vector<Nal>& out) {
  out.clear();
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
  // A byte above 1 starts no start code at itself or the two before it, so
  // the scan steps past all three.
  for (std::size_t i = 0; i + 3 <= size;) {
    if (data[i + 2] > 1) {
      i += 3;
    } else if (data[i + 2] == 1 && data[i] == 0 && data[i + 1] == 0) {
      if (in_nal && !close(i)) return false;
      begin = i + 3;
      in_nal = true;
      i += 3;
    } else {
      ++i;
    }
  }
  return in_nal && close(size);
}

std::optional<std::uint32_t> vps_id(const Nal& vps) {
  if (vps.size < 3) return std::nullopt;
  return static_cast<std::uint32_t>(vps.data[2] >> 4);
}

std::optional<Sps> parse_sps(const Nal& nal) {
  BitReader r(nal.data + 2, nal.size - 2);
  Sps sps;
  sps.vps_id = r.bits(4);
  const std::uint32_t max_sub_layers_minus1 = r.bits(3);
  if (max_sub_layers_minus1 > 6) return std::nullopt;
  r.skip(1);  // sps_temporal_id_nesting_flag
  skip_profile_tier_level(r, max_sub_layers_minus1);
  sps.id = r.ue();
  sps.chroma_format_idc = r.ue();
  if (sps.id > 15 || sps.chroma_format_idc > 3) return std::nullopt;
  if (sps.chroma_format_idc == 3) sps.separate_colour_plane = r.flag();
  sps.coded_width = r.ue();
  sps.coded_height = r.ue();
  if (sps.coded_width == 0 || sps.coded_height == 0 ||
      sps.coded_width > 16888 || sps.coded_height > 16888) {
    return std::nullopt;
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
      return std::nullopt;
    }
    sps.crop_left = sub_width * left;
    sps.crop_right = sub_width * right;
    sps.crop_top = sub_height * top;
    sps.crop_bottom = sub_height * bottom;
    if (sps.crop_left + sps.crop_right >= sps.coded_width ||
        sps.crop_top + sps.crop_bottom >= sps.coded_height) {
      return std::nullopt;
    }
  }
  sps.bit_depth_luma = r.ue() + 8;
  sps.bit_depth_chroma = r.ue() + 8;
  const std::uint32_t log2_max_poc_lsb = r.ue() + 4;
  if (sps.bit_depth_luma > 16 || sps.bit_depth_chroma > 16 ||
      log2_max_poc_lsb > 16) {
    return std::nullopt;
  }
  sps.log2_max_poc_lsb = log2_max_poc_lsb;
  const bool ordering_per_sub_layer = r.flag();
  for (std::uint32_t i = ordering_per_sub_layer ? 0 : max_sub_layers_minus1;
       i <= max_sub_layers_minus1; ++i) {
    const std::uint32_t buffering = r.ue() + 1;
    sps.max_num_reorder = r.ue();
    r.ue();  // sps_max_latency_increase_plus1
    if (buffering > 16 || sps.max_num_reorder >= buffering) {
      return std::nullopt;
    }
  }
  for (int i = 0; i < 6; ++i) r.ue();  // block sizes, transform depths
  if (r.flag() && r.flag() && !skip_scaling_list_data(r)) {
    return std::nullopt;  // scaling_list_enabled, sps_scaling_list_data
  }
  r.skip(2);       // amp_enabled_flag, sample_adaptive_offset_enabled_flag
  if (r.flag()) {  // pcm_enabled_flag
    r.skip(8);
    r.ue();
    r.ue();
    r.skip(1);
  }
  const std::uint32_t short_term_sets = r.ue();
  if (short_term_sets > 64) return std::nullopt;
  std::vector<ShortTermSet> sets;
  for (std::uint32_t i = 0; i < short_term_sets; ++i) {
    if (!parse_short_term_set(r, i, sets)) return std::nullopt;
  }
  if (r.flag()) {  // long_term_ref_pics_present_flag
    const std::uint32_t long_term = r.ue();
    if (long_term > 32) return std::nullopt;
    for (std::uint32_t i = 0; i < long_term; ++i) {
      r.skip(static_cast<int>(log2_max_poc_lsb) + 1);
    }
  }
  r.skip(2);  // sps_temporal_mvp_enabled, strong_intra_smoothing_enabled
  if (r.flag() && !parse_vui(r, max_sub_layers_minus1, sps.color)) {
    return std::nullopt;
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
      if (r.overrun()) return std::nullopt;
      return sps;
    }
  }
  if (r.overrun() || !r.at_trailing_bits()) return std::nullopt;
  return sps;
}

std::optional<Pps> parse_pps(const Nal& nal) {
  BitReader r(nal.data + 2, nal.size - 2);
  Pps pps;
  pps.id = r.ue();
  pps.sps_id = r.ue();
  pps.dependent_slice_segments_enabled = r.flag();
  pps.output_flag_present = r.flag();
  pps.num_extra_slice_header_bits = r.bits(3);
  if (r.overrun() || pps.id > 63 || pps.sps_id > 15) return std::nullopt;
  return pps;
}

std::optional<SliceStart> parse_slice_start(
    const Nal& nal, const std::map<std::uint32_t, Pps>& pps,
    const std::map<std::uint32_t, Sps>& sps) {
  BitReader r(nal.data + 2, nal.size - 2);
  SliceStart slice;
  slice.first_slice_segment = r.flag();
  if (is_irap(nal.type)) slice.no_output_of_prior_pics = r.flag();
  slice.pps_id = r.ue();
  const auto p = pps.find(slice.pps_id);
  if (r.overrun() || p == pps.end()) return std::nullopt;
  const auto s = sps.find(p->second.sps_id);
  if (s == sps.end()) return std::nullopt;
  if (!slice.first_slice_segment) return slice;
  r.skip(static_cast<int>(p->second.num_extra_slice_header_bits));
  if (r.ue() > 2) return std::nullopt;  // slice_type
  if (p->second.output_flag_present) slice.pic_output = r.flag();
  if (s->second.separate_colour_plane) r.skip(2);  // colour_plane_id
  if (!is_idr(nal.type)) {
    slice.poc_lsb = r.bits(static_cast<int>(s->second.log2_max_poc_lsb));
  }
  if (r.overrun()) return std::nullopt;
  return slice;
}

}  // namespace volumetric_kit::recon::sensor::video::hevc
