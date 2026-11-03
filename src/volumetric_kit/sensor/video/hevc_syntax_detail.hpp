// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// The walks under hevc_syntax.hpp's parsers, shared with the test-only slice
// reader (testing/hevc_slices.hpp): each parser reads its structure through
// one of these, which records, where it is handed a tail, what only the
// slice reader needs. The decode path hands none, so what it reads and
// refuses is as before. Internal to sensor_video_syntax.

#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/hevc_syntax.hpp"

namespace volumetric_kit::sensor::hevc::detail {

// One short-term reference picture set's delta POCs (7.4.8), the negative
// ones nearest first, then the positive ones, and whether the picture that
// uses the set refers to each.
struct ShortTermSet {
  int negative = 0;
  int positive = 0;
  std::array<std::int32_t, 16> s0{};
  std::array<std::int32_t, 16> s1{};
  std::array<bool, 16> used0{};
  std::array<bool, 16> used1{};
  int count() const noexcept { return negative + positive; }
  // The pictures it holds that the current picture refers to.
  int used() const noexcept;
};

// st_ref_pic_set(index) (7.3.7) into @p set: one of an SPS's @p count sets
// (index below count), or a slice header's own (index == count), predicted
// from @p sets where it says so. False where it does not parse.
bool read_short_term_set(BitReader& r, std::uint32_t index, std::uint32_t count,
                         const std::vector<ShortTermSet>& sets,
                         ShortTermSet& set);

// What an SPS says past hevc::Sps that a slice header's walk needs.
struct SpsTail {
  int profile_idc = 0;                      // general_profile_idc
  std::uint32_t max_sub_layers_minus1 = 0;  // sps_max_sub_layers_minus1
  std::uint32_t log2_ctb = 4;               // CtbLog2SizeY
  bool sao = false;  // sample_adaptive_offset_enabled_flag
  std::vector<ShortTermSet> short_term;
  bool long_term = false;            // long_term_ref_pics_present_flag
  std::vector<bool> long_term_used;  // used_by_curr_pic_lt_sps_flag
  bool temporal_mvp = false;         // sps_temporal_mvp_enabled_flag
};

// What a PPS says past hevc::Pps, through lists_modification_present_flag.
struct PpsTail {
  bool dependent_slices = false;
  bool cabac_init_present = false;
  std::uint32_t num_ref_idx_l0_default = 1;
  std::uint32_t num_ref_idx_l1_default = 1;
  int init_qp = 26;  // 26 + init_qp_minus26
  bool weighted_pred = false;
  bool weighted_bipred = false;
  bool lists_modification = false;
};

// parse_sps, and the tail into @p tail where it is not null.
core::Result<Sps> read_sps(const Nal& nal, SpsTail* tail);
// parse_pps, read on through the tail into @p tail where it is not null.
core::Result<Pps> read_pps(const Nal& nal, PpsTail* tail);

// The start of a slice segment header, and its slice_type.
struct SliceHead {
  SliceStart start;
  std::uint32_t slice_type = 0;
  const Pps* pps = nullptr;  // the PPS it names, in the map it was read with
  const Sps* sps = nullptr;  // that PPS's SPS, likewise
};

// parse_slice_start, read with @p r, which it leaves after the fields it
// read: after slice_pic_order_cnt_lsb of a picture's first segment.
core::Result<SliceHead> read_slice_start(
    BitReader& r, const Nal& nal, const std::map<std::uint32_t, Pps>& pps,
    const std::map<std::uint32_t, Sps>& sps);

}  // namespace volumetric_kit::sensor::hevc::detail
