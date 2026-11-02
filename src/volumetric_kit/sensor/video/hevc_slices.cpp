// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/testing/hevc_slices.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "hevc_syntax_detail.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::sensor::testing {
namespace {

using hevc::BitReader;

constexpr int kSliceB = 0;
constexpr int kSliceP = 1;

core::Status malformed(const std::string& why) {
  return core::Status::io_error(
      "SliceReader: a slice header that does not "
      "parse: " +
      why);
}

// Ceil(Log2(n)): the bits of a u(v) index into n entries.
int bits_for(std::uint64_t n) {
  int bits = 0;
  while ((std::uint64_t{1} << static_cast<unsigned>(bits)) < n) ++bits;
  return bits;
}

// pred_weight_table() (7.3.6.3) over @p l0 and @p l1 active references; in
// these profiles no reference shares the picture's POC, so each has its
// flags.
void skip_pred_weight_table(BitReader& r, bool chroma, std::uint32_t l0,
                            std::uint32_t l1) {
  r.ue();  // luma_log2_weight_denom
  if (chroma) r.se();
  for (const std::uint32_t refs : {l0, l1}) {
    std::vector<bool> luma(refs);
    std::vector<bool> colour(refs);
    for (std::uint32_t i = 0; i < refs; ++i) luma[i] = r.flag();
    if (chroma) {
      for (std::uint32_t i = 0; i < refs; ++i) colour[i] = r.flag();
    }
    for (std::uint32_t i = 0; i < refs; ++i) {
      if (luma[i]) {
        r.se();
        r.se();
      }
      if (colour[i]) {
        for (int k = 0; k < 4; ++k) r.se();
      }
    }
  }
}

}  // namespace

struct SliceReader::Impl {
  std::map<std::uint32_t, hevc::Sps> sps;
  std::map<std::uint32_t, hevc::detail::SpsTail> sps_tail;
  std::map<std::uint32_t, hevc::Pps> pps;
  std::map<std::uint32_t, hevc::detail::PpsTail> pps_tail;
  std::optional<Sequence> sequence;
  std::vector<hevc::Nal> nals;

  // The rest of a first slice segment's header (7.3.6.1), on from @p head.
  core::Result<int> slice_qp(BitReader& r, const hevc::Nal& nal,
                             const hevc::detail::SliceHead& head) const;
};

core::Result<int> SliceReader::Impl::slice_qp(
    BitReader& r, const hevc::Nal& nal,
    const hevc::detail::SliceHead& head) const {
  const hevc::Sps& s = *head.sps;
  const hevc::detail::SpsTail& st = sps_tail.at(s.id);
  const hevc::detail::PpsTail& pt = pps_tail.at(head.start.pps_id);
  const int type = static_cast<int>(head.slice_type);
  int used = 0;  // NumPicTotalCurr
  bool slice_tmvp = false;
  if (!hevc::is_idr(nal.type)) {
    const auto sets = static_cast<std::uint32_t>(st.short_term.size());
    if (!r.flag()) {  // short_term_ref_pic_set_sps_flag
      hevc::detail::ShortTermSet own;
      if (!hevc::detail::read_short_term_set(r, sets, sets, st.short_term,
                                             own)) {
        return malformed("its short-term set");
      }
      used = own.used();
    } else {
      const std::uint32_t idx = sets > 1 ? r.bits(bits_for(sets)) : 0;
      if (idx >= sets) return malformed("a short-term set the SPS lacks");
      used = st.short_term[idx].used();
    }
    if (st.long_term) {
      const auto in_sps = static_cast<std::uint32_t>(st.long_term_used.size());
      const std::uint32_t from_sps = in_sps > 0 ? r.ue() : 0;
      const std::uint32_t own = r.ue();
      if (from_sps > in_sps || own > 32) {
        return malformed("too many long-term pictures");
      }
      for (std::uint32_t i = 0; i < from_sps + own; ++i) {
        if (i < from_sps) {
          const std::uint32_t idx = in_sps > 1 ? r.bits(bits_for(in_sps)) : 0;
          if (idx >= in_sps) return malformed("a long-term picture none");
          used += st.long_term_used[idx] ? 1 : 0;
        } else {
          r.skip(static_cast<int>(s.log2_max_poc_lsb));  // poc_lsb_lt
          used += r.flag() ? 1 : 0;  // used_by_curr_pic_lt_flag
        }
        if (r.flag()) r.ue();  // delta_poc_msb_cycle_lt
      }
    }
    if (st.temporal_mvp) slice_tmvp = r.flag();
  }
  const bool chroma = !s.separate_colour_plane && s.chroma_format_idc != 0;
  if (st.sao) {
    r.skip(1);  // slice_sao_luma_flag
    if (chroma) r.skip(1);
  }
  if (type == kSliceP || type == kSliceB) {
    std::uint32_t l0 = pt.num_ref_idx_l0_default;
    std::uint32_t l1 = pt.num_ref_idx_l1_default;
    if (r.flag()) {  // num_ref_idx_active_override_flag
      l0 = r.ue() + 1;
      if (type == kSliceB) l1 = r.ue() + 1;
    }
    if (l0 > 16 || l1 > 16) return malformed("more than 16 references");
    if (pt.lists_modification && used > 1) {
      const int bits = bits_for(static_cast<std::uint64_t>(used));
      if (r.flag()) r.skip(bits * static_cast<int>(l0));
      if (type == kSliceB && r.flag()) r.skip(bits * static_cast<int>(l1));
    }
    if (type == kSliceB) r.skip(1);  // mvd_l1_zero_flag
    if (pt.cabac_init_present) r.skip(1);
    if (slice_tmvp) {
      const bool from_l0 = type == kSliceB ? r.flag() : true;
      if ((from_l0 && l0 > 1) || (!from_l0 && l1 > 1)) r.ue();
    }
    if ((pt.weighted_pred && type == kSliceP) ||
        (pt.weighted_bipred && type == kSliceB)) {
      skip_pred_weight_table(r, chroma, l0, type == kSliceB ? l1 : 0);
    }
    r.ue();  // five_minus_max_num_merge_cand
  }
  const int qp = pt.init_qp + r.se();
  if (r.overrun()) return malformed("cut short");
  return qp;
}

SliceReader::SliceReader() : impl_(std::make_unique<Impl>()) {}
SliceReader::~SliceReader() = default;

core::Result<SliceReader::Picture> SliceReader::read(const std::uint8_t* data,
                                                     std::size_t size) try {
  Impl& m = *impl_;
  VKC_TRY(hevc::split_annex_b(data, size, m.nals).with_context("SliceReader"));
  Picture picture;
  const hevc::Nal* slice = nullptr;
  for (const hevc::Nal& nal : m.nals) {
    picture.max_temporal_id =
        std::max(picture.max_temporal_id, nal.temporal_id);
    if (nal.type == hevc::kSps) {
      hevc::detail::SpsTail tail;
      VKC_ASSIGN(const hevc::Sps sps, hevc::detail::read_sps(nal, &tail));
      m.sequence = Sequence{sps, tail.profile_idc,
                            static_cast<int>(tail.max_sub_layers_minus1) + 1};
      m.sps[sps.id] = sps;
      m.sps_tail[sps.id] = std::move(tail);
    } else if (nal.type == hevc::kPps) {
      hevc::detail::PpsTail tail;
      VKC_ASSIGN(const hevc::Pps pps, hevc::detail::read_pps(nal, &tail));
      m.pps[pps.id] = pps;
      m.pps_tail[pps.id] = tail;
    } else if (nal.type < 32 && slice == nullptr) {
      slice = &nal;
    }
  }
  if (slice == nullptr) {
    return core::Status::io_error("SliceReader: an access unit with no slice");
  }
  BitReader r(*slice);
  VKC_ASSIGN(const hevc::detail::SliceHead head,
             hevc::detail::read_slice_start(r, *slice, m.pps, m.sps));
  if (!head.start.first_slice_segment) {
    return core::Status::io_error(
        "SliceReader: an access unit that does not start its picture");
  }
  picture.nal_type = slice->type;
  picture.slice_type = static_cast<int>(head.slice_type);
  VKC_ASSIGN(picture.slice_qp, m.slice_qp(r, *slice, head));
  return picture;
} catch (...) {
  return core::exception_status("SliceReader::read",
                                core::Status::Code::IoError);
}

const std::optional<SliceReader::Sequence>& SliceReader::sequence()
    const noexcept {
  return impl_->sequence;
}

}  // namespace volumetric_kit::sensor::testing
