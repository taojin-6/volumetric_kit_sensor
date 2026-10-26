// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The public H.265 syntax (sensor/video/hevc_syntax.hpp), on bytes made here
// and on the committed clips (tools/make_hevc_fixtures.sh), with no codec and
// no GPU, so it runs on every leg: the bit reader (the RBSP's bits past the
// emulation prevention bytes, Exp-Golomb codes, reads past the end, the
// trailing bits); the NAL units an access unit splits into; what each clip's
// SPS says (size, crop, format, the VUI's colour, the reorder depth); the
// picture order counts of its slices; what is refused -- every truncated
// SPS, a slice whose parameter sets were never given, a NAL unit of another
// type, and noise -- without a read out of bounds; and a host allocation
// failure in each call, returned as OutOfMemory.

#include "volumetric_kit/recon/sensor/video/hevc_syntax.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "hevc_fixture.hpp"
#include "test_allocation_failure.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/base/result.hpp"

namespace hevc = volumetric_kit::recon::sensor::hevc;
namespace vkc = volumetric_kit::core;
using Code = vkc::Status::Code;

namespace {

using Units = std::vector<std::vector<std::uint8_t>>;

std::string clip(const char* name) {
  return std::string(VR_HEVC_DATA) + "/" + name + ".h265";
}

// @p payload behind a two-byte header of NAL unit @p type, in @p storage.
hevc::Nal unit(std::vector<std::uint8_t>& storage, int type,
               std::initializer_list<std::uint8_t> payload) {
  storage = {static_cast<std::uint8_t>(type << 1), 0x01};
  storage.insert(storage.end(), payload);
  hevc::Nal nal;
  nal.data = storage.data();
  nal.size = storage.size();
  nal.type = type;
  return nal;
}

// A clip read as the decoder reads it: its parameter sets kept by id, and
// the POC LSB of each access unit's picture.
struct Read {
  std::vector<hevc::Sps> sps;  // each, in order
  std::map<std::uint32_t, hevc::Sps> sps_by_id;
  std::map<std::uint32_t, hevc::Pps> pps_by_id;
  std::vector<std::uint32_t> poc_lsb;
  std::vector<int> picture_types;
  bool ok = true;
};

Read read_clip(const char* name) {
  Read out;
  const Units units = vr_test::hevc_access_units(clip(name).c_str());
  out.ok = !units.empty();
  std::vector<hevc::Nal> nals;
  for (const auto& unit : units) {
    if (!hevc::split_annex_b(unit.data(), unit.size(), nals).ok()) {
      out.ok = false;
    }
    bool picture = false;
    for (const hevc::Nal& nal : nals) {
      if (nal.type == hevc::kSps) {
        const vkc::Result<hevc::Sps> sps = hevc::parse_sps(nal);
        if (!sps) {
          out.ok = false;
          continue;
        }
        out.sps.push_back(*sps);
        out.sps_by_id[sps->id] = *sps;
      } else if (nal.type == hevc::kPps) {
        const vkc::Result<hevc::Pps> pps = hevc::parse_pps(nal);
        if (!pps) {
          out.ok = false;
          continue;
        }
        out.pps_by_id[pps->id] = *pps;
      } else if (nal.type < 32 && !picture) {
        const vkc::Result<hevc::SliceStart> slice =
            hevc::parse_slice_start(nal, out.pps_by_id, out.sps_by_id);
        if (!slice || !slice->first_slice_segment) {
          out.ok = false;
          continue;
        }
        picture = true;
        out.poc_lsb.push_back(slice->poc_lsb);
        out.picture_types.push_back(nal.type);
      }
    }
    if (!picture) out.ok = false;
  }
  return out;
}

int check_size(const hevc::Sps& s, std::uint32_t coded_width,
               std::uint32_t coded_height, std::uint32_t width,
               std::uint32_t height) {
  CHECK(s.coded_width == coded_width && s.coded_height == coded_height);
  CHECK(s.width() == width && s.height() == height);
  return 0;
}

// Bits most significant first, from the byte after the header; the 03 of
// 00 00 03 skipped, and only there; zeros read past the end, which sets
// overrun; a copy reads on from where the original was.
int test_bits() {
  std::vector<std::uint8_t> bytes;
  hevc::BitReader r(unit(bytes, hevc::kSps, {0xa5, 0x0f, 0x80}));
  CHECK(r.bits(4) == 0xa && r.bits(4) == 0x5);
  hevc::BitReader copy = r;
  CHECK(r.bits(8) == 0x0f && copy.bits(8) == 0x0f);
  CHECK(r.flag() && !r.flag());
  r.skip(6);
  CHECK(!r.overrun());
  CHECK(r.bits(3) == 0 && r.overrun());

  hevc::BitReader word(unit(bytes, hevc::kSps, {0x12, 0x34, 0x56, 0x78}));
  CHECK(word.bits(32) == 0x12345678u && !word.overrun());
  CHECK(word.bits(0) == 0 && !word.overrun());

  // RBSP 00 00 01: the 03 goes.
  hevc::BitReader prevented(unit(bytes, hevc::kSps, {0, 0, 3, 1}));
  CHECK(prevented.bits(24) == 1 && !prevented.overrun());
  CHECK(prevented.bits(1) == 0 && prevented.overrun());
  // A 03 after one zero stays; a zero after a skipped 03 starts the count.
  hevc::BitReader kept(unit(bytes, hevc::kSps, {0, 3, 0}));
  CHECK(kept.bits(24) == 0x000300 && !kept.overrun());
  hevc::BitReader twice(unit(bytes, hevc::kSps, {0, 0, 3, 0, 0, 3, 0, 1}));
  twice.skip(40);
  CHECK(twice.bits(8) == 1 && !twice.overrun());
  CHECK(twice.bits(1) == 0 && twice.overrun());

  // A unit of no more than its header has no payload.
  hevc::BitReader none(unit(bytes, hevc::kSps, {}));
  CHECK(none.bits(1) == 0 && none.overrun());
  hevc::Nal short_unit = unit(bytes, hevc::kSps, {});
  short_unit.size = 1;
  hevc::BitReader shorter(short_unit);
  CHECK(!shorter.flag() && shorter.overrun());
  return 0;
}

// Exp-Golomb codes (9.2): 1, 010, 011, 00100, 00101 and 0001000 are 0, 1,
// 2, 3, 4 and 7 as ue(v), and 0, 1, -1, 2, -2 and 4 as se(v); the largest
// code, 31 zeros, is 2^32 - 2, and one of 32 zeros sets overrun.
int test_exp_golomb() {
  std::vector<std::uint8_t> bytes;
  hevc::BitReader u(unit(bytes, hevc::kSps, {0xa6, 0x42, 0x88}));
  for (const std::uint32_t want : {0u, 1u, 2u, 3u, 4u, 7u}) {
    CHECK(u.ue() == want);
  }
  CHECK(!u.overrun());
  hevc::BitReader s(unit(bytes, hevc::kSps, {0xa6, 0x42, 0x88}));
  for (const std::int32_t want : {0, 1, -1, 2, -2, 4}) CHECK(s.se() == want);
  CHECK(!s.overrun());

  // RBSP 00 00 00 01 ff ff ff fe: 31 zeros, a one and 31 ones.
  hevc::BitReader largest(
      unit(bytes, hevc::kSps, {0, 0, 3, 0, 1, 0xff, 0xff, 0xff, 0xfe}));
  CHECK(largest.ue() == 0xfffffffeu && !largest.overrun());
  // RBSP 00 00 00 00 80: 32 zeros.
  hevc::BitReader longer(unit(bytes, hevc::kSps, {0, 0, 3, 0, 0, 0x80}));
  CHECK(longer.ue() == 0 && longer.overrun());
  // A code the payload ends inside.
  hevc::BitReader cut(unit(bytes, hevc::kSps, {0x00}));
  CHECK(cut.ue() == 0 && cut.overrun());
  return 0;
}

// rbsp_trailing_bits: a one, then zeros to the end.
int test_trailing_bits() {
  std::vector<std::uint8_t> bytes;
  hevc::BitReader at(unit(bytes, hevc::kSps, {0xb0, 0x00}));  // 101|1 0000...
  CHECK(at.bits(3) == 5 && at.at_trailing_bits() && !at.overrun());
  hevc::BitReader early(unit(bytes, hevc::kSps, {0xb0}));  // 10|11 0000
  CHECK(early.bits(2) == 2 && !early.at_trailing_bits());
  hevc::BitReader late(unit(bytes, hevc::kSps, {0x80, 0x01}));
  CHECK(!late.at_trailing_bits());
  hevc::BitReader empty(unit(bytes, hevc::kSps, {}));
  CHECK(!empty.at_trailing_bits() && empty.overrun());
  return 0;
}

// Annex B: three- and four-byte start codes, the zeros before the next one
// left off, a two-byte header read; none, or a header the standard forbids,
// is refused.
int test_split() {
  const std::uint8_t au[] = {0, 0, 0,    1,    0x46, 0x01, 0x10, 0,
                             0, 1, 0x40, 0x01, 0x0c, 0,    0,    0,
                             0, 1, 0x02, 0x01, 0xd0, 0x00};
  std::vector<hevc::Nal> nals;
  CHECK(hevc::split_annex_b(au, sizeof(au), nals).ok());
  CHECK(nals.size() == 3);
  CHECK(nals[0].type == 35 && nals[0].size == 3);  // an AUD
  CHECK(nals[1].type == hevc::kVps && nals[1].size == 3);
  CHECK(nals[2].type == 1 && nals[2].size == 3);  // trailing zero gone
  CHECK(nals[0].layer_id == 0 && nals[0].temporal_id == 0);
  const std::uint8_t none[] = {0x40, 0x01, 0x0c};
  CHECK(hevc::split_annex_b(none, sizeof(none), nals).domain() ==
        Code::IoError);
  const std::uint8_t forbidden[] = {0, 0, 1, 0xc0, 0x01, 0x0c};
  CHECK(hevc::split_annex_b(forbidden, sizeof(forbidden), nals).domain() ==
        Code::IoError);
  const std::uint8_t no_temporal_id[] = {0, 0, 1, 0x40, 0x00, 0x0c};
  CHECK(hevc::split_annex_b(no_temporal_id, sizeof(no_temporal_id), nals)
            .domain() == Code::IoError);
  const std::uint8_t empty[] = {0, 0, 1, 0x40};
  CHECK(hevc::split_annex_b(empty, sizeof(empty), nals).domain() ==
        Code::IoError);
  CHECK(hevc::split_annex_b(nullptr, 4, nals).domain() ==
        Code::InvalidArgument);
  return 0;
}

// What each clip's SPS says, as libavcodec reads it (ffprobe).
int test_parameter_sets() {
  const Read patches = read_clip("patches_256x144");
  CHECK(patches.ok && !patches.sps.empty());
  for (const hevc::Sps& s : patches.sps) {
    if (check_size(s, 256, 144, 256, 144) != 0) return 1;
    CHECK(s.chroma_format_idc == 1 && s.bit_depth_luma == 8 &&
          s.bit_depth_chroma == 8);
    CHECK(s.color.matrix == 1 && s.color.transfer == 1 &&
          s.color.primaries == 1 && !s.color.full_range &&
          s.color.chroma_location == 0);
    CHECK(s.max_num_reorder == 0);
  }
  const Read full = read_clip("fullrange_256x144");
  CHECK(full.ok && full.sps.front().color.full_range);
  const Read unlabelled = read_clip("unlabelled_256x144");
  CHECK(unlabelled.ok);
  CHECK(unlabelled.sps.front().color.matrix == 2 &&
        unlabelled.sps.front().color.transfer == 2 &&
        unlabelled.sps.front().color.primaries == 2);
  const Read cropped = read_clip("cropped_240x128");
  CHECK(cropped.ok);
  if (check_size(cropped.sps.front(), 256, 144, 240, 128) != 0) return 1;
  CHECK(cropped.sps.front().crop_left == 16 &&
        cropped.sps.front().crop_top == 16);

  const Read t2 = read_clip("testsrc2_128x64");
  CHECK(t2.ok);
  if (check_size(t2.sps.front(), 128, 64, 128, 64) != 0) return 1;
  // No colour description: unspecified, limited range.
  CHECK(t2.sps.front().color.matrix == 2 && !t2.sps.front().color.full_range);
  CHECK(t2.sps.front().max_num_reorder > 0);  // B-frames
  const Read t2_full = read_clip("testsrc2_full_128x64");
  CHECK(t2_full.ok && t2_full.sps.front().color.full_range &&
        t2_full.sps.front().color.matrix == 1);
  const Read t2_cropped = read_clip("testsrc2_cropped_120x56");
  CHECK(t2_cropped.ok);
  if (check_size(t2_cropped.sps.front(), 128, 64, 120, 56) != 0) return 1;
  CHECK(t2_cropped.sps.front().crop_left == 0 &&
        t2_cropped.sps.front().crop_top == 0);
  const Read main10 = read_clip("testsrc2_main10_128x64");
  CHECK(main10.ok && main10.sps.front().bit_depth_luma == 10 &&
        main10.sps.front().bit_depth_chroma == 10);

  // The B-frame clip, then two pictures of 4:0:0 grey.
  const Read refused = read_clip("refused_256x144");
  CHECK(refused.ok && refused.sps.front().chroma_format_idc == 1 &&
        refused.sps.back().chroma_format_idc == 0);
  return 0;
}

// The pictures' order counts, in decode order: the patch clips' key frame
// every 4, testsrc2's B-frames (libavcodec shows access unit 1 third), and
// the open GOPs' CRA and the leading pictures sent after it.
int test_picture_order() {
  const Read patches = read_clip("patches_256x144");
  CHECK(patches.ok);
  CHECK(
      (patches.poc_lsb == std::vector<std::uint32_t>{0, 1, 2, 3, 0, 1, 2, 3}));
  CHECK(hevc::is_idr(patches.picture_types[0]));
  CHECK(hevc::is_irap(patches.picture_types[0]) &&
        !hevc::is_irap(patches.picture_types[1]));
  const Read t2 = read_clip("testsrc2_128x64");
  CHECK(t2.ok);
  CHECK((t2.poc_lsb == std::vector<std::uint32_t>{0, 3, 2, 1, 6, 5, 4, 7}));
  const Read open_gop = read_clip("open_gop_256x144");
  CHECK(open_gop.ok && open_gop.picture_types.size() == 16);
  CHECK(open_gop.picture_types[4] == hevc::kCraNut);
  CHECK(hevc::is_irap(hevc::kCraNut) && !hevc::is_idr(hevc::kCraNut));
  CHECK(hevc::is_rasl(open_gop.picture_types[5]) &&
        hevc::is_rasl(open_gop.picture_types[6]));
  return 0;
}

// Every SPS cut short is refused, as is a slice whose PPS or SPS was not
// given, and a NAL unit of a type the call does not read; noise behind each
// header is read without a read out of bounds.
int test_malformed() {
  const Units units =
      vr_test::hevc_access_units(clip("testsrc2_full_128x64").c_str());
  CHECK(!units.empty());
  std::vector<hevc::Nal> nals;
  CHECK(hevc::split_annex_b(units[0].data(), units[0].size(), nals).ok());
  const hevc::Nal* vps = nullptr;
  const hevc::Nal* sps = nullptr;
  const hevc::Nal* pps = nullptr;
  const hevc::Nal* slice = nullptr;
  for (const hevc::Nal& nal : nals) {
    if (nal.type == hevc::kVps) vps = &nal;
    if (nal.type == hevc::kSps) sps = &nal;
    if (nal.type == hevc::kPps) pps = &nal;
    if (nal.type < 32 && slice == nullptr) slice = &nal;
  }
  CHECK(vps != nullptr && sps != nullptr && pps != nullptr && slice != nullptr);
  const vkc::Result<std::uint32_t> vps_id = hevc::vps_id(*vps);
  const vkc::Result<hevc::Sps> read_sps = hevc::parse_sps(*sps);
  const vkc::Result<hevc::Pps> read_pps = hevc::parse_pps(*pps);
  CHECK(vps_id.ok() && read_sps.ok() && read_pps.ok());
  CHECK(read_sps->vps_id == *vps_id && read_pps->sps_id == read_sps->id);
  // Each cut in storage of its own size, so a read past it is out of bounds.
  for (std::size_t size = 2; size < sps->size; ++size) {
    const std::vector<std::uint8_t> bytes(sps->data, sps->data + size);
    hevc::Nal cut = *sps;
    cut.data = bytes.data();
    cut.size = bytes.size();
    CHECK(hevc::parse_sps(cut).status().domain() == Code::IoError);
  }

  // A slice with no parameter sets, then with its PPS but not its SPS.
  std::map<std::uint32_t, hevc::Pps> pps_by_id;
  std::map<std::uint32_t, hevc::Sps> sps_by_id;
  CHECK(
      hevc::parse_slice_start(*slice, pps_by_id, sps_by_id).status().domain() ==
      Code::IoError);
  pps_by_id[read_pps->id] = *read_pps;
  CHECK(
      hevc::parse_slice_start(*slice, pps_by_id, sps_by_id).status().domain() ==
      Code::IoError);
  sps_by_id[read_sps->id] = *read_sps;
  CHECK(hevc::parse_slice_start(*slice, pps_by_id, sps_by_id).ok());
  hevc::Nal header_only = *slice;
  header_only.size = 2;
  CHECK(hevc::parse_slice_start(header_only, pps_by_id, sps_by_id)
            .status()
            .domain() == Code::IoError);

  // A NAL unit of another type.
  CHECK(hevc::vps_id(*sps).status().domain() == Code::InvalidArgument);
  CHECK(hevc::parse_sps(*pps).status().domain() == Code::InvalidArgument);
  CHECK(hevc::parse_pps(*sps).status().domain() == Code::InvalidArgument);
  CHECK(hevc::parse_slice_start(*sps, pps_by_id, sps_by_id).status().domain() ==
        Code::InvalidArgument);

  // Noise behind each header: parsed or refused as bytes that do not parse.
  const int types[] = {hevc::kVps, hevc::kSps, hevc::kPps, hevc::kIdrWRadl, 1};
  std::uint32_t state = 12345;
  for (int round = 0; round < 2500; ++round) {
    const int type = types[round % 5];
    state = state * 1664525u + 1013904223u;
    std::vector<std::uint8_t> noise(2 + (state >> 8) % 62);
    noise[0] = static_cast<std::uint8_t>(type << 1);
    noise[1] = 0x01;
    for (std::size_t i = 2; i < noise.size(); ++i) {
      state = state * 1664525u + 1013904223u;
      noise[i] = static_cast<std::uint8_t>(state >> 24);
    }
    hevc::Nal nal;
    nal.data = noise.data();
    nal.size = noise.size();
    nal.type = type;
    vkc::Status read;
    if (type == hevc::kVps) {
      read = hevc::vps_id(nal).status();
    } else if (type == hevc::kSps) {
      read = hevc::parse_sps(nal).status();
    } else if (type == hevc::kPps) {
      read = hevc::parse_pps(nal).status();
    } else {
      read = hevc::parse_slice_start(nal, pps_by_id, sps_by_id).status();
    }
    CHECK(read.ok() || read.domain() == Code::IoError);
  }
  return 0;
}

// @p call with its largest host allocation failed: whether it returned
// OutOfMemory rather than throwing.
template <typename Call>
bool contains_allocation_failure(const Call& call) {
  auto& failure = vr_test::allocation_failure;
  failure = {};
  failure.measure = true;
  static_cast<void>(call());
  const std::size_t bytes = failure.largest;
  const void* site = failure.largest_site;
  failure = {};
  failure.bytes = bytes;
  failure.site = site;
  failure.armed = bytes != 0;
  const vkc::Status status = call();
  const bool injected = failure.injected;
  failure = {};
  return injected && status.domain() == Code::OutOfMemory;
}

// Each call returns a host allocation failure as OutOfMemory: the split
// growing its list, and each parser building a refusal.
int test_allocation_failure() {
  const Units units =
      vr_test::hevc_access_units(clip("testsrc2_full_128x64").c_str());
  CHECK(!units.empty());
  const std::vector<std::uint8_t>& au = units[0];
  std::vector<hevc::Nal> nals;
  CHECK(hevc::split_annex_b(au.data(), au.size(), nals).ok());
  const hevc::Nal* sps = nullptr;
  const hevc::Nal* slice = nullptr;
  for (const hevc::Nal& nal : nals) {
    if (nal.type == hevc::kSps) sps = &nal;
    if (nal.type < 32 && slice == nullptr) slice = &nal;
  }
  CHECK(sps != nullptr && slice != nullptr);
  hevc::Nal cut = *sps;
  cut.size /= 2;
  const std::map<std::uint32_t, hevc::Pps> no_pps;
  const std::map<std::uint32_t, hevc::Sps> no_sps;

  CHECK(contains_allocation_failure([&] {
    std::vector<hevc::Nal> grown;
    return hevc::split_annex_b(au.data(), au.size(), grown);
  }));
  CHECK(
      contains_allocation_failure([&] { return hevc::vps_id(*sps).status(); }));
  CHECK(contains_allocation_failure(
      [&] { return hevc::parse_sps(cut).status(); }));
  CHECK(contains_allocation_failure(
      [&] { return hevc::parse_pps(*sps).status(); }));
  CHECK(contains_allocation_failure([&] {
    return hevc::parse_slice_start(*slice, no_pps, no_sps).status();
  }));
  return 0;
}

}  // namespace

int main() {
  if (test_bits() != 0) return 1;
  if (test_exp_golomb() != 0) return 1;
  if (test_trailing_bits() != 0) return 1;
  if (test_split() != 0) return 1;
  if (test_parameter_sets() != 0) return 1;
  if (test_picture_order() != 0) return 1;
  if (test_malformed() != 0) return 1;
  if (test_allocation_failure() != 0) return 1;
  std::printf("hevc syntax tests passed\n");
  return 0;
}
