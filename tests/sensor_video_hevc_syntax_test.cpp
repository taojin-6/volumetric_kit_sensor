// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The H.265 syntax the VideoToolbox path reads itself (hevc_syntax.hpp), on
// the committed clips (tools/make_hevc_fixtures.sh), with no codec and no
// GPU, so it runs on every leg: the NAL units an access unit splits into,
// what each clip's SPS says (size, crop, format, the VUI's colour, the
// reorder depth), the picture order counts of its slices, and bytes that do
// not parse -- every truncated SPS, a slice whose PPS was never sent, and
// noise -- refused without a read out of bounds.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "hevc_fixture.hpp"
#include "hevc_syntax.hpp"
#include "test_check.hpp"

namespace hevc = volumetric_kit::recon::sensor::video::hevc;

namespace {

using Units = std::vector<std::vector<std::uint8_t>>;

std::string clip(const char* name) {
  return std::string(VR_HEVC_DATA) + "/" + name + ".h265";
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
    if (!hevc::split_annex_b(unit.data(), unit.size(), nals)) out.ok = false;
    bool picture = false;
    for (const hevc::Nal& nal : nals) {
      if (nal.type == hevc::kSps) {
        const std::optional<hevc::Sps> sps = hevc::parse_sps(nal);
        if (!sps) {
          out.ok = false;
          continue;
        }
        out.sps.push_back(*sps);
        out.sps_by_id[sps->id] = *sps;
      } else if (nal.type == hevc::kPps) {
        const std::optional<hevc::Pps> pps = hevc::parse_pps(nal);
        if (!pps) {
          out.ok = false;
          continue;
        }
        out.pps_by_id[pps->id] = *pps;
      } else if (nal.type < 32 && !picture) {
        const std::optional<hevc::SliceStart> slice =
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

// Annex B: three- and four-byte start codes, the zeros before the next one
// left off, a two-byte header read; none, or a header the standard forbids,
// is refused.
int test_split() {
  const std::uint8_t au[] = {0, 0, 0,    1,    0x46, 0x01, 0x10, 0,
                             0, 1, 0x40, 0x01, 0x0c, 0,    0,    0,
                             0, 1, 0x02, 0x01, 0xd0, 0x00};
  std::vector<hevc::Nal> nals;
  CHECK(hevc::split_annex_b(au, sizeof(au), nals));
  CHECK(nals.size() == 3);
  CHECK(nals[0].type == 35 && nals[0].size == 3);  // an AUD
  CHECK(nals[1].type == hevc::kVps && nals[1].size == 3);
  CHECK(nals[2].type == 1 && nals[2].size == 3);  // trailing zero gone
  CHECK(nals[0].layer_id == 0 && nals[0].temporal_id == 0);
  const std::uint8_t none[] = {0x40, 0x01, 0x0c};
  CHECK(!hevc::split_annex_b(none, sizeof(none), nals));
  const std::uint8_t forbidden[] = {0, 0, 1, 0xc0, 0x01, 0x0c};
  CHECK(!hevc::split_annex_b(forbidden, sizeof(forbidden), nals));
  const std::uint8_t no_temporal_id[] = {0, 0, 1, 0x40, 0x00, 0x0c};
  CHECK(!hevc::split_annex_b(no_temporal_id, sizeof(no_temporal_id), nals));
  const std::uint8_t empty[] = {0, 0, 1, 0x40};
  CHECK(!hevc::split_annex_b(empty, sizeof(empty), nals));
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
  CHECK(patches.picture_types[0] == hevc::kIdrWRadl ||
        patches.picture_types[0] == hevc::kIdrNLp);
  const Read t2 = read_clip("testsrc2_128x64");
  CHECK(t2.ok);
  CHECK((t2.poc_lsb == std::vector<std::uint32_t>{0, 3, 2, 1, 6, 5, 4, 7}));
  const Read open_gop = read_clip("open_gop_256x144");
  CHECK(open_gop.ok && open_gop.picture_types.size() == 16);
  CHECK(open_gop.picture_types[4] == hevc::kCraNut);
  CHECK(hevc::is_rasl(open_gop.picture_types[5]) &&
        hevc::is_rasl(open_gop.picture_types[6]));
  return 0;
}

// Every SPS cut short is refused, as is noise after a valid header, and a
// slice whose PPS or SPS was never sent; none reads out of bounds.
int test_malformed() {
  const Units units =
      vr_test::hevc_access_units(clip("testsrc2_full_128x64").c_str());
  CHECK(!units.empty());
  std::vector<hevc::Nal> nals;
  CHECK(hevc::split_annex_b(units[0].data(), units[0].size(), nals));
  const hevc::Nal* sps = nullptr;
  const hevc::Nal* slice = nullptr;
  for (const hevc::Nal& nal : nals) {
    if (nal.type == hevc::kSps) sps = &nal;
    if (nal.type < 32 && slice == nullptr) slice = &nal;
  }
  CHECK(sps != nullptr && slice != nullptr);
  CHECK(hevc::parse_sps(*sps).has_value());
  for (std::size_t size = 2; size < sps->size; ++size) {
    std::vector<std::uint8_t> cut(sps->data, sps->data + size);
    hevc::Nal nal = *sps;
    nal.data = cut.data();
    nal.size = cut.size();
    CHECK(!hevc::parse_sps(nal).has_value());
  }
  std::uint32_t state = 12345;
  std::vector<std::uint8_t> noise(64);
  for (int round = 0; round < 2000; ++round) {
    noise[0] = 0x42;  // an SPS header
    noise[1] = 0x01;
    for (std::size_t i = 2; i < noise.size(); ++i) {
      state = state * 1664525u + 1013904223u;
      noise[i] = static_cast<std::uint8_t>(state >> 24);
    }
    hevc::Nal nal;
    nal.data = noise.data();
    nal.size = 2 + (state % 62);
    nal.type = hevc::kSps;
    hevc::parse_sps(nal);
    hevc::parse_pps(nal);
    hevc::vps_id(nal);
  }
  const std::map<std::uint32_t, hevc::Pps> no_pps;
  const std::map<std::uint32_t, hevc::Sps> no_sps;
  CHECK(!hevc::parse_slice_start(*slice, no_pps, no_sps).has_value());
  return 0;
}

}  // namespace

int main() {
  if (test_split() != 0) return 1;
  if (test_parameter_sets() != 0) return 1;
  if (test_picture_order() != 0) return 1;
  if (test_malformed() != 0) return 1;
  std::printf("hevc syntax tests passed\n");
  return 0;
}
