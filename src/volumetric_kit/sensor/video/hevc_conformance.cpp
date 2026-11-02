// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The encoder rules (testing/hevc_conformance.hpp): each is checked through
// HevcEncoder's public calls, with the stream read by sensor's syntax
// readers (hevc_syntax.hpp, testing/hevc_slices.hpp) and decoded by
// HevcDecoder. Written apart from the encoder, so a rule here is never the
// encoder's own code agreeing with itself.

#include "volumetric_kit/sensor/testing/hevc_conformance.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/testing/hevc_slices.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/video/backend.hpp"
#include "volumetric_kit/sensor/video/frame_color.hpp"
#include "volumetric_kit/sensor/video/hevc_decoder.hpp"
#include "volumetric_kit/sensor/video/hevc_syntax.hpp"

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>

#include "volumetric_kit/sensor/testing/nv12_pixel_buffer.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder_apple.hpp"
#endif

namespace volumetric_kit::sensor::testing {
namespace {

using Encoder = HevcEncoder;

// The rules' run: a build whose encoder takes pixel buffers (Apple's).
#if defined(__APPLE__)
using Code = core::Status::Code;

constexpr std::uint32_t kRing = 8;  // distinct pictures a run sends

std::int64_t pts_of(std::uint32_t i) { return 1000 + 333 * std::int64_t{i}; }

std::string name(Code code) { return std::string(core::to_string(code)); }

std::string said(const core::Status& s) {
  return name(s.domain()) + " (" + s.message() + ")";
}

// The rule's own statement of normalization (HevcEncoder::Options::color):
// matrix 5 is 6; transfer 6, 14 and 15 are 1.
ColorCodes normalized(ColorCodes codes) {
  if (codes.matrix == 5) codes.matrix = 6;
  if (codes.transfer == 6 || codes.transfer == 14 || codes.transfer == 15) {
    codes.transfer = 1;
  }
  return codes;
}

std::string codes_text(const ColorCodes& c) {
  return "matrix " + std::to_string(c.matrix) + ", transfer " +
         std::to_string(c.transfer) + ", primaries " +
         std::to_string(c.primaries) + ", " +
         (c.full_range ? "full" : "video") + " range, siting " +
         std::to_string(c.chroma_location);
}

bool same(const ColorCodes& a, const ColorCodes& b) {
  return a.matrix == b.matrix && a.transfer == b.transfer &&
         a.primaries == b.primaries && a.full_range == b.full_range &&
         a.chroma_location == b.chroma_location;
}

// What one access unit is, as C3 reads it: empty when it keeps the rule.
std::string unit_breach(const std::vector<std::uint8_t>& au, bool key,
                        std::vector<hevc::Nal>& nals) {
  if (au.size() < 4 || au[0] != 0 || au[1] != 0 || au[2] != 0 || au[3] != 1) {
    return "an access unit that does not open with 00 00 00 01";
  }
  const core::Status split = hevc::split_annex_b(au.data(), au.size(), nals);
  if (!split.ok()) return "an access unit that does not split: " + said(split);
  bool slice = false;
  for (std::size_t i = 0; i < nals.size(); ++i) {
    const hevc::Nal& n = nals[i];
    const std::size_t at = static_cast<std::size_t>(n.data - au.data());
    if (at < 4 || n.data[-4] != 0 || n.data[-3] != 0 || n.data[-2] != 0 ||
        n.data[-1] != 1) {
      return "NAL unit " + std::to_string(i) + " after a 3-byte start code";
    }
    if (n.temporal_id != 0) {
      return "NAL unit " + std::to_string(i) + " in temporal layer " +
             std::to_string(n.temporal_id);
    }
    if (n.type < 32) {
      // first_slice_segment_in_pic_flag: one picture an access unit.
      if (slice && n.size > 2 && (n.data[2] & 0x80) != 0) {
        return "two pictures in one access unit";
      }
      slice = true;
      if (!key && hevc::is_irap(n.type)) {
        return "a P-picture with an IRAP slice (type " +
               std::to_string(n.type) + ")";
      }
    }
  }
  if (!slice) return "an access unit with no slice";
  if (key) {
    if (nals.size() < 4 || nals[0].type != hevc::kVps ||
        nals[1].type != hevc::kSps || nals[2].type != hevc::kPps) {
      return "an IDR whose access unit does not open with its VPS, SPS and "
             "PPS";
    }
    const std::optional<int> type =
        hevc::first_slice_type(au.data(), au.size());
    if (!type || !hevc::is_idr(*type)) {
      return "a key that is not an IDR (type " +
             (type ? std::to_string(*type) : std::string("none")) + ")";
    }
  }
  return {};
}

struct CfRelease {
  void operator()(CFTypeRef ref) const noexcept { CFRelease(ref); }
};
using Pixels =
    std::unique_ptr<std::remove_pointer_t<CVPixelBufferRef>, CfRelease>;

// The test pattern's picture @p index as a pixel buffer (pixel_buffer):
// '420f' or '420v' as @p tags' range, tagged with its matrix, transfer and
// primaries and no siting.
core::Result<Pixels> make_pixels(std::uint32_t width, std::uint32_t height,
                                 std::uint32_t index, const ColorCodes& tags) {
  VKC_ASSIGN(CVPixelBufferRef made,
             pixel_buffer(test_pattern(width, height, index), tags));
  return Pixels(made);
}

// The settings line VideoToolbox's session must report (C10), for @p o.
std::string pinned_settings(const Encoder::Options& o) {
  const auto cv = [](CFStringRef s) -> std::string {
    if (s == nullptr) return "unset";
    char text[64] = {};
    CFStringGetCString(s, text, sizeof text, kCFStringEncodingUTF8);
    return text;
  };
  const ColorCodes c = normalized(o.color);
  return "VideoToolbox HEVC " + std::to_string(o.width) + "x" +
         std::to_string(o.height) +
         ": hardware required, low-latency rate control, profile "
         "HEVC_Main_AutoLevel, reordering off, base layer fraction 1, base "
         "frame QP on (P " +
         std::to_string(o.qp) + ", IDR " +
         std::to_string(o.key_qp.value_or(o.qp)) +
         "), real time unset, YCbCr " +
         cv(CVYCbCrMatrixGetStringForIntegerCodePoint(c.matrix)) +
         ", transfer " +
         (c.transfer == 2 ? std::string("unset")
                          : cv(CVTransferFunctionGetStringForIntegerCodePoint(
                                c.transfer))) +
         ", primaries " +
         (c.primaries == 2
              ? std::string("unset")
              : cv(CVColorPrimariesGetStringForIntegerCodePoint(c.primaries))) +
         ", " + (c.full_range ? "'420f'" : "'420v'") + ", siting " +
         std::to_string(c.chroma_location) + " from the buffer";
}

class Run {
 public:
  explicit Run(const HevcEncoderCase& c) : c_(c), o_(c.options) {}

  core::Result<std::string> all();

 private:
  // Picture @p i of the ring, made the first time it is asked for.
  core::Result<CVPixelBufferRef> picture(std::uint32_t i);
  core::Status send(Encoder& e, std::uint32_t i, bool key) {
    VKC_ASSIGN(CVPixelBufferRef p, picture(i));
    return send_pixel_buffer(e, p, key, pts_of(i));
  }
  // The rules, each empty when kept.
  core::Result<std::string> refusals();
  core::Result<std::string> stream(Encoder& e);
  core::Result<std::string> decodes();
  core::Result<std::string> drop(Encoder& e);
  core::Result<std::string> lose(Encoder& e);
  core::Result<std::string> buffers(Encoder& e);
  core::Result<std::string> move(Encoder& e);
  core::Result<std::string> in_flight(Encoder& e);

  const HevcEncoderCase& c_;
  const Encoder::Options o_;
  OSType format_ = 0;
  Pixels ring_[kRing];
  // The main stream, as received, for C6.
  std::vector<std::vector<std::uint8_t>> units_;
};

core::Result<CVPixelBufferRef> Run::picture(std::uint32_t i) {
  Pixels& slot = ring_[i % kRing];
  if (slot == nullptr) {
    VKC_ASSIGN(slot, make_pixels(o_.width, o_.height, i % kRing, o_.color));
  }
  return slot.get();
}

core::Result<std::string> Run::refusals() {
  struct Bad {
    const char* what;
    Encoder::Options o;
    Code want;
  };
  std::vector<Bad> bad;
  const auto with = [this](auto change) {
    Encoder::Options o = o_;
    change(o);
    return o;
  };
  const Code inv = Code::InvalidArgument;
  const Code uns = Code::Unsupported;
  bad.push_back({"an odd width", with([](auto& o) { o.width += 1; }), inv});
  bad.push_back({"a width of 0", with([](auto& o) { o.width = 0; }), inv});
  bad.push_back({"a height of 0", with([](auto& o) { o.height = 0; }), inv});
  bad.push_back({"a side past kMaxSide", with([](auto& o) {
                   o.width = Encoder::kMaxSide + 2;
                   o.height = 2;
                 }),
                 inv});
  bad.push_back({"samples past kMaxLumaSamples", with([](auto& o) {
                   o.width = Encoder::kMaxSide;
                   o.height = Encoder::kMaxLumaSamples / Encoder::kMaxSide + 2;
                 }),
                 inv});
  bad.push_back({"QP -1", with([](auto& o) { o.qp = -1; }), inv});
  bad.push_back({"QP 52", with([](auto& o) { o.qp = 52; }), inv});
  bad.push_back({"key QP -1", with([](auto& o) { o.key_qp = -1; }), inv});
  bad.push_back({"key QP 52", with([](auto& o) { o.key_qp = 52; }), inv});
  bad.push_back({"effort 0", with([](auto& o) { o.effort = 0; }), inv});
  bad.push_back({"effort 8", with([](auto& o) { o.effort = 8; }), inv});
  bad.push_back({"matrix 2", with([](auto& o) { o.color.matrix = 2; }), inv});
  bad.push_back({"matrix 4", with([](auto& o) { o.color.matrix = 4; }), uns});
  bad.push_back(
      {"transfer 3", with([](auto& o) { o.color.transfer = 3; }), uns});
  bad.push_back(
      {"primaries 7", with([](auto& o) { o.color.primaries = 7; }), uns});
  bad.push_back(
      {"siting 6", with([](auto& o) { o.color.chroma_location = 6; }), uns});
  if (video_backend() == VideoBackend::VideoToolbox) {
    // VideoToolbox codes QP 0 as 1.
    bad.push_back({"QP 0", with([](auto& o) { o.qp = 0; }), uns});
    bad.push_back({"key QP 0", with([](auto& o) { o.key_qp = 0; }), uns});
  }
  for (const Bad& b : bad) {
    const core::Result<Encoder> made = Encoder::create(b.o);
    if (made.ok()) return "C1: " + std::string(b.what) + " opened a session";
    if (made.status().domain() != b.want) {
      return "C1: " + std::string(b.what) + " gave " + said(made.status()) +
             ", not " + name(b.want);
    }
  }
  return std::string();
}

core::Result<std::string> Run::stream(Encoder& e) {
  // C2: the first must be a key.
  const core::Status first = send(e, 0, false);
  if (first.domain() != Code::InvalidArgument || e.in_flight() != 0) {
    return "C2: a P-picture first gave " + said(first) +
           ", not InvalidArgument";
  }
  const std::uint32_t n = c_.pictures;
  const std::uint32_t middle = n / 2;
  SliceReader reader;
  std::vector<hevc::Nal> nals;
  Encoder::CodedPicture coded;
  units_.clear();
  std::uint32_t received = 0;
  const auto take = [&]() -> core::Result<std::string> {
    const core::Result<bool> got = e.receive(coded);
    if (!got.ok()) {
      return "C3: picture " + std::to_string(received) +
             " failed: " + said(got.status());
    }
    if (!*got) return "C13: nothing came back with a picture in flight";
    const std::uint32_t i = received++;
    const bool key = i == 0 || i == middle;
    if (coded.pts != pts_of(i) || coded.key != key) {
      return "C13: picture " + std::to_string(i) + " came back with pts " +
             std::to_string(coded.pts) + (coded.key ? ", a key" : ", no key") +
             ", out of order or mislabelled";
    }
    if (const std::string b = unit_breach(coded.access_unit, key, nals);
        !b.empty()) {
      return "C3: picture " + std::to_string(i) + ": " + b;
    }
    const core::Result<SliceReader::Picture> p =
        reader.read(coded.access_unit.data(), coded.access_unit.size());
    if (!p.ok()) {
      return "C3: picture " + std::to_string(i) +
             " does not parse: " + said(p.status());
    }
    const int want_type = key ? 2 : 1;
    if (p->slice_type != want_type) {
      return "C3: picture " + std::to_string(i) + " has slice type " +
             std::to_string(p->slice_type) + ", not " +
             std::to_string(want_type);
    }
    const int want_qp = key ? o_.key_qp.value_or(o_.qp) : o_.qp;
    if (p->slice_qp != want_qp) {
      return "C5: picture " + std::to_string(i) + "'s slice QP is " +
             std::to_string(p->slice_qp) + ", not " + std::to_string(want_qp);
    }
    units_.push_back(coded.access_unit);
    return std::string();
  };
  for (std::uint32_t i = 0; i < n; ++i) {
    if (e.in_flight() == Encoder::kInFlight) {
      VKC_ASSIGN(const std::string b, take());
      if (!b.empty()) return b;
    }
    const core::Status sent = send(e, i, i == 0 || i == middle);
    if (!sent.ok()) {
      return "C2: picture " + std::to_string(i) + " was refused: " + said(sent);
    }
  }
  while (received < n) {
    VKC_ASSIGN(const std::string b, take());
    if (!b.empty()) return b;
  }
  // C4: the sequence.
  const std::optional<SliceReader::Sequence>& seq = reader.sequence();
  if (!seq) return std::string("C4: no SPS in the stream");
  const hevc::Sps& sps = seq->sps;
  if (seq->profile_idc != 1) {
    return "C4: profile " + std::to_string(seq->profile_idc) + ", not Main";
  }
  if (sps.chroma_format_idc != 1 || sps.separate_colour_plane ||
      sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8) {
    return std::string("C4: not 8-bit 4:2:0");
  }
  if (sps.width() != o_.width || sps.height() != o_.height) {
    return "C4: a picture of " + std::to_string(sps.width()) + "x" +
           std::to_string(sps.height()) + " after the crop";
  }
  if (sps.max_num_reorder != 0) {
    return "C4: " + std::to_string(sps.max_num_reorder) + " pictures reordered";
  }
  const ColorCodes want = normalized(o_.color);
  if (!same(e.facts().color, want)) {
    return "C4: the session states " + codes_text(e.facts().color) +
           ", not the options normalized, " + codes_text(want);
  }
  if (!same(sps.color, want)) {
    return "C4: the VUI says " + codes_text(sps.color) + ", not " +
           codes_text(want);
  }
  // C10: the configuration.
  if (e.facts().settings != pinned_settings(o_)) {
    return "C10: the settings are\n  " + e.facts().settings + "\nnot\n  " +
           pinned_settings(o_);
  }
  return std::string();
}

core::Result<std::string> Run::decodes() {
  if (c_.device == nullptr) return std::string();
  HevcDecoder::Options d;
  d.device = c_.device;
  d.allocator = c_.allocator;
  const auto run = [&](std::size_t from) -> core::Result<std::string> {
    VKC_ASSIGN(HevcDecoder decoder, HevcDecoder::create(d));
    if (from != 0) VKC_TRY(decoder.reset());
    const ColorCodes want = normalized(o_.color);
    const YcbcrWeights weights =
        ycbcr_weights(resolve_matrix(want.matrix, o_.height));
    const std::optional<ColorEncoding> encoding =
        resolve_encoding(want.transfer, want.primaries);
    std::size_t out = 0;
    for (std::size_t i = from; i <= units_.size(); ++i) {
      const bool end = i == units_.size();
      const core::Status sent = decoder.send(
          end ? nullptr : units_[i].data(), end ? 0 : units_[i].size(),
          pts_of(static_cast<std::uint32_t>(i)));
      if (!sent.ok()) {
        return "C6: unit " + std::to_string(i) +
               " does not decode: " + said(sent);
      }
      for (;;) {
        core::Result<std::optional<DecodedPicture>> p = decoder.receive();
        if (!p.ok()) {
          return "C6: picture " + std::to_string(from + out) +
                 " does not decode: " + said(p.status());
        }
        if (!*p) break;
        const DecodedPicture& pic = **p;
        const std::uint32_t at = static_cast<std::uint32_t>(from + out++);
        if (pic.pts != pts_of(at) || pic.yuv.width != o_.width ||
            pic.yuv.height != o_.height) {
          return "C6: picture " + std::to_string(at) + " decodes as " +
                 std::to_string(pic.yuv.width) + "x" +
                 std::to_string(pic.yuv.height) + " at pts " +
                 std::to_string(pic.pts);
        }
        if (pic.yuv.kr != weights.kr || pic.yuv.kb != weights.kb ||
            pic.yuv.full_range != want.full_range ||
            pic.yuv.chroma_location !=
                resolve_chroma_location(want.chroma_location)) {
          return "C6: picture " + std::to_string(at) +
                 " decodes with another matrix, range or siting";
        }
        if (pic.encoding.has_value() != encoding.has_value() ||
            (encoding && (pic.encoding->transfer != encoding->transfer ||
                          pic.encoding->primaries != encoding->primaries))) {
          return "C6: picture " + std::to_string(at) +
                 " decodes with another encoding";
        }
      }
    }
    if (from + out != units_.size()) {
      return "C6: " + std::to_string(out) + " pictures decode from unit " +
             std::to_string(from) + ", not " +
             std::to_string(units_.size() - from);
    }
    return std::string();
  };
  VKC_ASSIGN(const std::string whole, run(0));
  if (!whole.empty()) return whole;
  return run(c_.pictures / 2);
}

core::Result<std::string> Run::drop(Encoder& e) {
  Encoder::CodedPicture coded;
  VKC_TRY(send(e, 1, e.key_owed()));
  VKC_ASSIGN(const bool first, e.receive(coded));
  if (!first) return std::string("C7: a picture before the drop is lost");
  e.inject(Encoder::Fault::Drop);
  VKC_TRY(send(e, 2, false));
  const core::Result<bool> dropped = e.receive(coded);
  if (dropped.ok() || dropped.status().domain() != Code::IoError) {
    return std::string("C7: a dropped picture came back ") +
           (dropped.ok() ? "coded" : said(dropped.status())) + ", not IoError";
  }
  if (!e.key_owed() || e.lost()) {
    return std::string("C7: after a drop, ") +
           (e.lost() ? "the session is lost" : "no key is owed");
  }
  const core::Status p = send(e, 3, false);
  if (p.domain() != Code::InvalidArgument) {
    return "C7: a P-picture after a drop gave " + said(p);
  }
  VKC_TRY(send(e, 3, true));
  VKC_ASSIGN(const bool key, e.receive(coded));
  if (!key || !coded.key) return std::string("C7: the key after a drop");
  return std::string();
}

core::Result<std::string> Run::lose(Encoder& e) {
  Encoder::CodedPicture coded;
  // Lost while idle: lost at once, a key owed, the next key reopens it.
  e.inject(Encoder::Fault::Lose);
  if (!e.lost() || !e.key_owed()) {
    return std::string("C8: lost idle, the encoder says ") +
           (e.lost() ? "no key is owed" : "it is not lost");
  }
  const core::Status p = send(e, 4, false);
  if (p.domain() != Code::InvalidArgument) {
    return "C8: a P-picture on a lost session gave " + said(p);
  }
  VKC_TRY(send(e, 4, true));
  if (e.lost()) return std::string("C8: a key did not reopen the session");
  VKC_ASSIGN(const bool reopened, e.receive(coded));
  if (!reopened || !coded.key) {
    return std::string("C8: the reopened session's key");
  }
  // Lost with pictures in flight: the one behind the one coding fails.
  VKC_TRY(send(e, 5, false));
  VKC_TRY(send(e, 6, false));
  e.inject(Encoder::Fault::Lose);
  const core::Result<bool> a = e.receive(coded);
  const core::Result<bool> b = e.receive(coded);
  if (b.ok() || b.status().domain() == Code::InvalidArgument) {
    return std::string("C8: a picture sent before the loss came back ") +
           (b.ok() ? "coded" : said(b.status()));
  }
  if (a.ok() && !*a) return std::string("C8: a picture vanished");
  if (!e.key_owed()) return std::string("C8: no key owed after a loss");
  VKC_TRY(send(e, 7, true));
  VKC_ASSIGN(const bool again, e.receive(coded));
  if (!again || !coded.key || e.lost()) {
    return std::string("C8: the session did not reopen at a key");
  }
  return std::string();
}

core::Result<std::string> Run::buffers(Encoder& e) {
  const auto refused = [&](const char* what,
                           CVPixelBufferRef p) -> std::string {
    const core::Status s = send_pixel_buffer(e, p, true, 0);
    if (s.domain() == Code::InvalidArgument && e.in_flight() == 0) return {};
    return "C11: " + std::string(what) + " gave " + said(s);
  };
  if (std::string b = refused("no buffer", nullptr); !b.empty()) return b;
  {
    VKC_ASSIGN(Pixels small, make_pixels(o_.width - 2, o_.height, 0, o_.color));
    if (std::string b = refused("another size", small.get()); !b.empty()) {
      return b;
    }
  }
  {
    ColorCodes other = o_.color;
    other.full_range = !other.full_range;
    VKC_ASSIGN(Pixels wrong, make_pixels(o_.width, o_.height, 0, other));
    if (std::string b = refused("the other range", wrong.get()); !b.empty()) {
      return b;
    }
  }
  {
    ColorCodes tags = o_.color;
    tags.matrix = normalized(tags).matrix == 1 ? 6 : 1;
    VKC_ASSIGN(Pixels tagged, make_pixels(o_.width, o_.height, 0, tags));
    if (std::string b = refused("another matrix tag", tagged.get());
        !b.empty()) {
      return b;
    }
  }
  {
    VKC_ASSIGN(Pixels sited, make_pixels(o_.width, o_.height, 0, o_.color));
    const int other = o_.color.chroma_location == 0 ? 1 : 0;
    CVBufferSetAttachment(sited.get(), kCVImageBufferChromaLocationTopFieldKey,
                          other == 0 ? kCVImageBufferChromaLocation_Left
                                     : kCVImageBufferChromaLocation_Center,
                          kCVAttachmentMode_ShouldPropagate);
    if (std::string b = refused("another siting tag", sited.get());
        !b.empty()) {
      return b;
    }
  }
  {
    CVPixelBufferRef host = nullptr;
    if (CVPixelBufferCreate(kCFAllocatorDefault, o_.width, o_.height, format_,
                            nullptr, &host) != kCVReturnSuccess) {
      return core::Status::out_of_memory("hevc_encoder_breach: a buffer");
    }
    const Pixels owned(host);
    if (CVPixelBufferGetIOSurface(host) == nullptr) {
      if (std::string b = refused("a buffer on no IOSurface", host);
          !b.empty()) {
        return b;
      }
    }
  }
  return std::string();
}

core::Result<std::string> Run::move(Encoder& e) {
  Encoder::CodedPicture coded;
  VKC_TRY(send(e, 8, e.key_owed()));
  Encoder moved(std::move(e));
  // NOLINTNEXTLINE(bugprone-use-after-move): the moved-from contract
  if (e.in_flight() != 0 || !e.key_owed() || e.lost() ||
      e.receive(coded).status().domain() != Code::InvalidArgument ||
      send(e, 9, true).domain() != Code::InvalidArgument) {
    return std::string("C12: a moved-from encoder is not inert");
  }
  VKC_ASSIGN(const bool got, moved.receive(coded));
  if (!got || coded.pts != pts_of(8)) {
    return std::string("C12: the picture sent before the move");
  }
  e = std::move(moved);
  return std::string();
}

core::Result<std::string> Run::in_flight(Encoder& e) {
  Encoder::CodedPicture coded;
  // Two in flight, in order, a third refused.
  const bool key = e.key_owed();
  VKC_TRY(send(e, 10, key));
  VKC_TRY(send(e, 11, false));
  if (e.in_flight() != 2) {
    return "C13: " + std::to_string(e.in_flight()) + " in flight, not 2";
  }
  const core::Status third = send(e, 12, false);
  if (third.domain() != Code::InvalidArgument) {
    return "C13: a third picture in flight gave " + said(third);
  }
  for (const std::uint32_t i : {10u, 11u}) {
    VKC_ASSIGN(const bool got, e.receive(coded));
    if (!got || coded.pts != pts_of(i)) {
      return "C13: picture " + std::to_string(i) + " out of order";
    }
  }
  if (e.in_flight() != 0) return std::string("C13: in_flight after both");
  VKC_ASSIGN(const bool none, e.receive(coded, 0));
  if (none) return std::string("C13: a picture from nowhere");
  // A drop fails the one sent after it, and a key is owed.
  e.inject(Encoder::Fault::Drop);
  VKC_TRY(send(e, 12, false));
  VKC_TRY(send(e, 13, false));
  const core::Result<bool> a = e.receive(coded);
  const core::Result<bool> b = e.receive(coded);
  if (a.ok() || b.ok()) {
    return std::string("C13: after a drop, the picture sent next came back");
  }
  if (!e.key_owed() || e.lost()) {
    return std::string("C13: after a drop in flight, ") +
           (e.lost() ? "the session is lost" : "no key is owed");
  }
  // Sent on this thread, received on another.
  constexpr std::uint32_t kPictures = 30;
  for (std::uint32_t i = 0; i < kRing; ++i) VKC_TRY(picture(i).status());
  std::atomic<std::uint32_t> back{0};
  std::atomic<bool> stopped{false};  // the receiver failed, or was let go
  // Written by the receiver, read once it has joined.
  std::string order;
  core::Status failed;
  std::thread receiver([&] {
    Encoder::CodedPicture got;
    while (back < kPictures && !stopped) {
      const core::Result<bool> r = e.receive(got, 1'000'000);
      if (!r.ok()) {
        failed = r.status();
        stopped = true;
        return;
      }
      if (!*r) continue;
      if (got.pts != pts_of(100 + back) && order.empty()) {
        order = "out of order";
      }
      ++back;
    }
  });
  core::Status sent;
  for (std::uint32_t i = 0; i < kPictures && sent.ok() && !stopped;) {
    if (e.in_flight() == Encoder::kInFlight) {
      std::this_thread::yield();
      continue;
    }
    sent =
        send_pixel_buffer(e, ring_[i % kRing].get(), i == 0, pts_of(100 + i));
    ++i;
  }
  if (!sent.ok()) stopped = true;  // let the receiver go
  receiver.join();
  if (!sent.ok()) return "C13: two threads, a send failed: " + said(sent);
  if (!failed.ok())
    return "C13: two threads, a picture failed: " + said(failed);
  if (!order.empty()) return "C13: two threads, " + order;
  return std::string();
}

core::Result<std::string> Run::all() {
  format_ = o_.color.full_range
                ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  VKC_ASSIGN(Encoder e, Encoder::create(o_));
  using Rule = core::Result<std::string> (Run::*)(Encoder&);
  const Rule rules[] = {&Run::stream,  &Run::drop, &Run::lose,
                        &Run::buffers, &Run::move, &Run::in_flight};
  VKC_ASSIGN(const std::string refused, refusals());
  if (!refused.empty()) return refused;
  for (const Rule rule : rules) {
    VKC_ASSIGN(const std::string b, (this->*rule)(e));
    if (!b.empty()) return b;
    // C6 decodes the main stream, once it is coded.
    if (rule == &Run::stream) {
      VKC_ASSIGN(const std::string d, decodes());
      if (!d.empty()) return d;
    }
  }
  return std::string();
}
#endif

}  // namespace

core::Result<std::string> hevc_encoder_breach(const HevcEncoderCase& c) try {
  if (c.pictures < 4) {
    return core::Status::invalid_argument(
        "hevc_encoder_breach: a case of fewer than 4 pictures");
  }
  {
    // No hardware path here: create's own refusal, for the runner.
    VKC_ASSIGN(const Encoder probe, Encoder::create(c.options));
  }
#if defined(__APPLE__)
  if (!c.pixel_buffers) {
    return core::Status::invalid_argument(
        "hevc_encoder_breach: this build's encoder takes pixel buffers only "
        "(HevcEncoderCase::pixel_buffers)");
  }
  Run run(c);
  return run.all();
#else
  return core::Status::invalid_argument(
      "hevc_encoder_breach: this build's encoder has no way in to run");
#endif
} catch (...) {
  return core::exception_status("hevc_encoder_breach",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor::testing
