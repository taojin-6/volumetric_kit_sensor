// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The encoder rules (testing/hevc_conformance.hpp): each is checked through
// HevcEncoder's public calls, with the stream read by sensor's syntax
// readers (hevc_syntax.hpp, testing/hevc_slices.hpp) and decoded by
// HevcDecoder, each picture a pixel buffer or a target the device writes
// (Nv12Writer). Written apart from the encoder, so a rule here is never the
// encoder's own code agreeing with itself.

#include "volumetric_kit/sensor/testing/hevc_conformance.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/image.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/testing/hevc_slices.hpp"
#include "volumetric_kit/sensor/testing/nv12_pattern.hpp"
#include "volumetric_kit/sensor/testing/nv12_writer.hpp"
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

// The session's options: for targets, opened on the case's device.
Encoder::Options options_of(const HevcEncoderCase& c) {
  Encoder::Options o = c.options;
  if (!c.pixel_buffers) {
    o.device = c.device;
    o.allocator = c.allocator;
  }
  return o;
}

// The rules' run: a build whose encoder takes pixel buffers (Apple's).
#if defined(__APPLE__)
using Code = core::Status::Code;

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kRing = 8;  // distinct pictures a run sends
// How long C13b holds the value the batch that writes the target waits for:
// this many times one picture's whole path, as the run measures it, and at
// least kHoldFloor, so a coding thread that did not wait has coded the
// picture well within it.
constexpr int kHoldPictures = 10;
constexpr auto kHoldFloor = std::chrono::milliseconds(20);
// Past the hold, when C13b's watchdog sets the value itself: only a send
// that waits for its writer lets it.
constexpr auto kWatchdog = std::chrono::seconds(5);

std::int64_t pts_of(std::uint32_t i) { return 1000 + 333 * std::int64_t{i}; }

std::uint64_t nanoseconds(Clock::duration d) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

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

// @p picture, a test pattern's, as a pixel buffer (pixel_buffer): '420f' or
// '420v' as @p tags' range, tagged with its matrix, transfer and primaries
// and no siting.
core::Result<Pixels> make_pixels(const HostNv12& picture,
                                 const ColorCodes& tags) {
  VKC_ASSIGN(CVPixelBufferRef made, pixel_buffer(picture, tags));
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

// What a target says its codes mean, as C6 compares it with the decoded
// pictures'.
struct Described {
  float kr = 0;
  float kb = 0;
  bool full_range = false;
  ChromaLocation siting = ChromaLocation::Left;
};

class Run {
 public:
  explicit Run(const HevcEncoderCase& c) : c_(c), o_(options_of(c)) {}

  core::Result<std::string> all();

 private:
  bool targets() const { return !c_.pixel_buffers; }
  // Picture @p i of the ring, made the first time it is asked for.
  core::Result<CVPixelBufferRef> picture(std::uint32_t i);
  // Targets: the writer, the timeline the writing batches set, and the
  // ring's pictures on the device.
  core::Status ready_targets();
  // Picture @p i of the ring written into a target taken now, by a batch
  // that sets written_'s next value, which written() then names; @p gate's
  // value first, when given. The batch is kept in pending_.
  core::Result<YuvImage> write_target(Encoder& e, std::uint32_t i,
                                      core::TimelinePoint gate = {});
  core::TimelinePoint written() { return {&written_, value_}; }
  // Picture @p i sent as @p pts: a pixel buffer, or a target written.
  core::Status send(Encoder& e, std::uint32_t i, bool key, std::int64_t pts) {
    if (targets()) {
      VKC_TRY(write_target(e, i).status());
      return e.send(key, pts, written());
    }
    VKC_ASSIGN(CVPixelBufferRef p, picture(i));
    return send_pixel_buffer(e, p, key, pts);
  }
  core::Status send(Encoder& e, std::uint32_t i, bool key) {
    return send(e, i, key, pts_of(i));
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
  core::Result<std::string> retake(Encoder& e);
  core::Result<std::string> waits(Encoder& e);

  const HevcEncoderCase& c_;
  const Encoder::Options o_;
  OSType format_ = 0;
  Pixels ring_[kRing];
  // The main stream, as received, for C6.
  std::vector<std::vector<std::uint8_t>> units_;
  // The first target's description, for C6.
  std::optional<Described> described_;
  // Targets. The pending batches last, so they go first: they name the
  // timelines, the pictures and the writer's sets.
  std::optional<Nv12Writer> writer_;
  core::TimelineSemaphore written_;
  core::TimelineSemaphore gate_;  // C13b's helper sets it
  std::uint64_t value_ = 0;       // written_'s last value
  std::uint64_t gate_value_ = 0;
  core::Buffer planes_[kRing];
  core::PendingBatch pending_[kRing];
};

core::Status Run::ready_targets() {
  VKC_ASSIGN(writer_, Nv12Writer::create(*c_.device));
  VKC_ASSIGN(written_, core::TimelineSemaphore::create(*c_.device));
  VKC_ASSIGN(gate_, core::TimelineSemaphore::create(*c_.device));
  core::CommandBatch batch(*c_.device, *c_.allocator);
  for (std::uint32_t i = 0; i < kRing; ++i) {
    VKC_ASSIGN(planes_[i],
               Nv12Writer::upload(batch, *c_.allocator,
                                  test_pattern(o_.width, o_.height, i)));
  }
  return batch.submit();
}

core::Result<YuvImage> Run::write_target(Encoder& e, std::uint32_t i,
                                         core::TimelinePoint gate) {
  const std::uint64_t v = value_ + 1;
  core::PendingBatch& kept = pending_[v % kRing];
  // The batch kRing writes ago, whose descriptor set the writer takes next:
  // done, as its picture was coded or given up long since.
  VKC_TRY(kept.wait());
  core::CommandBatch batch(*c_.device, *c_.allocator);
  VKC_ASSIGN(YuvImage t, e.target(batch));
  VKC_TRY(writer_->record(batch, planes_[i % kRing], t));
  std::vector<core::TimelinePoint> waits;
  if (gate.semaphore != nullptr) waits.push_back(gate);
  VKC_ASSIGN(kept, batch.submit_async(waits, {{&written_, v}}));
  value_ = v;
  if (!described_) {
    described_ = Described{t.kr, t.kb, t.full_range, t.chroma_location};
  }
  return t;
}

core::Result<CVPixelBufferRef> Run::picture(std::uint32_t i) {
  Pixels& made = ring_[i % kRing];
  if (made == nullptr) {
    VKC_ASSIGN(made, make_pixels(test_pattern(o_.width, o_.height, i % kRing),
                                 o_.color));
  }
  return made.get();
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
  bad.push_back({"an odd height", with([](auto& o) { o.height += 1; }), inv});
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
  // C2: the first must be a key. A refused target is kept, and sent as one.
  const core::Status first = send(e, 0, false);
  if (first.domain() != Code::InvalidArgument || e.in_flight() != 0) {
    return "C2: a P-picture first gave " + said(first) +
           ", not InvalidArgument";
  }
  std::uint32_t from = 0;
  if (targets()) {
    const core::Status kept = e.send(true, pts_of(0), written());
    if (!kept.ok()) {
      return "C2: the target a refused send kept, sent as a key, gave " +
             said(kept);
    }
    from = 1;
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
  for (std::uint32_t i = from; i < n; ++i) {
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
  VKC_ASSIGN(const std::string middle, run(c_.pictures / 2));
  if (!middle.empty()) return middle;
  // The target is described as the stream's pictures decode: its codes are
  // written by the matrix, range and siting the VUI states.
  if (described_) {
    const ColorCodes want = normalized(o_.color);
    const YcbcrWeights weights =
        ycbcr_weights(resolve_matrix(want.matrix, o_.height));
    if (described_->kr != weights.kr || described_->kb != weights.kb ||
        described_->full_range != want.full_range ||
        described_->siting != resolve_chroma_location(want.chroma_location)) {
      return std::string(
          "C6: a target is described by another matrix, range or siting than "
          "its pictures decode with");
    }
  }
  return std::string();
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
  // Lost with two in flight: each comes back coded or failed, as the loss
  // finds it, and none coded after one that failed (it would follow a gap).
  VKC_TRY(send(e, 5, false));
  VKC_TRY(send(e, 6, false));
  e.inject(Encoder::Fault::Lose);
  bool gap = false;
  for (const std::uint32_t i : {5U, 6U}) {
    const core::Result<bool> r = e.receive(coded);
    if (r.ok() && !*r) return std::string("C8: a picture vanished");
    if (!r.ok()) {
      if (r.status().domain() == Code::InvalidArgument) {
        return "C8: picture " + std::to_string(i) +
               ", sent before the loss, gave " + said(r.status());
      }
      gap = true;
    } else if (gap || coded.pts != pts_of(i)) {
      return "C8: picture " + std::to_string(i) +
             (gap ? " came back coded after one that failed"
                  : " came back out of order");
    }
  }
  if (!e.lost() || !e.key_owed()) {
    return std::string("C8: lost in flight, the encoder says ") +
           (e.lost() ? "no key is owed" : "it is not lost");
  }
  VKC_TRY(send(e, 7, true));
  VKC_ASSIGN(const bool again, e.receive(coded));
  if (!again || !coded.key || e.lost()) {
    return std::string("C8: the session did not reopen at a key");
  }
  // Lost while a picture codes, and a key sent at once: the reopen waits for
  // that picture, which comes back coded or failed, and the key comes back
  // coded on the new session. A round waits no time before the loss and one
  // half a picture's, so one likely finds the picture coding at any size;
  // what is checked holds whatever each finds.
  const auto start = std::chrono::steady_clock::now();
  VKC_TRY(send(e, 5, false));
  VKC_ASSIGN(const bool timed, e.receive(coded));
  if (!timed) return std::string("C8: a picture vanished");
  const auto half = (std::chrono::steady_clock::now() - start) / 2;
  for (const auto wait : {decltype(half)::zero(), half}) {
    VKC_TRY(send(e, 5, false));
    std::this_thread::sleep_for(wait);
    e.inject(Encoder::Fault::Lose);
    VKC_TRY(send(e, 6, true));
    if (e.lost()) return std::string("C8: a key did not reopen the session");
    const core::Result<bool> before = e.receive(coded);
    if (before.ok() ? !*before || coded.pts != pts_of(5) || coded.key
                    : before.status().domain() == Code::InvalidArgument) {
      return "C8: the picture coding at the loss came back " +
             (before.ok() ? std::string(*before ? "mislabelled" : "nothing")
                          : said(before.status()));
    }
    const core::Result<bool> key = e.receive(coded);
    if (!key.ok() || !*key || !coded.key || coded.pts != pts_of(6)) {
      return "C8: the key sent while the lost session coded came back " +
             (key.ok() ? std::string("wrong") : said(key.status()));
    }
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
  if (targets()) {
    // A target taken and not sent: a buffer waits for it. Sent, it codes.
    VKC_TRY(write_target(e, 1).status());
    VKC_ASSIGN(CVPixelBufferRef p, picture(1));
    if (std::string b = refused("a buffer while a target is taken", p);
        !b.empty()) {
      return b;
    }
    VKC_TRY(e.send(true, pts_of(1), written()));
    Encoder::CodedPicture coded;
    const core::Result<bool> got = e.receive(coded);
    if (!got.ok() || !*got || coded.pts != pts_of(1)) {
      return "C11: the target a buffer waited for came back " +
             (got.ok() ? std::string(*got ? "mislabelled" : "nothing")
                       : said(got.status()));
    }
  } else if (o_.device == nullptr && c_.device != nullptr) {
    // A session with no device has no target, and codes buffers still.
    core::CommandBatch batch(*c_.device, *c_.allocator);
    const core::Result<YuvImage> t = e.target(batch);
    if (t.ok() || t.status().domain() != Code::InvalidArgument) {
      return "C11: a session with no device handed out a target, or gave " +
             said(t.status());
    }
    if (e.send(true, 0, {}).domain() != Code::InvalidArgument) {
      return std::string("C11: a send with no target taken was taken");
    }
  }
  {
    VKC_ASSIGN(Pixels small,
               make_pixels(test_pattern(o_.width - 2, o_.height, 0), o_.color));
    if (std::string b = refused("another size", small.get()); !b.empty()) {
      return b;
    }
  }
  // The session's size, made once for the buffers below.
  const HostNv12 base = test_pattern(o_.width, o_.height, 0);
  {
    ColorCodes other = o_.color;
    other.full_range = !other.full_range;
    VKC_ASSIGN(Pixels wrong, make_pixels(base, other));
    if (std::string b = refused("the other range", wrong.get()); !b.empty()) {
      return b;
    }
  }
  {
    ColorCodes tags = o_.color;
    tags.matrix = normalized(tags).matrix == 1 ? 6 : 1;
    VKC_ASSIGN(Pixels tagged, make_pixels(base, tags));
    if (std::string b = refused("another matrix tag", tagged.get());
        !b.empty()) {
      return b;
    }
  }
  {
    VKC_ASSIGN(Pixels sited, make_pixels(base, o_.color));
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
  {
    // A buffer with no colour tags is the session's colour, which the VUI
    // states as for any other.
    ColorCodes none = o_.color;
    none.matrix = 2;
    none.transfer = 2;
    none.primaries = 2;
    VKC_ASSIGN(Pixels bare, make_pixels(base, none));
    const core::Status sent = send_pixel_buffer(e, bare.get(), true, 0);
    if (!sent.ok())
      return "C11: a buffer with no colour tags gave " + said(sent);
    Encoder::CodedPicture coded;
    const core::Result<bool> got = e.receive(coded);
    if (!got.ok() || !*got) {
      return "C11: a buffer with no colour tags came back " +
             (got.ok() ? std::string("nothing") : said(got.status()));
    }
    SliceReader reader;
    const core::Result<SliceReader::Picture> p =
        reader.read(coded.access_unit.data(), coded.access_unit.size());
    if (!p.ok() || !reader.sequence()) {
      return std::string("C11: the untagged buffer's IDR does not parse");
    }
    const ColorCodes want = normalized(o_.color);
    if (!same(reader.sequence()->sps.color, want)) {
      return "C11: a buffer with no colour tags is coded as " +
             codes_text(reader.sequence()->sps.color) + ", not " +
             codes_text(want);
    }
  }
  return std::string();
}

core::Result<std::string> Run::move(Encoder& e) {
  Encoder::CodedPicture coded;
  if (targets()) {
    // Moved between target() and send(): the target moves with it.
    VKC_TRY(write_target(e, 8).status());
    Encoder holder(std::move(e));
    VKC_TRY(holder.send(holder.key_owed(), pts_of(8), written()));
    e = std::move(holder);
  } else {
    VKC_TRY(send(e, 8, e.key_owed()));
  }
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
  if (targets()) {
    core::CommandBatch batch(*c_.device, *c_.allocator);
    const core::Result<YuvImage> refused = e.target(batch);
    if (refused.ok() || refused.status().domain() != Code::InvalidArgument) {
      return "C13: a third target with two in flight gave " +
             said(refused.status());
    }
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
  // A drop fails the one sent after it, and a key is owed. That one is sent
  // at once, while the drop codes; a loaded machine may finish the drop
  // first, which owes the key before it is sent, so it is refused: a race
  // the test lost, tried again, and at least one round must send it.
  constexpr int kRounds = 5;
  bool behind = false;
  for (int round = 0; round < kRounds && !behind; ++round) {
    if (e.key_owed()) {
      VKC_TRY(send(e, 11, true));
      VKC_ASSIGN(const bool got, e.receive(coded));
      if (!got || !coded.key) return std::string("C13: the key before a drop");
    }
    e.inject(Encoder::Fault::Drop);
    VKC_TRY(send(e, 12, false));
    const core::Status next = send(e, 13, false);
    if (!next.ok() &&
        (next.domain() != Code::InvalidArgument || !e.key_owed())) {
      return "C13: the picture sent after a drop gave " + said(next);
    }
    behind = next.ok();
    const core::Result<bool> dropped = e.receive(coded);
    if (dropped.ok() || dropped.status().domain() != Code::IoError) {
      return "C13: a dropped picture in flight came back " +
             (dropped.ok() ? std::string("coded") : said(dropped.status()));
    }
    if (behind) {
      const core::Result<bool> after = e.receive(coded);
      if (after.ok() || after.status().domain() != Code::IoError) {
        return "C13: after a drop, the picture sent next came back " +
               (after.ok() ? std::string("coded") : said(after.status()));
      }
    }
    if (!e.key_owed() || e.lost()) {
      return std::string("C13: after a drop in flight, ") +
             (e.lost() ? "the session is lost" : "no key is owed");
    }
  }
  if (!behind) {
    return "C13: in " + std::to_string(kRounds) +
           " rounds no picture was sent before the drop ahead of it coded";
  }
  // Sent on this thread, received on another, which waits a finite time:
  // it must sleep, not spin, while nothing is in flight, so the sender
  // starts after a few waits. Each false return takes a whole wait, so
  // they number at most the receiver's time over the wait.
  constexpr std::uint32_t kPictures = kRing;
  constexpr auto kWait = std::chrono::milliseconds(5);
  if (!targets()) {
    for (std::uint32_t i = 0; i < kRing; ++i) VKC_TRY(picture(i).status());
  }
  std::atomic<std::uint32_t> back{0};
  std::atomic<bool> stopped{false};  // the receiver failed, or was let go
  // Written by the receiver, read once it has joined.
  std::string order;
  core::Status failed;
  std::int64_t empty = 0;
  std::chrono::steady_clock::duration took{};
  std::thread receiver([&] {
    Encoder::CodedPicture got;
    const auto from = std::chrono::steady_clock::now();
    while (back < kPictures && !stopped) {
      const core::Result<bool> r = e.receive(
          got,
          std::chrono::duration_cast<std::chrono::nanoseconds>(kWait).count());
      if (!r.ok()) {
        failed = r.status();
        stopped = true;
        return;
      }
      if (!*r) {
        ++empty;
        continue;
      }
      if (got.pts != pts_of(100 + back) && order.empty()) {
        order = "out of order";
      }
      ++back;
    }
    took = std::chrono::steady_clock::now() - from;
  });
  std::this_thread::sleep_for(3 * kWait);
  core::Status sent;
  for (std::uint32_t i = 0; i < kPictures && sent.ok() && !stopped;) {
    if (e.in_flight() == Encoder::kInFlight) {
      std::this_thread::yield();
      continue;
    }
    sent = send(e, i % kRing, i == 0, pts_of(100 + i));
    ++i;
  }
  if (!sent.ok()) stopped = true;  // let the receiver go
  receiver.join();
  if (!sent.ok()) return "C13: two threads, a send failed: " + said(sent);
  if (!failed.ok())
    return "C13: two threads, a picture failed: " + said(failed);
  if (!order.empty()) return "C13: two threads, " + order;
  if (empty * kWait > took + kWait) {
    return "C13: two threads, a receive waiting " +
           std::to_string(kWait.count()) +
           " ms returned at once: " + std::to_string(empty) + " empty in " +
           std::to_string(
               std::chrono::duration_cast<std::chrono::milliseconds>(took)
                   .count()) +
           " ms";
  }
  return std::string();
}

core::Result<std::string> Run::retake(Encoder& e) {
  if (!targets()) return std::string();
  Encoder::CodedPicture coded;
  // What a target is: NV12 plane images of the session's size, which a
  // kernel stores into, on the device's own queue family.
  core::CommandBatch batch(*c_.device, *c_.allocator);
  {
    VKC_ASSIGN(const YuvImage t, e.target(batch));
    const auto plane = [](const YuvImage& y, int p, VkFormat format) {
      const auto& image = y.image[p];
      return image != nullptr && image->format() == format &&
             image->layout() == VK_IMAGE_LAYOUT_GENERAL &&
             (image->usage() & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
             image->view() != VK_NULL_HANDLE;
    };
    if (t.layout != YuvLayout::Nv12 || t.width != o_.width ||
        t.height != o_.height || t.device != nullptr ||
        t.queue_family != kQueueFamilyIgnored || t.ready != nullptr ||
        !plane(t, 0, VK_FORMAT_R8_UNORM) ||
        !plane(t, 1, VK_FORMAT_R8G8_UNORM)) {
      return std::string(
          "C9: a target is not NV12 R8 and R8G8 storage images in GENERAL of "
          "the session's size, on the device's queue family");
    }
  }
  // Taken twice before a send, the first is given up: one picture is sent,
  // and one access unit comes back.
  VKC_TRY(write_target(e, 2).status());
  VKC_TRY(write_target(e, 3).status());
  VKC_TRY(e.send(e.key_owed(), pts_of(3), written()));
  if (e.in_flight() != 1) {
    return "C9: a target taken twice put " + std::to_string(e.in_flight()) +
           " pictures in flight";
  }
  VKC_ASSIGN(const bool got, e.receive(coded));
  if (!got || coded.pts != pts_of(3)) {
    return std::string("C9: the target taken second did not come back");
  }
  VKC_ASSIGN(const bool more, e.receive(coded, 0));
  if (more || e.in_flight() != 0) {
    return std::string("C9: a target given up came back too");
  }
  // A target is sent once: another send needs another target.
  const core::Status again = e.send(true, pts_of(4), written());
  if (again.domain() != Code::InvalidArgument) {
    return "C9: a send with no target taken gave " + said(again);
  }
  return std::string();
}

core::Result<std::string> Run::waits(Encoder& e) {
  if (!targets()) return std::string();
  // C13b: the batch that writes the target waits for a value the run holds.
  // send returns at once, before the value is reached; nothing comes back
  // while it is held, many times one picture's whole path; and the picture
  // comes back once the value is set.
  Encoder::CodedPicture coded;
  const Clock::time_point start = Clock::now();
  VKC_TRY(send(e, 4, e.key_owed()));
  VKC_ASSIGN(const bool timed, e.receive(coded));
  if (!timed) return std::string("C13b: a picture vanished");
  const Clock::duration hold = std::max<Clock::duration>(
      kHoldFloor, kHoldPictures * (Clock::now() - start));
  const std::uint64_t gate = ++gate_value_;
  VKC_TRY(write_target(e, 5, {&gate_, gate}).status());
  const std::uint64_t value = value_;
  // Should send wait for the writer, a watchdog sets the value, well past
  // the hold, so the run fails rather than hangs. The run sets it otherwise,
  // whatever came of the send: the batch waits for it.
  std::mutex mutex;
  std::condition_variable released;
  bool set = false;
  core::Status signalled;
  const auto set_gate = [&] {
    const std::lock_guard<std::mutex> lock(mutex);
    if (!set) signalled = gate_.signal(gate);
    set = true;
  };
  std::thread watchdog([&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (released.wait_for(lock, hold + kWatchdog, [&] { return set; })) return;
    signalled = gate_.signal(gate);
    set = true;
  });
  const core::Status sent = e.send(e.key_owed(), pts_of(5), written());
  const core::Result<std::uint64_t> at_send = written_.value();
  core::Result<bool> early = false;
  if (sent.ok()) early = e.receive(coded, nanoseconds(hold));
  set_gate();
  released.notify_all();
  watchdog.join();
  VKC_TRY(signalled);
  if (!sent.ok()) return "C13b: the send gave " + said(sent);
  VKC_TRY(at_send.status());
  if (*at_send >= value) {
    return std::string(
        "C13b: the writer's value was reached when send returned: send "
        "waited for the writer");
  }
  if (!early.ok() || *early) {
    return early.ok()
               ? std::string(
                     "C13b: the access unit came back while the value its "
                     "writer waits for was held: the picture was coded "
                     "before it was written")
               : "C13b: while its writer waited, the picture gave " +
                     said(early.status());
  }
  const core::Result<bool> got = e.receive(coded);
  if (!got.ok() || !*got || coded.pts != pts_of(5)) {
    return "C13b: the picture came back " +
           (got.ok() ? std::string(*got ? "mislabelled" : "nothing")
                     : said(got.status()));
  }
  return std::string();
}

core::Result<std::string> Run::all() {
  format_ = o_.color.full_range
                ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
                : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  // create's own refusal, for the runner, comes first.
  VKC_ASSIGN(Encoder e, Encoder::create(o_));
  if (targets()) VKC_TRY(ready_targets());
  using Rule = core::Result<std::string> (Run::*)(Encoder&);
  const bool all = c_.rules == HevcEncoderCase::Rules::All;
  std::vector<Rule> rules = {&Run::stream};
  if (all) rules.insert(rules.end(), {&Run::drop, &Run::lose});
  rules.push_back(&Run::buffers);
  if (all) {
    rules.insert(rules.end(),
                 {&Run::move, &Run::in_flight, &Run::retake, &Run::waits});
    VKC_ASSIGN(const std::string refused, refusals());
    if (!refused.empty()) return refused;
  }
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
  if (!c.pixel_buffers && (c.device == nullptr || c.allocator == nullptr)) {
    return core::Status::invalid_argument(
        "hevc_encoder_breach: targets need the case's device and allocator");
  }
#if defined(__APPLE__)
  Run run(c);
  return run.all();
#else
  {
    // No hardware path here: create's own refusal, for the runner.
    VKC_ASSIGN(const Encoder probe, Encoder::create(options_of(c)));
  }
  return core::Status::invalid_argument(
      "hevc_encoder_breach: this build's encoder has no way in to run");
#endif
} catch (...) {
  return core::exception_status("hevc_encoder_breach",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor::testing
