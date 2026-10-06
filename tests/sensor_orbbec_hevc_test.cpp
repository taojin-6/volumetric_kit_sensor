// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's H.265 colour decoder, with no camera: committed clips
// wrapped in SDK frames, each paired with a depth frame and dated at 30 fps,
// decoded on this machine's hardware onto the device. Every pair comes out,
// in display order, with its timestamps and depth, its picture carried in its
// frame and gone with it, an unlabelled stream's colour described as the
// Femto Mega codes it (BT.601 full range), a labelled one's as it says. A
// frame lost from the stream, or empty, costs the frames up to the next key
// frame, a pause in the timestamps costs nothing, and decoding starts at the
// first key frame; a stream the hardware refuses stops the decoder. The
// frame-index gate is tested on its own, since a test cannot set an SDK
// frame's index.
//
// Where no device path opens the test skips; VR_TEST_HEVC_BACKEND, which CI
// sets on the legs that promise one, makes it fail instead.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "device_picture_readback.hpp"
#include "hevc_color.hpp"
#include "no_device.hpp"
#include "picture_frames.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/recon/core/device_requirements.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::recon::sensor;
namespace orbbec = volumetric_kit::recon::sensor::orbbec;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 144;
constexpr std::uint64_t kStartUs = 1000000;
constexpr std::uint64_t kPeriodUs = 33333;  // 30 fps

// The patch clip (tools/make_hevc_fixtures.sh) four ways: with no matrix,
// as the camera sends it; labelled BT.709 limited; with B-frames, whose
// first 8 access units are the patch frames, sent out of display order; and
// in open GOPs, 16 frames with a CRA every 6.
constexpr const char* kUnlabelled = VR_HEVC_DATA "/unlabelled_256x144.h265";
constexpr const char* kLabelled = VR_HEVC_DATA "/patches_256x144.h265";
constexpr const char* kBFrames = VR_HEVC_DATA "/refused_256x144.h265";
constexpr const char* kOpenGop = VR_HEVC_DATA "/open_gop_256x144.h265";

// The clip's pattern, as tools/make_hevc_fixtures.sh draws it.
int patch(int column, int row, int frame) {
  return (column + frame + 3 * row) % 8;
}

using Units = std::vector<std::vector<std::uint8_t>>;

Units access_units(const char* path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
  std::vector<std::size_t> starts;
  for (std::size_t i = 0; i + 3 < bytes.size(); ++i) {
    if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 &&
        ((bytes[i + 3] >> 1) & 0x3f) == 35) {  // access unit delimiter
      starts.push_back(i > 0 && bytes[i - 1] == 0 ? i - 1 : i);
    }
  }
  Units units;
  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t end =
        k + 1 < starts.size() ? starts[k + 1] : bytes.size();
    units.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(starts[k]),
                       bytes.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return units;
}

// Access unit `f`'s pair, dated in `slot`, its colour numbered f + 1 -- as
// the camera numbers its frames, in the order it sends them -- in the system
// timestamp, where the decoder's frame_index reads it.
std::shared_ptr<ob::FrameSet> pair(const std::vector<std::uint8_t>& unit, int f,
                                   int slot, bool with_color = true,
                                   bool with_depth = true) {
  const std::uint64_t t =
      kStartUs + static_cast<std::uint64_t>(slot) * kPeriodUs;
  auto depth =
      ob::FrameFactory::createVideoFrame(OB_FRAME_DEPTH, OB_FORMAT_Y16, 4, 4);
  ob::FrameHelper::setFrameDeviceTimestampUs(depth, t);
  auto set = ob::FrameFactory::createFrameSet();
  if (with_depth) set->pushFrame(depth);
  if (with_color) {
    auto color =
        ob::FrameFactory::createFrame(OB_FRAME_COLOR, OB_FORMAT_H265,
                                      static_cast<std::uint32_t>(unit.size()));
    if (!unit.empty()) {
      color->updateData(unit.data(), static_cast<std::uint32_t>(unit.size()));
    }
    ob::FrameHelper::setFrameDeviceTimestampUs(color, t);
    color->setSystemTimestampUs(static_cast<std::uint64_t>(f) + 1);
    set->pushFrame(color);
  }
  return set;
}

using Pairs = std::vector<std::shared_ptr<ob::FrameSet>>;

// `frames` (access unit indices; -1 is a pair with no colour, -f - 2 frame
// f's colour with no depth) as the SDK would hand them over, dated by `slots`
// (by default each frame's own index).
Pairs pairs(const Units& units, const std::vector<int>& frames,
            std::vector<int> slots = {}) {
  if (slots.empty()) slots = frames;
  Pairs out;
  for (std::size_t k = 0; k < frames.size(); ++k) {
    const int i = frames[k];
    if (i == -1) {
      out.push_back(pair(units[0], 0, 0, false));
    } else if (i <= -2) {  // frame -i - 2, colour without its depth
      const int f = -i - 2;
      out.push_back(
          pair(units[static_cast<std::size_t>(f)], f, f, true, false));
    } else {
      out.push_back(pair(units[static_cast<std::size_t>(i)], i, slots[k]));
    }
  }
  return out;
}

// The device the decoders run on, set by main.
vkc::Device* g_device = nullptr;
vkc::Allocator* g_allocator = nullptr;

// A pair the decoder handed on, its picture read back as it arrived, so no
// picture is held past its pair.
struct Out {
  std::uint64_t depth_us = 0;
  std::uint64_t color_us = 0;
  std::optional<sensor::DecodedPicture> meta;  // its storage let go
  bool images = false;  // VideoToolbox's images, not a buffer
  std::vector<std::uint8_t> planes[3];
};

Out read_out(const ob::FrameSet& set) {
  Out o;
  const auto depth = set.getDepthFrame();
  const auto color = set.getColorFrame();
  if (depth != nullptr) o.depth_us = depth->getTimeStampUs();
  if (color == nullptr) return o;
  o.color_us = color->getTimeStampUs();
  std::optional<sensor::DecodedPicture> p = orbbec::device_picture(*color);
  if (!p) return o;
  vr_test::read_device_picture(*p, *g_device, *g_allocator, o.planes);
  o.images = p->image[0] != nullptr;
  p->device.reset();
  p->image[0].reset();
  p->image[1].reset();
  o.meta = std::move(p);
  return o;
}

struct Collected {
  std::mutex mutex;
  std::vector<Out> out;
  std::vector<std::shared_ptr<ob::FrameSet>> frames;  // kept only when asked
  std::size_t size() {
    std::lock_guard<std::mutex> lock(mutex);
    return out.size();
  }
};

// Push `in`, wait for `expect` pairs out, and stop.
struct Run {
  std::vector<Out> out;
  std::vector<std::shared_ptr<ob::FrameSet>> frames;  // with keep_frames
  std::uint64_t lost = 0;
  vkc::Status failure;
};
Run run(const Pairs& in, std::size_t expect, bool keep_frames = false) {
  auto collected = std::make_shared<Collected>();
  orbbec::HevcColorDecoder::Options options;
  options.fps = 30;
  options.who = "test";
  options.frame_index = [](const ob::Frame& frame) {
    return frame.getSystemTimeStampUs();
  };
  options.device = g_device;
  options.allocator = g_allocator;
  auto decoder = orbbec::HevcColorDecoder::start(
      options, [collected, keep_frames](std::shared_ptr<ob::FrameSet> set) {
        Out o = read_out(*set);
        std::lock_guard<std::mutex> lock(collected->mutex);
        collected->out.push_back(std::move(o));
        if (keep_frames) collected->frames.push_back(std::move(set));
      });
  if (!decoder) {
    std::fprintf(stderr, "%s\n", decoder.status().message().c_str());
    std::exit(1);
  }
  for (const auto& p : in) decoder.value()->push(p);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (collected->size() < expect && decoder.value()->failure().ok() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  // Let anything that should not come out have its chance to.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  Run r;
  r.failure = decoder.value()->failure();
  decoder.value()->stop();
  r.lost = decoder.value()->lost();
  std::lock_guard<std::mutex> lock(collected->mutex);
  r.out = std::move(collected->out);
  r.frames = std::move(collected->frames);
  return r;
}

// The slot a decoded pair was dated in, from its colour's timestamp.
int slot_of(const Out& o) {
  return static_cast<int>((o.color_us - kStartUs) / kPeriodUs);
}

// A decoded pair: clip frame `frame`'s picture, dated in `slot`, on the
// device where this platform's hardware leaves it, each patch's value as the
// clip was made, described as the Femto Mega codes its unlabelled stream
// unless `matrix` says otherwise.
int check_pair(
    const Out& o, int frame, int slot,
    sensor::VideoColorMatrix matrix = sensor::VideoColorMatrix::Bt601,
    bool full_range = true) {
  const std::uint64_t t =
      kStartUs + static_cast<std::uint64_t>(slot) * kPeriodUs;
  CHECK(o.depth_us == t);
  CHECK(o.color_us == t);
  CHECK(o.meta.has_value());
  const sensor::DecodedPicture& p = *o.meta;
  CHECK(p.width == static_cast<std::uint32_t>(kWidth));
  CHECK(p.height == static_cast<std::uint32_t>(kHeight));
  CHECK(p.layout == sensor::VideoPixelLayout::Nv12);
#if defined(__APPLE__)
  CHECK(o.images);
#else
  CHECK(!o.images);
#endif
  CHECK(p.matrix == matrix && p.full_range == full_range);
  // Neither clip declares a transfer or primaries ColorEncoding cannot name.
  CHECK(p.encoding.has_value() && is_canonical(*p.encoding));
  const int cw = kWidth / 2;
  CHECK(o.planes[0].size() == static_cast<std::size_t>(kWidth * kHeight));
  CHECK(o.planes[1].size() == static_cast<std::size_t>(cw * (kHeight / 2)));
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 8; ++col) {
      const int k = patch(col, row, frame);
      const int x = 32 * col + 16;
      const int yy = 72 * row + 36;
      const int c = (yy / 2) * cw + x / 2;
      const int y = o.planes[0][yy * kWidth + x];
      const int cb = o.planes[1][c];
      const int cr = o.planes[2][c];
      if (std::abs(y - (40 + 24 * k)) > 2 ||
          std::abs(cb - (64 + 16 * ((3 * k) % 8))) > 2 ||
          std::abs(cr - (64 + 16 * ((5 * k) % 8))) > 2) {
        std::fprintf(stderr, "frame %d patch %d,%d: Y'CbCr %d %d %d\n", frame,
                     col, row, y, cb, cr);
        CHECK(false);
      }
    }
  }
  return 0;
}

// Each picture comes out carried in its frame, which has no pixels of its
// own, and placed in a frame's colour as the driver places it; it lives as
// long as its frame, which a copy of the frame does not extend.
int test_hands_on_device_pictures() {
  Run r =
      run(pairs(access_units(kUnlabelled), {0, 1, 2, 3, 4, 5, 6, 7}), 8, true);
  CHECK(r.frames.size() == 8 && r.lost == 0);
  std::shared_ptr<const void> held;
  std::shared_ptr<ob::Frame> copied;
  for (int f = 0; f < 8; ++f) {
    const auto color = r.frames[static_cast<std::size_t>(f)]->getColorFrame();
    CHECK(color != nullptr);
    const std::optional<sensor::DecodedPicture> p =
        orbbec::device_picture(*color);
    CHECK(p.has_value());
    CHECK(color->as<ob::VideoFrame>()->getWidth() == 0);  // no pixels
    CHECK((p->device != nullptr) != (p->image[0] != nullptr));
    sensor::YuvImage placed;
    orbbec::place_device_color(*p, &placed);
    CHECK(placed.layout == sensor::YuvLayout::Nv12);
    if (p->device != nullptr) {
      CHECK(placed.device == p->device && placed.image[0] == nullptr);
      CHECK(placed.queue_family == sensor::kQueueFamilyExternal);
      for (int i = 0; i < 2; ++i) {
        CHECK(placed.offset[i] == p->offset[i] &&
              placed.stride[i] == p->stride[i]);
      }
    } else {
      CHECK(placed.device == nullptr && placed.image[0] == p->image[0] &&
            placed.image[1] == p->image[1]);
    }
    if (f == 7) {
      if (p->device)
        held = p->device;
      else
        held = p->image[0];
      // Its bytes, copied into a frame of their own.
      auto* bytes = new std::uint8_t[color->getDataSize()];
      std::memcpy(bytes, color->getData(), color->getDataSize());
      copied = ob::FrameFactory::createFrameFromBuffer(
          OB_FRAME_COLOR, color->getFormat(), bytes,
          [](std::uint8_t* b) { delete[] b; }, color->getDataSize());
    }
  }
  CHECK(held.use_count() > 1);  // the frame still holds it
  CHECK(orbbec::device_picture(*copied).has_value());  // as its copy sees
  r.frames.clear();
  CHECK(held.use_count() == 1);  // and it went with the frame
  CHECK(!orbbec::device_picture(*copied).has_value());  // nor its copy
  // A frame that carries no picture reads as none.
  CHECK(!orbbec::device_picture(*ob::FrameFactory::createVideoFrame(
                                    OB_FRAME_COLOR, OB_FORMAT_NV12, 16, 16))
             .has_value());
  return 0;
}

int test_key_frames() {
  const auto units = access_units(kUnlabelled);
  CHECK(units.size() == 8);
  for (std::size_t i = 0; i < units.size(); ++i) {
    // keyint 4: frames 0 and 4.
    CHECK(orbbec::is_key_frame(units[i].data(), units[i].size()) ==
          (i % 4 == 0));
  }
  const std::uint8_t none[] = {0, 0, 1};
  CHECK(!orbbec::is_key_frame(none, sizeof(none)));
  return 0;
}

int test_every_pair_in_order() {
  const Run r =
      run(pairs(access_units(kUnlabelled), {0, 1, 2, 3, 4, 5, 6, 7}), 8);
  CHECK(r.out.size() == 8);
  CHECK(r.lost == 0);
  for (int i = 0; i < 8; ++i) {
    CHECK(slot_of(r.out[static_cast<std::size_t>(i)]) == i);
    if (check_pair(r.out[static_cast<std::size_t>(i)], i, i) != 0) return 1;
  }
  return 0;
}

// Frame 1 lost on the way: frames 2 and 3 are predicted from it, so they go
// too, and decoding picks up at the key frame, 4. The gap in the frame
// numbers is what shows it: FFmpeg 6.1 conceals the missing reference and
// decodes frames 2 and 3 wrongly without an error, where FFmpeg 9 refuses.
int test_gap_waits_for_key_frame() {
  const Run r = run(pairs(access_units(kUnlabelled), {0, 2, 3, 4, 5, 6, 7}), 5);
  CHECK(r.out.size() == 5);
  CHECK(r.lost == 2);
  const int want[] = {0, 4, 5, 6, 7};
  for (std::size_t i = 0; i < r.out.size(); ++i) {
    CHECK(slot_of(r.out[i]) == want[i]);
    if (check_pair(r.out[i], want[i], want[i]) != 0) return 1;
  }
  return 0;
}

// The clip intact, with four seconds between frames 0 and 1: a sync
// secondary's first frame comes as soon as it starts, the rest once the
// primary triggers it. Nothing is missing from the stream, so nothing is lost.
int test_pause_costs_nothing() {
  const Run r = run(pairs(access_units(kUnlabelled), {0, 1, 2, 3, 4, 5, 6, 7},
                          {0, 120, 121, 122, 123, 124, 125, 126}),
                    8);
  CHECK(r.out.size() == 8);
  CHECK(r.lost == 0);
  const int slots[] = {0, 120, 121, 122, 123, 124, 125, 126};
  for (int i = 0; i < 8; ++i) {
    CHECK(slot_of(r.out[static_cast<std::size_t>(i)]) == slots[i]);
    if (check_pair(r.out[static_cast<std::size_t>(i)], i, slots[i]) != 0) {
      return 1;
    }
  }
  return 0;
}

// The gate: key frames start a stream, a skipped or repeated index stops it
// until the next, an unnumbered frame (index 0) is never a gap, and resync()
// waits too. The key frame that ends a wait restarts the stream; one within
// an unbroken stream continues it.
int test_gate() {
  using A = orbbec::ColorStreamGate::Admission;
  orbbec::ColorStreamGate gate;
  CHECK(gate.admit(1, false) == A::Drop);  // before the first key frame
  CHECK(gate.admit(2, true) == A::Restart);
  CHECK(gate.admit(3, false) == A::Decode);
  CHECK(gate.admit(4, true) == A::Decode);
  CHECK(gate.admit(6, false) == A::Drop);  // 5 went missing
  CHECK(gate.admit(7, false) == A::Drop);
  CHECK(gate.admit(8, true) == A::Restart);
  CHECK(gate.admit(9, false) == A::Decode);
  CHECK(gate.admit(9, false) == A::Drop);  // repeated
  CHECK(gate.admit(10, true) == A::Restart);
  CHECK(gate.admit(2, false) == A::Drop);  // backwards: a restarted camera
  CHECK(gate.admit(3, true) == A::Restart);
  gate.resync();
  CHECK(gate.admit(4, false) == A::Drop);
  CHECK(gate.admit(5, true) == A::Restart);
  orbbec::ColorStreamGate unnumbered;
  CHECK(unnumbered.admit(0, true) == A::Restart);
  CHECK(unnumbered.admit(0, false) == A::Decode);
  CHECK(unnumbered.admit(0, false) == A::Decode);
  return 0;
}

// A stream joined mid-GOP starts at its first key frame.
int test_start_waits_for_key_frame() {
  const Run r = run(pairs(access_units(kUnlabelled), {2, 3, 4, 5}), 2);
  CHECK(r.out.size() == 2);
  CHECK(r.lost == 2);
  CHECK(slot_of(r.out[0]) == 4 && slot_of(r.out[1]) == 5);
  return 0;
}

// Frame 2's colour with no depth, as the SDK hands over a colour frame whose
// depth never came: decoded, so frame 3 still has the picture it is predicted
// from, and dropped. Only that pair is lost.
int test_color_without_depth() {
  const Run r =
      run(pairs(access_units(kUnlabelled), {0, 1, -4, 3, 4, 5, 6, 7}), 7);
  CHECK(r.out.size() == 7);
  CHECK(r.lost == 1);
  const int want[] = {0, 1, 3, 4, 5, 6, 7};
  for (std::size_t i = 0; i < r.out.size(); ++i) {
    CHECK(slot_of(r.out[i]) == want[i]);
    if (check_pair(r.out[i], want[i], want[i]) != 0) return 1;
  }
  return 0;
}

int test_pair_without_color() {
  const Run r = run(pairs(access_units(kUnlabelled), {0, -1}), 1);
  CHECK(r.out.size() == 1);
  CHECK(r.lost == 1);
  return 0;
}

// Frame 2 arrives empty: lost, as on the wire, and not sent, which would end
// the stream. Frame 3 goes with it, and decoding picks up at the key frame, 4.
int test_empty_frame() {
  const auto units = access_units(kUnlabelled);
  Pairs in = pairs(units, {0, 1, 2, 3, 4, 5, 6, 7});
  in[2] = pair({}, 2, 2);
  const Run r = run(in, 6);
  CHECK(r.out.size() == 6);
  CHECK(r.lost == 2);
  const int want[] = {0, 1, 4, 5, 6, 7};
  for (std::size_t i = 0; i < r.out.size(); ++i) {
    CHECK(slot_of(r.out[i]) == want[i]);
    if (check_pair(r.out[i], want[i], want[i]) != 0) return 1;
  }
  return 0;
}

// A stream that labels itself is decoded as it says: BT.709 limited range.
int test_labelled_stream() {
  const Run r =
      run(pairs(access_units(kLabelled), {0, 1, 2, 3, 4, 5, 6, 7}), 8);
  CHECK(r.out.size() == 8);
  CHECK(r.lost == 0);
  for (int i = 0; i < 8; ++i) {
    if (check_pair(r.out[static_cast<std::size_t>(i)], i, i,
                   sensor::VideoColorMatrix::Bt709, false) != 0) {
      return 1;
    }
  }
  return 0;
}

// Which frame each access unit shows, in display order: decoded on its own,
// each sent with its index as pts. Empty if it does not decode.
std::vector<int> display_order(const Units& units) {
  sensor::HevcDecoder::Options options;
  options.device = g_device;
  options.allocator = g_allocator;
  auto decoder = sensor::HevcDecoder::create(options);
  if (!decoder) return {};
  std::vector<int> shows(units.size(), -1);
  int shown = 0;
  const auto drain = [&]() {
    for (;;) {
      auto picture = decoder->receive();
      if (!picture.ok() || !picture.value()) return picture.ok();
      const auto k = static_cast<std::size_t>(picture.value()->pts);
      if (k >= shows.size()) return false;
      shows[k] = shown++;
    }
  };
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!decoder->send(units[i].data(), units[i].size(),
                       static_cast<std::int64_t>(i)) ||
        !drain()) {
      return {};
    }
  }
  if (!decoder->send(nullptr, 0, 0) || !drain()) return {};
  if (shown != static_cast<int>(units.size())) return {};
  return shows;
}

// `clip` appended to `units`, and to `shows` the frame each of its access
// units shows (`clip_shows`, from display_order), counted on from the frames
// already there.
void append(Units* units, std::vector<int>* shows, const Units& clip,
            const std::vector<int>& clip_shows) {
  const int base = static_cast<int>(shows->size());
  for (std::size_t i = 0; i < clip.size(); ++i) {
    units->push_back(clip[i]);
    shows->push_back(base + clip_shows[i]);
  }
}

// An FFmpeg before 7.1 hands out one picture per access unit, so a stream
// that changes sequence -- a clip's first key frame after another clip --
// ends with the pictures the first held back still coming out, one an
// access unit: two for these clips, at most this many.
constexpr std::size_t kHeldAtEnd = 3;

// Every pair in `want` (the slots they are dated in, in display order) came
// out, in order, but for the last kHeldAtEnd; each shows the frame
// `frame_of` names, as `labelled_below` or later slots are decoded (BT.709
// limited, or unlabelled).
int check_run(const Run& r, const std::vector<int>& want, int labelled_below,
              int (*frame_of)(int)) {
  CHECK(r.out.size() <= want.size());
  CHECK(r.out.size() + kHeldAtEnd >= want.size());
  for (std::size_t i = 0; i < r.out.size(); ++i) {
    const int slot = want[i];
    CHECK(slot_of(r.out[i]) == slot);
    const int status = slot < labelled_below
                           ? check_pair(r.out[i], frame_of(slot), slot,
                                        sensor::VideoColorMatrix::Bt709, false)
                           : check_pair(r.out[i], frame_of(slot), slot);
    if (status != 0) return 1;
  }
  return 0;
}

// B-frames: the frames are sent out of display order and their pictures come
// out in it, so no pair is lost for having been sent before one shown
// earlier. Each access unit is dated at the frame it shows, as a camera
// dates the frame it captured; the pairs come out in that order. The
// unlabelled clip after them is what lets the last of them out, as the next
// frames of a live stream would.
int test_b_frames() {
  Units b_frames = access_units(kBFrames);
  CHECK(b_frames.size() >= 8);
  b_frames.resize(8);  // the patch frames; the grey ones after are 4:0:0
  const std::vector<int> b_shows = display_order(b_frames);
  CHECK(b_shows.size() == 8);
  CHECK(!std::is_sorted(b_shows.begin(), b_shows.end()));  // out of order
  const Units unlabelled = access_units(kUnlabelled);
  const std::vector<int> u_shows = display_order(unlabelled);
  CHECK(u_shows.size() == 8);

  Units units;
  std::vector<int> shows;
  append(&units, &shows, b_frames, b_shows);
  append(&units, &shows, unlabelled, u_shows);
  std::vector<int> frames;
  std::vector<int> want;
  for (int i = 0; i < 16; ++i) {
    frames.push_back(i);
    want.push_back(i);
  }
  const Run r = run(pairs(units, frames, shows), want.size() - kHeldAtEnd);
  CHECK(r.lost == 0);
  return check_run(r, want, 8, [](int slot) { return slot % 8; });
}

// Open GOPs, access unit 3 lost: frame 1, which nothing refers to. The
// decoder cannot know that, so it restarts at the next key frame, the CRA
// showing frame 6, and skips its two leading pictures (frames 4 and 5):
// predicted from before the CRA, they could refer to a frame that was lost,
// and FFmpeg 6.1 would decode them from a concealed one. Frames 2 and 3,
// held for display at the restart, go too. From there every frame comes out
// -- the next CRA's leading pictures as well, nothing before it missing --
// and then the unlabelled clip, four times over: it lets the last of them
// out and runs on long enough for the skipped pictures' pairs to be counted,
// 32 access units without a picture. Lost: frames 2 and 3, and the leading
// two.
int test_open_gop_restart() {
  const Units open_gop = access_units(kOpenGop);
  CHECK(open_gop.size() == 16);
  const std::vector<int> o_shows = display_order(open_gop);
  CHECK(o_shows.size() == 16);
  CHECK(o_shows[3] == 1 && o_shows[4] == 6 && o_shows[5] == 5 &&
        o_shows[6] == 4);
  const Units unlabelled = access_units(kUnlabelled);
  const std::vector<int> u_shows = display_order(unlabelled);
  CHECK(u_shows.size() == 8);

  Units units;
  std::vector<int> shows;
  append(&units, &shows, open_gop, o_shows);
  for (int copy = 0; copy < 4; ++copy) {
    append(&units, &shows, unlabelled, u_shows);
  }
  std::vector<int> frames;
  std::vector<int> slots;
  for (int i = 0; i < static_cast<int>(units.size()); ++i) {
    if (i == 3) continue;
    frames.push_back(i);
    slots.push_back(shows[static_cast<std::size_t>(i)]);
  }
  std::vector<int> want = {0};
  for (int f = 6; f < static_cast<int>(units.size()); ++f) want.push_back(f);
  const Run r = run(pairs(units, frames, slots), want.size() - kHeldAtEnd);
  CHECK(r.lost == 4);
  return check_run(r, want, 16,
                   [](int slot) { return slot < 16 ? slot : (slot - 16) % 8; });
}

// A decoder with no device to decode onto does not start.
int test_start_needs_device() {
  orbbec::HevcColorDecoder::Options options;
  options.who = "test";
  auto decoder = orbbec::HevcColorDecoder::start(
      options, [](std::shared_ptr<ob::FrameSet>) {});
  CHECK(decoder.status().domain() == vkc::Status::Code::Unsupported);
  return 0;
}

// The B-frame clip's 4:0:0 grey, which no hardware path hands out: the
// decoder stops for good, Unsupported, having handed on patch frames before
// it.
int test_refused_stream() {
  const Units units = access_units(kBFrames);
  CHECK(units.size() == 10);
  std::vector<int> frames;
  for (int i = 0; i < 10; ++i) frames.push_back(i);
  const Run r = run(pairs(units, frames), 10);
  CHECK(r.failure.domain() == vkc::Status::Code::Unsupported);
  CHECK(!r.out.empty());
  return 0;
}

}  // namespace

int main() {
  // The SDK writes a log file into the working directory unless told not to.
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
  if (test_key_frames() != 0) return 1;
  if (test_gate() != 0) return 1;
  if (test_start_needs_device() != 0) return 1;

  auto instance = vkc::Instance::create({});
  if (!instance) {
    return vr_test::no_device("no Vulkan instance",
                              instance.status().message());
  }
  auto physical =
      instance.value().select_physical_device(vr::device_requirements());
  if (!physical) {
    return vr_test::no_device("no compute-capable device",
                              physical.status().message());
  }
  auto device = vkc::Device::create(instance.value(), physical.value(),
                                    vr::device_requirements());
  CHECK(device.ok());
  auto allocator =
      vkc::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());
  g_device = &device.value();
  g_allocator = &allocator.value();
  {
    sensor::HevcDecoder::Options options;
    options.device = g_device;
    options.allocator = g_allocator;
    auto probe = sensor::HevcDecoder::create(options);
    if (!probe) {
      if (probe.status().domain() != vkc::Status::Code::Unsupported) {
        std::fprintf(stderr, "FAIL: %s\n", probe.status().message().c_str());
        return 1;
      }
      return vr_test::no_decoder(probe.status().message());
    }
  }

  if (test_every_pair_in_order() != 0) return 1;
  if (test_hands_on_device_pictures() != 0) return 1;
  if (test_refused_stream() != 0) return 1;
  if (test_gap_waits_for_key_frame() != 0) return 1;
  if (test_pause_costs_nothing() != 0) return 1;
  if (test_start_waits_for_key_frame() != 0) return 1;
  if (test_color_without_depth() != 0) return 1;
  if (test_pair_without_color() != 0) return 1;
  if (test_empty_frame() != 0) return 1;
  if (test_labelled_stream() != 0) return 1;
  if (test_b_frames() != 0) return 1;
  if (test_open_gop_restart() != 0) return 1;
  std::puts("sensor_orbbec_hevc: OK");
  return 0;
}
