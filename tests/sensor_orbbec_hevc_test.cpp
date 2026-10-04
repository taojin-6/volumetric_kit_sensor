// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's H.265 colour decoder, with no camera: committed clips
// wrapped in SDK frames, each paired with a depth frame and dated at 30 fps.
// Every pair comes out, in display order, with its timestamps and depth, and
// an unlabelled stream's colour decoded as the Femto Mega codes it (BT.601
// full range), a labelled one's as it says. A frame lost from the stream, or
// empty, costs the frames up to the next key frame, a pause in the
// timestamps costs nothing, and decoding starts at the first key frame. The
// frame-index gate is tested on its own, since a test cannot set an SDK
// frame's index. Given a device, a picture the hardware leaves on it comes
// out carried in its frame, and goes with it.

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
#include "picture_frames.hpp"
#include "volumetric_kit/recon/core/allocator.hpp"
#include "volumetric_kit/recon/core/device.hpp"
#include "volumetric_kit/recon/core/instance.hpp"
#include "yuv_reference.hpp"

namespace vr = volumetric_kit::recon;
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
constexpr const char* kBFrames = VR_HEVC_DATA "/fallback_256x144.h265";
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

struct Collected {
  std::mutex mutex;
  std::vector<std::shared_ptr<ob::FrameSet>> sets;
  std::size_t size() {
    std::lock_guard<std::mutex> lock(mutex);
    return sets.size();
  }
};

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

// Push `in`, wait for `expect` pairs out, and stop; `yuv` has the decoder
// hand on I420 frames, for the GPU pass, or pictures on `device`.
struct Run {
  std::vector<std::shared_ptr<ob::FrameSet>> out;
  std::uint64_t lost = 0;
};
Run run(const Pairs& in, std::size_t expect, bool yuv = false,
        const vr::Device* device = nullptr,
        vr::Allocator* allocator = nullptr) {
  auto collected = std::make_shared<Collected>();
  orbbec::HevcColorDecoder::Options options;
  options.fps = 30;
  options.who = "test";
  options.frame_index = [](const ob::Frame& frame) {
    return frame.getSystemTimeStampUs();
  };
  options.yuv = yuv;
  options.device = device;
  options.allocator = allocator;
  auto decoder = orbbec::HevcColorDecoder::start(
      options, [collected](std::shared_ptr<ob::FrameSet> set) {
        std::lock_guard<std::mutex> lock(collected->mutex);
        collected->sets.push_back(std::move(set));
      });
  if (!decoder) {
    std::fprintf(stderr, "%s\n", decoder.status().message().c_str());
    std::exit(1);
  }
  for (const auto& p : in) decoder.value()->push(p);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (collected->size() < expect &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  // Let anything that should not come out have its chance to.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  decoder.value()->stop();
  Run r;
  r.lost = decoder.value()->lost();
  std::lock_guard<std::mutex> lock(collected->mutex);
  r.out = collected->sets;
  return r;
}

// The slot a decoded pair was dated in, from its colour's timestamp.
int slot_of(const ob::FrameSet& set) {
  const auto color = set.getColorFrame();
  return static_cast<int>((color->getTimeStampUs() - kStartUs) / kPeriodUs);
}

// A decoded pair: clip frame `frame`'s pixels, dated in `slot`, decoded as
// the Femto Mega codes its unlabelled stream unless `matrix` says otherwise.
int check_pair(
    const ob::FrameSet& set, int frame, int slot,
    sensor::VideoColorMatrix matrix = sensor::VideoColorMatrix::Bt601,
    bool full_range = true) {
  const auto depth = set.getDepthFrame();
  const auto color = set.getColorFrame();
  CHECK(depth != nullptr && color != nullptr);
  const std::uint64_t t =
      kStartUs + static_cast<std::uint64_t>(slot) * kPeriodUs;
  CHECK(depth->getTimeStampUs() == t);
  CHECK(color->getTimeStampUs() == t);
  CHECK(color->getFormat() == OB_FORMAT_RGB);
  const auto video = color->as<ob::VideoFrame>();
  CHECK(video->getWidth() == static_cast<std::uint32_t>(kWidth));
  CHECK(video->getHeight() == static_cast<std::uint32_t>(kHeight));
  CHECK(color->getDataSize() >=
        static_cast<std::uint32_t>(kWidth * kHeight * 3));
  const std::uint8_t* rgb = color->getData();
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 8; ++col) {
      const int p = patch(col, row, frame);
      const auto want =
          yuv_reference::rgb(40 + 24 * p, 64 + 16 * ((3 * p) % 8),
                             64 + 16 * ((5 * p) % 8), matrix, full_range);
      const int x = 32 * col + 16;
      const int y = 72 * row + 36;
      for (int k = 0; k < 3; ++k) {
        const int got = rgb[3 * (y * kWidth + x) + k];
        if (std::abs(got - want[k]) > 3) {
          std::fprintf(stderr, "frame %d patch %d,%d channel %d: %d vs %d\n",
                       frame, col, row, k, got, want[k]);
          CHECK(false);
        }
      }
    }
  }
  return 0;
}

// For the GPU pass: the decoded planes as an I420 frame, Y then Cb and Cr at
// half size, rows packed, each patch's value as the clip was made, carrying
// the matrix and range they are coded in -- the Femto Mega's BT.601 full range
// for the unlabelled stream, and what a labelled one says -- and a canonical
// encoding, since neither declares a transfer or primaries it cannot name.
int test_hands_on_i420() {
  for (const bool labelled : {false, true}) {
    const Run r = run(pairs(access_units(labelled ? kLabelled : kUnlabelled),
                            {0, 1, 2, 3, 4, 5, 6, 7}),
                      8, true);
    CHECK(r.out.size() == 8);
    for (const auto& set : r.out) {
      const auto color = set->getColorFrame();
      CHECK(color != nullptr);
      const std::optional<orbbec::PlanesColor> described =
          orbbec::planes_color(*color);
      CHECK(described.has_value());
      CHECK(described->matrix == (labelled ? sensor::VideoColorMatrix::Bt709
                                           : sensor::VideoColorMatrix::Bt601));
      CHECK(described->full_range == !labelled);
      CHECK(described->has_encoding && is_canonical(described->encoding));
    }
  }
  // A frame the decoder did not make carries none.
  CHECK(!orbbec::planes_color(*ob::FrameFactory::createVideoFrame(
                                  OB_FRAME_COLOR, OB_FORMAT_I420, 16, 16))
             .has_value());

  const Run r =
      run(pairs(access_units(kUnlabelled), {0, 1, 2, 3, 4, 5, 6, 7}), 8, true);
  CHECK(r.out.size() == 8);
  CHECK(r.lost == 0);
  for (int f = 0; f < 8; ++f) {
    const auto& set = *r.out[static_cast<std::size_t>(f)];
    CHECK(set.getDepthFrame() != nullptr);
    const auto color = set.getColorFrame();
    CHECK(color != nullptr && color->getFormat() == OB_FORMAT_I420);
    const auto video = color->as<ob::VideoFrame>();
    CHECK(video->getWidth() == static_cast<std::uint32_t>(kWidth));
    CHECK(video->getHeight() == static_cast<std::uint32_t>(kHeight));
    const int cw = kWidth / 2;
    const int ch = kHeight / 2;
    CHECK(color->getDataSize() ==
          static_cast<std::uint32_t>(kWidth * kHeight + 2 * cw * ch));
    const std::uint8_t* y = color->getData();
    const std::uint8_t* cb = y + kWidth * kHeight;
    const std::uint8_t* cr = cb + cw * ch;
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int p = patch(col, row, f);
        const int x = 32 * col + 16;
        const int yy = 72 * row + 36;
        CHECK(std::abs(y[yy * kWidth + x] - (40 + 24 * p)) <= 2);
        const int c = (yy / 2) * cw + x / 2;
        CHECK(std::abs(cb[c] - (64 + 16 * ((3 * p) % 8))) <= 2);
        CHECK(std::abs(cr[c] - (64 + 16 * ((5 * p) % 8))) <= 2);
      }
    }
  }
  return 0;
}

// Given a device the hardware decodes onto, each picture comes out carried in
// its frame, each patch's value as the clip was made, described as the Femto
// Mega codes the unlabelled stream, and placed in a raw frame's colour as the
// driver places it; and it lives as long as its frame, which a copy of the
// frame does not extend. Where no hardware leaves pictures on this device,
// they come as I420.
int test_hands_on_device_pictures() {
  auto instance = vr::Instance::create({});
  if (!instance) return 0;
  auto gpu = instance.value().select_physical_device(vr::device_requirements());
  if (!gpu) return 0;
  auto device = vr::Device::create(instance.value(), gpu.value(),
                                   vr::device_requirements());
  CHECK(device.ok());
  auto allocator =
      vr::Allocator::create(instance.value().handle(), device.value());
  CHECK(allocator.ok());

  Run r = run(pairs(access_units(kUnlabelled), {0, 1, 2, 3, 4, 5, 6, 7}), 8,
              true, &device.value(), &allocator.value());
  CHECK(r.out.size() == 8 && r.lost == 0);
  const bool on_device =
      orbbec::device_picture(*r.out.front()->getColorFrame()).has_value();
  // The legs whose hardware leaves its pictures on the device.
  const char* required = std::getenv("VR_TEST_HEVC_BACKEND");
  const std::string backend = required != nullptr ? required : "";
  if (backend == "videotoolbox" || (VR_TEST_WITH_CUDA && backend == "cuda")) {
    CHECK(on_device);
  }
  std::shared_ptr<const void> held;
  std::shared_ptr<ob::Frame> copied;
  for (int f = 0; f < 8; ++f) {
    const auto color = r.out[static_cast<std::size_t>(f)]->getColorFrame();
    CHECK(color != nullptr);
    const std::optional<sensor::DecodedPicture> p =
        orbbec::device_picture(*color);
    CHECK(p.has_value() == on_device);
    if (!p) {
      CHECK(color->getFormat() == OB_FORMAT_I420);
      continue;
    }
    CHECK(color->as<ob::VideoFrame>()->getWidth() == 0);  // no pixels
    CHECK(p->width == static_cast<std::uint32_t>(kWidth) &&
          p->height == static_cast<std::uint32_t>(kHeight));
    CHECK(p->layout == sensor::VideoPixelLayout::Nv12);
    CHECK((p->device != nullptr) != (p->image[0] != nullptr));
    CHECK(p->matrix == sensor::VideoColorMatrix::Bt601 && p->full_range);
    sensor::YuvImage placed;
    orbbec::place_device_color(*p, &placed);
    CHECK(placed.layout == sensor::YuvLayout::Nv12);
    CHECK(placed.plane[0] == nullptr);
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
    std::vector<std::uint8_t> got[3];
    vr_test::read_device_picture(*p, device.value(), allocator.value(), got);
    CHECK(got[0].size() == static_cast<std::size_t>(kWidth * kHeight));
    const int cw = kWidth / 2;
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 8; ++col) {
        const int k = patch(col, row, f);
        const int x = 32 * col + 16;
        const int yy = 72 * row + 36;
        CHECK(std::abs(got[0][yy * kWidth + x] - (40 + 24 * k)) <= 2);
        const int c = (yy / 2) * cw + x / 2;
        CHECK(std::abs(got[1][c] - (64 + 16 * ((3 * k) % 8))) <= 2);
        CHECK(std::abs(got[2][c] - (64 + 16 * ((5 * k) % 8))) <= 2);
      }
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
  if (held != nullptr) {
    CHECK(held.use_count() > 1);  // the frame still holds it
    CHECK(orbbec::device_picture(*copied).has_value());  // as its copy sees
    r.out.clear();
    CHECK(held.use_count() == 1);  // and it went with the frame
    CHECK(!orbbec::device_picture(*copied).has_value());  // nor its copy
  }
  // A frame that carries no picture reads as none.
  CHECK(!orbbec::device_picture(*ob::FrameFactory::createVideoFrame(
                                    OB_FRAME_COLOR, OB_FORMAT_NV12, 16, 16))
             .has_value());
  std::printf("  device pictures: %s\n",
              on_device ? "carried in their frames"
                        : "not offered here; I420 frames instead");
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
    CHECK(slot_of(*r.out[static_cast<std::size_t>(i)]) == i);
    if (check_pair(*r.out[static_cast<std::size_t>(i)], i, i) != 0) return 1;
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
    CHECK(slot_of(*r.out[i]) == want[i]);
    if (check_pair(*r.out[i], want[i], want[i]) != 0) return 1;
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
    CHECK(slot_of(*r.out[static_cast<std::size_t>(i)]) == slots[i]);
    if (check_pair(*r.out[static_cast<std::size_t>(i)], i, slots[i]) != 0) {
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
  CHECK(slot_of(*r.out[0]) == 4 && slot_of(*r.out[1]) == 5);
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
    CHECK(slot_of(*r.out[i]) == want[i]);
    if (check_pair(*r.out[i], want[i], want[i]) != 0) return 1;
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
    CHECK(slot_of(*r.out[i]) == want[i]);
    if (check_pair(*r.out[i], want[i], want[i]) != 0) return 1;
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
    if (check_pair(*r.out[static_cast<std::size_t>(i)], i, i,
                   sensor::VideoColorMatrix::Bt709, false) != 0) {
      return 1;
    }
  }
  return 0;
}

// Which frame each access unit shows, in display order: decoded on its own,
// each sent with its index as pts. Empty if it does not decode.
std::vector<int> display_order(const Units& units) {
  auto decoder = sensor::HevcDecoder::create({});
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
    CHECK(slot_of(*r.out[i]) == slot);
    const int status = slot < labelled_below
                           ? check_pair(*r.out[i], frame_of(slot), slot,
                                        sensor::VideoColorMatrix::Bt709, false)
                           : check_pair(*r.out[i], frame_of(slot), slot);
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

}  // namespace

int main() {
  // The SDK writes a log file into the working directory unless told not to.
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
  if (test_key_frames() != 0) return 1;
  if (test_every_pair_in_order() != 0) return 1;
  if (test_hands_on_i420() != 0) return 1;
  if (test_hands_on_device_pictures() != 0) return 1;
  if (test_gap_waits_for_key_frame() != 0) return 1;
  if (test_pause_costs_nothing() != 0) return 1;
  if (test_gate() != 0) return 1;
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
