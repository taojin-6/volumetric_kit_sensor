// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The Orbbec driver's H.265 colour decoder, with no camera: the committed
// patch clip wrapped in SDK frames, each paired with a depth frame and dated
// at 30 fps. Every pair comes out, in order, with its timestamps and depth,
// and its colour decoded as the Femto Mega codes it (BT.601 full range). A
// frame lost from the stream costs the frames up to the next key frame, a
// pause in the timestamps costs nothing, and decoding starts at the first key
// frame. The frame-index gate is tested on its own, since a test cannot set
// an SDK frame's index.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <libobsensor/ObSensor.hpp>

#include "hevc_color.hpp"
#include "yuv_reference.hpp"

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

// The clip's pattern, as tools/make_hevc_fixtures.sh draws it.
int patch(int column, int row, int frame) {
  return (column + frame + 3 * row) % 8;
}

std::vector<std::vector<std::uint8_t>> access_units() {
  std::ifstream in(VR_HEVC_DATA "/patches_256x144.h265", std::ios::binary);
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>());
  std::vector<std::size_t> starts;
  for (std::size_t i = 0; i + 3 < bytes.size(); ++i) {
    if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1 &&
        ((bytes[i + 3] >> 1) & 0x3f) == 35) {  // access unit delimiter
      starts.push_back(i > 0 && bytes[i - 1] == 0 ? i - 1 : i);
    }
  }
  std::vector<std::vector<std::uint8_t>> units;
  for (std::size_t k = 0; k < starts.size(); ++k) {
    const std::size_t end =
        k + 1 < starts.size() ? starts[k + 1] : bytes.size();
    units.emplace_back(bytes.begin() + static_cast<std::ptrdiff_t>(starts[k]),
                       bytes.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return units;
}

// Access unit `i` as the SDK would hand it over: depth and H.265 colour, both
// dated at frame `i`.
// Clip frame `f`'s pair, dated in `slot`, its colour numbered f + 1 -- as the
// camera numbers its frames -- in the system timestamp, where the decoder's
// frame_index reads it.
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
    color->updateData(unit.data(), static_cast<std::uint32_t>(unit.size()));
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

// Push `frames` (indices into the clip; -1 is a pair with no colour, -f - 2
// frame f's colour with no depth), dated
// by `slots` (by default the frame's own index), wait for `expect` pairs
// out, and stop.
struct Run {
  std::vector<std::shared_ptr<ob::FrameSet>> out;
  std::uint64_t lost = 0;
};
Run run(const std::vector<int>& frames, std::size_t expect,
        std::vector<int> slots = {}) {
  if (slots.empty()) slots = frames;
  const auto units = access_units();
  auto collected = std::make_shared<Collected>();
  orbbec::HevcColorDecoder::Options options;
  options.fps = 30;
  options.who = "test";
  options.frame_index = [](const ob::Frame& frame) {
    return frame.getSystemTimeStampUs();
  };
  auto decoder = orbbec::HevcColorDecoder::start(
      options, [collected](std::shared_ptr<ob::FrameSet> set) {
        std::lock_guard<std::mutex> lock(collected->mutex);
        collected->sets.push_back(std::move(set));
      });
  if (!decoder) {
    std::fprintf(stderr, "%s\n", decoder.status().message().c_str());
    std::exit(1);
  }
  for (std::size_t k = 0; k < frames.size(); ++k) {
    const int i = frames[k];
    if (i == -1) {
      decoder.value()->push(pair(units[0], 0, 0, false));
    } else if (i <= -2) {  // frame -i - 2, colour without its depth
      const int f = -i - 2;
      decoder.value()->push(
          pair(units[static_cast<std::size_t>(f)], f, f, true, false));
    } else {
      decoder.value()->push(
          pair(units[static_cast<std::size_t>(i)], i, slots[k]));
    }
  }
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

// A decoded pair: clip frame `frame`'s pixels, dated in `slot`.
int check_pair(const ob::FrameSet& set, int frame, int slot) {
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
      const auto want = yuv_reference::rgb(
          40 + 24 * p, 64 + 16 * ((3 * p) % 8), 64 + 16 * ((5 * p) % 8),
          sensor::VideoColorMatrix::Bt601, true);
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

int test_key_frames() {
  const auto units = access_units();
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
  const Run r = run({0, 1, 2, 3, 4, 5, 6, 7}, 8);
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
  const Run r = run({0, 2, 3, 4, 5, 6, 7}, 5);
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
  const Run r =
      run({0, 1, 2, 3, 4, 5, 6, 7}, 8, {0, 120, 121, 122, 123, 124, 125, 126});
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
// waits too.
int test_gate() {
  orbbec::ColorStreamGate gate;
  CHECK(!gate.admit(1, false));  // before the first key frame
  CHECK(gate.admit(2, true));
  CHECK(gate.admit(3, false));
  CHECK(!gate.admit(5, false));  // 4 went missing
  CHECK(!gate.admit(6, false));
  CHECK(gate.admit(7, true));
  CHECK(gate.admit(8, false));
  CHECK(!gate.admit(8, false));  // repeated
  CHECK(gate.admit(9, true));
  CHECK(!gate.admit(2, false));  // backwards: a restarted camera
  CHECK(gate.admit(3, true));
  gate.resync();
  CHECK(!gate.admit(4, false));
  CHECK(gate.admit(5, true));
  orbbec::ColorStreamGate unnumbered;
  CHECK(unnumbered.admit(0, true));
  CHECK(unnumbered.admit(0, false));
  CHECK(unnumbered.admit(0, false));
  return 0;
}

// A stream joined mid-GOP starts at its first key frame.
int test_start_waits_for_key_frame() {
  const Run r = run({2, 3, 4, 5}, 2);
  CHECK(r.out.size() == 2);
  CHECK(r.lost == 2);
  CHECK(slot_of(*r.out[0]) == 4 && slot_of(*r.out[1]) == 5);
  return 0;
}

// Frame 2's colour with no depth, as the SDK hands over a colour frame whose
// depth never came: decoded, so frame 3 still has the picture it is predicted
// from, and dropped. Only that pair is lost.
int test_color_without_depth() {
  const Run r = run({0, 1, -4, 3, 4, 5, 6, 7}, 7);
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
  const Run r = run({0, -1}, 1);
  CHECK(r.out.size() == 1);
  CHECK(r.lost == 1);
  return 0;
}

}  // namespace

int main() {
  // The SDK writes a log file into the working directory unless told not to.
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
  if (test_key_frames() != 0) return 1;
  if (test_every_pair_in_order() != 0) return 1;
  if (test_gap_waits_for_key_frame() != 0) return 1;
  if (test_pause_costs_nothing() != 0) return 1;
  if (test_gate() != 0) return 1;
  if (test_start_waits_for_key_frame() != 0) return 1;
  if (test_color_without_depth() != 0) return 1;
  if (test_pair_without_color() != 0) return 1;
  std::puts("sensor_orbbec_hevc: OK");
  return 0;
}
