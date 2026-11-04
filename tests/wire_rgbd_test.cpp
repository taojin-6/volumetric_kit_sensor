// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// An RGB-D sensor's frames as bytes (wire/rgbd_wire.hpp), host only, against
// goldens built here, field by field from the header's tables, never by the
// library:
// - The sensor: a Femto-like SensorInfo (all eight lens terms distinct, a
//   colour camera and its controls, primary, the device's clock) and an
//   ARKit-like one (no lens, no controls, tracked, the host's clock) encode
//   to their bytes and decode back field for field, every double bit for
//   bit (-0.0, a subnormal, all 52 fraction bits). Bytes past the model and
//   reserved bytes are ignored. Every truncation, an unknown flag, role 3
//   and pose 2 are refused, version 2 is Unsupported, and so is a sensor
//   with no depth camera at encode.
// - Depth records at 256 x 192 with a plane and 640 x 576 without, and at
//   1 x 1 and 3 x 5, whose payloads end at an odd size: the encode's runs
//   are the frame's own rows and plane, the prefix is the golden, the
//   decoded frame points at payload + 392 and + 392 + 2wh and holds the
//   payload, and every field is the encoded frame's, bit for bit, reserved
//   bytes ignored. Each refusal is named, a camera whose record's bytes
//   pass a size_t among them; values frame prep would refuse round-trip as
//   written.
// - Colour records: the prefix golden, the camera, codes and exposure read
//   back, reserved bytes ignored, the picture viewed in place, codes of 255
//   and siting 5 accepted, the refusals; the colour params.
// - Through the contract: pattern depth's records, decoded with an info
//   decoded from params, keep the frame contract and carry their index,
//   each posed as ARKit's driver hands a float pose out (made exact): the
//   pose comes back bit for bit, its bottom row exactly [0, 0, 0, 1]; one
//   left as float wrote it breaks the contract, named.
// - A host allocation failure is OutOfMemory.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/testing/test_allocation_failure.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/sensor/testing/rgbd_contract.hpp"
#include "volumetric_kit/sensor/testing/rgbd_pattern.hpp"
#include "volumetric_kit/sensor/wire/rgbd_wire.hpp"

namespace vkc = volumetric_kit::core;
namespace sensor = volumetric_kit::sensor;
namespace camera = volumetric_kit::sensor::camera;
namespace testing = volumetric_kit::sensor::testing;
namespace wire = volumetric_kit::sensor::wire;

namespace {

using Code = vkc::Status::Code;
using Bytes = std::vector<std::uint8_t>;

std::uint64_t bits(double v) {
  std::uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

std::uint32_t bits(float v) {
  std::uint32_t b;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

double from_bits(std::uint64_t b) {
  double v;
  std::memcpy(&v, &b, sizeof v);
  return v;
}

// 1.9999999999999998: every one of the 52 fraction bits set.
const double kAllFraction = from_bits(0x3FFFFFFFFFFFFFFFull);
const double kSubnormal = std::numeric_limits<double>::denorm_min();

// A layout written by hand: each field at the offset the header's tables
// give, least significant byte first.
struct Golden {
  Bytes bytes;

  explicit Golden(std::size_t size) : bytes(size, 0) {}

  void le(std::size_t offset, std::uint64_t value, int count) {
    for (int i = 0; i < count; ++i) {
      bytes.at(offset + i) = static_cast<std::uint8_t>(value >> (8 * i));
    }
  }
  void f64(std::size_t offset, double v) { le(offset, bits(v), 8); }
  void f32(std::size_t offset, float v) { le(offset, bits(v), 4); }

  // The camera table: width, height, fx fy cx cy, k1 k2 p1 p2 k3 k4 k5 k6.
  void camera(std::size_t offset, const camera::CameraModel& c) {
    le(offset + 0, c.size.width, 4);
    le(offset + 4, c.size.height, 4);
    f64(offset + 8, c.intrinsics.fx);
    f64(offset + 16, c.intrinsics.fy);
    f64(offset + 24, c.intrinsics.cx);
    f64(offset + 32, c.intrinsics.cy);
    f64(offset + 40, c.distortion.k1);
    f64(offset + 48, c.distortion.k2);
    f64(offset + 56, c.distortion.p1);
    f64(offset + 64, c.distortion.p2);
    f64(offset + 72, c.distortion.k3);
    f64(offset + 80, c.distortion.k4);
    f64(offset + 88, c.distortion.k5);
    f64(offset + 96, c.distortion.k6);
  }

  // 16 f64, column-major: column c's row r at 8 (4 c + r).
  void transform(std::size_t offset, const camera::Mat4d& m) {
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) f64(offset + 8 * (4 * c + r), m[c][r]);
    }
  }
};

camera::CameraModel pinhole(std::uint32_t width, std::uint32_t height) {
  camera::CameraModel m;
  m.size = {width, height};
  m.intrinsics = {0.75 * width + 0.125, 0.75 * width - 0.0625,
                  width / 2.0 - 0.25, height / 2.0 + 0.375};
  return m;
}

// Not symmetric, so a transposed write moves every value: a turn about z
// and a translation, with the special doubles among its terms.
camera::Mat4d transform(double tx) {
  camera::Mat4d m(1.0);
  m[0][0] = 0.8;
  m[0][1] = 0.6;
  m[1][0] = -0.6;
  m[1][1] = 0.8;
  m[2][2] = kAllFraction / 2.0;
  m[3][0] = tx;
  m[3][1] = -0.0;
  m[3][2] = kSubnormal;
  return m;
}

// A Femto Mega's: its NFOV depth, its 4K colour, each with a lens.
sensor::SensorInfo femto_info() {
  sensor::SensorInfo info;
  info.id = "CL8F25300AB";
  info.model = "Orbbec Femto Mega";
  camera::CameraModel depth = pinhole(640, 576);
  depth.distortion = {-0.0,   kSubnormal, kAllFraction, -1.25e-3,
                      0.0731, 0.512,      -0.0213,      0.00419};
  info.depth = depth;
  camera::CameraModel color = pinhole(3840, 2160);
  color.distortion = {0.0794, -0.105, 2.5e-4, -3.75e-4,
                      0.0431, 0.0,    0.0,    1.0e-5};
  info.color = color;
  info.depth_to_color = transform(0.5);
  info.role = sensor::SyncRole::Primary;
  info.clock = sensor::ClockDomain::Device;
  info.pose = sensor::PoseSource::Fixed;
  info.fps = 25;
  info.color_controls = camera::ColorControls{167, 60};
  return info;
}

// An iPad's: ARKit's depth and colour, no lens, posed by its own tracking.
sensor::SensorInfo arkit_info() {
  sensor::SensorInfo info;
  info.id = "ipad";
  info.model = "iPad Pro";
  info.depth = pinhole(256, 192);
  info.color = pinhole(1920, 1440);
  info.depth_to_color = camera::Mat4d(1.0);
  info.role = sensor::SyncRole::FreeRun;
  info.clock = sensor::ClockDomain::Host;
  info.pose = sensor::PoseSource::Tracked;
  info.fps = 60;
  return info;
}

// The sensor table, by hand.
Bytes sensor_golden(const sensor::SensorInfo& info, std::uint8_t role,
                    std::uint8_t pose) {
  Golden g(356 + info.model.size());
  g.le(0, 1, 2);
  g.le(2, role, 1);
  g.le(3, pose, 1);
  g.le(4, (info.color ? 1 : 0) | (info.color_controls ? 2 : 0), 2);
  g.camera(8, *info.depth);
  if (info.color) g.camera(112, *info.color);
  g.transform(216, info.depth_to_color);
  if (info.color_controls) {
    g.le(344, info.color_controls->exposure, 4);
    g.le(348, info.color_controls->gain, 4);
  }
  g.le(352, info.model.size(), 4);
  std::copy(info.model.begin(), info.model.end(), g.bytes.begin() + 356);
  return g.bytes;
}

std::string text(const Bytes& b) { return std::string(b.begin(), b.end()); }

bool same_camera(const camera::CameraModel& a, const camera::CameraModel& b) {
  const double x[12] = {a.intrinsics.fx, a.intrinsics.fy, a.intrinsics.cx,
                        a.intrinsics.cy, a.distortion.k1, a.distortion.k2,
                        a.distortion.p1, a.distortion.p2, a.distortion.k3,
                        a.distortion.k4, a.distortion.k5, a.distortion.k6};
  const double y[12] = {b.intrinsics.fx, b.intrinsics.fy, b.intrinsics.cx,
                        b.intrinsics.cy, b.distortion.k1, b.distortion.k2,
                        b.distortion.p1, b.distortion.p2, b.distortion.k3,
                        b.distortion.k4, b.distortion.k5, b.distortion.k6};
  for (int i = 0; i < 12; ++i) {
    if (bits(x[i]) != bits(y[i])) return false;
  }
  return a.size.width == b.size.width && a.size.height == b.size.height;
}

bool same_transform(const camera::Mat4d& a, const camera::Mat4d& b) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (bits(a[c][r]) != bits(b[c][r])) return false;
    }
  }
  return true;
}

bool same_optional_camera(const std::optional<camera::CameraModel>& a,
                          const std::optional<camera::CameraModel>& b) {
  return a.has_value() == b.has_value() && (!a || same_camera(*a, *b));
}

bool same_info(const sensor::SensorInfo& a, const sensor::SensorInfo& b) {
  return a.id == b.id && a.model == b.model &&
         same_optional_camera(a.color, b.color) &&
         same_optional_camera(a.depth, b.depth) &&
         same_transform(a.depth_to_color, b.depth_to_color) &&
         a.role == b.role && a.clock == b.clock && a.pose == b.pose &&
         a.fps == b.fps &&
         a.color_controls.has_value() == b.color_controls.has_value() &&
         (!a.color_controls || *a.color_controls == *b.color_controls);
}

// Whether `r` was refused with `code`, its message naming `why`.
template <typename T>
bool refused(const vkc::Result<T>& r, Code code, const std::string& why) {
  if (r.ok()) {
    std::fprintf(stderr, "accepted; expected a refusal naming \"%s\"\n",
                 why.c_str());
    return false;
  }
  const std::string& message = r.status().message();
  if (r.status().domain() != code || message.find(why) == std::string::npos) {
    std::fprintf(stderr, "refused with \"%s\"; expected \"%s\"\n",
                 message.c_str(), why.c_str());
    return false;
  }
  return true;
}

int test_sensor() {
  const sensor::SensorInfo femto = femto_info();
  const sensor::SensorInfo arkit = arkit_info();
  struct Case {
    const sensor::SensorInfo* info;
    std::uint8_t role;
    std::uint8_t pose;
    Bytes head;  // the first 8 bytes, literally
  };
  const Case cases[] = {
      {&femto, 1, 0, {0x01, 0x00, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00}},
      {&arkit, 0, 1, {0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00}},
  };
  for (const Case& c : cases) {
    const sensor::SensorInfo& info = *c.info;
    const Bytes golden = sensor_golden(info, c.role, c.pose);
    CHECK(std::equal(c.head.begin(), c.head.end(), golden.begin()));
    const vkc::Result<std::string> params = wire::encode_sensor(info);
    CHECK(params.ok());
    CHECK(*params == text(golden));

    const vkc::Result<sensor::SensorInfo> read =
        wire::decode_sensor(*params, info.id, info.clock, info.fps);
    CHECK(read.ok());
    CHECK(same_info(*read, info));
    // Bytes past the model are ignored.
    const vkc::Result<sensor::SensorInfo> longer =
        wire::decode_sensor(*params + "later", info.id, info.clock, info.fps);
    CHECK(longer.ok());
    CHECK(same_info(*longer, info));
    // Reserved bytes are ignored: 6 and 7 all ones read as written 0.
    std::string reserved = *params;
    reserved[6] = reserved[7] = static_cast<char>(0xFF);
    const vkc::Result<sensor::SensorInfo> ignored =
        wire::decode_sensor(reserved, info.id, info.clock, info.fps);
    CHECK(ignored.ok());
    CHECK(same_info(*ignored, info));

    // Every truncation, the model's bytes included.
    for (std::size_t n = 0; n < golden.size(); ++n) {
      const std::string cut = params->substr(0, n);
      CHECK(refused(wire::decode_sensor(cut, info.id, info.clock, info.fps),
                    Code::InvalidArgument, "bytes, fewer than"));
    }
    std::string bad = *params;
    bad[0] = 2;
    CHECK(refused(wire::decode_sensor(bad, info.id, info.clock, info.fps),
                  Code::Unsupported, "version 2; this library reads 1"));
    // Version 2 is read before the rest: its params may be shorter.
    CHECK(refused(
        wire::decode_sensor(bad.substr(0, 2), info.id, info.clock, info.fps),
        Code::Unsupported, "version 2"));
    bad = *params;
    bad[4] = static_cast<char>(bad[4] | 0x04);
    CHECK(refused(wire::decode_sensor(bad, info.id, info.clock, info.fps),
                  Code::InvalidArgument, "flags 0x"));
    bad = *params;
    bad[5] = 0x01;
    CHECK(refused(wire::decode_sensor(bad, info.id, info.clock, info.fps),
                  Code::InvalidArgument, "flags 0x1"));
    bad = *params;
    bad[2] = 3;
    CHECK(refused(wire::decode_sensor(bad, info.id, info.clock, info.fps),
                  Code::InvalidArgument, "role 3"));
    bad = *params;
    bad[3] = 2;
    CHECK(refused(wire::decode_sensor(bad, info.id, info.clock, info.fps),
                  Code::InvalidArgument, "pose 2"));
  }

  // The literal bytes of two fields: the depth camera's width (640) at 8,
  // and the extrinsic's x translation (0.5, column 3, row 0) at 216 + 96.
  const vkc::Result<std::string> params = wire::encode_sensor(femto);
  CHECK(params.ok());
  const Bytes width = {0x80, 0x02, 0x00, 0x00};
  CHECK(params->compare(8, 4, text(width)) == 0);
  const Bytes half = {0, 0, 0, 0, 0, 0, 0xE0, 0x3F};
  CHECK(params->compare(312, 8, text(half)) == 0);

  // The clock, id and rate are the caller's, from the track.
  const vkc::Result<sensor::SensorInfo> other =
      wire::decode_sensor(*params, "other", sensor::ClockDomain::Host, 0);
  CHECK(other.ok());
  CHECK(other->id == "other");
  CHECK(other->clock == sensor::ClockDomain::Host);
  CHECK(other->fps == 0);

  // Each role and pose by its wire number.
  sensor::SensorInfo info = arkit;
  info.role = sensor::SyncRole::Secondary;
  info.pose = sensor::PoseSource::Fixed;
  info.color.reset();
  info.color_controls = camera::ColorControls{1, 2};  // without a colour camera
  const vkc::Result<std::string> secondary = wire::encode_sensor(info);
  CHECK(secondary.ok());
  CHECK(*secondary == text(sensor_golden(info, 2, 0)));
  const vkc::Result<sensor::SensorInfo> back =
      wire::decode_sensor(*secondary, info.id, info.clock, info.fps);
  CHECK(back.ok());
  CHECK(same_info(*back, info));

  // The encode's refusals.
  info = femto;
  info.depth.reset();
  CHECK(refused(wire::encode_sensor(info), Code::Unsupported,
                "wire::encode_sensor: a sensor with no depth camera"));
  info = femto;
  info.role = static_cast<sensor::SyncRole>(7);
  CHECK(refused(wire::encode_sensor(info), Code::InvalidArgument, "role 7"));
  info = femto;
  info.pose = static_cast<sensor::PoseSource>(5);
  CHECK(refused(wire::encode_sensor(info), Code::InvalidArgument, "pose 5"));
  std::puts("sensor: the goldens, the round trips and the refusals");
  return 0;
}

// Depth at w x h covering 0, 1, 65535 and a ramp, and a plane covering 0, 1,
// 2 and 255.
struct DepthData {
  std::vector<std::uint16_t> depth;
  std::vector<std::uint8_t> confidence;
};

DepthData depth_data(std::uint32_t width, std::uint32_t height) {
  DepthData d;
  const std::size_t samples = std::size_t{width} * height;
  d.depth.resize(samples);
  d.confidence.resize(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    d.depth[i] = static_cast<std::uint16_t>(i * 257 + 3);
    const std::uint8_t levels[4] = {0, 1, 2, 255};
    d.confidence[i] = levels[i % 4];
  }
  const std::uint16_t ends[3] = {0, 1, 65535};
  for (std::size_t i = 0; i < 3 && i < samples; ++i) d.depth[i] = ends[i];
  return d;
}

sensor::SensorInfo sensor_with_depth(std::uint32_t width,
                                     std::uint32_t height) {
  sensor::SensorInfo info = femto_info();
  camera::CameraModel depth = *info.depth;
  depth.size = {width, height};
  info.depth = depth;
  return info;
}

// A driver's frame of `info`'s sensor over `d`: the depth camera refined
// from the factory's, a pose of its own, the fields the record leaves out
// set too.
sensor::RgbdFrame depth_frame(const sensor::SensorInfo& info,
                              const DepthData& d, bool plane) {
  sensor::RgbdFrame f;
  f.depth = d.depth.data();
  f.depth_camera = *info.depth;
  f.depth_camera.intrinsics.fx += 0.5;
  f.color_to_world = transform(-1.75);
  f.depth_to_color = info.depth_to_color;
  f.metres_per_unit = 0.001f;
  f.min_depth = 0.25f;
  f.max_depth = 5.5f;
  f.depth_noise = {0.0005f, 0.95e-3f, -0.0f};
  if (plane) {
    f.depth_confidence = d.confidence.data();
    f.min_confidence = 2;
  }
  f.timestamp_ns = 1'000'000'123;
  f.sequence = 7;
  f.arrival_ns = 99;
  return f;
}

// The depth table, by hand.
Bytes depth_prefix_golden(const sensor::RgbdFrame& f) {
  Golden g(392);
  g.camera(0, f.depth_camera);
  g.transform(104, f.color_to_world);
  g.transform(232, f.depth_to_color);
  g.f32(360, f.metres_per_unit);
  g.f32(364, f.min_depth);
  g.f32(368, f.max_depth);
  g.f32(372, f.depth_noise.a);
  g.f32(376, f.depth_noise.b);
  g.f32(380, f.depth_noise.c);
  g.le(384, f.depth_confidence != nullptr ? 1 : 0, 1);
  g.le(385, f.min_confidence, 1);
  return g.bytes;
}

// The record's runs, appended as a container does.
std::shared_ptr<Bytes> payload_of(const wire::DepthRecord& r) {
  auto p = std::make_shared<Bytes>(r.prefix.begin(), r.prefix.end());
  p->insert(p->end(), r.depth, r.depth + r.depth_bytes);
  if (r.confidence != nullptr) {
    p->insert(p->end(), r.confidence, r.confidence + r.confidence_bytes);
  }
  return p;
}

// Whether `b`'s depth half is `a`'s, bit for bit, samples included.
bool same_depth_half(const sensor::RgbdFrame& a, const sensor::RgbdFrame& b) {
  const std::size_t samples =
      std::size_t{a.depth_camera.size.width} * a.depth_camera.size.height;
  const bool plane = a.depth_confidence != nullptr;
  return same_camera(a.depth_camera, b.depth_camera) &&
         same_transform(a.color_to_world, b.color_to_world) &&
         same_transform(a.depth_to_color, b.depth_to_color) &&
         bits(a.metres_per_unit) == bits(b.metres_per_unit) &&
         bits(a.min_depth) == bits(b.min_depth) &&
         bits(a.max_depth) == bits(b.max_depth) &&
         bits(a.depth_noise.a) == bits(b.depth_noise.a) &&
         bits(a.depth_noise.b) == bits(b.depth_noise.b) &&
         bits(a.depth_noise.c) == bits(b.depth_noise.c) &&
         a.min_confidence == b.min_confidence &&
         plane == (b.depth_confidence != nullptr) && b.depth != nullptr &&
         std::equal(a.depth, a.depth + samples, b.depth) &&
         (!plane || std::equal(a.depth_confidence, a.depth_confidence + samples,
                               b.depth_confidence));
}

// Whether `f` leaves what the record does not carry at its defaults.
bool left_out(const sensor::RgbdFrame& f) {
  return !f.has_color() && f.depth_device.buffer == nullptr &&
         f.timestamp_ns == 0 && f.sequence == 0 && f.arrival_ns == 0 &&
         f.color_bitstream.codec == sensor::ColorBitstream::Codec::None &&
         f.color_exposure.duration_s == 0.0f && f.color_exposure.iso == 0.0f &&
         f.color_exposure.bias_ev == 0.0f;
}

int round_trip(std::uint32_t width, std::uint32_t height, bool plane) {
  const sensor::SensorInfo info = sensor_with_depth(width, height);
  const DepthData d = depth_data(width, height);
  const sensor::RgbdFrame frame = depth_frame(info, d, plane);
  const std::size_t samples = std::size_t{width} * height;

  const vkc::Result<wire::DepthRecord> record = wire::encode_depth(info, frame);
  CHECK(record.ok());
  // The runs are the frame's own: nothing copied.
  CHECK(record->depth == reinterpret_cast<const std::uint8_t*>(d.depth.data()));
  CHECK(record->depth_bytes == 2 * samples);
  CHECK(record->confidence == (plane ? d.confidence.data() : nullptr));
  CHECK(record->confidence_bytes == (plane ? samples : 0));
  const Bytes golden = depth_prefix_golden(frame);
  CHECK(std::equal(golden.begin(), golden.end(), record->prefix.begin()));

  const std::shared_ptr<Bytes> payload = payload_of(*record);
  CHECK(payload->size() == 392 + (plane ? 3 : 2) * samples);
  const vkc::Result<sensor::RgbdFrame> read =
      wire::decode_depth(info, payload->data(), payload->size(), payload);
  CHECK(read.ok());
  // Read in place: the views point into the payload, which the frame holds.
  CHECK(reinterpret_cast<const std::uint8_t*>(read->depth) ==
        payload->data() + 392);
  CHECK(read->depth_confidence ==
        (plane ? payload->data() + 392 + 2 * samples : nullptr));
  CHECK(read->pixels.get() == static_cast<const void*>(payload.get()));
  CHECK(same_depth_half(frame, *read));
  CHECK(left_out(*read));
  // Reserved bytes are ignored: 386 to 391 all ones read as written 0.
  const auto reserved = std::make_shared<Bytes>(*payload);
  std::fill(reserved->begin() + 386, reserved->begin() + 392, 0xFF);
  const vkc::Result<sensor::RgbdFrame> ignored =
      wire::decode_depth(info, reserved->data(), reserved->size(), reserved);
  CHECK(ignored.ok());
  CHECK(same_depth_half(frame, *ignored));
  CHECK(left_out(*ignored));
  std::printf("depth: %u x %u %s a plane, %zu bytes, read in place\n", width,
              height, plane ? "with" : "without", payload->size());
  return 0;
}

int test_depth_round_trips() {
  CHECK(round_trip(256, 192, true) == 0);
  CHECK(round_trip(640, 576, false) == 0);
  // Payloads of an odd size: 395 and 437 bytes.
  CHECK(round_trip(1, 1, true) == 0);
  CHECK(round_trip(3, 5, true) == 0);
  CHECK(round_trip(3, 5, false) == 0);
  return 0;
}

int test_depth_refusals() {
  const sensor::SensorInfo info = sensor_with_depth(3, 5);
  const DepthData d = depth_data(3, 5);
  const sensor::RgbdFrame frame = depth_frame(info, d, true);
  const vkc::Result<wire::DepthRecord> record = wire::encode_depth(info, frame);
  CHECK(record.ok());
  const std::shared_ptr<Bytes> payload = payload_of(*record);
  const Bytes& good = *payload;
  CHECK(good.size() == 437);
  const auto decode = [&](const sensor::SensorInfo& i, const Bytes& b,
                          std::size_t size) {
    return wire::decode_depth(i, b.data(), size, payload);
  };

  // The size: one byte short or long, the prefix alone, none.
  CHECK(refused(decode(info, good, 436), Code::InvalidArgument,
                "wire::decode_depth: 436 bytes, not the layout's 437 for 3 x "
                "5 with a plane"));
  Bytes longer = good;
  longer.push_back(0);
  CHECK(refused(decode(info, longer, 438), Code::InvalidArgument,
                "438 bytes, not the layout's 437"));
  CHECK(refused(decode(info, good, 392), Code::InvalidArgument,
                "392 bytes, not the layout's 437"));
  CHECK(refused(decode(info, good, 0), Code::InvalidArgument,
                "0 bytes, fewer than the prefix's 392"));
  CHECK(refused(decode(info, good, 391), Code::InvalidArgument,
                "391 bytes, fewer than the prefix's 392"));
  CHECK(refused(wire::decode_depth(info, nullptr, 437, payload),
                Code::InvalidArgument, "a null payload"));
  // An odd address, where no u16 row can start.
  Bytes shifted(good.size() + 1, 0);
  std::copy(good.begin(), good.end(), shifted.begin() + 1);
  CHECK(refused(
      wire::decode_depth(info, shifted.data() + 1, good.size(), payload),
      Code::InvalidArgument, "odd address"));
  // A depth camera of another width, or height, than the sensor's.
  CHECK(refused(decode(sensor_with_depth(4, 5), good, good.size()),
                Code::InvalidArgument,
                "a depth camera of 3 x 5, not the sensor's 4 x 5"));
  CHECK(refused(decode(sensor_with_depth(3, 6), good, good.size()),
                Code::InvalidArgument, "not the sensor's 3 x 6"));
  sensor::SensorInfo no_depth = info;
  no_depth.depth.reset();
  CHECK(refused(decode(no_depth, good, good.size()), Code::InvalidArgument,
                "a sensor without a depth camera"));
  // A flag this version does not name; the plane and its floor.
  Bytes bad = good;
  bad[384] = 0x03;
  CHECK(refused(decode(info, bad, bad.size()), Code::InvalidArgument,
                "flags 0x3"));
  bad = good;
  bad[385] = 0;
  CHECK(refused(decode(info, bad, bad.size()), Code::InvalidArgument,
                "a confidence plane at a floor of 0"));
  bad = good;
  bad[384] = 0;
  bad.resize(392 + 30);  // the rows alone, the floor still 2
  CHECK(refused(decode(info, bad, bad.size()), Code::InvalidArgument,
                "a confidence floor of 2 without a plane"));

  // The encode's.
  sensor::RgbdFrame f = frame;
  f.depth = nullptr;
  f.depth_confidence = nullptr;
  f.min_confidence = 0;
  CHECK(refused(wire::encode_depth(info, f), Code::InvalidArgument,
                "wire::encode_depth: a frame without depth"));
  f.depth_device.buffer = std::make_shared<const vkc::Buffer>();
  CHECK(refused(wire::encode_depth(info, f), Code::Unsupported,
                "depth on the device only"));
  f.depth_confidence = d.confidence.data();
  f.min_confidence = 2;
  CHECK(refused(wire::encode_depth(info, f), Code::InvalidArgument,
                "a confidence plane beside no host depth"));
  f = frame;
  f.depth_device.buffer = std::make_shared<const vkc::Buffer>();
  CHECK(refused(wire::encode_depth(info, f), Code::InvalidArgument,
                "both host and device depth"));
  f = frame;
  f.depth_camera.size = {3, 4};
  CHECK(refused(wire::encode_depth(info, f), Code::InvalidArgument,
                "a depth camera of 3 x 4, not the sensor's 3 x 5"));
  f = frame;
  f.min_confidence = 0;
  CHECK(refused(wire::encode_depth(info, f), Code::InvalidArgument,
                "a confidence plane at a floor of 0"));
  f = frame;
  f.depth_confidence = nullptr;
  CHECK(refused(wire::encode_depth(info, f), Code::InvalidArgument,
                "a confidence floor of 2 without a plane"));
  CHECK(refused(wire::encode_depth(no_depth, frame), Code::InvalidArgument,
                "a sensor without a depth camera"));

  // A depth camera whose record's bytes a size_t cannot count: w h is
  // 2^63 + 2, so an unchecked 392 + 2 w h wraps to 396. A 396-byte record
  // of it is refused, not read past, and so is a frame of it at encode (its
  // rows are never read). The sensor comes through its params.
  sensor::SensorInfo huge;
  camera::CameraModel huge_camera;
  huge_camera.size = {4294836226u, 2147549185u};
  huge.depth = huge_camera;
  const vkc::Result<std::string> huge_params = wire::encode_sensor(huge);
  CHECK(huge_params.ok());
  const vkc::Result<sensor::SensorInfo> huge_info =
      wire::decode_sensor(*huge_params, "huge", sensor::ClockDomain::Host, 0);
  CHECK(huge_info.ok());
  Golden wrapped(396);
  wrapped.camera(0, huge_camera);
  CHECK(refused(decode(*huge_info, wrapped.bytes, 396), Code::InvalidArgument,
                "396 bytes, not the layout's count, past a size_t for "
                "4294836226 x 2147549185 without a plane"));
  const std::uint16_t sample = 0;
  sensor::RgbdFrame huge_frame;
  huge_frame.depth = &sample;
  huge_frame.depth_camera = huge_camera;
  CHECK(refused(wire::encode_depth(*huge_info, huge_frame),
                Code::InvalidArgument,
                "a depth camera of 4294836226 x 2147549185, too large for a "
                "record's bytes to be counted"));

  // Values frame prep refuses are its to refuse: they round-trip as
  // written. A unit of 0, a NaN range, a reflected pose, unknown noise.
  f = frame;
  f.metres_per_unit = 0.0f;
  f.min_depth = std::numeric_limits<float>::quiet_NaN();
  f.max_depth = -std::numeric_limits<float>::infinity();
  f.color_to_world = camera::Mat4d(1.0);
  f.color_to_world[0][0] = -1.0;
  f.depth_noise = {};
  const vkc::Result<wire::DepthRecord> odd = wire::encode_depth(info, f);
  CHECK(odd.ok());
  const std::shared_ptr<Bytes> odd_payload = payload_of(*odd);
  const vkc::Result<sensor::RgbdFrame> odd_read = wire::decode_depth(
      info, odd_payload->data(), odd_payload->size(), odd_payload);
  CHECK(odd_read.ok());
  CHECK(same_depth_half(f, *odd_read));
  std::puts("depth: each refusal named; prep's values passed as written");
  return 0;
}

bool same_codes(const sensor::ColorCodes& a, const sensor::ColorCodes& b) {
  return a.matrix == b.matrix && a.transfer == b.transfer &&
         a.primaries == b.primaries && a.full_range == b.full_range &&
         a.chroma_location == b.chroma_location;
}

int test_color() {
  const sensor::SensorInfo info = femto_info();
  const std::uint8_t planes[1] = {0};
  sensor::RgbdFrame frame;
  frame.color.layout = sensor::YuvLayout::Nv12;
  frame.color.host = planes;
  frame.color.width = info.color->size.width;
  frame.color.height = info.color->size.height;
  frame.color_camera = *info.color;
  frame.color_camera.intrinsics.cx += 0.75;  // refined, the frame's own
  frame.color_exposure = {1.0f / 120.0f, 320.0f, -0.5f};
  sensor::ColorCodes codes;
  codes.matrix = 9;
  codes.transfer = 16;
  codes.primaries = 12;
  codes.full_range = true;
  codes.chroma_location = 2;

  using Prefix = std::array<std::uint8_t, wire::kColorPrefixBytes>;
  const vkc::Result<Prefix> prefix = wire::encode_color(info, frame, codes);
  CHECK(prefix.ok());
  // The colour table, by hand.
  Golden g(128);
  g.camera(0, frame.color_camera);
  g.le(104, 9, 1);
  g.le(105, 16, 1);
  g.le(106, 12, 1);
  g.le(107, 1, 1);
  g.le(108, 2, 1);
  g.f32(112, frame.color_exposure.duration_s);
  g.f32(116, frame.color_exposure.iso);
  g.f32(120, frame.color_exposure.bias_ev);
  CHECK(std::equal(g.bytes.begin(), g.bytes.end(), prefix->begin()));

  // A picture after the prefix: what the track's codec coded, opaque here.
  const Bytes picture = {0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01,
                         0xff, 0xff, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00,
                         0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00,
                         0x5d, 0x95, 0x98, 0x09, 0x00, 0x00, 0x00, 0x01,
                         0x26, 0x01, 0xaf, 0x1d, 0x80};
  Bytes payload(prefix->begin(), prefix->end());
  payload.insert(payload.end(), picture.begin(), picture.end());
  const vkc::Result<wire::ColorRecord> read =
      wire::decode_color(info, payload.data(), payload.size());
  CHECK(read.ok());
  CHECK(same_camera(read->camera, frame.color_camera));
  CHECK(read->codes.matrix == 9 && read->codes.transfer == 16 &&
        read->codes.primaries == 12 && read->codes.full_range &&
        read->codes.chroma_location == 2);
  CHECK(bits(read->exposure.duration_s) ==
        bits(frame.color_exposure.duration_s));
  CHECK(bits(read->exposure.iso) == bits(frame.color_exposure.iso));
  CHECK(bits(read->exposure.bias_ev) == bits(frame.color_exposure.bias_ev));
  CHECK(read->picture == payload.data() + 128);
  CHECK(read->picture_bytes == picture.size());
  CHECK(std::equal(picture.begin(), picture.end(), read->picture));
  // Reserved bytes are ignored: 109 to 111 and 124 to 127 all ones read as
  // written 0.
  Bytes reserved = payload;
  std::fill(reserved.begin() + 109, reserved.begin() + 112, 0xFF);
  std::fill(reserved.begin() + 124, reserved.begin() + 128, 0xFF);
  const vkc::Result<wire::ColorRecord> ignored =
      wire::decode_color(info, reserved.data(), reserved.size());
  CHECK(ignored.ok());
  CHECK(same_camera(ignored->camera, read->camera));
  CHECK(same_codes(ignored->codes, read->codes));
  CHECK(bits(ignored->exposure.duration_s) == bits(read->exposure.duration_s));
  CHECK(bits(ignored->exposure.iso) == bits(read->exposure.iso));
  CHECK(bits(ignored->exposure.bias_ev) == bits(read->exposure.bias_ev));
  CHECK(ignored->picture == reserved.data() + 128);
  CHECK(ignored->picture_bytes == picture.size());

  // The bounds, written and read: codes of 255, siting 5.
  sensor::ColorCodes bounds;
  bounds.matrix = 255;
  bounds.transfer = 255;
  bounds.primaries = 255;
  bounds.chroma_location = 5;
  const vkc::Result<Prefix> bounds_prefix =
      wire::encode_color(info, frame, bounds);
  CHECK(bounds_prefix.ok());
  CHECK((*bounds_prefix)[104] == 255 && (*bounds_prefix)[105] == 255 &&
        (*bounds_prefix)[106] == 255 && (*bounds_prefix)[108] == 5);
  Bytes bounds_payload(bounds_prefix->begin(), bounds_prefix->end());
  bounds_payload.push_back(0x26);
  const vkc::Result<wire::ColorRecord> bounds_read =
      wire::decode_color(info, bounds_payload.data(), bounds_payload.size());
  CHECK(bounds_read.ok());
  CHECK(same_codes(bounds_read->codes, bounds));

  // Limited range and an unknown exposure; a one-byte picture.
  sensor::RgbdFrame plain = frame;
  plain.color_exposure = {};
  sensor::ColorCodes limited = codes;
  limited.full_range = false;
  limited.chroma_location = 0;
  const vkc::Result<Prefix> limited_prefix =
      wire::encode_color(info, plain, limited);
  CHECK(limited_prefix.ok());
  CHECK((*limited_prefix)[107] == 0 && (*limited_prefix)[108] == 0);
  Bytes one(limited_prefix->begin(), limited_prefix->end());
  one.push_back(0x26);
  const vkc::Result<wire::ColorRecord> one_read =
      wire::decode_color(info, one.data(), one.size());
  CHECK(one_read.ok());
  CHECK(!one_read->codes.full_range && one_read->picture_bytes == 1);
  CHECK(one_read->exposure.duration_s == 0.0f &&
        one_read->exposure.iso == 0.0f && one_read->exposure.bias_ev == 0.0f);

  // The decode's refusals.
  CHECK(refused(wire::decode_color(info, payload.data(), 128),
                Code::InvalidArgument,
                "wire::decode_color: 128 bytes, no picture after the "
                "prefix's 128"));
  CHECK(refused(wire::decode_color(info, nullptr, payload.size()),
                Code::InvalidArgument, "a null payload"));
  sensor::SensorInfo other = info;
  camera::CameraModel hd = *other.color;
  hd.size = {1920, 1080};
  other.color = hd;
  CHECK(refused(wire::decode_color(other, payload.data(), payload.size()),
                Code::InvalidArgument,
                "a colour camera of 3840 x 2160, not the sensor's 1920 x "
                "1080"));
  sensor::SensorInfo no_color = info;
  no_color.color.reset();
  CHECK(refused(wire::decode_color(no_color, payload.data(), payload.size()),
                Code::InvalidArgument, "a sensor without a colour camera"));
  Bytes bad = payload;
  bad[108] = 6;
  CHECK(refused(wire::decode_color(info, bad.data(), bad.size()),
                Code::InvalidArgument, "chroma siting 6"));
  bad = payload;
  bad[107] = 0x03;
  CHECK(refused(wire::decode_color(info, bad.data(), bad.size()),
                Code::InvalidArgument, "flags 0x3"));

  // The encode's.
  sensor::ColorCodes bad_codes = codes;
  bad_codes.matrix = 256;
  CHECK(refused(wire::encode_color(info, frame, bad_codes),
                Code::InvalidArgument, "each is 0 to 255"));
  bad_codes = codes;
  bad_codes.transfer = -1;
  CHECK(refused(wire::encode_color(info, frame, bad_codes),
                Code::InvalidArgument, "each is 0 to 255"));
  bad_codes = codes;
  bad_codes.primaries = 256;
  CHECK(refused(wire::encode_color(info, frame, bad_codes),
                Code::InvalidArgument, "each is 0 to 255"));
  bad_codes = codes;
  bad_codes.chroma_location = 6;
  CHECK(refused(wire::encode_color(info, frame, bad_codes),
                Code::InvalidArgument, "chroma siting 6"));
  bad_codes.chroma_location = -1;
  CHECK(refused(wire::encode_color(info, frame, bad_codes),
                Code::InvalidArgument, "chroma siting -1"));
  CHECK(refused(wire::encode_color(no_color, frame, codes),
                Code::InvalidArgument,
                "wire::encode_color: a sensor without a colour camera"));
  sensor::RgbdFrame f = frame;
  f.color = sensor::YuvImage{};
  CHECK(refused(wire::encode_color(info, f, codes), Code::InvalidArgument,
                "a frame without colour"));
  f = frame;
  f.color_camera.size = {3840, 2176};
  CHECK(refused(wire::encode_color(info, f, codes), Code::InvalidArgument,
                "a colour camera of 3840 x 2176, not the sensor's 3840 x "
                "2160"));

  // The picture track's params.
  const std::string params =
      wire::encode_color_params(wire::ColorCodec::HevcMain8);
  CHECK(params == std::string("\x01\x00\x01\x00", 4));
  const vkc::Result<wire::ColorCodec> codec = wire::decode_color_params(params);
  CHECK(codec.ok() && *codec == wire::ColorCodec::HevcMain8);
  const vkc::Result<wire::ColorCodec> later =
      wire::decode_color_params(params + "later");
  CHECK(later.ok() && *later == wire::ColorCodec::HevcMain8);
  CHECK(refused(wire::decode_color_params(std::string("\x01\x00\x02\x00", 4)),
                Code::Unsupported, "codec 2"));
  CHECK(refused(wire::decode_color_params(std::string("\x02\x00\x01\x00", 4)),
                Code::Unsupported, "version 2"));
  CHECK(refused(wire::decode_color_params(std::string("\x02\x00", 2)),
                Code::Unsupported, "version 2"));
  for (std::size_t n = 0; n < params.size(); ++n) {
    CHECK(refused(wire::decode_color_params(params.substr(0, n)),
                  Code::InvalidArgument, "fewer than the params' 4"));
  }
  std::puts("colour: the golden, read in place, the refusals; the params");
  return 0;
}

// Pattern depth's records, decoded with an info decoded from the params,
// keep the frame contract, and each carries its own index.
// A tracked pose as a float device writes it: a turn of `angle` about +Y and a
// translation, rounded to float, its first column 2e-7 long (R^T R 4e-7 off
// the identity), the corner 3 ulp past 1.
camera::Mat4d float_pose(double angle) {
  camera::Mat4d m(1.0);
  m[0][0] = std::cos(angle) * (1.0 + 2.0e-7);
  m[0][2] = -std::sin(angle) * (1.0 + 2.0e-7);
  m[2][0] = std::sin(angle);
  m[2][2] = std::cos(angle);
  m[3] = glm::dvec4(0.5 * angle, 1.25, -2.0 + angle, 1.0);
  camera::Mat4d rounded = camera::Mat4d(glm::mat4(m));
  rounded[3][3] = double(
      std::nextafter(std::nextafter(std::nextafter(1.0f, 2.0f), 2.0f), 2.0f));
  return rounded;
}

int test_contract() {
  const sensor::SensorInfo arkit = arkit_info();
  const vkc::Result<std::string> params = wire::encode_sensor(arkit);
  CHECK(params.ok());
  const vkc::Result<sensor::SensorInfo> read =
      wire::decode_sensor(*params, arkit.id, sensor::ClockDomain::Host, 10);
  CHECK(read.ok());
  // A reader of the depth track alone names no colour camera: a frame of
  // depth alone is not whole for a sensor with one.
  sensor::SensorInfo info = *read;
  info.color.reset();

  constexpr std::uint32_t kFrames = 6;
  std::vector<sensor::RgbdFrame> frames;
  for (std::uint32_t i = 0; i < kFrames; ++i) {
    const testing::PatternDepth p = testing::pattern_depth(256, 192, 1000 + i);
    sensor::RgbdFrame f;
    f.depth = p.depth.data();
    f.depth_confidence = p.confidence.data();
    f.min_confidence = 2;
    f.depth_camera = *arkit.depth;
    f.min_depth = 0.25f;
    f.max_depth = 5.0f;
    // As ARKit's driver hands its float pose out.
    const camera::Mat4d rounded = float_pose(0.1 * i);
    CHECK(!camera::check_rigid(rounded).ok());
    const vkc::Result<camera::Mat4d> exact = camera::exact_rigid(rounded);
    CHECK(exact.ok());
    f.color_to_world = *exact;
    const vkc::Result<wire::DepthRecord> record = wire::encode_depth(arkit, f);
    CHECK(record.ok());
    const std::shared_ptr<Bytes> payload = payload_of(*record);
    vkc::Result<sensor::RgbdFrame> frame =
        wire::decode_depth(info, payload->data(), payload->size(), payload);
    CHECK(frame.ok());
    // The pose comes back bit for bit, its row exact.
    const camera::Mat4d& pose = frame->color_to_world;
    CHECK(same_transform(pose, *exact));
    CHECK(pose[0][3] == 0.0 && pose[1][3] == 0.0 && pose[2][3] == 0.0 &&
          pose[3][3] == 1.0);
    CHECK(camera::check_rigid(pose).ok());
    frame->timestamp_ns = 1'000'000'000 + 100'000'000ull * i;  // the pts
    frame->sequence = i;                                       // the seq
    frames.push_back(std::move(*frame));
  }
  // The pattern's own buffers are gone: each frame reads its payload.
  for (std::uint32_t i = 0; i < kFrames; ++i) {
    const std::optional<std::uint32_t> index =
        testing::depth_index(frames[i].depth, 256, 192);
    CHECK(index && *index == 1000 + i);
  }
  sensor::SensorStats stats;
  stats.received = kFrames;
  stats.delivered = kFrames;
  CHECK(testing::rgbd_breach(info, frames, stats).empty());
  CHECK(testing::rgbd_breach(*read, frames, stats) ==
        "frame 0: no colour from a sensor with a colour camera");
  // A pose as float wrote it, not made exact: the contract names it.
  frames[3].color_to_world = float_pose(0.3);
  CHECK(testing::rgbd_breach(info, frames, stats) ==
        "frame 3: color_to_world: transform's bottom row is not [0, 0, 0, 1]");
  std::puts(
      "contract: pattern depth's records keep it, their index and their "
      "exact poses");
  return 0;
}

// A host allocation failure inside a call is OutOfMemory, not an exception.
int test_allocation_failure() {
  auto& failure = vkc::test::allocation_failure;
  sensor::SensorInfo info = femto_info();
  info.model = std::string(200, 'm');  // past every small-string buffer

  failure.measure = true;
  const vkc::Result<std::string> params = wire::encode_sensor(info);
  failure.measure = false;
  CHECK(params.ok());
  failure.site = failure.measured_site;
  failure.armed = true;
  const vkc::Result<std::string> encoded = wire::encode_sensor(info);
  CHECK(failure.injected);
  failure = {};
  CHECK(refused(encoded, Code::OutOfMemory, "wire::encode_sensor"));

  failure.measure = true;
  const vkc::Result<sensor::SensorInfo> read =
      wire::decode_sensor(*params, "id", info.clock, info.fps);
  failure.measure = false;
  CHECK(read.ok());
  failure.site = failure.measured_site;
  failure.armed = true;
  const vkc::Result<sensor::SensorInfo> decoded =
      wire::decode_sensor(*params, "id", info.clock, info.fps);
  CHECK(failure.injected);
  failure = {};
  CHECK(refused(decoded, Code::OutOfMemory, "wire::decode_sensor"));
  std::puts("allocation failure: OutOfMemory");
  return 0;
}

}  // namespace

int main() {
  if (test_sensor() != 0 || test_depth_round_trips() != 0 ||
      test_depth_refusals() != 0 || test_color() != 0 || test_contract() != 0 ||
      test_allocation_failure() != 0) {
    return 1;
  }
  std::puts("sensor_wire_rgbd: OK");
  return 0;
}
