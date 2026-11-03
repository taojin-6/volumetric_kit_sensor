// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/wire/rgbd_wire.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

#include "volumetric_kit/core/base/exception_status.hpp"

// Every field is copied as the host holds it, so the wire's little-endian
// form, f32 and f64 bit for bit, needs a little-endian host with IEEE 754
// binary32 and binary64.
#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__)
#error "sensor_wire needs the compiler's byte-order macros"
#endif
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "sensor_wire builds only for a little-endian host");
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "the wire's f32 is IEEE 754 binary32");
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
              "the wire's f64 is IEEE 754 binary64");

// The public calls return a host allocation failure (the params, the model,
// a refusal's message) as OutOfMemory through core::exception_status: no
// exception leaves them.

namespace volumetric_kit::sensor::wire {
namespace {

// A camera's fields (kCameraBytes).
constexpr std::size_t kCameraWidth = 0;
constexpr std::size_t kCameraHeight = 4;
constexpr std::size_t kCameraIntrinsics = 8;
constexpr std::size_t kCameraDistortion = 40;
static_assert(kCameraDistortion + 8 * 8 == kCameraBytes);

constexpr std::size_t kTransformBytes = 16 * 8;
// Each track's params begin with the u16 version.
constexpr std::size_t kVersionBytes = 2;

// The sensor's params (encode_sensor).
constexpr std::size_t kSensorVersion = 0;
constexpr std::size_t kSensorRole = 2;
constexpr std::size_t kSensorPose = 3;
constexpr std::size_t kSensorFlags = 4;
constexpr std::size_t kSensorDepthCamera = 8;
constexpr std::size_t kSensorColorCamera = kSensorDepthCamera + kCameraBytes;
constexpr std::size_t kSensorDepthToColor = kSensorColorCamera + kCameraBytes;
constexpr std::size_t kSensorExposure = kSensorDepthToColor + kTransformBytes;
constexpr std::size_t kSensorGain = kSensorExposure + 4;
constexpr std::size_t kSensorModelBytes = kSensorGain + 4;
static_assert(kSensorColorCamera == 112 && kSensorDepthToColor == 216 &&
              kSensorExposure == 344 && kSensorModelBytes == 352);
static_assert(kSensorModelBytes + 4 == kSensorParamsBytes);
constexpr std::uint16_t kSensorHasColor = 1u << 0;
constexpr std::uint16_t kSensorHasControls = 1u << 1;
constexpr std::uint16_t kSensorFlagsKnown =
    kSensorHasColor | kSensorHasControls;

// A depth record's prefix (encode_depth).
constexpr std::size_t kDepthCamera = 0;
constexpr std::size_t kDepthColorToWorld = kDepthCamera + kCameraBytes;
constexpr std::size_t kDepthDepthToColor = kDepthColorToWorld + kTransformBytes;
constexpr std::size_t kDepthUnits = kDepthDepthToColor + kTransformBytes;
constexpr std::size_t kDepthMin = kDepthUnits + 4;
constexpr std::size_t kDepthMax = kDepthMin + 4;
constexpr std::size_t kDepthNoise = kDepthMax + 4;
constexpr std::size_t kDepthFlags = kDepthNoise + 12;
constexpr std::size_t kDepthFloor = kDepthFlags + 1;
static_assert(kDepthColorToWorld == 104 && kDepthDepthToColor == 232 &&
              kDepthUnits == 360 && kDepthNoise == 372 && kDepthFlags == 384);
static_assert(kDepthFloor + 1 + 6 == kDepthPrefixBytes);
static_assert(kDepthPrefixBytes % 8 == 0);
constexpr std::uint8_t kDepthHasPlane = 1u << 0;

// The picture track's params (encode_color_params).
constexpr std::size_t kColorParamsVersion = 0;
constexpr std::size_t kColorParamsCodec = 2;

// A colour record's prefix (encode_color).
constexpr std::size_t kColorCamera = 0;
constexpr std::size_t kColorMatrix = kColorCamera + kCameraBytes;
constexpr std::size_t kColorTransfer = kColorMatrix + 1;
constexpr std::size_t kColorPrimaries = kColorTransfer + 1;
constexpr std::size_t kColorFlags = kColorPrimaries + 1;
constexpr std::size_t kColorSiting = kColorFlags + 1;
constexpr std::size_t kColorExposure = kColorSiting + 1 + 3;
static_assert(kColorMatrix == 104 && kColorSiting == 108 &&
              kColorExposure == 112);
static_assert(kColorExposure + 12 + 4 == kColorPrefixBytes);
constexpr std::uint8_t kColorFullRange = 1u << 0;
constexpr int kMaxSiting = 5;  // ChromaSampleLocType's last

template <typename T>
void put(std::uint8_t* at, T value) noexcept {
  std::memcpy(at, &value, sizeof value);
}

template <typename T>
T get(const std::uint8_t* at) noexcept {
  T value;
  std::memcpy(&value, at, sizeof value);
  return value;
}

void put_camera(std::uint8_t* at, const camera::CameraModel& c) noexcept {
  put<std::uint32_t>(at + kCameraWidth, c.size.width);
  put<std::uint32_t>(at + kCameraHeight, c.size.height);
  const double intrinsics[4] = {c.intrinsics.fx, c.intrinsics.fy,
                                c.intrinsics.cx, c.intrinsics.cy};
  for (int i = 0; i < 4; ++i)
    put(at + kCameraIntrinsics + 8 * i, intrinsics[i]);
  const camera::RationalDistortion& d = c.distortion;
  const double lens[8] = {d.k1, d.k2, d.p1, d.p2, d.k3, d.k4, d.k5, d.k6};
  for (int i = 0; i < 8; ++i) put(at + kCameraDistortion + 8 * i, lens[i]);
}

camera::CameraModel get_camera(const std::uint8_t* at) noexcept {
  camera::CameraModel c;
  c.size.width = get<std::uint32_t>(at + kCameraWidth);
  c.size.height = get<std::uint32_t>(at + kCameraHeight);
  const auto f64 = [&](std::size_t offset, int i) {
    return get<double>(at + offset + 8 * i);
  };
  c.intrinsics = {f64(kCameraIntrinsics, 0), f64(kCameraIntrinsics, 1),
                  f64(kCameraIntrinsics, 2), f64(kCameraIntrinsics, 3)};
  c.distortion = {f64(kCameraDistortion, 0), f64(kCameraDistortion, 1),
                  f64(kCameraDistortion, 2), f64(kCameraDistortion, 3),
                  f64(kCameraDistortion, 4), f64(kCameraDistortion, 5),
                  f64(kCameraDistortion, 6), f64(kCameraDistortion, 7)};
  return c;
}

// Column-major: GLM's own order, m[column][row].
void put_transform(std::uint8_t* at, const camera::Mat4d& m) noexcept {
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      put(at + 8 * (4 * column + row), m[column][row]);
    }
  }
}

camera::Mat4d get_transform(const std::uint8_t* at) noexcept {
  camera::Mat4d m(1.0);
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      m[column][row] = get<double>(at + 8 * (4 * column + row));
    }
  }
  return m;
}

// The wire's numbers for SyncRole and PoseSource, stated here so that a C++
// enumeration's order is not the wire's.
std::optional<std::uint8_t> role_code(SyncRole role) noexcept {
  switch (role) {
    case SyncRole::FreeRun:
      return 0;
    case SyncRole::Primary:
      return 1;
    case SyncRole::Secondary:
      return 2;
  }
  return std::nullopt;
}

std::optional<SyncRole> role_of(std::uint8_t code) noexcept {
  switch (code) {
    case 0:
      return SyncRole::FreeRun;
    case 1:
      return SyncRole::Primary;
    case 2:
      return SyncRole::Secondary;
    default:
      return std::nullopt;
  }
}

std::optional<std::uint8_t> pose_code(PoseSource pose) noexcept {
  switch (pose) {
    case PoseSource::Fixed:
      return 0;
    case PoseSource::Tracked:
      return 1;
  }
  return std::nullopt;
}

std::optional<PoseSource> pose_of(std::uint8_t code) noexcept {
  switch (code) {
    case 0:
      return PoseSource::Fixed;
    case 1:
      return PoseSource::Tracked;
    default:
      return std::nullopt;
  }
}

core::Status refuse(const char* api, const std::string& why) {
  return core::Status::invalid_argument(std::string("wire::") + api + ": " +
                                        why);
}

core::Status unsupported(const char* api, const std::string& why) {
  return core::Status::unsupported(std::string("wire::") + api + ": " + why);
}

std::string size_text(const camera::ImageSize& size) {
  return std::to_string(size.width) + " x " + std::to_string(size.height);
}

bool same_size(const camera::ImageSize& a, const camera::ImageSize& b) {
  return a.width == b.width && a.height == b.height;
}

std::string hex_text(unsigned value) {
  char text[16];
  std::snprintf(text, sizeof text, "0x%x", value);
  return text;
}

// A depth record's runs for a depth camera of `size`: 2 w h bytes of rows,
// w h of plane with one, and the payload's bytes; empty where the payload's
// bytes cannot be counted in a size_t.
struct DepthLayout {
  std::size_t rows = 0;
  std::size_t plane = 0;
  std::size_t payload = 0;
};

std::optional<DepthLayout> depth_layout(const camera::ImageSize& size,
                                        bool plane) noexcept {
  const std::uint64_t samples = std::uint64_t{size.width} * size.height;
  const std::uint64_t per_sample = plane ? 3 : 2;
  constexpr std::uint64_t kMax = std::numeric_limits<std::size_t>::max();
  if (samples > (kMax - kDepthPrefixBytes) / per_sample) return std::nullopt;
  DepthLayout layout;
  layout.rows = static_cast<std::size_t>(2 * samples);
  layout.plane = plane ? static_cast<std::size_t>(samples) : 0;
  layout.payload = kDepthPrefixBytes + layout.rows + layout.plane;
  return layout;
}

}  // namespace

core::Result<std::string> encode_sensor(const SensorInfo& info) try {
  constexpr const char* kApi = "encode_sensor";
  if (!info.depth) {
    return unsupported(kApi,
                       "a sensor with no depth camera, which this version "
                       "does not carry");
  }
  const std::optional<std::uint8_t> role = role_code(info.role);
  if (!role) {
    return refuse(kApi, "role " + std::to_string(static_cast<int>(info.role)) +
                            ", which this version does not name");
  }
  const std::optional<std::uint8_t> pose = pose_code(info.pose);
  if (!pose) {
    return refuse(kApi, "pose " + std::to_string(static_cast<int>(info.pose)) +
                            ", which this version does not name");
  }
  if (info.model.size() > std::numeric_limits<std::uint32_t>::max()) {
    return refuse(kApi, "a model of " + std::to_string(info.model.size()) +
                            " bytes, more than a u32 counts");
  }
  std::string params(kSensorParamsBytes + info.model.size(), '\0');
  auto* p = reinterpret_cast<std::uint8_t*>(params.data());
  put<std::uint16_t>(p + kSensorVersion, kVersion);
  put<std::uint8_t>(p + kSensorRole, *role);
  put<std::uint8_t>(p + kSensorPose, *pose);
  std::uint16_t flags = 0;
  if (info.color) flags |= kSensorHasColor;
  if (info.color_controls) flags |= kSensorHasControls;
  put<std::uint16_t>(p + kSensorFlags, flags);
  put_camera(p + kSensorDepthCamera, *info.depth);
  if (info.color) put_camera(p + kSensorColorCamera, *info.color);
  put_transform(p + kSensorDepthToColor, info.depth_to_color);
  if (info.color_controls) {
    put<std::uint32_t>(p + kSensorExposure, info.color_controls->exposure);
    put<std::uint32_t>(p + kSensorGain, info.color_controls->gain);
  }
  put<std::uint32_t>(p + kSensorModelBytes,
                     static_cast<std::uint32_t>(info.model.size()));
  if (!info.model.empty()) {
    std::memcpy(p + kSensorParamsBytes, info.model.data(), info.model.size());
  }
  return params;
} catch (...) {
  return core::exception_status("wire::encode_sensor",
                                core::Status::Code::InvalidArgument);
}

core::Result<SensorInfo> decode_sensor(std::string_view params, std::string id,
                                       ClockDomain clock,
                                       std::uint32_t fps) try {
  constexpr const char* kApi = "decode_sensor";
  const auto* p = reinterpret_cast<const std::uint8_t*>(params.data());
  if (params.size() < kVersionBytes) {
    return refuse(kApi, std::to_string(params.size()) +
                            " bytes, fewer than the version's 2");
  }
  // The version first: another lays out the rest its own way.
  const auto version = get<std::uint16_t>(p + kSensorVersion);
  if (version != kVersion) {
    return unsupported(
        kApi, "version " + std::to_string(version) + "; this library reads 1");
  }
  if (params.size() < kSensorParamsBytes) {
    return refuse(kApi, std::to_string(params.size()) +
                            " bytes, fewer than the params' 356");
  }
  const std::optional<SyncRole> role = role_of(p[kSensorRole]);
  if (!role) {
    return refuse(kApi, "role " + std::to_string(p[kSensorRole]) +
                            ", which this version does not name");
  }
  const std::optional<PoseSource> pose = pose_of(p[kSensorPose]);
  if (!pose) {
    return refuse(kApi, "pose " + std::to_string(p[kSensorPose]) +
                            ", which this version does not name");
  }
  const auto flags = get<std::uint16_t>(p + kSensorFlags);
  if ((flags & ~kSensorFlagsKnown) != 0) {
    return refuse(kApi, "flags " + hex_text(flags) +
                            ", which this version does not name");
  }
  const auto model_bytes = get<std::uint32_t>(p + kSensorModelBytes);
  if (params.size() - kSensorParamsBytes < model_bytes) {
    return refuse(kApi, std::to_string(params.size()) +
                            " bytes, fewer than the params and the model's " +
                            std::to_string(model_bytes));
  }
  SensorInfo info;
  info.id = std::move(id);
  info.model.assign(params.data() + kSensorParamsBytes, model_bytes);
  info.depth = get_camera(p + kSensorDepthCamera);
  if ((flags & kSensorHasColor) != 0) {
    info.color = get_camera(p + kSensorColorCamera);
  }
  info.depth_to_color = get_transform(p + kSensorDepthToColor);
  info.role = *role;
  info.clock = clock;
  info.pose = *pose;
  info.fps = fps;
  if ((flags & kSensorHasControls) != 0) {
    camera::ColorControls controls;
    controls.exposure = get<std::uint32_t>(p + kSensorExposure);
    controls.gain = get<std::uint32_t>(p + kSensorGain);
    info.color_controls = controls;
  }
  return info;
} catch (...) {
  return core::exception_status("wire::decode_sensor",
                                core::Status::Code::InvalidArgument);
}

core::Result<DepthRecord> encode_depth(const SensorInfo& info,
                                       const RgbdFrame& frame) try {
  constexpr const char* kApi = "encode_depth";
  if (!info.depth) return refuse(kApi, "a sensor without a depth camera");
  if (!frame.has_depth()) return refuse(kApi, "a frame without depth");
  if (frame.depth != nullptr && frame.depth_device.buffer != nullptr) {
    return refuse(kApi, "a frame with both host and device depth");
  }
  if (!same_size(frame.depth_camera.size, info.depth->size)) {
    return refuse(kApi,
                  "a depth camera of " + size_text(frame.depth_camera.size) +
                      ", not the sensor's " + size_text(info.depth->size));
  }
  const bool plane = frame.depth_confidence != nullptr;
  if (plane && frame.depth == nullptr) {
    return refuse(kApi, "a confidence plane beside no host depth");
  }
  if (plane && frame.min_confidence == 0) {
    return refuse(kApi, "a confidence plane at a floor of 0");
  }
  if (!plane && frame.min_confidence != 0) {
    return refuse(kApi, "a confidence floor of " +
                            std::to_string(frame.min_confidence) +
                            " without a plane");
  }
  if (frame.depth == nullptr) {
    return unsupported(kApi,
                       "depth on the device only, which a writer reads back "
                       "first");
  }
  const std::optional<DepthLayout> layout =
      depth_layout(frame.depth_camera.size, plane);
  if (!layout) {
    return refuse(kApi, "a depth camera of " +
                            size_text(frame.depth_camera.size) +
                            ", too large for a record's bytes to be counted");
  }
  DepthRecord record;
  std::uint8_t* p = record.prefix.data();
  put_camera(p + kDepthCamera, frame.depth_camera);
  put_transform(p + kDepthColorToWorld, frame.color_to_world);
  put_transform(p + kDepthDepthToColor, frame.depth_to_color);
  put<float>(p + kDepthUnits, frame.metres_per_unit);
  put<float>(p + kDepthMin, frame.min_depth);
  put<float>(p + kDepthMax, frame.max_depth);
  put<float>(p + kDepthNoise, frame.depth_noise.a);
  put<float>(p + kDepthNoise + 4, frame.depth_noise.b);
  put<float>(p + kDepthNoise + 8, frame.depth_noise.c);
  p[kDepthFlags] = plane ? kDepthHasPlane : 0;
  p[kDepthFloor] = frame.min_confidence;
  record.depth = reinterpret_cast<const std::uint8_t*>(frame.depth);
  record.depth_bytes = layout->rows;
  if (plane) {
    record.confidence = frame.depth_confidence;
    record.confidence_bytes = layout->plane;
  }
  return record;
} catch (...) {
  return core::exception_status("wire::encode_depth",
                                core::Status::Code::InvalidArgument);
}

core::Result<RgbdFrame> decode_depth(const SensorInfo& info,
                                     const std::uint8_t* payload,
                                     std::size_t size,
                                     std::shared_ptr<const void> owner) try {
  constexpr const char* kApi = "decode_depth";
  if (!info.depth) return refuse(kApi, "a sensor without a depth camera");
  if (payload == nullptr) return refuse(kApi, "a null payload");
  if (reinterpret_cast<std::uintptr_t>(payload) % 2 != 0) {
    return refuse(kApi, "a payload at an odd address, where no u16 row starts");
  }
  if (size < kDepthPrefixBytes) {
    return refuse(kApi,
                  std::to_string(size) + " bytes, fewer than the prefix's 392");
  }
  const camera::CameraModel depth_camera = get_camera(payload + kDepthCamera);
  if (!same_size(depth_camera.size, info.depth->size)) {
    return refuse(kApi, "a depth camera of " + size_text(depth_camera.size) +
                            ", not the sensor's " +
                            size_text(info.depth->size));
  }
  const std::uint8_t flags = payload[kDepthFlags];
  if ((flags & ~kDepthHasPlane) != 0) {
    return refuse(kApi, "flags " + hex_text(flags) +
                            ", which this version does not name");
  }
  const bool plane = (flags & kDepthHasPlane) != 0;
  const std::uint8_t floor = payload[kDepthFloor];
  if (plane && floor == 0) {
    return refuse(kApi, "a confidence plane at a floor of 0");
  }
  if (!plane && floor != 0) {
    return refuse(kApi, "a confidence floor of " + std::to_string(floor) +
                            " without a plane");
  }
  const std::optional<DepthLayout> layout =
      depth_layout(depth_camera.size, plane);
  if (!layout || size != layout->payload) {
    return refuse(kApi, std::to_string(size) + " bytes, not the layout's " +
                            (layout ? std::to_string(layout->payload)
                                    : std::string("count, past a size_t")) +
                            " for " + size_text(depth_camera.size) +
                            (plane ? " with a plane" : " without a plane"));
  }
  RgbdFrame frame;
  frame.depth_camera = depth_camera;
  frame.color_to_world = get_transform(payload + kDepthColorToWorld);
  frame.depth_to_color = get_transform(payload + kDepthDepthToColor);
  frame.metres_per_unit = get<float>(payload + kDepthUnits);
  frame.min_depth = get<float>(payload + kDepthMin);
  frame.max_depth = get<float>(payload + kDepthMax);
  frame.depth_noise.a = get<float>(payload + kDepthNoise);
  frame.depth_noise.b = get<float>(payload + kDepthNoise + 4);
  frame.depth_noise.c = get<float>(payload + kDepthNoise + 8);
  // Even, and the prefix a multiple of 8: the rows are u16-aligned.
  frame.depth =
      reinterpret_cast<const std::uint16_t*>(payload + kDepthPrefixBytes);
  if (plane) {
    frame.depth_confidence = payload + kDepthPrefixBytes + layout->rows;
    frame.min_confidence = floor;
  }
  frame.pixels = std::move(owner);
  return frame;
} catch (...) {
  return core::exception_status("wire::decode_depth",
                                core::Status::Code::InvalidArgument);
}

std::string encode_color_params(ColorCodec codec) noexcept {
  // Four bytes: within every standard library's small-string buffer, so
  // nothing is allocated.
  std::string params(kColorParamsBytes, '\0');
  auto* p = reinterpret_cast<std::uint8_t*>(params.data());
  put<std::uint16_t>(p + kColorParamsVersion, kVersion);
  put<std::uint16_t>(p + kColorParamsCodec, static_cast<std::uint16_t>(codec));
  return params;
}

core::Result<ColorCodec> decode_color_params(std::string_view params) try {
  constexpr const char* kApi = "decode_color_params";
  const auto* p = reinterpret_cast<const std::uint8_t*>(params.data());
  if (params.size() < kVersionBytes) {
    return refuse(kApi, std::to_string(params.size()) +
                            " bytes, fewer than the params' 4");
  }
  // The version first: another lays out the rest its own way.
  const auto version = get<std::uint16_t>(p + kColorParamsVersion);
  if (version != kVersion) {
    return unsupported(
        kApi, "version " + std::to_string(version) + "; this library reads 1");
  }
  if (params.size() < kColorParamsBytes) {
    return refuse(kApi, std::to_string(params.size()) +
                            " bytes, fewer than the params' 4");
  }
  const auto codec = get<std::uint16_t>(p + kColorParamsCodec);
  if (codec != static_cast<std::uint16_t>(ColorCodec::HevcMain8)) {
    return unsupported(kApi, "codec " + std::to_string(codec) +
                                 ", which this version does not name");
  }
  return ColorCodec::HevcMain8;
} catch (...) {
  return core::exception_status("wire::decode_color_params",
                                core::Status::Code::InvalidArgument);
}

core::Result<std::array<std::uint8_t, kColorPrefixBytes>> encode_color(
    const SensorInfo& info, const RgbdFrame& frame,
    const ColorCodes& codes) try {
  constexpr const char* kApi = "encode_color";
  if (!info.color) return refuse(kApi, "a sensor without a colour camera");
  if (!frame.has_color()) return refuse(kApi, "a frame without colour");
  if (!same_size(frame.color_camera.size, info.color->size)) {
    return refuse(kApi,
                  "a colour camera of " + size_text(frame.color_camera.size) +
                      ", not the sensor's " + size_text(info.color->size));
  }
  const auto code_byte = [](int code) { return code >= 0 && code <= 255; };
  if (!code_byte(codes.matrix) || !code_byte(codes.transfer) ||
      !code_byte(codes.primaries)) {
    return refuse(kApi, "codes " + std::to_string(codes.matrix) + ", " +
                            std::to_string(codes.transfer) + ", " +
                            std::to_string(codes.primaries) +
                            " (matrix, transfer, primaries): each is 0 to 255");
  }
  if (codes.chroma_location < 0 || codes.chroma_location > kMaxSiting) {
    return refuse(kApi, "chroma siting " +
                            std::to_string(codes.chroma_location) +
                            ", which is 0 to 5");
  }
  std::array<std::uint8_t, kColorPrefixBytes> prefix{};
  std::uint8_t* p = prefix.data();
  put_camera(p + kColorCamera, frame.color_camera);
  p[kColorMatrix] = static_cast<std::uint8_t>(codes.matrix);
  p[kColorTransfer] = static_cast<std::uint8_t>(codes.transfer);
  p[kColorPrimaries] = static_cast<std::uint8_t>(codes.primaries);
  p[kColorFlags] = codes.full_range ? kColorFullRange : 0;
  p[kColorSiting] = static_cast<std::uint8_t>(codes.chroma_location);
  put<float>(p + kColorExposure, frame.color_exposure.duration_s);
  put<float>(p + kColorExposure + 4, frame.color_exposure.iso);
  put<float>(p + kColorExposure + 8, frame.color_exposure.bias_ev);
  return prefix;
} catch (...) {
  return core::exception_status("wire::encode_color",
                                core::Status::Code::InvalidArgument);
}

core::Result<ColorRecord> decode_color(const SensorInfo& info,
                                       const std::uint8_t* payload,
                                       std::size_t size) try {
  constexpr const char* kApi = "decode_color";
  if (payload == nullptr) return refuse(kApi, "a null payload");
  if (size <= kColorPrefixBytes) {
    return refuse(kApi, std::to_string(size) +
                            " bytes, no picture after the prefix's 128");
  }
  if (!info.color) return refuse(kApi, "a sensor without a colour camera");
  ColorRecord record;
  record.camera = get_camera(payload + kColorCamera);
  if (!same_size(record.camera.size, info.color->size)) {
    return refuse(kApi, "a colour camera of " + size_text(record.camera.size) +
                            ", not the sensor's " +
                            size_text(info.color->size));
  }
  const std::uint8_t flags = payload[kColorFlags];
  if ((flags & ~kColorFullRange) != 0) {
    return refuse(kApi, "flags " + hex_text(flags) +
                            ", which this version does not name");
  }
  const int siting = payload[kColorSiting];
  if (siting > kMaxSiting) {
    return refuse(
        kApi, "chroma siting " + std::to_string(siting) + ", which is 0 to 5");
  }
  record.codes.matrix = payload[kColorMatrix];
  record.codes.transfer = payload[kColorTransfer];
  record.codes.primaries = payload[kColorPrimaries];
  record.codes.full_range = (flags & kColorFullRange) != 0;
  record.codes.chroma_location = siting;
  record.exposure.duration_s = get<float>(payload + kColorExposure);
  record.exposure.iso = get<float>(payload + kColorExposure + 4);
  record.exposure.bias_ev = get<float>(payload + kColorExposure + 8);
  record.picture = payload + kColorPrefixBytes;
  record.picture_bytes = size - kColorPrefixBytes;
  return record;
} catch (...) {
  return core::exception_status("wire::decode_color",
                                core::Status::Code::InvalidArgument);
}

}  // namespace volumetric_kit::sensor::wire
