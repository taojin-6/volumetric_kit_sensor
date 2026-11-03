// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "calibration_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace volumetric_kit::sensor::camera::detail {

namespace {

constexpr std::string_view kRational = "rational";
constexpr char kHexDigits[] = "0123456789abcdef";

const char* to_string(IntrinsicsSource source) {
  return source == IntrinsicsSource::Calibrated ? "calibrated" : "factory";
}

// The value of hex digit `c`, either case; -1 for anything else.
int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// A session's 16 bytes from 32 hex digits, byte 0 first, each byte's high
// digit first.
core::Result<std::array<std::uint8_t, 16>> session_of(const std::string& hex) {
  std::array<std::uint8_t, 16> session{};
  if (hex.size() != 2 * session.size()) {
    return bad("\"session\" is " + std::to_string(hex.size()) +
               " bytes long; a session is 32 hex digits");
  }
  for (std::size_t i = 0; i < session.size(); ++i) {
    const int high = hex_value(hex[2 * i]);
    const int low = hex_value(hex[2 * i + 1]);
    if (high < 0 || low < 0) {
      return bad(
          "\"session\" has a digit that is not hex; a session is 32 "
          "hex digits");
    }
    session[i] = static_cast<std::uint8_t>(high * 16 + low);
  }
  return session;
}

std::string session_hex(const std::array<std::uint8_t, 16>& session) {
  std::string hex;
  for (const std::uint8_t byte : session) {
    hex += kHexDigits[byte >> 4];
    hex += kHexDigits[byte & 0xF];
  }
  return hex;
}

core::Result<PinholeIntrinsics> pinhole(const Json& block,
                                        const std::string& where) {
  PinholeIntrinsics k;
  VKC_ASSIGN(k.fx, number(block, "fx", where));
  VKC_ASSIGN(k.fy, number(block, "fy", where));
  VKC_ASSIGN(k.cx, number(block, "cx", where));
  VKC_ASSIGN(k.cy, number(block, "cy", where));
  return k;
}

core::Result<RationalDistortion> rational(const Json& block,
                                          const std::string& where) {
  const auto model = block.find("model");
  if (model == block.end() || !model->is_string()) {
    return bad(where + "missing string \"model\"");
  }
  if (model->get<std::string>() != kRational) {
    return unsupported(where + "model \"" + model->get<std::string>() +
                       "\"; this reader knows \"rational\"");
  }
  RationalDistortion d;
  VKC_ASSIGN(d.k1, number(block, "k1", where));
  VKC_ASSIGN(d.k2, number(block, "k2", where));
  VKC_ASSIGN(d.p1, number(block, "p1", where));
  VKC_ASSIGN(d.p2, number(block, "p2", where));
  VKC_ASSIGN(d.k3, number(block, "k3", where));
  VKC_ASSIGN(d.k4, number(block, "k4", where));
  VKC_ASSIGN(d.k5, number(block, "k5", where));
  VKC_ASSIGN(d.k6, number(block, "k6", where));
  return d;
}

// A whole number, signed, in an int's range; the file's checks judge its sign.
core::Result<int> integer(const Json& object, const char* key,
                          const std::string& where) {
  const auto it = object.find(key);
  const bool fits =
      it != object.end() &&
      (it->is_number_unsigned()
           ? it->get<std::uint64_t>() <=
                 std::uint64_t{std::numeric_limits<int>::max()}
           : it->is_number_integer() &&
                 it->get<std::int64_t>() >= std::numeric_limits<int>::min());
  if (!fits) return bad(where + "\"" + key + "\" is not a 32-bit integer");
  return static_cast<int>(it->get<std::int64_t>());
}

core::Result<Vec3d> vector3(const Json& object, const char* key,
                            const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_array() || it->size() != 3) {
    return bad(where + "\"" + key + "\" is not 3 numbers");
  }
  Vec3d v(0.0);
  for (std::size_t i = 0; i < 3; ++i) {
    const Json& e = (*it)[i];
    if (!e.is_number() || !std::isfinite(e.get<double>())) {
      return bad(where + "\"" + key + "\" is not 3 finite numbers");
    }
    v[static_cast<int>(i)] = e.get<double>();
  }
  return v;
}

}  // namespace

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument(what);
}

core::Status unsupported(const std::string& what) {
  return core::Status::unsupported(what);
}

bool is_utf8(const std::string& s) {
  try {  // JSON text is UTF-8, and dump() throws on anything else
    (void)Json(s).dump();
    return true;
  } catch (const Json::exception&) {
    return false;
  }
}

// --- reading -----------------------------------------------------------------

core::Status check_format(const Json& doc, std::string_view format,
                          std::uint64_t version) {
  const auto it = doc.find("format");
  if (it == doc.end() || !it->is_string() || *it != format) {
    return bad("\"format\" is not \"" + std::string(format) + "\"");
  }
  const auto v = doc.find("version");
  if (v == doc.end() || !v->is_number_unsigned()) {
    return bad("missing whole number \"version\"");
  }
  const std::uint64_t n = v->get<std::uint64_t>();
  if (n > version) {
    return unsupported("version " + std::to_string(n) +
                       "; this reader knows version " +
                       std::to_string(version));
  }
  if (n < 1) {
    return bad("version " + std::to_string(n) + "; versions start at 1");
  }
  return {};
}

core::Result<std::string> string_at(const Json& object, const char* key,
                                    const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_string()) {
    return bad(where + "missing string \"" + key + "\"");
  }
  return it->get<std::string>();
}

core::Result<double> number(const Json& object, const char* key,
                            const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    return bad(where + "missing number \"" + key + "\"");
  }
  const double v = it->get<double>();
  if (!std::isfinite(v)) return bad(where + "\"" + key + "\" is not finite");
  return v;
}

core::Result<std::uint32_t> whole_number(const Json& object, const char* key,
                                         const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number_unsigned() ||
      it->get<std::uint64_t>() > std::numeric_limits<std::uint32_t>::max()) {
    return bad(where + "\"" + key + "\" is not a 32-bit whole number");
  }
  return static_cast<std::uint32_t>(it->get<std::uint64_t>());
}

core::Result<const Json*> object_at(const Json& parent, const char* key,
                                    const std::string& where) {
  const auto it = parent.find(key);
  if (it == parent.end()) return static_cast<const Json*>(nullptr);
  if (!it->is_object()) return bad(where + "\"" + key + "\" is not an object");
  return &*it;
}

core::Result<Mat4d> transform(const Json& block, const std::string& where) {
  RodriguesTransform t;
  VKC_ASSIGN(t.rvec, vector3(block, "rvec", where));
  VKC_ASSIGN(t.tvec, vector3(block, "tvec", where));
  return matrix_from_rodrigues(t);
}

core::Result<WorldFrame> world_frame(const Json& block) {
  WorldFrame world;
  const auto sensor = block.find("sensor");
  VKC_ASSIGN(const Json* tag, object_at(block, "apriltag", "world: "));
  if (sensor == block.end() && tag == nullptr) {
    return unsupported(
        "world: this reader knows {\"sensor\": id} and {\"apriltag\": tag}");
  }
  if (sensor != block.end()) {
    if (!sensor->is_string()) return bad("world: \"sensor\" is not a string");
    world.sensor = sensor->get<std::string>();
    // Empty is the struct's "not a sensor", so a named sensor has an id.
    if (world.sensor.empty()) return bad("world: \"sensor\" is empty");
  }
  if (tag != nullptr) {
    const std::string where = "world: apriltag ";
    AprilTagWorld t;
    const auto family = tag->find("family");
    if (family == tag->end() || !family->is_string()) {
      return bad(where + "missing string \"family\"");
    }
    t.family = family->get<std::string>();
    VKC_ASSIGN(t.id, integer(*tag, "id", where));
    VKC_ASSIGN(t.size_m, number(*tag, "size_m", where));
    world.apriltag = std::move(t);
  }
  return world;
}

core::Result<CameraCalibration> camera_calibration(const Json& block,
                                                   const std::string& where) {
  CameraCalibration camera;
  const auto source = block.find("source");
  if (source == block.end() || !source->is_string()) {
    return bad(where + "missing string \"source\"");
  }
  if (*source == "factory") {
    camera.source = IntrinsicsSource::Factory;
  } else if (*source == "calibrated") {
    camera.source = IntrinsicsSource::Calibrated;
  } else {
    return bad(where + "\"source\" is \"" + source->get<std::string>() +
               "\", not \"factory\" or \"calibrated\"");
  }
  VKC_ASSIGN(camera.model.size.width, whole_number(block, "width", where));
  VKC_ASSIGN(camera.model.size.height, whole_number(block, "height", where));
  VKC_ASSIGN(const Json* intrinsics, object_at(block, "intrinsics", where));
  if (intrinsics == nullptr) return bad(where + "no \"intrinsics\" object");
  VKC_ASSIGN(camera.model.intrinsics,
             pinhole(*intrinsics, where + "intrinsics "));
  VKC_ASSIGN(const Json* distortion, object_at(block, "distortion", where));
  if (distortion != nullptr) {
    VKC_ASSIGN(camera.model.distortion,
               rational(*distortion, where + "distortion "));
  }
  return camera;
}

core::Result<TrackedSession> tracked_session(const Json& doc) {
  TrackedSession s;
  VKC_ASSIGN(s.sensor, string_at(doc, "sensor", ""));
  VKC_ASSIGN(const std::string hex, string_at(doc, "session", ""));
  VKC_ASSIGN(s.session, session_of(hex));
  VKC_ASSIGN(const std::uint32_t epoch, whole_number(doc, "world_epoch", ""));
  if (epoch > std::numeric_limits<std::uint16_t>::max()) {
    return bad("\"world_epoch\" " + std::to_string(epoch) + " is past 65535");
  }
  s.world_epoch = static_cast<std::uint16_t>(epoch);
  return s;
}

// --- checking ----------------------------------------------------------------

core::Status check_world_kind(const WorldFrame& world, bool allow_none) {
  if (const auto& tag = world.apriltag) {
    if (!world.sensor.empty()) {
      return bad("world: names both sensor " + world.sensor +
                 " and an AprilTag; it is one or the other");
    }
    const std::string where = "world: apriltag ";
    if (tag->family.empty()) return bad(where + "\"family\" is empty");
    if (!is_utf8(tag->family)) return bad(where + "\"family\" is not UTF-8");
    if (tag->id < 0) return bad(where + "\"id\" is negative");
    if (!std::isfinite(tag->size_m) || !(tag->size_m > 0.0)) {
      return bad(where + "\"size_m\" is not a finite length above 0");
    }
    return {};
  }
  if (world.sensor.empty() && !allow_none) {
    return bad("world: names neither a sensor nor an AprilTag");
  }
  return {};
}

core::Status check_tracked_session(const TrackedSession& s) {
  if (s.sensor.empty()) return bad("\"sensor\" is empty");
  if (!is_utf8(s.sensor)) return bad("\"sensor\" is not UTF-8");
  if (std::all_of(s.session.begin(), s.session.end(),
                  [](std::uint8_t b) { return b == 0; })) {
    return bad("\"session\" is all zero, which names no session");
  }
  return {};
}

// --- writing -----------------------------------------------------------------

std::string text(double v) { return Json(v).dump(); }
std::string text(std::uint64_t v) { return Json(v).dump(); }
std::string text(const std::string& s) { return Json(s).dump(); }

std::string vector_text(const Vec3d& v) {
  return "[" + text(v.x) + ", " + text(v.y) + ", " + text(v.z) + "]";
}

std::string transform_text(const Mat4d& transform) {
  const RodriguesTransform t = rodrigues_from_matrix(transform);
  return "{\"rvec\": " + vector_text(t.rvec) +
         ", \"tvec\": " + vector_text(t.tvec) + "}";
}

std::string world_text(const WorldFrame& world) {
  if (const auto& tag = world.apriltag) {
    return "{\"apriltag\": {\"family\": " + text(tag->family) +
           ", \"id\": " + text(static_cast<std::uint64_t>(tag->id)) +
           ", \"size_m\": " + text(tag->size_m) + "}}";
  }
  return "{\"sensor\": " + text(world.sensor) + "}";
}

std::string camera_text(const CameraCalibration& camera,
                        const std::string& indent) {
  const CameraModel& m = camera.model;
  const RationalDistortion& d = m.distortion;
  return std::string("{\"source\": \"") + to_string(camera.source) +
         "\", \"width\": " + text(std::uint64_t{m.size.width}) +
         ", \"height\": " + text(std::uint64_t{m.size.height}) + ",\n" +
         indent + "  \"intrinsics\": {\"fx\": " + text(m.intrinsics.fx) +
         ", \"fy\": " + text(m.intrinsics.fy) +
         ", \"cx\": " + text(m.intrinsics.cx) +
         ", \"cy\": " + text(m.intrinsics.cy) + "},\n" + indent +
         "  \"distortion\": {\"model\": \"rational\", \"k1\": " + text(d.k1) +
         ", \"k2\": " + text(d.k2) + ", \"p1\": " + text(d.p1) +
         ", \"p2\": " + text(d.p2) + ", \"k3\": " + text(d.k3) +
         ", \"k4\": " + text(d.k4) + ", \"k5\": " + text(d.k5) +
         ", \"k6\": " + text(d.k6) + "}}";
}

std::string tracked_session_text(const TrackedSession& session) {
  return "\"sensor\": " + text(session.sensor) +
         ",\n  \"session\": " + text(session_hex(session.session)) +
         ",\n  \"world_epoch\": " + text(std::uint64_t{session.world_epoch});
}

}  // namespace volumetric_kit::sensor::camera::detail
