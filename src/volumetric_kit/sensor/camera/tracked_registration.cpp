// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/camera/tracked_registration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "calibration_json.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::sensor::camera {

namespace {

using detail::bad;
using detail::Json;

// What this file's refusals are prefixed with, at its public functions.
constexpr const char* kWho = "tracked registration";

constexpr std::string_view kFormat = "volumetric_kit.tracked_registration";
constexpr std::uint64_t kVersion = 1;
constexpr char kHexDigits[] = "0123456789abcdef";

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
    return bad("\"session\" has " + std::to_string(hex.size()) +
               " digits; a session is 32 hex digits");
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

std::string session_text(const std::array<std::uint8_t, 16>& session) {
  std::string hex;
  for (const std::uint8_t byte : session) {
    hex += kHexDigits[byte >> 4];
    hex += kHexDigits[byte & 0xF];
  }
  return hex;
}

core::Result<std::string> string_at(const Json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_string()) {
    return bad(std::string("missing string \"") + key + "\"");
  }
  return it->get<std::string>();
}

// validate_tracked_registration's checks, their refusals not yet named.
core::Status check(const TrackedRegistration& r) {
  if (r.sensor.empty()) return bad("\"sensor\" is empty");
  if (!detail::is_utf8(r.sensor)) return bad("\"sensor\" is not UTF-8");
  if (std::all_of(r.session.begin(), r.session.end(),
                  [](std::uint8_t b) { return b == 0; })) {
    return bad("\"session\" is all zero, which names no session");
  }
  const std::string& world_sensor = r.world.sensor;
  if (r.world.apriltag) {
    if (!world_sensor.empty()) {
      return bad("world: names both sensor " + world_sensor +
                 " and an AprilTag; it is one or the other");
    }
    VKC_TRY(detail::check_apriltag(*r.world.apriltag));
  } else if (world_sensor.empty()) {
    return bad("world: names neither a sensor nor an AprilTag");
  } else if (!detail::is_utf8(world_sensor)) {
    return bad("world: \"sensor\" is not UTF-8");
  }
  const core::Status rigid = check_rigid(r.tracking_to_world);
  if (!rigid.ok()) return bad("tracking_to_world: " + rigid.message());
  if (!std::isfinite(r.rms_px) || !(r.rms_px >= 0.0)) {
    return bad("fit: \"rms_px\" is not a finite error of at least 0");
  }
  if (r.views == 0) {
    return bad("fit: \"views\" is 0; a fit uses at least one still");
  }
  if (!std::isfinite(r.edge_m) || !(r.edge_m > 0.0)) {
    return bad("fit: \"edge_m\" is not a finite length above 0");
  }
  return {};
}

// parse_tracked_registration's reading, its refusals not yet named.
core::Result<TrackedRegistration> parse_document(std::string_view json_text) {
  const Json doc = Json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) return bad("not JSON");
  if (!doc.is_object()) return bad("not a JSON object");
  const auto format = doc.find("format");
  if (format == doc.end() || !format->is_string() || *format != kFormat) {
    return bad("\"format\" is not \"" + std::string(kFormat) + "\"");
  }
  const auto version = doc.find("version");
  if (version == doc.end() || !version->is_number_unsigned()) {
    return bad("missing whole number \"version\"");
  }
  const std::uint64_t v = version->get<std::uint64_t>();
  if (v > kVersion) {
    return detail::unsupported("version " + std::to_string(v) +
                               "; this reader knows version 1");
  }
  if (v < kVersion) {
    return bad("version " + std::to_string(v) + "; versions start at 1");
  }
  TrackedRegistration r;
  VKC_ASSIGN(r.sensor, string_at(doc, "sensor"));
  VKC_ASSIGN(const std::string session, string_at(doc, "session"));
  VKC_ASSIGN(r.session, session_of(session));
  VKC_ASSIGN(const std::uint32_t epoch,
             detail::whole_number(doc, "world_epoch", ""));
  if (epoch > std::numeric_limits<std::uint16_t>::max()) {
    return bad("\"world_epoch\" " + std::to_string(epoch) + " is past 65535");
  }
  r.world_epoch = static_cast<std::uint16_t>(epoch);
  VKC_ASSIGN(const Json* world, detail::object_at(doc, "world", ""));
  if (world == nullptr) return bad("no \"world\" object");
  VKC_ASSIGN(r.world, detail::world_frame(*world));
  VKC_ASSIGN(const Json* to_world,
             detail::object_at(doc, "tracking_to_world", ""));
  if (to_world == nullptr) return bad("no \"tracking_to_world\" object");
  VKC_ASSIGN(r.tracking_to_world,
             detail::transform(*to_world, "tracking_to_world "));
  VKC_ASSIGN(const Json* fit, detail::object_at(doc, "fit", ""));
  if (fit == nullptr) return bad("no \"fit\" object");
  VKC_ASSIGN(r.rms_px, detail::number(*fit, "rms_px", "fit: "));
  VKC_ASSIGN(r.views, detail::whole_number(*fit, "views", "fit: "));
  VKC_ASSIGN(r.edge_m, detail::number(*fit, "edge_m", "fit: "));
  VKC_TRY(check(r));
  return r;
}

}  // namespace

core::Status validate_tracked_registration(
    const TrackedRegistration& registration) try {
  return check(registration).with_context(kWho);
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Result<TrackedRegistration> parse_tracked_registration(
    std::string_view json) try {
  auto registration = parse_document(json);
  if (!registration.ok()) return registration.status().with_context(kWho);
  return registration;
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Result<TrackedRegistration> read_tracked_registration(
    const std::string& path) try {
  VKC_ASSIGN(const std::string text, core::read_file(path));
  auto registration = parse_tracked_registration(text);
  if (!registration.ok()) return registration.status().with_context(path);
  return registration;
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Result<std::string> format_tracked_registration(
    const TrackedRegistration& registration) try {
  VKC_TRY(validate_tracked_registration(registration));
  using detail::text;
  const TrackedRegistration& r = registration;
  return "{\n  \"format\": " + text(std::string(kFormat)) +
         ",\n  \"version\": " + text(kVersion) +
         ",\n  \"sensor\": " + text(r.sensor) +
         ",\n  \"session\": " + text(session_text(r.session)) +
         ",\n  \"world_epoch\": " + text(std::uint64_t{r.world_epoch}) +
         ",\n  \"world\": " + detail::world_text(r.world) +
         ",\n  \"tracking_to_world\": " +
         detail::transform_text(r.tracking_to_world) +
         ",\n  \"fit\": {\"rms_px\": " + text(r.rms_px) +
         ", \"views\": " + text(std::uint64_t{r.views}) +
         ", \"edge_m\": " + text(r.edge_m) + "}\n}\n";
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Status write_tracked_registration(const std::string& path,
                                        const TrackedRegistration& registration,
                                        core::WriteMode mode) {
  VKC_ASSIGN(const std::string text, format_tracked_registration(registration));
  return core::write_file(path, text, mode);
}

}  // namespace volumetric_kit::sensor::camera
