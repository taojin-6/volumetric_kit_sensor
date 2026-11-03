// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/camera/tracked_registration.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

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

detail::TrackedSession session_of(const TrackedRegistration& r) {
  return {r.sensor, r.session, r.world_epoch};
}

// validate_tracked_registration's checks, their refusals not yet named.
core::Status check(const TrackedRegistration& r) {
  VKC_TRY(detail::check_tracked_session(session_of(r)));
  VKC_TRY(detail::check_world_kind(r.world, /*allow_none=*/false));
  if (!r.world.apriltag && !detail::is_utf8(r.world.sensor)) {
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
  VKC_TRY(detail::check_format(doc, kFormat, kVersion));
  VKC_ASSIGN(detail::TrackedSession session, detail::tracked_session(doc));
  TrackedRegistration r;
  r.sensor = std::move(session.sensor);
  r.session = session.session;
  r.world_epoch = session.world_epoch;
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
         ",\n  \"version\": " + text(kVersion) + ",\n  " +
         detail::tracked_session_text(session_of(r)) +
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
