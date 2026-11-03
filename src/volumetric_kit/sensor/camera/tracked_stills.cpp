// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/camera/tracked_stills.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "calibration_json.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"

namespace volumetric_kit::sensor::camera {

namespace {

using detail::bad;
using detail::Json;

// What this file's refusals are prefixed with, at its public functions.
constexpr const char* kWho = "tracked stills";

constexpr std::string_view kFormat = "volumetric_kit.tracked_stills";
constexpr std::uint64_t kVersion = 1;
// The stills' indent in the document, and their camera block's.
constexpr const char* kIndent = "    ";
constexpr const char* kCameraIndent = "     ";

std::string where_of(std::size_t i) {
  return "still " + std::to_string(i) + ": ";
}

// Why `image` names no file in the stills file's directory; empty when it
// does.
std::string name_fault(const std::string& image) {
  if (image.empty()) return "is empty";
  if (image == "." || image == "..") return "is \"" + image + "\", a directory";
  if (image.find('/') != std::string::npos) return "has a '/'";
  if (image.find('\0') != std::string::npos) return "has a NUL";
  if (!detail::is_utf8(image)) return "is not UTF-8";
  return {};
}

// validate_tracked_stills's checks, their refusals not yet named.
core::Status check(const TrackedStills& s) {
  VKC_TRY(detail::check_tracked_session({s.sensor, s.session, s.world_epoch}));
  if (s.stills.empty()) return bad("no stills; a file names at least one");
  std::map<std::string_view, std::size_t> named;
  for (std::size_t i = 0; i < s.stills.size(); ++i) {
    const TrackedStill& still = s.stills[i];
    const std::string where = where_of(i);
    const std::string fault = name_fault(still.image);
    if (!fault.empty()) return bad(where + "\"image\" " + fault);
    const auto [before, fresh] = named.emplace(still.image, i);
    if (!fresh) {
      return bad(where + "\"image\" " + detail::text(still.image) +
                 " is still " + std::to_string(before->second) + "'s too");
    }
    if (i > 0 && !(still.timestamp_ns > s.stills[i - 1].timestamp_ns)) {
      return bad(where + "\"timestamp_ns\" " +
                 std::to_string(still.timestamp_ns) + " is not after still " +
                 std::to_string(i - 1) + "'s " +
                 std::to_string(s.stills[i - 1].timestamp_ns));
    }
    const core::Status model = check_camera_model(still.model);
    if (!model.ok()) return bad(where + "camera: " + model.message());
    const core::Status rigid = check_rigid(still.cam_to_tracking);
    if (!rigid.ok()) return bad(where + "cam_to_tracking: " + rigid.message());
  }
  return {};
}

// The whole number at `key`, of 64 bits.
core::Result<std::uint64_t> time_at(const Json& object, const char* key,
                                    const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number_unsigned()) {
    return bad(where + "\"" + key + "\" is not a 64-bit whole number");
  }
  return it->get<std::uint64_t>();
}

core::Result<TrackedStill> still_of(const Json& block,
                                    const std::string& where) {
  if (!block.is_object()) return bad(where + "not an object");
  TrackedStill still;
  VKC_ASSIGN(still.image, detail::string_at(block, "image", where));
  VKC_ASSIGN(still.timestamp_ns, time_at(block, "timestamp_ns", where));
  VKC_ASSIGN(const Json* camera, detail::object_at(block, "camera", where));
  if (camera == nullptr) return bad(where + "no \"camera\" object");
  VKC_ASSIGN(const CameraCalibration calibration,
             detail::camera_calibration(*camera, where + "camera "));
  still.model = calibration.model;
  VKC_ASSIGN(const Json* pose,
             detail::object_at(block, "cam_to_tracking", where));
  if (pose == nullptr) return bad(where + "no \"cam_to_tracking\" object");
  VKC_ASSIGN(still.cam_to_tracking,
             detail::transform(*pose, where + "cam_to_tracking "));
  return still;
}

// parse_tracked_stills's reading, its refusals not yet named.
core::Result<TrackedStills> parse_document(std::string_view json_text) {
  const Json doc = Json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) return bad("not JSON");
  if (!doc.is_object()) return bad("not a JSON object");
  VKC_TRY(detail::check_format(doc, kFormat, kVersion));
  VKC_ASSIGN(detail::TrackedSession session, detail::tracked_session(doc));
  TrackedStills s;
  s.sensor = std::move(session.sensor);
  s.session = session.session;
  s.world_epoch = session.world_epoch;
  const auto stills = doc.find("stills");
  if (stills == doc.end() || !stills->is_array()) {
    return bad("no \"stills\" array");
  }
  s.stills.reserve(stills->size());
  for (std::size_t i = 0; i < stills->size(); ++i) {
    VKC_ASSIGN(TrackedStill still, still_of((*stills)[i], where_of(i)));
    s.stills.push_back(std::move(still));
  }
  VKC_TRY(check(s));
  return s;
}

}  // namespace

core::Status validate_tracked_stills(const TrackedStills& stills) try {
  return check(stills).with_context(kWho);
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Result<TrackedStills> parse_tracked_stills(std::string_view json) try {
  auto stills = parse_document(json);
  if (!stills.ok()) return stills.status().with_context(kWho);
  return stills;
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Result<TrackedStills> read_tracked_stills(const std::string& path) try {
  VKC_ASSIGN(const std::string text, core::read_file(path));
  auto stills = parse_tracked_stills(text);
  if (!stills.ok()) return stills.status().with_context(path);
  return stills;
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Result<std::string> format_tracked_stills(
    const TrackedStills& stills) try {
  VKC_TRY(validate_tracked_stills(stills));
  using detail::text;
  std::string out = "{\n  \"format\": " + text(std::string(kFormat)) +
                    ",\n  \"version\": " + text(kVersion) + ",\n  " +
                    detail::tracked_session_text(
                        {stills.sensor, stills.session, stills.world_epoch}) +
                    ",\n  \"stills\": [";
  const std::string indent = kIndent;
  for (std::size_t i = 0; i < stills.stills.size(); ++i) {
    const TrackedStill& s = stills.stills[i];
    // A tracked sensor's model is its SDK's (the header).
    const CameraCalibration camera{s.model, IntrinsicsSource::Factory};
    out += (i == 0 ? "\n" : ",\n") + indent + "{\"image\": " + text(s.image) +
           ", \"timestamp_ns\": " + text(s.timestamp_ns) + ",\n" + indent +
           " \"camera\": " + detail::camera_text(camera, kCameraIndent) +
           ",\n" + indent + " \"cam_to_tracking\": " +
           detail::transform_text(s.cam_to_tracking) + "}";
  }
  out += "\n  ]\n}\n";
  return out;
} catch (...) {
  return core::exception_status(kWho, core::Status::Code::InvalidArgument);
}

core::Status write_tracked_stills(const std::string& path,
                                  const TrackedStills& stills,
                                  core::WriteMode mode) {
  VKC_ASSIGN(const std::string text, format_tracked_stills(stills));
  return core::write_file(path, text, mode);
}

}  // namespace volumetric_kit::sensor::camera
