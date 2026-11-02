// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "calibration_json.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace volumetric_kit::sensor::camera::detail {

namespace {

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

// --- checking ----------------------------------------------------------------

core::Status check_apriltag(const AprilTagWorld& tag) {
  const std::string where = "world: apriltag ";
  if (tag.family.empty()) return bad(where + "\"family\" is empty");
  if (!is_utf8(tag.family)) return bad(where + "\"family\" is not UTF-8");
  if (tag.id < 0) return bad(where + "\"id\" is negative");
  if (!std::isfinite(tag.size_m) || !(tag.size_m > 0.0)) {
    return bad(where + "\"size_m\" is not a finite length above 0");
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

}  // namespace volumetric_kit::sensor::camera::detail
