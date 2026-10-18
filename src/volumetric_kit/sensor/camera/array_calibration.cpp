// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/camera/array_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "volumetric_kit/core/base/exception_status.hpp"

namespace volumetric_kit::recon::camera {

namespace {

using nlohmann::json;

constexpr std::string_view kFormat = "volumetric_kit.array_calibration";
constexpr std::uint64_t kVersion = 2;
constexpr std::string_view kRational = "rational";
// How far a world sensor's pose may be from the origin: a writer puts it there
// exactly, so this absorbs only the Rodrigues round trip.
constexpr double kOriginTolerance = 1e-9;

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("array calibration: " + what);
}

core::Status unsupported(const std::string& what) {
  return core::Status::unsupported("array calibration: " + what);
}

const char* to_string(IntrinsicsSource source) {
  return source == IntrinsicsSource::Calibrated ? "calibrated" : "factory";
}

bool is_utf8(const std::string& s) {
  try {  // JSON text is UTF-8, and dump() throws on anything else
    (void)json(s).dump();
    return true;
  } catch (const json::exception&) {
    return false;
  }
}

// --- reading -----------------------------------------------------------------

core::Result<double> number(const json& object, const char* key,
                            const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    return bad(where + "missing number \"" + key + "\"");
  }
  const double v = it->get<double>();
  if (!std::isfinite(v)) return bad(where + "\"" + key + "\" is not finite");
  return v;
}

core::Result<std::uint32_t> whole_number(const json& object, const char* key,
                                         const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number_unsigned() ||
      it->get<std::uint64_t>() > std::numeric_limits<std::uint32_t>::max()) {
    return bad(where + "\"" + key + "\" is not a 32-bit whole number");
  }
  return static_cast<std::uint32_t>(it->get<std::uint64_t>());
}

// A whole number, signed, in an int's range; validate judges its sign.
core::Result<int> integer(const json& object, const char* key,
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

// The object at `key`; null when there is none.
core::Result<const json*> object_at(const json& parent, const char* key,
                                    const std::string& where) {
  const auto it = parent.find(key);
  if (it == parent.end()) return static_cast<const json*>(nullptr);
  if (!it->is_object()) return bad(where + "\"" + key + "\" is not an object");
  return &*it;
}

core::Result<Vec3d> vector3(const json& object, const char* key,
                            const std::string& where) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_array() || it->size() != 3) {
    return bad(where + "\"" + key + "\" is not 3 numbers");
  }
  Vec3d v(0.0);
  for (std::size_t i = 0; i < 3; ++i) {
    const json& e = (*it)[i];
    if (!e.is_number() || !std::isfinite(e.get<double>())) {
      return bad(where + "\"" + key + "\" is not 3 finite numbers");
    }
    v[static_cast<int>(i)] = e.get<double>();
  }
  return v;
}

core::Result<Mat4d> transform(const json& block, const std::string& where) {
  RodriguesTransform t;
  VKC_ASSIGN(t.rvec, vector3(block, "rvec", where));
  VKC_ASSIGN(t.tvec, vector3(block, "tvec", where));
  return matrix_from_rodrigues(t);
}

core::Result<PinholeIntrinsics> pinhole(const json& block,
                                        const std::string& where) {
  PinholeIntrinsics k;
  VKC_ASSIGN(k.fx, number(block, "fx", where));
  VKC_ASSIGN(k.fy, number(block, "fy", where));
  VKC_ASSIGN(k.cx, number(block, "cx", where));
  VKC_ASSIGN(k.cy, number(block, "cy", where));
  return k;
}

core::Result<RationalDistortion> rational(const json& block,
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

core::Result<CameraCalibration> camera_v2(const json& block,
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
  VKC_ASSIGN(const json* intrinsics, object_at(block, "intrinsics", where));
  if (intrinsics == nullptr) return bad(where + "no \"intrinsics\" object");
  VKC_ASSIGN(camera.model.intrinsics,
             pinhole(*intrinsics, where + "intrinsics "));
  VKC_ASSIGN(const json* distortion, object_at(block, "distortion", where));
  if (distortion != nullptr) {
    VKC_ASSIGN(camera.model.distortion,
               rational(*distortion, where + "distortion "));
  }
  return camera;
}

core::Result<SensorCalibration> sensor_v2(const std::string& id,
                                          const json& block) {
  SensorCalibration sensor;
  sensor.id = id;
  const std::string where = "sensor " + id + ": ";
  if (!block.is_object()) return bad(where + "not an object");
  VKC_ASSIGN(const json* pose, object_at(block, "pose", where));
  if (pose != nullptr) {
    VKC_ASSIGN(const Mat4d extrinsic, transform(*pose, where + "pose "));
    sensor.color_to_world = rigid_inverse(extrinsic);
  }
  for (const auto& [key, field] :
       {std::pair{"color", &sensor.color}, std::pair{"depth", &sensor.depth}}) {
    VKC_ASSIGN(const json* camera, object_at(block, key, where));
    if (camera != nullptr) {
      VKC_ASSIGN(*field, camera_v2(*camera, where + key + " "));
    }
  }
  VKC_ASSIGN(const json* depth_to_color,
             object_at(block, "depth_to_color", where));
  if (depth_to_color != nullptr) {
    VKC_ASSIGN(sensor.depth_to_color,
               transform(*depth_to_color, where + "depth_to_color "));
  }
  return sensor;
}

// Both kinds are read where a world names both, for validate to refuse.
core::Result<WorldFrame> world_v2(const json& block) {
  WorldFrame world;
  const auto sensor = block.find("sensor");
  VKC_ASSIGN(const json* tag, object_at(block, "apriltag", "world: "));
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

core::Result<ArrayCalibration> parse_v2(const json& doc) {
  const auto version = doc.find("version");
  if (version == doc.end() || !version->is_number_unsigned()) {
    return bad("missing whole number \"version\"");
  }
  const std::uint64_t v = version->get<std::uint64_t>();
  if (v > kVersion) {
    return unsupported("version " + std::to_string(v) +
                       "; this reader knows version 2");
  }
  if (v != kVersion) {
    return bad("version " + std::to_string(v) +
               "; a file without \"format\" is version 1");
  }
  VKC_ASSIGN(const json* sensors, object_at(doc, "device_calibration", ""));
  if (sensors == nullptr) return bad("no \"device_calibration\" object");
  ArrayCalibration array;
  VKC_ASSIGN(const json* world, object_at(doc, "world", ""));
  if (world != nullptr) {
    VKC_ASSIGN(array.world, world_v2(*world));
  }
  for (auto it = sensors->begin(); it != sensors->end(); ++it) {
    VKC_ASSIGN(SensorCalibration sensor, sensor_v2(it.key(), *it));
    array.sensors.push_back(std::move(sensor));
  }
  return array;
}

core::Result<ArrayCalibration> parse_v1(const json& doc) {
  VKC_ASSIGN(const json* sensors, object_at(doc, "device_calibration", ""));
  if (sensors == nullptr) return bad("no \"device_calibration\" object");
  ArrayCalibration array;
  for (auto it = sensors->begin(); it != sensors->end(); ++it) {
    if (!it->is_object()) continue;  // e.g. "pose_calib_file_path"
    SensorCalibration sensor;
    sensor.id = it.key();
    const std::string where = "sensor " + sensor.id + ": ";
    VKC_ASSIGN(const json* pose, object_at(*it, "pose", where));
    if (pose == nullptr) return bad(where + "no \"pose\" object");
    VKC_ASSIGN(const Mat4d extrinsic, transform(*pose, where + "pose "));
    sensor.color_to_world = rigid_inverse(extrinsic);
    array.sensors.push_back(std::move(sensor));
  }
  return array;
}

// --- writing -----------------------------------------------------------------

// A value as JSON text: a number in a form that reads back to the same double
// (at most 17 significant digits), a string quoted and escaped.
std::string text(double v) { return json(v).dump(); }
std::string text(std::uint64_t v) { return json(v).dump(); }
std::string text(const std::string& s) { return json(s).dump(); }

std::string vector_text(const Vec3d& v) {
  return "[" + text(v.x) + ", " + text(v.y) + ", " + text(v.z) + "]";
}

std::string transform_text(const Mat4d& transform) {
  const RodriguesTransform t = rodrigues_from_matrix(transform);
  return "{\"rvec\": " + vector_text(t.rvec) +
         ", \"tvec\": " + vector_text(t.tvec) + "}";
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

core::Status check_sensor(const SensorCalibration& sensor) {
  const std::string who = "sensor " + sensor.id + ": ";
  if (sensor.color_to_world) {
    const core::Status rigid = check_rigid(*sensor.color_to_world);
    if (!rigid.ok()) return bad(who + "pose: " + rigid.message());
  }
  for (const auto& [name, camera] :
       {std::pair{"color", &sensor.color}, std::pair{"depth", &sensor.depth}}) {
    if (!*camera) continue;
    const core::Status model = check_camera_model((*camera)->model);
    if (!model.ok()) return bad(who + name + ": " + model.message());
  }
  if (sensor.depth_to_color) {
    if (!sensor.depth) {
      return bad(who + "depth_to_color without a depth camera");
    }
    const core::Status rigid = check_rigid(*sensor.depth_to_color);
    if (!rigid.ok()) return bad(who + "depth_to_color: " + rigid.message());
  }
  return {};
}

// A tag is not in the array, so nothing of it is checked against the poses.
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

core::Status check_world(const ArrayCalibration& array) {
  const std::string& id = array.world.sensor;
  if (array.world.apriltag) {
    if (!id.empty()) {
      return bad("world: names both sensor " + id +
                 " and an AprilTag; it is one or the other");
    }
    return check_apriltag(*array.world.apriltag);
  }
  if (id.empty()) return {};
  const SensorCalibration* sensor = find_sensor(array, id);
  if (sensor == nullptr) {
    return bad("world: sensor " + id + " is not in the array");
  }
  if (!sensor->color_to_world) {
    return bad("world: sensor " + id + " is not posed");
  }
  const Mat4d& m = *sensor->color_to_world;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (std::fabs(m[c][r] - (c == r ? 1.0 : 0.0)) > kOriginTolerance) {
        return bad("world: sensor " + id +
                   " is the world, so it must sit at the origin");
      }
    }
  }
  return {};
}

}  // namespace

const SensorCalibration* find_sensor(const ArrayCalibration& array,
                                     std::string_view id) {
  for (const SensorCalibration& sensor : array.sensors) {
    if (sensor.id == id) return &sensor;
  }
  return nullptr;
}

core::Status validate_array_calibration(const ArrayCalibration& array) try {
  if (array.sensors.empty()) return bad("no sensors");
  for (std::size_t i = 0; i < array.sensors.size(); ++i) {
    const std::string& id = array.sensors[i].id;
    if (id.empty()) return bad("sensor " + std::to_string(i) + ": no id");
    if (!is_utf8(id)) {
      return bad("sensor " + std::to_string(i) + ": id is not UTF-8");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (array.sensors[j].id == id) return bad("sensor " + id + " is twice");
    }
    VKC_TRY(check_sensor(array.sensors[i]));
  }
  return check_world(array);
} catch (...) {
  return core::exception_status("array calibration",
                                core::Status::Code::InvalidArgument);
}

core::Result<ArrayCalibration> parse_array_calibration(
    std::string_view json_text) try {
  const json doc = json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) return bad("not JSON");
  if (!doc.is_object()) return bad("not a JSON object");
  ArrayCalibration array;
  const auto format = doc.find("format");
  if (format != doc.end()) {
    if (!format->is_string() || *format != kFormat) {
      return bad("\"format\" is not \"" + std::string(kFormat) + "\"");
    }
    VKC_ASSIGN(array, parse_v2(doc));
  } else {
    // Version 1 has neither key. A version without a format is a later file
    // that lost its format, and read as version 1 it would lose its lenses.
    if (doc.contains("version")) return bad("\"version\" without \"format\"");
    VKC_ASSIGN(array, parse_v1(doc));
  }
  std::sort(array.sensors.begin(), array.sensors.end(),
            [](const SensorCalibration& a, const SensorCalibration& b) {
              return a.id < b.id;
            });
  VKC_TRY(validate_array_calibration(array));
  return array;
} catch (...) {
  return core::exception_status("array calibration",
                                core::Status::Code::InvalidArgument);
}

core::Result<ArrayCalibration> read_array_calibration(
    const std::string& path) try {
  VKC_ASSIGN(const std::string text, core::read_file(path));
  auto array = parse_array_calibration(text);
  if (!array.ok()) return array.status().with_context(path);
  return array;
} catch (...) {
  return core::exception_status("array calibration",
                                core::Status::Code::InvalidArgument);
}

core::Result<std::string> format_array_calibration(
    const ArrayCalibration& array) try {
  VKC_TRY(validate_array_calibration(array));
  std::vector<const SensorCalibration*> sorted;
  for (const SensorCalibration& sensor : array.sensors) {
    sorted.push_back(&sensor);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto* a, const auto* b) { return a->id < b->id; });

  std::string out = "{\n  \"format\": " + text(std::string(kFormat)) +
                    ",\n  \"version\": " + text(kVersion) + ",\n";
  if (!array.world.sensor.empty()) {
    out += "  \"world\": {\"sensor\": " + text(array.world.sensor) + "},\n";
  }
  if (const auto& tag = array.world.apriltag) {
    out += "  \"world\": {\"apriltag\": {\"family\": " + text(tag->family) +
           ", \"id\": " + text(static_cast<std::uint64_t>(tag->id)) +
           ", \"size_m\": " + text(tag->size_m) + "}},\n";
  }
  out += "  \"device_calibration\": {";
  const std::string indent = "      ";
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const SensorCalibration& s = *sorted[i];
    std::vector<std::string> fields;
    if (s.color_to_world) {
      fields.push_back("\"pose\": " +
                       transform_text(rigid_inverse(*s.color_to_world)));
    }
    if (s.color) {
      fields.push_back("\"color\": " + camera_text(*s.color, indent));
    }
    if (s.depth) {
      fields.push_back("\"depth\": " + camera_text(*s.depth, indent));
    }
    if (s.depth_to_color) {
      fields.push_back("\"depth_to_color\": " +
                       transform_text(*s.depth_to_color));
    }
    out += (i == 0 ? "\n    " : ",\n    ") + text(s.id) + ": {";
    for (std::size_t f = 0; f < fields.size(); ++f) {
      out += (f == 0 ? "\n" : ",\n") + indent + fields[f];
    }
    out += fields.empty() ? "}" : "\n    }";
  }
  out += "\n  }\n}\n";
  return out;
} catch (...) {
  return core::exception_status("array calibration",
                                core::Status::Code::InvalidArgument);
}

core::Status write_array_calibration(const std::string& path,
                                     const ArrayCalibration& array,
                                     core::WriteMode mode) {
  VKC_ASSIGN(const std::string text, format_array_calibration(array));
  return core::write_file(path, text, mode);
}

}  // namespace volumetric_kit::recon::camera
