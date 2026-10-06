// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/camera/array_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace volumetric_kit::recon::camera {

namespace {

using nlohmann::json;

constexpr std::string_view kFormat = "volumetric_kit.array_calibration";
constexpr std::uint64_t kVersion = 2;
constexpr std::string_view kRational = "rational";
// How far a world sensor's pose may be from the origin: a writer puts it there
// exactly, so this only absorbs the Rodrigues round trip.
constexpr double kOriginTolerance = 1e-9;

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("array calibration: " + what);
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

core::Result<RodriguesTransform> rodrigues(const json& block,
                                           const std::string& where) {
  RodriguesTransform t;
  VKC_ASSIGN(t.rvec, vector3(block, "rvec", where));
  VKC_ASSIGN(t.tvec, vector3(block, "tvec", where));
  return t;
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
  if (model != block.end()) {
    if (!model->is_string()) return bad(where + "\"model\" is not a string");
    if (model->get<std::string>() != kRational) {
      return core::Status::unsupported(
          "array calibration: " + where + "distortion model \"" +
          model->get<std::string>() + "\"; this reader knows \"rational\"");
    }
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

core::Result<WorldFrame> world_v2(const json& block) {
  const std::string where = "world: ";
  WorldFrame world;
  const auto sensor = block.find("sensor");
  VKC_ASSIGN(const json* tag, object_at(block, "apriltag", where));
  if ((sensor != block.end()) == (tag != nullptr)) {
    return bad(where + "needs one of \"sensor\" and \"apriltag\"");
  }
  if (sensor != block.end()) {
    if (!sensor->is_string()) return bad(where + "\"sensor\" is not a string");
    world.kind = WorldFrame::Kind::Sensor;
    world.sensor = sensor->get<std::string>();
    return world;
  }
  world.kind = WorldFrame::Kind::AprilTag;
  const auto family = tag->find("family");
  if (family == tag->end() || !family->is_string()) {
    return bad(where + "apriltag: missing string \"family\"");
  }
  world.tag.family = family->get<std::string>();
  VKC_ASSIGN(world.tag.id, whole_number(*tag, "id", where + "apriltag: "));
  VKC_ASSIGN(world.tag.size_m, number(*tag, "size_m", where + "apriltag: "));
  return world;
}

core::Result<ArrayCalibration> parse_v2(const json& doc) {
  const auto version = doc.find("version");
  if (version == doc.end() || !version->is_number_unsigned()) {
    return bad("missing whole number \"version\"");
  }
  if (version->get<std::uint64_t>() > kVersion) {
    return core::Status::unsupported(
        "array calibration: version " +
        std::to_string(version->get<std::uint64_t>()) +
        "; this reader knows version 2");
  }
  if (version->get<std::uint64_t>() != kVersion) {
    return bad("version " + std::to_string(version->get<std::uint64_t>()) +
               " is not a versioned format; version 1 is the "
               "\"device_calibration\" layout");
  }
  ArrayCalibration array;
  VKC_ASSIGN(const json* world, object_at(doc, "world", ""));
  if (world != nullptr) {
    VKC_ASSIGN(array.world, world_v2(*world));
  }
  VKC_ASSIGN(const json* sensors, object_at(doc, "sensors", ""));
  if (sensors == nullptr) return bad("no \"sensors\" object");
  for (auto it = sensors->begin(); it != sensors->end(); ++it) {
    SensorCalibration sensor;
    sensor.id = it.key();
    const std::string where = "sensor " + sensor.id + ": ";
    if (!it->is_object()) return bad(where + "not an object");
    VKC_ASSIGN(const json* pose, object_at(*it, "pose", where));
    if (pose != nullptr) {
      VKC_ASSIGN(const RodriguesTransform extrinsic,
                 rodrigues(*pose, where + "pose "));
      sensor.color_to_world = rigid_inverse(matrix_from_rodrigues(extrinsic));
    }
    for (const auto& [key, field] : {std::pair{"color", &sensor.color},
                                     std::pair{"depth", &sensor.depth}}) {
      VKC_ASSIGN(const json* camera, object_at(*it, key, where));
      if (camera != nullptr) {
        VKC_ASSIGN(*field, camera_v2(*camera, where + key + " "));
      }
    }
    VKC_ASSIGN(const json* depth_to_color,
               object_at(*it, "depth_to_color", where));
    if (depth_to_color != nullptr) {
      VKC_ASSIGN(const RodriguesTransform t,
                 rodrigues(*depth_to_color, where + "depth_to_color "));
      sensor.depth_to_color = matrix_from_rodrigues(t);
    }
    array.sensors.push_back(std::move(sensor));
  }
  return array;
}

core::Result<ArrayCalibration> parse_v1(const json& section) {
  ArrayCalibration array;
  for (auto it = section.begin(); it != section.end(); ++it) {
    if (!it->is_object()) continue;  // e.g. "pose_calib_file_path"
    SensorCalibration sensor;
    sensor.id = it.key();
    const std::string where = "camera " + sensor.id + ": ";
    VKC_ASSIGN(const json* pose, object_at(*it, "pose", where));
    if (pose == nullptr) return bad(where + "no \"pose\" object");
    VKC_ASSIGN(const RodriguesTransform extrinsic,
               rodrigues(*pose, where + "pose "));
    sensor.color_to_world = rigid_inverse(matrix_from_rodrigues(extrinsic));
    // The intrinsics form a camera only with the image they are in pixels of.
    VKC_ASSIGN(const json* intrinsics, object_at(*it, "intrinsics", where));
    if (intrinsics != nullptr && intrinsics->contains("width") &&
        intrinsics->contains("height")) {
      const std::string w = where + "intrinsics ";
      CameraCalibration color;
      VKC_ASSIGN(color.model.size.width, whole_number(*intrinsics, "width", w));
      VKC_ASSIGN(color.model.size.height,
                 whole_number(*intrinsics, "height", w));
      VKC_ASSIGN(color.model.intrinsics, pinhole(*intrinsics, w));
      VKC_ASSIGN(const json* distortion, object_at(*it, "distortion", where));
      if (distortion != nullptr) {
        VKC_ASSIGN(color.model.distortion,
                   rational(*distortion, where + "distortion "));
      }
      sensor.color = color;
    }
    array.sensors.push_back(std::move(sensor));
  }
  return array;
}

// --- writing -----------------------------------------------------------------

// A number as JSON text: the shortest form that reads back to the same value.
std::string text(double v) { return json(v).dump(); }
std::string text(std::uint64_t v) { return json(v).dump(); }
std::string text(const std::string& s) { return json(s).dump(); }

std::string vector_text(const Vec3d& v) {
  return "[" + text(v.x) + ", " + text(v.y) + ", " + text(v.z) + "]";
}

std::string rodrigues_text(const Mat4d& transform) {
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
         "  \"distortion\": {\"model\": \"rational\", " +
         "\"k1\": " + text(d.k1) + ", \"k2\": " + text(d.k2) +
         ", \"p1\": " + text(d.p1) + ", \"p2\": " + text(d.p2) +
         ", \"k3\": " + text(d.k3) + ", \"k4\": " + text(d.k4) +
         ", \"k5\": " + text(d.k5) + ", \"k6\": " + text(d.k6) + "}}";
}

std::string world_text(const WorldFrame& world) {
  if (world.kind == WorldFrame::Kind::Sensor) {
    return "{\"sensor\": " + text(world.sensor) + "}";
  }
  return "{\"apriltag\": {\"family\": " + text(world.tag.family) +
         ", \"id\": " + text(std::uint64_t{world.tag.id}) +
         ", \"size_m\": " + text(world.tag.size_m) + "}}";
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

core::Status check_world(const ArrayCalibration& array) {
  const WorldFrame& world = array.world;
  if (world.kind == WorldFrame::Kind::Sensor) {
    const SensorCalibration* sensor = find_sensor(array, world.sensor);
    if (sensor == nullptr) {
      return bad("world: sensor " + world.sensor + " is not in the array");
    }
    if (!sensor->color_to_world) {
      return bad("world: sensor " + world.sensor + " is not posed");
    }
    const Mat4d& m = *sensor->color_to_world;
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) {
        if (std::fabs(m[c][r] - (c == r ? 1.0 : 0.0)) > kOriginTolerance) {
          return bad("world: sensor " + world.sensor +
                     " is the world, so it must sit at the origin");
        }
      }
    }
  } else if (world.kind == WorldFrame::Kind::AprilTag) {
    if (world.tag.family.empty() || !is_utf8(world.tag.family)) {
      return bad("world: the AprilTag needs a UTF-8 family");
    }
    if (!(std::isfinite(world.tag.size_m) && world.tag.size_m > 0.0)) {
      return bad("world: the AprilTag's size must be finite and positive");
    }
  }
  return {};
}

// TODO: one read-whole-file helper in the core's base tier; the Orbbec sync
// config reader (orbbec_sync_config.cpp) repeats this loop.
core::Result<std::string> read_text(const std::string& path) {
  // stdio rather than a stream: a stream reports a failed read -- a directory,
  // say -- as the end of the file.
  std::FILE* in = std::fopen(path.c_str(), "rb");
  if (in == nullptr) {
    return core::Status::io_error("array calibration: cannot open " + path);
  }
  std::string text;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) text.append(buf, n);
  const bool failed = std::ferror(in) != 0;
  std::fclose(in);
  if (failed) {
    return core::Status::io_error("array calibration: cannot read " + path);
  }
  return text;
}

}  // namespace

const SensorCalibration* find_sensor(const ArrayCalibration& array,
                                     std::string_view id) {
  for (const SensorCalibration& sensor : array.sensors) {
    if (sensor.id == id) return &sensor;
  }
  return nullptr;
}

core::Status validate_array_calibration(const ArrayCalibration& array) {
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
}

core::Result<ArrayCalibration> parse_array_calibration(
    std::string_view json_text) {
  json doc;
  try {
    doc = json::parse(json_text);
  } catch (const std::exception& e) {  // json::parse_error names the byte
    return bad(std::string("not JSON: ") + e.what());
  }
  if (!doc.is_object()) return bad("not a JSON object");
  ArrayCalibration array;
  const auto format = doc.find("format");
  if (format != doc.end()) {
    if (!format->is_string() || *format != kFormat) {
      return bad("\"format\" is not \"" + std::string(kFormat) + "\"");
    }
    VKC_ASSIGN(array, parse_v2(doc));
  } else {
    VKC_ASSIGN(const json* section, object_at(doc, "device_calibration", ""));
    if (section == nullptr) {
      return bad("neither a \"format\" nor a \"device_calibration\" object");
    }
    VKC_ASSIGN(array, parse_v1(*section));
  }
  std::sort(array.sensors.begin(), array.sensors.end(),
            [](const SensorCalibration& a, const SensorCalibration& b) {
              return a.id < b.id;
            });
  VKC_TRY(validate_array_calibration(array));
  return array;
}

core::Result<ArrayCalibration> read_array_calibration(const std::string& path) {
  VKC_ASSIGN(const std::string text, read_text(path));
  auto array = parse_array_calibration(text);
  if (!array.ok()) {  // parsing refuses with these two codes only
    const std::string message = path + ": " + array.status().message();
    return array.status().domain() == core::Status::Code::Unsupported
               ? core::Status::unsupported(message)
               : core::Status::invalid_argument(message);
  }
  return array;
}

core::Result<std::string> format_array_calibration(
    const ArrayCalibration& array) {
  VKC_TRY(validate_array_calibration(array));
  std::vector<const SensorCalibration*> sorted;
  for (const SensorCalibration& sensor : array.sensors) {
    sorted.push_back(&sensor);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto* a, const auto* b) { return a->id < b->id; });

  std::string out = "{\n  \"format\": " + text(std::string(kFormat)) +
                    ",\n  \"version\": " + text(kVersion) + ",\n";
  if (array.world.kind != WorldFrame::Kind::Unspecified) {
    out += "  \"world\": " + world_text(array.world) + ",\n";
  }
  out += "  \"sensors\": {";
  const std::string indent = "      ";
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const SensorCalibration& s = *sorted[i];
    std::vector<std::string> fields;
    if (s.color_to_world) {
      fields.push_back("\"pose\": " +
                       rodrigues_text(rigid_inverse(*s.color_to_world)));
    }
    if (s.color)
      fields.push_back("\"color\": " + camera_text(*s.color, indent));
    if (s.depth)
      fields.push_back("\"depth\": " + camera_text(*s.depth, indent));
    if (s.depth_to_color) {
      fields.push_back("\"depth_to_color\": " +
                       rodrigues_text(*s.depth_to_color));
    }
    out += (i == 0 ? "\n" : ",\n") + std::string("    ") + text(s.id) + ": {";
    for (std::size_t f = 0; f < fields.size(); ++f) {
      out += (f == 0 ? "\n" : ",\n") + indent + fields[f];
    }
    out += fields.empty() ? "}" : "\n    }";
  }
  out += "\n  }\n}\n";
  return out;
}

core::Status write_array_calibration(const std::string& path,
                                     const ArrayCalibration& array) {
  VKC_ASSIGN(const std::string text, format_array_calibration(array));
  // Written beside the file -- a symbolic link's target, so the link stays --
  // synced, and renamed over it, so a failed write or a crash leaves the
  // calibration that was there. The temporary is this process's own, and
  // takes the old file's permissions.
  std::error_code error;
  const std::string target = std::filesystem::weakly_canonical(path, error);
  if (error) {
    return core::Status::io_error("array calibration: cannot resolve " + path);
  }
  const std::string temporary =
      target + "." + std::to_string(::getpid()) + ".tmp";
  std::FILE* file = std::fopen(temporary.c_str(), "wb");
  if (file == nullptr) {
    return core::Status::io_error("array calibration: cannot create " +
                                  temporary);
  }
  bool written =
      std::fwrite(text.data(), 1, text.size(), file) == text.size() &&
      std::fflush(file) == 0;
  struct stat old{};
  if (written && ::stat(target.c_str(), &old) == 0) {
    written = ::fchmod(::fileno(file), old.st_mode & 07777) == 0;
  }
  written = written && ::fsync(::fileno(file)) == 0;
  written = std::fclose(file) == 0 && written;
  if (!written) {
    std::remove(temporary.c_str());
    return core::Status::io_error("array calibration: cannot write " +
                                  temporary);
  }
  if (std::rename(temporary.c_str(), target.c_str()) != 0) {
    std::remove(temporary.c_str());
    return core::Status::io_error("array calibration: cannot replace " + path);
  }
  return {};
}

}  // namespace volumetric_kit::recon::camera
