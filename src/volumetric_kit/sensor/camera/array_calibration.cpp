// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/recon/camera/array_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace volumetric_kit::recon::camera {

namespace {

using nlohmann::json;

core::Status bad(const std::string& what) {
  return core::Status::invalid_argument("array calibration: " + what);
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
    const SensorCalibration& sensor = array.sensors[i];
    if (sensor.id.empty()) {
      return bad("sensor " + std::to_string(i) + ": no id");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (array.sensors[j].id == sensor.id) {
        return bad("sensor " + sensor.id + " is twice");
      }
    }
    const core::Status rigid = check_rigid(sensor.color_to_world);
    if (!rigid.ok()) {
      return bad("sensor " + sensor.id + ": pose: " + rigid.message());
    }
  }
  return {};
}

core::Result<ArrayCalibration> parse_array_calibration(
    std::string_view json_text) {
  // Without exceptions: calib compiles this tier with -fno-exceptions.
  const json doc = json::parse(json_text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) return bad("not JSON");
  if (!doc.is_object()) return bad("not a JSON object");
  const auto section = doc.find("device_calibration");
  if (section == doc.end() || !section->is_object()) {
    return bad("no \"device_calibration\" object");
  }
  ArrayCalibration array;
  for (auto it = section->begin(); it != section->end(); ++it) {
    if (!it->is_object()) continue;  // e.g. "pose_calib_file_path"
    SensorCalibration sensor;
    sensor.id = it.key();
    const std::string where = "sensor " + sensor.id + ": ";
    const auto pose = it->find("pose");
    if (pose == it->end() || !pose->is_object()) {
      return bad(where + "no \"pose\" object");
    }
    VKC_ASSIGN(const Vec3d rvec, vector3(*pose, "rvec", where + "pose "));
    VKC_ASSIGN(const Vec3d tvec, vector3(*pose, "tvec", where + "pose "));
    sensor.color_to_world = rigid_inverse(matrix_from_rodrigues({rvec, tvec}));
    array.sensors.push_back(std::move(sensor));
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
  if (!array.ok()) {
    return core::Status::invalid_argument(path + ": " +
                                          array.status().message());
  }
  return array;
}

}  // namespace volumetric_kit::recon::camera
