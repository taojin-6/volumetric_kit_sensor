// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sensor array's calibration file: version 2 read as calib writes it,
// version 1 (the family's device_calibration layout) still read for its poses,
// the refusals, a later version refused, and a calibration written and read
// back with every model and number exact and every pose to round-off, through
// the core's write_file. Host-only.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "scratch_dir.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;
namespace fs = std::filesystem;

namespace {

using Code = vkc::Status::Code;

constexpr double kPi = 3.14159265358979323846;

// Two sensors: N at the origin, the world, with a colour and a depth camera;
// B turned 90 degrees about +Y with tvec (0, 0, 2), its colour camera alone,
// calibrated; and F, unposed, its factory lens alone. Unknown keys are
// ignored.
const char* kVersion2 = R"({
  "format": "volumetric_kit.array_calibration", "version": 2,
  "world": {"sensor": "N"},
  "note": "a key this reader does not know",
  "device_calibration": {
    "N": {
      "pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]},
      "color": {"source": "factory", "width": 3840, "height": 2160,
                "intrinsics": {"fx": 2239.48193, "fy": 2238.95605,
                               "cx": 1891.20874, "cy": 1036.8446},
                "distortion": {"model": "rational", "k1": 0.0753582269,
                               "k2": -0.105282806, "p1": -0.000271808181,
                               "p2": 0.000335, "k3": 0.0436422117,
                               "k4": 0, "k5": 0, "k6": 0}},
      "depth": {"source": "factory", "width": 640, "height": 576,
                "intrinsics": {"fx": 504.2, "fy": 504.3, "cx": 321.1,
                               "cy": 330.6}},
      "depth_to_color": {"rvec": [0.002, 0.001, 0],
                         "tvec": [-0.032, -0.002, 0.004]}},
    "B": {
      "pose": {"rvec": [0, 1.5707963267948966, 0], "tvec": [0, 0, 2]},
      "color": {"source": "calibrated", "width": 1280, "height": 720,
                "intrinsics": {"fx": 746.5, "fy": 746.3, "cx": 630.4,
                               "cy": 345.6}, "later": true}},
    "F": {
      "color": {"source": "factory", "width": 1280, "height": 720,
                "intrinsics": {"fx": 746.5, "fy": 746.3, "cx": 630.4,
                               "cy": 345.6}}}}
})";

// The family's version-1 layout (config_nunga.json), trimmed: an unrelated
// section, a non-object entry, lens fields this reader ignores, and two posed
// cameras: A at the origin, and B as above.
const char* kVersion1 = R"({
  "kinect_config": {"device_count": 2, "master_serial": "A"},
  "device_calibration": {
    "pose_calib_file_path": "/somewhere/all_extrinsics.npy",
    "B": {"intrinsics": {"fx": 746.5, "fy": 746.3, "cx": 630.4, "cy": 345.6},
          "pose": {"rvec": [0.0, 1.5707963267948966, 0.0],
                   "tvec": [0.0, 0.0, 2.0]}},
    "A": {"intrinsics": {"fx": 2239.48193, "fy": 2238.95605,
                         "cx": 1891.20874, "cy": 1036.8446,
                         "width": 3840, "height": 2160},
          "pose": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}}
  }
})";

bool near(const camera::Mat4d& a, const camera::Mat4d& b, double eps) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (std::fabs(a[c][r] - b[c][r]) > eps) return false;
    }
  }
  return true;
}

bool same(const camera::CameraModel& a, const camera::CameraModel& b) {
  const camera::RationalDistortion& p = a.distortion;
  const camera::RationalDistortion& q = b.distortion;
  return a.size.width == b.size.width && a.size.height == b.size.height &&
         a.intrinsics.fx == b.intrinsics.fx &&
         a.intrinsics.fy == b.intrinsics.fy &&
         a.intrinsics.cx == b.intrinsics.cx &&
         a.intrinsics.cy == b.intrinsics.cy && p.k1 == q.k1 && p.k2 == q.k2 &&
         p.p1 == q.p1 && p.p2 == q.p2 && p.k3 == q.k3 && p.k4 == q.k4 &&
         p.k5 == q.k5 && p.k6 == q.k6;
}

bool refused(const std::string& json, Code code = Code::InvalidArgument) {
  const auto r = camera::parse_array_calibration(json);
  if (r.ok()) return false;
  std::printf("  refused as expected: %s\n", r.status().message().c_str());
  return r.status().domain() == code;
}

// A version-2 document with one sensor, A, whose fields are `body`.
std::string one_sensor(const std::string& body, const std::string& world = "") {
  return R"({"format": "volumetric_kit.array_calibration", "version": 2, )" +
         world + R"("device_calibration": {"A": {)" + body + "}}}";
}

// The camera turned 90 degrees about +Y, tvec (0, 0, 2), sits at (2, 0, 0).
int check_b(const camera::SensorCalibration& b) {
  CHECK(b.id == "B" && b.color_to_world);
  CHECK(glm::length(camera::Vec3d((*b.color_to_world)[3]) -
                    camera::Vec3d(2.0, 0.0, 0.0)) < 1e-15);
  return 0;
}

int test_version2() {
  const auto r = camera::parse_array_calibration(kVersion2);
  CHECK(r.ok());
  const camera::ArrayCalibration& a = r.value();
  CHECK(a.world.sensor == "N");
  CHECK(a.sensors.size() == 3);
  CHECK(a.sensors[0].id == "B" && a.sensors[1].id == "F" &&
        a.sensors[2].id == "N");  // in id order
  if (check_b(a.sensors[0]) != 0) return 1;
  CHECK(a.sensors[0].color &&
        a.sensors[0].color->source == camera::IntrinsicsSource::Calibrated);
  CHECK(!a.sensors[0].depth && !a.sensors[0].depth_to_color);
  // A factory record of a lens: listed, unposed.
  const camera::SensorCalibration& f = a.sensors[1];
  CHECK(!f.color_to_world && f.color && !f.depth);
  CHECK(f.color->source == camera::IntrinsicsSource::Factory);

  const camera::SensorCalibration* n = camera::find_sensor(a, "N");
  CHECK(n != nullptr && n->color && n->depth && n->depth_to_color);
  CHECK(near(*n->color_to_world, camera::Mat4d(1.0), 0.0));
  CHECK(n->color->model.size.width == 3840 &&
        n->color->model.size.height == 2160);
  CHECK(n->color->model.intrinsics.fx == 2239.48193);
  CHECK(n->color->model.distortion.p1 == -0.000271808181);
  CHECK(n->color->model.distortion.k3 == 0.0436422117);
  // A camera without a distortion block has a pinhole lens.
  CHECK(n->depth->model.distortion.k1 == 0.0);
  // depth_to_color is OpenCV's form of x_color = R x_depth + t.
  CHECK(glm::length(camera::Vec3d((*n->depth_to_color)[3]) -
                    camera::Vec3d(-0.032, -0.002, 0.004)) < 1e-15);
  CHECK(camera::find_sensor(a, "missing") == nullptr);
  return 0;
}

int test_version1() {
  const auto r = camera::parse_array_calibration(kVersion1);
  CHECK(r.ok());
  const camera::ArrayCalibration& a = r.value();
  CHECK(a.world.sensor.empty());
  CHECK(a.sensors.size() == 2);
  const camera::SensorCalibration& sa = a.sensors[0];
  CHECK(sa.id == "A" && sa.color_to_world &&
        near(*sa.color_to_world, camera::Mat4d(1.0), 0.0));
  // Read for the poses alone: the lens fields are ignored.
  CHECK(!sa.color && !sa.depth && !sa.depth_to_color);
  if (check_b(a.sensors[1]) != 0) return 1;
  return 0;
}

int test_refusals() {
  const std::string pose = R"("pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]})";
  const std::string color =
      R"("color": {"source": "factory", "width": 64, "height": 48,
         "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})";
  const std::string world_a = R"("world": {"sensor": "A"}, )";
  CHECK(camera::parse_array_calibration(one_sensor(pose + ", " + color)).ok());
  CHECK(camera::parse_array_calibration(one_sensor(pose, world_a)).ok());

  CHECK(refused("{not json"));
  CHECK(refused("[1, 2]"));
  CHECK(refused(R"({"kinect_config": {}})"));
  CHECK(refused(R"({"device_calibration": {}})"));  // no sensors
  CHECK(refused(R"({"device_calibration": [1]})"));
  // A later version is refused as unsupported, an earlier one as not this
  // format: a file with no "format" is version 1.
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 3, "device_calibration": {"A": {}}})",
                Code::Unsupported));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 3, "sensors": {"A": {}}})",
                Code::Unsupported));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 1, "device_calibration": {"A": {}}})"));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "device_calibration": {"A": {}}})"));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": "2", "device_calibration": {"A": {}}})"));
  CHECK(refused(R"({"format": "another", "version": 2,
                    "device_calibration": {"A": {}}})"));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 2, "device_calibration": {"A": 1}})"));
  // Fields of the wrong shape.
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0], "tvec": [0, 0, 0]})")));
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0, 0],
                                       "tvec": [0, "0", 0]})")));
  CHECK(refused(one_sensor(R"("pose": [0, 0, 0])")));
  CHECK(refused(one_sensor(R"("color": {"width": 64, "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "guessed", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": -64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64.5,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48})")));
  // A model check_camera_model refuses.
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 0, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("depth": {"source": "factory", "width": 0,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  // A lens: OpenCV's rational model only, named, with its eight numbers.
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5},
      "distortion": {"model": "fisheye", "k1": 0, "k2": 0, "p1": 0, "p2": 0,
                     "k3": 0, "k4": 0, "k5": 0, "k6": 0}})"),
                Code::Unsupported));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5},
      "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0, "k4": 0,
                     "k5": 0, "k6": 0}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5},
      "distortion": {"model": "rational", "k1": 0, "k2": 0, "p1": 0, "p2": 0,
                     "k3": 0}})")));
  // depth_to_color beside a depth camera only, and rigid.
  CHECK(refused(one_sensor(
      R"("depth_to_color": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]})")));
  // The world must be a sensor of the array, posed at the origin.
  CHECK(refused(one_sensor(pose, R"("world": {"sensor": "Z"}, )")));
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0, 0], "tvec": [1, 0, 0]})",
                           world_a)));
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0.1, 0],
                                       "tvec": [0, 0, 0]})",
                           world_a)));
  CHECK(refused(one_sensor(color, world_a)));
  CHECK(refused(one_sensor(pose, R"("world": {"sensor": 1}, )")));
  CHECK(refused(one_sensor(pose, R"("world": "A", )")));
  // A world this reader does not know, as a later writer may name.
  CHECK(refused(one_sensor(pose, R"("world": {"apriltag": {"id": 0}}, )"),
                Code::Unsupported));
  // Version 1: every listed camera needs a pose.
  CHECK(refused(R"({"device_calibration": {"A": {"intrinsics":
      {"fx": 1, "fy": 1, "cx": 0, "cy": 0}}}})"));
  CHECK(refused(R"({"device_calibration": {"pose_calib_file_path": "x"}})"));

  // What validate refuses directly.
  camera::SensorCalibration a;
  a.id = "A";
  camera::ArrayCalibration array;
  CHECK(!camera::validate_array_calibration(array).ok());  // no sensors
  array.sensors = {a, a};
  CHECK(!camera::validate_array_calibration(array).ok());  // twice
  array.sensors = {a};
  CHECK(camera::validate_array_calibration(array).ok());  // a bare id
  array.sensors[0].color_to_world = camera::Mat4d(2.0);
  CHECK(!camera::validate_array_calibration(array).ok());  // not rigid
  array.sensors[0] = {};
  CHECK(!camera::validate_array_calibration(array).ok());  // no id
  array.sensors[0].id = "\xff";
  CHECK(!camera::validate_array_calibration(array).ok());  // not UTF-8
  array.sensors[0] = a;
  array.sensors[0].depth.emplace();  // a model with a zero size
  CHECK(!camera::validate_array_calibration(array).ok());
  return 0;
}

// Poses at Rodrigues' edge cases, models with numbers that have no short
// decimal form, an unposed sensor, and a world, written and read back.
camera::ArrayCalibration edge_cases() {
  const struct {
    const char* id;
    camera::Vec3d rvec;
    camera::Vec3d tvec;
  } poses[] = {{"0-world", camera::Vec3d(0.0), camera::Vec3d(0.0)},
               {"1-tiny",
                1e-9 * glm::normalize(camera::Vec3d(0.3, 0.5, 0.8)),
                {0.1, 0.2, 0.3}},
               {"2-general", {0.3, -0.8, 0.52}, {1.25, -2.5, 3.75e-3}},
               {"3-pi_x", {kPi, 0.0, 0.0}, {0.0, 0.0, 1.0}},
               {"4-pi_xy",
                kPi * glm::normalize(camera::Vec3d(1.0, 1.0, 0.0)),
                {0.5, 0.0, 0.0}},
               {"5-near_pi",
                (kPi - 1e-6) * glm::normalize(camera::Vec3d(0.2, 0.9, -0.4)),
                {0.0, 1.0, 0.0}}};
  camera::ArrayCalibration array;
  array.world.sensor = "0-world";
  for (const auto& p : poses) {
    camera::SensorCalibration s;
    s.id = p.id;
    s.color_to_world =
        camera::rigid_inverse(camera::matrix_from_rodrigues({p.rvec, p.tvec}));
    array.sensors.push_back(s);
  }
  camera::SensorCalibration& general = array.sensors[2];
  general.color = camera::CameraCalibration{
      {{3840, 2160},
       {2239.0 + 1.0 / 3.0, 2238.95605, 1891.20874, 1036.0 + 1.0 / 7.0},
       {0.0753582269, -0.105282806, -1e-300, 0.000335, 0.0436422117, 0.1 / 3.0,
        -2.0 / 3.0, 5e-17}},
      camera::IntrinsicsSource::Calibrated};
  general.depth = camera::CameraCalibration{
      {{640, 576},
       {504.2, 504.3, 321.1, 330.6},
       {29.29, 0.1 / 7.0, 1e-5, -2e-5, 0.0, 29.6, 12.0 / 7.0, 3.77}},
      camera::IntrinsicsSource::Factory};
  general.depth_to_color = camera::matrix_from_rodrigues(
      {{0.002, 0.001, 0.0}, {-0.0325, -0.0005, 0.0022}});
  // An unposed sensor: a factory record of its lens alone.
  camera::SensorCalibration unposed;
  unposed.id = "6-unposed";
  unposed.color =
      camera::CameraCalibration{{{1280, 720}, {746.5, 746.3, 630.4, 345.6}, {}},
                                camera::IntrinsicsSource::Factory};
  array.sensors.push_back(unposed);
  return array;
}

int check_same(const camera::ArrayCalibration& written,
               const camera::ArrayCalibration& read) {
  CHECK(read.world.sensor == written.world.sensor);
  CHECK(read.sensors.size() == written.sensors.size());
  for (std::size_t i = 0; i < written.sensors.size(); ++i) {
    const camera::SensorCalibration& w = written.sensors[i];
    const camera::SensorCalibration& r = read.sensors[i];
    CHECK(r.id == w.id);
    CHECK(r.color_to_world.has_value() == w.color_to_world.has_value());
    if (w.color_to_world) {
      CHECK(near(*r.color_to_world, *w.color_to_world, 1e-14));
    }
    CHECK(r.color.has_value() == w.color.has_value());
    if (w.color) {
      CHECK(same(r.color->model, w.color->model));  // exact
      CHECK(r.color->source == w.color->source);
    }
    CHECK(r.depth.has_value() == w.depth.has_value());
    if (w.depth) {
      CHECK(same(r.depth->model, w.depth->model));
      CHECK(r.depth->source == w.depth->source);
    }
    CHECK(r.depth_to_color.has_value() == w.depth_to_color.has_value());
    if (w.depth_to_color) {
      CHECK(near(*r.depth_to_color, *w.depth_to_color, 1e-15));
    }
  }
  return 0;
}

int test_round_trip() {
  const camera::ArrayCalibration written = edge_cases();
  const auto text = camera::format_array_calibration(written);
  CHECK(text.ok());
  const auto read = camera::parse_array_calibration(text.value());
  CHECK(read.ok());
  if (check_same(written, read.value()) != 0) return 1;

  // The document is version 2 under version 1's key, so a version-1 reader
  // finds each pose where it looks; the factory record has none.
  const nlohmann::json doc = nlohmann::json::parse(text.value());
  CHECK(doc.at("format") == "volumetric_kit.array_calibration");
  CHECK(doc.at("version") == 2);
  CHECK(doc.at("world").at("sensor") == "0-world");
  const nlohmann::json& sensors = doc.at("device_calibration");
  CHECK(sensors.size() == written.sensors.size());
  CHECK(sensors.at("2-general").at("pose").at("tvec").size() == 3);
  CHECK(!sensors.at("6-unposed").contains("pose"));
  // Every number is the double it was: shortest form, read back exactly.
  const nlohmann::json& lens = sensors.at("2-general").at("color");
  CHECK(lens.at("intrinsics").at("fx").get<double>() == 2239.0 + 1.0 / 3.0);
  CHECK(lens.at("distortion").at("p1").get<double>() == -1e-300);
  CHECK(lens.at("distortion").at("model") == "rational");
  CHECK(lens.at("source") == "calibrated");
  // The tiny turn's pose is kept, not rounded to zero.
  const nlohmann::json& rvec = sensors.at("1-tiny").at("pose").at("rvec");
  CHECK(std::fabs(rvec[2].get<double>() - 1e-9 * 0.8 / std::sqrt(0.98)) <
        1e-22);

  // A factory record: every sensor unposed, no world.
  camera::ArrayCalibration factory;
  factory.sensors.push_back(written.sensors[6]);
  camera::SensorCalibration depth_only;
  depth_only.id = "7-depth";
  depth_only.depth = written.sensors[2].depth;
  depth_only.depth_to_color = written.sensors[2].depth_to_color;
  factory.sensors.push_back(depth_only);
  const auto factory_text = camera::format_array_calibration(factory);
  CHECK(factory_text.ok());
  CHECK(factory_text.value().find("\"world\"") == std::string::npos);
  const auto factory_read =
      camera::parse_array_calibration(factory_text.value());
  CHECK(factory_read.ok());
  if (check_same(factory, factory_read.value()) != 0) return 1;

  // A calibration validate refuses is not formatted.
  camera::ArrayCalibration off_origin = written;
  (*off_origin.sensors[0].color_to_world)[3].x = 0.5;
  CHECK(camera::format_array_calibration(off_origin).status().domain() ==
        Code::InvalidArgument);
  return 0;
}

int test_files() {
  const vr_test::Scratch scratch("recon_array_calibration_");
  const std::string path = (scratch.path / "array_calibration.json").string();
  const camera::ArrayCalibration written = edge_cases();
  CHECK(camera::write_array_calibration(path, written,
                                        vkc::WriteMode::CreateNew));
  const auto read = camera::read_array_calibration(path);
  CHECK(read.ok());
  if (check_same(written, read.value()) != 0) return 1;
  // The file holds exactly what format gives.
  const auto text = camera::format_array_calibration(written);
  const auto bytes = vr_test::read_bytes(path);
  CHECK(std::string(bytes.begin(), bytes.end()) == text.value());

  // CreateNew refuses the file now there, and leaves it.
  camera::ArrayCalibration other;
  other.sensors.push_back(written.sensors[6]);
  CHECK(camera::write_array_calibration(path, other, vkc::WriteMode::CreateNew)
            .domain() == Code::InvalidArgument);
  CHECK(vr_test::read_bytes(path) == bytes);
  // A refused calibration touches nothing, even under Replace.
  camera::ArrayCalibration none;
  CHECK(camera::write_array_calibration(path, none, vkc::WriteMode::Replace)
            .domain() == Code::InvalidArgument);
  CHECK(vr_test::read_bytes(path) == bytes);
  const std::string untouched = (scratch.path / "refused.json").string();
  CHECK(!camera::write_array_calibration(untouched, none,
                                         vkc::WriteMode::CreateNew));
  CHECK(!fs::exists(untouched));
  // Replace swaps the whole file and leaves nothing else beside it.
  CHECK(camera::write_array_calibration(path, other, vkc::WriteMode::Replace));
  const auto replaced = camera::read_array_calibration(path);
  CHECK(replaced.ok() && replaced.value().sensors.size() == 1 &&
        replaced.value().sensors[0].id == "6-unposed");
  CHECK(std::distance(fs::directory_iterator(scratch.path),
                      fs::directory_iterator()) == 1);
  // A directory that is not there.
  const std::string orphan = (scratch.path / "missing" / "a.json").string();
  CHECK(camera::write_array_calibration(orphan, other, vkc::WriteMode::Replace)
            .domain() == Code::NotFound);

  // Reading: a missing file is the core's NotFound; a refused document's
  // message names the file, its domain kept.
  CHECK(camera::read_array_calibration(path + ".missing").status().domain() ==
        Code::NotFound);
  const std::string bad = (scratch.path / "bad.json").string();
  CHECK(vkc::write_file(bad, "{\"device_calibration\": {}}",
                        vkc::WriteMode::CreateNew));
  const auto refused_file = camera::read_array_calibration(bad);
  CHECK(!refused_file.ok() &&
        refused_file.status().domain() == Code::InvalidArgument &&
        refused_file.status().message().rfind(bad + ": ", 0) == 0);
  const std::string later = (scratch.path / "later.json").string();
  CHECK(vkc::write_file(later,
                        R"({"format": "volumetric_kit.array_calibration",
                            "version": 3, "device_calibration": {}})",
                        vkc::WriteMode::CreateNew));
  const auto later_file = camera::read_array_calibration(later);
  CHECK(!later_file.ok() && later_file.status().domain() == Code::Unsupported &&
        later_file.status().message().rfind(later + ": ", 0) == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_version2() != 0) return 1;
  if (test_version1() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_round_trip() != 0) return 1;
  if (test_files() != 0) return 1;
  std::printf("array calibration tests passed\n");
  return 0;
}
