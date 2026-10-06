// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The sensor array's calibration file: version 2 read as calib writes it,
// version 1 (the family's device_calibration layout) still read, the
// refusals, and a written file read back with every model exact and every
// pose to round-off. Host-only.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/camera/array_calibration.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

namespace {

constexpr double kPi = 3.14159265358979323846;

// Two sensors: N at the origin, the world, with a colour and a depth camera;
// B turned 90 degrees about +Y with tvec (0, 0, 2), its colour camera alone,
// calibrated. An unknown key is ignored.
const char* kVersion2 = R"({
  "format": "volumetric_kit.array_calibration", "version": 2,
  "world": {"sensor": "N"},
  "note": "a key this reader does not know",
  "sensors": {
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
                               "cy": 345.6}}}}
})";

// The family's version-1 layout (config_nunga.json), trimmed: an unrelated
// section, a non-object entry, a camera whose intrinsics record their image
// and one whose do not.
const char* kVersion1 = R"({
  "kinect_config": {"device_count": 2, "master_serial": "A"},
  "device_calibration": {
    "pose_calib_file_path": "/somewhere/all_extrinsics.npy",
    "A": {
      "intrinsics": {"fx": 2239.48193, "fy": 2238.95605, "cx": 1891.20874,
                     "cy": 1036.8446, "width": 3840, "height": 2160},
      "distortion": {"k1": 0.0753582269, "k2": -0.105282806,
                     "k3": 0.0436422117, "k4": 0, "k5": 0, "k6": 0,
                     "p1": -0.000271808181, "p2": 0.000335},
      "optimal_intrinsics": {"fx": 1892.1, "fy": 1861.0, "cx": 1933.5,
                             "cy": 1111.1},
      "pose": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}
    },
    "B": {"intrinsics": {"fx": 746.5, "fy": 746.3, "cx": 630.4, "cy": 345.6},
          "pose": {"rvec": [0.0, 1.5707963267948966, 0.0],
                   "tvec": [0.0, 0.0, 2.0]}}
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

bool refused(const std::string& json,
             vkc::Status::Code code = vkc::Status::Code::InvalidArgument) {
  const auto r = camera::parse_array_calibration(json);
  if (r.ok()) return false;
  std::printf("  refused as expected: %s\n", r.status().message().c_str());
  return r.status().domain() == code;
}

// A version-2 document with one sensor whose fields are `body`.
std::string one_sensor(const std::string& body, const std::string& world = "") {
  return R"({"format": "volumetric_kit.array_calibration", "version": 2, )" +
         world + R"("sensors": {"A": {)" + body + "}}}";
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
  CHECK(a.world.kind == camera::WorldFrame::Kind::Sensor &&
        a.world.sensor == "N");
  CHECK(a.sensors.size() == 2);
  CHECK(a.sensors[0].id == "B" && a.sensors[1].id == "N");  // in id order
  if (check_b(a.sensors[0]) != 0) return 1;
  CHECK(a.sensors[0].color &&
        a.sensors[0].color->source == camera::IntrinsicsSource::Calibrated);
  CHECK(!a.sensors[0].depth && !a.sensors[0].depth_to_color);

  const camera::SensorCalibration* n = camera::find_sensor(a, "N");
  CHECK(n != nullptr && n->color && n->depth && n->depth_to_color);
  CHECK(near(*n->color_to_world, camera::Mat4d(1.0), 0.0));
  CHECK(n->color->source == camera::IntrinsicsSource::Factory);
  CHECK(n->color->model.size.width == 3840);
  CHECK(n->color->model.intrinsics.fx == 2239.48193);
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
  CHECK(a.world.kind == camera::WorldFrame::Kind::Unspecified);
  CHECK(a.sensors.size() == 2);
  const camera::SensorCalibration& sa = a.sensors[0];
  CHECK(sa.id == "A" && sa.color_to_world &&
        near(*sa.color_to_world, camera::Mat4d(1.0), 0.0));
  // Intrinsics that record their image become the colour camera.
  CHECK(sa.color && sa.color->source == camera::IntrinsicsSource::Factory);
  CHECK(sa.color->model.size.width == 3840 &&
        sa.color->model.size.height == 2160);
  CHECK(sa.color->model.distortion.p2 == 0.000335);
  // B's do not, so they are dropped; its pose stays.
  if (check_b(a.sensors[1]) != 0) return 1;
  CHECK(!a.sensors[1].color);
  return 0;
}

int test_refusals() {
  using Code = vkc::Status::Code;
  const std::string pose = R"("pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]})";
  const std::string color =
      R"("color": {"source": "factory", "width": 64, "height": 48,
         "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})";
  CHECK(camera::parse_array_calibration(one_sensor(pose + ", " + color)).ok());

  CHECK(refused("{not json"));
  CHECK(refused("[1, 2]"));
  CHECK(refused(R"({"kinect_config": {}})"));
  CHECK(refused(R"({"format": "another", "version": 2, "sensors": {}})"));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 3, "sensors": {}})",
                Code::Unsupported));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 1, "sensors": {}})"));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 2})"));
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 2, "sensors": {}})"));  // no sensors
  CHECK(refused(R"({"format": "volumetric_kit.array_calibration",
                    "version": 2, "sensors": {"A": 1}})"));
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0], "tvec": [0, 0, 0]})")));
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0, 0],
                                       "tvec": [0, "0", 0]})")));
  CHECK(refused(one_sensor(R"("color": {"width": 64, "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "guessed", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": -64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 0, "fy": 50, "cx": 31.5, "cy": 23.5}})")));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5},
      "distortion": {"model": "fisheye", "k1": 0, "k2": 0, "p1": 0, "p2": 0,
                     "k3": 0, "k4": 0, "k5": 0, "k6": 0}})"),
                Code::Unsupported));
  CHECK(refused(one_sensor(R"("color": {"source": "factory", "width": 64,
      "height": 48,
      "intrinsics": {"fx": 50, "fy": 50, "cx": 31.5, "cy": 23.5},
      "distortion": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0}})")));
  CHECK(refused(one_sensor(
      R"("depth_to_color": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]})")));
  // The world must be a sensor of the array, posed at the origin.
  CHECK(refused(one_sensor(pose, R"("world": {"sensor": "Z"}, )")));
  CHECK(refused(one_sensor(R"("pose": {"rvec": [0, 0, 0], "tvec": [1, 0, 0]})",
                           R"("world": {"sensor": "A"}, )")));
  CHECK(refused(one_sensor(color, R"("world": {"sensor": "A"}, )")));
  CHECK(refused(
      one_sensor(pose, R"("world": {"apriltag": {"family": "tag36h11", "id": 0,
                                      "size_m": 0}}, )")));
  CHECK(refused(one_sensor(pose, R"("world": {}, )")));
  // Version 1: every camera needs a pose.
  CHECK(refused(R"({"device_calibration": {"A": {"intrinsics":
      {"fx": 1, "fy": 1, "cx": 0, "cy": 0}}}})"));
  CHECK(refused(R"({"device_calibration": {"pose_calib_file_path": "x"}})"));
  return 0;
}

int test_round_trip() {
  // Poses at Rodrigues' edge cases: none, a tiny turn, a general one, and
  // turns of pi and just under it, where sin(theta) vanishes.
  const struct {
    const char* id;
    camera::Vec3d rvec;
    camera::Vec3d tvec;
  } poses[] = {{"0-world", camera::Vec3d(0.0), camera::Vec3d(0.0)},
               {"1-tiny",
                1e-7 * glm::normalize(camera::Vec3d(0.3, 0.5, 0.8)),
                {0.1, 0.2, 0.3}},
               {"2-general", {0.3, -0.8, 0.52}, {1.25, -2.5, 3.75e-3}},
               {"3-pi_x", {kPi, 0.0, 0.0}, {0.0, 0.0, 1.0}},
               {"4-pi_xy",
                kPi * glm::normalize(camera::Vec3d(1.0, 1.0, 0.0)),
                {0.5, 0.0, 0.0}},
               {"5-near_pi",
                3.1415 * glm::normalize(camera::Vec3d(0.2, 0.9, -0.4)),
                {0.0, 1.0, 0.0}}};
  camera::ArrayCalibration written;
  written.world.kind = camera::WorldFrame::Kind::AprilTag;
  written.world.tag = {"tag36h11", 0, 0.22};
  for (const auto& p : poses) {
    camera::SensorCalibration s;
    s.id = p.id;
    s.color_to_world =
        camera::rigid_inverse(camera::matrix_from_rodrigues({p.rvec, p.tvec}));
    written.sensors.push_back(s);
  }
  // Models whose numbers have no short decimal form.
  camera::SensorCalibration& general = written.sensors[2];
  general.color = camera::CameraCalibration{
      {{3840, 2160},
       {2239.0 + 1.0 / 3.0, 2238.95605, 1891.20874, 1036.0 + 1.0 / 7.0},
       {0.0753582269, -0.105282806, -1e-300, 0.000335, 0.0436422117, 0.1 / 3.0,
        -2.0 / 3.0, 5e-17}},
      camera::IntrinsicsSource::Calibrated};
  general.depth =
      camera::CameraCalibration{{{640, 576}, {504.2, 504.3, 321.1, 330.6}, {}},
                                camera::IntrinsicsSource::Factory};
  general.depth_to_color = camera::matrix_from_rodrigues(
      {{0.002, 0.001, 0.0}, {-0.032, -0.002, 0.004}});
  // An unposed sensor: a factory record of its lens alone.
  camera::SensorCalibration unposed;
  unposed.id = "6-unposed";
  unposed.color =
      camera::CameraCalibration{{{1280, 720}, {746.5, 746.3, 630.4, 345.6}, {}},
                                camera::IntrinsicsSource::Factory};
  written.sensors.push_back(unposed);

  const std::string path =
      std::string(VR_TEST_SCRATCH_DIR) + "/array_calibration.json";
  CHECK(camera::write_array_calibration(path, written).ok());
  std::FILE* leftover = std::fopen((path + ".tmp").c_str(), "rb");
  CHECK(leftover == nullptr);  // renamed into place
  const auto read = camera::read_array_calibration(path);
  CHECK(read.ok());
  const camera::ArrayCalibration& a = read.value();
  CHECK(a.world.kind == camera::WorldFrame::Kind::AprilTag);
  CHECK(a.world.tag.family == "tag36h11" && a.world.tag.id == 0 &&
        a.world.tag.size_m == 0.22);
  CHECK(a.sensors.size() == written.sensors.size());
  for (std::size_t i = 0; i < written.sensors.size(); ++i) {
    const camera::SensorCalibration& w = written.sensors[i];
    const camera::SensorCalibration& r = a.sensors[i];
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
    if (w.depth) CHECK(same(r.depth->model, w.depth->model));
    CHECK(r.depth_to_color.has_value() == w.depth_to_color.has_value());
    if (w.depth_to_color) {
      CHECK(near(*r.depth_to_color, *w.depth_to_color, 1e-15));
    }
  }

  // A world sensor posed at the origin survives the trip.
  camera::ArrayCalibration sensor_world = written;
  sensor_world.world = {};
  sensor_world.world.kind = camera::WorldFrame::Kind::Sensor;
  sensor_world.world.sensor = "0-world";
  const auto text = camera::format_array_calibration(sensor_world);
  CHECK(text.ok());
  const auto parsed = camera::parse_array_calibration(text.value());
  CHECK(parsed.ok() && parsed.value().world.sensor == "0-world");
  return 0;
}

int test_writer_refusals() {
  const std::string path =
      std::string(VR_TEST_SCRATCH_DIR) + "/array_calibration_refused.json";
  std::remove(path.c_str());
  camera::ArrayCalibration array;
  CHECK(!camera::write_array_calibration(path, array).ok());  // no sensors
  camera::SensorCalibration a;
  a.id = "A";
  array.sensors = {a, a};
  CHECK(!camera::write_array_calibration(path, array).ok());  // twice
  a.color_to_world = camera::Mat4d(2.0);
  array.sensors = {a};
  CHECK(!camera::write_array_calibration(path, array).ok());  // not rigid
  a.color_to_world.reset();
  a.id = "\xff";
  array.sensors = {a};
  CHECK(!camera::write_array_calibration(path, array).ok());  // not UTF-8
  std::FILE* untouched = std::fopen(path.c_str(), "rb");
  CHECK(untouched == nullptr);  // nothing was written

  const auto missing = camera::read_array_calibration(path + ".missing");
  CHECK(!missing.ok() &&
        missing.status().domain() == vkc::Status::Code::IoError);
  // A refused document's message names the file.
  const std::string bad =
      std::string(VR_TEST_SCRATCH_DIR) + "/array_calibration_bad.json";
  std::FILE* f = std::fopen(bad.c_str(), "wb");
  CHECK(f != nullptr);
  std::fputs("{\"device_calibration\": {}}", f);
  std::fclose(f);
  const auto refused_file = camera::read_array_calibration(bad);
  CHECK(!refused_file.ok() &&
        refused_file.status().message().rfind(bad, 0) == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_version2() != 0) return 1;
  if (test_version1() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_round_trip() != 0) return 1;
  if (test_writer_refusals() != 0) return 1;
  std::printf("array calibration tests passed\n");
  return 0;
}
