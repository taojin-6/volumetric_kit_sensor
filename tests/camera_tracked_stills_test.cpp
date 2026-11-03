// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A tracked sensor's stills file: a golden document read to known values (the
// pose's direction, the session's byte order, either case of hex, a pinhole's
// missing lens), the writer's numbers in that direction and its camera block
// the array file's, stills formatted and read back with their models exact and
// their poses to round-off, written and read through the core's write_file,
// each refusal named, keys this reader does not know ignored, and a later
// version refused. Host-only.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>

#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/tracked_stills.hpp"

namespace vkc = volumetric_kit::core;
namespace cam = volumetric_kit::sensor::camera;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

using Code = vkc::Status::Code;

const double kHalfPi = glm::half_pi<double>();
constexpr std::uint64_t kLastTime = std::numeric_limits<std::uint64_t>::max();

// Two stills. The first is a pinhole (no "distortion") posed a quarter turn
// about +z, then (1, 2, 3): its camera's point (1, 0, 0) sits at (1, 3, 3) in
// the tracking world. The second has a lens, a "calibrated" source and the
// last time. The session is the bytes 0x00 to 0x0f. Keys this reader does not
// know, at every level, are ignored.
const char* kGolden = R"({
  "format": "volumetric_kit.tracked_stills", "version": 1,
  "note": "a key this reader does not know",
  "sensor": "ipad-pro-m5",
  "session": "000102030405060708090a0b0c0d0e0f", "world_epoch": 3,
  "stills": [
    {"image": "still_000.png", "timestamp_ns": 1761957000000000000,
     "later": true,
     "camera": {"source": "factory", "width": 1920, "height": 1440,
                "intrinsics": {"fx": 1596.2, "fy": 1596.0, "cx": 959.1,
                               "cy": 719.6, "later": true},
                "later": true},
     "cam_to_tracking": {"rvec": [0, 0, 1.5707963267948966],
                         "tvec": [1, 2, 3], "later": true}},
    {"image": "still_001.png", "timestamp_ns": 18446744073709551615,
     "camera": {"source": "calibrated", "width": 960, "height": 720,
                "intrinsics": {"fx": 798.1, "fy": 798.0, "cx": 479.55,
                               "cy": 359.8},
                "distortion": {"model": "rational", "k1": 0.01, "k2": -0.02,
                               "p1": 0.001, "p2": -0.002, "k3": 0.003,
                               "k4": 0.004, "k5": 0.005, "k6": 0.006}},
     "cam_to_tracking": {"rvec": [0, 0, 0], "tvec": [0, 0, 0]}}
  ]
})";

cam::Vec3d apply(const cam::Mat4d& m, const cam::Vec3d& p) {
  return cam::Vec3d(m * glm::dvec4(p, 1.0));
}

bool near(const cam::Vec3d& a, const cam::Vec3d& b, double eps) {
  return glm::length(a - b) <= eps;
}

bool near(const cam::Mat4d& a, const cam::Mat4d& b, double eps) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (std::fabs(a[c][r] - b[c][r]) > eps) return false;
    }
  }
  return true;
}

bool same(const cam::CameraModel& a, const cam::CameraModel& b) {
  const cam::RationalDistortion& p = a.distortion;
  const cam::RationalDistortion& q = b.distortion;
  return a.size.width == b.size.width && a.size.height == b.size.height &&
         a.intrinsics.fx == b.intrinsics.fx &&
         a.intrinsics.fy == b.intrinsics.fy &&
         a.intrinsics.cx == b.intrinsics.cx &&
         a.intrinsics.cy == b.intrinsics.cy && p.k1 == q.k1 && p.k2 == q.k2 &&
         p.p1 == q.p1 && p.p2 == q.p2 && p.k3 == q.k3 && p.k4 == q.k4 &&
         p.k5 == q.k5 && p.k6 == q.k6;
}

// The golden stills, as a struct.
cam::TrackedStills golden() {
  cam::TrackedStills s;
  s.sensor = "ipad-pro-m5";
  for (std::size_t i = 0; i < s.session.size(); ++i) {
    s.session[i] = static_cast<std::uint8_t>(i);
  }
  s.world_epoch = 3;
  cam::TrackedStill first;
  first.image = "still_000.png";
  first.timestamp_ns = 1761957000000000000u;
  first.model = {{1920, 1440}, {1596.2, 1596.0, 959.1, 719.6}, {}};
  first.cam_to_tracking = cam::matrix_from_rodrigues(
      {cam::Vec3d(0.0, 0.0, kHalfPi), cam::Vec3d(1.0, 2.0, 3.0)});
  cam::TrackedStill second;
  second.image = "still_001.png";
  second.timestamp_ns = kLastTime;
  second.model = {{960, 720},
                  {798.1, 798.0, 479.55, 359.8},
                  {0.01, -0.02, 0.001, -0.002, 0.003, 0.004, 0.005, 0.006}};
  s.stills = {first, second};
  return s;
}

// Every field the same, the poses to `eps`.
int check_same(const cam::TrackedStills& a, const cam::TrackedStills& b,
               double eps) {
  CHECK(a.sensor == b.sensor);
  CHECK(a.session == b.session);
  CHECK(a.world_epoch == b.world_epoch);
  CHECK(a.stills.size() == b.stills.size());
  for (std::size_t i = 0; i < a.stills.size(); ++i) {
    const cam::TrackedStill& p = a.stills[i];
    const cam::TrackedStill& q = b.stills[i];
    CHECK(p.image == q.image);
    CHECK(p.timestamp_ns == q.timestamp_ns);
    CHECK(same(p.model, q.model));
    CHECK(near(p.cam_to_tracking, q.cam_to_tracking, eps));
  }
  return 0;
}

// Whether `status` is in `code`'s domain, names the file and says `reason`:
// a refusal for another reason does not count.
bool says(const vkc::Status& status, Code code, const std::string& reason) {
  if (status.ok()) return false;
  std::printf("  refused as expected: %s\n", status.message().c_str());
  return status.domain() == code &&
         status.message().rfind("tracked stills: ", 0) == 0 &&
         status.message().find(reason) != std::string::npos;
}

bool refused(const json& doc, Code code, const std::string& reason) {
  return says(cam::parse_tracked_stills(doc.dump()).status(), code, reason);
}

bool invalid(const cam::TrackedStills& s, const std::string& reason) {
  return says(cam::validate_tracked_stills(s), Code::InvalidArgument, reason) &&
         says(cam::format_tracked_stills(s).status(), Code::InvalidArgument,
              reason);
}

int test_golden() {
  const auto read = cam::parse_tracked_stills(kGolden);
  if (!read) std::fprintf(stderr, "%s\n", read.status().message().c_str());
  CHECK(read.ok());
  const cam::TrackedStills& s = read.value();
  CHECK(s.sensor == "ipad-pro-m5");
  for (std::size_t i = 0; i < s.session.size(); ++i) {
    CHECK(std::size_t{s.session[i]} == i);
  }
  CHECK(s.world_epoch == 3);
  CHECK(s.stills.size() == 2);
  CHECK(s.stills[0].image == "still_000.png");
  CHECK(s.stills[0].timestamp_ns == 1761957000000000000u);
  CHECK(s.stills[1].timestamp_ns == kLastTime);
  // A pinhole's lens may be absent; the source is read and not kept.
  CHECK(s.stills[0].model.distortion.k1 == 0.0 &&
        s.stills[0].model.distortion.k6 == 0.0);
  CHECK(s.stills[1].model.distortion.k6 == 0.006);
  // The direction: camera to tracking, not tracking to camera.
  CHECK(near(apply(s.stills[0].cam_to_tracking, cam::Vec3d(1.0, 0.0, 0.0)),
             cam::Vec3d(1.0, 3.0, 3.0), 1e-12));
  CHECK(near(apply(s.stills[0].cam_to_tracking, cam::Vec3d(0.0)),
             cam::Vec3d(1.0, 2.0, 3.0), 1e-12));
  if (check_same(s, golden(), 1e-15) != 0) return 1;

  // Upper-case hex reads the same.
  json upper = json::parse(kGolden);
  upper["session"] = "000102030405060708090A0B0C0D0E0F";
  const auto upper_read = cam::parse_tracked_stills(upper.dump());
  CHECK(upper_read.ok() && upper_read.value().session == s.session);

  // The writer writes the same direction, the session lowercase, byte 0
  // first, and the reader's numbers: what an independent parser finds.
  const auto text = cam::format_tracked_stills(golden());
  CHECK(text.ok());
  std::printf("%s", text.value().c_str());
  const json doc = json::parse(text.value());
  CHECK(doc.at("format") == "volumetric_kit.tracked_stills");
  CHECK(doc.at("version") == 1);
  CHECK(doc.at("sensor") == "ipad-pro-m5");
  CHECK(doc.at("session") == "000102030405060708090a0b0c0d0e0f");
  CHECK(doc.at("world_epoch") == 3);
  const json& stills = doc.at("stills");
  CHECK(stills.size() == 2);
  CHECK(stills[0].at("image") == "still_000.png");
  CHECK(stills[0].at("timestamp_ns").get<std::uint64_t>() ==
        1761957000000000000u);
  CHECK(stills[1].at("timestamp_ns").get<std::uint64_t>() == kLastTime);
  const json& pose = stills[0].at("cam_to_tracking");
  const auto vec = [&pose](const char* key) {
    const json& v = pose.at(key);
    return cam::Vec3d(v[0].get<double>(), v[1].get<double>(),
                      v[2].get<double>());
  };
  CHECK(near(vec("rvec"), cam::Vec3d(0.0, 0.0, kHalfPi), 1e-12));
  CHECK(near(vec("tvec"), cam::Vec3d(1.0, 2.0, 3.0), 1e-12));
  // The camera block is the array file's, as its writer writes a factory
  // camera: the lens written, zero for a pinhole, every source "factory".
  CHECK(stills[0].at("camera") ==
        json::parse(R"({"source": "factory", "width": 1920, "height": 1440,
          "intrinsics": {"fx": 1596.2, "fy": 1596.0, "cx": 959.1, "cy": 719.6},
          "distortion": {"model": "rational", "k1": 0.0, "k2": 0.0,
                         "p1": 0.0, "p2": 0.0, "k3": 0.0, "k4": 0.0,
                         "k5": 0.0, "k6": 0.0}})"));
  CHECK(stills[1].at("camera").at("source") == "factory");
  cam::ArrayCalibration array;
  for (std::size_t i = 0; i < 2; ++i) {
    cam::SensorCalibration sensor;
    sensor.id = "S" + std::to_string(i);
    sensor.color = cam::CameraCalibration{golden().stills[i].model,
                                          cam::IntrinsicsSource::Factory};
    array.sensors.push_back(std::move(sensor));
  }
  const auto array_text = cam::format_array_calibration(array);
  CHECK(array_text.ok());
  const json array_doc = json::parse(array_text.value());
  for (std::size_t i = 0; i < 2; ++i) {
    CHECK(array_doc.at("device_calibration")
              .at("S" + std::to_string(i))
              .at("color") == stills[i].at("camera"));
  }
  return 0;
}

int test_round_trip() {
  // Numbers that need all 17 digits, names JSON must escape, every byte
  // value's digits, the last epoch, times 0 and the last, and a turn past
  // pi/2 about a skew axis.
  cam::TrackedStills s;
  s.sensor = "iPad \"Pro\" \xc3\xa9\\ M5";
  s.session = {0xff, 0x00, 0x7f, 0x80, 0x01, 0x10, 0xab, 0xcd,
               0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde};
  s.world_epoch = 65535;
  cam::TrackedStill a;
  a.image = "still \"\xc3\xa9\" \\ 0.png";
  a.timestamp_ns = 0;
  a.model = {{4294967295u, 1},
             {1.0 / 3.0, 2.0 / 7.0, -0.1 - 0.2, 1e300},
             {0.1 + 0.2, -1e-17, 5e-324, -0.0, 1.0 / 9.0, 7.0, -3.5, 1e-9}};
  a.cam_to_tracking = cam::matrix_from_rodrigues(
      {cam::Vec3d(0.31, -1.7, 2.2), cam::Vec3d(-1.0 / 3.0, 2.0 / 7.0, 1e-9)});
  cam::TrackedStill b = golden().stills[0];
  b.timestamp_ns = kLastTime;
  s.stills = {a, b};
  const auto text = cam::format_tracked_stills(s);
  CHECK(text.ok());
  const auto read = cam::parse_tracked_stills(text.value());
  if (!read) std::fprintf(stderr, "%s\n", read.status().message().c_str());
  CHECK(read.ok());
  if (check_same(read.value(), s, 1e-12) != 0) return 1;
  // A second trip writes the same document.
  const auto again = cam::format_tracked_stills(read.value());
  CHECK(again.ok());
  const json first = json::parse(text.value());
  const json second = json::parse(again.value());
  CHECK(first.at("session") == second.at("session"));
  CHECK(first.at("stills")[0].at("camera") ==
        second.at("stills")[0].at("camera"));

  // The golden stills the same way.
  const auto golden_text = cam::format_tracked_stills(golden());
  CHECK(golden_text.ok());
  const auto golden_read = cam::parse_tracked_stills(golden_text.value());
  CHECK(golden_read.ok());
  if (check_same(golden_read.value(), golden(), 1e-12) != 0) return 1;
  return 0;
}

int test_refusals() {
  const json base = json::parse(kGolden);
  CHECK(cam::parse_tracked_stills(base.dump()).ok());
  const auto with = [&base](const char* key, json value) {
    json doc = base;
    doc[key] = std::move(value);
    return doc;
  };
  const auto without = [&base](const char* key) {
    json doc = base;
    doc.erase(key);
    return doc;
  };
  // The second still's `key` set to `value`, or erased for null.
  const auto with_still = [&base](const char* key, json value) {
    json doc = base;
    if (value.is_null()) {
      doc["stills"][1].erase(key);
    } else {
      doc["stills"][1][key] = std::move(value);
    }
    return doc;
  };
  const auto with_camera = [&base](const char* key, json value) {
    json doc = base;
    if (value.is_null()) {
      doc["stills"][1]["camera"].erase(key);
    } else {
      doc["stills"][1]["camera"][key] = std::move(value);
    }
    return doc;
  };
  const Code bad = Code::InvalidArgument;

  // Not stills at all.
  CHECK(says(cam::parse_tracked_stills("{").status(), bad, "not JSON"));
  CHECK(
      says(cam::parse_tracked_stills("[]").status(), bad, "not a JSON object"));
  CHECK(refused(with("format", "volumetric_kit.tracked_registration"), bad,
                "\"format\" is not \"volumetric_kit.tracked_stills\""));
  CHECK(refused(without("format"), bad, "\"format\" is not"));
  CHECK(refused(without("version"), bad, "missing whole number \"version\""));
  CHECK(refused(with("version", 0), bad, "version 0; versions start at 1"));
  CHECK(refused(with("version", 2), Code::Unsupported,
                "version 2; this reader knows version 1"));

  // The sensor, the session and its epoch.
  CHECK(refused(with("sensor", ""), bad, "\"sensor\" is empty"));
  CHECK(refused(without("sensor"), bad, "missing string \"sensor\""));
  CHECK(refused(with("sensor", 7), bad, "missing string \"sensor\""));
  CHECK(refused(with("session", std::string(32, '0')), bad, "all zero"));
  CHECK(
      refused(with("session", std::string(31, '1')), bad, "is 31 bytes long"));
  CHECK(refused(with("session", "g00102030405060708090a0b0c0d0e0f"), bad,
                "not hex"));
  CHECK(refused(without("session"), bad, "missing string \"session\""));
  CHECK(refused(with("world_epoch", 65536), bad, "past 65535"));
  CHECK(refused(without("world_epoch"), bad, "\"world_epoch\""));

  // The stills.
  CHECK(refused(without("stills"), bad, "no \"stills\" array"));
  CHECK(refused(with("stills", json::object()), bad, "no \"stills\" array"));
  CHECK(refused(with("stills", json::array()), bad, "no stills"));
  json not_object = base;
  not_object["stills"][1] = 7;
  CHECK(refused(not_object, bad, "still 1: not an object"));

  // A still's name.
  CHECK(refused(with_still("image", nullptr), bad,
                "still 1: missing string \"image\""));
  CHECK(refused(with_still("image", 7), bad, "still 1: missing string"));
  CHECK(refused(with_still("image", ""), bad, "still 1: \"image\" is empty"));
  CHECK(refused(with_still("image", "."), bad, "a directory"));
  CHECK(refused(with_still("image", ".."), bad, "a directory"));
  CHECK(refused(with_still("image", "takes/still_001.png"), bad, "has a '/'"));
  CHECK(refused(with_still("image", std::string("still\0.png", 10)), bad,
                "has a NUL"));
  CHECK(refused(with_still("image", "still_000.png"), bad,
                "still 1: \"image\" \"still_000.png\" is still 0's too"));
  json dotted = with_still("image", "..png");
  CHECK(cam::parse_tracked_stills(dotted.dump()).ok());

  // A still's time.
  CHECK(refused(with_still("timestamp_ns", nullptr), bad,
                "still 1: \"timestamp_ns\" is not a 64-bit whole number"));
  CHECK(refused(with_still("timestamp_ns", -1), bad, "\"timestamp_ns\""));
  CHECK(refused(with_still("timestamp_ns", 1.5), bad, "\"timestamp_ns\""));
  CHECK(refused(with_still("timestamp_ns", 1761957000000000000u), bad,
                "is not after still 0's"));
  CHECK(refused(with_still("timestamp_ns", 1761956999999999999u), bad,
                "is not after still 0's"));
  json next = with_still("timestamp_ns", 1761957000000000001u);
  CHECK(cam::parse_tracked_stills(next.dump()).ok());

  // A still's camera: the array file's refusals, then its model's check.
  CHECK(refused(with_still("camera", nullptr), bad,
                "still 1: no \"camera\" object"));
  CHECK(refused(with_camera("source", "phone"), bad,
                "still 1: camera \"source\" is \"phone\""));
  CHECK(
      refused(with_camera("width", nullptr), bad, "still 1: camera \"width\""));
  CHECK(refused(with_camera("intrinsics", nullptr), bad,
                "still 1: camera no \"intrinsics\" object"));
  json no_fx = base;
  no_fx["stills"][1]["camera"]["intrinsics"]["fx"] = 0;
  CHECK(refused(no_fx, bad, "still 1: camera: "));
  json brown = base;
  brown["stills"][1]["camera"]["distortion"]["model"] = "brown";
  CHECK(refused(brown, Code::Unsupported,
                "still 1: camera distortion model \"brown\""));
  json no_k6 = base;
  no_k6["stills"][1]["camera"]["distortion"].erase("k6");
  CHECK(refused(no_k6, bad, "missing number \"k6\""));

  // A still's pose.
  CHECK(refused(with_still("cam_to_tracking", nullptr), bad,
                "still 1: no \"cam_to_tracking\" object"));
  CHECK(
      refused(with_still("cam_to_tracking",
                         json::parse(R"({"rvec": [0, 0], "tvec": [0, 0, 0]})")),
              bad, "still 1: cam_to_tracking \"rvec\" is not 3 numbers"));

  // What JSON text cannot carry, refused by validate and by format alike.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(cam::validate_tracked_stills(golden()).ok());
  cam::TrackedStills s = golden();
  s.sensor = "\xff";
  CHECK(invalid(s, "\"sensor\" is not UTF-8"));
  s = golden();
  s.session = {};
  CHECK(invalid(s, "all zero"));
  s = golden();
  s.stills.clear();
  CHECK(invalid(s, "no stills"));
  s = golden();
  s.stills[1].image = "\xfe.png";
  CHECK(invalid(s, "still 1: \"image\" is not UTF-8"));
  s = golden();
  s.stills[1].image = std::string("a\0b", 3);
  CHECK(invalid(s, "still 1: \"image\" has a NUL"));
  s = golden();
  s.stills[1].timestamp_ns = s.stills[0].timestamp_ns;
  CHECK(invalid(s, "still 1: \"timestamp_ns\""));
  s = golden();
  s.stills[1].model.intrinsics.fx = nan;
  CHECK(invalid(s, "still 1: camera: "));
  s = golden();
  s.stills[1].model.distortion.k3 = inf;
  CHECK(invalid(s, "still 1: camera: "));
  s = golden();
  s.stills[1].model.size.height = 0;
  CHECK(invalid(s, "still 1: camera: "));
  s = golden();
  s.stills[1].cam_to_tracking[0] *= 1.01;
  CHECK(invalid(s, "still 1: cam_to_tracking: transform's rotation is not"));
  s = golden();
  s.stills[1].cam_to_tracking[3][2] = inf;
  CHECK(invalid(s, "still 1: cam_to_tracking: transform is not finite"));
  CHECK(invalid(cam::TrackedStills{}, "\"sensor\" is empty"));
  return 0;
}

int test_files() {
  const vkc::test::Scratch scratch("sensor_tracked_stills_");
  const std::string path = (scratch.path / "stills.json").string();
  const cam::TrackedStills written = golden();
  CHECK(cam::write_tracked_stills(path, written, vkc::WriteMode::CreateNew));
  const auto read = cam::read_tracked_stills(path);
  CHECK(read.ok());
  if (check_same(read.value(), written, 1e-12) != 0) return 1;
  // The file holds exactly what format gives.
  const auto text = cam::format_tracked_stills(written);
  const auto bytes = vkc::test::read_bytes(path);
  CHECK(std::string(bytes.begin(), bytes.end()) == text.value());

  // CreateNew refuses the file now there, and leaves it.
  cam::TrackedStills other = golden();
  other.world_epoch = 4;
  const vkc::Status exists =
      cam::write_tracked_stills(path, other, vkc::WriteMode::CreateNew);
  CHECK(exists.domain() == Code::InvalidArgument &&
        exists.message().find(path) != std::string::npos);
  CHECK(vkc::test::read_bytes(path) == bytes);
  // Refused stills touch nothing, even under Replace.
  CHECK(cam::write_tracked_stills(path, cam::TrackedStills{},
                                  vkc::WriteMode::Replace)
            .domain() == Code::InvalidArgument);
  CHECK(vkc::test::read_bytes(path) == bytes);
  const std::string untouched = (scratch.path / "refused.json").string();
  CHECK(says(cam::write_tracked_stills(untouched, cam::TrackedStills{},
                                       vkc::WriteMode::CreateNew),
             Code::InvalidArgument, "\"sensor\" is empty"));
  CHECK(!fs::exists(untouched));
  // Replace replaces it, and leaves nothing else beside it.
  CHECK(cam::write_tracked_stills(path, other, vkc::WriteMode::Replace));
  const auto replaced = cam::read_tracked_stills(path);
  CHECK(replaced.ok() && replaced.value().world_epoch == 4);
  CHECK(std::distance(fs::directory_iterator(scratch.path),
                      fs::directory_iterator()) == 1);

  // Reading: a missing file is the core's NotFound; a refused document's
  // message names the file, its domain kept.
  CHECK(cam::read_tracked_stills(path + ".missing").status().domain() ==
        Code::NotFound);
  const std::string later = (scratch.path / "later.json").string();
  json later_doc = json::parse(kGolden);
  later_doc["version"] = 2;
  CHECK(vkc::write_file(later, later_doc.dump(), vkc::WriteMode::CreateNew));
  const auto later_file = cam::read_tracked_stills(later);
  CHECK(!later_file.ok() && later_file.status().domain() == Code::Unsupported &&
        later_file.status().message().rfind(later + ": tracked stills: ", 0) ==
            0);
  return 0;
}

}  // namespace

int main() {
  if (test_golden() != 0) return 1;
  if (test_round_trip() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_files() != 0) return 1;
  std::printf("tracked stills tests passed\n");
  return 0;
}
