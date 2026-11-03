// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// A tracked sensor's registration file: a golden document read to known
// values (the transform's direction, the session's byte order, either case of
// hex), the writer's numbers in that direction, a registration formatted and
// read back with its fit exact and its transform to round-off, written and
// read through the core's write_file, each refusal named, keys this reader
// does not know ignored, and a tracked frame placed as a tracked source does.
// Host-only.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <nlohmann/json.hpp>

#include "volumetric_kit/core/base/file_io.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/testing/scratch_dir.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/camera/tracked_registration.hpp"

namespace vkc = volumetric_kit::core;
namespace cam = volumetric_kit::sensor::camera;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

using Code = vkc::Status::Code;

const double kHalfPi = glm::half_pi<double>();

// A quarter turn about +z, then (1, 2, 3): the tracking point (1, 0, 0) sits
// at (1, 3, 3) in the array's world. The session is the bytes 0x00 to 0x0f.
// Keys this reader does not know, at every level, are ignored.
const char* kGolden = R"({
  "format": "volumetric_kit.tracked_registration", "version": 1,
  "note": "a key this reader does not know",
  "sensor": "ipad-pro-m5",
  "session": "000102030405060708090a0b0c0d0e0f", "world_epoch": 3,
  "world": {"apriltag": {"family": "tag36h11", "id": 0, "size_m": 0.15,
                         "later": true}, "later": true},
  "tracking_to_world": {"rvec": [0, 0, 1.5707963267948966],
                        "tvec": [1, 2, 3], "later": true},
  "fit": {"rms_px": 0.61, "views": 6, "edge_m": 0.1471, "later": true}
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

// The golden registration, as a struct.
cam::TrackedRegistration golden() {
  cam::TrackedRegistration r;
  r.sensor = "ipad-pro-m5";
  for (std::size_t i = 0; i < r.session.size(); ++i) {
    r.session[i] = static_cast<std::uint8_t>(i);
  }
  r.world_epoch = 3;
  r.world.apriltag = cam::AprilTagWorld{"tag36h11", 0, 0.15};
  r.tracking_to_world = cam::matrix_from_rodrigues(
      {cam::Vec3d(0.0, 0.0, kHalfPi), cam::Vec3d(1.0, 2.0, 3.0)});
  r.rms_px = 0.61;
  r.views = 6;
  r.edge_m = 0.1471;
  return r;
}

// Every field the same, the transform to `eps`.
int check_same(const cam::TrackedRegistration& a,
               const cam::TrackedRegistration& b, double eps) {
  CHECK(a.sensor == b.sensor);
  CHECK(a.session == b.session);
  CHECK(a.world_epoch == b.world_epoch);
  CHECK(a.world.sensor == b.world.sensor);
  CHECK(a.world.apriltag.has_value() == b.world.apriltag.has_value());
  if (a.world.apriltag) {
    CHECK(a.world.apriltag->family == b.world.apriltag->family);
    CHECK(a.world.apriltag->id == b.world.apriltag->id);
    CHECK(a.world.apriltag->size_m == b.world.apriltag->size_m);
  }
  CHECK(near(a.tracking_to_world, b.tracking_to_world, eps));
  CHECK(a.rms_px == b.rms_px);
  CHECK(a.views == b.views);
  CHECK(a.edge_m == b.edge_m);
  return 0;
}

// Whether `status` is in `code`'s domain, names the file and says `reason`:
// a refusal for another reason does not count.
bool says(const vkc::Status& status, Code code, const std::string& reason) {
  if (status.ok()) return false;
  std::printf("  refused as expected: %s\n", status.message().c_str());
  return status.domain() == code &&
         status.message().rfind("tracked registration: ", 0) == 0 &&
         status.message().find(reason) != std::string::npos;
}

bool refused(const json& doc, Code code, const std::string& reason) {
  return says(cam::parse_tracked_registration(doc.dump()).status(), code,
              reason);
}

bool invalid(const cam::TrackedRegistration& r, const std::string& reason) {
  return says(cam::validate_tracked_registration(r), Code::InvalidArgument,
              reason) &&
         says(cam::format_tracked_registration(r).status(),
              Code::InvalidArgument, reason);
}

// The sample in the PR: a tracked camera's frame placed in the array's
// world, the session's id and epoch, and the frame's pose, as its depth record
// carries them.
vkc::Status place(const std::string& path,
                  const std::array<std::uint8_t, 16>& session,
                  std::uint16_t epoch, const cam::Mat4d& cam_to_tracking) {
  VKC_ASSIGN(const cam::TrackedRegistration reg,
             cam::read_tracked_registration(path));
  if (reg.session != session || reg.world_epoch != epoch) {
    return vkc::Status::invalid_argument(
        path + ": registered for another session or world epoch");
  }
  const cam::Mat4d color_to_world = reg.tracking_to_world * cam_to_tracking;
  std::printf("%s at (%.3f, %.3f, %.3f) m; fit %.2f px over %u stills\n",
              reg.sensor.c_str(), color_to_world[3].x, color_to_world[3].y,
              color_to_world[3].z, reg.rms_px, reg.views);
  return {};
}

int test_golden() {
  const auto read = cam::parse_tracked_registration(kGolden);
  if (!read) std::fprintf(stderr, "%s\n", read.status().message().c_str());
  CHECK(read.ok());
  const cam::TrackedRegistration& r = read.value();
  CHECK(r.sensor == "ipad-pro-m5");
  for (std::size_t i = 0; i < r.session.size(); ++i) {
    CHECK(std::size_t{r.session[i]} == i);
  }
  CHECK(r.world_epoch == 3);
  CHECK(r.world.sensor.empty() && r.world.apriltag.has_value());
  CHECK(r.world.apriltag->family == "tag36h11");
  CHECK(r.world.apriltag->id == 0 && r.world.apriltag->size_m == 0.15);
  CHECK(r.rms_px == 0.61 && r.views == 6 && r.edge_m == 0.1471);
  // The direction: tracking to world, not world to tracking.
  CHECK(near(apply(r.tracking_to_world, cam::Vec3d(1.0, 0.0, 0.0)),
             cam::Vec3d(1.0, 3.0, 3.0), 1e-12));
  CHECK(near(apply(r.tracking_to_world, cam::Vec3d(0.0)),
             cam::Vec3d(1.0, 2.0, 3.0), 1e-12));
  if (check_same(r, golden(), 1e-15) != 0) return 1;

  // Upper-case hex reads the same.
  json upper = json::parse(kGolden);
  upper["session"] = "000102030405060708090A0B0C0D0E0F";
  const auto upper_read = cam::parse_tracked_registration(upper.dump());
  CHECK(upper_read.ok() && upper_read.value().session == r.session);

  // The writer writes the same direction, the session lowercase, byte 0
  // first, and the reader's numbers: what an independent parser finds.
  const auto text = cam::format_tracked_registration(golden());
  CHECK(text.ok());
  std::printf("%s", text.value().c_str());
  const json doc = json::parse(text.value());
  CHECK(doc.at("format") == "volumetric_kit.tracked_registration");
  CHECK(doc.at("version") == 1);
  CHECK(doc.at("sensor") == "ipad-pro-m5");
  CHECK(doc.at("session") == "000102030405060708090a0b0c0d0e0f");
  CHECK(doc.at("world_epoch") == 3);
  CHECK(doc.at("world") ==
        json::parse(R"({"apriltag": {"family": "tag36h11", "id": 0,
                                     "size_m": 0.15}})"));
  const json& t = doc.at("tracking_to_world");
  const auto vec = [&t](const char* key) {
    const json& v = t.at(key);
    return cam::Vec3d(v[0].get<double>(), v[1].get<double>(),
                      v[2].get<double>());
  };
  const cam::Vec3d rvec = vec("rvec");
  const cam::Vec3d tvec = vec("tvec");
  CHECK(near(rvec, cam::Vec3d(0.0, 0.0, kHalfPi), 1e-12));
  CHECK(near(tvec, cam::Vec3d(1.0, 2.0, 3.0), 1e-12));
  CHECK(doc.at("fit") ==
        json::parse(R"({"rms_px": 0.61, "views": 6, "edge_m": 0.1471})"));
  return 0;
}

int test_round_trip() {
  // Numbers that need all 17 digits, a sensor id JSON must escape, every
  // byte value's digits, the last epoch, a world at a sensor, and a turn past
  // pi/2 about a skew axis.
  cam::TrackedRegistration r;
  r.sensor = "iPad \"Pro\" \xc3\xa9\\ M5";
  r.session = {0xff, 0x00, 0x7f, 0x80, 0x01, 0x10, 0xab, 0xcd,
               0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde};
  r.world_epoch = 65535;
  r.world.sensor = "CL2A141000N";
  r.tracking_to_world = cam::matrix_from_rodrigues(
      {cam::Vec3d(0.31, -1.7, 2.2), cam::Vec3d(-1.0 / 3.0, 2.0 / 7.0, 1e-9)});
  r.rms_px = 0.1 + 0.2;
  r.views = 4294967295u;
  r.edge_m = std::nextafter(0.15, 1.0);
  const auto text = cam::format_tracked_registration(r);
  CHECK(text.ok());
  const auto read = cam::parse_tracked_registration(text.value());
  if (!read) std::fprintf(stderr, "%s\n", read.status().message().c_str());
  CHECK(read.ok());
  if (check_same(read.value(), r, 1e-12) != 0) return 1;
  // A second trip writes the same numbers.
  const auto again = cam::format_tracked_registration(read.value());
  CHECK(again.ok());
  const json a = json::parse(text.value());
  const json b = json::parse(again.value());
  CHECK(a.at("fit") == b.at("fit") && a.at("session") == b.at("session"));

  // The golden registration, at a tag, the same way.
  const auto golden_text = cam::format_tracked_registration(golden());
  CHECK(golden_text.ok());
  const auto golden_read = cam::parse_tracked_registration(golden_text.value());
  CHECK(golden_read.ok());
  if (check_same(golden_read.value(), golden(), 1e-12) != 0) return 1;
  return 0;
}

int test_refusals() {
  const json base = json::parse(kGolden);
  CHECK(cam::parse_tracked_registration(base.dump()).ok());
  const auto with = [&base](const char* key, json value) {
    json doc = base;
    doc[key] = std::move(value);
    return doc;
  };
  const auto with_fit = [&base](const char* key, json value) {
    json doc = base;
    doc["fit"][key] = std::move(value);
    return doc;
  };
  const auto without = [&base](const char* key) {
    json doc = base;
    doc.erase(key);
    return doc;
  };
  const Code bad = Code::InvalidArgument;

  // Not a registration at all.
  CHECK(says(cam::parse_tracked_registration("{").status(), bad, "not JSON"));
  CHECK(says(cam::parse_tracked_registration("[]").status(), bad,
             "not a JSON object"));
  CHECK(refused(with("format", "volumetric_kit.array_calibration"), bad,
                "\"format\" is not"));
  CHECK(refused(without("format"), bad, "\"format\" is not"));
  CHECK(refused(without("version"), bad, "missing whole number \"version\""));
  CHECK(refused(with("version", 0), bad, "version 0"));
  CHECK(refused(with("version", 2), Code::Unsupported, "version 2"));

  // The sensor.
  CHECK(refused(with("sensor", ""), bad, "\"sensor\" is empty"));
  CHECK(refused(without("sensor"), bad, "missing string \"sensor\""));
  CHECK(refused(with("sensor", 7), bad, "missing string \"sensor\""));

  // The session and its epoch.
  CHECK(refused(with("session", std::string(32, '0')), bad, "all zero"));
  CHECK(
      refused(with("session", std::string(31, '1')), bad, "is 31 bytes long"));
  CHECK(
      refused(with("session", std::string(33, '1')), bad, "is 33 bytes long"));
  // A digit that is not hex, in a byte's low digit and in its high one.
  CHECK(refused(with("session", "000102030405060708090a0b0c0d0e0g"), bad,
                "not hex"));
  CHECK(refused(with("session", "0x0102030405060708090a0b0c0d0e0f"), bad,
                "not hex"));
  CHECK(refused(with("session", "g00102030405060708090a0b0c0d0e0f"), bad,
                "not hex"));
  CHECK(refused(without("session"), bad, "missing string \"session\""));
  CHECK(refused(with("world_epoch", 65536), bad, "past 65535"));
  CHECK(refused(with("world_epoch", -1), bad, "\"world_epoch\""));
  CHECK(refused(without("world_epoch"), bad, "\"world_epoch\""));
  json last_epoch = with("world_epoch", 65535);
  CHECK(cam::parse_tracked_registration(last_epoch.dump()).ok());

  // The world.
  CHECK(refused(without("world"), bad, "no \"world\" object"));
  CHECK(refused(with("world", json::parse(R"({"sensor": "N",
                  "apriltag": {"family": "tag36h11", "id": 0,
                               "size_m": 0.15}})")),
                bad, "names both"));
  CHECK(refused(with("world", json::parse(R"({"marker": 3})")),
                Code::Unsupported, "this reader knows"));
  CHECK(refused(with("world", json::parse(R"({"sensor": ""})")), bad,
                "world: \"sensor\" is empty"));
  CHECK(refused(with("world", json::parse(R"({"apriltag": {"family":
                  "tag36h11", "id": 0, "size_m": 0}})")),
                bad, "\"size_m\" is not a finite length above 0"));
  CHECK(refused(with("world", json::parse(R"({"apriltag": {"family":
                  "tag36h11", "id": -1, "size_m": 0.15}})")),
                bad, "\"id\" is negative"));
  CHECK(refused(with("world", json::parse(R"({"apriltag": {"family": "",
                  "id": 0, "size_m": 0.15}})")),
                bad, "\"family\" is empty"));

  // The transform.
  CHECK(refused(without("tracking_to_world"), bad,
                "no \"tracking_to_world\" object"));
  CHECK(refused(with("tracking_to_world",
                     json::parse(R"({"rvec": [0, 0], "tvec": [0, 0, 0]})")),
                bad, "tracking_to_world \"rvec\" is not 3 numbers"));

  // The fit.
  CHECK(refused(without("fit"), bad, "no \"fit\" object"));
  CHECK(refused(with_fit("rms_px", -0.01), bad, "\"rms_px\""));
  CHECK(refused(with_fit("views", 0), bad, "\"views\" is 0"));
  CHECK(refused(with_fit("views", 1.5), bad, "\"views\""));
  CHECK(refused(with_fit("edge_m", 0), bad, "\"edge_m\""));
  CHECK(refused(with_fit("edge_m", -0.15), bad, "\"edge_m\""));
  json zero_rms = with_fit("rms_px", 0);
  CHECK(cam::parse_tracked_registration(zero_rms.dump()).ok());

  // What JSON text cannot carry, refused by validate and by format alike.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  CHECK(cam::validate_tracked_registration(golden()).ok());
  cam::TrackedRegistration r = golden();
  r.sensor = "\xff";
  CHECK(invalid(r, "\"sensor\" is not UTF-8"));
  r = golden();
  r.session = {};
  CHECK(invalid(r, "all zero"));
  r = golden();
  r.world = {};
  CHECK(invalid(r, "names neither"));
  r = golden();
  r.world.sensor = "N";
  CHECK(invalid(r, "names both"));
  r = golden();
  r.world = {};
  r.world.sensor = "\xfe";
  CHECK(invalid(r, "world: \"sensor\" is not UTF-8"));
  r = golden();
  r.world.apriltag->size_m = nan;
  CHECK(invalid(r, "\"size_m\""));
  r = golden();
  r.tracking_to_world[0] *= 1.01;
  CHECK(invalid(r, "tracking_to_world: transform's rotation is not"));
  r = golden();
  r.tracking_to_world[3][2] = inf;
  CHECK(invalid(r, "tracking_to_world: transform is not finite"));
  r = golden();
  r.rms_px = nan;
  CHECK(invalid(r, "\"rms_px\""));
  r = golden();
  r.rms_px = inf;
  CHECK(invalid(r, "\"rms_px\""));
  r = golden();
  r.views = 0;
  CHECK(invalid(r, "\"views\" is 0"));
  r = golden();
  r.edge_m = 0.0;
  CHECK(invalid(r, "\"edge_m\""));
  r = golden();
  r.edge_m = nan;
  CHECK(invalid(r, "\"edge_m\""));
  CHECK(invalid(cam::TrackedRegistration{}, "\"sensor\" is empty"));
  return 0;
}

int test_files() {
  const vkc::test::Scratch scratch("sensor_tracked_registration_");
  const std::string path = (scratch.path / "take.registration.json").string();
  const cam::TrackedRegistration written = golden();
  CHECK(cam::write_tracked_registration(path, written,
                                        vkc::WriteMode::CreateNew));
  const auto read = cam::read_tracked_registration(path);
  CHECK(read.ok());
  if (check_same(read.value(), written, 1e-12) != 0) return 1;
  // The file holds exactly what format gives.
  const auto text = cam::format_tracked_registration(written);
  const auto bytes = vkc::test::read_bytes(path);
  CHECK(std::string(bytes.begin(), bytes.end()) == text.value());

  // CreateNew refuses the file now there, and leaves it.
  cam::TrackedRegistration other = golden();
  other.world_epoch = 4;
  const vkc::Status exists =
      cam::write_tracked_registration(path, other, vkc::WriteMode::CreateNew);
  CHECK(exists.domain() == Code::InvalidArgument &&
        exists.message().find(path) != std::string::npos);
  CHECK(vkc::test::read_bytes(path) == bytes);
  // A refused registration touches nothing, even under Replace.
  CHECK(cam::write_tracked_registration(path, cam::TrackedRegistration{},
                                        vkc::WriteMode::Replace)
            .domain() == Code::InvalidArgument);
  CHECK(vkc::test::read_bytes(path) == bytes);
  const std::string untouched = (scratch.path / "refused.json").string();
  CHECK(says(
      cam::write_tracked_registration(untouched, cam::TrackedRegistration{},
                                      vkc::WriteMode::CreateNew),
      Code::InvalidArgument, "\"sensor\" is empty"));
  CHECK(!fs::exists(untouched));
  // Replace replaces it, and leaves nothing else beside it.
  CHECK(cam::write_tracked_registration(path, other, vkc::WriteMode::Replace));
  const auto replaced = cam::read_tracked_registration(path);
  CHECK(replaced.ok() && replaced.value().world_epoch == 4);
  CHECK(std::distance(fs::directory_iterator(scratch.path),
                      fs::directory_iterator()) == 1);

  // Reading: a missing file is the core's NotFound; a refused document's
  // message names the file, its domain kept.
  CHECK(cam::read_tracked_registration(path + ".missing").status().domain() ==
        Code::NotFound);
  const std::string later = (scratch.path / "later.json").string();
  json later_doc = json::parse(kGolden);
  later_doc["version"] = 2;
  CHECK(vkc::write_file(later, later_doc.dump(), vkc::WriteMode::CreateNew));
  const auto later_file = cam::read_tracked_registration(later);
  CHECK(!later_file.ok() && later_file.status().domain() == Code::Unsupported &&
        later_file.status().message().rfind(later + ": tracked registration: ",
                                            0) == 0);

  // A tracked source places a frame only for the session and epoch it was
  // registered for. The camera 1 m along the tracking world's x is at
  // (1, 3, 3) in the array's.
  cam::Mat4d cam_to_tracking(1.0);
  cam_to_tracking[3] = glm::dvec4(1.0, 0.0, 0.0, 1.0);
  CHECK(place(path, other.session, 4, cam_to_tracking));
  CHECK(near(apply(other.tracking_to_world * cam_to_tracking, cam::Vec3d(0.0)),
             cam::Vec3d(1.0, 3.0, 3.0), 1e-12));
  CHECK(place(path, other.session, 3, cam_to_tracking).domain() ==
        Code::InvalidArgument);
  std::array<std::uint8_t, 16> another = other.session;
  another[15] ^= 1;
  CHECK(place(path, another, 4, cam_to_tracking).domain() ==
        Code::InvalidArgument);
  return 0;
}

}  // namespace

int main() {
  if (test_golden() != 0) return 1;
  if (test_round_trip() != 0) return 1;
  if (test_refusals() != 0) return 1;
  if (test_files() != 0) return 1;
  std::printf("tracked registration tests passed\n");
  return 0;
}
