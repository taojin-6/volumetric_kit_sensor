// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The rig's colour responses fitted by hand (DESIGN.md, "colour response
// fit"): opens the Orbbec rig at its calibration's colour controls, H.265 as
// it streams live, takes a few sets of an empty, static room a second apart,
// fits every camera's response, prints the report, and writes the calibration
// with the responses to a new file; it never replaces one.
//
//   color_fit --sync rig_sync.json --calibration in.json --out out.json
//             [--sets 8]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "color_fit.hpp"
#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/instance.hpp"
#include "volumetric_kit/sensor/array/sensor_array.hpp"
#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/orbbec/orbbec_sensor.hpp"

namespace camera = volumetric_kit::sensor::camera;
namespace color_fit = volumetric_kit::sensor::color_fit;
namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

namespace {

struct Args {
  std::string sync;
  std::string calibration;
  std::string out;
  int sets = 8;
};

int usage() {
  std::fprintf(stderr,
               "usage: color_fit --sync <rig_sync.json> --calibration "
               "<in.json> --out <new.json> [--sets <n>]\n");
  return 2;
}

int fail(const std::string& what) {
  std::fprintf(stderr, "color_fit: %s\n", what.c_str());
  return 1;
}

int fail(const vkc::Status& s) { return fail(s.message()); }

std::optional<Args> parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const char* value = argv[i + 1];
    if (key == "--sync") {
      args.sync = value;
    } else if (key == "--calibration") {
      args.calibration = value;
    } else if (key == "--out") {
      args.out = value;
    } else if (key == "--sets") {
      args.sets = std::atoi(value);
    } else {
      return std::nullopt;
    }
  }
  if (argc % 2 == 0 || args.sync.empty() || args.calibration.empty() ||
      args.out.empty() || args.sets < 2 || args.sets > 64) {
    return std::nullopt;
  }
  return args;
}

int run(const Args& args) {
  // Refused before the capture; the write's CreateNew is the guarantee.
  std::error_code error;
  if (std::filesystem::exists(args.out, error)) {
    return fail(args.out + " exists: color_fit writes a new file only");
  }
  auto read = camera::read_array_calibration(args.calibration);
  if (!read) {
    const std::string what = read.status().message();
    if (what.find("color_response") == std::string::npos) return fail(what);
    return fail(what +
                "; after a relock, delete every color_response, then "
                "fit again");
  }
  camera::ArrayCalibration calibration = std::move(read).value();
  if (!calibration.color_controls) {
    return fail("the calibration names no color_controls to fit at");
  }
  // The fit reads the colour as the cameras give it.
  for (camera::SensorCalibration& s : calibration.sensors) {
    s.color_response = {};
  }

  auto instance = vkc::Instance::create({});
  if (!instance) return fail(instance.status());
  const vkc::DeviceRequirements reqs = sensor::device_requirements();
  auto physical = instance->select_physical_device(reqs);
  if (!physical) return fail(physical.status());
  auto device = vkc::Device::create(*instance, *physical, reqs);
  if (!device) return fail(device.status());
  auto allocator = vkc::Allocator::create(instance->handle(), *device);
  if (!allocator) return fail(allocator.status());

  sensor::OrbbecSensor::Options streams;  // H.265, the rig's live codec
  streams.color_controls = calibration.color_controls;
  streams.device = &*device;
  streams.allocator = &*allocator;
  auto sensors = sensor::open_orbbec_sensors(args.sync, streams);
  if (!sensors) return fail(sensors.status());
  sensor::SensorArray::Options options;
  options.calibration = calibration;
  options.device = &*device;
  options.allocator = &*allocator;
  auto array = sensor::SensorArray::open(std::move(sensors).value(), options);
  if (!array) return fail(array.status());
  std::vector<sensor::SensorInfo> infos;
  for (std::size_t i = 0; i < array->size(); ++i) {
    infos.push_back(array->sensor(i).info());
  }
  color_fit::Options fit_options;
  fit_options.sets = static_cast<std::uint32_t>(args.sets);
  auto fit = color_fit::ColorFit::create(*device, *allocator, std::move(infos),
                                         fit_options);
  if (!fit) return fail(fit.status());

  if (const vkc::Status s = array->start(); !s.ok()) return fail(s);
  int taken = 0;
  for (int tries = 0; taken < args.sets && tries < 4 * args.sets; ++tries) {
    auto set = array->wait_set(std::chrono::seconds(2));
    if (!set) {
      array->stop();
      return fail(set.status());
    }
    if (!*set || !(*set)->complete()) continue;
    auto prepared = array->process(**set);
    if (!prepared) {
      array->stop();
      return fail(prepared.status());
    }
    if (const vkc::Status s = (*fit)->add(*prepared); !s.ok()) {
      array->stop();
      return fail(s);
    }
    ++taken;
    std::this_thread::sleep_for(std::chrono::seconds(1));  // AWB's wander
  }
  array->stop();
  if (taken < args.sets) return fail("too few complete sets: a camera dropped");

  auto report = (*fit)->solve(calibration);
  if (!report) return fail(report.status());
  std::fputs(color_fit::format_report(*report).c_str(), stdout);
  const vkc::Status written = camera::write_array_calibration(
      args.out, report->calibration, vkc::WriteMode::CreateNew);
  if (!written.ok()) return fail(written);
  std::printf("wrote %s\n", args.out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::optional<Args> args = parse(argc, argv);
  if (!args) return usage();
  return run(*args);
}
