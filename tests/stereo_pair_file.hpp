// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// A calibration file holding a pair of cine cameras, each an entry of its
// own, and a Femto: the fixture stereo_calibration's test and the stereo
// tier's take a pair from. Camera tier only.

#include <utility>

#include "volumetric_kit/sensor/camera/array_calibration.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"

namespace stereo_pair_file {

namespace camera = volumetric_kit::sensor::camera;

// A 2160p lens with the 09-29 Blackmagic model's intrinsics, scaled from
// 1080p, and a mild rational distortion; `dx` moves its principal point.
inline camera::CameraModel lens(double dx) {
  camera::CameraModel c;
  c.size = {3840, 2160};
  c.intrinsics = {1702.41, 1702.12, 1917.38 + dx, 1108.27};
  c.distortion.k1 = 0.05;
  c.distortion.k2 = -0.01;
  c.distortion.p1 = 2e-4;
  c.distortion.k4 = 0.02;
  return c;
}

// The left camera's point in the right one: 0.4 m along +x of the left
// camera, a little toed in and tilted.
inline camera::Mat4d left_to_right() {
  return camera::matrix_from_rodrigues({camera::Vec3d(0.004, -0.03, 0.002),
                                        camera::Vec3d(-0.398, 0.006, -0.011)});
}

// A file of a Femto and two cine cameras, "bm-1" (left) and "bm-2"
// (right), posed in a world neither of them is at.
inline camera::ArrayCalibration pair_file() {
  camera::ArrayCalibration file;
  camera::SensorCalibration femto;
  femto.id = "femto";
  femto.color_to_world = camera::Mat4d(1.0);
  camera::SensorCalibration left;
  left.id = "bm-1";
  left.color_to_world = camera::matrix_from_rodrigues(
      {camera::Vec3d(0.3, -1.1, 0.2), camera::Vec3d(1.5, -0.4, 2.25)});
  left.color = camera::CameraCalibration{lens(0.0),
                                         camera::IntrinsicsSource::Calibrated};
  camera::SensorCalibration right = left;
  right.id = "bm-2";
  right.color_to_world =
      *left.color_to_world * camera::rigid_inverse(left_to_right());
  right.color->model = lens(3.5);
  file.sensors = {std::move(left), std::move(right), std::move(femto)};
  return file;
}

}  // namespace stereo_pair_file
