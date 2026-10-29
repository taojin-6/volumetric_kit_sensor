// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor {

core::Status IStereoSensor::set_frame_notify(std::function<void()> /*notify*/) {
  return core::Status::unsupported(
      "IStereoSensor::set_frame_notify: this source notifies nothing; poll "
      "it");
}

}  // namespace volumetric_kit::sensor
