// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file stereo/stereo_calibration.hpp
/// @brief Where `stereo_calibration` was: it is the camera tier's now
///        (`camera/stereo_rectification.hpp`). This header includes that one
///        and `stereo/stereo_sensor.hpp`, as before, so an include of this
///        path keeps compiling.
///
/// TODO(sensor): remove once the family's consumers include
/// `camera/stereo_rectification.hpp`.

#include "volumetric_kit/sensor/camera/stereo_rectification.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"
