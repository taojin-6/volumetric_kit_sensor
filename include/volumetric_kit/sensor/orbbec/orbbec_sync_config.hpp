// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file sensor/orbbec/orbbec_sync_config.hpp
/// @brief A rig's hardware sync configuration: which cameras, which is the
///        primary, and each one's sync settings.
///
/// The Orbbec SDK's `MultiDeviceSyncConfig.json` layout, so the SDK's own tools
/// read the same file, plus an optional `master_serial` that must name the
/// primary:
///
/// @code{.json}
/// {"master_serial": "CL2A141000N",
///  "devices": [{"sn": "CL2A141000N", "syncConfig": {
///    "syncMode": "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY", "depthDelayUs": 0,
///    "colorDelayUs": 0, "trigger2ImageDelayUs": 0, "triggerOutEnable": true,
///    "triggerOutDelayUs": 0, "framesPerTrigger": 1}}]}
/// @endcode

#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/recon/sensor/orbbec/export.hpp"

namespace volumetric_kit::recon::sensor {

/// @brief A camera's role in a hardware-synchronised rig, as the camera itself
///        reports it.
///
/// Read when a camera is opened and written only when asked
/// (`OrbbecRig::Options::apply_sync_config`): the role persists on the
/// camera, and changing it for a one-camera run would break the rig for the
/// next. See @ref waits_for_primary for what it means to a caller.
enum class OrbbecSyncMode {
  FreeRun,     ///< Not synchronised; streams on its own clock.
  Standalone,  ///< Not synchronised; streams on its own clock.
  Primary,     ///< Streams on its own and drives the sync line.
  Secondary,   ///< Captures only on the primary's sync signal.
  /// The rig's secondaries. The SDK says this mode captures on its own and
  /// re-times to a signal; measured, it delivers nothing without one, so this
  /// driver treats it as @ref Secondary.
  SecondarySynced,
  /// Captures only on a host trigger, which this driver never sends; a
  /// camera's open refuses it.
  SoftwareTriggering,
  /// Captures only on a trigger another camera sends down the sync line.
  HardwareTriggering,
  Other,  ///< A mode this driver does not name.
};

/// @param mode  A camera's reported sync mode.
/// @return `true` when the camera produces frames only on another camera's
///         signal. Started on its own, such a camera delivers nothing -- a
///         poll keeps returning no frame, as if it were merely slow. `false`
///         for @ref OrbbecSyncMode::SoftwareTriggering, which waits for the
///         host and which a camera's open refuses.
VR_SENSOR_ORBBEC_API bool waits_for_primary(OrbbecSyncMode mode) noexcept;

/// @return A stable lowercase name for @p mode (`"primary"`,
///         `"secondary-synced"`, ...), for logs.
VR_SENSOR_ORBBEC_API const char* to_string(OrbbecSyncMode mode) noexcept;

/// @brief One camera's sync settings: the SDK's `OBMultiDeviceSyncConfig`.
struct OrbbecSyncSettings {
  OrbbecSyncMode mode = OrbbecSyncMode::Standalone;
  int depth_delay_us = 0;
  int color_delay_us = 0;
  int trigger_to_image_delay_us = 0;
  bool trigger_out_enable = false;
  int trigger_out_delay_us = 0;
  int frames_per_trigger = 1;
};

/// @brief One camera of a rig's sync configuration.
struct OrbbecSyncDevice {
  std::string serial;
  OrbbecSyncSettings sync;
};

/// @brief A rig's sync configuration, one entry per camera in file order.
struct OrbbecRigSyncConfig {
  std::vector<OrbbecSyncDevice> devices;
};

/// @brief Parse a sync configuration document.
/// @return The configuration; or `Status::Code::InvalidArgument` naming what
///         is wrong: not JSON, no `devices`, a device without an `sn` or a
///         `syncConfig` whose fields are all present and typed, a `syncMode`
///         this driver does not name, a negative delay, a repeated serial, or
///         a `master_serial` that is not the one primary.
VR_SENSOR_ORBBEC_API core::Result<OrbbecRigSyncConfig> parse_orbbec_sync_config(
    const std::string& json);

/// @brief Read and parse the sync configuration file at @p path.
/// @return As @ref parse_orbbec_sync_config, messages naming the file; or
///         `Status::Code::IoError` if it cannot be read.
VR_SENSOR_ORBBEC_API core::Result<OrbbecRigSyncConfig> read_orbbec_sync_config(
    const std::string& path);

}  // namespace volumetric_kit::recon::sensor
