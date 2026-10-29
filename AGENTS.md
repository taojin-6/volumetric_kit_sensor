# Sensor package instructions

`volumetric_kit_sensor`: cameras, capture and the device-frame contract, in
namespace `volumetric_kit::sensor`, headers `volumetric_kit/sensor/<tier>/…`,
targets `volumetric_kit::sensor_<tier>`, macros `VS_`. It depends on
volumetric_kit_core alone and never includes recon (`sensor_recon_free`).

While it lives in recon's `sensor/`, recon's AGENTS.md applies to it too.
