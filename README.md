# volumetric_kit_sensor

Cameras, capture and the device-frame contract for the volumetric_kit family:
the camera vocabulary and rig calibration file, the RGB-D sensor interface, GPU
frame preparation, the sensor array, the video decoders, the Apple importer,
the Orbbec driver and image I/O. On volumetric_kit_core alone.

Lives in volumetric_kit_recon's `sensor/` until it moves to its own repository.

## Building

```sh
cmake -S sensor -B build-sensor -DCMAKE_BUILD_TYPE=Release
cmake --build build-sensor --parallel
ctest --test-dir build-sensor --output-on-failure
```

Options: `VS_WITH_ORBBEC`, `VS_WITH_FFMPEG`, `VS_WITH_CUDA` (Linux decoders),
`VS_CAMERA_ONLY` (the camera tier alone, no Vulkan), `VS_BUILD_TESTS`.
