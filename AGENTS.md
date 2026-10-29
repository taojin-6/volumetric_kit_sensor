# Sensor package instructions

The working guide for every agent, Codex and Claude Code alike, on
`volumetric_kit_sensor`. `CLAUDE.md` imports this file and duplicates none
of it. While the package lives in volumetric_kit_recon's `sensor/`, recon's
AGENTS.md covers the repository's Git, CI and formatting practice; the rules
for this code are here.

## Project and scope

`volumetric_kit_sensor` is the family's camera side: the camera vocabulary
and the rig's calibration file, the RGB-D sensor interface, GPU frame
preparation, the sensor array, the hardware video decoders, Apple's
pixel-buffer importer, the Orbbec driver and image files. It hands
consumers posed, undistorted RGB-D frames on the device (`DeviceFrame`).

- It builds on [volumetric_kit_core](https://github.com/taojin-6/volumetric_kit_core)
  alone (fetched, pinned by commit), under the core's names (`core::Status`,
  `core::Device`). Code two or more siblings need belongs in the core.
- It never includes recon, which sits above it (`sensor_recon_free`).
  recon's libraries link `sensor_frame` alone, so a change there is a change
  to recon's contract.
- A sensor driver belongs here only if this package can build and test it.
  Platform-only drivers such as ARKit belong with their downstream
  application.
- Vulkan compute is the baseline on every platform, MoltenVK on Apple. The
  decoders use the platform's hardware (VideoToolbox, NVDEC, nvJPEG) and hand
  their pictures out on the device only.

## Read what the task needs

| Task | Read |
| --- | --- |
| Build, options, consuming the package | [README.md](README.md) |
| Tiers and their contracts | [DESIGN.md](DESIGN.md#tiers), then the tier under [Implementation reference](DESIGN.md#implementation-reference) |
| Colour or camera conventions | [Colour at the boundary](DESIGN.md#colour-at-the-boundary) and the tier's public headers |
| Changing a contract or understanding a constraint | [Decision index](DECISIONS.md#decision-index), then the dated entries; newest context wins |
| Old recon names | [CHANGELOG.md](CHANGELOG.md), 0.1.0 |

## Architecture and conventions

- A tier links only the tiers below it: `camera` the core's base tier and
  GLM; `frame` the core's Vulkan tier; the interface `camera` and `frame`;
  `prep` the interface; `array` the prep; `apple` `frame`; `video` `frame`
  (and, on Apple, `apple`); `orbbec` the interface and `video`;
  `video_syntax` and `io` the base tier.
  Drivers, decoders and preparation stay targets of their own.
- C++17, no compiler extensions. Namespace `volumetric_kit::sensor`, the
  camera vocabulary in `sensor::camera`; headers
  `include/volumetric_kit/sensor/<tier>/…`; macros `VS_` (`VK_` is
  Vulkan's); targets `volumetric_kit::sensor_<tier>`.
- Fallible APIs return `core::Status` / `core::Result<T>` and propagate with
  `VKC_TRY` / `VKC_ASSIGN`; no exception crosses the API. Build with C++
  exceptions enabled for the handlers that translate backend errors.
  Programmer errors use `VKC_CHECK`.
- Include Vulkan through `volumetric_kit/core/vulkan/vulkan.hpp`. Host PODs
  and their GLSL `layout(scalar)` mirrors stay byte-identical. A `create`
  that builds kernels first calls `check_device_requirements`.
- Kernel memory is device-local, reached through `CommandBatch`; the host
  does I/O. Preserve each producer's queue-family and lifetime contract
  (`frame/device_frame.hpp`).
- 8-bit colour is encoded; float colour is linear, in linear BT.709/D65.
  Drivers declare the encoding they produce and convert nothing.
- Full Doxygen on public classes and functions, as in `rgbd_sensor.hpp`.
  `@ref` only this package's names; write the core's in backticks.
- No `friend` declarations; test through public APIs.
- Mark deferred work `TODO(sensor)`.
- Keep first-party code warning-clean with `VS_WARNINGS_AS_ERRORS=ON`.
- Every Vulkan, Metal, CUDA or SDK handle owner deletes copy, provides the
  move pair, resets every member on move and `destroy()`, guards self-move,
  and keeps backend types out of public headers. A class that only
  aggregates such owners deletes copy and move, and its `create()` returns
  `Result<std::unique_ptr<T>>`.

## Build and validation

```sh
recon_root="$(git rev-parse --show-toplevel)"
cmake -S "$recon_root/sensor" -B "$recon_root/build-sensor" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$recon_root/build-sensor" --parallel
ctest --test-dir "$recon_root/build-sensor" --output-on-failure
```

- Always set the build type; measure in Release.
- Build the package alone for a change here, with the options the change
  touches (README, "Building"), and run recon's suite too when `frame`
  changes.
- Tests use no real hardware: a camera is checked by hand through recon's
  `fuse_orbbec`. Report such a check as run or not run, never as a test.
- Tests are `main()`s on the core's test support; GPU tests run on its
  policy, and CI sets `VKC_TEST_VALIDATION=1`. `VS_TEST_HEVC_BACKEND` makes
  the decoder tests fail rather than skip where no device path opens.

## Keeping the guidance current

- Keep this file short. Implementation detail goes in DESIGN.md, decision
  evidence in DECISIONS.md, each version's changes in CHANGELOG.md.
- When changing a locked decision, update its index entry and dated entry
  in DECISIONS.md in the same commit, and the DESIGN.md contract with it.
- `CLAUDE.md` stays an import of this file.
