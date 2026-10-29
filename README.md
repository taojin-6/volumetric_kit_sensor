# volumetric_kit_sensor

Cameras, capture and the device-frame contract of the `volumetric_kit`
family: the camera vocabulary and the rig's calibration file, the RGB-D
sensor interface, GPU frame preparation, the sensor array, the hardware video
decoders, Apple's pixel-buffer importer, the Orbbec driver and image files.
A camera stack is anything that produces `DeviceFrame`s, the posed,
undistorted RGB-D frames on the device that fusion reads.

It lives in [volumetric_kit_recon](https://github.com/taojin-6/volumetric_kit_recon)'s
`sensor/` until it moves, with its history, to a repository of its own.

> [AGENTS.md](AGENTS.md) holds the working rules, [DESIGN.md](DESIGN.md) the
> tiers and their contracts, [DECISIONS.md](DECISIONS.md#decision-index) the
> dated decisions, and [CHANGELOG.md](CHANGELOG.md) the versions, with the
> old recon names mapped to the new ones in 0.1.0.

## In the family

The package builds on [volumetric_kit_core](https://github.com/taojin-6/volumetric_kit_core)
alone and never includes recon. An application declares the packages in the
order core → sensor → grid → recon: recon's libraries link the frame tier
(`sensor_frame`), and grid takes nothing from this package.

## Tiers

Targets are `volumetric_kit::sensor_<tier>`, headers
`volumetric_kit/sensor/<tier>/…`, namespace `volumetric_kit::sensor`. The
umbrella `volumetric_kit::sensor` links every tier the build has.

| Target | Headers | What | Links |
| --- | --- | --- | --- |
| `sensor_camera` | `camera/` | `CameraModel`, rigid geometry, the lens model, the calibration file; no Vulkan | core base, GLM |
| `sensor_frame` | `frame/` | `DeviceFrame`, the pinhole cameras, `ColorEncoding` and the sRGB curve, `pinhole.glsl`, `device_requirements()`; header-only | core Vulkan |
| `sensor_interface` | the root | `IRgbdSensor`, `RgbdFrame`, `YuvImage`, camera and colour conventions | camera, frame |
| `sensor_prep` | `prep/` | `GpuFramePrep`, `PictureReadback` | interface |
| `sensor_array` | `array/` | `SensorArray`, `RigSets` | prep |
| `sensor_video_syntax` | `video/` | H.265 syntax read on the host, every build | core base |
| `sensor_video` | `video/` | `HevcDecoder`, `JpegDecoder`, CUDA interop | frame |
| `sensor_apple` | `apple/` | `PixelBufferImporter`, every Apple build | frame |
| `sensor_orbbec` | `orbbec/` | `OrbbecSensor`, the rig's sync file | interface, video |
| `sensor_image_io` | `io/` | image files | core base |

## Building

Requires CMake ≥ 3.21, a C++17 compiler and the Vulkan toolchain (MoltenVK
on Apple). The package fetches the core pinned by commit and checks it is
0.29.0 or newer.

```sh
recon_root="$(git rev-parse --show-toplevel)"
cmake -S "$recon_root/sensor" -B "$recon_root/build-sensor" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$recon_root/build-sensor" --parallel
ctest --test-dir "$recon_root/build-sensor" --output-on-failure
```

| Option | Default | Effect |
| --- | --- | --- |
| `VS_BUILD_TESTS` | top level | the package's tests |
| `VS_WARNINGS_AS_ERRORS` | top level | `-Werror` on first-party code |
| `VS_CAMERA_ONLY` | OFF | `sensor_camera` alone, with no Vulkan; refuses the drivers |
| `VS_WITH_ORBBEC` | OFF | the Orbbec driver |
| `VS_WITH_FFMPEG` | OFF | the video decoders off Apple; needs `VS_WITH_CUDA`, refused on Apple |
| `VS_WITH_CUDA` | OFF | NVIDIA's hardware for the decoders |
| `VS_SANITIZE` | empty | sanitizers, e.g. `address;undefined` |

The GPU tests follow the core's test policy (its README, "Build and test"):
`VKC_REQUIRE_VULKAN_DEVICE`, `VKC_TEST_VALIDATION` and
`VKC_TEST_SYNC_VALIDATION`.

## Consuming the package

recon declares it after the core, by `SOURCE_DIR`, so recon's configure
takes the `VS_` options. An application fetches it pinned by commit (the
package has no install rules), declares it before anything that also does,
and checks the floor it needs; FetchContent keeps the first declaration:

```cmake
include(FetchContent)
FetchContent_Declare(volumetric_kit_core GIT_REPOSITORY
  https://github.com/taojin-6/volumetric_kit_core.git GIT_TAG <commit>)
FetchContent_Declare(volumetric_kit_sensor GIT_REPOSITORY
  https://github.com/taojin-6/volumetric_kit_recon.git GIT_TAG <commit>
  SOURCE_SUBDIR sensor)
set(VKC_WITH_VULKAN ON)
FetchContent_MakeAvailable(volumetric_kit_core volumetric_kit_sensor)
vkc_require_sibling(volumetric_kit::sensor 0.1.0)
target_link_libraries(my_app PRIVATE volumetric_kit::sensor_array)
```

A consumer with no GPU work, such as calib's solver, sets `VS_CAMERA_ONLY`
ON and links `volumetric_kit::sensor_camera`, with no Vulkan SDK.

A consumer's kernels include the exported pinhole through the core's
`INCLUDE_TARGETS`, as recon's do:

```cmake
vkc_compile_shaders(my_kernels INCLUDE_TARGETS volumetric_kit::sensor_frame
                    SHADERS my.comp)
# my.comp: #include "volumetric_kit/sensor/frame/shaders/pinhole.glsl"
```

An application that opens one device for this package and recon merges the
two packages' requirements:
`core::merge(sensor::device_requirements(), recon::device_requirements())`.

## Image files

`volumetric_kit::sensor_image_io` provides encoded RGB image loading, 16-bit
grayscale depth PNG loading with explicit units, and RGBA8 and 16-bit
grayscale PNG writing (`io/image_io.hpp`, namespace `sensor::io`); it links
the core's base tier alone, with no Vulkan. These are host file operations;
they allocate no Vulkan resources.

## Orbbec driver

The Orbbec (Femto Mega) driver is off by default and needs the
[Orbbec SDK](https://github.com/orbbec/OrbbecSDK_v2/releases) ≥ 2.10.6
installed. It is not fetched: install it once, outside the repo (the family
convention is `<workspace>/third_party/OrbbecSDK_v<version>`), and point the
build at its root:

```sh
cmake -S "$recon_root/sensor" -B "$recon_root/build-sensor" \
    -DCMAKE_BUILD_TYPE=Release -DVS_WITH_ORBBEC=ON -DOrbbecSDK_ROOT=<sdk>
# on Linux, add -DVS_WITH_FFMPEG=ON -DVS_WITH_CUDA=ON
# or once, for every repo that finds it:  export OrbbecSDK_ROOT=<sdk>
```

The driver decodes each camera's colour itself, so it needs the video
decoders too (below: every Apple build has them; on Linux, FFmpeg and
CUDA). That builds `volumetric_kit::sensor_orbbec` -- `OrbbecSensor`, one
camera as an `IRgbdSensor` (`orbbec/orbbec_sensor.hpp`), which a
`SensorArray` reads as a synced rig. recon's `fuse_orbbec` is the live
example.

## Video decoders

The HEVC and JPEG decoders (`volumetric_kit::sensor_video`) decode on the
GPU's hardware and hand their pictures out on the device only. Every Apple
build has them, on VideoToolbox with no FFmpeg; their pictures reach Vulkan
through `PixelBufferImporter` (`volumetric_kit::sensor_apple`). On Linux
they are off by default: NVDEC through FFmpeg ≥ 6.1, found through
pkg-config, and nvJPEG, with the CUDA 13 toolkit's headers
(`cmake/vs_cuda.cmake`). `VS_WITH_FFMPEG` on Apple fails the configure. The
H.265 syntax the Apple decoder reads (`volumetric_kit::sensor_video_syntax`,
`video/hevc_syntax.hpp`) is host code, in every build.

```sh
sudo apt install pkg-config libavcodec-dev libavutil-dev
cmake -S "$recon_root/sensor" -B "$recon_root/build-sensor" \
    -DCMAKE_BUILD_TYPE=Release -DVS_WITH_FFMPEG=ON -DVS_WITH_CUDA=ON
# fail, rather than skip, where the decoders find no device path
# (videotoolbox on a Mac):
VS_TEST_HEVC_BACKEND=cuda ctest --test-dir "$recon_root/build-sensor" -R video
```

## Versions

The version is `project()`'s (0.1.0), carried by the umbrella as
`VKC_PACKAGE_VERSION`. Before 1.0, a change to a public header, the
exported GLSL, a target or option name, or the calibration file's format
bumps the minor in the PR that makes it; CHANGELOG.md has a section per
version.

## License

MIT — see [LICENSE](LICENSE).
