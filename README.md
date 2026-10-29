# volumetric_kit_sensor

[![ci](https://github.com/taojin-6/volumetric_kit_sensor/actions/workflows/ci.yml/badge.svg)](https://github.com/taojin-6/volumetric_kit_sensor/actions/workflows/ci.yml)

Cameras, capture and the device-frame contract of the `volumetric_kit`
family: the camera vocabulary and the rig's calibration file, the RGB-D
sensor interface, GPU frame preparation, the sensor array, the hardware video
decoders, Apple's pixel-buffer importer, the Orbbec and RealSense drivers
and image files.
A camera stack is anything that produces `DeviceFrame`s, the posed,
undistorted RGB-D frames on the device that fusion reads.

It was extracted from [volumetric_kit_recon](https://github.com/taojin-6/volumetric_kit_recon)'s
`sensor/`, with its history (see [CHANGELOG.md](CHANGELOG.md)).

> [AGENTS.md](AGENTS.md) holds the working rules, [DESIGN.md](DESIGN.md) the
> tiers and their contracts, [DECISIONS.md](DECISIONS.md#decision-index) the
> dated decisions, and [CHANGELOG.md](CHANGELOG.md) the versions, with the
> old recon names mapped to the new ones in 0.1.0.

## In the family

The package builds on [volumetric_kit_core](https://github.com/taojin-6/volumetric_kit_core)
alone and never includes recon or grid. An application declares the
packages in the order core → sensor → grid → recon: recon's libraries link
the frame tier (`sensor_frame`), and grid takes nothing from this package.

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
| `sensor_stereo` | `stereo/` | `IStereoSensor`, `StereoFrame`, `StereoRectifier` (a calibrated pair rectified on the GPU) | interface; prep, privately |
| `sensor_video_syntax` | `video/` | H.265 syntax read on the host, every build | core base |
| `sensor_video` | `video/` | `HevcDecoder`, `JpegDecoder`, CUDA interop | frame |
| `sensor_apple` | `apple/` | `PixelBufferImporter`, every Apple build | frame |
| `sensor_orbbec` | `orbbec/` | `OrbbecSensor`, the rig's sync file | interface, video |
| `sensor_realsense` | `realsense/` | `RealSenseSensor` | interface |
| `sensor_ffs` | `ffs/` | `FfsStereo` (Fast-FoundationStereo on TensorRT), `StereoDepthSensor` (any stereo camera as an `IRgbdSensor`); NVIDIA only | stereo; video, privately |
| `sensor_image_io` | `io/` | image files | core base |

## Building

Requires CMake ≥ 3.21, a C++17 compiler and the Vulkan toolchain (MoltenVK
on Apple). The package fetches the core pinned by commit and checks it is
0.29.0 or newer.

```sh
sensor_root="$(git rev-parse --show-toplevel)"
cmake -S "$sensor_root" -B "$sensor_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$sensor_root/build" --parallel
ctest --test-dir "$sensor_root/build" --output-on-failure
```

| Option | Default | Effect |
| --- | --- | --- |
| `VS_BUILD_TESTS` | top level | the package's tests |
| `VS_WARNINGS_AS_ERRORS` | top level | `-Werror` on first-party code |
| `VS_CAMERA_ONLY` | OFF | `sensor_camera` alone, with no Vulkan; refuses the drivers |
| `VS_WITH_ORBBEC` | OFF | the Orbbec driver |
| `VS_WITH_REALSENSE` | OFF | the RealSense driver |
| `VS_WITH_FFMPEG` | OFF | the video decoders off Apple; needs `VS_WITH_CUDA`, refused on Apple |
| `VS_WITH_CUDA` | OFF | NVIDIA's hardware for the decoders |
| `VS_WITH_FFS` | OFF | `sensor_ffs`: stereo depth by FFS; needs `VS_WITH_CUDA` and TensorRT ≥ 11 |
| `VS_SANITIZE` | empty | sanitizers, e.g. `address;undefined` |

The GPU tests follow the core's test policy (its README, "Build and test"):
`VKC_REQUIRE_VULKAN_DEVICE`, `VKC_TEST_VALIDATION` and
`VKC_TEST_SYNC_VALIDATION`.

## Consuming the package

An application fetches it pinned by commit (the package has no install
rules), declares it before anything that also does, and checks the floor it
needs; FetchContent keeps the first declaration. Its configure takes the
`VS_WITH_` options; fetched, the package's tests and `-Werror` default off:

```cmake
include(FetchContent)
FetchContent_Declare(volumetric_kit_core GIT_REPOSITORY
  https://github.com/taojin-6/volumetric_kit_core.git GIT_TAG <commit>)
FetchContent_Declare(volumetric_kit_sensor GIT_REPOSITORY
  https://github.com/taojin-6/volumetric_kit_sensor.git GIT_TAG <commit>)
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
cmake -S "$sensor_root" -B "$sensor_root/build" \
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

## RealSense driver

The RealSense (D400, the D455) driver is off by default and needs
[librealsense](https://github.com/realsenseai/librealsense) ≥ 2.59.1
installed from the platform's packages, which carry its CMake package:

```sh
brew install librealsense                     # macOS
# Ubuntu 24.04 / 26.04: RealSense's apt repository, librealsense2-dev
#   (librealsense's doc/distribution_linux.md; CI's install-librealsense action)
cmake -S "$sensor_root" -B "$sensor_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DVS_WITH_REALSENSE=ON
```

That builds `volumetric_kit::sensor_realsense` -- `RealSenseSensor`, one
camera as an `IRgbdSensor` (`realsense/realsense_sensor.hpp`): the camera's
own depth and its colour camera's YUYV, which `GpuFramePrep` converts on the
device.

## Stereo pairs

`volumetric_kit::sensor_stereo` is built in every build with Vulkan, with
no option and no SDK. A stereo camera's driver implements `IStereoSensor`
(`stereo/stereo_sensor.hpp`); `StereoRectifier`
(`stereo/stereo_rectifier.hpp`) undistorts and rectifies its pairs on the
GPU into a row-aligned pinhole pair, `Luma8` or `Rgba8`, at the size a
matcher takes:

```cpp
VKC_ASSIGN(auto rectifier,
           sensor::StereoRectifier::create(
               device, allocator, camera.info(),
               {{960, 544}, 0.0f, sensor::StereoRectifier::Output::Luma8}));
VKC_ASSIGN(sensor::RectifiedPair pair, rectifier->rectify(frame));
```

`tools/stereo_rectify_reference.py` regenerates the OpenCV numbers its test
pins (OpenCV in a virtual environment; the build never needs it).

## Stereo depth (FFS)

`volumetric_kit::sensor_ffs` (`VS_WITH_FFS`, Linux, NVIDIA only) turns
rectified pairs into depth with Fast-FoundationStereo (FFS) on TensorRT:
`FfsStereo` reads the rectifier's views through CUDA and writes 16-bit depth
straight into a Vulkan buffer (`RgbdFrame::depth_device`), which
`GpuFramePrep` reads in place; `StereoDepthSensor` wraps any
`IStereoSensor`, the rectifier and `FfsStereo` as an `IRgbdSensor`, on a
thread of its own. It needs `VS_WITH_CUDA` (with nvcc, for its two
kernels) and TensorRT ≥ 11, found and never fetched (`cmake/vs_ffs.cmake`;
a tarball through `TensorRT_ROOT`):

```sh
sudo apt install libnvinfer-dev libnvonnxparsers-dev  # NVIDIA's CUDA repository
cmake -S "$sensor_root" -B "$sensor_root/build" -DCMAKE_BUILD_TYPE=Release \
    -DVS_WITH_FFMPEG=ON -DVS_WITH_CUDA=ON -DVS_WITH_FFS=ON
```

The model is an ONNX export of FFS that `tools/ffs/export_onnx.py` makes,
once per checkpoint and network size, from a local checkout of NVlabs'
[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) and
its research weights (NVIDIA's, under their licence; never vendored or
fetched). The network is the rig's size rounded up to multiples of 32
(960 x 544 stays, 1280 x 720 is exported as 1280 x 736). `FfsStereo` builds
its TensorRT plan once (minutes) and keeps it in a cache directory:

```sh
venv/bin/python tools/ffs/export_onnx.py --ffs ~/Fast-FoundationStereo \
    --size 960x544 --out ffs_23-36-37_960x544.onnx
```

```cpp
VKC_ASSIGN(auto sensor,
           sensor::StereoDepthSensor::create(
               device, allocator, std::move(camera),  // any IStereoSensor
               {{{960, 544}, 0.0f, sensor::StereoRectifier::Output::Luma8},
                {"ffs_23-36-37_960x544.onnx", "ffs_plans"}}));
VKC_TRY(sensor->start());  // then poll and prepare as for any IRgbdSensor
```

The tests run TensorRT and the kernels on two 593-byte models with FFS's
inputs and output (`tests/data/ffs`, made by `tools/ffs/make_test_onnx.py`),
and skip where CUDA does not reach the GPU unless `VS_TEST_HEVC_BACKEND`
is set.

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
cmake -S "$sensor_root" -B "$sensor_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DVS_WITH_FFMPEG=ON -DVS_WITH_CUDA=ON
# fail, rather than skip, where the decoders find no device path
# (videotoolbox on a Mac):
VS_TEST_HEVC_BACKEND=cuda ctest --test-dir "$sensor_root/build" -R video
```

## Versions

The version is `project()`'s (0.6.0), carried by the umbrella as
`VKC_PACKAGE_VERSION`. Before 1.0, a change to a public header, the
exported GLSL, a target or option name, or the calibration file's format
bumps the minor in the PR that makes it; CHANGELOG.md has a section per
version.

## License

MIT — see [LICENSE](LICENSE).
