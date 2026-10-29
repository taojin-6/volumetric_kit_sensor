# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-10-29

### Added

- **Stereo depth by Fast-FoundationStereo, `sensor_ffs`** (`VS_WITH_FFS`,
  Linux, NVIDIA only; needs `VS_WITH_CUDA` and TensorRT ≥ 11, found by
  `cmake/vs_ffs.cmake`, never fetched). `FfsStereo` (`ffs/ffs_stereo.hpp`)
  runs an FFS ONNX export on TensorRT, its plan built once and cached by
  GPU, TensorRT version, model and size: a CUDA kernel makes the network's
  input from a `StereoRectifier`'s views, read in place, and another turns
  its disparity into 16-bit depth written straight into a Vulkan buffer,
  the frame's `RgbdFrame::depth_device`. `StereoDepthSensor`
  (`ffs/stereo_depth_sensor.hpp`) is any `IStereoSensor` as an
  `IRgbdSensor`, rectifying and matching on a thread of its own.
  `tools/ffs/export_onnx.py` exports FFS from a local NVlabs checkout with
  the changes TensorRT needs; `tools/ffs/make_test_onnx.py` makes the
  tests' models (DECISIONS.md, the 2026-10-29 FFS entry).

### Changed

- `vs_target_warnings` gives its flags to C++ and Objective-C++ only, and
  CUDA sources nvcc's equivalents.

## [0.7.0] - 2026-10-29

### Added

- **The RealSense infrared cameras as a stereo source.** `RealSenseStereo`
  (`realsense/realsense_stereo.hpp`, in `sensor_realsense`), a RealSense
  D400's two infrared cameras as an `IStereoSensor`: their 8-bit pictures
  as the camera rectifies them, luma on the host, and the colour camera's
  YUYV, the frameset held, nothing converted on the host; the projector
  off by default; the factory calibration as `StereoInfo`; librealsense's
  global time (DECISIONS.md, the 2026-10-29 RealSense stereo entry).
- `IStereoSensor::set_frame_notify`, as `IRgbdSensor`'s: a source that
  notifies nothing returns `Unsupported`, the default.
- `StereoInfo` carries what the source is, as `SensorInfo` has it: `id`,
  `model`, `clock`, `fps` and `color_controls`; `RealSenseStereo` fills
  them, and `StereoRectifier::depth_info()` copies them, so a depth source
  on a pair can join a `SensorArray`.

### Changed

- `RealSenseSensor` and `RealSenseStereo` run on one internal camera
  (`realsense/camera.hpp`); a frameset is one moment when it holds a frame
  of every mode opened, each within half a frame of the first.
  `sensor_realsense` links `sensor_stereo` publicly.

## [0.6.0] - 2026-10-29

### Fixed

- **`StereoRectifier`: rays past a lens's fold are 0.** Past the fold,
  where the radial model stops carrying rays outward, the model turns back
  and put rays the lens cannot image inside the picture, mirrored and
  marked covered: a tenth of a strong lens's view at alpha 1. `create` finds
  each camera's fold; the pass compares one radius.
- **`StereoRectifier`: a buffer the pictures share is taken over once.** A
  side-by-side frame's one buffer was taken over twice. Pictures in one
  buffer, a colour picture beside them included, must name the family that
  wrote it (refused otherwise), as `GpuFramePrep`'s inputs must, through
  one shared check, and `RectifiedPair::source` names the pass's family for
  each of them.

### Changed

- **`StereoRectifier`'s views are taken back with no transfer.** The pass
  writes every word, so it no longer acquires its views back from
  `VK_QUEUE_FAMILY_EXTERNAL`; a Vulkan reader need not hand them back.
  `RectifiedPair` states the contract: use pairs in turn (rectify, read,
  drop) and the views are the same two buffers, mapped once; drop a pair
  only once its reads have completed; map a view by its open descriptor;
  hand on `RectifiedPair::source`.
- **A pair rectified already keeps square pixels:** it is scaled by one
  factor about the centres, alpha choosing the picture cropped (0) or
  fitted (1), where it was scaled across and down apart.
- Luma alone (`YuvLayout::Luma`) is not checked for a matrix or a chroma
  siting, in `GpuFramePrep` and the rectifier.

## [0.5.0] - 2026-10-29

### Added

- **Depth on the device.** `RgbdFrame::depth_device`, a `DeviceDepth` (a
  buffer, the first sample's byte and its writer's queue family), holds the
  frame's 16-bit depth where a GPU stage left it, instead of the host
  `depth`; `RgbdFrame::has_depth()` counts either. `GpuFramePrep` reads it
  in place, at any even byte of its buffer, after taking it over from that
  queue family, with no copy; the depth pass's push constants gain
  `raw_first` (`DepthParams` is 72 bytes). A buffer several of a batch's
  depths and planes share is taken over once, and references to it naming
  different writers are refused (DECISIONS.md, the 2026-10-29 device-depth
  entry).

## [0.4.0] - 2026-10-29

### Added

- **The RealSense driver.** `RealSenseSensor` (`sensor_realsense`,
  `VS_WITH_REALSENSE`), one RealSense D400 camera as an `IRgbdSensor`: the
  camera's own depth and its colour camera's YUYV on the host, which
  `GpuFramePrep` converts on the device, with each camera's factory model
  and the factory extrinsic, on the host's clock. librealsense 2.59.1 or
  later is a prerequisite, found through its CMake package; CI installs it
  (DECISIONS.md, the 2026-10-29 RealSense entry).

## [0.3.0] - 2026-10-29

### Added

- **The stereo tier, `sensor_stereo`.** `StereoCalibration`, `StereoInfo`,
  `StereoFrame` and `IStereoSensor` (`stereo/stereo_sensor.hpp`): what a
  stereo camera's driver implements and hands out, no SDK. And
  `StereoRectifier` (`stereo/stereo_rectifier.hpp`): the rectification
  computed once as OpenCV's `stereoRectify` computes it (Bouguet's,
  `CALIB_ZERO_DISPARITY`, `alpha`, any size), an already rectified pair
  only resampled, then one Vulkan compute pass a view, every layout from
  the host or a device buffer, `Luma8` or `Rgba8` views in device buffers
  exported where the device exports memory (`RectifiedPair`, `StereoRig`,
  `depth_info()`). `tools/stereo_rectify_reference.py` pins OpenCV's numbers
  in its test (DECISIONS.md, the 2026-10-29 stereo entry).
- **`YuvLayout::Luma`**, a mono camera's luma alone, one plane.
  `GpuFramePrep` takes it, from the host or a buffer, and turns it grey;
  `PictureReadback` reads it, with no chroma.

### Changed

- The colour pass's push constants gain `luma_only` (120 bytes), and its
  checks and plane binding are one function the rectifier shares; a
  `PictureReadback` YUYV refusal now reads "reads 4:2:0 pictures and luma
  alone".

## [0.2.0] - 2026-10-29

### Added

- **Colour from the host as Y'CbCr planes, and YUYV.** `YuvImage::host`
  points at a picture's planes on the host, as a camera's SDK hands them
  over, and `YuvLayout::Yuyv` is a UVC camera's 4:2:2 (Y0 Cb Y1 Cr, even
  width). `GpuFramePrep` stages host planes beside the depth, as it stages
  packed words, and converts them on the device; YUYV reads from a buffer
  too. `RgbdFrame::has_color()` counts host planes. `PictureReadback`
  refuses YUYV. A planes-in-several-places refusal now reads "one of them"
  (DECISIONS.md, the 2026-10-29 host-planes entry).

## [0.1.1] - 2026-10-29

### Changed

- **Prepared depth mixes where the raw pixels are one surface.** The depth
  pass samples each pinhole pixel's sub-pixel source position as the
  bilinear mix of the four raw pixels around it when all four return, after
  the depth offset, within 4 cm of each other, and as the nearest raw pixel,
  as before, otherwise. A slope no longer comes out in half-pixel terraces,
  which fusion shaded as bands on floors and walls; an edge or a hole is
  never blended. No API or option changes. The pass takes about 1 µs more a
  frame on an RTX 4090 and 2 µs on an M5 Max; a consumer that re-pins gets
  different prepared depth on slopes (DECISIONS.md, the 2026-10-29 depth-mix
  entry).

## [0.1.0] - 2026-10-29

Extracted from `volumetric_kit_recon` at `5eb28085`; earlier history in that
repository. The commits here are recon's that touched the package's files,
each file at its path here and each `#N` written as
`taojin-6/volumetric_kit_recon#N`; a file's history from before an earlier
move inside recon stays there, and recon's CHANGELOG has the changes up to
`ac1b792e`. The tiers, tests, tools and fixtures moved into recon's
`sensor/`, on volumetric_kit_core alone, and took the package's names
(DECISIONS.md, the 2026-10-29 package entry); recon's design text for them
and the decisions that govern them moved with them, into DESIGN.md and
DECISIONS.md.

### Changed

- **BREAKING for consumers: every name changed**, with no alias, forwarding
  header or shim left. A consumer renames at its re-pin by the tables below,
  declares the package by this repository's URL and a SHA, and checks it
  with `vkc_require_sibling(volumetric_kit::sensor 0.1.0)`.

#### Headers

| Old (`volumetric_kit/recon/...`) | New (`volumetric_kit/sensor/...`) | Target |
| --- | --- | --- |
| `camera/array_calibration.hpp` | `camera/array_calibration.hpp` | sensor_camera |
| `camera/camera_model.hpp` | `camera/camera_model.hpp` | sensor_camera |
| `camera/color_controls.hpp` | `camera/color_controls.hpp` | sensor_camera |
| `camera/geometry.hpp` | `camera/geometry.hpp` | sensor_camera |
| `camera/projection.hpp` | `camera/projection.hpp` | sensor_camera |
| `camera/export.hpp` (generated) | `camera/export.hpp` | sensor_camera |
| `version.hpp` (recon's; recon keeps it) | `version.hpp` (new, the package's) | sensor_camera |
| `core/camera_params.hpp` | `frame/camera_params.hpp` | sensor_frame |
| `core/color_space.hpp` | `frame/color_space.hpp` | sensor_frame |
| (in `sensor/utils/gpu_frame_prep.hpp`: `DeviceFrame`) | `frame/device_frame.hpp` | sensor_frame |
| (in `sensor/array/sensor_array.hpp`: `DeviceFrameSet`) | `frame/device_frame.hpp` | sensor_frame |
| `core/math/vector_types.hpp` (recon keeps its own) | `frame/vector_types.hpp` (new, the package's) | sensor_frame |
| `core/device_requirements.hpp` (recon keeps its own) | `frame/device_requirements.hpp` (new, the package's) | sensor_frame |
| `sensor/rgbd_sensor.hpp` | `rgbd_sensor.hpp` | sensor_interface |
| `sensor/rgbd_frame.hpp` | `rgbd_frame.hpp` | sensor_interface |
| `sensor/yuv_image.hpp` | `yuv_image.hpp` | sensor_interface |
| `sensor/camera_conventions.hpp` | `camera_conventions.hpp` | sensor_interface |
| `sensor/color_conventions.hpp` | `color_conventions.hpp` | sensor_interface |
| `sensor/chroma_location.hpp` | `chroma_location.hpp` | sensor_interface |
| `sensor/export.hpp` (generated) | `export.hpp` | sensor_interface |
| `sensor/utils/gpu_frame_prep.hpp` | `prep/gpu_frame_prep.hpp` | sensor_prep |
| `sensor/utils/picture_readback.hpp` | `prep/picture_readback.hpp` | sensor_prep |
| `sensor/utils/export.hpp` (generated) | `prep/export.hpp` | sensor_prep |
| `sensor/array/sensor_array.hpp`, `rig_sets.hpp`, `export.hpp` | `array/...` (same names) | sensor_array |
| `sensor/video/hevc_syntax.hpp`, `color_codes.hpp`, `syntax_export.hpp` | `video/...` (same names) | sensor_video_syntax |
| `sensor/video/hevc_decoder.hpp`, `jpeg_decoder.hpp`, `decoded_picture.hpp`, `cuda_interop.hpp`, `export.hpp` | `video/...` (same names) | sensor_video |
| `sensor/apple/pixel_buffer_importer.hpp`, `export.hpp` | `apple/...` | sensor_apple |
| `sensor/orbbec/orbbec_sensor.hpp`, `orbbec_stream.hpp`, `orbbec_sync_config.hpp`, `export.hpp` | `orbbec/...` | sensor_orbbec |
| `io/image_io.hpp`, `io/image_io_export.hpp` (generated) | `io/image_io.hpp`, `io/image_io_export.hpp` | sensor_image_io |

Mesh files are grid's `mesh_io` (recon's CHANGELOG, 0.1.0).

#### Namespaces

| Old | New |
| --- | --- |
| `volumetric_kit::recon::sensor` (and nested: `::hevc`, `::orbbec`, `::video`, `::detail`) | `volumetric_kit::sensor` (nested names kept: `sensor::hevc`, ...) |
| `volumetric_kit::recon::camera` | `volumetric_kit::sensor::camera` |
| `volumetric_kit::recon::{DepthCameraParams, ColorCameraParams, ColorEncoding, is_canonical, srgb_to_linear, linear_to_srgb, unpack_srgb_to_linear, pack_linear_to_srgb, primaries_to_working}` | `volumetric_kit::sensor::...` (same names) |
| `volumetric_kit::recon::io::{load_color_packed, load_depth_u16, write_png_rgba8, encode_png_u16, write_png_u16}` | `volumetric_kit::sensor::io::...` |
| `volumetric_kit::recon::device_requirements()` for the decoders' extensions | `volumetric_kit::sensor::device_requirements()` (recon's keeps `scalarBlockLayout` only; merge both with `core::merge`) |
| `volumetric_kit::recon::{Vec2f, Vec3f, Vec4f, Mat3f, Mat4f}` as used by the sensor headers | `volumetric_kit::sensor::{Vec2f, Vec3f, Vec4f, Mat3f, Mat4f}` (recon keeps its own aliases) |
| `volumetric_kit::recon::Vec3d` (where sensor code used it) | `volumetric_kit::sensor::camera::Vec3d` |
| — (new) | `volumetric_kit::sensor::{version_string, version_major, version_minor, version_patch}` |

recon code writes `sensor::X` / `sensor::camera::X` (no `recon::sensor`
namespace remains). Test helpers: `vr_test::{no_decoder, no_jpeg_decoder,
bare_device, read_device_picture, hevc_access_units, Md5, metal_allocated_size,
bare_plane_textures_size}` -> `vs_test::...` (tests/).

#### CMake targets (each with a `volumetric_kit::` alias)

| Old | New |
| --- | --- |
| `recon_camera` | `sensor_camera` |
| — (new; recon_core's camera_params/color_space/pinhole.glsl + DeviceFrame) | `sensor_frame` (INTERFACE) |
| `recon_sensor` | `sensor_interface` (the real target `sensor` is the umbrella's) |
| `recon_sensor_utils` | `sensor_prep` |
| `recon_sensor_array` | `sensor_array` |
| `recon_sensor_video` | `sensor_video` |
| `recon_sensor_video_syntax` | `sensor_video_syntax` |
| `recon_sensor_apple` | `sensor_apple` |
| `recon_sensor_orbbec` | `sensor_orbbec` |
| `recon_image_io` | `sensor_image_io` |
| (umbrella `volumetric_kit::recon` reached them) | umbrella `volumetric_kit::sensor` (real target `sensor`, `VKC_PACKAGE_VERSION` 0.1.0) |
| `INCLUDE_TARGETS volumetric_kit::recon_core` (for pinhole.glsl) | `INCLUDE_TARGETS volumetric_kit::sensor_frame` |
| `recon_tsan_tests` (sensor members) | `sensor_tsan_tests` |

#### Options and cache variables

| Old | New |
| --- | --- |
| `VR_CAMERA_ONLY` | `VS_CAMERA_ONLY` (the package's; recon's option is removed: recon needs sensor_frame) |
| `VR_WITH_ORBBEC` | `VS_WITH_ORBBEC` |
| `VR_WITH_FFMPEG` | `VS_WITH_FFMPEG` |
| `VR_WITH_CUDA` | `VS_WITH_CUDA` |
| — | `VS_BUILD_TESTS` and `VS_WARNINGS_AS_ERRORS` (default: top level; in recon's build, recon's `VR_` ones), `VS_SANITIZE` |
| `VR_ORBBEC_SDK_VERSION`, `_MAJOR`, `_MINOR`, `_PATCH`, `VR_ORBBEC_SDK_MIN_VERSION` | `VS_ORBBEC_SDK_VERSION...` (set in the sensor package's root directory scope) |
| `VR_FFMPEG` (pkg-config prefix; `PkgConfig::VR_FFMPEG`) | `VS_FFMPEG` (`PkgConfig::VS_FFMPEG`) |
| `VR_NVJPEG_INCLUDE_DIR` | `VS_NVJPEG_INCLUDE_DIR` |
| `VR_JSON_INCLUDE_DIR` (recon's third_party) | `VS_JSON_INCLUDE_DIR` (the package's third_party) |
| `VR_STB_INCLUDE_DIR` | `VS_STB_INCLUDE_DIR` |
| `VR_CORE_MIN_VERSION` (recon keeps its own) | `VS_CORE_MIN_VERSION` (0.29.0) |
| — | `VR_SENSOR_MIN_VERSION` (recon's floor on the package, 0.1.0) |

#### CMake functions and modules

| Old | New |
| --- | --- |
| `cmake/vr_orbbec.cmake`, `vr_link_orbbec_sdk()` | `cmake/vs_orbbec.cmake`, `vs_link_orbbec_sdk()` |
| `cmake/vr_ffmpeg.cmake` | `cmake/vs_ffmpeg.cmake` |
| `cmake/vr_cuda.cmake` | `cmake/vs_cuda.cmake` |
| `vr_embed_shaders` / `vr_target_warnings` / `VR_SANITIZE` (recon keeps them) | the package's own `vs_embed_shaders` (symbol prefix `vs_`), `vs_target_warnings`, `VS_SANITIZE` |
| — | recon's `vr_embed_shaders(... INCLUDE_TARGETS <target>... WITH_SRC)`: recon's src/ plus a sibling's exported GLSL |

#### Macros

| Old | New |
| --- | --- |
| `VR_CAMERA_API`, `VR_CAMERA_STATIC_DEFINE` | `VS_CAMERA_API`, `VS_CAMERA_STATIC_DEFINE` |
| `VR_SENSOR_API`, `VR_SENSOR_STATIC_DEFINE` | `VS_INTERFACE_API`, `VS_INTERFACE_STATIC_DEFINE` |
| `VR_SENSOR_UTILS_API`, `_STATIC_DEFINE` | `VS_PREP_API`, `VS_PREP_STATIC_DEFINE` |
| `VR_SENSOR_ARRAY_API`, `_STATIC_DEFINE` | `VS_ARRAY_API`, `VS_ARRAY_STATIC_DEFINE` |
| `VR_SENSOR_VIDEO_API`, `_STATIC_DEFINE` | `VS_VIDEO_API`, `VS_VIDEO_STATIC_DEFINE` |
| `VR_SENSOR_VIDEO_SYNTAX_API`, `_STATIC_DEFINE` | `VS_VIDEO_SYNTAX_API`, `VS_VIDEO_SYNTAX_STATIC_DEFINE` |
| `VR_SENSOR_VIDEO_WITH_CUDA` | `VS_VIDEO_WITH_CUDA` |
| `VR_SENSOR_APPLE_API`, `_STATIC_DEFINE` | `VS_APPLE_API`, `VS_APPLE_STATIC_DEFINE` |
| `VR_SENSOR_ORBBEC_API`, `_STATIC_DEFINE` | `VS_ORBBEC_API`, `VS_ORBBEC_STATIC_DEFINE` |
| `VR_IMAGE_IO_API`, `_STATIC_DEFINE` | `VS_IMAGE_IO_API`, `VS_IMAGE_IO_STATIC_DEFINE` |
| — | `VS_VERSION_MAJOR`, `VS_VERSION_MINOR`, `VS_VERSION_PATCH`, `VS_VERSION_STRING` |
| `VR_CUDA_*`, `VR_NVJPEG_*` (cuda_driver/nvjpeg internals) | `VS_CUDA_*`, `VS_NVJPEG_*` |
| test-only: `VR_HEVC_DATA`, `VR_JPEG_DATA`, `VR_SYNC_FIXTURES`, `VR_TEST_*` (cuda tests) | `VS_HEVC_DATA`, `VS_JPEG_DATA`, `VS_SYNC_FIXTURES`, `VS_TEST_*` |

#### GLSL

| Old | New |
| --- | --- |
| `volumetric_kit/recon/core/shaders/pinhole.glsl` | `volumetric_kit/sensor/frame/shaders/pinhole.glsl` (exported by sensor_frame) |
| guard `VR_CORE_PINHOLE_GLSL` | `VS_FRAME_PINHOLE_GLSL` |
| `vr_world_to_camera`, `vr_unproject_pinhole`, `vr_pinhole_pixel`, `vr_project_pinhole`, `vr_nearest_pixel` | `vs_world_to_camera`, `vs_unproject_pinhole`, `vs_pinhole_pixel`, `vs_project_pinhole`, `vs_nearest_pixel` |
| `sensor/utils/shaders/{lens.glsl, undistort_color.glsl, undistort_*.comp}` (private) | `sensor/prep/shaders/...` (private to sensor_prep); guards `VR_SENSOR_LENS_GLSL` -> `VS_PREP_LENS_GLSL`, `VR_SENSOR_UNDISTORT_COLOR_GLSL` -> `VS_PREP_UNDISTORT_COLOR_GLSL`; embedded symbols `vr_undistort_*_comp_spv` -> `vs_undistort_*_comp_spv` |

#### Environment variables

| Old | New |
| --- | --- |
| `VR_TEST_HEVC_BACKEND` | `VS_TEST_HEVC_BACKEND` |

#### Tests (ctest names)

`recon_<name>` -> `sensor_<name>` for the moved tests, with these special cases:
`recon_core_color_space` -> `sensor_frame_color_space`,
`recon_core_pinhole_gpu` -> `sensor_frame_pinhole_gpu`,
`recon_io_image` -> `sensor_io_image`,
`recon_orbbec_sdk_smoke` -> `sensor_orbbec_sdk_smoke`,
`recon_decoder_test_harness` -> `sensor_decoder_test_harness`,
`recon_camera_*` -> `sensor_camera_*`,
`recon_sensor_X` -> `sensor_X`,
`recon_subproject_camera_only` -> `sensor_subproject_camera_only`,
`recon_subproject_apple_ffmpeg` -> `sensor_subproject_apple_ffmpeg`,
`recon_subproject_glsl` -> `sensor_subproject_glsl` (the pinhole was recon's
last exported GLSL).
Split: `recon_sensor_gpu_frame_prep`'s fusion half -> `recon_tsdf_prepared_frame`;
`recon_sensor_orbbec_sync_config`'s committed-rig case -> `recon_example_rig_sync_config`.
New: `sensor_smoke`, `sensor_package_boundary`, `sensor_subproject_{clean, defaults,
old_sensor, old_core, core_without_vulkan}`, `recon_library_sensor_links`.

#### Files

| Old | New |
| --- | --- |
| `tools/make_hevc_fixtures.sh`, `make_jpeg_fixtures.sh`, `ios_build_check.sh` | `tools/...` here |
| `tests/data/{hevc, jpeg, orbbec_sync}` | `tests/data/...` here (recon's examples read `patches_256x144.jpg` through `VS_TEST_DATA_DIR`) |
| `config/femto_mega_sync.json` | unchanged (recon's example configuration) |
| `.github/actions/install-orbbec-sdk` | the same path here; a consumer's CI uses it by commit |
