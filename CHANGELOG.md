# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-29

Seeded from `volumetric_kit_recon`'s camera and sensor tiers at `ac1b792e`;
history in recon's CHANGELOG up to that commit. The tiers, tests, tools and
fixtures moved into recon's `sensor/`, on volumetric_kit_core alone, and
took the package's names (DECISIONS.md, the 2026-10-29 package entry);
recon's design text for them and the decisions that govern them moved with
them, into DESIGN.md and DECISIONS.md.

### Changed

- **BREAKING for consumers: every name changed**, with no alias, forwarding
  header or shim left. A consumer renames at its re-pin by the tables below,
  declares the package by URL, SHA and `SOURCE_SUBDIR sensor`, and checks it
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
bare_plane_textures_size}` -> `vs_test::...` (sensor/tests).

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
| — | `VS_BUILD_TESTS` (default: top level; recon's CI passes ON), `VS_WARNINGS_AS_ERRORS`, `VS_SANITIZE` |
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
| `cmake/vr_orbbec.cmake`, `vr_link_orbbec_sdk()` | `sensor/cmake/vs_orbbec.cmake`, `vs_link_orbbec_sdk()` |
| `cmake/vr_ffmpeg.cmake` | `sensor/cmake/vs_ffmpeg.cmake` |
| `cmake/vr_cuda.cmake` | `sensor/cmake/vs_cuda.cmake` |
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
`recon_subproject_apple_ffmpeg` -> `sensor_subproject_apple_ffmpeg`.
Split: `recon_sensor_gpu_frame_prep`'s fusion half -> `recon_tsdf_prepared_frame`;
`recon_sensor_orbbec_sync_config`'s committed-rig case -> `recon_example_rig_sync_config`.
New: `sensor_smoke`, `sensor_package_boundary`, `sensor_subproject_{defaults, old_sensor,
glsl, old_core, core_without_vulkan}`, `recon_library_sensor_links`.

#### Files

| Old | New |
| --- | --- |
| `tools/make_hevc_fixtures.sh`, `make_jpeg_fixtures.sh`, `ios_build_check.sh` | `sensor/tools/...` |
| `tests/data/{hevc, jpeg, orbbec_sync}` | `sensor/tests/data/...` (recon's examples read `patches_256x144.jpg` through `${volumetric_kit_sensor_SOURCE_DIR}`) |
| `config/femto_mega_sync.json` | unchanged (recon's example configuration) |
| `.github/actions/install-orbbec-sdk` | unchanged (recon's root, until X1) |
