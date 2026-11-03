# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.40.0] - 2026-11-03

### Added

- **A tracked sensor's stills** (`camera/tracked_stills.hpp`,
  `sensor_camera`, every build, camera-only included): `TrackedStills`, a
  tracked sensor's stills of one capture session's world epoch, each a
  `TrackedStill` with its image's file name (beside the file), its frame's
  time, its `CameraModel` and the rigid `cam_to_tracking` its tracking gave
  it. `validate_`, `parse_`, `read_`, `format_` and `write_tracked_stills`
  read and write the file, `"volumetric_kit.tracked_stills"` version 1:
  the registration's `sensor`, `session` and `world_epoch`, and the array
  file's camera block and transform form for each still. A tracked
  sensor's recorder writes it beside the session's images; calib's
  `register_tracked` reads it. DECISIONS.md, the 2026-11-03 tracked stills
  entry.

### Changed

- **`TrackedRegistration::edge_m` is the tag's edge the solve fitted the
  corners to**, which the solver is given, not one the stills measure; the
  file is unchanged.
- The array file's camera block, and the registration's head and session,
  are read and written in the internal `calibration_json.{hpp,cpp}`, which
  the stills share. Both files read, refuse and write as before, every
  message the same.

### Fixed

- A camera-only build with tests configures on Apple: it no longer adds the
  VideoToolbox clip's maker (`tools/vt_pattern_clip`), which needs the video
  tier.

## [0.36.0] - 2026-11-03

### Added

- **`HevcEncoder` on NVENC** (`video/hevc_encoder.hpp`, `sensor_video`, a
  Linux build with `VS_WITH_FFMPEG` and `VS_WITH_CUDA`): FFmpeg's
  `hevc_nvenc` behind the same API, rules and coding thread as VideoToolbox.
  NVENC codes targets alone, so `create` needs `Options::device` and
  `Options::allocator` (`InvalidArgument` without), on a GPU CUDA reaches
  whose device exports memory. A target is one of `kInFlight + 1` exported
  buffers made at `create` (`YuvImage::device`: the luma's rows from 0, the
  chroma's after them, at a stride of the width rounded up to the GPU's
  texture alignment), which NVENC reads through CUDA with no copy; `target`
  records the buffer's crossing into the batch. Effort is the preset (p1 to
  p7), every QP is set from the options, QP 0 to 51 is coded, sitings 1 to 5
  are refused, and a picture of 4096 rows or more is coded in 3 strips
  (`Facts::strips`), which an FFmpeg before 7.1 refuses in `create`.
- **`HevcEncoderCase::Rules::Sequence`** (`testing/hevc_conformance.hpp`):
  what the sequence says on one key picture (C3-C5, C10), for a sweep over
  colour codes or efforts. The rules run on NVENC too: C9 checks NVENC's
  buffer target; pixel buffers (C11) stay Apple's.
- **`Nv12Writer` writes a buffer's rows** (`testing/nv12_writer.hpp`), as
  NVENC's targets are, the bytes past each row up to its stride 0.

### Removed

- **`HevcDecoder::Options::configure_ffmpeg_logging`**: the first coder that
  opens on NVIDIA sets FFmpeg's log level to ERROR, once a process. No
  caller in the family set the option.

### Changed

- **NVDEC opens FFmpeg's CUDA device as NVENC does**, through one internal
  helper on the Vulkan device's GPU; its pictures are unchanged.
- **The colour-code sweep** in `sensor_video_hevc_encoder` runs the
  sequence's rules, on one key picture each.
- **CI**: the 26.04 leg builds FFmpeg and CUDA and promises NVDEC and NVENC;
  the 24.04 leg promises NVENC too.

## [0.33.0] - 2026-11-03

### Added

- **`HevcEncoderCase::rules`** (`testing/hevc_conformance.hpp`): `All`, the
  default, runs every rule as before; `Stream` runs only those a session's
  options change (C2-C6, C10, C11), for a sweep over colour codes, QPs or
  sizes. A caller that sets no `rules`, as the iOS self-test, is unchanged.

### Changed

- **The encoder tests are fast.** `sensor_video_hevc_encoder` runs its rules
  on 256x144 pictures (and 258x146, a 4n+2 width), 6 pictures a case, and its
  colour-code sweep on the stream's rules alone: 3.0 s in Release (was
  25.6 s) and 16.2 s under ThreadSanitizer (was 163-178 s) on an M5 Max.
  Full-size coding stays in the benchmarks and the iOS self-test.
- **C13b holds its writer's value instead of waiting 1 s.** It holds ten times
  one picture's whole path, as the run measures it (at least 20 ms); nothing
  may come back while the value is held, and the access unit comes back once
  the run sets it. A watchdog sets the value 5 s past the hold, so a coding
  thread that never waits fails the rule rather than hanging the run.

## [0.32.0] - 2026-11-03

### Changed

- **`order_breach`** (`testing/rgbd_contract.hpp`) takes a time of 0 as
  none: the sequence is still checked, the time only where both are
  non-zero. `RgbdContract` calls it with the last non-zero time, as
  before, and its answers are unchanged.
- **The stereo contract** (`tests/stereo_contract.hpp`) takes a time of 0
  as none too: a pair's sequence is still checked, and its time, where it
  has one, on the last one reported. DECISIONS.md, the 2026-11-02 frame
  contract entry, amended.

## [0.31.0] - 2026-11-03

### Fixed

- **A GPU fault leaves no reader waiting for good.** `HevcDecoder` (NVDEC)
  returns every error from `send` or `receive` but `InvalidArgument` once
  each ready value it handed out is set, from the host where a failed
  stream dropped CUDA's signal; with three pictures kept, `receive`'s host
  wait for the oldest asks the stream between 10 ms slices, so a failed one
  returns CUDA's error, settled, and one that runs dry with the value unset
  a Backend error, rather than waiting for good. Each slice that blocks
  counts as one of the device's `host_waits`. DECISIONS.md, the 2026-11-03
  fault entry.
- `sensor_video_handover`'s CUDA table drops ready values' signals and
  reports a faulted stream, as a GPU fault leaves them. Under a validation
  layer older than 1.4.341, which reports a host signal after a read of an
  exported timeline (VUID-VkSemaphoreSignalInfo-value-03259), those tests
  run in a second, unvalidated run.

## [0.30.0] - 2026-11-02

### Added

- **`HevcEncoder::target(batch)`** and **`send(key, pts, written)`**
  (`video/hevc_encoder.hpp`): the encoder hands out its next picture on the
  device, NV12 R8 and R8G8 storage images over a picture of VideoToolbox's
  pool, described by the session's colour, for the caller's kernel to write
  in its own batch; `send` hands over the timeline value that batch sets,
  returns at once, and the coding thread waits for that value on the host
  before it codes. A second `target` before `send` gives up the first once
  it succeeds; a refused `send` keeps the target. The images are kept for
  the batch that writes them across a lost session too. `create` readies
  the targets' pictures (one throwaway picture coded first, as VideoToolbox
  replaces its pool at its first), so a producer's targets wait for no
  batch; one on a picture new to the session, as after a reopen, waits on
  the host for the device's earlier work, without the encoder's lock. A
  session whose pool is gone, as after the system reclaims it, fails the
  target (`IoError`) and is lost. DECISIONS.md, the 2026-11-02 target
  entry.
- `HevcEncoder::Options::device` and `Options::allocator`, and
  **`hevc_encoder_device_requirements()`**: `device_requirements()` plus
  `shaderStorageImageExtendedFormats`, which `create` checks of a device it
  is given.
- **`PixelBufferImporter::Access`** (`apple/pixel_buffer_importer.hpp`):
  `Write` imports planes a compute shader stores into as well (`STORAGE`
  and `TRANSFER_DST` usage, Metal textures written in place), refused where
  the device has no R8 or R8G8 storage images; `Read`, the default, is
  unchanged.
- **Test support:** `testing::Nv12Writer` (`testing/nv12_writer.hpp`,
  `sensor_video_testing`): a compute kernel that stores an NV12 test picture
  into plane images in the caller's batch, as a producer's kernel writes a
  target. `hevc_encoder_breach` runs on targets when the case is not
  `pixel_buffers`, and checks C9 (the target, taken twice) and C13b (the
  coding thread waits for each target's writer).

### Changed

- The encoder's destructor waits for the writer of each target sent, with
  no limit, as a pending batch does; `send_pixel_buffer` refuses while a
  target is taken and not sent.
- `HevcEncoderCase::device` and `allocator` also write the targets; a case
  that is not `pixel_buffers` runs on targets rather than being refused.
- The encoder test runs on a device made with
  `hevc_encoder_device_requirements()` on every leg, the test writer first.

## [0.29.0] - 2026-11-02

### Added

- **`HevcEncoder`** (`video/hevc_encoder.hpp`, `sensor_video`), the twin
  of `HevcDecoder`: one hardware H.265 Main session in one low-latency
  configuration, a slice QP for P-pictures and one for IDRs fixed for the
  session, keys only when asked, the VUI from `Options::color` normalized
  and checked against one code set. Pipelined: a send returns at once, a
  coding thread of its own codes (allocating nothing; user-interactive
  quality of service on Apple), and `receive(picture, timeout_ns)` hands
  back each Annex B access unit in the order sent, swapping buffers, at most
  `kInFlight` (2) in flight; a finite wait sleeps even with none in flight.
  `key_owed()`, `lost()`, `facts()` and
  `inject(Fault)` as the header says. On VideoToolbox in every Apple build;
  `create` refuses on Linux until NVENC lands. DECISIONS.md, the
  2026-11-02 encoder entry.
- **`send_pixel_buffer(encoder, pixels, key, pts)`**
  (`video/hevc_encoder_apple.hpp`, Apple): a camera's `CVPixelBuffer` coded
  as it is, with no Vulkan device.
- **`video_backend()`** (`video/backend.hpp`): `VideoToolbox` or `Nvidia`.
- **`color_codes_of(pixels)`** (`apple/core_video_color.hpp`,
  `sensor_apple`): a `CVPixelBuffer`'s colour tags as H.273 codes, through
  Core Video's own mapping.
- **Test support:** `testing::SliceReader` (`testing/hevc_slices.hpp`,
  `sensor_video_syntax`): each picture's first slice type and QP and each
  SPS's profile and sub-layers; `hevc_encoder_breach`
  (`testing/hevc_conformance.hpp`, new static target
  `sensor_video_testing`): the encoder rules C1-C13 on one case;
  `hevc_encoder_promised()` (`VS_TEST_HEVC_ENCODER`);
  `testing/nv12_pattern.hpp`, the host half of `nv12_picture.hpp`, with
  `pattern_index`, the band's reader; and on Apple
  `testing/nv12_pixel_buffer.hpp`'s `pixel_buffer`.
- `tests/data/hevc/vt_pattern_1920x1440.h265`, coded by VideoToolbox's
  encoder (`tools/vt_pattern_clip`): the test pattern, 12 pictures, keys at
  0 and 6. The decoder test decodes it on NVDEC and VideoToolbox to
  libavcodec's MD5.

### Changed

- `sensor_apple` links `sensor_video_syntax` PUBLIC (for `ColorCodes`);
  `sensor_video` links CoreVideo PUBLIC on Apple (the encoder's Apple entry
  names `CVPixelBufferRef`).
- `testing/nv12_picture.hpp` includes `testing/nv12_pattern.hpp`, where
  `HostNv12` and `test_pattern` now live, unchanged, with the band's layout
  from `testing/index_band.hpp`.
- The H.265 parsers read through internal walks the slice reader shares;
  what they read and refuse is unchanged.
- The iOS build check also compiles `sensor_video_testing` and the Apple
  headers as plain C++, and links an app's encoder calls into an iOS
  executable.

## [0.28.0] - 2026-11-02

### Added

- **A picture's ready value** (`YuvImage::ready`, `YuvImage::ready_value`):
  the timeline value a writer that did not wait for its work sets once the
  picture is written, which a reader's submission waits for on the device.
  NVDEC's and nvJPEG's pictures carry one; VideoToolbox's, a camera's and
  host planes none. DECISIONS.md, the 2026-11-02 hand-over entry.
- **`ready_points(frame)`** and **`ready_points(frames)`**
  (`prep/gpu_frame_prep.hpp`, `sensor_prep`): the values a batch that reads
  a frame's colour, or a set's, waits for, for a caller of
  `GpuFramePrep::record`, `record_set` or `SensorArray::record_process`
  (`ready_points(set.frames)`) to pass among its `submit_async` waits.
- **`add_ready_point(image, points)`** (`frame/ready_points.hpp`,
  `sensor_frame`): a picture's ready value added to a submission's waits,
  one a timeline at its highest value, for a reader of its own.
- `device_requirements()` names `VK_KHR_external_semaphore_fd` among its
  optional extensions.

### Changed

- **NVIDIA's decoders hand their pictures over on the GPU.** `HevcDecoder`
  (NVDEC) and `JpegDecoder` (nvJPEG) queue the copy or decode, signal an
  exported timeline CUDA imported after it, and hand the picture out with
  that value, in place of `cuStreamSynchronize` and the event wait.
  `HevcDecoder` keeps each frame FFmpeg handed out until its value is
  reached, whoever holds the picture, and up to three: with three kept, the
  next `receive` waits on the host for the oldest. FFmpeg's own copy of
  NVDEC's output stays (dropping it costs about 100 MB more a 4K decoder).
  On NVIDIA `create` is `Unsupported` on a device that does not export
  semaphores (`Device::exports_semaphores`).
- **Every reader here waits for a picture's ready value on the device:**
  `GpuFramePrep::prepare` and `prepare_batch`, `PictureReadback`,
  `LumaPass` and `StereoRectifier`; each refuses a ready timeline made on
  another device (`InvalidArgument`). **A caller that records
  `GpuFramePrep` into its own batch, or reads a decoder's picture in its
  own, must pass the picture's ready value among its waits**
  (`ready_points`, `add_ready_point`), or it reads an NVIDIA picture CUDA
  may not have written yet.
- The core is pinned at 0.41.0 (core #88: `create_exported_timeline`,
  `Device::exports_semaphores`, `fwd.hpp`'s `TimelineSemaphore`), and every
  build but a camera-only one needs it.

## [0.27.0] - 2026-11-02

### Added

- **An RGB-D sensor's frames as bytes** (`wire/rgbd_wire.hpp`, the new
  target `sensor_wire`, every build, iOS included), version 1, for a
  recording or a live link of any `IRgbdSensor`:
  - `encode_sensor` / `decode_sensor`: a `SensorInfo` as a depth track's
    params, less the id, clock and rate a container's track holds;
  - `encode_depth` / `decode_depth`: a frame's depth half as a record, a
    392-byte prefix (the depth camera, `color_to_world`, `depth_to_color`,
    units, range, noise, the confidence flag and floor) then the rows and
    the plane, raw. The encode borrows the frame's rows and plane
    (`DepthRecord`); the decode returns an `RgbdFrame` pointing into the
    payload, which its `pixels` holds;
  - `encode_color_params` / `decode_color_params` (`ColorCodec::HevcMain8`)
    and `encode_color` / `decode_color`: a frame's colour half as a
    128-byte prefix (the colour camera, its `ColorCodes`, its exposure)
    before one coded picture, read in place (`ColorRecord`).
  The decode checks the layout and passes every value as written; another
  version is `Unsupported`. DECISIONS.md, the 2026-11-02 wire-form entry.
- **`ColorExposure` and `RgbdFrame::color_exposure`** (`rgbd_frame.hpp`):
  a picture's exposure time, ISO and bias as the camera reported it, all 0
  where the driver does not say; the colour record carries it.
- `tools/ios_build_check.sh` compiles `sensor_wire` too.

## [0.26.0] - 2026-11-02

### Added

- **The frame contract**, in `IRgbdSensor`'s doc (`rgbd_sensor.hpp`): a
  frame's colour is its own capture's picture; a sensor that names a depth
  or colour camera hands out whole frames only, a lost or undecoded picture
  counted in `SensorStats::failed`; a frame's cameras are the sizes
  `SensorInfo` names; a gap in `RgbdFrame::sequence` is a frame lost after
  the sensor numbered it, and a rate gate's skips leave none
  (`RgbdFrame::sequence`'s doc says so too); a timestamp of 0 is none, so
  only reported times increase; a confidence plane comes beside host depth
  at a floor of 1 or more, and a floor only with one, as `GpuFramePrep`
  takes it (0.24.0's plane). DECISIONS.md, the 2026-11-02 frame contract
  entry.
- **`color_from_registered_depth(depth, width, height)`**
  (`camera_conventions.hpp`, `sensor_interface`): the colour camera a depth
  image is registered to, a `camera::CameraModel` in and out, the inverse
  of `depth_from_registered_color`'s rescale through the same per-axis
  code, the lens kept; `InvalidArgument` for a zero size or a depth camera
  `check_camera_model` refuses.
- **Test support** (`testing/`, header-only):
  - `testing/rgbd_contract.hpp`: `RgbdContract`, which checks a source's
    frames one at a time, one start each, and keeps none, `rgbd_breach`
    over a vector, and the rules they share with the stereo contract,
    `sequence_breach`, `order_breach` and `stats_breach`.
  - `testing/rgbd_pattern.hpp` (standard library alone): `PatternDepth`,
    `pattern_depth(width, height, index)`, u16 millimetres carrying the
    index band at the picture's pixels and a confidence plane, and
    `depth_index`, which reads it back; `kPatternNearMm`, `kPatternFarMm`.
  - `testing/index_band.hpp` (standard library alone): `kIndexSquare` and
    `kIndexBits`, moved from `nv12_picture.hpp`, which includes it;
    `test_pattern`'s bytes are unchanged.

### Changed

- `tests/stereo_contract.hpp` calls `order_breach` and `stats_breach`; the
  stereo suites are unchanged.
- The Orbbec camera stub stands in for `CameraStream`'s mailbox, take and
  read, playing a script of captures with counts of its own, and
  `tests/realsense_camera_stub.cpp` stands in for the RealSense camera
  (`camera.cpp`) with librealsense's software device: the Orbbec,
  RealSense and FFS depth sensors' tests run `rgbd_breach` under injected
  faults through each sensor's poll, drain and stats. The cameras' own
  counting is checked only on a camera.

## [0.25.0] - 2026-11-02

### Added

- **A tracked sensor's registration** (`camera/tracked_registration.hpp`,
  `sensor_camera`, every build): `TrackedRegistration`, the one rigid
  `tracking_to_world` that places a tracked sensor's world (ARKit's) in a
  sensor array's world for one capture session's world epoch, with the
  array's world it registers into (one sensor or one tag) and the tag fit
  that solved it (`rms_px`, `views`, `edge_m`). A frame posed
  `cam_to_tracking` sits at `tracking_to_world * cam_to_tracking`.
  `validate_`, `parse_`, `read_`, `format_` and `write_tracked_registration`
  read and write the file, `"volumetric_kit.tracked_registration"` version
  1, through the core's `read_file` and `write_file`; the session is 32 hex
  digits, byte 0 first. calib's `register_tracked` writes it; a tracked
  source reads it at open. DECISIONS.md, the 2026-11-02 tracked
  registration entry.

### Changed

- The array calibration file's world block, its transforms, and the JSON
  numbers, objects and text under them are read and written in one internal
  place (`camera/calibration_json.{hpp,cpp}`), which the registration
  shares, and so is the check that a world is one sensor or one tag. The
  file reads, refuses and writes as before, every message the same.

## [0.24.0] - 2026-11-02

### Added

- **A depth confidence plane** (`RgbdFrame::depth_confidence`,
  `RgbdFrame::min_confidence`): beside host depth, one byte a sample in the
  sensor's own levels (ARKit's 0 low, 1 medium, 2 high), and the floor below
  which `GpuFramePrep` drops a sample on the GPU, per raw sample, before the
  depth offset, the mix, the colour-view test and the outlier filter, so a
  dropped sample is a hole. Level 0 is never integrated: a producer that
  invalidates a sample, such as one it masks as a moving person, writes 0
  there, and every floor drops it. A plane comes with a floor of 1 or more,
  which its producer sets, and is copied into the pass's slot after the
  depth; a frame without a plane is prepared bit for bit as before.
  `prepare` refuses a plane beside device depth or at a floor of 0, and a
  floor without a plane (`InvalidArgument`). DECISIONS.md, the 2026-11-02
  confidence plane entry.
- `sensor_gpu_frame_prep_confidence`: the gate at every floor, per raw
  sample, the slot, sets and the refusals.

## [0.23.0] - 2026-11-02

### Added

- **What H.273 colour codes mean, public** (`video/frame_color.hpp`,
  `sensor_video`): `resolve_matrix(matrix, height)`,
  `resolve_encoding(transfer, primaries)` and
  `resolve_chroma_location(code)` give a `ColorCodes`' `VideoColorMatrix`,
  `ColorEncoding` and `ChromaLocation`, as the decoders describe their
  pictures by them, for any caller holding codes (an SPS's, a capture
  source's tags). They resolve as they did inside; `resolve_matrix` takes
  the height as `std::uint32_t`. DECISIONS.md, the 2026-11-02 colour codes
  entry.
- **Test support** (`testing/`, header-only, no test framework):
  - `testing/hevc_clip.hpp`: `split_access_units` and
    `access_units_of_file`, a clip's access units at its delimiters, and
    `hevc_decoder_promised()`, whether `VS_TEST_HEVC_BACKEND` promises a
    hardware decoder.
  - `testing/nv12_picture.hpp`: `HostNv12`; `test_pattern(width, height,
    index)`, the family's NV12 test picture, whose index band names
    `index % 16384` in 16 x 16 squares beside a 255 and a 0 reference
    (`kIndexSquare`, `kIndexBits`); and `device_picture(device, allocator,
    picture, pad)`, the picture as R8 and R8G8 plane images in `GENERAL`,
    as a hardware decoder hands one out, written through the core's
    `core_test_policy`.
- `tests/data/hevc/centred_full_256x144.h265` (`tools/make_hevc_fixtures.sh`):
  testsrc2 at BT.709 full range, its chroma sited at the centre, an IDR
  every 4. The decoder test decodes it to libavcodec's MD5 on both paths and
  checks its range and siting; the syntax test reads its SPS.

### Changed

- `sensor_video` links `sensor_video_syntax` PUBLIC (was PRIVATE):
  `frame_color.hpp` resolves the codes the syntax reader returns.
- The Metal device under a Vulkan device is found in one place,
  `sensor_apple`'s internal `metal_device.mm`, which `PixelBufferImporter`
  and the importer's test both call; the importer behaves as before.
- The package's tests are on the test support: `tests/hevc_fixture.hpp` is
  gone, `decoder_test.hpp`'s promise is `hevc_decoder_promised()`, and the
  frame prep, readback and luma tests make their NV12 plane images with
  `device_picture` (the readback and luma tests' 3 columns wider and 3 rows
  taller, where they were 3 and 2).
  The frame colour test calls the public resolvers and still runs on every
  leg, with or without the decoders.

## [0.22.0] - 2026-11-02

### Added

- **`KeyGate`** (`video/key_gate.hpp`, `volumetric_kit::sensor_video_syntax`,
  every build): which access units of a predicted H.265 stream to send after
  a loss. `admit(seq, key)` is `Skip` while it waits for a key (at the
  start, after a number other than one up from the last, and after
  `lose()`), `Restart` for the key that ends a wait (reset the decoder, then
  send it), and `Take` otherwise; an empty `seq` is an unnumbered unit,
  never a gap. `waiting()` says whether it waits.
- **`hevc::first_slice_type(data, size)`** (`video/hevc_syntax.hpp`): the
  nal_unit_type of an access unit's first slice segment, found by start
  codes alone, with no allocation and no failure; empty if there is none.
  With `is_irap`, whether the unit is a key.
  DECISIONS.md, the 2026-11-02 key gate entry.

### Changed

- The Orbbec driver's H.265 colour decoder uses them: its internal gate and
  `is_key_frame` go, its behaviour unchanged, and
  `sensor_orbbec` links `sensor_video_syntax` privately. Their unit cases
  move to `sensor_video_key_gate` and `sensor_video_hevc_syntax`.

## [0.21.0] - 2026-11-01

### Added

- **`LumaPass`** (`prep/luma_pass.hpp`, `sensor_prep`): every picture of a
  list made into its 8-bit luma and its half on the GPU, in one batch and
  one submit, from any layout a driver hands out -- V210 (its 10-bit Y
  rounded once, `(code + 2) / 4`), NV12, I420, YUYV and luma alone, in a
  buffer or on the host, and NV12 plane images. Each `DeviceLuma` holds its
  `full` and `half` (`(w / 2) x (h / 2)`, each the rounded mean of a 2x2
  block) as `YuvLayout::Luma` pictures in a pooled device buffer, reused
  once no copy holds it, and CONCURRENT across the queue families
  `create` is given. DECISIONS.md, the 2026-11-01 luma entry.
- `PictureReadback::read(pictures, planes, metrics)`: a list of pictures
  read in one submit, a view each; pictures that share a buffer, as a
  `DeviceLuma`'s two do, are taken over once.

### Changed

- `PictureReadback` refuses pictures in one buffer that name two writers'
  queue families (the buffer is taken over once), and its V210 refusal
  (`Unsupported`, as before) names `LumaPass`.

## [0.20.0] - 2026-11-01

### Added

- **A stereo pair's rectification and a lens's fold, on the host**
  (`camera/stereo_rectification.hpp`, `volumetric_kit::sensor_camera`, so
  camera-only builds have them):
  - `rectify_stereo(pair, size, alpha)`: Bouguet's rectification, as
    `StereoRectifier::create` computes it, now public. It returns a
    `StereoRectification`: the `StereoRig` (`R1` is its
    `rectified_to_left`'s rotation, transposed) and `right_to_rectified`,
    OpenCV's `R2`. `InvalidArgument` for a bad camera, cameras of different
    sizes, a `left_to_right` that is not rigid, an alpha or size out of
    range, a coincident, vertical or swapped pair, or a degenerate result.
  - `fold_radius2(distortion)`: where a lens's radial model folds, the
    squared radius the rectifier cuts its views at, now public.
  DECISIONS.md, the 2026-11-01 host rectification entry.

### Changed

- `StereoCalibration` (was `stereo/stereo_sensor.hpp`'s), `StereoRig` (was
  `stereo/stereo_rectifier.hpp`'s) and `stereo_calibration` (was
  `sensor_stereo`'s) move to `camera/stereo_rectification.hpp` and
  `sensor_camera`, names and namespace unchanged. The stereo headers
  include it, and `stereo/stereo_calibration.hpp` forwards to it, so every
  include compiles as before; `sensor_stereo` links `sensor_camera`, so no
  consumer's link changes.
- `StereoRectifier::create` calls `rectify_stereo` and `fold_radius2`: its
  rigs and views are bit-identical. Its refusals of the pair's sizes,
  `left_to_right`, alpha and the size now read
  `StereoRectifier: rectify_stereo: ...`, with the same codes.
- `StereoInfo::calibration` states the rule for every source: both cameras
  0 x 0 is a source opened without a calibration, which `StereoRectifier`
  and `StereoDepthSensor` refuse; in a source opened with one, neither
  camera has a side 0. `tests/stereo_contract.hpp` checks it on every
  driver's unit (`Opened`, `calibration_breach`).
- CI: the 24.04 leg builds the DeckLink driver's SDK part
  (`VS_WITH_DECKLINK`) against the SDK's headers the core keeps, checked out
  at the core pin and held to their SHA256s, and runs its SDK smoke test.
- The core pin moves to core #85's merge (`1972079`, 0.39.0), whose
  `tools/ci/decklink-sdk-16.0` holds those headers; the floors are
  unchanged, so nothing changes for a consumer.

## [0.19.0] - 2026-11-01

### Added

- **`DecklinkStereo`** (`decklink/decklink_stereo.hpp`,
  `volumetric_kit::sensor_decklink`, `VS_WITH_DECKLINK`, Linux x86-64): two
  SDI inputs of a DeckLink 8K Pro as one `IStereoSensor`. Each camera's
  10-bit V210 picture is written by the card into host pages the device
  imports and copied once into a device buffer of its own; free-running
  cameras are paired by the card's frame times within `max_skew_ns`, each
  pair stamped on the host's clock with its skew. `Options::calibration` is
  optional, for the raw pairs a calibration is made from. Needs the
  DeckLink SDK's headers (`DeckLinkSDK_ROOT`, `cmake/vs_decklink.cmake`)
  and, to capture, Desktop Video 16.0 or newer, loaded at run time.
  DECISIONS.md, the 2026-11-01 DeckLink entry.
- `decklink_skew_report` (with `VS_WITH_DECKLINK`): the two inputs' frames,
  losses, rates and the pairs' skew and drift, by hand.
- `device_requirements()` asks for `VK_EXT_external_memory_host` where
  offered, which `DecklinkStereo` needs.

### Changed

- `StereoRectifier::create`, and so `StereoDepthSensor::create`, refuses a
  source with no calibration (`StereoInfo::calibration` both cameras
  0 x 0) with `InvalidArgument`, naming it.
- The core is pinned at 0.38.0 (core #82, `create_imported_host_buffer`).
  0.38.0 is the floor where the DeckLink driver's units build
  (`VS_WITH_DECKLINK`, or the tests); elsewhere the floor stays 0.29.0, so
  a decoder-only consumer at an older core still builds.

## [0.18.0] - 2026-11-01

### Added

- **FFS's near gate.** `FfsStereo::Options::max_disparity`, the network's
  disparity search range at the rig's size (192 for the exported
  23-36-37 checkpoint): depth below `1.05 * fx * baseline / max_disparity`
  is no return, as nearer surfaces come back saturated at about that
  depth, and the frame's `min_depth` is the bound kept. 0, the default, is
  no gate; a gate not below `max_depth` is refused.

### Changed

- **The stereo rectifier prefilters where it shrinks a picture.** A view
  pixel whose footprint in the source (from where the points half a pixel
  either side of it land) is longer than 1.5 source pixels on a side is
  the mean of 5 x 5 bilinear samples spread over that footprint, Y'CbCr
  averaged and converted once, so a 4K picture shrunk 4.5:1 to 960 x 544
  does not alias, and a fisheye's edge, shrunk less than its centre, is
  averaged over less. Smaller footprints, a D455's 1.33:1 among them, take
  one sample as before. DECISIONS.md, the 2026-11-01 prefilter entry.

## [0.17.0] - 2026-11-01

### Added

- **A stereo frame's skew and its source's bound.** `StereoFrame::skew_ns`,
  the right picture's capture time less the left's (`timestamp_ns`), and
  `StereoInfo::max_skew_ns`, the most any two pictures of a frame lie apart:
  frames past it are counted failed, never handed out. `RealSenseStereo`
  states its two infrared stamps' difference, 0 on a D455 (one exposure),
  and half a frame, the bound its framesets keep.
- **The right picture's own owner.** `StereoFrame::right_pixels`, for a
  source whose right picture is apart from `pixels`.
- **`stereo_calibration(file, left_id, right_id)`**
  (`stereo/stereo_calibration.hpp`): the `StereoCalibration` two entries of
  a calibration file make, each camera posed with a colour model,
  `left_to_right = inverse(right.color_to_world) * left.color_to_world`;
  `InvalidArgument` for a file the validator refuses, one id twice, an id
  missing, unposed or without colour, cameras at one place, or lenses of
  different sizes.
- `tests/stereo_contract.hpp`: the sync contract every stereo source's
  frames keep, one check for a test camera and each driver's unit.

### Changed

- **BREAKING: `StereoRectifier::rectify` lets the right picture go** once
  its pass has read it: `RectifiedPair::source` has no right planes and no
  `right_pixels`. `StereoDepthSensor` lets the camera's frame go once
  rectified, so the right picture is free before FFS runs and a depth frame
  holds the left picture alone. DECISIONS.md, the 2026-11-01 stereo-sync
  entry.
- A RealSense frameset is one moment when every two of its pictures lie
  within half a frame, in the whole-nanosecond stamps its frames carry (each
  was held to the first, in milliseconds), so `RealSenseStereo`'s frames
  keep their bound exactly. Both drivers share the rule; a D455's framesets
  pass as before.

## [0.16.0] - 2026-11-01

### Added

- **V210 pictures (SDI's 10-bit 4:2:2).** `YuvLayout::V210`, one plane of
  words, six pixels in four, as a DeckLink card captures it: `GpuFramePrep`
  and `StereoRectifier` read it from the host or a buffer at 10 bits and
  round to their 8-bit outputs once. Limited range only, of even width,
  rows `ceil(width / 6) * 16` bytes or more, offset and stride whole words;
  `PictureReadback` refuses it (`Unsupported`) until it reads it.
  DECISIONS.md, the 2026-11-01 V210 entry.

## [0.13.0] - 2026-10-31

### Changed

- **BREAKING: frame prep drops depth outliers by default.**
  `DepthOutlierFilter::enabled` is `true` by default, so every
  `GpuFramePrep`, and every pass of a `SensorArray`, filters depth (8 px,
  20 points) unless its config sets it `false`; the refusals are unchanged.
  DECISIONS.md, the 2026-10-31 default-on entry.

## [0.12.0] - 2026-10-31

### Added

- **A depth outlier filter in frame prep.**
  `GpuFramePrepConfig::depth_outliers` (`DepthOutlierFilter`, off by
  default): after undistortion and the colour-view test, a return is kept
  where at least `min_points` returns, itself included, lie strictly within
  `radius_px` pixel footprints at its own depth (`radius_px z / fx` metres)
  of its point, and is 0 elsewhere; 8 px and 20 by default. On, it costs
  one more dispatch (`remove_depth_outliers.comp`) and one more depth-sized
  buffer per pass, and `GpuFramePrep::create` refuses a `radius_px` that is
  not finite in [1, 32] or a `min_points` below 2. A `SensorArray` applies
  its `Options::prep` filter to every sensor. DECISIONS.md, the 2026-10-31
  depth-outlier entry.

## [0.11.0] - 2026-10-30

### Added

- **Audio sources.** `IAudioSensor` (`audio_sensor.hpp`, in
  `sensor_interface`): any audio source, polled as `IRgbdSensor` is; `drain`
  appends every `AudioChunk` held, oldest first, each interleaved signed
  16-bit samples stamped with its first sample's capture time
  (`AudioInfo::clock`) and numbered since the start; a running source that
  delivers nothing for 1 s fails its drains (`IoError`). DECISIONS.md, the
  2026-10-30 audio entry.
- **`Microphone`** (`audio/microphone.hpp`, `sensor_microphone`, new option
  `VS_WITH_PORTAUDIO`, off by default): any OS input through PortAudio
  (Core Audio, ALSA), a prerequisite found through pkg-config, never
  fetched; `Microphone::inputs()` lists the inputs and `overflows()` counts
  input the backend lost. `tools/mic_check` checks a mic by hand, built
  with the tests.

## [0.10.0] - 2026-10-30

### Added

- **A frame states its depth's noise.** `DepthNoise {a, b, c}`
  (`frame/depth_noise.hpp`, in `sensor_frame`): the frame-to-frame standard
  deviation of one depth sample, `sigma(z) = a + b z + c z^2` metres;
  `known()`, and `DepthNoise::stereo(sigma_d, f, B)` for rectified stereo,
  `c = sigma_d / (f B)`. All zero, the default, means unknown.
  `RgbdFrame::depth_noise` is the driver's, and `GpuFramePrep` copies it to
  the new `DeviceFrame::depth_noise` unchanged. Both default to unknown, so
  no source breaks; a consumer that needs a model refuses a frame without
  one (DECISIONS.md, the 2026-10-30 depth-noise entry).
- Every producer states its own: `OrbbecSensor` the Femto Mega's 0.95 mm per
  metre (`{0, 0.95e-3, 0}`) in every depth mode; `RealSenseSensor`
  `DepthNoise::stereo(0.13, fx, B)` from the depth camera's fx and the
  factory baseline it now reads at open; `FfsStereo` and
  `StereoDepthSensor` `DepthNoise::stereo` of the new
  `FfsStereo::Options::disparity_sigma` over the rectified rig, which
  defaults to 0, unknown: the caller states the camera's measured value.
  `FfsStereo::create` refuses a `disparity_sigma` that is not finite and at
  least 0.

### Fixed

- The `FfsStereo` and `StereoDepthSensor` examples call
  `GpuFramePrep::create(device, allocator)`, as its signature is.

## [0.9.0] - 2026-10-29

### Changed

- **BREAKING: a frame with colour keeps only what both its cameras see, by
  their fields of view.** `GpuFramePrep` makes every colour pixel whose
  direction lies outside its depth camera's view a 0 word (coverage 0), and
  zeroes every depth pixel, from the host or the device, whose direction lies
  outside the colour camera's view or whose point lands on colour it drops;
  the view test leaves the baseline out, and the depth range plays no part.
  A frame without colour keeps all its depth.
  `GpuFramePrepConfig::depth_within_color` is gone: a consumer that set it
  drops the line, and one that left it off gets the mask (DECISIONS.md, the
  2026-10-29 same-view entry).

## [0.8.0] - 2026-10-29

### Added

- **Stereo depth by Fast-FoundationStereo, `sensor_ffs`** (`VS_WITH_FFS`,
  Linux, NVIDIA only; needs `VS_WITH_CUDA` and TensorRT ≥ 11, found by
  `cmake/vs_ffs.cmake`, never fetched). `FfsStereo` (`ffs/ffs_stereo.hpp`)
  runs an FFS ONNX export on TensorRT, its plan built once and cached by
  GPU, TensorRT version, model and size: a CUDA kernel makes the network's
  input from a `StereoRectifier`'s views, read in place, and another turns
  its disparity into 16-bit depth written straight into a Vulkan buffer,
  the frame's `RgbdFrame::depth_device`, with no return where an `Rgba8`
  view saw nothing; the host sleeps through the network run.
  `StereoDepthSensor` (`ffs/stereo_depth_sensor.hpp`) is any
  `IStereoSensor` as an `IRgbdSensor`, rectifying and matching on a thread
  of its own that the camera's frame notification wakes; it reports the
  camera's identity, so a `SensorArray` takes it, and the camera's drops.
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
