# Sensor package design

The architecture and implementation reference for `volumetric_kit_sensor`.
[AGENTS.md](AGENTS.md) holds the working rules;
[DECISIONS.md](DECISIONS.md#decision-index) holds the dated rules,
rationale and measurements. Read the sections the task needs, and update
them with the code when a contract changes. The text moved here from
volumetric_kit_recon's DESIGN.md with the package (2026-10-29).

- [Tiers](#tiers) and [names](#names)
- [Colour at the boundary](#colour-at-the-boundary)
- Implementation: [camera](#camera), [frame](#frame), [the interface and
  the tiers on it](#sensor), [stereo](#stereo), [stereo depth
  (FFS)](#stereo-depth-ffs), [audio](#audio), [wire](#wire), [image
  I/O](#image-io)
- [Next work](#next-work)

## Tiers

The package sits on volumetric_kit_core alone, and nothing in it includes
recon or grid, its sibling on the core (`sensor_package_boundary`). recon's
libraries link the `frame` tier alone (`recon_library_sensor_links`);
recon's examples assemble the rest.

- **`camera`** — the camera vocabulary the drivers and the interface
  share: the double-precision `CameraModel` (image size, intrinsics, OpenCV's
  rational lens), the lens as a scalar template, rigid transforms, the
  sensor array's calibration file (`camera/array_calibration.hpp`), a
  tracked sensor's registration into its world
  (`camera/tracked_registration.hpp`), and a stereo pair, its
  rectification and a lens's fold, on the host
  (`camera/stereo_rectification.hpp`). No Vulkan (the 2026-10-06
  decisions), so a tool that solves or checks a calibration without a GPU
  links it alone.

- **`frame`** — the contract recon's fusion, texturing and sender read,
  header-only on the family core's Vulkan tier: `DeviceFrame` and
  `DeviceFrameSet` (`frame/device_frame.hpp`), the posed pinhole
  `DepthCameraParams`/`ColorCameraParams` (`frame/camera_params.hpp`), the
  depth noise each producer states (`frame/depth_noise.hpp`),
  `ColorEncoding`, `is_canonical` and the sRGB curve
  (`frame/color_space.hpp`), and the GLSL pinhole and sRGB curve it
  exports; beside them
  the package's GLM aliases and `device_requirements()`.
- **`sensor`** — the sensor *interface*: `IRgbdSensor`, the `RgbdFrame` it
  hands out as captured (each camera's `camera::CameraModel`, the pixels it
  holds), and the boundary conversions a capture integration gets silently
  wrong — camera conventions (pose handedness, registered-depth intrinsics)
  and colour (`to_canonical`). Reads `camera`'s models, and `frame`'s
  pinhole cameras and `ColorEncoding`, so it
  depends on **`camera` and `frame` alone** — it sits beside recon's fusion tiers,
  not on top of them — and bundles **no drivers**: one ships here only if this package can build
  *and* test it (the 2026-08-02 decision). The ones that do, Orbbec,
  RealSense and DeckLink, are targets of their own (`orbbec/`, `realsense/`,
  `decklink/`), so
  `sensor_interface` never links a vendor SDK. The HEVC and JPEG decoders and the HEVC encoder are another target of
  their own (`video/`, on the GPU's hardware) and link no camera tier:
  `frame` and, on Apple, the Apple importer (`apple/`), which links
  `frame` and the syntax tier. The H.265 syntax they read on the host, and the key gate
  beside it, are a target of their own as well, on the family core's base
  tier alone, in every build. The encoder rules any runner checks are a
  static target on `video` (`sensor_video_testing`).
  The GPU pre-processing, any picture's luma and the pictures' readback
  are a fourth (`prep/`, Vulkan and shaders), so `sensor_interface` itself
  stays free of all of them.
- **`array`** — several sensors read as one (`SensorArray`, `RigSets`),
  on the GPU pass.
- **`stereo`** — calibrated stereo pairs: what a stereo camera driver
  implements and hands out (`IStereoSensor`, `StereoFrame`), and
  `StereoRectifier`, the pair undistorted and rectified on the GPU for a
  stereo matcher, through the camera tier's rectification. On the
  interface; it links `prep` privately, whose colour pass it shares.
- **`decklink`** — two SDI inputs of a DeckLink card as one stereo camera
  (`DecklinkStereo`, `VS_WITH_DECKLINK`, Linux x86-64). On `stereo`; its
  SDK-free units are a static target of their own, built and tested on
  every leg.
- **`ffs`** — stereo depth on NVIDIA's GPUs (`VS_WITH_FFS`): `FfsStereo`,
  Fast-FoundationStereo on TensorRT with CUDA kernels either side, and
  `StereoDepthSensor`, a stereo camera as an `IRgbdSensor`. On `stereo`; it
  links `video` privately, for CUDA's view of Vulkan memory.
- **`audio`** — audio sources: `IAudioSensor` and its chunks
  (`audio_sensor.hpp`, in `sensor_interface`), and `Microphone`, an OS input
  through PortAudio (`sensor_microphone`, `VS_WITH_PORTAUDIO`), on the
  interface.
- **`wire`** — an RGB-D sensor's frames as bytes for a recording or a
  live link (`sensor_wire`, `wire/rgbd_wire.hpp`): the sensor once, each
  frame's depth half and colour half a record each, read back in place.
  On the interface and `video_syntax`'s colour codes, with no device code.
- **`io`** — image files (`sensor_image_io`, `io/image_io.hpp`): encoded
  images, depth PNGs and PNG export, on the family core's base tier alone,
  with stb private.

## Names

- Package: `volumetric_kit_sensor`; its version is `project()`'s, in
  `version.hpp` too.
- Namespace: `volumetric_kit::sensor`, the camera vocabulary in
  `sensor::camera`, the image files in `sensor::io`, the H.265 syntax in
  `sensor::hevc`.
- Headers: `include/volumetric_kit/sensor/<tier>/…`, the interface's
  directly under `volumetric_kit/sensor/`.
- Macros: `VS_` (`VS_PREP_API`, `VS_WITH_CUDA`); GLSL functions `vs_`.
- CMake targets `volumetric_kit::sensor_<tier>`: `sensor_camera`,
  `sensor_frame`, `sensor_interface`, `sensor_prep`, `sensor_array`,
  `sensor_stereo`, `sensor_video`, `sensor_video_syntax`,
  `sensor_video_testing`, `sensor_wire`, `sensor_apple`,
  `sensor_orbbec`, `sensor_realsense`, `sensor_ffs`, `sensor_microphone` and `sensor_image_io`; the umbrella `volumetric_kit::sensor` (real target
  `sensor`, which is why the interface's is `sensor_interface`). `io` and
  `audio` are the exceptions to `sensor_<tier>` and `<tier>/export.hpp`:
  their targets, export headers and macros are named after what they hold
  (`sensor_image_io`, `io/image_io_export.hpp`, `VS_IMAGE_IO_API`, as they
  were in recon; `sensor_microphone`, `audio/microphone_export.hpp`,
  `VS_MICROPHONE_API`).
- Package-wide headers: `version.hpp` at the root, built by `sensor_camera`
  so camera-only builds have it; `frame/device_requirements.hpp` and
  `frame/vector_types.hpp` in `frame`, the lowest tier on Vulkan and the one
  every tier past `camera` links (the camera tier names its double types in
  `camera/geometry.hpp`).

## Colour at the boundary

The family's colour rule is the 2026-08-02 colour-space decision: 8-bit
colour is encoded, float colour is linear, converted once at the sensor
boundary and encoded once at presentation. The working space is linear
BT.709 primaries with D65 white, and its canonical encoded form is the exact
sRGB curve. recon's DESIGN.md applies the rule to fusion, meshing and
presentation; this package holds the boundary.

### Declaring an encoding, not converting it

A driver states what it produces; it never converts. The declaration is a plain
value in `frame`, because it travels with a frame through every tier:

```cpp
// frame/color_space.hpp
struct ColorEncoding {
  enum class Transfer { Srgb, Bt709, Linear, Bt2020Pq };
  enum class Primaries { Bt709, DisplayP3, Bt2020 };

  Transfer transfer = Transfer::Srgb;
  Primaries primaries = Primaries::Bt709;
};
```

It rides **beside** the camera — a field on `sensor::RgbdFrame` and
`tsdf::ColorFrame` — and deliberately *not* inside `ColorCameraParams`. That
struct is uploaded verbatim to the fusion kernels under scalar block layout,
pinned at 88 bytes by `static_assert`s with GLSL mirrors in recon's `tsdf/shaders/`
and `texture/shaders/`; putting a field in it would spend a cross-tier shader ABI
change to carry something the kernel needs at most as a push constant. Both
frame structs already hold a `ColorCameraParams` by value, so a sibling field
costs nothing.

There is no `Range` field, and the omission is the point. By the time a frame
reaches this contract it is packed R'G'B', so the YCbCr matrix and any
limited-range expansion have already been applied — the contract requires
full-range R'G'B'. That conversion (de-planarize, the 601-vs-709 matrix, range
expansion) is the driver's, done before the contract, and it is as easy to get
silently wrong as the transfer function; what makes it the driver's is that it
is the one part that varies with the pixel format rather than with the color.
A field nothing consumes is a label free to drift. `Primaries` earns its place
because the working space gives it a consumer; `Range` would not have one.

ARKit declares `{Transfer::Bt709, Primaries::Bt709}` — spelled out, because
scoped enums do not name themselves from an aggregate initializer and this is
the line an out-of-tree driver copies. Its `capturedImage` is bi-planar YCbCr
that a BT.601 *matrix* converts to gamma-encoded R'G'B'; the matrix and the
transfer function are independent, and conflating them is its own silent
error. Its
primaries are already the working basis, so only the transfer is in question —
and BT.709's camera OETF differs from the sRGB EOTF by a couple of codes in the
toe. **`Transfer::Bt709` 8-bit content is accepted as sRGB rather than
converted**, which is what display pipelines do in practice (BT.1886) and what
keeps the common capture path a genuine no-op instead of a per-frame pass over
1920×1440 pixels. A bounded, stated error beats an unstated one. A depth camera
emitting linear 16-bit color declares `Transfer::Linear` and is decoded by
nothing.

Adding a sensor is then a declaration rather than a conversion, which is the
property that makes this scale: the failure mode of a new integration becomes a
wrong *label* — inspectable, and testable against a known patch — instead of a
bespoke curve buried in a platform driver.

The **type and the `is_canonical` predicate live with the curve in `frame`**,
not in the interface tier. recon's `TsdfIntegrator::integrate` refuses
a non-canonical frame rather than fusing it through the wrong curve, and recon's
libraries link `frame` alone; a property of `ColorEncoding` belongs with
`ColorEncoding`. `color_conventions.hpp` keeps what is genuinely the
boundary's: `to_canonical`, which walks a frame, and the cost and limits of
doing so. That refusal is what makes "convert once at the sensor boundary" a
contract rather than a hope — a mislabelled frame is an error, not a quietly
wrong reconstruction.

## Implementation reference

### camera

`sensor_camera` links `core_base` and GLM; the drivers, the interface and
calib build on it. It holds only what a caller uses (the second 2026-10-06
decision): unprojection and rescaling come with their first caller.

- **`camera/camera_model.hpp`** — `CameraModel`: `ImageSize`,
  `PinholeIntrinsics` and `RationalDistortion` (OpenCV's eight coefficients),
  all double, and `check_camera_model`. Conventions: +Z forward, +Y down,
  pixel centres at integers, intrinsics in pixels of the model's size.
- **`camera/projection.hpp`** — `distort_rational<T>`, the lens's forward
  model, which `prep`'s `lens.glsl` mirrors in float; the tests'
  reference.
- **`camera/geometry.hpp`** — GLM's double types, `check_rigid`,
  `rigid_inverse`, `nearest_rotation` (the polar factor, for an SDK's
  rotation that is not one), and Rodrigues both ways
  (`rotation_from_rodrigues`, `rodrigues_from_rotation`, accurate near 0 and
  pi). `RodriguesTransform` is OpenCV's `x' = R(rvec) x + tvec`.
- **`camera/color_controls.hpp`** — `ColorControls`, a colour camera's
  manual exposure (an exposure time and a gain in the camera's units, above
  0; auto exposure off, auto white balance on), which a rig's calibration
  records and a driver writes and reads back.
- **`camera/color_response.hpp`** — `ColorResponse`, what a colour camera's
  prepared colour is multiplied by in linear light so a rig's cameras agree:
  `gain[c] * F(r)`, `F = 1 + a q + b q^2`, `q = min(r, r_max)^2`, r the
  pinhole pixel's normalised radius (the 2026-10-31 colour-response
  decision); `validate` is the one check the file and `GpuFramePrep` share.
- **`camera/array_calibration.hpp`** — the sensor array's calibration file,
  version 2 under version 1's `device_calibration` key: per sensor by serial,
  an optional colour-camera pose, colour and depth `CameraCalibration`s
  (model and `factory` / `calibrated` source) and `depth_to_color`, and an
  optional world: a sensor at the origin, or a printed tag
  (`AprilTagWorld`, in OpenCV's marker axes, z up out of a floor tag).
  Version 3 adds a correction: a sensor's `depth_offset_m` (metres, + where
  its depth reads too far, at most 0.05 m), written only when it is not 0,
  or the array's `color_controls`, the colour cameras' manual exposure,
  written only when present; the file is version 3 only then, so an
  uncorrected calibration is written as version 2, byte for byte, and an
  older reader refuses a corrected one; a correction in a file that says
  version 2 is refused. Version 4 adds a sensor's `color_response`, beside
  the `color_controls` it holds at and only with them (its `fitted_at` must
  equal them), written only when it is not the identity, so a calibration
  without one keeps its version-2 or -3 bytes; a response in an older
  version is refused.
  `format_` / `write_array_calibration` write it
  through the core's `write_file`; version 1 is read for its poses.
- **`camera/tracked_registration.hpp`** — a tracked sensor's world (ARKit's,
  which it starts afresh each session and moves at each reset) placed in an
  array's world for one session's world epoch: `TrackedRegistration`, its
  sensor's id, the session's 16-byte id (never all zero) and the epoch, the
  array's world (one sensor or one tag, the array file's block), the rigid
  `tracking_to_world` in `depth_to_color`'s form, and the tag fit's RMS,
  stills and mean edge. A frame posed `cam_to_tracking` sits at
  `tracking_to_world * cam_to_tracking`; a reader refuses frames of another
  session or epoch, and the registration for an array of another world.
  The file is `"volumetric_kit.tracked_registration"` version 1, the
  session as 32 hex digits, byte 0 first, written lowercase and read in
  either case; it reads, checks and writes as the array file does. calib's
  `register_tracked` writes it, and a tracked source reads it at open.
  The two files' world block (read, written, and checked as one sensor or
  one tag), transforms, numbers and text live in one internal place
  (`calibration_json.{hpp,cpp}`); each file's public functions prefix its
  refusals with its name.
- **`camera/stereo_rectification.hpp`** — a stereo pair on the host, in
  `sensor`'s namespace as when it was the stereo tier's (the 2026-11-01
  host rectification decision): `StereoCalibration` (the two cameras,
  lenses included, and `left_to_right`, OpenCV's `R`, `T`) and
  `stereo_calibration(file, left, right)`, which makes one from two entries
  of a calibration file (a pair of cine cameras, each a sensor of its own);
  `rectify_stereo(pair, size, alpha)`, OpenCV's `stereoRectify` with
  `CALIB_ZERO_DISPARITY` (Bouguet's method) step for step in double,
  returning the `StereoRig` and OpenCV's `R2`; and `fold_radius2(lens)`,
  where a radial model stops carrying rays outward. `StereoRectifier` uses
  both, so a tool without a GPU checks a pair or a lens as the runtime
  does. The method, what each function checks and what it refuses are the
  header's. A pair rectified already is only resampled, its camera scaled
  by one factor so its pixels keep their shape (a learned matcher assumes
  square ones); a vertical pair is refused (`TODO(sensor)`).

### frame

`sensor_frame` is header-only. `DeviceFrame` is one posed, undistorted RGB-D
frame on the device, and `DeviceFrameSet` one trigger's frames in the
producer's sensor order (`frame/device_frame.hpp`). Every producer keeps one
contract, which the header states: device-local storage buffers on the
consumer's `VkDevice`, written completely before the frame is handed over,
owned by the producer's queue family or shared CONCURRENT with the families
it was given, and held by the frame, so the producer reuses a buffer only
once no frame holds it. `GpuFramePrep` is today's producer.

Every producer also states its depth's noise, `DeviceFrame::depth_noise`
(`frame/depth_noise.hpp`, the 2026-10-30 depth-noise decision): a
`DepthNoise {a, b, c}`, the frame-to-frame standard deviation of one sample,
`sigma(z) = a + b z + c z^2` metres. It is temporal, so an error that
repeats every frame (calibration, a matcher's fixed pattern) is not in it.
All zero, the default, means unknown, and a consumer that needs a model
refuses such a frame (`DepthNoise::known`); nothing defaults it.
`DepthNoise::stereo(sigma_d, f, B)` is rectified stereo's model, `c =
sigma_d / (f B)`. A driver states it on `RgbdFrame::depth_noise` for each
frame, as it states its depth range, and `GpuFramePrep::finish` copies it
unchanged; its depth mix only lowers the noise, so the stated value bounds
the prepared one. The producers, each its own value:

- `OrbbecSensor`: `{0, 0.95e-3, 0}` in every depth mode
  (`kFemtoMegaDepthNoise`, internal): measured face on at 640 x 576, the
  other modes not measured.
- `RealSenseSensor`: `DepthNoise::stereo(0.13, fx, B)`, the depth camera's
  fx at the mode it streams and B the length of its factory extrinsic to the
  right infrared camera, read at open (`realsense::chip_depth_noise`,
  internal); 0.13 px is a D455's measured value at 1280 x 720, averaged
  over a room's slants.
- `FfsStereo` (and so `StereoDepthSensor`):
  `DepthNoise::stereo(Options::disparity_sigma, fx, B)` of the rectified rig,
  made at `create`, which refuses a `disparity_sigma` that is not finite and
  at least 0; it defaults to 0, unknown, since it is the camera's and the
  model's, so whoever opens a camera states it.

No per-pixel sigma, no `SensorInfo` field and no calibration-file key: no
producer has a per-pixel confidence that sets a noise (a phone's confidence
plane only gates samples, `RgbdFrame::min_confidence`), recon's libraries
cannot read `SensorInfo`, and no unit has been measured to differ from its
model.
`device_requirements()` names what the package's kernels and decoders need
(`scalarBlockLayout`; `VK_KHR_external_memory_fd`,
`VK_KHR_external_semaphore_fd`, `VK_EXT_metal_objects` and
`VK_EXT_external_memory_host` where offered); an application opening one device for recon and this package
merges the two lists with `core::merge`.

`frame/shaders/pinhole.glsl` is the package's GLSL pinhole: the rigid
inverse, unprojection, divide, in-image projection and nearest pixel
(`vs_*`) that recon's tsdf and texture and the prep's colour test go
through, and that `sensor_frame` exports to its consumers' kernels
(README, "Consuming the package"). Its functions take values, so a
scalar-layout camera and a std430 one both pass theirs. The
acceptance stays two callers' own: texture extrapolates past the image, and
the colour test bounds the rounded pixel. grid's depth allocation
unprojects through its own copy of the one line it needs
(`volume/shaders/depth_pinhole.glsl`), which recon's
`recon_volume_depth_unproject` holds to this one bit for bit.
`frame/shaders/srgb.glsl` is `color_space.hpp`'s curve on the device
(`vs_srgb_to_linear`, `vs_linear_to_srgb`), exported beside it and held to
the host's on all 256 codes (`sensor_frame_srgb_gpu`); the colour pass's
response decodes and encodes through it.

### sensor

the sensor *interface*: `IRgbdSensor` (below) polled for an `RgbdFrame`
and asked `exhausted()` after an empty poll, since "nothing this tick" from
a live device and "nothing, ever" from a replay are the same empty optional
(2026-09-14; non-pure, `false` by default, so a live driver overrides
nothing) — plus the boundary math that is silently wrong when guessed —
`cv_from_gl_camera`, `depth_from_registered_color` and its inverse
`color_from_registered_depth` (a `camera::CameraModel` in and out, through
the one per-axis rescale, the lens kept), `to_canonical`, which
only the iOS scanner and the tests call until the scanner moves onto the
interface (2026-10-07). Every driver keeps **the frame contract**, stated
in `IRgbdSensor`'s doc (the 2026-11-02 frame contract entry): same
capture, whole frames, the sizes `SensorInfo` names, loss read off the
sequence. `testing/rgbd_contract.hpp` checks it (below, under test
support). Links `sensor_camera` and `sensor_frame` alone; drivers live
with the platform that can build *and* test them. The
implementer every recon dataset example runs is
recon's `examples/common/replica_sensor.hpp`, which plays a Replica sequence back
through the interface, its frames prepared on the GPU as a camera's.
**`orbbec/`** (`VS_WITH_ORBBEC`, which
needs the colour decoders, `VS_WITH_FFMPEG` off Apple, and the Orbbec SDK
2.10.6 or later) is the live Femto Mega, through the sensor interface below:
**`OrbbecSensor`**, one camera. A synced rig is an `OrbbecSensor` per camera
in a `SensorArray` (the 2026-10-07 rig decision), opened from its sync file by
`open_orbbec_sensors`. It hands out every frame
as captured, for the GPU pass; nothing on the host undistorts, registers or
converts (the 2026-10-06 raw-frames decision). It reads the camera's rig
sync role (`waits_for_primary`), and given the camera's entry of the rig's
**sync configuration** (`Options::sync`; `orbbec_sync_config.hpp`, the
SDK's `femto_mega_sync.json` layout) refuses a camera whose stored settings
differ, unless `apply_sync` writes them, only where they differ, since they
persist in its flash. The SDK's effective settings are compared again after
the write: normalization that still differs is refused, `Unsupported`;
its cached response does not verify persistence on the camera. At every
open it then sets the colour camera's exposure,
`OrbbecStreamOptions::color_controls` (a manual one, or auto exposure when
empty; auto white balance on either way, never off): checked against the
camera's controls and ranges before any write, written only where the
camera differs, and read back into `SensorInfo::color_controls`, a
difference refused naming the field (`write_color_controls`, internal; the
2026-10-27 colour-controls decision).
The lab rig's is recon's `config/femto_mega_sync.json`, which
only recon's examples and tests name. No test opens a camera (the 2026-10-06
no-hardware decision); recon's `fuse_orbbec` is how one is checked. Behind it is the
internal `CameraStream`, and the types of `orbbec_stream.hpp`: the stream
options, the camera's report, and its counters (`OrbbecStreamStats`).
Every frame states the Femto Mega's depth noise (above, "### frame").
H.265 at 3840x2160 and 25 fps, depth 640x576, is the default
(`color_codec = Hevc`; `Mjpeg` stays selectable, for calibration captures
and diagnostics): each camera's
`HevcColorDecoder` decodes every colour frame, in order, ahead of the
mailbox, as BT.601 full range unless the stream names its matrix, and posts
the picture with its depth, so everything after is MJPEG's path. A frame's
colour camera is read off the profile of the mode streamed. The SDK hands
over every colour frame, depth or not; a pair without depth is dropped
after decoding. A gap in the frame index, or an empty frame, waits for the
next key frame, where the decoder is reset (`KeyGate`, given the frame index,
0 unnumbered); pictures come out in display
order, each settling its own pair. All of it is counted in `stats().lost`
(the 2026-09-28 decision).
**`video/`'s decoders** (every Apple build, and elsewhere
`VS_WITH_FFMPEG`) hand out pictures on the device only (the 2026-10-06
device-only decoder decision), on the hardware of the device they are given,
fixed per build: VideoToolbox on Apple, directly, with no FFmpeg (the
2026-10-25 decision), and through CUDA elsewhere, so `VS_WITH_FFMPEG` needs
`VS_WITH_CUDA` (the CUDA 13 toolkit's headers; libcuda and libnvjpeg are
loaded at run time). A **`DecodedPicture`** is a frame's colour as it is: its
`YuvImage` (`DecodedPicture::yuv`: the planes, the queue family that wrote
them, the size, the matrix's weights, the range and the chroma siting), with
the stream's pts and its transfer and primaries as an optional
`ColorEncoding` (empty when that type cannot name them) beside it.
**`HevcDecoder`** decodes H.265 access units on the hardware. Off Apple,
through FFmpeg's hardware decoding: NVDEC's picture, which FFmpeg copies out
of NVDEC's output surface into a pool frame (without that copy NVDEC keeps an
output surface per decode surface, past the VRAM bar; the 2026-11-02
hand-over decision), copied device to device by one CUDA copy into two plane
images CUDA imported, an R8 luma and an R8G8 chroma of the picture's size
(`core`'s `create_exported_image`, on a device that `exports_memory`), handed
out as NV12's `YuvImage::image`, as VideoToolbox's are, with
`kQueueFamilyExternal` (the 2026-10-26 plane-image decision). The copy is
queued, not waited on: the picture carries its ready value (below), and the
decoder keeps FFmpeg's frame until that value is reached, never as long as a
holder keeps the picture, so held pictures hold none of FFmpeg's frames; with
three kept, the next `receive` waits on the host for the oldest, so CUDA runs
at most three pictures behind. A set of images is made, mapped and handed over to CUDA in
`GENERAL` once, and reused only once no picture holds either image; a
reader takes each picture back from outside Vulkan, and none hands it back.
The images are shared with `Options::queue_families` as
`PixelBufferImporter`'s are. Each CUDA mapping is held from the moment it
exists, so a failed mapping or host allocation releases it, in its context,
before the Vulkan image it maps. On Apple,
through a `VTDecompressionSession` of its own
(`vt_hevc.cpp`): it reads the parameter sets and each picture's slice
header itself (`hevc_syntax.hpp`, below), hands VideoToolbox the
picture's slices as one length-prefixed sample of a format made from the
sequence's VPS, SPS and PPSs (a new session only where a changed set needs
one), and puts the pictures, which come back in decode order, into display
order by their picture order count: shown once more wait than the SPS
reorders, and all at a sequence started afresh and at the end. The two
IOSurface planes are images in `YuvImage::image`, through `apple/`'s
`PixelBufferImporter`, written at the range the VUI declares (video range
without one), so the samples are the stream's own. A picture whose
reference is missing, after a skipped access unit, VideoToolbox reports as
bad data, `IoError`.
`Options::unlabelled_color` stands in for a stream that names no matrix,
and `reset()` restarts a stream after lost access units (on Apple, on a
fresh session at the next key frame). A stream the hardware cannot decode
or hand out -- not 8-bit 4:2:0, or on VideoToolbox a display window off the
coded corner, which it fills from the corner instead -- is refused,
`Unsupported`, from then on, after the pictures decoded before it. On
Apple each SPS is read whole, to its trailing bits, before it can refuse
the stream: a malformed one is `IoError` and cannot latch a refusal.
Both paths describe a picture from the H.273 codes its stream declares
through **`video/frame_color.hpp`**, public, the one home of what the codes
mean: `resolve_matrix` (a matrix `VideoColorMatrix` cannot name taken by
the height, as players do), `resolve_encoding` (empty for an encoding
`ColorEncoding` cannot name) and `resolve_chroma_location` (left for a code
H.273 does not name), so capture code that holds codes, an SPS's or a
pixel buffer's tags, reads them as the decoders do.
VideoToolbox's decoder sessions for both decoders open in one place
(`videotoolbox.cpp`): NV12 on Metal-compatible IOSurfaces, on the hardware
decoder or not at all.
**`JpegDecoder`** decodes a baseline 8-bit 4:2:0 JPEG, BT.601 full
range as JFIF defines it, with no FFmpeg: on nvJPEG's hardware JPEG engine
into the same kind of buffer, I420, handed out with its ready value once the
decode is queued (the next decode waits for the last one's work before it
rewrites nvJPEG's buffers), or through VideoToolbox's hardware
decoder (`vt_jpeg.cpp`) into the same images. Another subsampling, or a
JPEG larger than the hardware takes (16384 a side on nvJPEG's engine, the
device's image extent on Apple), is `Unsupported`, as is every JPEG on a GPU
with no hardware JPEG engine (an RTX 4090, say). For both decoders the
codes are the contract: `IoError` is data that does not decode and costs
only itself (and, for H.265, the frames to the next key frame);
`Unsupported` is a stream the hardware refuses, or a `create` with no
device path (no device or allocator, or a GPU the build's path cannot
reach); `Backend` or `OutOfMemory` is the device path failing. VideoToolbox
errors retain that distinction from session creation, the decode call and
its callback, for both decoders. Its decode-time `kVTParameterErr` also
denotes corrupt data (including a truncated JPEG fixture), so it is
`IoError` there. The
Orbbec driver's colour decoders stop for good on any code but `IoError`, and
the sensor's `poll` returns it, so a refused stream or a failing device path is
not quiet. A `Backend` failure names the library that failed
(`Status::backend()`): VideoToolbox, CUDA or nvJPEG, Vulkan for the hand-over
onto the device, or FFmpeg for its `AVERROR_EXTERNAL` (a hardware call
failed); a hardware failure FFmpeg reports otherwise reads as lost frames.
`VS_TEST_HEVC_BACKEND` makes the decoder tests fail rather than skip where
no device path opens, which is how CI holds its legs to NVDEC on 24.04 (and
nvJPEG where the GPU has the engine) and VideoToolbox on the Mac.
**`video/`'s CUDA interop** (`cuda_interop.hpp`, `VS_WITH_CUDA`) is
public, and is how both decoders reach CUDA: `cuda_ordinal_of` finds the
CUDA device whose UUID is the Vulkan device's; `CudaContext` retains that
GPU's primary context, which every holder shares and none leaves current;
`CudaStream` is a non-blocking stream of its holder's own;
`CudaBufferMapping::map` imports a `create_exported_buffer` buffer's memory,
taking its descriptor (`InvalidArgument` once taken), and maps the whole
buffer, which stays the caller's; and `CudaImageMapping::map` does the same
for a `create_exported_image` image of 8-bit `UNORM` with one, two or four
channels, mapped as a one-level CUDA array of its format and extent,
flagged as a colour attachment where its usage makes it one, as CUDA
requires. The one crossing it documents is CUDA writes, Vulkan reads: the writing stream
or an event is synchronized, then the reader acquires the buffer or image
from `VK_QUEUE_FAMILY_EXTERNAL`, an image once handed over to CUDA in
`GENERAL` before its first write; or, with no host wait, CUDA signals a
timeline it imported after the writes and the reader's batch waits for that
value on the device.
**A picture's ready value** (`YuvImage::ready`, `ready_value`; the 2026-11-02
hand-over decision) is that second form, for NVIDIA's decoders: each
decoder's `core::create_exported_timeline` (on a device that
`exports_semaphores`, else `create` is `Unsupported`), imported into its CUDA
context once (internal `cuda_timeline.cpp`), and each picture the next value,
signalled on the decoder's stream after its copy or decode
(`cuSignalExternalSemaphoresAsync`), in place of the stream synchronize. A
slot of either decoder's ring that no picture holds is let go only once its
last write's value is reached, since a holder may drop a picture CUDA has not
yet written. Every reader here waits for the value on the device in the
submission that reads the picture: `GpuFramePrep::prepare` and
`prepare_batch`, `PictureReadback`, `LumaPass` and `StereoRectifier`, each
one wait a timeline at its highest value (`frame/ready_points.hpp`'s
`add_ready_point`, which a reader of its own uses too), and each refuses a
timeline made on another device before it records anything. A caller of
`GpuFramePrep::record` passes `ready_points(frame)` among its `submit_async`
waits, one of `record_set` or `SensorArray::record_process`
`ready_points(frames)` (`set.frames`). After a CUDA failure, before NVDEC's
`send` or `receive` returns any error but `InvalidArgument`, and in its
destructor before CUDA's import goes, a decoder settles: it synchronizes its
stream, then sets the last value it handed out from the host if the counter
is below it, so no batch waits for good on a signal a failed stream dropped.
NVDEC's host wait for the oldest of three kept pictures asks the stream
between 10 ms slices, so a failed one, or one run dry short of the value,
returns an error, settled (the
2026-11-03 fault decision). VideoToolbox's pictures, a camera's and host planes carry none.
**`video/`'s H.265 syntax** (`hevc_syntax.hpp`, target
`sensor_video_syntax`) is public and builds on every platform: host
code on the family core's base tier, with no codec, GPU or FFmpeg (the
2026-10-25 public-syntax decision). It is what VideoToolbox's path reads.
`split_annex_b` finds an access unit's NAL units (`hevc::Nal`, borrowed,
emulation prevention left in), and `first_slice_type` its first slice's
type by start codes alone, allocating nothing and never failing (with
`is_irap`, whether it is a key); `vps_id`, `parse_sps` and `parse_pps` read
the parameter sets' ids and an SPS's size, conformance window, format,
reorder depth and VUI colour (`ColorCodes`, `color_codes.hpp`: the H.273
codes FFmpeg's frames are described from too), each SPS through its
trailing bits; and `parse_slice_start` reads a slice header as far as its
picture order count, given the parameter sets by id. Bytes that do not
parse are `IoError`, naming why; a NAL unit of another type than the call
reads is `InvalidArgument`; a host allocation failure is `OutOfMemory`,
never an exception. Every parser reads through `hevc::BitReader`:
the RBSP's bits, MSB first past the emulation prevention bytes, as u(n),
ue(v) and se(v), zeros past the end with `overrun()` set, and whether what
is left is the trailing bits.
**`video/`'s key gate** (`key_gate.hpp`, the same target) says which
access units of a predicted stream to send after a loss. `KeyGate::admit`
takes each unit's number (empty for an unnumbered one) and whether it is a
key: it is `Skip` while it waits for a key, at the start, after a number
other than one up from the last and after `lose()`; `Restart` for the key
that ends a wait, which the caller sends after `HevcDecoder::reset()`; and
`Take` otherwise. An unnumbered unit is never a gap, and the one after it
is not checked.
**`video/`'s HEVC encoder** (`hevc_encoder.hpp`, `sensor_video`; the
2026-11-02 encoder decision) is the decoder's twin: `HevcEncoder::create`
opens one hardware session in one configuration (low latency, P-pictures
only, no reordering, every NAL unit in temporal layer 0, a slice QP for
P-pictures and one for IDRs fixed for the session, keys only when asked,
the VUI from `Options::color` normalized and checked against one code set),
or says what is missing (`Unsupported`): VideoToolbox on Apple, NVENC
elsewhere. There are two ways in. With `Options::device`
(a device with `hevc_encoder_device_requirements()`, `device_requirements()`
plus `shaderStorageImageExtendedFormats`, which `create` checks),
`target(batch)` hands out the next picture on the device: on VideoToolbox a
picture of the session's pool, tagged with the session's siting, its planes
R8 and R8G8 storage images in `GENERAL` through `PixelBufferImporter`'s
`Access::Write`, described by the session's colour (the matrix's weights,
range and siting), with nothing to record into the batch since the planes
stay on the device's family. The caller's kernel writes them in that batch,
submits it, and `send(key, pts, written)` hands over the value it sets: the
coding thread waits for it on the host (`vkWaitSemaphores`, which allocates
nothing), then codes the picture, so neither host waits for the device and
the picture never crosses to the host. A second `target` gives up the
first once it succeeds; a refused send keeps it. The targets' images stay
with the encoder, a lost session's passing to the next, since a batch may
still be writing one; the coding thread waits for each sent target's writer
whatever comes of the picture, and so does the destructor, with no limit,
as a pending batch does. The importer readies a surface new to it with a
host-waited submit, behind the device's earlier work, so `create` readies
the targets' surfaces first: it codes one throwaway picture, as the session
replaces its pool at its first (it picks the level then), and imports
`kInFlight + 2` pictures of the pool it keeps, which a producer with two in
flight, or one target given up, cycles through. A target on a surface new
to the session (after a reopen, or one more the pool makes) waits so,
without the encoder's lock; a session whose pool is gone, as after the
system reclaims it, fails the target and is lost. On Apple the other way in is
`send_pixel_buffer(encoder, pixels, key, pts)` (`hevc_encoder_apple.hpp`):
an NV12 IOSurface buffer of the session's size and range, its tags agreeing
with the session's colour, coded as it is, refused while a target is taken.
A send returns at once; a coding
thread of the encoder's own codes each picture, since VideoToolbox's
low-latency mode codes inside `VTCompressionSessionEncodeFrame`; and
`receive` hands back each Annex B access unit in the order sent, its buffer
swapped with the caller's, at most `kInFlight` (2) pictures in flight; a
finite wait sleeps even with none in flight, so a receive thread of its own
does not spin. The coding thread allocates nothing and never throws (fixed
rings, reserved buffers), and on Apple runs at user-interactive quality of
service. The first picture is a key, and so is the next after a failure (`key_owed`): a
dropped picture keeps the session, any other failure loses it, a failed
picture fails those already sent behind it, and the next send opens a lost
session again. `inject` drives a drop or a loss through the backend's own
path, for tests. The rules and the thread are every backend's
(`hevc_encoder.cpp`); the session is the backend's, one definition a build
(`hevc_encode_session.hpp`; VideoToolbox's `vt_hevc_encode.cpp`, NVENC's
`nvenc_session.cpp`). Its
colour goes to VideoToolbox as Core Video names each code; VideoToolbox
codes QP 1 to 51 (0 as 1, so 0 is refused there) and writes the stream's
chroma siting from the buffer's tag, which `send_pixel_buffer` adds to a
buffer that lacks it. The session is torn down before what its callback
uses, since VideoToolbox hands a pending output to the callback inside the
teardown. NVENC (FFmpeg's `hevc_nvenc` on FFmpeg's CUDA device, the
Vulkan device's GPU's primary context, which NVDEC opens alike) codes
targets alone, so it needs `Options::device` and `Options::allocator`, a
device that exports memory, and its GPU reached by CUDA. Its targets are
`kInFlight + 1` exported buffers made at `create`, each mapped into CUDA
once, which FFmpeg registers with NVENC once, by pointer, and NVENC reads
with no copy: one coding, one the device writes, and one for a target taken
while the one before it is held. The free one handed out longest ago goes
next, so a target given up comes back two targets later at the soonest. A
target is NV12 rows, the luma's from 0 and the chroma's after them, at a
stride of the width rounded up to the GPU's texture alignment, as FFmpeg
pads its own CUDA frames; `target` records the buffer's crossing from
outside Vulkan before the batch's commands and back after them
(`video/cuda_interop.hpp`). The coding thread codes with `delay` 0, so
FFmpeg's send waits for the whole picture: NVENC on Linux is synchronous.
The session sets every QP from the options (P, B and the IDR's), so FFmpeg
chooses none, forces IDRs, and asks for 3 strips from 4096 rows (the
driver's choice below); an FFmpeg that cannot ask for them (before 7.1, or
built without nv-codec-headers 12.1) refuses such a picture in `create`.
It writes the VUI's codes as FFmpeg numbers them, H.273's, and no siting, so
sitings 1 to 5 are refused; it codes QP 0 to 51. A lost session is closed,
and its buffers pass to the session that opens again. `facts().settings` is
the configuration as the session reports it, one line, which the rules pin.
The first coder that opens on NVIDIA sets FFmpeg's log level to ERROR, once
a process. `video_backend()` (`backend.hpp`) names the build's backend.
**`apple/`'s `PixelBufferImporter`** (`sensor_apple`, on every
Apple build, iOS included, with no FFmpeg) hands an 8-bit NV12
`CVPixelBuffer` on an IOSurface to Vulkan in place: each plane becomes a
Metal texture imported as an image (`VK_EXT_metal_objects`), with a view and
`SAMPLED` and `TRANSFER_SRC` usage, so `GpuFramePrep` samples it and
`PictureReadback` copies it, made once per
surface and kept while pictures keep arriving on it (one no picture arrived
on in 64 is let go), the images holding the
pixel buffer. Made for `Access::Write`, the images also have `STORAGE` and
`TRANSFER_DST` usage and the textures are written as well as read, so a
compute shader stores into the surface in place (`r8`, `rg8`): the
encoder's targets, on a device with R8 and R8G8 storage images. The Metal device the textures are made on is read once, in
`create`, off a 1x1 probe image's exported texture (internal
`metal_device.mm`, which the importer's test reads Metal's allocations
through too). The images are EXCLUSIVE unless `create` is given the queue
families that read them, by `BufferDesc::queue_families`' rule with the
device's own family among them: two or more distinct make them CONCURRENT, so
a renderer on another family (gfx's, on MoltenVK) reads them in place. The
decoders' `Options::queue_families` and
`OrbbecStreamOptions::color_queue_families` pass the families through, and
NVDEC's plane images take them by the same rule (one check,
`sharing_families.hpp`); nvJPEG's buffers ignore them and are taken over
from `kQueueFamilyExternal`. It sets a `YuvImage`'s planes and leaves the
matrix, range and chroma siting to its caller, who alone knows how the
samples are coded (a stream's unlabelled colour, say). An iOS camera's
pictures, such as ARKit's `capturedImage`, can come through it with no
conversion on the host. **`color_codes_of`** (`core_video_color.hpp`,
the same target) reads what a buffer's tags state as H.273 codes, through
Core Video's own mapping and the six chroma-location strings (internal
`core_video_tags.hpp`, which the encoder shares); a tag Core Video maps to
no code is refused. An IOSurface buffer keeps a tag as a code, and Core
Video gives some back as another string (matrix 11 and 12 as primaries'
names, matrix 15 not at all, on macOS 27): what it reads is what Core Video
keeps.
**`realsense/`** (`VS_WITH_REALSENSE`, librealsense 2.59.1 or later, found
through its own CMake package) is a RealSense D400 (the D455):
**`RealSenseSensor`**, one free-running camera (the 2026-10-29 RealSense
decision). Each frame is the camera's own depth (Z16, in the left infrared
camera, rectified, so the factory model has no lens) and its colour camera's
YUYV on the host (`YuvImage::host`, BT.601 limited range, chroma sited left:
what librealsense converts by), which `GpuFramePrep` stages and converts on
the device; nothing converts on the host. Each frame states the depth noise
the setup holds, rectified stereo's model from the depth camera's fx and the
baseline (above, "### frame"). The factory colour lens,
librealsense's inverse Brown-Conrady, is taken as OpenCV's (k1, k2, p1, p2,
k3): it projects within 0.016 px of `rs2_project_point_to_pixel` over the
D455's picture, the difference being the tangential term applied after the
radial one; other lens models with coefficients are refused. Timestamps are
librealsense's global time, on the host's `std::chrono::system_clock`, so
`SensorInfo::clock` is `Host`; a frameset without depth or colour, or whose
colour lies more than half a frame from its depth, as some do while the
D455 restarts its depth in its first 1.2 s, is counted failed and never
queued, as is a frame `poll` or `drain` refuses; with a manual colour
exposure the colour runs out of step with the depth, up to half a frame. The
counters restart at `start`, not at `stop`. The projector is
on by default (`Options::emitter`); `Options::color_controls` writes a
manual exposure, checked against the camera's ranges first and read back,
as Orbbec's. The frames wait in an internal queue (`realsense/frame_queue.hpp`)
that keeps the `IRgbdSensor` contract: newest wins, a deeper queue drains
oldest first, every frame passed over or pushed out counted dropped. Each
frame holds its frameset, so librealsense's bounded frame pool is the limit
on how many a consumer may keep.
**`RealSenseStereo`** (`realsense/realsense_stereo.hpp`) is the same camera's
two infrared cameras as an `IStereoSensor` (the 2026-10-29 RealSense stereo
decision), the projector off by default: a stereo matcher sees the scene's
own texture, which the dots would cover. Each stereo frame is the left and
right infrared cameras' Y8 as the camera rectifies them, as
`YuvLayout::Luma` on the host (full range), and the colour camera's YUYV as
`RealSenseSensor`'s, the frameset held; `StereoRectifier` copies them to the
device once. `info()` is the factory calibration: both infrared cameras a
pinhole (a lens on either is refused), `left_to_right` librealsense's
extrinsic from the left infrared camera to the right one (on the lab D455
no rotation and 95.03 mm along x, so the rectifier only resamples), and
the colour camera with its lens and `left_to_color`; and what the camera is,
as `RealSenseSensor`'s `SensorInfo` has it (serial, name, the host's clock,
rate, the colour exposure read back), and its bound, half a frame
(`realsense::half_frame_ns`). Its time is the left infrared frame's global
time, and its skew the right one's less it, 0 on a D455, whose pair is one
exposure. It holds the newest stereo frame (`IStereoSensor` has no queue
depth) and wakes a consumer through `set_frame_notify`. Both drivers run on
one internal camera (`realsense/camera.hpp`): finding the camera, the modes,
the projector, the colour exposure, the pipeline with global time on, the
disconnect watch, the frame queue and the counters. A frameset is one moment
(`realsense::together`) when it holds a frame of every mode opened, every
two within half a frame in the whole-nanosecond stamps the frames carry, so
the stereo frames keep their bound exactly; any other is counted failed.
**`IRgbdSensor`** (`rgbd_sensor.hpp`) is one sensor: its
`SensorInfo` from when it opens (id, the cameras' factory models at the
opened modes, `depth_to_color`, rig role, clock, pose source, rate, the
colour exposure as read back), and
frames held up to `set_queue_depth`, the newest taken by `poll` or all,
oldest first, by `drain`. **`OrbbecSensor`** implements it: `open` refuses
a `min_depth` of 0, which the GPU pass would, a camera in a sync mode the
driver does not know, and one whose sync settings differ from
`Options::sync` without `apply_sync`; `sync_clock_to_host` stamps each frame
with the SDK's global timestamp, the camera's clock mapped onto the host's
`std::chrono::system_clock` and re-fitted as the two drift, refusing at
`open` a camera without one, and sets the camera's clock to the host's once
at each `start`, before it streams (`timerSyncWithHost`, never the
process-wide `enableDeviceClockSync`); and its `stats()` counts the driver's
lost frames failed, a JPEG the decoder had no time for being dropped.
**`SensorArray`** (`array/`, target `sensor_array`) reads
several `IRgbdSensor`s as one: it starts the secondaries before the primary,
drains every sensor each `poll_set`, groups the frames into a `FrameSet` by
its internal `TriggerGrouper`, around the primary's frames on the host
clock within `tolerance_us` (by default 0.4 of the fastest member's frame
period, 13.3 ms at 30 fps), and stamps each frame with its sensor's pose
from the `ArrayCalibration`, which must pose every sensor and, where it
names `color_controls`, find every sensor opened at them
(`SensorInfo::color_controls`), or the open is refused. `wait_set`
blocks for a set on one condition variable that every sensor's
notification wakes (`IRgbdSensor::set_frame_notify`, which `open` asks of
each; the Orbbec driver notifies from its mailbox as a pair lands, the camera
goes or the decoder fails), and on the grouping's next deadline, when a
trigger stops waiting for a late sensor; it never sleeps a fixed time. A
set's `arrival_ns` is its latest frame's (`RgbdFrame::arrival_ns`, the
system clock at the driver's frame callback). Opened with a
device, it makes a pass per sensor from `Options::prep`, each with its
sensor's `depth_offset_m` and `color_response` from the calibration, their
one source (an offset, or a response other than the identity, in
`Options::prep` is refused); its `record_process(batch, set)`
records a set's preparation into a
caller's batch through `GpuFramePrep::record_set`: every frame checked, then
every pass's copies and every pass's kernels; `process(set)` is that, one
submit and its wait. A `FrameSet`'s
`skew_ns(i)` is sensor i's frame's time less the trigger's, signed, and its
`spread_ns()` the latest of its frames' times less the earliest.
`RigSets(array, frames)` (`array/rig_sets.hpp`) reads a run of N
frames from a started array: `next` waits for each set (`wait_set`), keeps
each secondary's skew to the primary, and cuts the set that reaches N to its
first frames in the array's order; after `kSilenceLimit` (10 s) with no set
it returns `IoError`, naming a lone sync secondary's cause. The prep stays
the caller's, recorded into the caller's batch (2026-10-25).
**`prep/`'s `GpuFramePrep`** undistorts an `RgbdFrame`
(`rgbd_frame.hpp`) on the device, each camera keeping its intrinsics.
Depth is sampled where the lens puts each pinhole pixel, a sub-pixel
position of the raw image: the bilinear mix of the four raw pixels around it
where all four return within 4 cm of each other (`kBlendSpanM`, the
truncation band the rig fuses with), else the nearest raw pixel. A slope
comes out smooth, not in the half-pixel terraces the nearest pixel alone
leaves, which fusion shades as bands on floors and walls; an edge or a hole
is never blended (2026-10-29). Colour is sampled bilinearly and converted
from Y'CbCr in the same pass. The frame carries each camera's
`camera::CameraModel` in double, narrowed to float once for the passes; the
colour camera's pose, `color_to_world`, and the sensor's `depth_to_color`,
both refused unless rigid, pose the outputs; a sequence number; and
`pixels`, the owner of its host depth, any host colour and any
`color_bitstream` (the colour as the camera coded it, when that decodes on
its own), so a consumer may keep frames past the next poll and past the
capture; depth on the device is held by its buffer, as a device picture is.
`ChromaLocation` travels in the picture's `YuvImage`: JPEG is centred, and
HEVC keeps the decoded tag with left alignment when unspecified.
The GPU samples the location consistently for planes in a buffer and NV12
images. Existing callers that leave the field unset keep left alignment. The
resulting `DeviceFrame` feeds the `Buffer` overloads of grid's
`allocate_from_depth` (its camera through recon's `to_depth_pinhole`) and
recon's `integrate` (and
`ColorFrame::buffer`, with `coverage_in_alpha`, since a
pixel the lens maps outside the picture, or outside the depth camera's
view, is a 0 word), so nothing is
uploaded and nothing registered. Y'CbCr colour comes as I420, NV12, YUYV
(4:2:2, one plane), V210 (SDI's 10-bit 4:2:2, one plane of words) or luma
alone (`YuvLayout::Luma`, a mono camera's,
which comes out grey: R, G and B the luma's code by its range, no chroma
read): on the device (the 2026-10-06 device-only `GpuFramePrep`
decision), planes in a buffer (`YuvImage::device`, with per-plane offsets
and strides) are bound where they are, from
the first plane, once the batch has taken them over from the queue family
that owns them (`YuvImage::queue_family`; the 2026-09-28 decoded-frame
decision). NV12's planes may come as images instead (`YuvImage::image`,
with `SAMPLED` usage and a view), taken over the same way in the layout they
are in (an EXCLUSIVE image only from outside Vulkan), which the colour pass
samples where they are, with no copy: a second entry over the same body
(`undistort_color_images.comp`, `undistort_color.glsl`) binds them as two
combined image samplers, reads them at integer coordinates and makes each
sample its 8-bit code again (`round(v * 255)`), so its words are the buffer
form's, bit for bit (the 2026-10-25 plane-image decision). On the host, a
camera's SDK hands its planes over (`YuvImage::host`: a UVC camera's YUYV;
the 2026-10-29 host-planes decision), copied into the pass's slot after the
depth from the first plane's first byte and read there by the buffer entry;
YUYV's luma is every second byte and its chroma every fourth, on every row.
V210's six pixels are four words of three 10-bit samples, chroma on every
row; `color_planes.glsl`'s `v210_at`, the one unpack prep and the rectifier
share, finds each sample's word and bits and returns it as code / 4, on the
8-bit layouts' scale, so their levels and conversion apply unchanged and the
10 bits are rounded once, to the output's 8 (the 2026-11-01 V210 decision).
V210 is limited range only, its offset and stride whole words.
A dataset decoded on the host hands its colour over as packed R'G'B' words
instead (`RgbdFrame::color_packed`, `io::load_color_packed`'s layout; the
2026-10-07 packed-colour decision), copied the same way; the colour pass
samples R, G and B as byte planes four bytes apart, through the same
bilinear sampler and coverage test. Depth comes from the host
(`RgbdFrame::depth`, as a sensor's SDK hands it over) or from the device
(`RgbdFrame::depth_device`, a `DeviceDepth`: the same 16-bit samples, rows
packed, in a storage buffer from an even byte, as a GPU stage such as stereo
matching leaves them; the 2026-10-29 device-depth decision), read where it
is once the batch has taken the buffer over from its writer's queue family:
bound from its first sample rounded down to the binding alignment, the
image's first sample in the binding a push constant (`raw_first`), so any
even offset is read with no copy, and the buffer must hold the word with
the last sample, which the kernel reads whole. A buffer that several of a
batch's depths and planes name (a frame's depth and planes, every camera's
depth) is taken over once, before any frame's inputs, and references to it
that name different writers are refused. Host depth, and any host
colour after it at the binding alignment, is copied into the pass's
one persistent slot (`core::HostRing`), which the batch binds in place where
mapped device memory is cached and copies once into its device-only mirror
elsewhere; nothing is allocated per frame (a batch staging per call made VMA
allocate a block for every set: 46 ms a rig set on an RTX 5090, 4.8 ms
kept). `record(batch, frame)` records the copy and the passes into a
caller's batch, and `prepare` is that, one submit and its wait; a record
before the pass's last batch resolves is refused, as it binds the slot (a
frame with nothing on the host claims a word of it, which its batch binds
all the same). The
frame *holds* its device-local
buffers through a lease, and the pass reuses one only once no frame holds
it, so a frame kept past the next is still itself; the whole frame is
checked before anything is recorded or timed, a depth range from 0 and a
plane's queue family included, and `prepare` is `prepare_batch` over a set
of one. The driver's depth range, depth noise and times go on as stated;
the pass judges no noise model. Both return `OutOfMemory` for a host
allocation failure, preserving previously returned frames.
A frame with colour keeps only what both its cameras see by their fields of
view (the 2026-10-29 same-view decision), from one buffer of both cameras
and the transform between them, which both passes bind
(`rgbd_overlap.glsl`). A pixel's direction is turned into the other camera
by the extrinsic's rotation alone. Its colour is a 0 word where that
direction lands on no pixel the depth pass covers. Its
depth, from the host or the device, is zeroed where that direction lands
outside the colour picture (the colour pass's rounding and lens test), or
where its point lands on colour the colour pass does not keep. A frame
without colour keeps all its depth. `depth_offset_m`
(0 by default) is subtracted from every raw return in the depth pass, before
the mix and that test: for `z > 0`, `z = max(z - offset, 0)`, so 0 stays no
return whatever the sign; a `SensorArray` sets it per sensor from the
calibration. Host depth may carry a confidence plane
(`RgbdFrame::depth_confidence`, a byte a sample in the producer's levels,
ARKit's 0 low, 1 medium, 2 high) and, with it, a floor of 1 or more that
the producer sets (`min_confidence`; the 2026-11-02 confidence plane
decision). The plane is copied into the slot at the word after the depth,
inside the depth's binding, which grows to cover it
(`DepthParams::confidence_first`, the plane's first byte there); host
colour goes after both. The depth pass reads a raw pixel's level first, in
`metres_at`, and a level below the floor is no return, before the offset,
so the mix skips it, the colour-view test never sees it and the outlier
filter counts it as none: exactly as if the sensor had returned 0 there.
Level 0 is what a producer writes on the samples it invalidates, such as
those it masks as a moving person, and every floor drops it. No plane
leaves the slot, the binding and the copy as before and the branch
untaken, so the output is bit-identical; a plane beside device depth or at
a floor of 0, and a floor without one, are refused. With `depth_outliers`
on, the depth pass writes a buffer of the pass's own
(`sensor.undistorted_depth`, grown as larger frames ask), and a third pass
(`remove_depth_outliers.comp`, the internal
`detail::DepthOutlierPass`) filters it into the output: a thread per pixel
scans the box that holds the image of the return's ball, in square rings from
the centre, until the count is reached or can no longer be; its arithmetic is
the 2026-10-31 depth-outlier decision's.
`color_response` (`camera::ColorResponse`, the identity by default; the
2026-10-31 colour-response decision) is applied in both colour entries for
every layout, between the R'G'B' the matrix forms (`ycbcr_rgb`, or packed
words' samples) and the 8-bit code: clamped to [0, 1], decoded through
`srgb.glsl`, times `gain * F(r)` at the pinhole pixel, encoded and coded
(`responded_word`). Its parameters ride at the end of the overlap buffer
(+28 B, not push constants: `ColorParams` is 124 of Vulkan's minimum 128),
with an identity flag the host derives from them, at which the entries
write `ycbcr_word`'s words, today's, byte for byte. The stereo rectifier
calls `ycbcr_word` and, where it averages a footprint, `ycbcr_to_word`
alone, and `LumaPass` reads luma alone; neither binds the overlap buffer.
The rectifier's views feed a stereo matcher and `LumaPass`'s a detector,
with no response; an FFS frame's colour is its colour camera's own
picture (`StereoRectifier::depth_info`), which gets it once, here. A
luma-only picture on a pass with a response is refused: gains would tint
grey. The response is the pass's, as the depth offset is: `RgbdFrame`,
the frame contract and `sensor_wire` carry a frame as its driver handed
it out, and
`stereo_calibration` a pair's lenses and poses, so none carries one; the
receiver's `SensorArray` sets it from its calibration.
**`prep/`'s `PictureReadback`** copies a set's colour pictures to the
host, luma alone or every plane, for calib's detector: NV12 images from
their corner, and planes in a buffer where they sit, after the batch takes
either over from `YuvImage::queue_family`, images in the layout they are in;
every copy in one batch, one submit and one wait. Both read overloads take
mutable pictures/frames and, after a successful submit and wait, update each
picture's `queue_family` to the readback's family (EXCLUSIVE) or
`kQueueFamilyIgnored` (CONCURRENT, and images that needed no transfer).
Subsequent reads and preparation use the updated picture;
copies made before readback still carry the old ownership metadata.
Creation, including the GPU timer, and reading return `OutOfMemory` for
host allocation failure. The copies land in host memory it keeps (`Staging`,
cached for reading), grown to the largest read, and each `PictureView` points
into it until the next read, so the frames may go at once. A picture is
checked by the rules `GpuFramePrep` applies to one (`yuv_planes.hpp`), with
`TRANSFER_SRC` where the pass needs a buffer's storage usage, or images'
`SAMPLED` usage and views. It reads 4:2:0 and luma alone (its view has no
chroma, whichever planes are asked): a YUYV picture, whose luma is no plane
of its own, is refused, a V210 one is `Unsupported` (`LumaPass` makes its
8-bit luma, which this reads), and a frame whose picture is on the host has
no view. A list read takes pictures in any number, a view each; a buffer
several of them share (a `DeviceLuma`'s full and half) is taken over once,
so they must name one writer.
**`prep/`'s `LumaPass`** makes each picture of a list into its 8-bit luma
and its half on the device, for a detector that searches the half and
refines in the full picture, whatever the camera: one batch and one submit
for the list (the 2026-11-01 luma decision). Planes in a buffer or on the
host go through `bind_planes` as the colour pass binds them, a 4:2:0
picture narrowed to its luma plane first so its chroma is neither bound nor
staged, and the first kernel (`luma.comp`) reads each sample through
`color_planes.glsl`'s `luma_at`, the one unpack, writing four pixels a word:
an 8-bit layout's code as it is, V210's code / 4 rounded half up once,
`(code + 2) / 4`, at most 255. Host planes are copied into one persistent
slot, as frame prep stages them; NV12's luma image is copied from its
corner (`CommandBatch::copy`), with no kernel. The second kernel
(`luma_half.comp`), after every first one, makes each half pixel the rounded
mean of a 2x2 block of the full luma, `(a + b + c + d + 2) / 4`, its centre
at full pixel `2u + 0.5`. Both outputs are `YuvLayout::Luma` pictures, rows
packed, in one device buffer, the half at the binding alignment after the
full; the buffers are a pool, each run taking the smallest free one that
holds a picture, else replacing a free one too small, else adding one, and
a buffer is free once no copy of its lease (`DeviceLuma::owner`, either
picture's `device`) is left, as frame prep's outputs are. They are
EXCLUSIVE to the pass's family, or CONCURRENT across the families `create`
is given (the device's and a renderer's), and say `kQueueFamilyIgnored`.
Inputs are taken over from their writers, each buffer and image once
however many pictures name it, so those pictures must name one writer (the
pass's own family and `kQueueFamilyIgnored` both mean nothing to transfer),
and their `queue_family` is updated after the wait. A failed
submit lets go of the outputs and inputs it used, and the pass refuses
every later run.
An Orbbec frame holds its SDK pair and the SDK
context, with the models and the depth-to-colour extrinsic from the factory
calibration (the extrinsic's rotation made one: the Femto Mega's is not, the
2026-10-06 sensor-frame decision; `open` refuses one that does not come out a
rotation), the depth frame's index as the sequence, and the planes
converted by the matrix and range the stream codes them in. Given
`OrbbecStreamOptions::device`, its H.265 colour is decoded onto that device
and stays there, the picture carried through the mailbox by an SDK frame
whose bytes only name it, so a copy of the frame owns nothing
(`picture_frames.hpp`); for MJPEG it streams the camera's JPEGs, and each
camera's `JpegColorDecoder` decodes them on a thread of its own onto the
device, its picture's frame keeping the camera's frame as well, so the
frame's `color_bitstream` is the camera's JPEG, byte for byte. An H.265 frame
carries none, since it decodes only after the frames before it.
`OrbbecDeviceInfo` names each camera's lens model as the SDK reports it at
the opened mode (`brown-conrady-k6`, say). Each colour decoder, MJPEG's and
H.265's, decodes on a `DecodeWorker` of its own (`decode_worker.hpp`), which
owns its queue, what a full one drops, its thread, and the failure that ends
decoding.

### stereo

`sensor_stereo` (no SDK; every platform) is a calibrated stereo pair as a
stereo camera's driver presents it, and its rectification on the GPU. A
consumer of depth never sees it: a depth source built on a stereo camera
(rectification, then a matcher) is an `IRgbdSensor`.

- **`stereo/stereo_sensor.hpp`** — `StereoInfo` (the pair, the camera
  tier's `StereoCalibration`: both cameras 0 x 0 for a source opened
  without a calibration, whose pictures a calibration is made from, which
  the rectifier and so `StereoDepthSensor` refuse, and neither camera with
  a side 0 for one opened with one, with no other flag; any third, colour
  camera with `left_to_color`; what the
  source is as `SensorInfo` has it: id, model, clock, rate, colour exposure,
  and `max_skew_ns`, its sync guarantee: the most a frame's pictures lie
  apart, frames past it counted failed and never handed out), `StereoFrame`
  (one picture from each camera as the source delivered it, in any layout
  the GPU reads, on the host or in a device buffer; the colour to fuse, a
  third camera's or none when the left is the colour; times as `RgbdFrame`'s,
  the left picture's, and `skew_ns`, the right one's less it; `pixels`
  holding host planes, and `right_pixels` the right picture's own owner
  where it has one), and `IStereoSensor`, polled
  for the newest pair as `IRgbdSensor` is, waking a consumer through
  `set_frame_notify` where it can (`Unsupported`, the default, where it
  cannot), which a RealSense's infrared pair (`RealSenseStereo`), a
  Blackmagic pair or a ZED implements.
- **`stereo/stereo_calibration.hpp`** includes
  `camera/stereo_rectification.hpp`, where `stereo_calibration` is now, and
  `stereo/stereo_sensor.hpp`, so an include of its old path keeps compiling
  (`TODO(sensor)`: remove once the family includes the camera header).
- **The contract** every stereo source keeps is checked by one test helper
  (`tests/stereo_contract.hpp`): the calibration as the source was opened
  (`StereoInfo`'s rule, above), skew within the bound, sequences increasing
  and the times reported too (0 is no time, as for an `RgbdFrame`), and
  the counters' rule, on a test camera's pairs and on each driver's own unit
  (DeckLink's opened both ways, RealSense's factory calibration), not by a
  check of each driver's own.
- **`stereo/stereo_rectifier.hpp`** — `StereoRectifier`. `create` checks
  the source (a calibration, each camera within float's range, a colour
  camera and `left_to_color`), takes the rectification once from the camera
  tier's `rectify_stereo`, which checks the rest and refuses with its own
  reason under the rectifier's name, and builds its pass. `rig()` is that
  rectification's `StereoRig`; the right view's rotation is its
  `right_to_rectified`. `depth_info()` is the `SensorInfo` a
  matcher's depth is prepared with: depth in the rectified left camera,
  colour the third camera or the left one with its lens (its picture is the
  colour), and `depth_to_color = left_to_color * rectified_to_left`; the
  source's id, model, clock, rate and colour exposure, so a sensor array
  takes the depth source as any other sensor.
- **The pass** (`stereo/shaders/rectify.comp`): one dispatch a view, both in
  one batch, one submit and its wait. Each rectified pixel's ray is turned
  into the source camera (the view's rotation undone, from a storage buffer
  written once at create), through the source lens (`lens.glsl`'s
  `distort_normalized`), and the source picture is sampled there
  bilinearly, luma and chroma at their sitings, and converted as the colour
  pass converts: the kernel includes `undistort_color.glsl` and
  `color_planes.glsl`, pushes the colour pass's `ColorParams` with the
  source camera as its `cam`, and checks and binds the planes through the
  same function (`prep/color_pass.hpp`, internal; exported for
  `sensor_stereo`), so the two passes read pictures alike, one home for the
  lens and the sampling. Where the view shrinks the picture, a pixel is
  prefiltered over its own footprint: the map's Jacobian there, from where
  the points half a pixel either side of it land across and down; where
  its longer side is over 1.5 source pixels, the pixel is the mean of 5 x 5
  bilinear samples, each at the centre of one of 25 equal cells of that
  parallelogram, Y'CbCr averaged and converted once
  (`undistort_color.glsl`'s `ycbcr_sample` and `ycbcr_to_word`, which
  `ycbcr_word` is). Else, and where a neighbour's ray is behind the camera
  or past the fold, one sample, as before: a D455's 1.33:1 is unchanged.
  `Luma8` writes Y' by the picture's range, full
  range, four pixels a word; `Rgba8` R'G'B' and the coverage byte, as
  `GpuFramePrep`'s colour. A ray behind the camera, past the source lens's
  fold or out of the picture is 0: alpha above 0 reaches past the fold on a
  strong lens, where the model would put rays back inside the picture,
  mirrored. `create` takes each camera's fold once from the camera tier's
  `fold_radius2`, and the pass compares one squared radius. Pictures in a
  buffer are bound where they are, taken over from `YuvImage::queue_family`
  once a buffer: a
  side-by-side frame's two pictures in one buffer, a colour picture beside
  them, must name the family that wrote it, as `GpuFramePrep`'s inputs must
  (`transfer_family`, `sharing_families.hpp`, which both call), and
  `RectifiedPair::source` names the pass's family for each of them. Once
  read, the right picture is let go: `RectifiedPair::source` keeps no right
  planes and no `right_pixels`, so a source's right buffer is free while a
  matcher runs and a depth frame holds the left picture alone. Host planes
  are copied into one persistent
  `HostRing` slot, left then right, as `GpuFramePrep` stages them; NV12
  plane images are refused (`TODO(sensor)`, when a stereo source decodes on
  the GPU).
- **The views** are device-local storage buffers the rectifier owns, one a
  view, leased to each `RectifiedPair` as `GpuFramePrep`'s outputs are:
  written again only once no pair holds them. The contract is sequential:
  rectify, read, drop, and every pair has the same two buffers, so nothing is
  allocated a frame and CUDA maps each once; a pair kept past the next costs
  the next a new buffer (and a new import). A reader drops a pair only once
  its device reads have completed, as the lease orders the host alone.
  Overlapping one pair's matching with the next's rectifying would save a
  fraction of a millisecond against a matcher's tens of milliseconds, so no
  pool is kept.
  Where the device exports memory they are `create_exported_buffer`'s
  (decided at create: one that cannot export falls back to plain buffers),
  each `ExportedBuffer`'s descriptor open until an importer takes it (CUDA
  through `CudaBufferMapping::map`, which an open descriptor, not an
  address, calls for), and released to `VK_QUEUE_FAMILY_EXTERNAL` after the
  pass (`RectifiedPair::queue_family`), so a reader outside Vulkan reads them
  as they are once `rectify` returns. A Vulkan reader takes them over from
  there and need not hand them back: the next pass takes them without a
  transfer, which Vulkan allows an `EXCLUSIVE` buffer whose contents may be
  lost, since it writes every word. A submit or wait that fails leaves the
  rectifier refusing every later frame, as the device may still run the
  pass.

### decklink

`sensor_decklink` (`VS_WITH_DECKLINK`; the DeckLink SDK's headers, Desktop
Video at run time) is `DecklinkStereo`, two inputs of a DeckLink 8K Pro as
one `IStereoSensor`. Three targets keep the SDK to one file: the units
(`sensor_decklink_units`, static, internal, `EXCLUDE_FROM_ALL`, on every
leg), the capture (`sensor_decklink_capture`, static, internal:
`capture.cpp` and the SDK's dispatch file, which loads
`libDeckLinkAPI.so`), and the public class, which maps its options onto the
units' driver and hands it the SDK's capture.

- **Inputs.** Connectors are named as on the bracket and mapped to the
  sub-devices of the profile "four sub-devices half-duplex" (`ports.hpp`);
  `open` finds the first DeckLink 8K Pro, refuses another profile with the
  fix, checks each input can capture the mode as v210
  (`DoesSupportVideoMode`), and logs (`Info`) the SDK device each connector
  opened, to check against the cabling. `start` enables both at the fixed mode with
  format detection, waits for each to lock (no picture-less frames), then
  starts them; a detected mode other than the one opened fails `start`, or
  later `poll`, with the camera's own fix -- never a silent re-mode.
- **Pages.** The card writes each picture through the SDK's capture
  allocator (`EnableVideoInputWithAllocatorProvider`) into a `PagePool`'s
  host pages, which the device imports (`core::create_imported_host_buffer`),
  made on demand at the SDK's `bufferSize`, at most 8 an input, past which
  the allocator answers `E_OUTOFMEMORY` and the SDK drops a frame. A
  frame's pages are found again by the address the SDK hands back.
- **Callbacks** (one SDK thread an input) stamp and pair and touch no
  pixel (`pairing.hpp`): a frame's host time is its card time plus the two
  clocks' offset sampled together in its own callback, so the card clock's
  drift against the host's is tracked frame by frame; the newest unpaired
  frame of each input is kept, and a frame pairs with the other's within
  `max_skew_ns`, taking the left one's time; the left frames' sequence adds
  whole periods, so a gap is a lost frame and a camera off nominal never
  slips one.
- **Upload** (`upload.hpp`, one thread): a pair's two pictures copied into
  device buffers in one batch, its fence waited, then the SDK's frames
  handed back -- never before, as the card may write those pages again; the
  pair published, the newest winning. The device buffers (`DevicePictures`)
  are each picture's own owner (`YuvImage::device`), back in the pool once
  no frame holds them, so none is made after warm-up; past 24 a pair fails
  rather than fill the device. A copy that fails keeps its pages from the
  card for good and fails every later poll.
- **Counts**, in left frames: received, every left frame; failed, one not
  delivered in a pair (no partner, no signal, past the bound, no device
  buffer free); dropped, a pair replaced before a poll. `stop` logs the
  pools' high marks (`Info`).
- **`decklink_skew_report`** times both inputs through the same capture,
  on the SDK's own buffers (`skew_stats.hpp`): each input's frames, losses
  and rate against the card clock, and the pairs' skew, right less left,
  with its fit at the start and its drift a minute.

### stereo depth (FFS)

`sensor_ffs` (`VS_WITH_FFS`: Linux, `VS_WITH_CUDA`, TensorRT ≥ 11 and nvcc;
`cmake/vs_ffs.cmake`) is depth from any rectified pair by
Fast-FoundationStereo, NVIDIA only. No CUDA or TensorRT type reaches its
headers; TensorRT (shared libraries) and the CUDA runtime (static, which
loads libcuda at run time) are private.

- **The model** is an ONNX export of FFS (`tools/ffs/export_onnx.py`, run
  once per checkpoint and size against a local NVlabs checkout; research
  use, the 23-36-37 checkpoint by default): inputs `left_image` and
  `right_image`, float `1 x 3 x H x W`, ImageNet-normalised R'G'B'; output
  `disparity`, float `1 x 1 x H x W`. H and W are the rig's size rounded up
  to multiples of 32; `create` refuses a model of another size, checking the
  parsed network before the build.
- **The plan cache.** `create` names a plan for the GPU (CUDA's UUID), the
  TensorRT version, the model's bytes (64-bit FNV-1a) and the network's
  size, loads it from `Options::plan_cache` if TensorRT takes it, and else
  builds it (TensorRT's defaults, strongly typed, as the export's AMP casts
  say), writes it under a name of this process's and renames it. A plan
  TensorRT will not load is built again. The cache must be named, and is
  made before anything is built; a plan that cannot be written is logged
  and runs unkept.
- **A frame** (`depth`), on one CUDA stream, in the primary context of the
  device's GPU (`CudaContext`, made current around each call through the
  video tier's internal `CudaContextScope`), then one wait:
  1. the rectifier's two views, read in place: each `ExportedBuffer` mapped
     into CUDA once (`CudaBufferMapping`, which takes its descriptor, so the
     first matcher to read a rectifier's views is their one reader), the
     mappings kept for up to 8 buffers, least recently read let go first; a
     view whose descriptor is closed and that this matcher has no mapping of
     is refused. A new descriptor at a mapped view's address is a new buffer,
     whose old mapping goes. A buffer the rectifier replaced (a pair kept
     past the next) is freed with its last pair, unseen here: its mapping,
     and its memory with it (the import holds the descriptor's reference),
     stays until let go. Their layout is read from their size (`Luma8` a
     byte a pixel, `Rgba8` a word).
  2. the input kernel (`ffs_kernels.cu`): a thread a network pixel and view,
     the view's pixel at the clamped position (padding repeats the last row
     and column), luma in all three channels or R, G, B, each
     `(value - mean) / deviation`, into TensorRT's input buffers, which are
     CUDA's own, allocated at create.
  3. TensorRT (`enqueueV3`) on the same stream.
  4. the depth kernel: a thread a rig pixel, `z = fx * baseline / d` at
     `(x, y)` of the network's disparity; 0 where `d` is not above 0 (NaN
     included), `d > x` (the match is off the right view), or `z` is outside
     `[min_depth, max_depth]`, `min_depth` raised to the near gate
     `1.05 * fx * baseline / max_disparity` where that is farther
     (`Options::max_disparity`), and, for `Rgba8` views, where the left view's
     coverage byte at `(x, y)` or the right view's at `(rint(x - d), y)` is
     0 (the rectifier saw nothing there, and the network matched black);
     else `rint(z / metres_per_unit)`, clamped to 1..65535, with
     `metres_per_unit = max(0.1 mm, max_depth / 65535)`. Written as 16-bit
     samples, rows packed, straight into an exported Vulkan buffer that CUDA
     mapped when it was made. `Luma8` has no coverage: a border (alpha above
     0, or 1 on a pair rectified already whose aspect changes) is matched as
     picture.
  5. the wait: a blocking-sync event recorded after the depth kernel, so
     the host sleeps through the network run (a stream synchronize in the
     primary context, whose flags the decoders share and this leaves alone,
     spins a core for all of it).
  The frame's `depth_device` is that buffer, `kQueueFamilyExternal`, its
  writer finished; colour (the third camera's picture, or the left one's),
  `color_encoding`, `color_to_world` and times come from the pair, the
  colour camera and `depth_to_color` from the rectifier's `depth_info()`,
  and its depth noise from the rig (above, "### frame").
  The depth buffers are leased as the rectifier's views are: one is written
  again only once no frame holds it, and is not handed back from the family
  that read it, since CUDA writes every sample (as the decoders' pictures).
  A launch or wait that fails lets that frame's views and buffer go rather
  than reusing them, and every later frame is refused. Optional metrics:
  `ffs input`, `ffs network` and `ffs depth` device rows (CUDA events) and
  an `ffs` host row with its wait.
- **`StereoDepthSensor`** owns the camera, a rectifier and an `FfsStereo`
  and runs a thread from `start` to `stop`: it polls the camera and, when
  nothing is new, sleeps until the camera's `set_frame_notify` (set at
  `create`) wakes it, or 1 ms for a camera that notifies nothing; then
  rectifies, letting the camera's frame go once rectified, runs FFS, and
  holds the frame (`set_queue_depth`, oldest let go), then calls the frame
  notification. The camera's notification may
  run on its own thread after the camera let it go, so it owns what it
  touches (a shared wake-up). The camera is called from one thread at a
  time: `start` and `stop` on the caller's, `poll` and `stats` on the
  thread, which `stop` joins first. Its `info` is the rectifier's
  `depth_info()`, with the camera's id, model, clock, rate and colour
  exposure, so a `SensorArray` takes it. Its `stats` are the camera's
  `received`, its own `delivered`, and both's `dropped` and `failed` (the
  camera's as of the thread's last poll); a stop lets the frames held go
  uncounted, as every driver's. A pair the rectifier or FFS refuses
  (`InvalidArgument`) is counted `failed` and passed over; any other
  failure -- the camera's poll, a GPU or CUDA failure, after which the
  rectifier and FFS refuse every pair, so it takes a new sensor -- ends the
  thread and is what `poll` and `drain` return until `stop` and `start`. The thread's Vulkan submits share the queue with the caller's
  (`GpuFramePrep`): `core::Device` gives every submit a pool of its own and
  serialises `vkQueueSubmit` under its queue's mutex, and the allocator
  locks its own state.

### audio

`Microphone` holds PortAudio behind a pointer. Its stream callback
(`audio/portaudio_callback.hpp`), on PortAudio's realtime thread, reads
`system_clock` and hands the chunk to `audio::Capture::on_input`
(`audio/capture.hpp`), which takes no lock, allocates nothing and logs
nothing:
- It stamps the chunk with `chunk_pts_ns`:
  `now - (currentTime - inputBufferAdcTime)`. Both times are on
  PortAudio's stream clock, so only their difference crosses clocks; a wait
  that is not finite, negative or 1 s or more counts as 0. The stamp is
  clamped to the chunk before's.
- It copies the chunk into `audio::ChunkRing`, a ring of
  `Options::queue_chunks` entries allocated at `open`, one writer and one
  reader. A full ring refuses the push.

`drain` copies the chunks out on the caller's thread and runs the 1 s
silence check on a steady time passed in. `Capture` holds no PortAudio, so
its tests run in every build. The input opens at its default high latency:
a deeper host buffer rides out a late callback, and the ADC time places
each chunk whatever the latency.

### wire

`wire/rgbd_wire.hpp` (`sensor_wire`; the layout is the header's, the
rules the 2026-11-02 wire-form entry's) is the form a recording or a live
link carries any `IRgbdSensor` in: the `SensorInfo` once, as the depth
track's params, then each frame as a depth record and a colour record. The
cameras, units, range and noise travel per record, not per track (the
entry's "Per record" says why). The encode borrows
the frame's rows and plane and the decode points into the payload it
holds, so neither copies or reads a sample. The decode checks the layout,
not the values, which their users check. It links no device code, so it
builds wherever the interface does, iOS included (the iOS build check
compiles it). The container is grid's `.vrst` session, which this package
does not include, so the writer and the reader live above both.

### image I/O

`io::load_color_packed` decodes JPEG/PNG to top-left row-major encoded RGB
bytes in the existing `R | G<<8 | B<<16` packing. It performs no implicit
linearization or profile conversion; the capture adapter declares the input
encoding. `io::load_depth_u16` requires a genuine 16-bit single-channel PNG
and returns its samples as stored: what a unit is, is the caller's to say
(the Replica sensor's `RgbdFrame::metres_per_unit`).
Expected dimensions are explicit so a camera cannot silently use different
intrinsics. `io::write_png_rgba8` takes already-encoded RGBA8.
`io::encode_png_u16` encodes 16-bit gray samples as stored, and
`io::write_png_u16` writes them; stb writes 8-bit only, so it encodes the
samples' big-endian bytes as two 8-bit channels and IHDR is rewritten to
16-bit gray. The writers open their output only after encoding succeeds.
Errors name the file and backend reason. Public functions validate their input
and translate backend exceptions that reach them into `Status`/`Result`;
private third-party types never appear in public headers. The backend is the
pinned stb source, a private implementation dependency of `sensor_image_io`.

### colour response fit (tools/color_fit)

`tools/color_fit/` fits each camera's `camera::ColorResponse` from posed sets
of a static scene, prepared with identity responses (the 2026-10-31
colour-fit decision). `sensor_color_fit` is an internal static library
(`color_fit::ColorFit`, no export macro, not installed), built with the tests
on every leg so its test, `sensor_color_fit`, runs there. The `color_fit`
tool opens the rig with H.265 colour, the rig's live codec; it builds with
`VS_WITH_ORBBEC`, and no test runs it. The fit takes any array of two or
more sensors of one `SensorInfo::model`, each with colour and depth; only
the tool's open is Orbbec's. It reads the prepared frames alone, where
depth below its frame's confidence floor is already a hole (frame prep's
gate), which no sample's window may hold: low-confidence depth feeds no
sample.

Every per-pixel step is a kernel (`tools/color_fit/shaders/`, over the
exported `pinhole.glsl` and `srgb.glsl`). `add(set)`, one submit: each
camera's words decoded to linear light and blurred at sigma 2 and 8, weighted
by coverage (`blur.comp`); each ordered pair sampled from A's depth pixels and
gated, counted, then kept by a hashed ratio to `Options::samples` (1 M) over
`Options::sets` sets, each pair's samples a run of their own, with both
pixels' radii in a per-camera histogram (`sample.comp`). `solve`: r_max is
each camera's 99th-percentile sample radius; the Huber IRLS normal equations,
the residuals of each hold-out half and the gauge's moments are summed per
ordered pair (`normal_equations.comp`, chunks, then a compensated reduce per
pair), and the host solves the 5N - 3 system by Levenberg-Marquardt,
stopping where a step's predicted gain falls below the float sums' noise;
the share of covered pixels a response takes to 255 is counted on the last
set (`pushed.comp`). A set that fails after its sampling is recorded leaves
the fit refusing every later call. It returns the report and the input
calibration with every response; the tool prints the one and writes the
other to a new file (`WriteMode::CreateNew`), and refuses an existing
`--out` before it opens the rig.

**Runbook.** Geometry calibration first; then carry the array's
`color_controls`, depth offsets and responses into the new file by hand.
Run `color_fit --sync <rig_sync.json> --calibration <in.json> --out
<new.json>` in an empty, static room under the rig's usual lights: 8 sets a
second apart by default. Refit only after a relock or an intrinsics change:
the response is the lens's and the unit's, not the pose's. After a relock,
delete every `color_response` from the file first: the reader refuses a
response fitted at other controls, so the tool cannot read the file.

### test support

`include/volumetric_kit/sensor/testing/` is the package's test support,
for its own tests and a consumer's, header-only and with no test framework
(the 2026-11-02 colour codes and test support entry). `hevc_clip.hpp`
splits a clip at its access unit delimiters and reads
`VS_TEST_HEVC_BACKEND` and `VS_TEST_HEVC_ENCODER`, which make a decoder or
encoder test fail rather than skip. `nv12_pattern.hpp` (the standard
library alone) holds `HostNv12`, the family's test pattern and
`pattern_index`, its reader. The pattern's index band, luma rows 0
to 15 and columns 0 to 255, is 16 squares of 16 x 16: a 255 and a 0
reference, then bits 0 to 13 of the index, each square 255 or 0, so a
reader recovers the index from square means after lossy coding; elsewhere
the planes are smooth triangle waves that move with the index.
The band's layout, `kIndexSquare` and `kIndexBits`, is `index_band.hpp`'s
(the standard library alone), which the pattern and the test depth both
include. `nv12_picture.hpp`'s `device_picture` writes a picture into R8 and
R8G8 plane images in `GENERAL` through the core's `write_image`, padded
past the planes where a test asks, as VideoToolbox's images can be; on
Apple `nv12_pixel_buffer.hpp`'s `pixel_buffer` makes it an IOSurface
`CVPixelBuffer`, tagged, as a camera's. `hevc_slices.hpp`'s `SliceReader`
(in `sensor_video_syntax`, test use only) reads each picture's first slice
type and QP and each SPS's profile and sub-layers, through the parsers'
own walks (internal `hevc_syntax_detail.hpp`); no decode path calls it.
`hevc_conformance.hpp`'s `hevc_encoder_breach` (`sensor_video_testing`,
static, built wherever `sensor_video` is, with no core test support) runs
the encoder rules C1 to C13b on one case through the public calls, on pixel
buffers or on targets, and names the first one broken; sensor's encoder
test runs it on the public sizes and every code in the set, and an app's
device test can run it as it is. `nv12_writer.hpp`'s `Nv12Writer` (the same
target) writes a test picture into plane images from one compute kernel
(`write_nv12.comp`, embedded), recorded into the caller's batch, as a
producer's kernel writes a target; the rules' C13b needs the write inside a
batch a timeline value orders, and the target's `STORAGE` planes need a
kernel's stores, which a copy would not show.

`rgbd_pattern.hpp` (standard library alone; camera-only builds run its
test) is that test depth, as a tracked sensor delivers it:
`pattern_depth(width, height, index)` writes u16 millimetres, the band at
the same pixels as the picture's luma, a set square 1000 and a clear one
2000, the plane at 2000 and no return across the left half of the bottom
eighth, beside a confidence plane of ARKit's levels, 2 on every square and
0, 1, 2 by column elsewhere. `depth_index` reads it back only where every
sample of each square's central half holds the square's one value and the
references are 1000 and 2000, so torn depth is no index rather than a wrong
one.

`rgbd_contract.hpp` checks the frame contract on what an `IRgbdSensor`
handed out. `RgbdContract` takes each frame as it is handed out and keeps
none, for a run of any length, one checker a start, and its
`breach(stats)` names the first rule broken with its frame's place: with
`SensorInfo::depth`, depth present and its camera that size; a confidence
plane only beside host depth at a floor of 1 or more, and a floor only with
one; with `SensorInfo::color`, colour present, its camera that size and the picture
its camera's, without it no colour; then `order_breach` (sequences
increase, and times where both are reported, each on the last one
reported: 0 is no time) and
`stats_breach` (`delivered` the frames taken, `delivered + dropped +
failed <= received`), which the stereo contract
(`tests/stereo_contract.hpp`) calls too, the same way. `rgbd_breach` runs
it over a vector. The Orbbec, RealSense and FFS depth
sensors' tests run it under injected faults. The Orbbec camera stub stands
in for `CameraStream`'s mailbox, take and read, and plays a script of
captures with counts of its own, so `OrbbecSensor`'s poll, drain (its
discards after a refusal) and stats (failed plus lost) run on it. The
RealSense camera stub (`tests/realsense_camera_stub.cpp`, in place of
`camera.cpp`, the SDK callback's received count and `stats()` included)
sends librealsense software-device framesets into the driver's frame
queue, so `RealSenseSensor`'s conversion, poll, drain, its delivered and
failed counts and the queue's drops run. `CameraStream`'s and
`camera.cpp`'s own counting is checked only on a camera.

## Next work

**Cameras and sensors for the family (the 2026-10-06 plan).** This package is
what calib and ios build on for cameras and sensors. The `camera` tier has
landed; the stack continues:
1. **The sensor interface.** `RgbdFrame` (each camera's `CameraModel`,
   `color_to_world` and `depth_to_color`, a sequence number, pixels it holds)
   and `IRgbdSensor` with `SensorInfo` have landed, the Femto Mega as
   `OrbbecSensor`, and `OrbbecCapture` and the SDK's host path are gone. The
   Replica source is an `IRgbdSensor` too, its colour packed host words
   (`RgbdFrame::color_packed`), and `ICameraCapture` and `CapturedFrame`
   are gone (2026-10-07). The frame gains no float depth: the iOS scanner,
   which still calls the host-depth and convention entry points kept for
   it, quantises ARKit's float depth to 16 bits when it moves onto the
   interface.
2. **`SensorArray`**, vendor-neutral, has landed: start order from the sync
   roles, trigger grouping, poses from the calibration file
   (nearest-frame and tracked members later), and `process(set)`, every
   stream of every sensor in one GPU batch. `OrbbecRig` is gone:
   `fuse_orbbec` and `rig_viewer` open a `SensorArray` of `OrbbecSensor`s on
   the SDK's global timestamps (2026-10-07), their skew measured flat over a
   20-minute `fuse_orbbec` run; sessions of hours are still to measure.
3. **Luma readback** has landed in `prep` (`PictureReadback`),
   wherever the decoder left the picture, for calib's detector, and any
   layout's 8-bit luma and half on the GPU before it (`LumaPass`).
4. **Later: mixed arrays** of fixed sensors and tracked ones (iPhones):
   nearest-frame members, registration of a tracked sensor's world, a network
   sensor with an ios sender, and clock offsets.

**On `sensor`**, each a `TODO(sensor)`: the GPU pre-processing has landed for
one camera (`GpuFramePrep`, the 2026-09-28 GPU pre-processing decision: at 4K
it takes the host from 15.5 ms of undistortion and registration a frame to
none, and the run's CPU eightfold down) and for the rig's raw sets. NVDEC
and nvJPEG hand their pictures over on the device, and VideoToolbox both
kinds, and a raw Orbbec frame's colour stays on the device over either
codec, VideoToolbox's and NVDEC's plane images sampled where they are. For
H.265: the camera's encoder settings, its key-frame interval above all,
which sets what a lost frame costs (`camera_stream.cpp`). The rig's next consumer is calib's
viewer, showing its synchronised sets.

**Depth noise, later.** A per-pixel sigma once a producer has a confidence
(the Femto Mega's infrared amplitude, which needs an infrared stream); a
total (spatial) sigma, a field of its own beside the temporal one, with the
consumer that needs it, such as a 1/sigma^2 fusion weight; and measured
values for the Femto Mega's other depth modes and the D455's other modes.
