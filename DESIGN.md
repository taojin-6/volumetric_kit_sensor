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
  (FFS)](#stereo-depth-ffs), [image I/O](#image-io)
- [Next work](#next-work)

## Tiers

The package sits on volumetric_kit_core alone, and nothing in it includes
recon or grid, its sibling on the core (`sensor_package_boundary`). recon's
libraries link the `frame` tier alone (`recon_library_sensor_links`);
recon's examples assemble the rest.

- **`camera`** — the camera vocabulary the drivers and the interface
  share: the double-precision `CameraModel` (image size, intrinsics, OpenCV's
  rational lens), the lens as a scalar template, rigid transforms, and the
  sensor array's calibration file (`camera/array_calibration.hpp`). No
  Vulkan (the 2026-10-06 decisions).

- **`frame`** — the contract recon's fusion, texturing and sender read,
  header-only on the family core's Vulkan tier: `DeviceFrame` and
  `DeviceFrameSet` (`frame/device_frame.hpp`), the posed pinhole
  `DepthCameraParams`/`ColorCameraParams` (`frame/camera_params.hpp`),
  `ColorEncoding`, `is_canonical` and the sRGB curve
  (`frame/color_space.hpp`), and the GLSL pinhole it exports; beside them
  the package's GLM aliases and `device_requirements()`.
- **`sensor`** — the sensor *interface*: `IRgbdSensor`, the `RgbdFrame` it
  hands out as captured (each camera's `camera::CameraModel`, the pixels it
  holds), and the boundary conversions a capture integration gets silently
  wrong — camera conventions (pose handedness, registered-depth intrinsics)
  and colour (`to_canonical`). Reads `camera`'s models, and `frame`'s
  pinhole cameras and `ColorEncoding`, so it
  depends on **`camera` and `frame` alone** — it sits beside recon's fusion tiers,
  not on top of them — and bundles **no drivers**: one ships here only if this package can build
  *and* test it (the 2026-08-02 decision). The ones that do, Orbbec and
  RealSense, are targets of their own (`orbbec/`, `realsense/`), so
  `sensor_interface` never links a vendor SDK. The HEVC and JPEG decoders are another target of
  their own (`video/`, on the GPU's hardware) and link no camera tier:
  `frame` and, on Apple, the Apple importer (`apple/`), which links
  `frame` alone. The H.265 syntax they read on the host is a target of its
  own as well, on the family core's base tier alone, in every build.
  The GPU pre-processing and the pictures' readback are a fourth
  (`prep/`, Vulkan and shaders), so `sensor_interface` itself stays free
  of all of them.
- **`array`** — several sensors read as one (`SensorArray`, `RigSets`),
  on the GPU pass.
- **`stereo`** — calibrated stereo pairs: what a stereo camera driver
  implements and hands out (`IStereoSensor`, `StereoFrame`), and
  `StereoRectifier`, the pair undistorted and rectified on the GPU for a
  stereo matcher. On the interface; it links `prep` privately, whose colour
  pass it shares.
- **`ffs`** — stereo depth on NVIDIA's GPUs (`VS_WITH_FFS`): `FfsStereo`,
  Fast-FoundationStereo on TensorRT with CUDA kernels either side, and
  `StereoDepthSensor`, a stereo camera as an `IRgbdSensor`. On `stereo`; it
  links `video` privately, for CUDA's view of Vulkan memory.
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
  `sensor_stereo`, `sensor_video`, `sensor_video_syntax`, `sensor_apple`,
  `sensor_orbbec`, `sensor_realsense`, `sensor_ffs` and `sensor_image_io`; the umbrella `volumetric_kit::sensor` (real target
  `sensor`, which is why the interface's is `sensor_interface`). `io` is
  the one exception to `sensor_<tier>` and `<tier>/export.hpp`: its
  target, export header and macro are named after the image files it holds
  (`sensor_image_io`, `io/image_io_export.hpp`, `VS_IMAGE_IO_API`), as they
  were in recon.
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
  version 2 is refused.
  `format_` / `write_array_calibration` write it
  through the core's `write_file`; version 1 is read for its poses.

### frame

`sensor_frame` is header-only. `DeviceFrame` is one posed, undistorted RGB-D
frame on the device, and `DeviceFrameSet` one trigger's frames in the
producer's sensor order (`frame/device_frame.hpp`). Every producer keeps one
contract, which the header states: device-local storage buffers on the
consumer's `VkDevice`, written completely before the frame is handed over,
owned by the producer's queue family or shared CONCURRENT with the families
it was given, and held by the frame, so the producer reuses a buffer only
once no frame holds it. `GpuFramePrep` is today's producer.
`device_requirements()` names what the package's kernels and decoders need
(`scalarBlockLayout`; `VK_KHR_external_memory_fd` and `VK_EXT_metal_objects`
where offered); an application opening one device for recon and this package
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

### sensor

the sensor *interface*: `IRgbdSensor` (below) polled for an `RgbdFrame`
and asked `exhausted()` after an empty poll, since "nothing this tick" from
a live device and "nothing, ever" from a replay are the same empty optional
(2026-09-14; non-pure, `false` by default, so a live driver overrides
nothing) — plus the boundary math that is silently wrong when guessed —
`cv_from_gl_camera`, `depth_from_registered_color`, `to_canonical`, which
only the iOS scanner and the tests call until the scanner moves onto the
interface (2026-10-07). Links `sensor_camera` and `sensor_frame` alone;
drivers live with the platform that can build *and* test them. The
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
H.265 at 3840x2160 and 25 fps, depth 640x576, is the default
(`color_codec = Hevc`; `Mjpeg` stays selectable, for calibration captures
and diagnostics): each camera's
`HevcColorDecoder` decodes every colour frame, in order, ahead of the
mailbox, as BT.601 full range unless the stream names its matrix, and posts
the picture with its depth, so everything after is MJPEG's path. A frame's
colour camera is read off the profile of the mode streamed. The SDK hands
over every colour frame, depth or not; a pair without depth is dropped
after decoding. A gap in the frame index, or an empty frame, waits for the
next key frame, where the decoder is reset; pictures come out in display
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
through FFmpeg's hardware decoding: NVDEC's picture copied device to device,
by one CUDA copy, into two plane images CUDA imported, an R8 luma and an
R8G8 chroma of the picture's size (`core`'s `create_exported_image`, on a
device that `exports_memory`), handed out as NV12's `YuvImage::image`, as
VideoToolbox's are, with `kQueueFamilyExternal` (the 2026-10-26 plane-image
decision). A set of images is made, mapped and handed over to CUDA in
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
VideoToolbox's decoder sessions for both decoders open in one place
(`videotoolbox.cpp`): NV12 on Metal-compatible IOSurfaces, on the hardware
decoder or not at all.
**`JpegDecoder`** decodes a baseline 8-bit 4:2:0 JPEG, BT.601 full
range as JFIF defines it, with no FFmpeg: on nvJPEG's hardware JPEG engine
into the same kind of buffer, I420, or through VideoToolbox's hardware
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
`GENERAL` before its first write.
**`video/`'s H.265 syntax** (`hevc_syntax.hpp`, target
`sensor_video_syntax`) is public and builds on every platform: host
code on the family core's base tier, with no codec, GPU or FFmpeg (the
2026-10-25 public-syntax decision). It is what VideoToolbox's path reads.
`split_annex_b` finds an access unit's NAL units (`hevc::Nal`, borrowed,
emulation prevention left in); `vps_id`, `parse_sps` and `parse_pps` read
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
**`apple/`'s `PixelBufferImporter`** (`sensor_apple`, on every
Apple build, iOS included, with no FFmpeg) hands an 8-bit NV12
`CVPixelBuffer` on an IOSurface to Vulkan in place: each plane becomes a
Metal texture imported as an image (`VK_EXT_metal_objects`), with a view and
`SAMPLED` and `TRANSFER_SRC` usage, so `GpuFramePrep` samples it and
`PictureReadback` copies it, made once per
surface and kept while pictures keep arriving on it, the images holding the
pixel buffer. The images are EXCLUSIVE unless `create` is given the queue
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
conversion on the host.
**`realsense/`** (`VS_WITH_REALSENSE`, librealsense 2.59.1 or later, found
through its own CMake package) is a RealSense D400 (the D455):
**`RealSenseSensor`**, one free-running camera (the 2026-10-29 RealSense
decision). Each frame is the camera's own depth (Z16, in the left infrared
camera, rectified, so the factory model has no lens) and its colour camera's
YUYV on the host (`YuvImage::host`, BT.601 limited range, chroma sited left:
what librealsense converts by), which `GpuFramePrep` stages and converts on
the device; nothing converts on the host. The factory colour lens,
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
rate, the colour exposure read back). Its time is the left infrared frame's
global time. It holds the newest stereo frame (`IStereoSensor` has no queue
depth) and wakes a consumer through `set_frame_notify`. Both drivers run on
one internal camera (`realsense/camera.hpp`): finding the camera, the modes,
the projector, the colour exposure, the pipeline with global time on, the
disconnect watch, the frame queue and the counters. A frameset is one moment
(`realsense::together`) when it holds a frame of every mode opened, each
within half a frame of the first; any other is counted failed.
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
sensor's `depth_offset_m` from the calibration, the offset's one source (an
offset in `Options::prep` is refused); its `record_process(batch, set)`
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
pixel the lens maps outside the picture is a 0 word), so nothing is
uploaded and nothing registered. Y'CbCr colour comes as I420, NV12, YUYV
(4:2:2, one plane) or luma alone (`YuvLayout::Luma`, a mono camera's,
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
kept). `record(batch, frame)` records the copy and both passes into a
caller's batch, and `prepare` is that, one submit and its wait; a record
before the pass's last batch resolves is refused, as it binds the slot (a
frame with nothing on the host claims a word of it, which its batch binds
all the same). The
frame *holds* its device-local
buffers through a lease, and the pass reuses one only once no frame holds
it, so a frame kept past the next is still itself; the whole frame is
checked before anything is recorded or timed, a depth range from 0 and a
plane's queue family included, and `prepare` is `prepare_batch` over a set
of one. Both return `OutOfMemory` for a host allocation failure,
preserving previously returned frames.
`depth_within_color` (off by
default; `rig_viewer` turns it on) zeroes depth outside the colour camera's
view, by the colour pass's own coverage test, so nothing is fused that no
colour camera can colour. `depth_offset_m` (0 by default) is subtracted
from every raw return in the depth pass, before the mix and that test: for
`z > 0`, `z = max(z - offset, 0)`, so 0 stays no return whatever the sign; a
`SensorArray` sets it per sensor from the calibration.
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
of its own, is refused, and a frame whose picture is on the host has no
view.
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

- **`stereo/stereo_sensor.hpp`** — `StereoCalibration` (the two cameras,
  lenses included, and `left_to_right`, OpenCV's `R`, `T`), `StereoInfo`
  (the pair and any third, colour camera with `left_to_color`, and what the
  source is as `SensorInfo` has it: id, model, clock, rate, colour exposure),
  `StereoFrame` (one picture from each camera as the source delivered it, in
  any layout the GPU reads, on the host or in a device buffer; the colour to
  fuse, a third camera's or none when the left is the colour; times as
  `RgbdFrame`'s; `pixels` holding host planes), and `IStereoSensor`, polled
  for the newest pair as `IRgbdSensor` is, waking a consumer through
  `set_frame_notify` where it can (`Unsupported`, the default, where it
  cannot), which a RealSense's infrared pair (`RealSenseStereo`), a
  Blackmagic pair or a ZED implements.
- **`stereo/stereo_rectifier.hpp`** — `StereoRectifier`. `create` computes
  the rectification once, on the host, as OpenCV's `stereoRectify` does with
  `CALIB_ZERO_DISPARITY` (Bouguet's method), step for step in double: the
  relative rotation split in half between the cameras, the baseline then
  turned onto x, one focal length for both views (the cameras' vertical
  ones, scaled to the rectified height), one principal point where the
  corners, undistorted and turned, centre, and `alpha`'s scale between the
  rectangle every view's pixels are valid in (0) and the one holding every
  source pixel (1), from the border of a 9 x 9 grid; the undistortion of
  corners and borders is OpenCV's own five fixed-point iterations
  (`undistortPoints`' default). It refuses a vertical pair and one whose
  right camera is on the left one's -x side (`TODO(sensor)`: vertical
  pairs). A pair that is rectified already -- no rotation, no lens, one
  camera model, the right camera along +x alone, each within 1e-9 -- is only
  resampled, with no rotation: its camera scaled to the size by one factor
  `s`, so its pixels keep their shape (a learned matcher assumes square
  ones), about the centres (`fx * s`, `(w' - 1) / 2 + (cx - (w - 1) / 2) * s`).
  Alpha picks `s` between the larger of `w' / w` and `h' / h` (0: the view
  filled, the picture cropped) and the smaller (1: all of the picture, the
  view bordered); at the picture's own aspect ratio both are the scale edge
  to edge. `rig()` is the views' camera (pinhole), the baseline (metres,
  above 0) and `rectified_to_left` (OpenCV's `R1` transposed); the right
  view's rotation is `R1 * R^T`. `depth_info()` is the `SensorInfo` a
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
  lens and the sampling. `Luma8` writes Y' by the picture's range, full
  range, four pixels a word; `Rgba8` R'G'B' and the coverage byte, as
  `GpuFramePrep`'s colour. A ray behind the camera, past the source lens's
  fold or out of the picture is 0. The fold is where the radial model stops
  carrying rays outward (`r * radial(r^2)` stops growing, or its denominator
  reaches 0); past it the model turns back and would put rays the lens
  cannot image inside the picture, mirrored, which alpha above 0 reaches on
  a strong lens. `create` finds it once a camera on the derivative's sign --
  with `s = r^2`, `N D + 2 s (N' D - N D')` or `D` no longer positive --
  marched by the angle off the axis, 0.01 degrees a step, then bisected,
  and the pass compares one squared radius. Pictures in a buffer are bound
  where they are, taken over from `YuvImage::queue_family` once a buffer: a
  side-by-side frame's two pictures in one buffer, a colour picture beside
  them, must name the family that wrote it, as `GpuFramePrep`'s inputs must
  (`transfer_family`, `sharing_families.hpp`, which both call), and
  `RectifiedPair::source` names the pass's family for each of them. Host planes are copied into one persistent
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
  TensorRT will not load is built again.
- **A frame** (`depth`), on one CUDA stream, in the primary context of the
  device's GPU (`CudaContext`, made current around each call through the
  video tier's internal `CudaContextScope`), then one wait:
  1. the rectifier's two views, read in place: each `ExportedBuffer` mapped
     into CUDA once (`CudaBufferMapping`, which takes its descriptor, so the
     first matcher to read a rectifier's views is their one reader), the
     mappings kept for up to 8 buffers, least recently read let go first; a
     view whose descriptor is closed and that this matcher never mapped is
     refused. Their layout is read from their size (`Luma8` a byte a pixel,
     `Rgba8` a word).
  2. the input kernel (`ffs_kernels.cu`): a thread a network pixel and view,
     the view's pixel at the clamped position (padding repeats the last row
     and column), luma in all three channels or R, G, B, each
     `(value - mean) / deviation`, into TensorRT's input buffers, which are
     CUDA's own, allocated at create.
  3. TensorRT (`enqueueV3`) on the same stream.
  4. the depth kernel: a thread a rig pixel, `z = fx * baseline / d` at
     `(x, y)` of the network's disparity; 0 where `d` is not above 0 (NaN
     included), `d > x` (the match is off the right view), or `z` is outside
     `[min_depth, max_depth]`; else `rint(z / metres_per_unit)`, clamped to
     1..65535, with `metres_per_unit = max(0.1 mm, max_depth / 65535)`.
     Written as 16-bit samples, rows packed, straight into an exported
     Vulkan buffer that CUDA mapped when it was made.
  The frame's `depth_device` is that buffer, `kQueueFamilyExternal`, its
  writer finished; colour (the third camera's picture, or the left one's),
  `color_encoding`, `color_to_world` and times come from the pair, the
  colour camera and `depth_to_color` from the rectifier's `depth_info()`.
  The depth buffers are leased as the rectifier's views are: one is written
  again only once no frame holds it, and is not handed back from the family
  that read it, since CUDA writes every sample (as the decoders' pictures).
  A launch or wait that fails lets that frame's views and buffer go rather
  than reusing them, and every later frame is refused. Optional metrics:
  `ffs input`, `ffs network` and `ffs depth` device rows (CUDA events) and
  an `ffs` host row with its wait.
- **`StereoDepthSensor`** owns the camera, a rectifier and an `FfsStereo`
  and runs a thread from `start` to `stop`: it polls the camera, waiting
  1 ms when nothing is new (`IStereoSensor` tells no one), rectifies, runs
  FFS, and holds the frame (`set_queue_depth`, oldest let go), then calls
  the frame notification. The camera is called from one thread at a time:
  `start` and `stop` on the caller's, `poll` on the thread, which `stop`
  joins first. A pair the rectifier or FFS refuses (`InvalidArgument`) is
  counted `failed` and passed over; any other failure -- the camera's poll,
  a GPU or CUDA failure, after which the rectifier and FFS refuse every pair
  -- ends the thread and is what `poll` and `drain` return until the next
  `start`. The thread's Vulkan submits share the queue with the caller's
  (`GpuFramePrep`): `core::Device` gives every submit a pool of its own and
  serialises `vkQueueSubmit` under its queue's mutex, and the allocator
  locks its own state.

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
   wherever the decoder left the picture, for calib's detector.
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
