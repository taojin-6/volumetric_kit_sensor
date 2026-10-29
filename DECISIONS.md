# Locked decisions

The dated design record for `volumetric_kit_sensor`: each rule's rationale,
measurements, review findings, and verification. Newest context wins.

[AGENTS.md](AGENTS.md) is the concise working guide; [DESIGN.md](DESIGN.md)
describes the implementation and detailed contracts. When a decision changes,
update its index entry and dated rationale here in the same commit, plus the
affected contract in DESIGN.md and any essential rule in AGENTS.md.

These entries governed the camera and sensor tiers while they were recon's,
and moved here with them, dates unchanged and text as it was but for a few
words. They keep the names of their day (`recon_sensor_*`, `VR_WITH_*`,
`vr::sensor`), which [CHANGELOG.md](CHANGELOG.md)'s 0.1.0 maps to the names
here. An entry they name that is not here is in
[recon's record](https://github.com/taojin-6/volumetric_kit_recon/blob/main/DECISIONS.md#decision-index),
which holds the rest of the family's history and its measured lessons.

## Decision index

The index and dated entries are in the same chronological order. Read the
entries relevant to your task; later amendments supersede earlier rules.

- [**2026-08-02**](#2026-08-02--the-sensor-tier-is-a-contract-not-a-driver-collection-a-capture-driver-lives-here-only-if-this-repo-can-build) —
  The `sensor` tier is a *contract*, not a driver collection: a capture driver
  lives here only if this repo can build (`ICameraCapture` went 2026-10-07)
- [**2026-08-02**](#2026-08-02--one-color-space-rule-and-a-named-working-space-8-bit-color-is-encoded-float-color-is-linear-converted-once-at-the-sensor-boundary-and-encoded-once-at-presentation) —
  One color-space rule, and a *named* working space: 8-bit color is encoded,
  float color is linear, converted once at the sensor boundary and encoded once
  at presentation. *Amended 2026-10-28:* the voxel colour attribute's GLSL
  and its host half are `volume`'s, and the PLY writer has a copy of its own.
- [**2026-09-14**](#2026-09-14--the-examples-poll-their-frames-through-the-sensor-contract-the-replica-reader-is-an-icameracapture-a-source-says-when-it-is-exhausted-and-a-frame-kept-past-the-next-poll-is-copied--the-last-one-included) —
  The examples poll their frames through the sensor contract: the Replica
  reader is an `ICameraCapture`, a source says when it is exhausted, and a
  frame kept past the next poll is copied — the last one included. (The
  reader became an `IRgbdSensor`, and the copy went, on 2026-10-07.)
- [**2026-09-24**](#2026-09-24--the-orbbec-sdk-is-a-prerequisite-behind-vr_with_orbbec-installed-once-for-the-family-found-and-never-fetched) —
  The Orbbec SDK is a prerequisite behind `VR_WITH_ORBBEC`: installed once for
  the family, found, and never fetched. The floor rose to 2.10.6 on
  2026-10-07.
- [**2026-09-26**](#2026-09-26--the-orbbec-driver-lands-as-sensororbbec-a-target-of-its-own-it-undistorts-colour-and-then-registers-depth-to-it-on-the-host-and-it-reads-the-rigs-sync-roles-without-writing-them) —
  The Orbbec driver lands as `sensor/orbbec`, a target of its own: it
  undistorts colour and then registers depth to it on the host, and it reads
  the rig's sync roles without writing them. The host path went on
  2026-10-06: every frame is handed out as captured.
- [**2026-09-27**](#2026-09-27--the-rig-is-orbbecrig-it-checks-the-cameras-against-the-rigs-sync-configuration-and-writes-it-only-when-asked-starts-the-secondaries-before-the-primary-keeps-the-cameras-on-the-hosts-clock-and-builds-each-set-around-a-primary-frame-poses-come-from-the-calibration-file) —
  The rig is `OrbbecRig`: it checks the cameras against the rig's sync
  configuration and writes it only when asked, starts the secondaries before
  the primary, keeps the cameras on the host's clock, and builds each set
  around a primary frame; poses come from the calibration file. The file
  moved to the `camera` tier on 2026-10-06, and the rig's sets became raw
  only the same day. `OrbbecRig` went on 2026-10-07: a rig is a
  `SensorArray` of `OrbbecSensor`s, on the SDK's global timestamps.
- [**2026-09-27**](#2026-09-27--the-video-decoder-is-recon_sensor_video-a-target-of-its-own-over-an-installed-ffmpeg-hevc-in-host-pictures-out-on-the-first-hardware-back-end-that-decodes-a-built-in-clip-nvidia-ahead-of-an-integrated-gpu-each-ci-leg-requires-the-hardware-it-has) —
  The video decoder is `recon_sensor_video`, a target of its own over an
  installed FFmpeg: HEVC in, host pictures out, on the first hardware back end
  that decodes a built-in clip, NVIDIA ahead of an integrated GPU; each CI leg
  requires the hardware it has. The decoders hand out device pictures only
  since 2026-10-06.
- [**2026-09-28**](#2026-09-28--the-orbbec-driver-streams-colour-as-h265-on-request-every-colour-frame-is-decoded-in-order-on-a-thread-per-camera-ahead-of-the-mailbox-a-lost-frame-is-read-off-the-frame-index-not-the-clock-and-the-femto-megas-stream-is-decoded-as-bt601-full-range-which-it-codes-and-does-not-say) —
  The Orbbec driver streams colour as H.265 on request: every colour frame is
  decoded, in order, on a thread per camera ahead of the mailbox; a lost frame
  is read off the frame index, not the clock; and the Femto Mega's stream is
  decoded as BT.601 full range, which it codes and does not say. Its RGB
  output, and the RGB mode its calibration was read from, went on
  2026-10-06, and its host decoding the same day. *Amended 2026-10-27:*
  H.265 at 3840x2160 and 25 fps is the default.
- [**2026-09-28**](#2026-09-28--gpu-pre-processing-is-recon_sensor_utils-a-driver-hands-out-the-frame-as-captured-the-device-undistorts-depth-and-undistorts-and-converts-colour-and-fusion-reads-the-buffers-in-place-depth-and-colour-each-with-its-own-camera-rather-than-registered) —
  GPU pre-processing is `recon_sensor_utils`: a driver hands out the frame as
  captured, the device undistorts depth and undistorts and converts colour,
  and fusion reads the buffers in place, depth and colour each with its own
  camera rather than registered.
- [**2026-09-28**](#2026-09-28--a-decoded-colour-frame-never-leaves-the-gpu-the-platforms-hardware-decoder-for-h265-and-mjpeg-alike-hands-its-picture-to-vulkan-in-place-and-gpuframeprep-takes-i420-or-nv12-from-the-host-or-the-device) —
  A decoded colour frame never leaves the GPU: the platform's hardware
  decoder, for H.265 and MJPEG alike, hands its picture to Vulkan in place,
  and `GpuFramePrep` takes I420 or NV12 from the host or the device. The
  software decoding and fallbacks it kept went on 2026-10-06.
  *Amended 2026-10-25:* VideoToolbox's plane images are sampled where they
  are, not copied (the plane-image entry).
- [**2026-10-03**](https://github.com/taojin-6/volumetric_kit_grid/blob/main/DECISIONS.md#2026-10-03--asset-loading-and-export-belong-to-io-with-assimp-as-an-optional-private-backend) —
  Asset loading and export move into the installed `io` tier; Assimp adds
  optional static mesh import, while dataset and normalization policy stay
  with consumers. (grid's entry, mostly mesh files; the 2026-10-28 `io`
  entry below amends it for images.)
- [**2026-10-06**](#2026-10-06--the-familys-camera-vocabulary-is-recons-camera-tier-a-double-precision-camera-model-and-the-sensor-arrays-calibration-file-version-2) —
  The family's camera vocabulary is recon's `camera` tier: a double-precision
  camera model with one templated projection, rigid transforms, and the
  sensor array's calibration file, version 2. It links the core's base tier
  and GLM, not Vulkan. *Amended 2026-10-11:* `VR_CAMERA_ONLY` configures it
  alone. Cut back the same day, below.
- [**2026-10-06**](#2026-10-06--the-camera-tier-keeps-only-what-a-capture-uses-amends-the-entry-above) —
  The camera tier keeps only what a capture uses: the camera model and its
  check, the lens's forward template, rigid checks, and each sensor's pose
  from the calibration file's version-1 layout. Unprojection, rescaling,
  version 2 and the writer go until a caller needs them. *Amended
  2026-10-12:* version 2 and the writer are back, for calib.
- [**2026-10-06**](#2026-10-06--tests-use-no-real-hardware-amends-the-2026-09-26-orbbec-driver-entry) —
  Tests use no real hardware: the three camera tests are removed, and a
  camera is checked by hand through `fuse_orbbec`. *Amended 2026-10-07:* no
  test looks for a camera either.
- [**2026-10-06**](#2026-10-06--a-sensors-frame-is-rgbdframe-each-cameras-cameramodel-the-poses-in-double-a-sequence-number-and-the-pixels-it-holds-and-the-femto-megas-factory-extrinsic-is-made-a-rotation) —
  A sensor's frame is `RgbdFrame`: each camera's `CameraModel`, the poses in
  double, a sequence number, and the pixels it holds. The Femto Mega's factory
  depth-to-colour rotation is not one, and is made one.
- [**2026-10-06**](#2026-10-06--a-sensor-is-an-irgbdsensor-what-it-is-before-a-frame-sensorinfo-frames-polled-newest-or-drained-oldest-first-and-the-femto-mega-is-orbbecsensor) —
  A sensor is an `IRgbdSensor`: what it is before a frame (`SensorInfo`),
  frames polled newest or drained oldest first. The Femto Mega is
  `OrbbecSensor`. (The Replica source followed on 2026-10-07, and
  `sync_clock_to_host` became the SDK's global timestamps.)
- [**2026-10-06**](#2026-10-06--several-sensors-are-a-sensorarray-vendor-neutral-started-in-their-rigs-order-grouped-by-trigger-or-by-sequence-number-and-posed-by-the-calibration) —
  Several sensors are a `SensorArray`, vendor-neutral: started in their rig's
  order, grouped by trigger or by sequence number, and posed by the
  calibration. (Sequence grouping went later that day; on 2026-10-07
  `TriggerGrouper` became its internals and the tolerance 0.4 of a frame
  period.)
- [**2026-10-06**](#2026-10-06--an-arrays-set-is-prepared-in-one-batch-every-stream-of-every-sensor-one-submit-and-one-wait-host-colour-staged-on-a-thread-per-camera-first) —
  An array's set is prepared in one batch, every stream of every sensor, one
  submit and one wait; host colour is staged on a thread per camera first,
  and `prepare_set` goes. (Host colour and the threads went later that day.)
- [**2026-10-06**](#2026-10-06--every-orbbec-frame-is-handed-out-as-captured-the-sdks-host-path-orbbeccapture-and-fuse_orbbec---gpu-go-amends-the-2026-09-26-2026-09-27-and-2026-09-28-orbbec-entries) —
  Every Orbbec frame is handed out as captured: the SDK's host path,
  `OrbbecCapture` and `fuse_orbbec --gpu` go, and the Orbbec driver needs
  the video decoders.
- [**2026-10-06**](#2026-10-06--the-video-decoders-hand-out-pictures-on-the-device-only-the-hardware-path-is-fixed-per-build-a-stream-the-hardware-refuses-is-an-error-and-off-apple-vr_with_ffmpeg-needs-vr_with_cuda-amends-the-2026-09-27-decoder-2026-09-28-decoded-frame-and-2026-10-06-raw-frames-entries) —
  The video decoders hand out pictures on the device only: the hardware path
  is fixed per build, a stream the hardware refuses is an error, and off
  Apple `VR_WITH_FFMPEG` needs `VR_WITH_CUDA`. *Amended 2026-10-25:* on
  Apple the H.265 decoder uses VideoToolbox directly, with no FFmpeg, and
  `VR_WITH_FFMPEG` there is a configure error.
- [**2026-10-06**](#2026-10-06--gpuframeprep-takes-colour-on-the-device-only-and-a-sets-depth-is-staged-on-one-thread-amends-the-2026-09-28-decoded-frame-and-2026-10-06-one-batch-entries) —
  `GpuFramePrep` takes colour on the device only, and a set's depth is
  staged on one thread. (Packed host colour came back the next day.)
- [**2026-10-06**](#2026-10-06--a-sensor-array-groups-its-frames-by-trigger-only-syncmode-and-sequence-grouping-go-amends-the-2026-10-06-sensorarray-entry) —
  A sensor array groups its frames by trigger only: `SyncMode` and sequence
  grouping go.
- [**2026-10-07**](#2026-10-07--gpuframeprep-stages-a-host-frames-packed-colour-beside-its-depth-amends-the-2026-10-06-device-only-gpuframeprep-entry) —
  `GpuFramePrep` stages a host frame's packed colour beside its depth
  (`RgbdFrame::color_packed`): one copy in the pass's own batch, no host
  conversion, no threads.
- [**2026-10-07**](#2026-10-07--the-replica-source-is-an-irgbdsensor-and-the-examples-fuse-what-gpuframeprep-prepares-icameracapture-capturedframe-and-ownedframe-go-and-io-loads-depth-as-stored-amends-the-2026-08-02-sensor-tier-2026-09-14-examples-and-2026-10-06-irgbdsensor-entries) —
  The Replica source is an `IRgbdSensor` and the examples fuse what
  `GpuFramePrep` prepares: `ICameraCapture`, `CapturedFrame` and
  `OwnedFrame` go, and `io` loads depth as stored. What the iOS scanner
  calls stays until it migrates; the frame gains no float depth. The viewer
  retains its last successful keyframe through a failed preparation or fuse.
- [**2026-10-07**](#2026-10-07--a-rig-is-an-orbbecsensor-per-camera-in-a-sensorarray-on-the-sdks-global-timestamps-sdk-2106-or-later-each-camera-is-checked-against-its-sync-settings-at-open-and-written-only-when-asked-the-tolerance-is-04-of-a-frame-period-and-orbbecrig-its-start-order-and-the-public-triggergrouper-go-amends-the-2026-09-24-sdk-2026-09-27-rig-and-2026-10-06-irgbdsensor-and-sensorarray-entries) —
  A rig is an `OrbbecSensor` per camera in a `SensorArray`, on the SDK's
  global timestamps (SDK 2.10.6 or later): each camera is checked against
  its sync settings at `open`, written only when asked and checked for SDK
  normalization after a write; the tolerance is 0.4 of a frame period, and
  `OrbbecRig`, its start order and the public `TriggerGrouper` go.
  *Amended 2026-10-08:* `open_orbbec_sensors` opens a rig's cameras from its
  sync file, for every caller.
- [**2026-10-08**](#2026-10-08--a-decoded-picture-is-its-yuvimage-with-its-pts-and-encoding-the-one-description-of-a-device-picture-which-a-frame-takes-as-it-is-amends-the-2026-09-28-decoded-frame-and-2026-10-02-gpu-regression-entries) —
  A decoded picture is its `YuvImage`, with its pts and encoding: one
  description of a device picture, which a frame takes as it is.
  *Amended 2026-10-26:* NVDEC's picture is two plane images, which CUDA
  wrote; nvJPEG's alone is a buffer.
- [**2026-10-08**](#2026-10-08--apples-pixel-buffer-importer-is-public-pixelbufferimporter-in-recon_sensor_apple-on-every-apple-build-ios-included-with-no-ffmpeg) —
  Apple's pixel-buffer importer is public: `PixelBufferImporter` in
  `recon_sensor_apple`, on every Apple build, iOS included, with no FFmpeg.
  *Amended 2026-10-25:* CI compiles the importer for iOS 26.0.
  *Amended again 2026-10-25:* `recon_sensor_video` is no longer behind
  `VR_WITH_FFMPEG` on Apple, and CI's iOS check links it with the importer.
  *And 2026-10-25 (the plane-image entry):* its planes gain `SAMPLED` usage
  and a view.
- [**2026-10-11**](#2026-10-11--decoded-pictures-name-the-queue-families-that-read-them-videotoolboxs-images-are-concurrent-across-two-or-more) —
  Decoded pictures name the queue families that read them: VideoToolbox's
  images are CONCURRENT across two or more.
- [**2026-10-11**](#2026-10-11--picture-readback-publishes-buffer-ownership-for-subsequent-stages) —
  Picture readback updates ownership before subsequent reads or preparation.
- [**2026-10-12**](#2026-10-12--the-calibration-file-records-lenses-and-is-written-version-2-and-its-writer-return-with-calib-its-first-caller-amends-the-2026-10-06-camera-tier-entries) —
  The calibration file records lenses and is written: version 2 under
  version 1's `device_calibration` key, its writer and
  `rodrigues_from_rotation` return with calib; `SensorArray::open` refuses an
  unposed sensor, and the file I/O is the core's. *Amended 2026-10-18:* the
  world may be a printed AprilTag. *Amended 2026-10-27:* version 3, a
  sensor's depth offset or the colour cameras' exposure, written only for a
  corrected calibration.
- [**2026-10-16**](#2026-10-16--a-shared-glsl-pinhole-for-recon-and-its-consumers-coreshaderspinholeglsl-exported-under-the-include-root-recon_core-names-amends-the-2026-08-11-texturing-entrys-per-tier-copies) —
  A shared GLSL pinhole for recon and its consumers,
  `core/shaders/pinhole.glsl` (amends the 2026-08-11 texturing entry's
  per-tier copies): values, not structs, so scalar-layout and std430 kernels
  both call it; recon's kernels project through it, keeping only their own
  acceptance rules; and `recon_core`'s `VR_SHADER_INCLUDE_DIR` names the root
  a consumer includes it under, fetched or installed. *Amended 2026-10-22:*
  fetched only; recon installs nothing. *Amended 2026-10-28:* volume
  unprojects through a copy of its own, held to it bit for bit.
- [**2026-10-16**](#2026-10-16--a-prepared-frames-depth-is-shared-like-its-colour-gpuframeprepconfigdepth_queue_families-recon-builds-no-point-clouds-and-makes-no-calibration-measurements-the-same-k-depth-crop-and-nearest-resampling-are-accepted-on-a-stand-in-lens-pending-a-fold-check-of-the-rigs-own-amends-the-2026-09-29-rig_viewer-entry) —
  A prepared frame's depth may be shared with another queue family, as its
  colour is: `GpuFramePrepConfig::depth_queue_families`, with no vertex
  usage. recon builds no point clouds and makes no calibration
  measurements. The same-K depth crop and nearest resampling are accepted on
  a stand-in lens, pending a fold check of each rig camera's own (by hand,
  not yet run).
- [**2026-10-18**](#2026-10-18--the-calibration-files-world-may-be-a-printed-apriltag-in-opencvs-marker-axes-and-the-file-stays-version-2-amends-the-2026-10-12-calibration-file-entry) —
  The calibration file's world may be a printed AprilTag, in OpenCV's
  marker axes rather than libapriltag's, so a floor tag's z is up. A world
  is a sensor or a tag, never both, and a tag puts no sensor at the origin.
  The file stays version 2: an older reader already refuses a tag world as
  `Unsupported`.
- [**2026-10-25**](#2026-10-25--hevcdecoder-decodes-through-videotoolbox-directly-on-apple-no-ffmpeg-the-hardware-decoder-or-none-pictures-in-display-order-by-their-order-count-and-vr_with_ffmpeg-on-apple-a-configure-error-amends-the-2026-10-06-device-only-decoder-and-2026-10-08-pixel-buffer-importer-entries) —
  `HevcDecoder` decodes through VideoToolbox directly on Apple: no FFmpeg,
  the hardware decoder or none, pictures put in display order by their
  order count and written at the range the stream declares;
  `recon_sensor_video` builds on every Apple platform, and `VR_WITH_FFMPEG`
  on Apple is a configure error.
- [**2026-10-25**](#2026-10-25--the-h265-syntax-reader-is-public-in-a-target-of-its-own-on-every-platform-amends-the-2026-10-25-videotoolbox-decoder-entry) —
  The H.265 syntax reader is public (`sensor/video/hevc_syntax.hpp`,
  `recon_sensor_video_syntax`): host code on the core's base tier alone,
  built on every platform, its parsers returning `Result` and its
  `BitReader` public with them.
- [**2026-10-25**](#2026-10-25--frame-prep-samples-nv12-plane-images-where-they-are-in-a-second-entry-of-the-colour-pass-and-the-plane-copies-go-amends-the-2026-09-28-decoded-frame-and-2026-10-08-pixel-buffer-importer-entries) —
  Frame prep samples NV12 plane images where they are, through combined
  image samplers in a second entry over the colour pass's body, to the
  buffer form's words bit for bit; the two plane copies and their scratch
  buffer go. Plane images need `SAMPLED` usage and a view for frame prep,
  `TRANSFER_SRC` for readback, and the importer's carry both.
  *Amended 2026-10-26:* NVDEC's pictures are plane images too, which both
  readers take over from outside Vulkan (the NVDEC plane-image entry).
- [**2026-10-25**](#2026-10-25--a-runs-sets-the-fusion-grid-and-the-block-layout-are-library-calls-rigsets-in-the-arrays-tier-create_fusion_grid-in-tsdf-and-block_grid_params-in-volume-opening-a-rig-stays-the-callers) —
  A run's sets, the fusion grid and the block layout move from the examples
  into the library: `sensor::RigSets` in `recon_sensor_array`,
  `tsdf::create_fusion_grid` and `fusion_buckets` in `recon_tsdf`, and
  `volume::block_grid_params`, which `codec::make_decode_grid` builds with
  too. Opening a rig stays the three public calls, and a set's prep and all
  printing stay the caller's.
- [**2026-10-26**](#2026-10-26--nvdecs-pictures-land-in-two-plane-images-videotoolboxs-form-cudas-one-copy-writes-them-a-reader-takes-each-over-from-outside-vulkan-and-nothing-hands-it-back-amends-the-2026-10-08-decoded-picture-and-2026-10-25-plane-image-entries) —
  NVDEC's pictures land in two plane images, an R8 luma and an R8G8 chroma,
  VideoToolbox's form: CUDA's one copy writes them, handed to it once in
  `GENERAL`; a reader takes each picture over from outside Vulkan in the
  layout it is in, and nothing hands it back. The images take the
  decoder's queue families; nvJPEG keeps its I420 buffer.
- [**2026-10-27**](#2026-10-27--a-sensors-depth-offset-lives-in-the-calibration-file-version-3-and-frame-prep-subtracts-it-the-calibration-its-one-source-amends-the-2026-10-12-calibration-file-entry) —
  A sensor's depth offset (`SensorCalibration::depth_offset_m`, + where it
  reads too far) lives in the calibration file, version 3, which the writer
  writes only for a corrected calibration; frame prep's depth pass
  subtracts it, 0 staying no return for either sign, before the
  within-colour mask. `SensorArray` sets each pass's offset from the
  calibration, its one source, and refuses one in `Options::prep`.
  *Amended the same day:* the colour cameras' exposure joins version 3.
- [**2026-10-27**](#2026-10-27--the-orbbec-driver-defaults-to-h265-at-3840x2160-and-25-fps-and-opens-each-colour-camera-at-the-calibrations-exposure-written-where-the-camera-differs-auto-white-balance-on-and-read-back-an-array-refuses-a-sensor-opened-at-another-amends-the-2026-09-28-h265-entry-and-the-depth-offset-entry-above) —
  The Orbbec driver defaults to H.265 at 3840x2160 and 25 fps, as core
  0.25.0's scenarios do; MJPEG stays for calibration captures and
  diagnostics. Every colour camera opens at the calibration's exposure
  (`ArrayCalibration::color_controls`, a version-3 field; empty is auto
  exposure), written at each open where the camera differs, auto white
  balance always on, checked before the first write and read back after the
  last; an array refuses a sensor opened at another.
- [**2026-10-28**](#2026-10-28--io-is-two-targets-images-on-the-family-cores-base-tier-alone-mesh-files-on-mesh-amends-the-2026-10-03-asset-io-entry) —
  `io` is two targets: `recon_image_io` (images, stb) links the family
  core's base tier alone; `recon_mesh_io` (PLY, the triangle containers,
  normalization, tinyply) links `mesh`, and `recon_mesh_io_assimp` replaces
  `recon_io_assimp`. Headers and namespace are unchanged; `recon_io` is
  gone.
- [**2026-10-29**](#2026-10-29--the-camera-and-sensor-tiers-are-the-package-volumetric_kit_sensor-in-recons-sensor-until-it-has-a-repository-of-its-own-and-recons-libraries-link-its-frame-tier-alone-amends-the-2026-08-02-sensor-tier-2026-10-06-camera-tier-2026-10-16-glsl-pinhole-and-2026-10-28-io-entries) —
  The camera and sensor tiers are the package `volumetric_kit_sensor`, in
  recon's `sensor/` until it has a repository of its own: tiers `sensor_*`,
  namespace `volumetric_kit::sensor`, macros and options `VS_`, on the core
  alone. recon's libraries link `sensor_frame` alone, and nothing in the
  package includes recon or grid. The owner's decisions of 2026-10-28:
  repositories now, these names, a future stereo depth producer NVIDIA-only
  at first.

## Decision record

### 2026-08-02 — The `sensor` tier is a *contract*, not a driver collection: a capture driver lives here only if this repo can build

and test** it.** The
tier's first slice is platform-neutral C++ — an `ICameraCapture` interface, a
`CapturedFrame` POD, and the coordinate/intrinsics conversion helpers — with
**no** driver implementations bundled in. (*Amended 2026-10-07 (below):* the
interface is `IRgbdSensor`, handing out `RgbdFrame`s; `ICameraCapture`,
`CapturedFrame` and the conventions test's `FakeCapture` are gone.) The
first real source, **ARKit,
lives in `volumetric_kit_ios`**, not here. The rule that places it: a driver
belongs in this repo when this repo's CI can compile and exercise it (a
cross-platform C++ SDK — an Orbbec driver would qualify); otherwise it belongs
with the platform that can. ARKit is iOS-only Objective-C, buildable only in
an iOS cross-compile recon's CI does not run and verifiable only on LiDAR
hardware, so hosting it here would mean an Objective-C++ TU that **no CI
compiles and no test exercises** — unbuilt code that rots silently. *Refines*
the provenance policy's "its sensor driver framework is rebuilt behind a clean
`ICameraCapture`": the **interface** is rebuilt here; drivers are placed by the
buildable-and-testable rule. This mirrors the sibling precedent exactly —
gfx's windowing tier is *window-system* agnostic (`Surface` adopts a raw
`VkSurfaceKHR`, `WindowedApp` takes a consumer-supplied `SurfaceFactory`),
which is precisely why gfx ported to iOS with zero source changes and why the
app, not the library, owns the `CAMetalLayer`. The `sensor` tier is
*capture-system* agnostic for the same reason and buys the same portability.
What deliberately **does** stay here is the conversion math, because it is the
part most likely to silently ruin a reconstruction and it needs no hardware to
test: (1) ARKit's camera is +Y **up** / −Z **forward** while recon projects +Z
forward, so poses convert as `T_world_cv = T_world_arkit · diag(1, −1, −1, 1)`
— get this wrong and the mesh is smeared or doubled, never an error; and (2)
ARKit's depth (`sceneDepth`, 256×192 float metres) and colour
(`capturedImage`, 1920×1440) come from **one physical camera**, so depth is
*registered* to colour — the two share a pose and differ only by an intrinsics
scale, which lands fusion in the registered case and avoids the
occlusion/partial-colouring caveats `TsdfIntegrator::integrate` documents for
an unregistered colour camera. (`sceneDepth.confidenceMap` gates low-confidence
pixels to depth 0, so the integrator skips them.) Both are pinned by plain
host-side unit tests before any Objective-C touches them. Revisit the
placement if ARKit capture ever gains a second consumer — a headless capture
tool, a visionOS target — since that is reuse across consumers rather than one
app's platform glue; moving it down stays cheap precisely because the contract
already lives here. (*2026-09-26:* the Orbbec driver that "would qualify" has
landed, as a target of its own beside the contract — see that entry.)
**Two consequences of the implementer being out of tree, both found by review
and fixed on the same PR.** (1) *The camera-parameter structs are `core`
vocabulary, and the contract depends on `core` alone.* (*2026-10-06:* and on
`camera`, which links no Vulkan; see that entry.) They had been placed by
which tier first needed one — `DepthCameraParams` in `volume` (block
allocation unprojects a depth frame), `ColorCameraParams` in `tsdf` (the first
tier that fuses colour) — which split a matched pair across two tiers and, far
worse, buried both inside `voxel_hash_map.hpp` / `tsdf_integrator.hpp`, which
reach `core/vulkan.hpp` through the VMA allocator and the pipeline wrappers.
Including the sensor contract therefore preprocessed **105 k lines with 1 412
Vulkan handle references**, all of it paid by an ARKit Objective-C++ TU that
only wants to fill in an intrinsics struct. Both now live in
**`core/camera_params.hpp`** as `vr::DepthCameraParams` / `vr::ColorCameraParams`
— a posed pinhole camera is pure math over `Mat4f`, the same kind of vocabulary
as `Status` and the GLM aliases, and four tiers take one. `recon_sensor`
consequently links **`recon_core` and nothing else**: 0 Vulkan references,
77 k lines, the remainder GLM, which `Mat4f` requires. They stay two *types*
(88 vs 96 bytes — colour carries no depth range), and each still pins its host
packing against the GLSL mirrors in `volume/`, `tsdf/` and `texture/`
shaders.
(2) *`Result<std::optional<T>>` has no natural spelling*, and a contract whose
only implementers are out of tree cannot afford that. `Result` converts
implicitly from exactly its value type, so `return std::nullopt;` and
`return frame;` are both two user-defined conversions and neither compiles;
`return {};` is ambiguous between the value and `Status` constructors; and
`return Status{};` compiles and then **aborts**, since an OK `Status` is not a
failure. `ICameraCapture` therefore ships `no_frame()` and `some_frame(f)` and
documents why. That trap was invisible until something implemented the
interface: the tier had shipped an interface **nothing in the repo
implemented**, so `tests/sensor_conventions_test.cpp` now carries a
`FakeCapture` driven through a base-class reference — it caught the
success-path half of this on its first compile.

### 2026-08-02 — One color-space rule, and a *named* working space: 8-bit color is encoded, float color is linear, converted once at the sensor boundary and encoded once at presentation.

*Amended 2026-10-28 (the grid-contract entry):* the GLSL the kernels decode
and encode with is no longer `core`'s. The voxel colour attribute's format,
the curve included, is `volume/shaders/voxel_color.glsl`, which `volume`
exports, with its host half `volume/voxel_color.hpp`; the PLY writer encodes
with a copy of its own. `core/color_space.hpp` stays the sensor boundary's.
The volume test holds the voxel format to the published curve and the PLY
test holds the PLY copy to the voxel format, so the grid tiers' tests take
no colour vocabulary from `core`; `recon_tsdf_voxel_color_space`, in tsdf,
which links both, holds the voxel format to `color_space.hpp`. The rule is
unchanged.

Color arrives from a sensor encoded for a
display and is then **averaged** four times — the TSDF running mean
(`tsdf_integrate.comp`), the marching-cubes edge lerp
(`marching_cubes_common.glsl`), texture filtering, and gfx's shading multiply
— before a `_SRGB` swapchain encodes it a second time. Averaging is linear and
a display encoding deliberately is not, so each is wrong by construction, and
wrong *quietly*: linear `0.0` and `1.0` fuse to `0.214` rather than `0.5` (a
plausibly darker surface, not a visible error), a few percent between similar
samples, worst across high-contrast pairs — so it concentrates on silhouettes
and depth discontinuities, exactly where a reconstruction gets inspected.
Surfaced by the iOS scanner, where only the double-encode is visible.
The rule decides the integrator, marching cubes, the atlas format, the vertex
color and the render targets with no judgment left over, and it is not
arbitrary: eight bits suffice for color *because* a display curve spends them
perceptually, so 8-bit storage stays encoded (naively linearizing trades a
blending error for banding in the darks); a float has the range to be linear,
so it is. It says *color* deliberately — a future 8-bit confidence or
occupancy channel is a number and carries no curve.
**"Linear" alone does not name a space, so one is named**: the working space
is **linear BT.709/D65** (the linear half of sRGB), its canonical encoded form
is that space through the *exact piecewise* sRGB transfer, full range, and
every 8-bit color below the sensor boundary is in that one form. Without this
the rule is under-determined — a driver can declare four transfers and three
primaries, so "encoded" would name twelve different things — and, worse,
`Primaries` would be a label nothing consumes: two sensors with different
primaries produce linear values in *different RGB bases*, and averaging those
is wrong the same way. The boundary conversion is therefore two steps, decode
the transfer then 3×3 into the working basis, both identity for a BT.709
source (so the ARKit path is a genuine no-op). Cost booked: BT.709 is the
narrowest declarable gamut, so a P3/BT.2020 source clips — widening the
working space means widening the storage with it, both named constants rather
than assumptions spread through the kernels. Presentation is the symmetric
half: the surface declares its space (`VkFormat` + `VkColorSpaceKHR`) exactly
as a driver does, so a P3/HDR display is reached by that declaration changing
and the presentation step gaining the primaries half — nothing else moves.
**Placement follows the 2026-08-02 camera-params precedent, not the sensor
tier.** `ColorEncoding` and the curve live in `core` (`core/color_space.hpp` +
a `core/shaders/color_common.glsl` mirror), because `sensor` branches off
`core` *beside* the fusion tiers — so `tsdf` and `mesh`, which do the decoding
in GLSL, cannot include from it — and because a curve four tiers need is
vocabulary. `sensor/color_conventions.hpp` keeps what is genuinely the
boundary's — `to_canonical`, which walks a frame. *Implementing this moved
the line*: the `ColorEncoding` type **and** `is_canonical` went to `core` too,
because `TsdfIntegrator::integrate` **refuses** a non-canonical colour frame
rather than fusing it through the wrong curve, and `tsdf` cannot include from
`sensor` — a property of the type belongs with the type. That refusal is what
makes "convert once at the sensor boundary" a contract rather than a hope.
The cross-tier GLSL include (the repo's first) is spelled like the C++ header
path so its provenance is visible at the include site; `vr_compile_shaders`
passes `src/` as the shader include root.
The declaration rides **beside** the camera
(a field on `sensor::CapturedFrame` / `tsdf::ColorFrame`), never *inside*
`ColorCameraParams`: that struct is uploaded verbatim under scalar block
layout, pinned at 88 bytes with GLSL mirrors in `tsdf/` and `texture/`, so a
field there would spend a cross-tier shader ABI change to carry what the
kernel needs at most as a push constant. No `Range` field — the contract
requires full-range packed R'G'B', so the YCbCr matrix and range expansion are
the driver's, applied before the contract; `Primaries` earns its place because
the working space gives it a consumer, and a field nothing consumes is a label
free to drift. ARKit declares `{Bt709, Bt709}` (its YCbCr→R'G'B' uses a BT.601
*matrix*, independent of the transfer function — conflating the two is its own
silent error), and `Transfer::Bt709` 8-bit content is **accepted as sRGB
rather than converted**: they differ by a couple of codes in the toe, which is
what display pipelines assume anyway (BT.1886), and converting would cost a
per-frame pass over 1920×1440 to buy nothing.
Two consequences worth stating. (1) **Presentation is three sites, not one** —
the swapchain (already `_SRGB`), `fuse_render`'s offscreen target (`_UNORM`
today; it renders straight to PNG, so linear values would simply come out
dark, and it is the CI-visible leg), and host export, where PLY `uchar` colors
are read as sRGB by every external viewer and must be encoded on write while
glTF `COLOR_0` is linear and passes through — the exporters cannot share one
path. (2) **`volumetric_kit_gfx` needs no change at all**: every format is
caller-set, the swapchain already prefers `_SRGB`, and `hybrid_mesh.frag`'s
`albedo *= ambient + diffuse` was never the bug, only its operands. Both the
host curve and its GLSL mirror must be the *exact piecewise* sRGB function,
not `pow(x, 2.2)` — hardware `_SRGB` sampling uses the exact curve and that
shader selects atlas-vs-vertex-color per triangle across one surface, so an
approximation shows as a seam exactly where texturing stops. Storage stays
`uint32` (convert in the shader) and is measured; the escalation trigger is
**not** banding but the running mean *latching* — re-quantized to 8 bits it
stops moving once the per-frame delta falls under half a code. A historical
**host arithmetic experiment**, with `max_weight = 5` and a 2 m observation
(weight 0.25), repeatedly unpacked the current and target codes to linear,
averaged them with weights 5 and 0.25, and packed the result back to sRGB: the
mean stops **~10 codes short, uniformly across the range**
(0→64 settles at 55, 0→255 at 245), and a gap narrower than ~10 codes never
moves the voxel at all — the residual is range-independent because the sRGB
curve turns a fixed fraction of the linear gap into a roughly fixed number of
codes. So fused colour accuracy is ceilinged at ~4%, a *convergence* limit
rather than a precision one, which is precisely why banding was the wrong
thing to watch. The 2026-10-10 test cleanup keeps this experiment here: it
models the arithmetic and does not exercise the production integrator, so it
cannot detect a change to fusion or storage. The color-space tests retain the
actual conversion and host/shader parity checks. Authoritative detail:
DESIGN.md → "Color space".

### 2026-09-14 — The examples poll their frames through the sensor contract: the Replica reader is an `ICameraCapture`, a source says when it is exhausted, and a frame kept past the next poll is copied — the last one included.

*Amended 2026-10-07 (below):* the Replica reader is `ReplicaSensor`, an
`IRgbdSensor`, and the examples fuse what `GpuFramePrep` prepares from its
frames. A frame holds its pixels, so nothing is copied to be kept:
`OwnedFrame` is gone, and the viewers keep the prepared `DeviceFrame`. The
move constructor is defaulted, the frames to play moving with their vector,
so a moved-from sensor plays nothing; assignment is deleted.

The sensor tier had shipped a contract with no producer and no consumer in
this tree. Its only implementer was the ARKit driver in `volumetric_kit_ios`,
its only in-repo exercise the test fake, and the three fuse examples — the
code that actually drives the pipeline — read frames through a dataset API of
their own (`ReplicaDataset::frame(i)` returning a `FrameView`, plus
`make_depth_camera` to build the camera structs the loop then patched per
frame). With an external RGB-D source (Orbbec) next, that left a live driver
with nothing to plug into: every consumer was written against the dataset,
not the contract, and the contract's ergonomics on a real frame path — a
frame that borrows its pixels, a range stamped by the driver, a pose that
rides on the frame — had never been paid for by anything that ran.

**What changed.** `ReplicaDataset` / `FrameView` / `example_camera.hpp` are
gone, replaced by `examples/common/replica_capture.hpp`: a `ReplicaCapture
final : sensor::ICameraCapture` that plays the sequence back through
`poll()`. All three examples take an `ICameraCapture&` for the fuse loop and
touch the concrete type only where it is opened (options, `preload`,
`frame_count`, the colour camera the viewers size their render camera from).
No compatibility layer: the repo is under active development, and a shim
would have kept the loop's old shape alive beside the new one. Beside it, two
header-only helpers in `examples/common` that the review of the first cut
asked for: `rgbd_frame.hpp` (the one frame type, below)
and `fuse_frame.hpp` (one frame's allocate-and-integrate, which three loops
had each carried a drifting copy of — one printed the overflow and timed the
grow, one did neither, and the encoding hand-off into `ColorFrame` was the
kind of line a fourth copy drops with no error, since a `ColorFrame` left
defaulted *declares* canonical). The contract itself gained one member, and
`ReplicaCapture` gained a host-only test. Three points of the design carry
the decision; the first and third are corrections the review made to the
first cut, recorded as such.

1. **A source says when it is exhausted; the loop does not infer it from an
   empty poll.** The contract's `no_frame()` means "nothing this tick" for a
   live device, and a replay has no such state — every poll yields the next
   frame until there is none — so the first cut treated the first empty poll
   as the end of the sequence, rejected an `exhausted()` accessor as "a branch
   nothing reaches today", and deferred a wait-and-retry to "the day a live
   source sits beside the replay". The review showed that made the headers'
   own claim false as merged: a live driver constructed in place of
   `ReplicaCapture` returns `no_frame()` on its first idle tick and all three
   loops `break` — `fuse_replica` writes a header-only PLY and exits 0 with
   "done: fused 0 frames" — and the deferral could not be executed at the
   loop at all, because replay-exhausted and live-idle were one signal. So
   `ICameraCapture::exhausted()` exists: non-pure, defaulting to `false` (a
   live device is never exhausted, only stopped), so the test fake and the
   out-of-tree ARKit driver compile untouched, and overridden by the replay
   to `next_ >= end_`. Every loop now waits a millisecond and polls again on
   an empty poll from a source that is not exhausted, and ends on one that
   is; nothing on that branch is deferred any more. A decode failure on a
   frame that is present stays a real error and stops the run, which is what
   `fuse_render` already required after a partial-room PNG once exited 0.

2. **The frame cap, the stride and the depth gate are the capture's options,
   stamped on the frame — and the probe looks only at the frames those options
   select.** The ARKit driver stamps its own range constants on every frame's
   depth camera; the replay does the same with the CLI's `--min-depth` /
   `--max-depth` (now exposed by all three examples, where the viewers had
   offered only the far end and so refused `--max-depth 0.1` with a message
   naming a knob they never took — the refusal is also re-labelled as this
   capture's, with both values in it), deriving the depth camera from the
   colour one through `depth_from_registered_color` at `open`. That
   validates the range before a frame exists, keeps the two poses from a
   single trajectory entry so they cannot drift, and is the only copy of the
   intrinsics: the first cut kept a third `CameraModel` beside the two camera
   structs and size-checked the decode against *it*, so a later depth
   downscale could have stamped one extent and held another. `frame_count()`
   is probed from disk at `open` rather than read off the trajectory: room0
   lists 2000 poses against 400 images, which is what `fuse_replica`'s banner
   printed ("dataset: 2000 poses") and what the viewer's "fused N / M" showed
   under a `--max-frames` above 400 (its default of 400 clamped it; the first
   cut of this entry claimed the default over-promised, and it did not), while
   the old `preload_bytes_projected` probed for itself but only under
   `--preload`. The first cut probed every contiguous index from 0 and only
   afterwards applied the limit: the review measured 801 file opens to learn
   `end_ = 5` at `--max-frames 5`, on `fuse_viewer`'s main thread after the
   window was up, and a sequence thinned on disk to every N-th frame — which
   `main` fused completely under `--stride N` — played one frame with
   nothing said. The probe now visits index 0, `stride`, `2·stride`, … below
   the limit and stops at the first absent one, through a `stat` rather than
   an `ifstream` open (~10x cheaper for the same answer), so it scales with
   the frames the run will play and an index the stride never visits cannot
   end the sequence. One probe, one count, and a missing image at poll time
   becomes the error it is.

3. **A frame kept past the next poll is copied — every one, the last
   included.** `CapturedFrame` is a view the capture recycles on the next
   `poll()`. `fuse_render` used to re-read its keyframe from the dataset after
   fusion — random access a live source cannot offer — and now retains a copy
   of that frame as it goes by (the `--follow` frame, else the middle of the
   sequence, chosen against `frame_count()`). The first cut let `fuse_viewer`
   hold the *last* polled frame as a borrowed view for its final texture
   pass, "valid precisely because nothing polls after the loop" — and that
   named the wrong boundary. The retention sat at the bottom of a successful
   iteration, so every failure exit (a map that would not allocate after
   five rounds, a resize that failed, an integrate error) left the loop after
   the poll that had already move-assigned the next decoded frame over the
   one the view pointed at, and the final `publish` uploaded ~3 MB of freed
   depth and packed a freed colour plane — heap-use-after-free under ASan,
   a corrupted atlas or a SIGSEGV in Release, on the default streaming path.
   The ordinary exit survived only because `ReplicaCapture`'s exhausted
   branch happens to return before touching its frame, a detail no document
   promised and no test pinned; the terminating empty poll *is* a poll under
   the contract, and the ARKit driver rotates its buffers on one. So both
   viewers keep a keyframe the same way, in an `RgbdFrame` of their own —
   the type `ReplicaCapture` decodes into, made public and given the whole
   header, so there is one frame type in the examples and `CapturedFrame` is
   its view (`assign` from the view, capacity reused; `view()` back to the
   contract type, built on access so the defaulted moves stay correct), and `fuse_viewer` copies
   the newest fused frame at the bottom of every iteration: the review
   measured the 6.5 MB retain at 0.08–0.09 ms at `-O2` on an M5 Max — ~5% of
   a preloaded fuse iteration, ~1% of a streaming one. `ReplicaCapture`'s
   header no longer promises anything about the empty poll, and the in-loop
   publish still consumes the frame synchronously, as it may.

**What else the review corrected.** `ReplicaCapture`'s defaulted move pair
emptied the vectors but copied `end_`, `running_` and `next_` — the shape the
2026-08-03 entry records for the extractor — so a moved-from capture still
claimed frames and, polled, indexed an emptied pose table; the pair is
hand-written to leave the source default-constructed (nothing on disk, not
running, exhausted), and `load()` guards the pose table as well as `end_`.
The `current_borrowed_` member and its resolve-on-access `current()` had one
caller, in the same `poll()` that set both members under an "owned wins"
precedence rule; a `poll()`-local pointer does the same job with one member,
and the comment that said the decode was "swapped in only on success" now
says what the code does — move-assigned in place on success, releasing the
previous frame's storage, which is exactly the property point 3 tripped
over. `pack_color_rgba8` re-derived, byte for byte, the identity branch of
`sensor::to_canonical` and never read the frame's encoding declaration, so
the atlas leg assumed canonical bytes where the fuse leg checks them; both
viewers now run the keyframe through `to_canonical` (the identity plus an
opaque alpha for Replica's sRGB JPEGs) into the packed words an RGBA8 upload
reads on the little-endian hosts every Vulkan platform here is, and the
function is gone.

**What the review asked for and did not get.** A host-only test of
`ReplicaCapture`, since the examples had come to depend on three of its
behaviours — the empty poll and a failed decode leaving the previous frame
intact, and `frame_count()` agreeing with `preload()` and `poll()` — that no
test pinned and no CI leg ran (`viewer.yml` is build-only; nothing in `tests/`
links `vr_example_common`). Declined. The first two dependencies are gone
with the copy, and a test that reaches `poll()` at all needs decodable
images — an encoder in the test, or a JPEG as a byte array in source — which
is a fixture out of all proportion to an example's file reader. The replay
is verified the way the examples are, against the dataset (below); the
contract member it implements is pinned where the contract is tested.

**Verified.** `fuse_replica room0 --max-frames 60 --stride 3 --device-extract`
after the review's fixes: 20 frames played, 6 057 blocks, 628 833 vertices /
209 611 triangles — the same figures the first cut matched against the
pre-refactor Release binary (where the two PLYs were identical as triangle
multisets, the byte diff being the atomics' arrival-order permutation).
`fuse_render --follow 30 --max-frames 100` textures the retained frame-30
keyframe in register through the `to_canonical` atlas (PNG inspected);
`fuse_viewer --max-frames 12 --remesh-every 4`, *streaming*, fuses 12/12 and
publishes the final mesh textured with the copied last frame, nothing on
stderr. 27/27 tests pass; the examples build under `-Werror` on MoltenVK; an
ASan/UBSan build of the whole tree, viewer included, runs the suite and both
of those example runs clean.

### 2026-09-24 — The Orbbec SDK is a prerequisite behind `VR_WITH_ORBBEC`: installed once for the family, found, and never fetched.

*Amended 2026-10-07 (the rig entry, below):* the floor and CI's pin are
2.10.6, for its global timestamps, and CI installs it through one composite
action, `.github/actions/install-orbbec-sdk`.

The next capture source is the Orbbec rig — Femto Mega units, calibrated by
`volumetric_kit_calib` for their poses — and its driver needs the Orbbec SDK
v2. The SDK is a prebuilt binary (a universal dylib plus plugin libraries in
`lib/extensions/` that it loads from beside itself), and recon is not its only
consumer: calib's planned `capture/desktop` tier drives the same cameras.

**The rule.** `VR_WITH_ORBBEC` (OFF by default) requires an installed SDK
≥ 2.9.3 and exposes `ob::OrbbecSDK`; `cmake/vr_orbbec.cmake` does the finding.
The SDK is installed **once, outside every repo** — the family convention is
`<workspace>/third_party/OrbbecSDK_v<version>` — and every build that needs it
points at that copy with `-DOrbbecSDK_ROOT=<sdk>`, or the `OrbbecSDK_ROOT`
environment variable set once for every repo. It is linked in place, so no
build tree or worktree carries a copy of its own, and nothing is copied beside
a binary for it to run. That is the Vulkan SDK's treatment, not VMA's.

**Why not a pinned download.** A FetchContent of the release tarball, hashed
per platform, with a local-copy override, was the first proposal: it would
have let a fresh clone and CI build with no setup. The owner's call was to
make the SDK a requirement instead — one copy the whole family links, the
version stated as a floor this repo checks, not a download each build tree
repeats. Vendoring it into git (≈12 MB per platform) was ruled out, as was
assuming a system install (`/usr/local`: sudo, and a machine-wide version that
drifts under every repo); a system install still works, it is just not
assumed.

**Why off by default.** The SDK ships macOS, Linux (x86_64, arm64) and Windows
builds and no iOS one, and recon cross-compiles to iOS unchanged (the
2026-08-01 decision); a required SDK would break that build, and make every
consumer that never touches a camera install it. The iOS half is checked, not
just stated (the 2026-08-04 rule): with the option on, an Apple target other
than macOS stops the configure and says to turn it off, where it would
otherwise find the macOS SDK and fail at link time on the wrong platform's
dylib.

**Two quirks of the SDK's package files**, handled once in the module rather
than rediscovered by each caller:
- `OrbbecSDKConfig.cmake` sits in `<sdk>/lib`, which none of `find_package`'s
  standard layouts search under a prefix. Without `PATH_SUFFIXES lib`,
  `OrbbecSDK_ROOT` and `CMAKE_PREFIX_PATH` both miss it; only
  `OrbbecSDK_DIR=<sdk>/lib` finds it.
- The version file is `OrbbecSDKVersion.cmake`, not
  `OrbbecSDKConfigVersion.cmake`, so `find_package` never reads it and a
  versioned `find_package(OrbbecSDK 2.9.3)` rejects even a matching SDK. The
  module reads `PACKAGE_VERSION` out of that file and checks it itself: the
  floor *and* the same major version — the rule the file encodes — since the
  first cut's bare `VERSION_LESS` let a 3.0.0 through.

**A named root is authoritative.** `find_package` caches the directory it
found in `OrbbecSDK_DIR` for the life of the build tree, so review found that
re-pointing `OrbbecSDK_ROOT` at an upgrade was ignored without a word — the
smoke test still passing, since it compared the runtime against the same
stale configured version — and that after an "older than" error, re-pointing
it at a newer SDK repeated the error (the 2026-08-04 rule: a staleness the
caller cannot see is the library's to check). So whenever `OrbbecSDK_ROOT` is
set, as a variable or in the environment, the module drops the cached
directory and finds again. With no root named the cache stands, so a tree
configured with a one-off `OrbbecSDK_ROOT=<sdk> cmake …`, or reopened by an
IDE that never read the shell profile, still re-runs cleanly. A fuller
version — walking every candidate on the search path through its own version
file, skipping the incompatible ones, keeping a record to tell a
caller-supplied `OrbbecSDK_DIR` from the cache's — was written and dropped as
not worth its ~170 lines: the root is how the family points at the SDK, the
cases it added (two SDKs on one search path, a 32-bit build) do not occur
here, and an SDK the check rejects stops the configure with the fix named.

**CI installs it the way a developer does.** `_build.yml` takes an
`orbbec_sdk` input, on for the ubuntu-24.04 and macos-26 legs and off for the
other two, so both configurations build on every PR: a step downloads the
pinned 2.9.3 release for the runner's platform, checks its SHA-256, unpacks it
under `RUNNER_TEMP` and exports `OrbbecSDK_ROOT`, and the configure passes
`-DVR_WITH_ORBBEC=ON`. That is not the rejected pinned download: the *build*
still only finds, and the step is the one-time install a developer does,
repeated per job because a CI job starts with nothing. The pin is the
module's floor, so CI builds against exactly the oldest SDK it accepts; the
two move together.

**The smoke test's own fixes.** The SDK's `setLoggerSeverity` sets *every*
sink, the file one included, so calling it after `setLoggerToFile(OFF, "")`
switched file logging back on and the test wrote `./Log/OrbbecSDK.log.txt`
into `build/tests` — the thing its comment said it prevented. It now sets each
sink separately (`setLoggerToConsole(WARN)`), and both calls sit inside the
`try`, since they throw `ob::Error` like the rest. The device-list strings go
through a null guard before `printf("%s")`. And the listing is described as
what it is: `queryDeviceList()` blocks about a second probing the network, and
still an Ethernet camera can miss it from cold — polled every 250 ms on the
rig, the three Femto Megas answered the first query in 12 of 14 runs; in one
of the other two none answered for ~2.5 s before all three did, and in the
last none answered in the whole ~4 s window. So "0 device(s)" is a
log line, not proof that none is attached, and the driver's open path will
need to wait for discovery rather than trust one query. On Windows, where a
DLL is found on `PATH` rather than through an rpath and the SDK is not copied
beside the test (it loads `extensions/` from beside its own DLL), the test
prepends the SDK's DLL directory through `ENVIRONMENT_MODIFICATION` (CMake ≥
3.22).

**Verified** on macOS (Apple silicon) against the 2.9.3 install at
`~/ws/volumetric_kit/third_party/OrbbecSDK_v2.9.3`. `recon_orbbec_sdk_smoke`
asserts the runtime version equals the one configured against (the failure a
linked-in-place install invites is a different copy answering at runtime, not
a build error), then opens an SDK context and enumerates devices: it lists the
rig's three Femto Megas, all over Ethernet, passes with none attached, and
leaves no `Log/` behind. The module was run against scratch SDKs (2.7.6,
2.9.3, 2.10.0, 3.0.0) under both CMake 3.21.4 — the declared minimum — and
4.4.3, with identical results: re-pointing `OrbbecSDK_ROOT` (by `-D` and by
environment) in an existing tree switches SDKs, including out of an "older
than" error; 2.7.6 and 3.0.0 are each refused with the fix named; dropping the
environment variable in a later configure keeps the SDK already found, a
mistyped `-DOrbbecSDK_ROOT` stops the configure, and a found SDK deleted from
disk is not kept; and an iOS target stops at configure.
With the option on, all 28 tests pass under `-Werror` in Release; with it off,
the build is unchanged — 27 tests, the SDK never looked for.

**Open.** The Windows `PATH` has never run — no CI leg builds on Windows at
all (a `TODO(ci)` beside it). The rig's driver has not landed; under the
2026-08-02 rule it may, now that CI builds and tests against the SDK, and its
open path owns the discovery wait above. (*2026-09-26:* it has landed, discovery
wait included — see that entry.)

### 2026-09-26 — The Orbbec driver lands as `sensor/orbbec`, a target of its own: it undistorts colour and then registers depth to it on the host, and it reads the rig's sync roles without writing them.

*Amended 2026-10-06 (below):* the host path is gone. `OrbbecCapture`, the
SDK's undistortion and registration, and the host conversion are removed;
every frame is handed out as captured and prepared on the GPU.
*Amended 2026-10-07 (the rig entry, below):* `OrbbecSensor` writes a
camera's sync settings when asked (`apply_sync`), and only where they
differ; software triggering is refused after any write.

The 2026-08-02 rule admits a driver this repo can build and test, and CI builds
against the Orbbec SDK (the 2026-09-24 decision), so the Femto Mega driver lives
here: `OrbbecCapture final : sensor::ICameraCapture`, target
`volumetric_kit::recon_sensor_orbbec`, built only with `VR_WITH_ORBBEC`.

**Placement.** A target of its own, so `recon_sensor` stays what an out-of-tree
driver compiles against — no vendor SDK, no Vulkan. The SDK is `PRIVATE` behind
a pointer to an implementation, so the public header names no SDK type; the
driver installs its own headers, and the package config re-finds the SDK, only
when the package was built with it. The config hints the SDK the build checked
and does not re-check a version itself: a consumer that points
`OrbbecSDK_ROOT` elsewhere chose that SDK. The folder names the driver rather than a
category (`sensor/drivers/`): the rule keeps platform-bound drivers out, so a
second may never come. calib's planned capture tier is the known second
consumer and links this target through the package; if the reuse outgrows
linking, moving the folder to a shared repo is the whole change.

**The frame path: undistort colour, then register depth to it, then convert.**
Measured on the rig's primary (CL2A141000N, 1280x720 colour, 640x576 depth):
- The SDK's depth-to-colour alignment **ignores the colour lens.** Aligning a
  raw frame set and one whose colour was undistorted first gives bit-identical
  depth, and the registered depth matches an independent *pinhole* projection
  to 0.90 mm mean / 1.36 mm RMS over 1,755 samples; predicting the distorted
  colour location instead misses by 28.6 mm RMS. The colour lens moves pixels
  5.25 px on average (10.5 px at most), so without undistorting first every
  colour sample would sit that far from its depth. Undistorting first puts both
  images on one pinhole camera — the registered case `tsdf` and `texture` are
  written for.
- The SDK's `UnDistortionFilter` runs in the right direction: 0.25/255 per
  channel from an independent forward remap, against 17.6 for the opposite
  direction and 14.3 for none.
- The depth lens is the large correction (31.5 px mean, 220.9 px at the edge)
  and the SDK's registration makes it; nothing here re-implements it.
- Both filters keep the intrinsics and zero the distortion, so both of a
  frame's cameras are the colour camera, the depth one derived through
  `depth_from_registered_color`. The first pair of every start is held to that
  — both images reporting the stamped intrinsics, colour reporting no
  distortion, a NaN failing — and a mismatch is an `IoError` rather than a
  misprojected reconstruction (the 2026-08-04 rule). Registration is asked for
  at the colour image's full size (`setMatchTargetResolution(true)`) rather
  than left to the SDK's default.
- Cost on the polling thread (M5 Max, -O2): undistort 1.33 ms + register
  1.76 ms median; a whole `poll()` that hands out a frame, 3.56 ms median /
  4.20 ms worst. The SDK's frame thread only swaps the newest pair into a slot,
  so a pair nobody polls costs nothing past the SDK's own decode.

Depth to colour, not colour to depth: the texture tier's atlas must be
registered to the frame's camera, and colour sampled through depth would be a
640x576 atlas with a hole wherever depth has none. The camera's **hardware**
registration (`ALIGN_D2C_HW_MODE`) is deferred — a `TODO(sensor)` — because it
ships 2.5x the depth pixels over Ethernet, which three cameras on one link will
feel, and how it treats the colour lens is unmeasured.

**Colour is requested as RGB; the wire carries MJPG either way.** At 1280x720 an
RGB and an MJPG request move the same traffic (31 and 27 Mbit/s averaged over a
run), since the camera compresses and the SDK decodes on its own thread. The
camera also encodes H.265, which the SDK passes through undecoded: 21 Mbit/s
against MJPG's 185 at 4K, where MJPG reaches only 16.7 of 25 fps. A decoder of
our own is the step to 4K on three cameras, not this one.

**Sync roles are read and never written.** The rig is hardware-synced:
CL2A141000N is the primary, CL2A141000G and CL2A14100A4 are `SECONDARY_SYNCED`.
A secondary opened alone produced one frame set in ~6 s — to a consumer, a
camera that is merely slow. That contradicts the SDK's description of the mode
(capture at once, re-time to a signal when one arrives), so `waits_for_primary`
follows the measurement. The role persists on the camera, and a driver that
"fixed" it for a one-camera run would break the rig for the next one, so
`OrbbecDeviceInfo::sync_mode` reports it and nothing writes it; starting a rig
in its wiring's order belongs to the multi-camera source. Software triggering
is refused at `open`: it waits for a host trigger this driver never sends. A
camera reporting its image mirrored, flipped or rotated is refused too — its
intrinsics describe the unmirrored image, and nothing downstream could tell.

**Discovery waits.** `open` re-queries every 250 ms until the camera answers or
`discovery_timeout_ms` (8 s) runs out, since one query is no proof of absence
for an Ethernet camera (the 2026-09-24 measurement) — and, by the same
measurement, a list of one is no proof a camera is alone. So an empty serial
waits out the whole window and is refused as soon as a second camera answers;
a named camera opens the moment it answers, which is why the examples and the
hardware test name one.

**The SDK's callbacks hold a shared mailbox, not the capture.** The SDK calls a
*copy* of the device-changed callback after releasing its lock, so
unregistering does not wait for a call in flight, and a pipeline stop that
throws may keep the frame callback. Both capture the pending pair, the fault and
the counters by `shared_ptr`, so whichever of the capture and the SDK lets go
last frees them.

**Failures are graded.** A disconnect is permanent — the handle is gone, and a
camera that comes back is a new device — so the capture becomes `exhausted()`
and `start()` refuses it. A pair the SDK fails on is skipped and counted
(`stats().failed`), and only 30 in a row (a second at 30 fps) are an `IoError`,
since a camera whose every pair fails would otherwise look merely slow. A pair
that contradicts the negotiated profile (size, format, intrinsics) is an error
at once.

**The hardware test opens only the camera it is told to.**
`recon_sensor_orbbec_capture` skips unless `VR_ORBBEC_TEST_SERIAL` names a
camera — discovery is a broadcast, the cameras are exclusive, the Debug and
Release legs run at once, and a person may be using the rig — and skips on a
secondary. `recon_sensor_orbbec_conversion` pins the arithmetic with no camera:
units (0 kept exact), byte order, the camera struct (a negative `int16_t` size
refused, not wrapped), the sync-mode mapping, and the option checks `open`
makes before touching the SDK.

**A live camera is its own example, `fuse_orbbec`,** not a flag on
`fuse_replica`, whose preload, stride, dirty-block survey and phase breakdown a
camera takes none of. It fuses through the shared
`examples/common/fuse_frame.hpp`, stops after `--frames` frames (300 by
default; a camera is never exhausted), and gives up after 10 s without a frame,
naming a sync secondary as the likely reason.

**Verified** on macOS (Apple silicon) against the 2.9.3 SDK and the rig. The
capture test against CL2A141000N: 30 frames, 0 dropped, 0 failed, 63.6% of
registered pixels carrying depth, increasing timestamps, restart and moves; on
CL2A141000G it skips as a secondary. `fuse_orbbec --serial CL2A141000N`:
29.6 fps (the camera's rate), none dropped, a planar floor with its marker's
texture on its own geometry; on the secondary it stops after 10 s naming the
role, and with no serial and three cameras it refuses with all three serials.
With the option on, 30 tests pass in Release under `-Werror` and clean under
ASan/UBSan (the capture test and a `fuse_orbbec` run included), and a
`BUILD_SHARED_LIBS=ON` build links; with it off the build is unchanged — 27
tests, no `orbbec` file installed, no SDK asked for. A consumer project against
an install links the target through `find_package`, and `fuse_replica` on 60
room0 frames produces the same 636,624-vertex mesh as before.

**Open.** The disconnect and skip-and-count paths are implemented but
unexercised: no run has unplugged a camera, and no healthy run fails a pair.
Each camera stamps frames with its own hardware clock, so a rig's frames are not
yet on one clock. That, HEVC decoding, GPU pre-processing and the hardware
registration are `TODO(sensor)`s in `orbbec_capture.cpp`; Windows is unbuilt.

### 2026-09-27 — The rig is `OrbbecRig`: it checks the cameras against the rig's sync configuration and writes it only when asked, starts the secondaries before the primary, keeps the cameras on the host's clock, and builds each set around a primary frame; poses come from the calibration file.

*Amended 2026-10-06 (below):* a set's frames are as captured, read through
`poll_set`; the rig processes nothing on the host and is no longer an
`ICameraCapture`.
*Amended 2026-10-07 (below):* `OrbbecRig` is gone. Each camera is an
`OrbbecSensor`, checked against its entry of the sync file at `open` and
written only with `apply_sync`; a `SensorArray` starts and groups them, on
the SDK's global timestamps rather than `enableDeviceClockSync`, within 0.4
of a frame period rather than 5 ms. A read after a write comes from the
SDK's cache, so it does not prove the write, as this entry says it does.

**The rule.** `OrbbecRig` (`sensor/orbbec/orbbec_rig.hpp`) is opened from the
rig's sync configuration (`orbbec_sync_config.hpp`: the SDK's
`MultiDeviceSyncConfig.json` layout, as in `femto_mega_sync.json`), which
names the cameras, the primary, and each one's sync settings. `open` compares
every camera's stored settings with it and refuses a difference, naming each
field, unless `apply_sync_config` writes the configuration to the cameras that
differ. It then requires one primary and its secondaries. `start` has the SDK
sync the cameras' clocks to the host's (`enableDeviceClockSync`) and starts
the secondaries first, since the primary's first trigger is what they wait
for. Each set is a primary frame plus each secondary's frame within
`sync_tolerance_us` of it on that clock, and only a set's frames are
processed, through the `CameraStream` that `OrbbecCapture` now wraps.
`poll_set()` hands out one set per trigger, with an empty slot for a missing
secondary; `poll()` hands out the same frames one at a time, so the fuse loop
reads a rig as it reads one camera. A caller reads with one of the two per
`start`.

**Which sync fields are compared.** On the Femto Mega, a written `SECONDARY`
reads back as `SECONDARY_SYNCED`, `trigger2ImageDelayUs` reads back as the
depth delay, and `framesPerTrigger` as 0; writes persist (a 200 µs delay read
back from a fresh process). So the comparison covers the role (primary or
secondary), the depth and colour delays and the trigger output, and
`framesPerTrigger` only in a triggering mode; anything more would refuse
every open. The camera has one secondary mode, and it waits for the trigger:
started alone for 4 s, the three secondaries delivered one frame between
them. The SDK answers a read in the same process with what was written, so
the check after a write proves the write, and the next `open` proves the
camera. The lab rig's file is committed as `config/femto_mega_sync.json`,
which a test keeps valid; the library never reads `config/` itself.

**Why these numbers.** On the four-camera rig over the 2.5 Gbit/s cable,
720p colour, 640x576 depth, 30 fps:
- With clock sync, a secondary's frame lands +0.4 to +1.7 ms from the
  primary's, stragglers to 3.4 ms. Without it the cameras' clocks are ~30 s
  apart and host arrival times jitter by ±20 ms. The 5 ms tolerance clears
  every straggler; half a frame period (16.7 ms), where one frame could match
  two triggers, is refused.
- Each secondary drops 1-6% of its frames and the primary none, so 90-99% of
  sets are complete. A set goes out once each missing secondary has sent a
  later frame or stayed silent for 1.5 frame periods, rather than costing the
  other cameras their frames.
- The primary's frames name the triggers, so its first triggers, which reach
  the secondaries ~525 ms before its own first frame reaches the host, a late
  secondary frame, and a camera whose clock is off by more than the tolerance
  all cost frames, never an extra set.
- A set's four frames take ~11 ms to process. A rig camera's mailbox keeps
  its four newest frames, so a poll slower than the cameras still finds one
  trigger in all of them: at 58 ms a set (ASan), one-frame mailboxes left
  75.6% of sets complete, four-frame ones 98.9%.
- Over Wi-Fi the rig ran at 13-28 fps. Depth crosses the network uncompressed
  at ~170 Mbit/s per camera, so the rig needs the wired link.

**The calibration file** (`sensor/rig_calibration.hpp`, in `recon_sensor`;
*amended 2026-10-06 (below):* now `camera/array_calibration.hpp`, in
`recon_camera`, which `recon_sensor` links, and it reads only the poses) is
the family's config
layout: `device_calibration.<serial>` with `intrinsics`, `distortion`,
`optimal_intrinsics` and `pose {rvec, tvec}`; other sections are ignored.
`pose` is the colour camera's OpenCV extrinsic
(world to camera; Rodrigues radians; metres), the one required field, turned
into this repo's camera-to-world on read. The lens fields are parsed and kept
but not used: they are meant to equal the factory calibration the capture
undistorts with. The inverse Rodrigues takes its angle from `atan2`, since
`acos` near pi turned float round-off into an 8e-5 pose error; every pose now
round-trips to 1.2e-7. It is in `recon_sensor` because calib writes it and
nothing in it is Orbbec's. Both files are parsed with nlohmann/json 3.12.0
(hash-pinned in `third_party/CMakeLists.txt`), a private `SYSTEM` include like
VMA's.

**Verified** on the rig. `recon_sensor_orbbec_rig` (`VR_ORBBEC_TEST_RIG`
names the sync file) opens against `femto_mega_sync.json` with nothing
written, refuses a copy with one delay changed, naming the camera, and gets
90 sets, 98.9% complete, worst skew 1.59 ms, each frame posed from a written
calibration file. `fuse_orbbec --rig --apply-sync` wrote only the one camera
that differed; applying the real file put it back, matching a backup taken
first. `recon_sensor_rig_calibration`, `recon_sensor_orbbec_sync_config` and
`recon_sensor_orbbec_grouping` need no camera; 36 tests pass with the driver
built, under `-Werror` in Release.

**Open.** Why the secondaries drop frames with bandwidth to spare is
unmeasured. Discovery once found no camera for 8 s right after another
process released one. A set's frames are processed one camera after another
(a `TODO(sensor)`). No real rig poses exist yet; that waits on calib.

### 2026-09-27 — The video decoder is `recon_sensor_video`, a target of its own over an installed FFmpeg: HEVC in, host pictures out, on the first hardware back end that decodes a built-in clip, NVIDIA ahead of an integrated GPU; each CI leg requires the hardware it has.

*Amended 2026-10-06 (below):* host pictures, `Auto`, the probe clip and
software decoding are gone. The decoders hand out device pictures only, on
NVDEC through CUDA or on VideoToolbox, and only the 24.04 and macOS legs
build them.

**The rule.** `HevcDecoder` (`sensor/video/hevc_decoder.hpp`) takes an Annex B
H.265 stream one access unit at a time and hands out host pictures, as
`Rgb24` or as the decoded `Yuv420` planes with the matrix and range to convert
them by. It knows nothing of cameras, so the Orbbec driver will link it, not
the other way round. FFmpeg is a prerequisite behind `VR_WITH_FFMPEG`, found
through pkg-config and never fetched, like the Orbbec SDK; the floor is
Ubuntu 22.04's FFmpeg 4.4. No FFmpeg type is public.

**Back ends.** `Auto` tries VideoToolbox on Apple; CUDA (NVDEC), then VAAPI
on Linux; CUDA, then D3D11VA on Windows. A machine with an NVIDIA and an
Intel GPU therefore decodes on the NVIDIA one. A back end counts only if it
decodes a built-in 157-byte clip, because an open device says nothing about
HEVC: an older Intel GPU opens under VAAPI without decoding HEVC. VideoToolbox is asked instead, through
`VTIsHardwareDecodeSupported`, the one back end with a single call for it.
The others each have a query of their own (NVDEC's `cuvidGetDecoderCaps`,
VAAPI's profiles, D3D11's decoder profiles),
each with its own dependency and two with no CI leg, while the clip tests
the path FFmpeg will decode on. Each is probed once per process, with
FFmpeg's log silenced (a missing back end says so at ERROR: "Cannot load
libcuda.so.1"), and `Auto` probes down the list only until one decodes and
opens, so an NVIDIA machine never probes VAAPI. A named back end
that is not there is refused, and the message lists those that are. Only
`Auto` falls back to software, including mid-stream when the hardware refuses
a stream (4:4:4 or 4:0:0, say); `backend()` reports the switch. Whether a
picture needs copying off the GPU is asked of the picture (`hw_frames_ctx`),
not of the back end, because the pictures the hardware decoded before the
switch are still waiting for display. A named back end meeting such a stream
returns `Unsupported`, after the pictures decoded before it.

**No Vulkan.** FFmpeg can also decode HEVC through Vulkan video, on NVIDIA's
driver and Mesa's, and it is left out. At a format change mid-stream (the
fallback clip's 4:0:0), FFmpeg tears its Vulkan decoder down before asking
for the new format, and freeing the decoder afterwards segfaults in NVIDIA's
Vulkan driver: `avcodec_free_context`, then libavcodec, then
`libnvidia-eglcore.so`, on FFmpeg 6.1 and 8.0 with driver 615.71.09. CUDA
passes the same clip. Nothing outside FFmpeg avoids it, and CUDA and VAAPI
already cover NVIDIA, Intel and AMD. Its draw comes with the GPU module:
pictures decoded straight into `VkImage`s, taken without a copy.

**Cropping.** A stream's display window can start right of or below the
coded picture's corner (an SPS conformance window). FFmpeg crops a hardware
picture only at the right and bottom, so the decoder crops the host copy at
the left and top. VideoToolbox cannot be fixed that way: FFmpeg sizes its
output at the display size, which VideoToolbox fills from the coded corner,
and then clears the frame's crop, so the right pixels never reach the host.
The decoder therefore reads each SPS itself on VideoToolbox (the conformance
window, H.265 7.3.2.2; only the NAL units ahead of a slice) and treats a left
or top offset like a stream the hardware refuses. Software decoding sets
`AV_CODEC_FLAG_UNALIGNED`, without which FFmpeg keeps up to 64 columns of a
left crop to keep the planes aligned.

**Colour.** `matrix` and `full_range` describe the planes handed out, which
is what a `Yuv420` consumer converts by. So a converted `Yuv420` keeps its
source's range (swscale alone takes a YUVJ source to limited range), and an
RGB-coded stream is converted to `Yuv420` by the matrix it is labelled with.
SMPTE 240M and FCC have their own matrices, since swscale has tables for
both. `encoding` carries the stream's transfer function and primaries as the
`ColorEncoding` a capture source declares, and is empty for anything that
type cannot name (HLG, BT.601's own primaries), so a driver refuses rather
than guesses.

**Modules.** Only `hevc_decoder.cpp` is HEVC-specific. `ffmpeg.hpp` is the one
FFmpeg include (owning pointers, errors), `hw_backend` holds the platform
order and opens devices, and `picture_converter` is the swscale step and the
matrix choice. An H.264 decoder, or a GPU path that skips the host copy,
reuses the three; the converter reports errors under the name its owner
gives it.

**Packaging.** A static `recon_sensor_video` is code compiled against one
FFmpeg's headers, so the package config holds a consumer to that FFmpeg: its
major versions, no older than its minors. The floor applies only to the
build. A consumer whose pkg-config finds another FFmpeg is told which to
point `PKG_CONFIG_PATH` at, rather than linking one that lays `AVFrame` out
differently.

**Why these numbers.** M-series Mac, FFmpeg 9.0.2, 120-frame IPPP clips,
ms per frame:

| | 720p YUV | 720p RGB | 4K YUV | 4K RGB |
|---|---|---|---|---|
| VideoToolbox | 0.93 | 2.90 | 4.21 | 22.70 |
| software, FFmpeg's threads | 0.45 | 2.05 | 2.07 | 18.54 |
| software, one thread | 1.50 | 3.46 | 13.38 | 32.09 |

- Converting to RGB on the host costs ~18 ms at 4K, and nearly all of that
  is the flags. An earlier note here said full chroma interpolation and
  accurate rounding cost nothing measurable; that is wrong on FFmpeg 9.0.2.
  `SWS_ACCURATE_RND | SWS_FULL_CHR_H_INT` take swscale off its unscaled SIMD
  path, and plain `SWS_BILINEAR` converts 4K NV12 or YUV420P to RGB24 in
  0.96 or 0.90 ms instead of 18.9 or 17.9. On solid colours both are within 1
  code of the standards: 99.7% exact with the flags, 77.7% without. What the
  fast path gives up is chroma interpolation, since it repeats each chroma
  sample. `SWS_ACCURATE_RND` alone is the case that came out 3 codes off
  (10 ms). swscale is CPU-only, and its own threads did not speed up either
  path. 4K colour for a rig is to be converted on the GPU, from `Yuv420`
  (the next step), so the host path's flags are left as they are for now.
- VideoToolbox's times include copying each picture out of GPU memory.
- Software with FFmpeg's thread count held back 15 pictures, half a second at
  30 fps, so a live stream sets `threads = 1`. Hardware held back none.
- A hardware decoder opens with the caller's `threads` as *slice* threads, for
  a stream `Auto` moves to software. They hold nothing back, since frame
  threads would hold pictures back on the hardware too, but they help only a
  wavefront (WPP) stream: on 4K x265, 12.3 ms a frame on one thread against
  3.7 on slice threads (frame threads: 4.2), and 10.8 either way without WPP.

**CI.** Every build leg builds the decoder against its own FFmpeg and sets
`VR_TEST_HEVC_BACKEND`: `cuda` in the Linux GPU containers, `videotoolbox` on
the Mac. The decoder test fails unless that back end decodes and `Auto` picks
it, so a container that lost NVDEC, or one that picked an integrated GPU,
fails instead of passing on software. The sanitizer job decodes in software.

**Verified** on the Mac, and on NVDEC in CI (FFmpeg 4.4, 6.1 and 8.0).
`recon_sensor_video_hevc` decodes the committed clip
(`tools/make_hevc_fixtures.sh`: 256x144, the height NVDEC's minimum allows,
8 frames of solid patches at qp 4). In software each patch is within 2 codes
of the pattern in YUV and 3 in RGB, BT.709 limited. On VideoToolbox and
NVDEC the YUV is bit-identical to software and the RGB within 1 code.
`recon_sensor_video_converter` checks each matrix and range on hand-built
frames against the standards' constants. A wrong matrix, wrong coefficients or
a dropped pts each fails a test, and `VR_TEST_HEVC_BACKEND=cuda` fails on the
Mac.

Two more clips came out of review. `cropped_240x128.h265` is the same
frames with a window 16 right and 16 down; it must equal the uncropped
decode's pixels there, exactly, on every back end that crops, and named
VideoToolbox must refuse it. `fallback_256x144.h265` has B-frames and then
4:0:0 grey, for which FFmpeg offers no hardware format in any version from
4.4 to 9.0.2. `Auto` must hand out all 10 pictures, in display order and
each pts sent once, and end on software; a named back end must refuse it.
Taking back each of these fixes fails a test: deciding the GPU copy by the
back end, the SPS read, `AV_CODEC_FLAG_UNALIGNED`, a YUVJ source's range, the
target matrix of an RGB-coded stream, and SMPTE 240M. The SPS reader was
also checked against an x265 stream with three temporal sub-layers.

**Review (PR #92).** Fixed as described above: the lost pictures after a
fallback, the left and top crop on hardware and in software, the range of a
converted YUVJ source, the transfer and primaries, the matrices, a failed
`sws_scale` (it returned OK), single-threaded decoding after a fallback, the
unpinned FFmpeg in the package config, `Auto` giving up after the first
hardware back end, the probe's cost and noise, and the converter's
hard-coded name. Measured and left as it was: FFmpeg allocates a fresh host
picture for each hardware picture, and keeping one buffer measured no faster
(3.70 against 3.63 ms a 4K picture on VideoToolbox), so the simpler code
stays. Kept on purpose: a decoder opens its own device after the probe
closes its own, since sharing one would tie every decoder to a device the
process holds.

**Open.** VAAPI decoding is untested, and D3D11VA has no leg. The
left and top crop on hardware is tested only on the back ends CI has
(NVDEC). Pictures pass through host memory (a `TODO(sensor)`).

### 2026-09-28 — The Orbbec driver streams colour as H.265 on request: every colour frame is decoded, in order, on a thread per camera ahead of the mailbox; a lost frame is read off the frame index, not the clock; and the Femto Mega's stream is decoded as BT.601 full range, which it codes and does not say.

*Amended 2026-10-06 (below):* the decoder hands on the planes, or the picture
the hardware left on the device, never an RGB frame: the `Rgb24` frame
stamped with the RGB profile described here fed the host path, which is gone.
A frame's colour camera is read off the H.265 mode's own profile, so the
byte comparison under "The calibration" went with the RGB mode.
*Amended again the same day (the device-only decoder entry, below):* it
decodes on NVDEC or VideoToolbox only, hands on the picture on the device
only, and a stream the hardware refuses stops it.
*Amended 2026-10-27 (the colour-controls entry, below):* `Hevc` at
3840x2160 and 25 fps is the default; `Mjpeg` stays selectable.

**The rule.** `OrbbecStreamOptions::color_codec` is `Mjpeg` (the default)
or `Hevc`. With `Hevc` the camera sends H.265, and each camera's
`HevcColorDecoder` (`sensor/orbbec/hevc_color.hpp`, built only with
`VR_WITH_FFMPEG`) decodes every colour frame, in order, on a thread of its
own, before the mailbox. It uses `HevcDecoder` (`Auto`, one thread, `Rgb24`),
turns each picture into an RGB frame stamped with the RGB mode's profile, and
posts it with its depth. From the mailbox on, nothing differs from MJPEG.
Pairs the decoder cannot hand on are counted in `OrbbecCaptureStats::lost`.
Without FFmpeg, `open` refuses `Hevc` before it touches the SDK.

**Why ahead of the mailbox.** The mailbox keeps the newest pairs and drops
the rest. An H.265 frame dropped before decoding corrupts the frames predicted
from it, up to the next key frame, which the camera sends every 30 frames.

**The colour.** The camera writes no VUI (`vui_parameters_present_flag = 0`),
so its stream would decode as BT.709 limited range, the default for an
unlabelled 720p stream. Its luma spans 0-255 in every frame, though. Fitting
the decoded colour to the camera's own MJPEG output of the same still view,
per channel (`a = gain * b + offset`, which absorbs the exposure change
between the two captures), stable over two runs:

| decoded as | residual RMS | on saturated pixels | gains (R, G, B) |
|---|---|---|---|
| BT.601 full | 4.16 | 3.50 | 0.998, 0.999, 0.996 |
| BT.709 full | 4.54 | 5.18 | 0.977, 1.007, 0.983 |
| BT.601 limited | 5.12 | 4.52 | 0.886, 0.884, 0.879 |
| BT.709 limited | 5.31 | 5.83 | 0.878, 0.890, 0.868 |

So the driver decodes it as BT.601 full range, through
`HevcDecoder::Options::unlabelled_color`: the matrix and range for a stream
that declares no matrix. A stream that declares one is decoded as it says, so
a camera or firmware that labels its stream is taken at its word. The range
goes with the matrix because FFmpeg reads a stream with no VUI as limited
range (ffprobe's `color_range=tv`), just as one that declares limited, so
only the missing matrix tells the unlabelled stream apart. BT.601 full is
also how JPEG codes the MJPEG stream, so a frame's colour is the same sRGB
either way.

**The calibration.** The decoded frames carry the RGB mode's profile. Its
intrinsics, distortion and depth extrinsic are the H.265 mode's byte for byte,
measured at 720p, 1080p and 4K, so undistortion and registration run
unchanged. `open` compares those bytes on every camera and refuses one where
they differ (`Unsupported`): the first-pair check compares the frames with
the RGB profile they were stamped with, so it could never catch it.

**What loses a pair.** Two things measured on the rig shaped this:
- **The SDK's pairing.** Handed only complete pairs, the SDK dropped a
  secondary's colour frame whenever its depth frame was missing, under 2% of
  triggers. With MJPEG that costs the one pair; with H.265 each gap cost the
  frames up to the next key frame, 8-30% of a secondary's pairs over two
  runs. So for H.265
  the SDK hands over every colour frame (`COLOR_FRAME_REQUIRE`), each is
  decoded, and a pair without depth is dropped after. Colour index gaps over
  a run went from 22 to 0.
- **The clock.** A sync secondary's first frame comes as soon as it starts,
  and the rest once the primary triggers it, 0.3-4.2 s later, with no frame
  lost between. A rule that dated a gap from the timestamps threw away 29
  good frames per start. The camera's frame index, one up per encoded frame,
  is what shows a loss (the colour gate, `hevc_color.hpp`). A skipped or
  repeated index, an
  empty frame (which, sent, would end the stream), a decode error, or a queue
  two seconds behind waits for the next key frame; so does the start.

Where decoding picks up again, and what is counted:
- **The restart.** The key frame that ends a wait resets the decoder
  (`HevcDecoder::reset`, FFmpeg's flush). What it still holds belongs to pairs
  from before the loss and is counted lost, and a CRA's leading pictures are
  skipped rather than decoded from a reference that may be the lost frame;
  FFmpeg 6.1 conceals a missing reference instead of refusing it. The reset
  also clears a decoder left holding pictures after a failed send or a
  thrown call, which would otherwise refuse every send after it.
- **Display order.** Pictures come out in display order, so a picture settles
  its own pair and no other. A pair sent 32 access units ago with no picture
  is one the decoder skipped, and is counted then: H.265 holds at most 16
  pictures back, one thread adds none, and an FFmpeg before 7.1 (6.1 on
  Ubuntu 24.04) hands out one picture per access unit, so the pictures a
  sequence holds back can wait behind the last one's too. A window of 16,
  the first cut's, would have dropped such a pair as lost. The Femto Mega
  sends no B-frames; a camera that did would otherwise have lost about half
  its pairs.

**Why these numbers.** The four-camera rig at 720p30 over the cable,
1200 frames, Release:

| | MJPEG | H.265 |
|---|---|---|
| complete sets | 291 of 304 | 292 of 304 |
| lost per camera | 0 | 1-4 (colour without its depth) |
| fused | 115.6 fps | 114.3 fps |
| CPU, user + sys | 9.37 + 2.58 s | 9.49 + 2.88 s |

Colour on the wire is 37-38 Mbit/s per camera as MJPEG and 21.6 as H.265
at 720p30, and 185 against 21 at 4K25. Decoding costs about what the SDK's
JPEG decoding does. One camera at 4K25 fuses at about 24 fps either way,
using about 55 ms of CPU a frame, nearly all of it undistorting, registering
and converting at 4K. So a 4K rig waits on the GPU module, not on the
decoder.

**Verified.** `recon_sensor_orbbec_hevc` runs the decoder on committed clips
wrapped in SDK frames, with no camera. It checks order, timestamps, depth
and colour; that a lost frame costs the frames to the next key frame, an
empty one too, and a pause costs nothing; that colour without depth is
decoded and dropped, and the frames after it still decode; that a labelled
stream keeps its label; that B-frames lose nothing; that a restart in open
GOPs skips the CRA's leading pictures and counts them 32 frames on; and it
tests the colour gate (`hevc_color.hpp`) on its own. The clips are the patch
clip unlabelled
(`unlabelled_256x144.h265`, its SPS's matrix, primaries and transfer set to
unspecified, which is what FFmpeg reads off the camera), labelled, with
B-frames, and in open GOPs (`open_gop_256x144.h265`).
A test cannot set an SDK frame's index, so it numbers its frames through
`Options::frame_index`. It first relied on the decoder noticing the loss,
which FFmpeg 9 does and FFmpeg 6.1 does not: 6.1 conceals the missing
reference and decodes the next frames wrongly, without an error. Removing the gate's index check,
the decode of a colour frame without depth, or the unlabelled colour each
fails it. So does taking back each review fix below: settling earlier pairs
with a picture, the reset at a restart, the count 32 frames on, applying the
unlabelled colour to a labelled stream, a reset that does not flush or does
not undo the end, and the early refusal without FFmpeg. Sending an empty
frame alone does not, since the reset at the next key frame now undoes the
end of stream it caused; sending it without that reset does.
`recon_sensor_video_hevc` tests the reset and the unlabelled colour on every
back end. The first cut of the B-frame test failed on CI's Ubuntu 24.04 leg:
it waited for every picture, and FFmpeg 6.1, handing out one picture per
access unit, still held two when the stream ended, where FFmpeg 9 lets them
all out at the next clip's key frame. Replayed against libavcodec 6.1, the
open-GOP run hands out 41 of its 43 pictures, none later than 4 access
units after its own. So the tests take every picture in order but the last
three, and the window above is 32, not 16. On the rig, `recon_sensor_orbbec_capture` holds H.265 to the MJPEG
contract, and its colour to MJPEG's within a 5% gain (0.997, 0.984, 0.974
measured). `recon_sensor_orbbec_rig` streams 90 sets with each codec (H.265:
89-96% complete, worst skew 1.6-2.2 ms, 0-3 lost per camera). Both passed
again after the review fixes, the calibration check included (gains 1.000,
0.987, 0.974; H.265 sets 98.9% complete, 0-1 lost per camera).

**Review (PR #97).** Fixed as described above: an empty frame ending the
stream, the calibration assumed rather than checked, the colour forced on a
labelled stream, frames lost to display order, a CRA's leading pictures
decoded after a loss, and a decoder wedged by pictures left waiting. Also
fixed: a queue allocation that throws inside `noexcept push` loses the pair
rather than terminating; the decoding thread failing to start is `IoError`,
not an exception; `start()` documents the decoder's `Unsupported` and
`IoError`; the refusal without FFmpeg moved ahead of discovery;
`configure_sdk_logging` says it sets FFmpeg's level, and does so at the
first start only; the stats say what `received` and `lost` count under
H.265; and `stop()` releases the decoder and the RGB profile rather than
leaving them to the last reference, which the SDK's callback may hold past
the context. Measured and left as it was, from a 21 Mbit/s 4K x265 clip on
an M5 Max, Release:
- Each decoded picture is copied into a freshly allocated SDK frame: 0.5 ms
  at 4K, allocation included, against 24.4 ms a picture on VideoToolbox. A
  pool, or swscale writing into the SDK frame, would save 2%.
- Software decoding stays on one thread: 32.5 ms a picture, about 13 of it
  decoding and the rest the RGB conversion, inside the 40 ms of 4K25. Frame
  threads would shorten only the decode, and hold pictures back. A host
  slower than this one with no hardware decoder would fall behind, which
  the queue's overflow would show in `lost` (a `TODO(sensor)`).

**Open.** A 4K rig waits on the GPU module, and so does keeping the decoded
picture off the host. The camera's own H.265 settings (key-frame interval,
bitrate) are left at its defaults (a `TODO(sensor)`). The SDK's "Stream have
not been started!" at every close and its RTP "Metadata size is too large!"
come with MJPEG too.

### 2026-09-28 — GPU pre-processing is `recon_sensor_utils`: a driver hands out the frame as captured, the device undistorts depth and undistorts and converts colour, and fusion reads the buffers in place, depth and colour each with its own camera rather than registered.

*Amended 2026-10-06 (below):* raw frames no longer come through
`ICameraCapture`: `OrbbecSensor::poll` and `OrbbecRig::poll_set` hand them
out, and the examples fuse them with `fuse_set` alone.

**The rule.** `GpuFramePrep` (`sensor/utils/gpu_frame_prep.hpp`) takes a
`RawFrame` (`sensor/raw_frame.hpp`): raw depth, the decoded Y'CbCr 4:2:0
planes, and each camera's `LensCamera` (`sensor/lens.hpp`) and pose. Two
compute passes undistort depth, and undistort and convert colour; the result
is a `DeviceFrame` of storage buffers that `VoxelHashMap::allocate_from_depth`
and `TsdfIntegrator::integrate` read in place (their `Buffer` overloads, and
`ColorFrame::buffer`). Nothing registers depth to colour: each keeps its own
intrinsics and pose, and `tsdf` fuses a separate colour camera as it always
could. The `DeviceFrame` holds its buffers. A source hands raw frames out
through the capture contract, `ICameraCapture::poll_raw`, and says with
`raw_frames` that it does. The Orbbec driver is one when opened with
`OrbbecStreamOptions::raw` (H.265 colour only), and `fuse_orbbec --gpu` fuses
its frames.

**Why not registered.** The SDK can register depth to colour two ways, and
neither suits a 4K rig. Its software `Align` runs on the CPU, and at the
colour image's full size it made each fuse allocate over 8.3 M depth samples
where the sensor has 0.37 M (9.1 ms of host allocation at 4K against 1.3 ms
here). The camera's hardware registration sends depth at the colour size:
about 8.3 M x 2 B x 25 fps, 415 MB/s a camera at 4K, past gigabit Ethernet.
Fusing the two cameras as they are removes the step, and the depth resampling
with it. What it gives up is the registered case's colour visibility: a voxel
hidden from the colour camera but seen by depth takes the occluder's colour
(`tsdf`'s documented caveat), and the texture tier still wants a frame
registered to one camera.

**Placement.** A target of its own under `sensor/utils/`, because it carries
Vulkan and shaders and `recon_sensor` must not (the 2026-09-26 decision: an
out-of-tree driver compiles against it). `RawFrame` and `LensCamera` stay in
`recon_sensor`, so a driver produces them without a GPU API.
`LensDistortion` moved from `rig_calibration.hpp` to `lens.hpp`, beside the
host model the GLSL mirrors.

**The kernels.** Undistorting keeps each camera's intrinsics and drops its
lens, as the SDK's filter does: each pinhole pixel is sampled where OpenCV's
rational model (`distort_normalized`) images it, so only the forward model is
needed. Depth is sampled at the nearest pixel, since between a foreground and
a background return there is no surface to interpolate. Colour is sampled
bilinearly, luma and chroma each at its own siting (chroma with the even luma
columns, between two luma rows, H.265's default), and converted by the
picture's `kr`, `kb` and range in the same pass, so no 4K R'G'B' intermediate
is written. Only an encoding `is_canonical` accepts is converted. Where the
lens maps a pixel outside the captured image, depth is 0, "no return", and
colour a 0 word. Every other colour word carries 0xFF in its high byte, so
the high byte is the pixel's coverage, and `ColorFrame::coverage_in_alpha`
has fusion skip a pixel without it.

**The calibration** is the camera's, read at open: each stream profile's
intrinsics and lens, and `depth_profile->getExtrinsicTo(color_profile)` for
the depth camera's pose (`p_colour = R p_depth + t`, `rot` row-major,
millimetres). The calibration file's `pose` is still the colour camera's, so
it needs no new fields. On CL2A141000N the depth lens is the rational model
with all six radial terms (k1 29.29 ... k6 3.77), the colour lens plain
Brown-Conrady (k1 0.073, k2 -0.10, k3 0.04), and the depth camera sits at
(-32.5, -0.5, 2.2) mm in the colour camera's frame. The SDK's modified,
inverse and Kannala-Brandt models are refused.

**The Orbbec raw path.** The decoder thread decodes to `Yuv420` and posts the
planes as an SDK I420 frame through the same mailbox. It must not stamp the RGB
mode's stream profile on it, as the RGB path does: the profile restamps the
frame's format as RGB, which the first rig run caught. The frame carries the
matrix and range the decoder resolved in its metadata (`PlanesColor`, since
an SDK frame has no field for them), with the transfer and primaries the
stream declares. A labelled stream is therefore converted by its own labels
on both paths, and an unlabelled one as the Femto Mega's BT.601 full range. A
stream declaring a transfer or primaries `ColorEncoding` cannot name is
refused. `OrbbecRig` refuses `raw` for now.

**Why these numbers.** `fuse_orbbec` on CL2A141000N at 4K25, 300 frames,
Release, M-series Mac, each path's own run:

| per fused frame | host path (`--hevc`) | GPU path (`--gpu`) |
|---|---|---|
| fps | 24.4 | 24.9 (the camera's 25) |
| poll, host | 15.5 ms | 0.01 ms |
| frame prep | -- | 2.6 ms host, 0.89 ms device |
| allocate | 9.1 ms host, 5.5 ms device | 1.3 ms host, 0.76 ms device |
| integrate | 3.4 ms host, 0.46 ms device | 1.7 ms host, 0.58 ms device |
| CPU for the run, user + sys | 15.96 + 1.82 s | 0.94 + 1.23 s |
| peak resident | 1.43 GB | 1.05 GB |

The CPU saved is more than the poll row shows: the host path's decode thread
also converted every 4K picture to RGB through swscale. The frame prep's host
time is mostly copying 12 MB of planes into the upload buffer. (Since step 5a
of the residency decision below, into a batch's staging, and the copy up is
timed on the device.)

**Verified.** `recon_tsdf_device_input` fuses one frame from host arrays and
from buffers and compares every observed voxel. `recon_sensor_gpu_frame_prep`
holds both passes to a host reference (depth exact, colour within a code),
checks the lens is undone rather than applied (a checkerboard imaged through
an independently inverted lens comes back where it was drawn: 0 of 49 920
pixels off), anchors the matrix to a forward conversion, and fuses the
output. Breaking the lens direction, the matrix weights, the range, the
depth rounding or the chroma siting each fails it. On the camera,
`recon_sensor_orbbec_gpu_prep` captures a still scene through both paths.
At 720p every box of a 5 x 3 grid, corners included, matches the SDK's
undistorted colour best unshifted, 1.4-3.0 RMS apart. At 4K every textured
box does (1.9-8.2 RMS, a 2 px shift raising it two to four times). The pass's
depth, moved into the colour camera by the extrinsic, lies a median 5.1-6.0
mm from the SDK's registration, best within a pixel of unshifted. A reversed
extrinsic would shift it by about f * baseline / z, some 30 px there. Two 4K
meshes, one from each path at 2 cm voxels, sit a median 10 mm apart
vertex to vertex (half a voxel). 7-16% of vertices have no counterpart within
2 cm, unexplained: the scene changed between the runs.

**Review.** The PR's review changed eight things, each where the first cut
would have been silently wrong.
- **Colour labels.** The raw path converted every picture as unlabelled
  BT.601 full range, so a labelled stream fused other colours than the host
  path. The labels now travel with the planes, as above.
- **Coverage.** A pixel outside the picture was black and fused as colour.
  Since a voxel's first colour is assigned rather than blended, the view's
  edges fused black patches. The flag is opt-in because a host image's high
  byte is undefined.
- **The depth gate.** An unset range (`RawFrame`'s zeros) fused nothing, and
  a range from 0 fused every "no return" as a surface at the camera. The pass
  refuses anything but a finite `0 < min < max`.
- **Binding ranges.** The device-input overloads bound a caller's buffer at
  `VK_WHOLE_SIZE`, past `maxStorageBufferRange` for a large enough buffer.
  They bind the image's exact range now, through one `core` type,
  `StorageInput` (a host array or a device buffer, never both). Both tiers
  and the colour image bind through it, and its O(1) check runs before the
  compaction dispatch, so an empty grid refuses what a full one does.
- **Order.** `prepare` checks both halves of a frame before either is
  uploaded, so a refused colour picture costs no depth dispatch.
- **Lifetime.** A `DeviceFrame` was a borrowed view "valid until the next
  prepare". A kept one then silently fused the next frame's buffers with its
  own cameras. A generation stamp would need checking where the frame is
  consumed, and the fusion tiers take a `Buffer`, not a `DeviceFrame`.
  Ownership removes the staleness instead: the frame shares its buffers, and
  `prepare` reuses one only when no frame still holds it (the 2026-08-04
  rule, met by construction).
- **Device-local outputs.** They were host-visible, so on a discrete GPU the
  integrate kernel's reads would cross the bus. They are now `DeviceLocal`
  and `TRANSFER_SRC` (a test copies them out).
- **The contract.** `fuse_orbbec --gpu` reached into the concrete
  `OrbbecCapture`, against the 2026-09-14 rule. Raw frames now come through
  `ICameraCapture`, non-pure like `exhausted()`. The `DeviceFrame` overload
  of `fuse_frame` lives in `fuse_device_frame.hpp`, so the dataset examples
  do not link `recon_sensor_utils`.

The plain Brown-Conrady model also stopped reading k4..k6, which only the K6
model has. One suggestion was not taken: recording both passes in one submit
would save a fence wait a frame. It needed a several-kernel dispatch in
`core`, which `ExtractTimings`' device half wants too, so it waited for that.
`CommandBatch` is that dispatch, and step 5a of the residency decision below
has since put both passes in one submit. None of this has run on the camera
yet: the hardware test was not re-run after the review.

**Open.** The colour-occlusion fringe of an unregistered camera is
unmeasured, and the texture tier's separate colour camera is a later slice.
The planes are copied from the decoder. Zero-copy input from VideoToolbox or
NVDEC is a `TODO(sensor)`, and so are the rig's raw sets. Nothing has run on
an NVIDIA host with a camera. This PR's first test, 320 x 240 pixels of one flat wall, contended on
the hash map's bucket locks past the driver's 7 s watchdog in CI (Xid 109, and
a runner lost), while they were host-visible; PR #81 has since moved them into
device memory, the fix the fleet's first watchdog report called for.

*Amended 2026-09-29:* **the depth pass can keep depth to the colour camera's
view.** With `GpuFramePrepConfig::depth_within_color` set, a frame with colour
has a depth pixel zeroed ("no return") when its point, moved into the colour
camera by the two poses, lies behind that camera or lands on a pixel the colour
pass marks uncovered. The test uses the colour pass's own rounding and lens
test, so depth survives exactly where `DeviceFrame::color`'s coverage byte is
set. The option is off by default, and a frame without colour keeps all its
depth.

- **Why.** A Femto Mega's depth covers about 65 degrees vertically and its
  16:9 colour about 51. The depth outside the colour view fuses into surfaces
  no camera can colour or texture, and the mesher draws them white. On the
  lab rig that was about half the viewer's untextured triangles (the
  `rig_viewer` entry below).
- **Why here.** The frame prep is the one place that holds both cameras,
  their poses and the colour's coverage on the device. A zero is the "no
  return" that `allocate_from_depth` and `integrate` already skip, so neither
  fusion tier changes. The host Orbbec path needs no option: it registers
  depth to colour, so its depth already lies inside the colour view.
- **Not registration.** Depth keeps its own resolution and camera, as this
  entry chose. The mask costs each depth pixel one rigid transform and one
  lens evaluation, plus a 120-byte inline upload per frame. That device cost
  has not been measured apart from the pass.
- **The view, not the line of sight.** A point inside the colour view that a
  nearer surface hides from the colour camera is kept. That is `tsdf`'s
  documented colour-occlusion caveat, unchanged.

`recon_sensor_gpu_frame_prep` checks three cases:

- a pinhole depth camera behind a pincushion colour lens, with the same
  intrinsics and pose: depth is zeroed exactly where the colour's coverage
  byte is 0 (63 807 pixels kept, 12 993 zeroed, 0 wrong);
- a colour camera 1.5 times narrower and 10 cm to the side: the depth kept
  is the region a host projection puts inside its image;
- the option off, or a frame without colour: nothing is zeroed.

Each of three mutations fails the test: ignoring the pose, skipping the lens
coverage test, and ignoring the flag. The test passes under the Khronos
validation layer with no messages.

### 2026-09-28 — A decoded colour frame never leaves the GPU: the platform's hardware decoder, for H.265 and MJPEG alike, hands its picture to Vulkan in place, and `GpuFramePrep` takes I420 or NV12 from the host or the device.

*Amended 2026-10-06 (below):* the MJPG mode's calibration is no longer
compared with the RGB mode's; a frame's colour camera is read off the mode
streamed.
*Amended 2026-10-08 (below):* `place_device_color` goes: a decoded picture
carries its `YuvImage`, which a frame takes as it is.
*Amended 2026-10-06 (below):* the software decoding and fallbacks this entry
kept are gone, with nvJPEG's `GPU_HYBRID` back end: every picture is a
device picture, and a device path that fails is an error.
*Amended again the same day (below):* `GpuFramePrep` takes colour from the
device only.

**The rule.** The camera stream arrives on the host, the RGB-D frame goes to
the GPU once, and undistortion, colour conversion and fusion all run there.
Depth already did: raw 16-bit samples, one upload. Colour did not. A
hardware-decoded picture came back to the host (`av_hwframe_transfer_data`),
swscale converted it from NV12 to I420, the Orbbec driver copied it into a
new 12 MB buffer, `GpuFramePrep` copied it into staging, and the batch
uploaded it again. Per 4K picture per camera, Release, that was:

| | M5 Max | RTX 5090 box |
|---|---|---|
| copy to host | 0.20-0.34 ms | 1.1 ms (3.9 with a fresh driver buffer) |
| NV12 to I420 | 0.31-0.46 ms | 0.6 ms |
| driver copy (`i420_frame`) | 0.18-0.28 ms | 0.57 ms (3.8 fresh) |
| copy into staging | 0.2 ms | 0.6 ms |
| upload, device time | 0.04 ms | 0.47 ms |

On Linux a fresh 12 MB buffer a frame page-faults, which is what the driver
does today; the bench that measured it decoded a 3840x2160 clip through
`HevcDecoder` and repeated the driver's copy.

Instead, each platform's hardware decoder hands its picture to Vulkan where it
lies:

- **NVIDIA.** NVDEC, for H.265, copies its surface device to device
  (`cuMemcpy2DAsync`) into a ring of Vulkan buffers imported into CUDA
  (`VK_KHR_external_memory_fd`). Its pool is small, so the surface is copied
  out at once rather than held. nvJPEG's hardware back end decodes MJPEG
  straight into the same buffers, and `GPU_HYBRID` stands in on a GPU without
  the JPEG engine. Both sit behind `VR_WITH_CUDA`, on the CUDA 13 toolkit.
- **Apple.** VideoToolbox, for H.265 through FFmpeg and for JPEG directly,
  gives an IOSurface-backed NV12 `CVPixelBuffer`. Each plane becomes a Metal
  texture over the IOSurface, imported with `VK_EXT_metal_objects`, and the
  pass copies the planes into its input in its own batch.
- **`GpuFramePrep`** takes I420 or NV12, as host planes or as a device
  buffer with per-plane offsets and strides (`YuvImage::device`), and binds
  device planes where they are. The frame holds that buffer, so a decoder's
  ring cannot reuse it while the frame is prepared.

**Why each piece**, measured on throwaway spikes:

- **MoltenVK 1.4.2 refuses an IOSurface as one two-plane image**: `IOSurface
  bytes per element 1 does not match VkImage bytes per element 6`. A Metal
  texture per plane imports in 0.16-0.24 ms. VideoToolbox cycled through 5
  surfaces for 110 pictures, so the import can be cached per surface. The
  planes read back identical to FFmpeg's own host copy: 0 of 8.3 M luma and
  4.1 M chroma bytes differ. Copying both planes image to buffer takes
  0.24-0.30 ms of device time, which the kernel reading the images directly
  would save.
- **MJPEG needs a hardware decoder.** The Femto Mega's JPEGs are 4:2:0, full
  range, BT.601, like its H.265, about 1.4 MB at 4K (its limit is 25 fps),
  and carry no restart markers, so each frame's entropy decode is serial.
  CPU per 4K frame, with CUDA waiting on blocking sync, since its default
  spin bills every wait as CPU:

  | decoder | wall | CPU |
  |---|---|---|
  | FFmpeg software, M5 Max | 13.2 ms | 14.4 ms |
  | VideoToolbox JPEG, M5 Max | 5.2 ms | 0.25-0.29 ms |
  | FFmpeg software, 5090 | 14.7 ms | 16.2 ms |
  | FFmpeg through NVDEC, 5090 | 12.9 ms | 14.3 ms |
  | nvJPEG 13.2, CPU Huffman, 5090 | 10.9 ms | 10.0 ms |
  | nvJPEG 13.2, `GPU_HYBRID`, 5090 | 4.6 ms | 4.2 ms |
  | nvJPEG 13.2, `HARDWARE`, 5090 | 3.8 ms | 0.38 ms |

  The hardware engine's picture differs from the others only by IDCT
  rounding (a luma sum 350 apart over 8.3 M pixels), so its tests compare
  with a tolerance. FFmpeg's NVDEC MJPEG path keeps the picture on the GPU
  but not the work off the CPU. An MJPEG camera sends about 280 Mbit/s at 4K
  against about 30 for H.265, so H.265 stays the rig's default.
- **Vulkan Video decode was set aside.** FFmpeg 6.1 decodes H.265 through
  it on the 5090 at 12.4x real time, against 14.4x for CUDA. It needs Vulkan
  1.3 (recon's instance asks for 1.2), a decode queue in both device seams
  (gfx's bootstrap too), and FFmpeg's `AVVkFrame` protocol, and MoltenVK has
  none of it. It becomes the better choice if AMD or Intel on Linux, or
  Windows, become targets.
- **Host-side trims were set aside**: pooled driver buffers, and NV12
  through the host. They polish the path this removes. The page-fault
  finding stays true for the paths that keep host frames.

**Landed.** `GpuFramePrep` takes I420 or NV12, from the host or the device.
The colour kernel reads each plane at an offset and row stride of its own,
with a chroma step of 1 or 2, so its push constants grow from 76 to 96 bytes.
A test prepares one odd-sized, lensed picture every way it can come and
requires the outputs to be identical: host I420, host NV12 with tight and
padded rows, and device I420 and NV12 at odd offsets with padded rows, 4 KiB
into their buffer, taken over from outside Vulkan or not, the I420 with Cb
and Cr rows of different lengths. It also refuses:

- planes on both the host and the device, a stale third one included, and
  an NV12 host picture with a third plane;
- a plane past its buffer, or its last word past it;
- planes that overlap, as I420's do with their offsets left at zero;
- a row shorter than its picture;
- a buffer that is empty or without storage usage;
- a queue family the device lacks, before any work.

A kernel that ignores the chroma step, as a mutant, fails the NV12 case; one
that ignores the luma offset fails the device case.

Its review settled the device path, which a picture another queue or API
wrote reaches:

- **Ownership.** Reading an EXCLUSIVE buffer another queue family wrote, or
  memory CUDA wrote through an import, is undefined without a queue-family
  ownership transfer, and appears to work on Apple, where Metal has no such
  thing. `YuvImage::queue_family` names the writer: another family of the
  device, `kQueueFamilyExternal` for CUDA, or `kQueueFamilyIgnored`, the
  default, for the pass's own. The pass opens its batch with
  `CommandBatch::acquire`, new in `core`: the receiving half, from
  `VK_QUEUE_FAMILY_EXTERNAL` for an EXCLUSIVE or a CONCURRENT buffer, and
  from another family for an EXCLUSIVE one, whose writer records the
  release. The batch waits on no semaphore, so the writer must have
  finished, a fence waited or the stream synchronized, which
  `YuvImage::device` now says. An acquire naming this family for a
  CONCURRENT buffer, rather than every family, fails the batch test under
  the validation layer (VUID-09051); a pass that skips the acquire fails its
  refusal of an unknown family.
- **Lifetime.** A failed wait already leaked the staging the device may
  still read. It now leaks a reference to the device planes' buffer too,
  which the caller could otherwise drop, or a decoder's ring reuse, as soon
  as `prepare` returned.
- **Binding.** Device planes are bound from the first plane, rounded down to
  `minStorageBufferOffsetAlignment`, not from byte 0, so
  `maxStorageBufferRange` (2^27 on Mali and many Intel drivers) and the
  kernel's 32-bit addressing weigh the picture, not where it sits: a
  decoder ring's slot 128 MB into one buffer would have been refused on
  those drivers. A kernel reading each
  plane where it sits in the buffer rather than the binding fails the test;
  binding from byte 0 does not, since no device CI runs has a limit that
  low.
- **Checks.** Overlapping planes are refused, which catches I420's offsets
  left at zero, and so is a host pointer beside device planes, whichever
  plane it is. The kernel takes Cb's and Cr's strides separately, so the
  device path accepts Cb and Cr rows of different lengths, as FFmpeg's
  linesizes allow, rather than refusing them. The borrowed buffer is checked
  through `StorageInput::check`, so an empty one is called empty. A mutant
  dropping any of the new checks fails the test.

**NVDEC has landed.** `HevcDecoder::Options::device` hands a CUDA decode's
pictures out on the device:

- The decoder opens NVDEC on the CUDA device whose UUID is the Vulkan
  device's.
- It keeps a ring of buffers from `core`'s `create_exported_buffer`, each
  imported into CUDA once.
- For each picture it runs two `cuMemcpy2DAsync`, luma and chroma, and a
  stream sync on FFmpeg's stream.
- A picture takes a buffer no earlier picture holds, so a kept one is still
  itself.
- CUDA wrote it, so a reader acquires it from `VK_QUEUE_FAMILY_EXTERNAL`
  first, and a `YuvImage` of it carries `kQueueFamilyExternal`.

Anything else comes to the host as before: another back end, a 10-bit
stream, a left or top crop by an odd count, or a build without
`VR_WITH_CUDA`. FFmpeg has already taken the right and bottom crop off a
hardware picture's size, so the device path offsets its pointers by the
left and top crop and copies the rest.

`VR_WITH_CUDA` needs the CUDA 13 toolkit, and `nvcc` with it, since CMake
3.28's FindCUDAToolkit will not find the toolkit without it, though recon
compiles no CUDA code. CI's Ubuntu 24.04 leg installs the packages from
NVIDIA's repository. Measured on the RTX 5090, `receive` per 4K picture:

| | wall | CPU |
|---|---|---|
| host (copy to host, NV12 to I420) | 1.73-1.75 ms | 1.34-1.35 ms |
| device | 0.17-0.19 ms | 0.03-0.05 ms |

The host path then pays the 0.6 ms staging copy and the 0.47 ms upload; the
device path neither. On the 5090 and on CI's CUDA leg, the device pictures
of the plain and the cropped clips equal software's, byte for byte. Without
the left-crop offset, as a mutant, the cropped clip fails. A build without
CUDA, or any other back end, hands the same device option's pictures to the
host, which the Mac checks.

Its review changed six things:

- **libcuda is loaded, not linked.** Linking `CUDA::cuda_driver` stopped
  every program built with `VR_WITH_CUDA` from starting without the NVIDIA
  driver. It is `dlopen`ed now, as FFmpeg does, each entry point named
  through `cuda.h`'s macros so it is the versioned symbol its prototype
  declares. Only `cuda.h` comes from the toolkit, so the installed package
  finds no CUDA.
- **A failure falls back.** A slot that would not allocate, or a CUDA call
  that failed, failed the picture, and the next one the same way. Now the
  device path is let go, and that picture and every later one come to the
  host. Retrying each picture was not taken: a lasting failure would pay an
  allocation a picture.
- **Export is asked, not assumed.** The extension does not promise that a
  storage buffer exports, so `create_exported_buffer` asks
  `vkGetPhysicalDeviceExternalBufferProperties`.
- **A shared device hears of it.** `DeviceRequirements::external_memory`
  names the extension as optional, and `fuse_viewer`'s bootstrap enables it
  where offered. Before, an adopted device never exported memory.
- **The matrix guess reads the displayed height.** The device path passed
  the hardware frame's, which still counts the top crop, so a stream near
  576 rows could take BT.709 on one path and BT.601 on the other.
- **A free slot too small is let go**, so a change of picture size strands
  no memory. Slots added while pictures were held stay, a high-water mark
  like the mesh arena's.

`VR_WITH_CUDA` is Linux only, since CUDA imports the memory as a file
descriptor. The device must outlive every picture on it, documented rather
than checked, as for every `Buffer`. The test expects device pictures only
on an NVIDIA Vulkan device, one CUDA sees.

**nvJPEG has landed.** `JpegDecoder` decodes the JPEGs an MJPEG camera
sends, to I420, BT.601 full range as JFIF defines it:

- Given a device, in a `VR_WITH_CUDA` build, it opens nvJPEG in the primary
  context of the CUDA device whose UUID is the Vulkan device's, with a
  decoder on the GPU's hardware JPEG engine where it has one and one on its
  cores (`GPU_HYBRID`) for what the engine refuses; `backend()` names the
  first.
- An 8-bit 4:2:0 JPEG that one of them supports decodes into a buffer from the
  same ring as NVDEC's, Y, Cb and Cr each packed and starting on 256 bytes,
  so `DecodedPicture::offset` grows a third plane. The decoder waits on an
  event made to block, since CUDA's default wait spins and bills the wait as
  CPU. The picture carries `kQueueFamilyExternal` like NVDEC's.
- Anything else decodes in software through FFmpeg, 4:2:2 converted to
  4:2:0 on the host: another subsampling or precision, a JPEG nvJPEG cannot
  parse or decode, a device on another GPU, no device.
- libnvjpeg is loaded at run time, like libcuda, so a program built with
  CUDA starts where neither is. CI's CUDA leg installs it again.
- A JPEG nvJPEG refuses (`BAD_JPEG`, `JPEG_NOT_SUPPORTED`,
  `INCOMPLETE_BITSTREAM`) goes to software alone, since one corrupt frame
  from a camera must not cost every later one 14 ms of CPU. Any other
  failure, nvJPEG's or CUDA's, lets the device path go, as NVDEC's does.

Measured on the RTX 5090, per 4K frame of the Femto Mega's MJPEG, Release:

| | wall | CPU |
|---|---|---|
| software | 14.27 ms | 14.34 ms |
| nvJPEG, `HARDWARE` | 3.83 ms | 0.38 ms |

The software picture then pays the staging copy and the upload; the nvJPEG
one is already on the device. The committed JPEGs
(`tools/make_jpeg_fixtures.sh`: the HEVC clips' pattern at 4:2:0, at an odd
size and at 4:2:2) decode to the pattern in software, and to within 2 codes
of software on the device, the two inverse DCTs rounding apart. A buffer is
reused only once no picture holds it. Swapping the Cb and Cr offsets, as a
mutant, fails the test. Both CI hosts are Blackwell and have the JPEG
engine, so `GPU_HYBRID` runs there only on a fixture 16400 pixels wide.

Its review changed five things, measured on the RTX 5090 against 4K JPEGs
spoiled the ways a camera link might spoil them: entropy data flipped or
zeroed, and the file cut short.

- **A status is read, not only its failure.** Every nvJPEG failure sent the
  JPEG to software, so a lost GPU or a device out of memory cost every later
  frame a parse and 14 ms of CPU while `backend()` still named nvJPEG. Only
  the three statuses about the JPEG are refusals now, and a decode that fails
  still waits for what it queued before the buffers are reused. No spoiled
  frame returned another status: a flipped one fails to parse as
  `JPEG_NOT_SUPPORTED`, and a zeroed or cut one decodes, CUDA sound. The test
  holds a garbage and a cut frame to keeping the device path.
- **The cores take what the engine refuses**, a JPEG past 16384 pixels a
  side, which `GPU_HYBRID` decodes; nothing from 2x2 to 8192x8192 was
  refused. Both decoders share one handle, made on the engine.
- **A device path let go releases what it held**: the primary context, the
  stream, nvJPEG's handle, states and staging buffers. So does one that
  never finished opening. Both kept them for the decoder's life.
- **libnvjpeg is the major version `nvjpeg.h` names**, and configure refuses
  a toolkit without `nvjpeg.h`, which is a package of its own.
- **The test checks what it reads**: each picture against its fixture's
  size, and every plane whole before it indexes one. Dropping the cores'
  decoder, or refusing no status, as mutants, fails it on the 5090.

Not taken:

- Remembering a refusal. A full parse of a 350 KB 4K JPEG is 0.05 ms, under
  0.4% of the software decode that follows it. A header-only parse is not
  enough: the engine's support check answered differently after one.
- One loader for libcuda and libnvjpeg. Each table would still need its own
  macro list, so sharing saved no lines.
- Releasing a reused buffer back to CUDA. CUDA overwrites the picture, a
  family may take a buffer without a transfer when its contents need not
  survive, and the reader that let the picture go had finished with it.

**VideoToolbox's H.265 has landed.** Given `HevcDecoder::Options::device`
on a device that imports Metal textures, a VideoToolbox decode hands its
picture out as two images:

- `Device::create` enables `VK_EXT_metal_objects` where MoltenVK offers it
  (`imports_metal_textures()`), and `DeviceRequirements::metal_objects` asks
  a shared device for it.
- Each plane of the picture's IOSurface becomes a Metal texture on the
  device's own `MTLDevice`, imported as an image that `core`'s new `Image`
  holds. The `MTLDevice` is read off the exported texture of a 1x1 probe
  image: exporting the device itself (`vkExportMetalObjectsEXT`) needs the
  instance to have asked for it when it was made, which an embedder's may
  not have, where an image asks for its own texture. VideoToolbox cycles its pictures through
  a few surfaces (five for the lab's 4K clip), so each surface's images are
  made the first time a picture arrives on it and kept until none has for 64
  pictures. A picture's images hold its pixel buffer, so VideoToolbox does
  not reuse the surface while a frame reads it.
- `DecodedPicture::image` carries the two, and `YuvImage::image` takes them.
  `GpuFramePrep` copies both into its input in its batch
  (`CommandBatch::copy` from an `Image`, new), then reads them as it reads
  packed host planes.
- An image stays in one layout, one a copy reads, which its maker puts it
  in: `CommandBatch::copy` refuses any but GENERAL and TRANSFER_SRC_OPTIMAL.
  A surface's images go from UNDEFINED to GENERAL once, when they are made.
  Vulkan does not promise that keeps their contents; MoltenVK keeps them,
  Metal having no layouts, so the assumption lives with the Metal import,
  not in `core`. The images the tests fill by hand are GENERAL too, so the
  copy is exercised on every GPU CI has.
- A picture that is not 8-bit NV12 on an IOSurface comes to the host, and a
  failed import lets the device path go, as for NVDEC. A device path that
  cannot be set up, on either, leaves every picture to the host rather than
  failing the decoder. VideoToolbox has cut
  the right and bottom already, and a stream cropped at the left or top never
  reaches it.

Measured on the M5 Max, Release, per picture of the lab's 4K clip, with a
640x576 depth frame beside it:

| | receive, CPU | prepare, CPU | prepare, GPU |
|---|---|---|---|
| host planes | 0.57-0.75 ms | 0.43-0.58 ms | 0.94-1.01 ms |
| plane images | 0.003-0.005 ms | 0.31-0.45 ms | 1.09-1.17 ms |

Unified memory makes the host path cheap here, so the saving is modest: the
copy to the host, the conversion and the staging go, and the device copies
the planes instead, 0.28-0.31 ms alone and some 0.15 ms more than the staged
copy it replaces. Receiving is only handing the images over: made for every
picture, two textures, two images and two allocations cost 0.16-0.33 ms of
it. The pictures read back
identical to software's, byte for byte, the cropped clip going to software.
Colour copied from images larger than the picture, read from their corner,
prepares identically to host planes, and a mutant copying the chroma a
column short at an odd width fails that test.

**VideoToolbox's JPEG has landed.** Given a device that imports Metal
textures, `JpegDecoder` opens VideoToolbox's hardware JPEG decoder
(`JpegDecodeBackend::VideoToolbox`):

- It takes a baseline or extended 8-bit 4:2:0 JPEG in three components, the
  frame header read off the bytes, as nvJPEG takes it. Anything else goes to
  software, as does one past the device's `maxImageDimension2D`, before a
  session starts: Metal aborts the process on a texture past it, which the
  16400-wide fixture is on the M4 runner (16384).
- A session holds for one size, asked for full-range NV12 on an IOSurface
  and the hardware only. A size no session opens for is not asked again each
  frame.
- The decode is synchronous. Its picture goes through the same import as
  H.265's, into `DecodedPicture::image`.
- A JPEG the session refuses goes to software alone, and the session is
  started afresh for the next; only a failed import lets the device path go.

Measured on the M5 Max, Release, per frame of the Femto Mega's 4K MJPEG, CPU
the whole process's:

| | decode, wall | decode, CPU | prepare, CPU | prepare, GPU |
|---|---|---|---|---|
| software | 13.1-13.3 ms | 13.1-13.3 ms | 0.38-0.39 ms | 0.98-1.00 ms |
| VideoToolbox | 6.0-6.1 ms | 0.92-0.94 ms | 0.64-0.67 ms | 1.14-1.18 ms |

A frame costs 1.6 ms of CPU where it cost 13.7. The images of every 4:2:0
fixture, the 16400-wide one included where the extent takes it, are within 2
codes of software's; 4:2:2 goes to software.

**The Orbbec raw path carries H.265's device picture.** Given
`OrbbecStreamOptions::device` with `raw`, each camera's colour decoder
decodes onto that device, and a raw frame's colour is the picture NVDEC or
VideoToolbox left there (`YuvImage::device`, taken over from outside Vulkan,
or `YuvImage::image`). Any other picture comes as the I420 planes it came as
before.

The picture crosses the mailbox with an SDK colour frame
(`picture_frames.hpp`). The decode thread, the mailbox, a rig's
grouping and the held pair all handle frame sets, as before. The picture then
lives exactly as long as its frame, and is dropped, grouped and held with it;
changing the mailbox's element type instead would have touched every one of
them.

The frame's bytes are a tag and a serial, and the picture waits in a table
under that serial until the frame's deleter erases it. The first cut stored
the picture itself as the bytes, `shared_ptr`s and all, and review found that
a byte copy of the frame (`copyFrameInfo` then `updateData`, or an SDK filter)
would duplicate them without a reference, and outlive what they point at. Now
such a copy owns nothing, and once the frame is freed it carries no picture.
The frame is a 0x0 video frame of `OB_FORMAT_H264`, where the first cut's
NV12 invited a reader to take its bytes for pixels. The SDK refuses
`OB_FORMAT_UNKNOWN` for a colour frame, and a compressed format is sized by
its bytes. No stream here carries H.264, so nothing decodes them either.

No CI leg has a camera, so `process_raw` itself never runs there. What it
does with a device picture is `place_device_color`, which the offline test
runs on NVDEC's buffers on the CUDA leg and on VideoToolbox's images on the
Mac. Both tests require the pictures on the device only on the legs whose
hardware leaves them there: VideoToolbox, and NVDEC in a `VR_WITH_CUDA`
build. The three decoder tests read a device picture back through one
helper, `tests/device_picture_readback.hpp`.
`fuse_orbbec --gpu` now creates the GPU before it opens the cameras, and
hands the capture that device.

The committed clip, fed through the decoder with a device on the M5 Max,
comes out as VideoToolbox images holding each patch's value within 2 codes.
They are described as the Femto Mega codes its stream, and released with
their frames. On the live camera, raw frames arrive with their colour as
images, and the pass still matches the SDK's own undistortion and
registration at 720p and at 4K (depth 5 mm median).

Measured on the lab rig, four Femto Megas at 4K and 25 fps over the wired
link, `fuse_orbbec --rig --gpu --hevc`, 600 frames, Release, two runs each:

| colour | user CPU | sys CPU | fused |
|---|---|---|---|
| on the device | 0.56-0.57 s | 1.61-1.64 s | 94.8-97.4 fps |
| host planes | 1.37 s | 1.60-1.65 s | 92.7-97.4 fps |

That is about 1.4 ms less CPU per camera frame: the copy to the host, the
conversion, the I420 frame and the staging copy. The cameras set the rate
either way.

**MJPEG raw frames have landed.** A raw stream no longer needs H.265. For
MJPEG it streams the camera's MJPG mode, the JPEGs themselves, where the
host path takes the SDK's RGB mode. Each camera's `JpegColorDecoder` decodes
every pair on a thread of its own, ahead of the mailbox as H.265's decoder
does, through `JpegDecoder` onto the device where nvJPEG or VideoToolbox
takes the JPEG and in software to I420 otherwise. It hands the picture on in
the same frame (`picture_frames.hpp`, which now holds both kinds of colour
frame, and `PlanesColor`).

- On its own thread, so a rig's four cameras decode at once rather than one
  after another on the polling thread, where VideoToolbox's 6 ms a 4K frame
  would have cost a set 24.
- A JPEG depends on no other frame, so there is no gate: a pair missing a
  frame, an empty colour frame, a JPEG that does not decode, or a pair a
  newer one replaced while the decoder was busy each cost only themselves,
  all counted in `stats().lost`. The queue holds as many pairs as the
  mailbox after it, newest kept, so a decoder slower than the camera skips
  pairs rather than handing them out seconds late.
- The MJPG mode's calibration must be the RGB mode's, byte for byte, as
  H.265's must.
- `VR_ORBBEC_WITH_HEVC` becomes `VR_ORBBEC_WITH_VIDEO`, since it now gates
  both decoders, and raw frames without them are refused at open.

Measured on the lab rig, four Femto Megas at 4K and 25 fps over the wired
link, `fuse_orbbec --rig`, 600 frames, M5 Max, Release:

| path | user CPU | sys CPU | poll a frame |
|---|---|---|---|
| host MJPEG (the SDK decodes, the host undistorts and registers) | 60.0 s | 11.5 s | 18.6 ms |
| raw MJPEG, `--gpu --mjpeg` | 0.83-1.09 s | 3.66-4.37 s | 0.02 ms |
| raw H.265, `--gpu --hevc` | 0.61 s | 1.71-1.75 s | 0.02 ms |

Raw MJPEG takes about a sixteenth of the host path's CPU, and a little more
than H.265: its system time is mostly the kernel receiving the JPEGs, about
185 Mbit/s a camera against H.265's 21. H.265 stays the rig's default for
that reason and for the link's headroom: `fuse_orbbec --gpu` still streams
it, and `--mjpeg` asks for MJPEG by name. Before, `--gpu` forced H.265, and
the first MJPEG runs measured H.265 for that reason; the table is the rerun.
The offline test holds the decoder to the committed JPEG. It checks I420
frames in software, the device's pictures given one (VideoToolbox's images,
or nvJPEG's buffer on a CUDA leg), each released with its frame, each loss
counted once, and a decoder held back skipping to the newest pair. On the
live camera, raw MJPEG frames arrive as images and match the SDK's
undistortion and registration at 720p and 4K, as H.265's do, and the rig's
raw pass runs over both codecs.

**Next**: measured first, the colour kernel reading Apple's plane images
directly, which saves the copy's 0.28-0.31 ms of GPU.
*Done 2026-10-25 (the plane-image entry):* the colour pass samples them
where they are, and the copy is gone.

*Amended 2026-10-01 (PERF.md's D1 and D3):* the fallback is no longer
silent, and an exported picture buffer was already out of the BAR.

- **The fallback.** Every way a decoder given a device leaves its pictures
  on the host is said once, as a warning through `core`'s log handler: a
  device path that does not open (the back end has none, or its setup
  failed), one that fails mid-stream, and Auto moving a stream to software.
  `Options::label` puts the camera ahead of it, since a rig runs a decoder a
  camera. `OrbbecCaptureStats::host_pictures` counts, of the raw frames
  handed out, those with host colour although the stream decodes onto a
  device, and a frame the rig withdraws from a failed set comes back out of
  it. `fuse_orbbec` and `rig_viewer`'s Rig panel show it when it is not 0.
  On a discrete GPU each such 4K frame costs 12 MB across the bus.
- **Pinned by the tests.** Each decoder's test opens it on a device adopted
  with none of its optional extensions, where no device path opens, and
  holds it to one warning that names the label. The camera's GPU
  pre-processing test holds the count to the host-colour frames it sees (0
  for H.265 and MJPEG on the M5 Max, where a mutant counting every frame
  failed it), then decodes H.265 onto that bare device, where every frame
  must come to the host and be counted, so a count that never moves fails
  too.
- **Already out of the BAR.** Vulkan orders a memory type ahead of any whose
  flags strictly contain its own, so the first device-local type a buffer
  allows is one the host cannot map wherever the buffer allows one:
  `create_exported_buffer` never took the BAR heap over plain VRAM, and D3
  had nothing to fix. What it leaves is `core`'s `find_memory_type`, which
  says so, replaces the three hand-written pickers (`create_exported_buffer`,
  `vt_pictures.mm`'s, the tests' image), and is checked against each CI
  GPU's types by the external-memory test. `vt_pictures.mm` still excludes
  host-visible types, since MoltenVK would back that memory with a buffer.

### 2026-10-06 — The family's camera vocabulary is recon's `camera` tier: a double-precision camera model and the sensor array's calibration file, version 2.

*Amended the same day (below):* the tier keeps only what a capture uses.

calib solves for camera models and writes the rig's calibration; recon's
drivers report their factory calibration and read the poses; ios fills the
same types from ARKit. Until now each had its own: recon a float
`sensor::LensCamera` and the rig file in `sensor`, which links `recon_core`
and so Vulkan; calib a double `PinholeBrownConrady` of five coefficients,
which cannot hold the Femto Mega's factory `k4`-`k6`. Core PR #14 proposed
a camera tier in the core and was closed (camera stays out of the core). The
2026-10-06 plan makes recon the library calib and ios build on for cameras and
sensors; this tier is its first step. The sensor interface (`IRgbdSensor`,
`RgbdFrame`), the vendor-neutral `SensorArray` that replaces `OrbbecRig`, and
pipelined processing follow (DESIGN.md, [Next work](DESIGN.md#next-work)).

- **A tier of its own, under `sensor`.** `recon_camera` links `core_base` and
  GLM only, so calib's solver links no Vulkan to use it. Configuring recon
  still finds Vulkan for its other tiers; calib finds it anyway, for its own
  GPU work on the core's vulkan tier, so a camera-only configure would serve
  no consumer. *Amended 2026-10-11:* calib's solver, which builds with no
  Vulkan SDK, will take its camera model from this tier (decided 2026-10-11;
  calib's CA1), so it is that consumer: `VR_CAMERA_ONLY` builds and installs
  the tier alone, with no Vulkan, glslc or core vulkan tier.
- **Double on the host, float once at upload.** Calibration solves in double
  and the file is written in the shortest form that reads back exactly. float
  would save nothing measurable (a set converts a few models) and cost both:
  Ceres evaluates in double, and float rounds a written calibration.
- **One projection for every scalar.** `distort_rational<T>` and
  `project_rational<T>` take plain arrays: double on the host, float in the
  sensor tier's lens, which `lens.glsl` mirrors, and a solver's dual numbers.
- **OpenCV's rational model only.** Its eight coefficients cover
  Brown-Conrady (`k4 = k5 = k6 = 0`) and the Femto's `BROWN_CONRADY_K6`.
  `CameraModel` carries no model tag yet; the file's `distortion.model` does,
  so a fisheye model is a tagged alternative, not a format change.
- **Projection and unprojection stay inside the invertible radius**: past
  it, a pixel has several rays, and past a pole a point images at a
  mirrored pixel. `project` refuses a point out there; `unproject` is calib's
  Newton with backtracking, generalised, and measures its residual with
  `project`'s own arithmetic, so its 1e-9 px bounds the round trip exactly.
  The radius is where the distorted radius stops growing or the denominator
  reaches zero, found by a 2% scan in r² and bisection, since the rational
  model has no closed form for it. The scan takes 1.5-1.7 µs against an
  unprojection's 0.044 µs (M5 Max, Release), so a `CameraProjection` checks
  a model and finds its radius once, and projects and unprojects with it; a
  free function taking the radius as a number could be handed another
  lens's.
- **Rescaling refuses another aspect ratio** (the half-pixel rule otherwise):
  the Femto's 4:3 colour modes crop its 16:9 sensor.
- **The calibration file, version 2** (`camera/array_calibration.hpp`):
  `format` and `version`; sensors by id, each with its colour camera's pose
  (OpenCV's extrinsic, as version 1), colour and depth models that record
  their `width` and `height` and whether they are `factory` or `calibrated`,
  and `depth_to_color`; and an optional world, a sensor or an AprilTag. Every
  field of a sensor is optional, so a calib session records factory lenses in
  the same format, unposed. Unknown keys are ignored and a later version is
  refused. Version 1 is read and never written: its intrinsics are kept only
  where they record their image, and `optimal_intrinsics`, the legacy
  pipeline's undistortion target, is dropped. A write goes to a temporary
  file of the process's own, synced and renamed into place, beside a
  symbolic link's target, with the replaced file's permissions.
- `OrbbecRig::Options::calibration` takes the `ArrayCalibration`, which must
  pose every camera of the rig. *Amended 2026-10-07 (below):* `OrbbecRig` is
  gone; `SensorArray::Options::calibration` takes it.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec,
FFmpeg and the viewer: the build is warning-free and the 54 tests pass; the
camera tests that need the rig skip without it. `project` reproduces
`cv::projectPoints` to 1e-9 px and `rotation_from_rodrigues` `cv::Rodrigues`
to 1e-15 (OpenCV 5.0.0). Mutation check: dropping the half-pixel term,
swapping `p1` and `p2`, ignoring the denominator's pole, taking the axis
from `sin` past pi/2, keeping version-1 intrinsics without a size, loosening
the world sensor's origin check, letting Newton leave the invertible radius,
and inverting `depth_to_color` each fail a test; so do dropping `project`'s
radius check, measuring `unproject`'s residual in normalized coordinates (as
first written: 5 of 518,400 pixels of a rig lens came back past 1e-9 px),
and writing through a symbolic link or without the old permissions. Every
pixel of the four rig lenses round-trips within 1e-9 px.

### 2026-10-06 — The camera tier keeps only what a capture uses (amends the entry above).

*Amended 2026-10-12 (below):* calib is the first caller; version 2 (under
`device_calibration`), its writer and `rodrigues_from_rotation` return.

Once the tier merged, most of it had no caller outside its own tests:
`CameraProjection` (host projection and Newton unprojection),
`invertible_radius2`, `scale_camera_model`, `project_rational`, version 2 of
the calibration file with its writer, and `rodrigues_from_rotation`.
Reconstruction undistorts on the GPU with the forward model alone (the
2026-09-28 pre-processing decision), so it never inverts a lens, and calib,
the intended user, keeps its own camera model and links none of this. They
are removed, about 1,400 lines with their tests; each comes back with its
first caller.

- **What stays.** `CameraModel` and `check_camera_model` (`RgbdFrame` and the
  GPU frame prep carry and check it); `distort_rational<T>` (the sensor
  tier's float lens, which `lens.glsl` mirrors, and the tests' double
  reference); `check_rigid`, `rigid_inverse` and `rotation_from_rodrigues`
  (the driver's and the array's checks, and reading a pose).
- **The calibration file is the family's `device_calibration` layout
  again**, read for each sensor's pose by serial; lens fields and every other
  key are ignored. Version 2 had no writer outside the tests and no file in
  use: the rig's file is version 1.
- **`SensorCalibration::color_to_world` is a plain `Mat4d`**, since every
  sensor the file lists has a pose.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec,
FFmpeg and the viewer: the 54 tests pass, and the rig's calibration file
reads all four poses. The rig hardware test, which now writes its file as
JSON, was not run.

### 2026-10-06 — Tests use no real hardware (amends the 2026-09-26 Orbbec driver entry).

*Amended 2026-10-07:* no test looks for a camera either. The SDK smoke test
checks only the runtime version, and no longer opens a context and
enumerates devices, which probed USB and the network; the conversion test's
refusal of an absent camera went with the same probe. Every test passes
with the process denied network access.

`recon_sensor_orbbec_capture`, `recon_sensor_orbbec_gpu_prep` and
`recon_sensor_orbbec_rig` are removed, about 1,260 lines. Each skipped unless
an environment variable named a camera or the rig's sync file, and no CI leg
sets one, so they ran only by hand at the rig. A test now runs wherever the
suite builds; a camera is checked by hand through `fuse_orbbec`, and reported
as run or not run.

- **What stays, with no camera:** the conversion test (units, layouts, the
  sync-mode mapping, the option checks `open` makes, and now the refusal of
  H.265 colour without the decoders, moved from the capture test), the
  sync-configuration, grouping, H.265 and MJPEG tests, and the SDK smoke
  test, which passes with no device.
- **What goes:** the checks only a camera gives — the frame contract on a
  live stream, restart, the GPU pass against the SDK's host alignment, and
  the rig's trigger sets. The measurements they produced stay in the entries
  that cite them.

### 2026-10-06 — A sensor's frame is `RgbdFrame`: each camera's `CameraModel`, the poses in double, a sequence number, and the pixels it holds; and the Femto Mega's factory extrinsic is made a rotation.

`RgbdFrame` (`sensor/rgbd_frame.hpp`) replaces `RawFrame`, the second step of
the plan above. It is the frame every driver will hand out once
`IRgbdSensor` lands; `sensor::LensCamera` and `LensDistortion` are gone.

- **Each camera's `camera::CameraModel`**, lens included, in double.
  `GpuFramePrep` narrows it to float once, as `lens.glsl` reads it, and
  refuses a model past float's range.
- **The colour camera's pose and the sensor's extrinsic**, not two poses:
  `color_to_world` is where the sensor sits, from a calibration or a
  tracker; `depth_to_color` is the sensor's own. The pass poses the depth
  camera at their product and refuses either unless rigid.
  `OrbbecCapture::Options::cam_to_world` is double too and refused at `open`
  unless rigid, so the driver never hands the pass a pose it refuses.
- **A sequence number**, the device's count of its frames, so a gap is a
  lost frame; the Orbbec driver stamps the depth frame's index.
- **The pixels it holds** (`pixels`), not ones it borrows until the next
  poll: a sensor array groups frames by trigger before it prepares any, so
  it keeps several of each sensor. The Orbbec driver hands over the SDK's
  pair with the SDK context, so its buffers go back once every copy of the
  frame is gone, before the SDK does, even after the capture.

**The Femto Mega's factory depth-to-colour rotation is not a rotation.**
Read through the SDK (2.9.3, firmware 1.3.1) from all four cameras of the
lab rig, it describes the depth camera's tilt, 6.1° to 6.7° about x, but its
first element is cos θ where Rodrigues' formula gives
cos θ + (1 − cos θ) k_x² ≈ 1: the other eight match that formula to 1e-5,
and the rows are 1.15% to 1.36% off orthonormal. The raw path used the
matrix as it came, so depth landed 0.6% short along the colour camera's x
(about 4 px at a 720p image's edge), and fusion, which inverts a pose by its
transpose, doubled the inconsistency. `camera::nearest_rotation`, the polar
factor, recovers the Rodrigues rotation to about 1e-5 rad, and the driver
applies it to every extrinsic it reads, refusing at `open` one that does not
come out a rotation (a zeroed or reflected matrix). The hardware test's comparison of the
prepared depth with the SDK's own host alignment moved from a 5.3 mm to a
6.7 mm median, consistent with the SDK aligning through the defective
matrix. Which is closer to the scene is for calib to measure, against the
colour board.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec,
FFmpeg and the viewer: the 51 tests pass, `recon_sensor_gpu_frame_prep` on
the device with the new pose composition and refusals. Before the camera
tests went (the entry above), on the rig's primary, CL2A141000N, over Wi-Fi:
`recon_sensor_orbbec_gpu_prep` passed raw H.265, raw MJPEG and the forced
host-colour path, the first frame kept intact through the 45 polls after it
and prepared, and the sequence rising.

### 2026-10-06 — A sensor is an `IRgbdSensor`: what it is before a frame (`SensorInfo`), frames polled newest or drained oldest first; and the Femto Mega is `OrbbecSensor`.

The third step of the 2026-10-06 plan (`sensor/rgbd_sensor.hpp`): the
interface every driver implements, used on its own or by the sensor array
that replaces `OrbbecRig`.

- **`SensorInfo` is what a sensor is before a frame exists**: its id (the
  serial a calibration names), its cameras' factory models at the opened
  modes, its factory `depth_to_color`, its rig role, which clock its
  timestamps are on, whether a calibration or the sensor itself poses it,
  and its rate. An array needs each of these before it starts: the roles
  order the starts, grouping needs one clock, the id finds the pose.
- **Polled, not called back, and two ways to take frames.** A sensor holds
  up to `set_queue_depth` frames, set before `start`; `poll` hands out the
  newest and lets the older go, `drain` hands out every held frame oldest
  first. A standalone consumer polls; an array sets a deeper queue and drains,
  so frames of one trigger are still held when it groups them -- the
  Orbbec mailbox's `take`/`take_all`, made the contract. A `drain` that
  fails partway keeps the frames before the failure, handed out, rather
  than losing good frames to a sensor that just broke.
- **Counters every driver has** (`SensorStats`: received, delivered, dropped,
  failed). A driver keeps its own beside them: `OrbbecSensor::orbbec_stats`
  splits failed into failed and lost and counts host pictures. A raw MJPEG
  pair the JPEG decoder had no time for is dropped, as the mailbox's are,
  not lost, so backpressure never reads as decode failures.
- **`OrbbecSensor`** is the raw path behind the interface: frames as
  captured, each holding its SDK pair, models and extrinsic from the
  factory calibration, the pose given in double. `open` refuses what would
  only fail later: a `min_depth` of 0, which the GPU pass refuses, and a
  sync mode the driver does not know, whose start an array could not order.
- **`sync_clock_to_host` syncs this camera alone, once per `start`.** The
  context's `enableDeviceClockSync` runs one SDK thread over every camera
  the process created (`DeviceManager`, SDK 2.9.3), so one sensor's option
  would re-sync another that reports its own clock, and run on after it. The
  camera's own `timerSyncWithHost`, before it streams, keeps timestamps
  monotonic within a session, and a camera that cannot sync fails `start`
  rather than reporting `ClockDomain::Host` falsely. Its drift over a
  session is unmeasured (a `TODO(sensor)`). *Amended 2026-10-07 (below):*
  measured, 3–6 µs/s between cameras, so the option now stamps the SDK's
  global timestamps, the camera's clock mapped onto the host's; the one
  sync before each `start` stays.
- **Beside `OrbbecCapture` for now.** `OrbbecCapture`, `ICameraCapture`,
  `CapturedFrame` and the SDK's host path go when the Replica source moves
  onto the interface (a `TODO(sensor)`), and `OrbbecRig` when the array
  lands. *Amended the same day (below):* `OrbbecCapture` and the host path
  went first, apart from the Replica source. *Amended 2026-10-07 (below):*
  the Replica source is an `IRgbdSensor`, and `ICameraCapture` and
  `CapturedFrame` are gone.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec,
FFmpeg and the viewer: the 52 tests pass. `recon_sensor_rgbd_sensor` drives
a fake through the interface, and `recon_sensor_orbbec_conversion` checks
what `OrbbecSensor::open` refuses before it touches the SDK. `OrbbecSensor`
was checked by hand, not by a test (tests use no hardware), through
`fuse_orbbec --gpu`, which reads one camera through it, on CL2A141000N: H.265
and MJPEG each fused 60 frames at about 30 fps, printing both cameras'
factory models, every received pair accounted for (H.265: 64 received, 60
fused, 3 dropped, 1 lost). With `--host-clock` the first frame sat 197-243
ms behind the host's system clock; without it, 276 ms, the camera keeping
the clock the run before set. The example does not `drain`: the fake checks
it, and the array's example will at a camera. *Amended 2026-10-10:* the fake
queue only tested its own implementation. The sensor vocabulary test now
checks defaults, return helpers, copied-frame ownership and enum names directly;
production array and decoder behavior remains covered separately.

### 2026-10-06 — Several sensors are a `SensorArray`, vendor-neutral: started in their rig's order, grouped by trigger or by sequence number, and posed by the calibration.

*Amended 2026-10-06 (trigger only, below):* `SyncMode::Sequence` went, and
with it `SyncMode`: an array groups by trigger only.
*Amended 2026-10-07 (the rig entry, below):* `OrbbecRig` and its start order
are gone, `TriggerGrouper` is the array's internals, and `tolerance_us`
defaults to 0.4 of the fastest member's frame period.

The fourth step of the 2026-10-06 plan (`sensor/array/sensor_array.hpp`,
target `recon_sensor_array`): the array of `IRgbdSensor`s that replaces
`OrbbecRig`, built from what `OrbbecRig` did that no vendor owns.

- **What it reads of each sensor is its `SensorInfo`.** The roles order the
  starts (secondaries, then free-running sensors, the primary last) and the
  stops (the reverse); the id finds the pose in the `ArrayCalibration`, which
  must pose every sensor and is stamped on each frame's `color_to_world`.
  What a vendor's rig needs beyond that -- checking and writing sync settings,
  the SDK's clock sync -- is set when its driver opens the sensor.
- **Two ways frames form sets.** `SyncMode::Trigger` is the rig's grouping,
  `TriggerGrouper` moved out of the Orbbec driver into the sensor tier
  unchanged: around the primary's frames, within `tolerance_us` on the host's
  clock, a missing secondary waited for a frame and a half of the slowest
  sensor, the newest ready set handed out. It needs one primary, the rest
  secondaries, all on the host clock. `SyncMode::Sequence` groups equal
  sequence numbers, every set in order once each sensor has sent its frame,
  moved past it or ended; it reads no clock, so a recording replays the same
  sets every time, which is what calib's session replay needs. A frame for a
  set already handed out joins none, and a sensor ahead of one that has
  fallen behind holds `queue_depth` frames, its oldest let go, so the array's
  memory is bounded in both modes.
- **Lifecycle.** A failed start, a stop, the destructor and a move over a
  running array all stop the primary first. Starting a running array starts
  each sensor again, as `OrbbecRig` did, so a camera that has gone away says
  so. A refused `open` leaves the sensors with the caller, and a drain that
  fails partway has its earlier frames, counted delivered, grouped before the
  failure is returned.
- **Refused for now, each a `TODO:`:** tracked sensors (an iPhone's
  pose needs registering to the array's world) and free-running members of a
  triggered array (joined by their nearest frame).
- `process(set)`, every stream of every sensor in one GPU batch, is the next
  PR; then `OrbbecRig` goes and `rig_viewer` and `fuse_orbbec` open an array of
  `OrbbecSensor`s.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec,
FFmpeg and the viewer: the 54 tests pass. `recon_sensor_array` drives
scripted sensors through the refusals, the start and stop order, a failed
start, a start of a running array and a move over one, trigger sets
(complete, a silent secondary, the newest set, frames without a clock) and
sequence sets (out of turn, a missing frame, waiting on a live sensor, a
recording's end, a late frame, a sensor far ahead, a restart), and a drain
that fails partway; 50 runs in a row pass, its waits notwithstanding. Not
applying the poses, not waiting on a live sensor in sequence mode, and not
checking the clock each fail it, as does undoing any lifecycle or sequence
rule above. `recon_sensor_trigger_grouper` is the grouping test the rig's
was; the rig's start order is `recon_sensor_orbbec_start_order`.

### 2026-10-06 — An array's set is prepared in one batch: every stream of every sensor, one submit and one wait, host colour staged on a thread per camera first.

*Amended the same day (the device-only `GpuFramePrep` entry, below):* colour
comes on the device only, and with only depth to stage, the thread per
camera measured level with one thread and went.

`SensorArray::process(set)` prepares a set through
`GpuFramePrep::prepare_batch`, a pass per sensor: every frame is checked
first, so a refused set costs no work; device planes are taken over; each
pass stages its host data on a thread of its own; then every pass records
its uploads, and after them every pass its kernels, into one `CommandBatch`,
submitted and waited on once. Recording the uploads first saves a barrier
a camera: they write buffers of their own, so none is needed between them.
`prepare` is the same steps for one frame. `prepare_set`, a thread and a submit per camera,
is removed: `fuse_orbbec --rig` and `rig_viewer` call `prepare_batch` until
they open an array, rather than keep two ways to prepare a set. *They open
one since 2026-10-07 (below).*

**Why one batch, given PERF.md's P2 found it no faster.** It is not: it is
level. The 2026-10-01 measurement lost on the Mac's host colour because the
batch staged four 4K frames one after another; staging them on threads
first closes that. Median of 100 sets of four 640 x 576 depth and
3840 x 2160 colour frames, M5 Max, Release, each pair interleaved, three
runs: one batch against a thread and a submit per camera, 1.68 vs 1.66 ms
for host colour and 0.97 vs 0.95 ms for device colour; the uploads recorded
first against each pass's uploads and kernels together, 1.69 vs 1.74 ms and
0.99 vs 1.01 ms.

What it buys is one batch a set, the shape the pipelined stages (DESIGN.md's
Next work, step 4) need: a set's work submitted once, through the core's
`submit_async`, behind the previous stage on a timeline, where four threads
would make four submits to order. Until then it submits and waits, which a
`TODO:` at the submit marks. A failed submit abandons every pass's staging
and device planes, since the work may still run and `CommandBatch::submit`
cannot say whether it does; `PendingBatch::in_flight` will. The RTX 5090 was
level before the threaded staging and is not re-measured.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors,
Orbbec, FFmpeg and the viewer: the 55 tests pass.
`recon_sensor_array_process` prepares a three-sensor recording's sets -- one
missing a frame -- and holds each prepared frame to `prepare` on its own,
byte for byte, posed by the calibration. `recon_sensor_gpu_frame_prep` does
the same for a batch whose colour comes as host planes, device planes taken
over from outside Vulkan and NV12 plane images, with depth kept within
colour; it checks that a refused set times no row, so no work began, and
that a dozen cameras' set is still timed (the timer's default window, 32
spans, held eight cameras' host colour). Both run clean under the Khronos
layer's synchronization validation. Dropping the timer's reserve, or opening
the stage before the checks, fails the test. The split `prepare` was checked
by hand on CL2A141000N over Wi-Fi through `fuse_orbbec --gpu`: the default
stream, `--mjpeg` and `--hevc` each fuse 30 frames into a mesh. `fuse_orbbec
--rig` and `rig_viewer` on `prepare_batch` build, and were not run on the
rig.

### 2026-10-06 — Every Orbbec frame is handed out as captured: the SDK's host path, `OrbbecCapture` and `fuse_orbbec --gpu` go (amends the 2026-09-26, 2026-09-27 and 2026-09-28 Orbbec entries).

**The rule.** Once a frame arrives it goes to the GPU and is processed there.
The Orbbec driver therefore has one path: raw depth and the decoded colour,
each camera with its factory model, for `GpuFramePrep`. Removed:

- `OrbbecCapture`, the SDK's host undistortion and registration
  (`ob::UnDistortionFilter`, `ob::Align`), the host conversion to metres and
  packed RGB, and `OrbbecStreamOptions::raw`.
- The rig's host reads (`poll`, the processed `poll_set`) and `poll_raw`:
  `OrbbecRig` is no longer an `ICameraCapture`. Its one read is `poll_set`,
  formerly `poll_raw_set`, handing out an `OrbbecRigSet` of `RgbdFrame`s, and
  `color_camera` is the factory `CameraModel`, with `color_to_world` beside it.
- `ICameraCapture::poll_raw` and `raw_frames`, which only the two Orbbec
  classes implemented.
- The H.265 decoder's RGB frames and the RGB profile it stamped on them.
- The RGB mode `open` read each colour camera from, and its byte comparison
  with the mode streamed. Only the host path streamed the RGB mode, so the
  comparison could refuse a camera over a calibration nothing used, with no
  path left to open it. A colour camera is read off the H.265 or MJPG mode
  streamed, the pictures the decoder hands out. The two matched byte for byte
  on the Femto Mega at 720p, 1080p and 4K, so the rig's models are unchanged.
- `fuse_orbbec`'s `--gpu`. It always reads raw, H.265 unless `--mjpeg`, and
  every poll goes through `prepare_batch` and `fuse_set`, one camera as a
  set of one. `--host-clock` stays for one camera: it is not host
  processing, and it is how `OrbbecSensor::Options::sync_clock_to_host`,
  which a `SensorArray` grouping by trigger needs, is checked by hand.
- `VR_ORBBEC_WITH_VIDEO`: `VR_WITH_ORBBEC` now needs `VR_WITH_FFMPEG`, since
  a driver without the decoders could open no camera. Both CI legs that
  build the driver build FFmpeg.
- The `TODO(calib)` comparing the made depth-to-colour rotation with the
  SDK's matrix through its host alignment: that alignment is gone, and
  calib's own solve answers the question.

The types the two remaining classes share move to `orbbec_stream.hpp`
(`OrbbecStreamOptions`, `OrbbecDeviceInfo`, `OrbbecColorCodec`, and
`OrbbecStreamStats`, formerly `OrbbecCaptureStats`), and `OrbbecSyncMode` to
`orbbec_sync_config.hpp`.

**Why now, not with the Replica source.** The plan tied the removal to the
Replica source moving onto `IRgbdSensor`, but nothing in it depends on
`OrbbecCapture`. Meanwhile the host path was `fuse_orbbec`'s default, at
15.5 ms of host time a 4K frame and eight times the run's CPU (the
2026-09-28 GPU pre-processing decision), and a rig through it fused its
cameras one at a time, uploading float depth and packed colour for each.

**Left for the next step.** The video decoders still hand out host pictures
where no device path opens (software, VAAPI, NVDEC without `VR_WITH_CUDA`)
or the stream was given no device or allocator. `VideoPixelLayout::Rgb24`
and swscale's RGB conversion, which lose their caller here, go with that
step rather than be rewritten twice. *Done the same day (below).*
`OrbbecRig` goes once the examples read a `SensorArray`, which first needs
`OrbbecSensor` to check and write the sync settings the rig does. *Done
2026-10-07 (below).*

**Validation.** Apple M5 Max, macOS, Release with warnings as errors,
Orbbec, FFmpeg and the viewer: the 55 tests pass. The conversion test now
checks the stream options, `OrbbecSensor::open`'s refusals and the factory
models' intrinsics checks; the H.265 test checks every decoded pair's I420
planes and their colour description. `fuse_orbbec` and `rig_viewer` build;
neither was run on a camera.

### 2026-10-06 — The video decoders hand out pictures on the device only: the hardware path is fixed per build, a stream the hardware refuses is an error, and off Apple `VR_WITH_FFMPEG` needs `VR_WITH_CUDA` (amends the 2026-09-27 decoder, 2026-09-28 decoded-frame and 2026-10-06 raw-frames entries).

**The rule.** Once a frame arrives it goes to the GPU and is processed there,
decoding included. `HevcDecoder` and `JpegDecoder` hand out pictures on the
device only, on one hardware path fixed per build: VideoToolbox on Apple
(H.265 through FFmpeg, JPEG directly), and elsewhere NVDEC through FFmpeg and
nvJPEG's hardware engine, both through CUDA. Removed:

- software decoding, VAAPI, NVDEC without `VR_WITH_CUDA`, and copying a
  hardware picture to the host;
- the fallbacks: `Auto`'s move to software on a refusal, a failed device path
  sending every later picture to the host, and nvJPEG's `GPU_HYBRID` back end
  with the software decode of what it refused. `GPU_HYBRID` decodes the
  entropy on the host, 4.2 ms of CPU a 4K frame against the engine's 0.38
  (the 2026-09-28 decoded-frame decision);
- the selection that had nothing left to choose: `VideoDecodeBackend`,
  `hardware_backends()`, the probe clip, `JpegDecodeBackend`, `backend()`,
  H.265's `layout` and `threads`, both decoders' `label`, and the JPEG
  decoder's FFmpeg (it uses none now);
- `PictureConverter` and libswscale with it, `VideoPixelLayout::Rgb24` and
  `DecodedPicture::plane`. The colour helpers stay, as `frame_color.hpp`;
- the host-picture warning, `OrbbecStreamStats::host_pictures` (PERF.md's D1)
  and the Orbbec driver's I420 host frames.

**The codes are the contract.**
- `IoError`: data that does not decode. It costs that frame, and for H.265
  the frames to the next key frame.
- `Unsupported`: a stream or JPEG the hardware does not take, or a `create`
  with no device path (no device or allocator, or a GPU the build's path
  cannot reach). An H.265 stream is held to 8-bit 4:2:0 at its SPS, through
  `sw_pix_fmt` (`YUVJ420P` too, a stream whose VUI says full range), and on
  VideoToolbox to no crop at the left or top; a JPEG to baseline 8-bit 4:2:0
  within 16384 a side on nvJPEG's engine and the device's image extent on
  Apple. An H.265 refusal stands from then on, after the pictures decoded
  before it.
- `Backend` or `OutOfMemory`: the device path failed. A CUDA error is
  `Backend` now, not `IoError`.

The Orbbec colour decoders stop on any code but `IoError`, and
`CameraStream::take` returns it, so `OrbbecSensor::poll` and
`OrbbecRig::poll_set` report a stream that can no longer be decoded rather
than starve quietly. The audit of 2026-10-06 found that, once the fallbacks
went, a decoder that kept failing would have looked like a slow camera.

**Off Apple, `VR_WITH_FFMPEG` needs `VR_WITH_CUDA`**, refused at configure,
and the FFmpeg floor rises to 6.1 (Ubuntu 24.04). In CI the 24.04 leg (CUDA:
NVDEC, and nvJPEG where the GPU has the engine) and the Mac (VideoToolbox)
build and test the decoders. A GPU with no hardware JPEG engine decodes no
MJPEG: one CI host's is now an RTX 4090, which has none (nvJPEG status 7),
so the JPEG tests skip there and run on the RTX 5090 host. The 22.04, 26.04 and sanitizer jobs no longer build them, so FFmpeg
4.4 and 7.x on NVDEC, and the decoders under the sanitizers, lose coverage.

**The tests** read each device picture back and hold it to the pattern the
fixture was made from, rather than to a software decode; bit-exactness against
software and swscale's RGB checks go with it. They skip where no device path
opens, unless `VR_TEST_HEVC_BACKEND` promises one, as CI's legs do; a GPU
with no JPEG engine skips the JPEG tests regardless.

**Not taken.** A failed device path is not latched inside the decoders: their
consumer stops at the first. The Orbbec driver still opens without a device
and refuses at `start`, which keeps its option checks testable with no GPU.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec,
FFmpeg and the viewer: the 54 tests pass with
`VR_TEST_HEVC_BACKEND=videotoolbox`. Mutating VideoToolbox's crop refusal,
the MJPEG decoder's stop on `Unsupported`, or the H.265 decoder's on a
refused stream fails its test. CI's 24.04 leg on an RTX 4090 passes the
H.265 tests on NVDEC, the first run of that path. Not run on a camera.

### 2026-10-06 — `GpuFramePrep` takes colour on the device only, and a set's depth is staged on one thread (amends the 2026-09-28 decoded-frame and 2026-10-06 one-batch entries).

*Amended 2026-10-07 (below):* host colour is back, as packed R'G'B' words
(`RgbdFrame::color_packed`) staged beside the depth on the calling thread
and copied up in the pass's own batch; Y'CbCr planes stay device-only.

**The rule.** `GpuFramePrep` takes a frame's colour on the device only: I420
or NV12 planes in a buffer (`YuvImage::device`), or NV12's planes as images
(`YuvImage::image`). `YuvImage::plane`, the host planes, goes, with their
staging and the packing of their rows. Only depth goes up from the host:
one copy into the pass's staging, and one in its batch. `prepare_batch`
stages each camera's depth on the calling thread; its thread per camera
(`run_each`) goes.

**Why.** Nothing made host colour any more. The decoders hand out device
pictures only (the entry above), so the host planes were a path only the
tests took, against the rule that a frame is on the GPU from its arrival.
With colour on the device, the threads staged only depth, 0.7 MB a camera
at 640 x 576, and measured level with one thread. Median of 400 sets of four
cameras, each pair interleaved, Apple M5 Max, Release, NV12 colour in a
device buffer, threads against one thread:

| depth, colour | threads | one thread |
|---|---|---|
| 640 x 576, 3840 x 2160 | 1.035 ms | 1.025 ms |
| 1024 x 1024, 3840 x 2160 | 1.054 ms | 1.063 ms |
| 640 x 576, 1280 x 720 | 1.033 ms | 1.024 ms |

The RTX 5090 is not re-measured: the staging is the same host copy there.

**Tests.** The prep tests upload each picture as a decoder leaves it, I420 in
a device buffer, its rows padded where a test asks, and keep their buffer
and image layouts; the host-plane cases go. `recon_sensor_array_process`'s
recording holds its colour in a device buffer.

**Validation.** Apple M5 Max, macOS, Release, Orbbec and FFmpeg: the 54 tests
pass with `VR_TEST_HEVC_BACKEND=videotoolbox`, and
`recon_sensor_gpu_frame_prep` and `recon_sensor_array_process` run clean
under the Khronos layer's synchronization validation.

### 2026-10-06 — A sensor array groups its frames by trigger only: `SyncMode` and sequence grouping go (amends the 2026-10-06 `SensorArray` entry).

**Why.** The owner: "we don't need it. the main use case for sensor array is
for live synced camera capture which frames arrive within fixed time
interval." Nothing outside recon's tests grouped by sequence -- no example,
calib or ios code opens a `SensorArray` yet -- and it had a known bug: a
sensor repeating a sequence number sent its set out twice.

**What goes.** `SyncMode`, its `to_string` and `SensorArray::Options::sync`,
which with one mode left chose nothing; the per-sensor queues, the last set's
number, and the start-order slot for a second primary. An array groups by
`TriggerGrouper` around the primary's frames on the host clock; more than one
sensor needs one primary, the rest secondaries, all on that clock, and a
single sensor needs neither. `SensorArray::exhausted` is `true` once any
sensor is. `queue_depth`, `tolerance_us`, `IRgbdSensor::exhausted`,
`RgbdFrame::sequence` and `FrameSet::sequence` (the primary's frame's) stay.
A replay that must give the same sets every time -- calib's session replay,
which sequence grouping was for -- is a new decision when one needs it.

**Verified.** Apple M5 Max, macOS, Release with warnings as errors:
`recon_sensor_array` drops its sequence cases and runs the failed drain and
an exhausted sensor through trigger grouping, and opens one free-running
sensor on its own clock; `recon_sensor_array_process` groups its three
sensors by trigger, one set a poll, the set missing a sensor waited for and
then handed out without it.

### 2026-10-07 — `GpuFramePrep` stages a host frame's packed colour beside its depth (amends the 2026-10-06 device-only `GpuFramePrep` entry).

**The rule.** A frame may carry its colour on the host as packed R'G'B'
words, `RgbdFrame::color_packed`: the colour camera's size, row-major,
`R | G << 8 | B << 16` with the high byte ignored (`io::load_color_packed`'s
layout), held by `pixels`, and set instead of `YuvImage`'s planes, never
beside them. `GpuFramePrep` stages the words as it stages host depth: one
copy into the pass's staging, after the depth, and one in its batch into the
colour input. The colour pass reads R, G and B as byte planes four bytes
apart through the planes' own bilinear sampler, so its clamping, weights and
coverage byte are the Y'CbCr path's; a push constant, `packed_rgb`, picks
the branch (`ColorParams` is 108 bytes). `prepare_batch` stages each
camera's packed colour with its depth, on the calling thread. Depth stays
16-bit with `metres_per_unit`: the frame gains no float depth.

**Why.** The Replica examples' source, a dataset decoded on the host,
becomes an `IRgbdSensor` whose frames go through the pass as the cameras'
do, so host colour has a real caller again; the 2026-10-06 entry dropped it
because nothing made it. The words go into the staging as decoded, with no
conversion on the host, so a frame is still one upload in the pass's own
batch, and Y'CbCr planes stay on the device.

**The threads stay gone.** Host colour staged one camera after another lost
to a thread per camera at four 4K cameras on the Mac (the 2026-10-01
measurement, in the one-batch entry). The one caller is a single camera; a
rig with host colour measures that again before the threads come back.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec
and FFmpeg: the 56 tests pass. `recon_sensor_gpu_frame_prep` holds packed
words through a pinhole to their input exactly, the junk high byte replaced
by full coverage, and through a lens to a host bilinear reference within a
code, no coverage byte off; refuses words beside planes before any work, and
a PQ transfer; and prepares a fifth camera's packed colour in a batch byte
for byte as `prepare` does alone, depth kept within colour. It runs clean
under the Khronos layer's synchronization validation.

### 2026-10-07 — The Replica source is an `IRgbdSensor` and the examples fuse what `GpuFramePrep` prepares: `ICameraCapture`, `CapturedFrame` and `OwnedFrame` go, and `io` loads depth as stored (amends the 2026-08-02 sensor-tier, 2026-09-14 examples and 2026-10-06 `IRgbdSensor` entries).

*Amended 2026-10-08 (the streamed-atlas entry, below):* `fuse_viewer` copies
its keyframe's colour into the atlas on the device, and its
`atlas readback` row is gone.

The first step of the 2026-10-06 plan finishes: one interface for every
source, one frame type, one fusion path.

**The rule.**
- **The Replica source is an `IRgbdSensor`**: `ReplicaSensor`
  (`examples/common/replica_sensor.hpp`, formerly `ReplicaCapture`). Its
  `SensorInfo` names one pinhole `camera::CameraModel`, no distortion, for
  both cameras, since Replica renders depth and colour from one, with an
  identity `depth_to_color`; `FreeRun`, `Device` clock (every timestamp 0,
  "none"), `PoseSource::Tracked` (each frame carries its trajectory pose;
  the enum's doc now names a recording) and fps 0. A frame's depth is the
  PNG's 16-bit samples as stored, with `metres_per_unit = 1.0f / scale`
  computed in float as the old loader did; its colour is
  `RgbdFrame::color_packed`, declared canonical; its `color_to_world` is the
  trajectory pose, parsed in float as before and widened, so the pass
  narrows it back exactly; its `sequence` is how many frames the sensor
  handed out before it, so a stride is not a loss; and `pixels` owns both
  images. `poll` stays consumer-paced, `drain` appends what `poll` would
  return, `set_queue_depth` accepts at least 1 before `start` and changes
  nothing, and `stats` counts received and delivered. `open` refuses a
  `min_depth` of 0, which the pass reads as "no return", as
  `OrbbecSensor::open` does, and the three examples' `--min-depth` takes
  (0, max). The preload holds 6 bytes a pixel, not 8. The move constructor is
  defaulted and assignment deleted: the frames to play are one vector of
  entries, so a moved-from sensor plays nothing and is exhausted.
- **Every example fuses a `DeviceFrame`.** `fuse_replica`, `codec_replica`,
  `fuse_render` and `fuse_viewer` prepare each polled frame with one
  `GpuFramePrep` and fuse it through `fuse_set`, which `fuse_frame.hpp` now
  holds (`fuse_device_frame.hpp` folded into it; the host-depth
  `allocate_band` and `fuse_frame(CapturedFrame)` went, and
  `allocate_band_with` folded into the one `allocate_band` left). The
  viewers keep their keyframe as its `DeviceFrame`, texture through its
  colour camera with the single-view `TextureView`, and read its colour
  back with a `CommandBatch` for the atlas; `to_canonical` leaves them, as
  `prepare` refuses colour that is not canonical. `fuse_viewer` keeps its
  last successfully fused frame until the next preparation and fusion both
  succeed (`fuse_keyframe`), so a failed attempt leaves the final and later
  remeshes textured. This holds one previous output pair while the pass
  allocates the next; releasing it before preparation reused the buffers
  but lost the keyframe on a refused pose. It shows a `frame prep` row;
  its `atlas pack` row is `atlas readback`,
  with a `TODO(examples)` to copy the atlas on the device as `rig_viewer`
  does.
- **`ICameraCapture`, `CapturedFrame` and `OwnedFrame` go**, with the
  conventions test's `FakeCapture`; `recon_sensor_rgbd_sensor` covers the
  interface.
- **`io::load_depth_metres` is `io::load_depth_u16`**: the samples as
  stored, under the same 16-bit grayscale checks. Its one non-test caller
  was the Replica source, which now hands the samples on with their unit.

**Why.** A frame borrowed its pixels until the next poll, so a consumer
copied one to keep it (`OwnedFrame`); an `RgbdFrame` holds them, so the copy
has nothing to do. Every Replica run now goes through the pass a camera's
frame does, so the dataset examples exercise the path the live rig runs
rather than a host path no camera uses.

**What stays for the iOS scanner, and why.** The scanner, pinned at an older
recon, implements `ICameraCapture` and fuses host frames. What it calls
stays until it moves onto the interface: `VoxelHashMap::allocate_from_depth`
over host depth, `TsdfIntegrator::integrate`'s host overloads,
`ColorFrame::pixels`, `ProjectiveTexturer`'s host-depth overloads,
`sensor::to_canonical`, `cv_from_gl_camera` and
`depth_from_registered_color`. The 2026-10-06 test-only API rule counts a
sibling app's caller, so these are the scanner's, not test-only; they go or
move with its migration. `depth_from_registered_color` returns a
`DepthCameraParams`, which `RgbdFrame` has no slot for, so that migration
needs a `CameraModel` rescale for the 256x192 depth; it is not written
before it has that caller.

**No float depth.** DESIGN's Next work had the frame gaining float depth
for ARKit. It does not: depth stays 16-bit with `metres_per_unit`, one depth
format for the pass's one depth kernel, 2 bytes a pixel up. When the scanner
moves onto the interface it quantises ARKit's float `sceneDepth` on the
host, in 0.1 mm units, say, which reach 6.55 m.

**Verified.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec
and FFmpeg: the 56 tests pass, `fuse_replica`, `codec_replica` and
`fuse_orbbec` build, and `recon_io_image` reads the 16-bit fixture as
stored. `fuse_replica`, streaming and with `--preload --stride 2`, and
`codec_replica` ran on a synthetic three-frame 64x48 scene in Replica's
layout, each vertex's colour the one its depth's column was painted, and
`--min-depth 0` is refused. `fuse_render` and `fuse_viewer` were not
compiled here (the viewer is off; CI's viewer leg builds them), nor was
`rig_viewer`, whose source lost only an include. On room0, the first 200
frames at 2 cm, `fuse_replica` before and after this change, streaming and
with `--preload`, give the same 309,353 triangles, positions, normals and
colours bit for bit; only their order differs, as compaction's does from
run to run.

**Review regression.** `recon_example_fuse_keyframe` drives the viewer's
fusion helper without a window or dataset. A refused pose and an integration
failure after successful preparation both leave the previous depth, atlas
bytes, coverage and cameras intact, and the retained frame still textures
the surface. A successful retry replaces it. The viewer publishes the pose
only after the helper succeeds, alongside the fused-frame count.

### 2026-10-07 — A rig is an `OrbbecSensor` per camera in a `SensorArray`, on the SDK's global timestamps (SDK 2.10.6 or later): each camera is checked against its sync settings at `open` and written only when asked, the tolerance is 0.4 of a frame period, and `OrbbecRig`, its start order and the public `TriggerGrouper` go (amends the 2026-09-24 SDK, 2026-09-27 rig and 2026-10-06 `IRgbdSensor` and `SensorArray` entries).

*Amended 2026-10-08:* opening a rig is the library's.
`open_orbbec_sensors` (`orbbec_sensor.hpp`) reads the sync file and opens
each camera, in the file's order, with its entry as `Options::sync` and on
the host's clock, returning the sensors `SensorArray::open` takes; the
caller's options give the streams and `apply_sync`, never a serial or sync
settings, which the file gives, or a pose, which the array's calibration
gives. `fuse_orbbec` and `rig_viewer` call it, as calib's live app is to, so
a new rule for a rig's cameras is made once. Test:
`recon_sensor_orbbec_open_sensors`, against the camera stub the sync-apply
test uses (`tests/orbbec_camera_stub.cpp`).

The array's plan finishes (DESIGN.md's Next work, step 2): `fuse_orbbec`
and `rig_viewer` open one `OrbbecSensor` per camera of the rig's sync
configuration and read them as one `SensorArray`. What `OrbbecRig` did for
the Orbbec SDK moves into `OrbbecSensor`, one camera at a time; the rest was
already the array's.

**The rule.**
- **A camera's sync settings are checked at its `open`.**
  `OrbbecSensor::Options::sync` takes the camera's entry of the rig's file
  (`read_orbbec_sync_config`). `open` compares it with what the camera
  stores (`sync_differences`, which reads `SECONDARY_SYNCED` as `SECONDARY`)
  and refuses a difference, `Unsupported`, naming the camera and each field,
  unless `apply_sync` writes them. `apply_sync` without `sync` is
  `InvalidArgument`, before the SDK is touched. The role and the
  software-triggering and unknown-mode refusals are decided after any write,
  so `apply_sync` can repair a camera left in software triggering.
- **Written only on a difference, then checked for SDK normalization.**
  The settings persist in the camera's flash, which Orbbec's sample warns
  frequent writes wear. After
  a write the SDK answers a read from its cache of the effective settings,
  not the camera (`DeviceSyncConfiguratorOldProtocol::getSyncConfig`).
  That cache still matters: its setter forces a primary's
  `triggerOutEnable = true` and `triggerOutDelayUs = 0` before writing and
  caching. `open` compares again and refuses any remaining difference,
  `Unsupported`, naming the camera and fields. Removing the old rig's
  comparison falsely accepted a request for `false` / `250`; a public
  `OrbbecSensor::open` test with a camera stub reproduces it without hardware.
  This checks normalization, not persistence: the next `open` reads the
  camera and checks the stored settings.
- **`sync_clock_to_host` means the SDK's global timestamps.** `open` refuses
  a camera without them (`isGlobalTimestampSupported`) and enables them
  (`enableGlobalTimestamp(true)`; the SDK's own config leaves the Femto
  Mega's off). Each `start` still sets the camera's clock to the host's once,
  before it streams (`timerSyncWithHost`), and the SDK then re-fits its
  mapping synchronously, so the first frames are mapped. `timestamp_ns` is
  the depth frame's `getGlobalTimeStampUs()`: the camera's clock mapped onto
  the host's, the mapping re-fitted every 10 s (2.10.6). Nothing re-syncs a
  streaming camera, and nothing calls `enableDeviceClockSync`. A zero, no
  mapping yet, stays zero, and the array counts it unmatched.
  `ClockDomain::Host` is `std::chrono::system_clock`, the SDK's default host
  clock; recon never calls `setTimestampClockType`, which is process-wide.
- **The tolerance follows the frame rate.** `SensorArray::Options::tolerance_us`
  defaults to 0, meaning 0.4 of the fastest member's frame period
  (13 333 µs at 30 fps), or 5 000 µs when no member reports a rate. An
  explicit value keeps the old checks: non-zero, under half a period.
- **`TriggerGrouper` is the array's internals**
  (`src/volumetric_kit/recon/sensor/array/`, unexported, not installed). Its
  test compiles the source, as the Orbbec internal tests do.
- **What goes.** `OrbbecRig`, `OrbbecRigSet`, `OrbbecRigStats`, the rig's
  start order (`rig_start_order` and `recon_sensor_orbbec_start_order`; the
  array's order is tested by `recon_sensor_array`) and
  `orbbec::validate(OrbbecRig::Options)`; `CameraStream`'s rig-only
  `timestamp_us`, `withdraw` and `color_to_world`; and `discover`'s list of
  serials. The redundant `Pipeline::enableFrameSync()` goes too: the
  `Pipeline` constructor turns frame sync on. `OrbbecRigSyncConfig` and
  `read_orbbec_sync_config` stay, for the rig's file.
- **SDK 2.10.6 is the floor** in `cmake/vr_orbbec.cmake`, and CI's pin, now
  one composite action (`.github/actions/install-orbbec-sdk`) that
  `_build.yml` and `viewer.yml` share. The viewer leg installs FFmpeg and the
  SDK, so it compiles `rig_viewer` with the other viewers on every PR.

**Why: measured on the rig.** Four Femto Megas on the wired 2.5 GbE link,
each triggered by the primary through a sync hub, depth 640 x 576 at
30 fps; a secondary's timestamp minus the primary's for
the same trigger, from probes on the SDK outside recon:

| Clock | Skew | Over time |
|---|---|---|
| device, one `timerSyncWithHost` per camera | 1–2.6 ms at start | 3.1, 3.5 and 6.0 µs/s apart: medians 6.3–8.4 ms after 21 min; a 5 ms window fails after ~10 min |
| device, `timerSyncWithHost` every 60 s, as `OrbbecRig`'s `enableDeviceClockSync` did | about ±2.5 ms | bounded, but each sync lands on a new offset |
| global, SDK 2.9.3, one sync before start | medians 0.8–4.2 ms | bounded but biased; worst ~5.8 ms over 15 min |
| global, SDK 2.10.6, one sync before start | medians under 0.5 ms, typically −1.2 to +2.5 ms | flat over 15 min, while the same run's device timestamps drifted from 1.0–1.6 to 4.0–6.1 ms |

Rare single frames land 10–15 ms off, in the device timestamps as much as
the global ones, so they are not the mapping's. One past the window is let
go, costing that camera one set.

This is Orbbec's current recipe (SDK 2.10.6's
`3.advanced.multi_devices_sync` sample and `docs/tutorial/timestamp.md`):
global timestamps on every camera, one `timerSyncWithHost` each before any
stream, the secondaries started before the primary, frames matched on the
global timestamp within half a frame period, and no periodic re-sync. Its
5 ms is only the spread its `analyze_sync.py` flags as abnormal.

**Why not the alternatives.**
- **Device timestamps synced once** drift apart without bound: even
  13.3 ms is gone within about half an hour.
- **Periodic re-sync** steps every camera to a new random offset at each
  sync and blocks ~60 ms a camera. Once, in the 60 s run, the SDK's network
  enumerator reported two streaming cameras removed, and every later sync
  failed with "Device is deactivated/disconnected". Global timestamps bound
  the error without touching the camera's clock.
- **`Context::enableDeviceClockSync`** is process-wide, since the SDK's
  `Context` and `DeviceManager` are singletons: it re-syncs every camera the
  process opened, calib's included, belongs to whichever context called it
  last, returns before its first pass (`OrbbecRig::start` raced it), and
  holds the device-manager mutex through each pass, stalling every device
  open and close. Orbbec's sample replaced it with the per-camera call on
  2026-09-29.
- **The hardware timer reset** over the sync cable gives a rig clock, not
  the host's, so frames could not report `ClockDomain::Host`. Orbbec's staff
  call it exclusive with host sync on the Femto Mega, the SDK registers only
  its enable flag and signal there, and the clocks drift on after it.
  Grouping needs no sub-millisecond agreement: the trigger already aligns
  the exposures.
- **The grouper tracking each member's offset** would let device timestamps
  group, but leaves Orbbec frames off the host clock that `ClockDomain::Host`
  promises and calib needs, and adds per-member state to a vendor-neutral
  class for an unmeasured gain. It is worth a look only if global timestamps
  show a large, constant per-camera bias.

**Why 0.4 of a period.** The trigger exposes every camera once a period, the
0–480 µs depth delays aside, so a timestamp only says which trigger a frame
belongs to. Any window under half a period does that while the clock error
stays inside it: at 30 fps a frame within 13.3 ms of its trigger matches it,
and the neighbouring trigger's, 20 ms or more away, never does. The fixed
5 ms was tighter than anything Orbbec uses and refused 2.9.3's global worst
case.

**Not handled.** If a re-fit fails, the SDK keeps its previous mapping and
only logs it, so a global timestamp can be non-zero and off by the step
until the next re-fit; the array's zero check cannot see that. An
`OrbbecSDKConfig.xml` in the working directory can switch the SDK's host
clock to a monotonic one; recon does not read `getTimestampClockType`.

**Verified.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec
SDK 2.10.6 and FFmpeg: the 57 tests pass. `recon_sensor_array` groups by the
derived default with and without a rate (a secondary just inside it joins
the set, one just outside joins none), refuses an explicit 500 µs at 1 kHz
and accepts 499 µs; `recon_sensor_orbbec_conversion` refuses `apply_sync`
without `sync`. The public-open test
`recon_sensor_orbbec_sync_apply` uses a camera stub to check that SDK
normalization is refused (and fails without the post-write comparison),
matching settings are never written, a write repairs software triggering,
and a write error is preserved. Configure refuses an SDK below 2.10.6.
`rig_viewer` was not compiled here (the viewer is off locally); CI's viewer
leg compiles it.

On the rig, by hand (not a test), `fuse_orbbec --rig
config/femto_mega_sync.json`, H.265 720p at 30 fps, opened the four cameras
in 2.1–2.3 s and ran 19.8 min: 35 279 sets, 3.0% missing a camera. Each
secondary's mean skew to the primary was +0.13, +0.32 and +0.49 ms, its
160, 320 and 480 µs depth delay, flat over the run; the worst 2.8, 2.7 and
10.9 ms (one stray frame). The SDK reported no camera removed. The sets
missing a camera are H.265 colour lost to the decoder, 0.2–1% of each
camera's frames, through the same path as before. A sync file with one
secondary's depth delay changed was refused, naming the camera and the
field, and the camera kept its setting. Restarted three times with MJPEG,
1–3 of the first 31 sets missed a camera, so the start needs no settling
wait. One camera fused at 30 fps with and without `--host-clock`. Not run:
`apply_sync`'s write (it rewrites flash) and `rig_viewer`.

### 2026-10-08 — A decoded picture is its `YuvImage` with its pts and encoding: the one description of a device picture, which a frame takes as it is (amends the 2026-09-28 decoded-frame and 2026-10-02 GPU-regression entries).

**The rule.** `YuvImage` (`sensor/yuv_image.hpp`) is the one description of
a picture on the device: its planes and the queue family that wrote them, its
size, the matrix's weights, the range and the chroma siting. A decoder's
`DecodedPicture` is that `YuvImage` (`DecodedPicture::yuv`) with the pts and
the optional `ColorEncoding` beside it, and a frame takes it as its colour
unchanged. Whoever writes the picture describes it: `CudaPictures` and
nvJPEG give a CUDA buffer `kQueueFamilyExternal` (`place_cuda_buffer`),
`describe_color` an H.265 stream's weights, range and siting, and
`JpegDecoder::decode` JFIF's BT.601 full range and centred chroma, once for
both back ends. `VideoPixelLayout`, `DecodedPicture`'s own copies of the
fields and the Orbbec driver's `place_device_color` go. Two types had
described one picture, so each new source of decoded pictures had to map one
onto the other, and a source that missed the queue family or the matrix
would read CUDA's writes without taking them over, or fuse the wrong colour,
which `GpuFramePrep`'s checks cannot tell.

**Where it lives.** `yuv_image.hpp` is a header of `recon_sensor` that names
no camera, so `recon_sensor_video` describes its pictures with it without
linking the camera tier, as it already used `chroma_location.hpp`. A picture
carries the weights, which name any matrix. `VideoColorMatrix` and
`ycbcr_weights`, a named matrix's weights, live beside it, so a source that
is no decoder, such as an iOS camera whose buffer names its matrix, maps the
matrix with no header of the FFmpeg-gated decoders.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, Orbbec
and FFmpeg: the 58 tests pass with `VR_TEST_HEVC_BACKEND=videotoolbox`.
Without JFIF's description or `describe_color`'s weights, a test fails. The
decoder tests check the queue family each producer sets, so NVDEC's and
nvJPEG's `kQueueFamilyExternal` is CI's CUDA leg to check, as is the CUDA
code itself. Not run on a camera.

### 2026-10-08 — Apple's pixel-buffer importer is public: `PixelBufferImporter` in `recon_sensor_apple`, on every Apple build, iOS included, with no FFmpeg.

`PixelBufferImporter` (`sensor/apple/pixel_buffer_importer.hpp`, target
`recon_sensor_apple`), formerly the decoders' private `VtPictures`, is built
on every Apple platform, iOS included, and needs no FFmpeg, which it never
used, so an iOS camera can hand its colour (ARKit's `capturedImage`) to
`GpuFramePrep` on the device rather than convert it on the host. Its import,
surface cache and hold are as they were. It fills a `YuvImage`'s planes,
drops any buffer the image named, and leaves the colour description to its
caller, since a pixel buffer's format need not say how its samples are
coded.

- **In recon, not the family core.** What it fills is recon's `YuvImage`,
  and the iOS app reaches it through recon.
- **`VtJpeg` stays inside `JpegDecoder`**, its public face, and so behind
  `VR_WITH_FFMPEG`: on iOS it would reach a VideoToolbox key iOS has only
  from 17, above the app's floor of 16.
- **CI holds both promises.** The macOS leg, whose own build has FFmpeg,
  builds and runs the importer's test again without it, and compiles the
  importer for iOS 16 with exceptions enabled and `-Werror`.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors: the 59
tests pass with Orbbec and FFmpeg (`VR_TEST_HEVC_BACKEND=videotoolbox`), and
the 49 of a build with neither, `recon_sensor_apple_pixel_buffer_importer`
among them. It and the H.265 and JPEG decoder tests run clean under the
Khronos validation layer with synchronization validation; without the
import's drop of a stale buffer it fails. The importer compiles against the
iOS 16 SDK with `-Werror`, once it includes `IOSurfaceRef.h` (iOS has no
`IOSurface.h`). Not run on an iPhone.


**Rebase verification (2026-10-10).** All 71 tests pass on macOS Release
with synchronization validation, warnings as errors, FFmpeg, Orbbec SDK
2.10.6, Assimp and the viewers enabled. The importer uses the core GPU test
policy introduced in #208; the existing OOM and viewer teardown regressions
remain intact. The no-FFmpeg importer build and test pass under the same
validation policy, and its implementation and public header compile for
arm64 iOS 16 with the iOS 27 SDK. The CI compile check keeps exceptions
enabled. No camera, iPhone or interactive viewer was exercised.

### 2026-10-11 — Decoded pictures name the queue families that read them: VideoToolbox's images are CONCURRENT across two or more.

calib's live view draws each decoder's NV12 picture as a tile on gfx's queue
while recon reads the same images on its own. MoltenVK gives the two
libraries different queue families, and the importer made its images
EXCLUSIVE, so gfx's copy was undefined in the way that appears to work on
Apple. `PixelBufferImporter::create` now takes the families, and the decoders
and `OrbbecStreamOptions` pass them on.

- **Families, not a mode**, as for buffers (2026-08-03): a caller passes
  recon's and gfx's unconditionally, and they collapse to EXCLUSIVE where
  they are one family.
- **The device's own family must be among them.** The importer moves the
  images to `GENERAL` on it and the GPU pass copies them there; a list
  without it would make both undefined.
- **`OrbbecStreamOptions` holds them by value**, like
  `GpuFramePrepConfig::color_queue_families`, since each start opens the
  decoders afresh. The decoders read a pointer once, at `create`, as
  `BufferDesc` does.
- **NVDEC and nvJPEG wait** (`TODO(sensor)`): a Linux tile also needs gfx
  to map I420.
- **The importer repeats the core allocator's family checks for now.** They
  live in the core's internal `sharing_for`, and the importer creates its
  images itself, so it cannot reach them; a `TODO(sensor)` replaces the copy
  once the core makes them public and recon's core pin reaches that.

**Validation.** Apple M5 Max, MoltenVK, Release with warnings as errors,
FFmpeg 9.0.2 and Orbbec SDK 2.10.6, under the Khronos validation layer with
synchronization validation: the importer's test imports a picture on recon's
family of a windowless `SharedDevice` (`TwoFamilies`) and copies both planes
on the renderer's, bytes exact. The layer checks a CONCURRENT image's create
info (it reports a duplicate index or a single family) but does not track
which family owns an EXCLUSIVE image: made EXCLUSIVE, the same copy passes
with no message. So the guard is the sharing mode recorded on the image,
which the tests check. No camera was opened.

### 2026-10-11 — Picture readback publishes buffer ownership for subsequent stages

This amends the 2026-09-28 decoded-frame ownership contract:
`YuvImage::queue_family` names the current owner, initially the writer.
`PictureReadback::read` acquires a buffer from that family. Both overloads
now take mutable pictures/frames and update the field only after a successful
submit and wait: the readback's family for EXCLUSIVE buffers, and
`kQueueFamilyIgnored` for CONCURRENT buffers. Later stages use that updated
descriptor; copies made before the read still carry the old metadata.
`fuse_orbbec` passes the updated set to preparation.

The regression releases an EXCLUSIVE picture from one queue family to
another, then reads, prepares and reads it again through both overloads.
Before the fix, synchronization validation reports an acquire without a
matching release (`VUID-vkQueueSubmit-pSubmits-02207`). A rejected picture
keeps its original ownership metadata. The two-family case is reported as
skipped on devices that provide only one family.

Timer creation also belongs inside `PictureReadback::create`'s allocation
handler. A failure injected at a captured timer allocation size and call
site must return `OutOfMemory`, and creation must succeed on retry. Before
the fix, that regression catches an escaping `std::bad_alloc`.

### 2026-10-12 — The calibration file records lenses and is written: version 2 and its writer return with calib, its first caller (amends the 2026-10-06 camera-tier entries).

*Amended 2026-10-18 (the AprilTag-world entry, below):* the world may also
be a printed AprilTag, in OpenCV's marker axes.

*Amended 2026-10-27 (the depth-offset entry, below):* version 3 adds a
sensor's depth offset; the writer writes it only for a calibration that
carries one, and version 2 otherwise. *Amended again the same day (the
colour-controls entry, below):* version 3 also carries the colour cameras'
exposure, `color_controls`.

calib's rig app is the caller the 2026-10-06 cut-back waited for. Each
session records every camera's factory colour and depth models, unposed, and
each solve writes the posed rig with the lenses it solved with; both are
recon's file, written by recon, so the family has one schema and one writer.

- **What returns:** version 2 with `CameraCalibration` (the model and its
  `factory` or `calibrated` source), an optional pose, `depth_to_color` and a
  world sensor; `format_` and `write_array_calibration`;
  `rodrigues_from_rotation` and `rodrigues_from_matrix`.
- **What stays removed:** `CameraProjection`, `invertible_radius2`,
  `scale_camera_model` and `project_rational`. calib projects and unprojects
  with OpenCV, and the driver reports factory models at the streamed mode.
- **Version 2 keeps version 1's `device_calibration` key**, not 2026-10-06's
  `sensors`, so a version-1 reader (`verify_rig.py`, and recon before this)
  still finds the poses. `format` and `version` tell the two apart; a later
  version is `Unsupported`. Version 1 is still read for its poses alone.
- **Every field of a sensor is optional**, so a factory record is unposed;
  `SensorArray::open` refuses a sensor the file lists without a pose, as one
  it does not list.
- **The world is a sensor's colour camera only.** A floor AprilTag comes with
  calib's re-anchor, whose axes (z up out of the floor) are not
  libapriltag's; until then another world is `Unsupported`.
- **A written distortion names its model**, which the reader requires: the
  eight numbers mean nothing without it.
- **File I/O is the core's.** The writer takes the core's `WriteMode` (calib
  never overwrites a session file); the whole-file reads of this file, the
  Orbbec sync file and the Replica example's `cam_params.json` are
  `core::read_file`, so a missing file is `NotFound`; and `io`'s exception
  mapping is `core::exception_status`. recon needs core 0.2.0.
- **`rodrigues_from_rotation` is accurate to round-off at every angle**:
  `atan2` for the angle, first order only below 1e-8 rad, and past pi/2 the
  axis from the symmetric part. `cv::Rodrigues` returns exactly zero below
  about 1e-5 rad, so calib converts with this.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors, FFmpeg,
Orbbec SDK 2.10.6, Assimp and the viewer: the 92 tests pass under the
validation layer; the camera tier alone, with no Vulkan, builds and passes
with Apple Clang and GCC 16; and the camera, sensor-array and io tests pass
under ASan and UBSan. `rodrigues_from_rotation` comes back within 3e-16 of
the angle, relative, at 0, 1e-9, 1e-6, 1e-5, pi - 1e-6 and pi. Mutation
check: a `cv::Rodrigues`-style zero below 1e-5 rad, `acos` for the angle,
first order up to 1e-6 rad, the axis from `sin` past pi/2, the pose written
uninverted, six significant digits, the `sensors` key, a distortion without
its model, the write mode ignored, a later version accepted, and an array
taking an unposed sensor each fail a test. No camera was opened.

### 2026-10-16 — A shared GLSL pinhole for recon and its consumers: `core/shaders/pinhole.glsl`, exported under the include root `recon_core` names (amends the 2026-08-11 texturing entry's per-tier copies).

*Amended 2026-10-22 (the install entry):* recon installs nothing, so the
root is the build tree's copy alone.

*Amended 2026-10-28 (the camera-free volume entry):* volume's depth
allocation no longer goes through this file; it unprojects through
`volume/shaders/depth_pinhole.glsl`, a copy of `vr_unproject_pinhole`,
operation for operation, that `recon_volume_depth_unproject` holds to it
bit for bit.

calib's depth checks run kernels of their own over recon's prepared depth,
and they must project and unproject it by recon's rules. A copy in calib
would be a second pinhole. The core cannot hold it, since its decisions lock
"No camera or sensor tier". Nor can recon's `camera` tier, the family's
camera vocabulary: it builds without Vulkan (`VR_CAMERA_ONLY`), and `volume`
and `tsdf` do not depend on it, so they could not include it. So recon's
`core` tier holds it, beside `color_common.glsl`, and recon exports it. gfx
depends on no recon, so its planned point-cloud shader restates the in-image
test and the nearest-pixel rule on its own, and its tests must hold that copy
to recon's rule.

- **Five functions, taking values:** `vr_world_to_camera(cam_to_world,
  world)`, `vr_unproject_pinhole(px, z, k)`, `vr_pinhole_pixel(p_cam, k, out
  px)` (the divide, wherever the pixel lands), `vr_project_pinhole(p_cam, k,
  size, out px)` (the same within the image) and `vr_nearest_pixel(px,
  size)`, with `k` = (fx, fy, cx, cy) and `size` a `uvec2`. A struct would
  fix one buffer layout; values let recon's scalar-layout cameras and calib's
  std430 ones pass theirs.
- **Today's rules:** every test negated, so a NaN fails it; the domain
  `[0, W) × [0, H)`; the last half pixel clamped to the edge pixel
  (2026-10-08).
- **recon's own kernels go through it.** tsdf's `project_pinhole` and
  `nearest_pixel` are thin wrappers, volume's depth allocation unprojects
  with it, and texture's `world_to_camera` and `camera_to_image` call its
  rigid inverse and divide. Each keeps the order of operations its caller
  had, so none changes a bit. The sensor prep's colour test unprojects and
  divides through it too, a rounding-level change its tests pass unchanged.
  What stays each caller's is the acceptance: texture's projection
  extrapolates past the image (2026-08-11), and the colour test bounds the
  rounded pixel, `[-0.5, W - 0.5)`, as the colour pass marks coverage.
  `lens.glsl`'s normalization at z = 1 stays with the lens model it mirrors
  (`camera::distort_rational`).
- **Exported through a target property.** `recon_core`'s
  `VR_SHADER_INCLUDE_DIR` names the root a consumer passes its shader
  compiler, as the core's `VKC_CORE_VERSION` rides `core_base`, so it
  reaches a project that fetches recon and one that finds it. In the build
  tree the root is a copy of the exported files alone, so a kernel that
  compiles against a fetched recon cannot include recon's internal GLSL and
  then fail against an installed one. Installed, it is the include
  directory, which the package config sets on the imported target, found
  from the config's own directory: under CMake 3.29 and older a
  dependency's package config resets `PACKAGE_PREFIX_DIR` (Ubuntu's Assimp
  gave `/usr`).

**Validation.** Apple M5 Max, MoltenVK, Release with warnings as errors,
under the Khronos validation layer. `recon_core_pinhole_gpu` runs the five
functions in a std430 kernel against the host's double: 28 written-out
edges, 3 unprojections, 8 unbounded projections and 2 rigid poses in exact
arithmetic, a general rotation, and 2420 grid cases around five cameras,
from 1 × 1 to 3840 × 2160. `recon_subproject_glsl` and `recon_package_glsl`
compile that kernel through the fetched and the installed root, and
`recon_package_quiet_not_found` checks a refused package leaves no `_vr_`
variable behind. The tsdf, volume, mesh, texture and sensor suites pass,
with four cases added: `recon_tsdf_integrate` an 8 × 2 colour image, so its
nearest pixel's clamp is checked against the width; `recon_volume_allocate`
a camera that tells a swapped focal length or principal point apart;
`recon_sensor_gpu_frame_prep` colour principal points a fraction of a pixel
off, which pin the rounded bounds; and `recon_texture_projective` a vertex
past the left border, so the extrapolation holds on both sides.

### 2026-10-16 — A prepared frame's depth is shared like its colour, `GpuFramePrepConfig::depth_queue_families`; recon builds no point clouds and makes no calibration measurements; the same-K depth crop and nearest resampling are accepted on a stand-in lens, pending a fold check of the rig's own (amends the 2026-09-29 `rig_viewer` entry).

**Depth may be shared.** gfx is to draw points from each camera's prepared
depth in place, on its own queue family (GF5), for calib's 3D view of a rig
of any size. `GpuFramePrepConfig::depth_queue_families` names the families a
pass's depth is shared with, as `color_queue_families` does for colour and
independently of it. For either output, `create` refuses a count past
`kMaxQueueFamilies` or a family the device does not have, naming the output.
A missing family used to surface only at the first output, as core's
allocation error, and for colour not until a frame carried colour. The
buffer has no usage beyond the pass's storage and transfer usage, so a
consumer binds it as a storage buffer, not as vertices. Left empty, depth
stays EXCLUSIVE to the pass's family, as before; `rig_viewer`, which draws no
points, leaves it so. `SensorArray::Options::prep` gives every pass of an
array the same config.

**recon prepares RGB-D frames and stops there.** It builds no point clouds
and makes no calibration measurements: gfx is to draw points from depth and
a camera, and calib assembles the rig's views and runs its depth checks as its
own kernels. The point-cloud pass planned for recon is withdrawn.

**The depth crop and nearest resampling are accepted on a stand-in lens.**
The pass keeps the depth camera's K, so its pinhole image is the lens's view
cropped to that K and resampled at the nearest pixel: about 65 x 59 degrees
at the test's K (fx about 504 px), of the stand-in's raw 73 x 66. A K
reaching further would pass the lens's monotonic radius, where
`distort_rational` folds and two output pixels read one raw pixel.
`recon_sensor_depth_lens_fold` scans `distort_rational` forward along each
border pixel's ray. Its lens is a stand-in, not an SDK report: of
CL2A141000N's depth lens only k1 29.29 and k6 3.77 are on record (the
2026-09-28 GPU pre-processing entry). The other terms are calib's Femto-like
test lens, and the test holds the stand-in to the rest of the record. Its
rays reach its whole raw image, and undistorting moves a raw pixel 30.7 px
on average and 224.7 px at most, against the 31.5 px and 220.9 px measured
on CL2A141000N (the 2026-09-26 Orbbec driver entry). On it the widest output
pixel is at r = 0.914, and nothing folds within r = 3. Where a lens folds
depends on the terms not on record, so the crop holds for a rig only once
each camera's depth lens, as the SDK reports it, passes the same scan: a
hand check, not yet run.

**Verified.** Apple M5 Max, MoltenVK, Release with warnings as errors, core
`1315d80` (core #36): the 102 tests pass under the validation layer with
`VKC_REQUIRE_VULKAN_DEVICE=1`, and the sensor, texture and tsdf tests under
synchronization validation too. The changed tests also pass under ASan and
UBSan (Debug). `recon_sensor_gpu_frame_prep` crosses families on core's
`test::TwoFamilyDevice`: sets of 1, 3 and 5 frames prepared on recon's
family read the same on the renderer's. `recon_sensor_array_process` opens
arrays of 1, 3 and 5 sensors. `recon_sensor_apple_pixel_buffer_importer`'s
and `recon_sensor_picture_readback`'s cross-family cases move onto the same
fixture, and `test_image::make` fills through `test::write_image`. Planted
bugs each failed a test: depth's families ignored, depth given colour's,
colour given depth's, no check of the counts, no check that the device has
a family, colour's families checked as depth's, colour refused as depth,
vertex usage on the outputs, the array dropping its prep options, the fold
scan finding no fold, the guard missing the image's corners, a raw pixel no
ray reaches let through, and the stand-in replaced by a lens that folds
before its raw corners or one that moves raw pixels less than measured.

### 2026-10-18 — The calibration file's world may be a printed AprilTag, in OpenCV's marker axes, and the file stays version 2 (amends the 2026-10-12 calibration-file entry).

calib re-anchors its solved rig at a tag on the floor (calib's CA12b
plan). Its file could only leave `world` out, which records nothing, so
only calib's report would name the world. A world named in recon's file
keeps one home for what the world is.

- **OpenCV's marker axes, not libapriltag's.** calib finds the tag with
  OpenCV's detector and fits the frame to its corners in OpenCV's order. The
  re-anchor wants z up out of the floor. libapriltag's frame puts a floor
  tag's z down; the legacy detector used it, and so did the 2026-10-06
  draft (`d5dd23c6`). The draft's `{family, id, size_m}` is kept, but not
  its axes.
- **A sensor or a tag, never both, and no origin check for a tag.** A tag
  is not in the array, so it poses no sensor, and only its own fields are
  checked. A file that names both kinds has both read, so validate refuses
  it, for a file as for a struct.
- **The family is a free string, non-empty UTF-8.** recon detects nothing,
  so it keeps no list of families; the writer names one (`tag36h11`).
  `size_m` records what was printed, and nothing in recon reads it.
- **Still version 2.** Readers before this already refuse every world but
  a sensor as `Unsupported`, so they refuse a tag-world file rather than
  misread it. A file with a sensor world or none is unchanged byte for
  byte.

**Validation.** Apple M5 Max, macOS, Release with warnings as errors:
the default build's 89 tests pass under the validation layer, and so do
108 of 112 with the Orbbec SDK 2.10.6, FFmpeg, Assimp and the viewer. The
other four start `fuse_orbbec`, `fuse_render` and the viewers, and were
not run on that laptop; CI runs them. The camera tier alone, with no
Vulkan, builds and passes. Six planted bugs each failed a test: the
writer dropping the tag, a world with both kinds accepted, an id of -1
accepted, an infinite size accepted, the tag read but not kept, and a tag
world that needs a sensor at the origin. No camera was opened.

### 2026-10-25 — `HevcDecoder` decodes through VideoToolbox directly on Apple: no FFmpeg, the hardware decoder or none, pictures in display order by their order count, and `VR_WITH_FFMPEG` on Apple a configure error (amends the 2026-10-06 device-only decoder and 2026-10-08 pixel-buffer importer entries).

**The rule.** On Apple, `HevcDecoder` decodes through a
`VTDecompressionSession` of its own (`vt_hevc.cpp`) and hands its pictures
out through `PixelBufferImporter`, as before, with no copy. FFmpeg's
VideoToolbox path is deleted. The class, `DecodedPicture` and the codes are
unchanged, and so is Linux, NVDEC through FFmpeg. `recon_sensor_video`
builds on every Apple platform, as `recon_sensor_apple` does, and the
Orbbec driver builds on macOS without FFmpeg. Nothing keeps the decoders
off iOS now: they need no FFmpeg, and the floor is 26.0, above the iOS 17
that VideoToolbox's hardware key needs (the 2026-10-08 entry), so they
cross-compile to iOS unchanged, as every tier does (the 2026-08-01 entry).
`VR_WITH_FFMPEG` on Apple fails the configure and names the fix, so no
option is left that does nothing.

**Why.** A Mac decodes the Femto's H.265 with nothing but the platform:
nothing to install beside the Orbbec SDK, and one decoder from the
bitstream to the picture, `rig_viewer` and `fuse_orbbec` through the Orbbec
sensor as before.

**What the decoder reads itself** (`hevc_syntax.cpp`); VideoToolbox decodes
the slices:
- **The parameter sets.** Each SPS is read whole, to its trailing bits,
  before it can refuse the stream, so bytes that do not parse are
  `IoError` and latch nothing. It refuses what the device path cannot hand
  out: not 8-bit 4:2:0 (Main 10, 4:0:0), and a window off the coded corner,
  which VideoToolbox fills from the corner (measured on an M4: the
  left/top-cropped clip decodes to other pixels than libavcodec's). It
  crops the right and the bottom itself. A format is made from the
  sequence's VPS, SPS and PPSs; a changed set makes a new one, on the same
  session where `VTDecompressionSessionCanAcceptFormatDescription` allows
  (an M4 allows none of the test clips' SPS changes, the range alone
  included, so each takes a new session).
- **Display order.** VideoToolbox hands pictures back in decode order. The
  decoder computes each picture's order count (8.3.1) and holds pictures
  as the output process does (C.5.2): shown once more wait than the SPS
  reorders (`sps_max_num_reorder_pics`), and all at a sequence started
  afresh, unless it drops them, at an end of sequence and at the end. A
  stream with no B-frames, the Femto's, shows each picture as it decodes.
- **Where decoding starts.** The first key frame after `create` or
  `reset()` starts a sequence afresh; its leading pictures, predicted from
  before it, are skipped, as FFmpeg skips them, and a picture before any
  key frame is `IoError`. `reset()` drops the session, so that key frame
  decodes on a fresh one.
- **The range.** The session writes NV12 at the range the VUI declares,
  video range without one. Asked for the other range, VideoToolbox converts
  the samples (an M4 decoding the full-range clip as video range no longer
  matches libavcodec). `frame_color.cpp` describes them from H.273 codes,
  which the VUI and FFmpeg's frame fields share, one home for both paths.

**Hardware or nothing.** `create` returns `Unsupported` where VideoToolbox
has no hardware HEVC decoder (`VTIsHardwareDecodeSupported`), and the
session requires it
(`kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder`),
with no software fallback: a stream it cannot open is refused. iOS 26, the
floor, has the key, so there is no availability check. One internal home,
`videotoolbox.cpp`, opens the sessions of both decoders (NV12 on
Metal-compatible IOSurfaces, the range its caller asks for: the stream's,
or JPEG's full range), decodes a sample and maps a failure to its code.

**A skipped access unit is reported.** VideoToolbox reports a picture whose
reference is missing as bad data, so each picture predicted from a skipped
access unit is `IoError` up to the next key frame (on an M4: the patch clip
without frame 1 fails frames 2 and 3 and decodes from 4).

**Tests.** Three `testsrc2` clips at 128x64 with B-frames
(`tools/make_hevc_fixtures.sh`) decode on VideoToolbox to the MD5 of
libavcodec's own decode, in display order: one with no colour description
(video range), one full range, and the first cropped 8 at the right and the
bottom. NVDEC takes nothing under 144x144, so they are Apple's. Main 10 and
the left/top crop are refused; a skipped access unit is reported; one
decoder sent the patch clip, its full-range copy and two `testsrc2` clips
in turn hands out each sequence at its own size and range, the samples the
stream's own; the decoder, Orbbec and JPEG tests pass on the direct path.
`recon_sensor_video_hevc_syntax` reads every committed clip's syntax and
refuses every truncated SPS, on every leg, the sanitizers' included. On
Apple, a configure with `VR_WITH_FFMPEG` fails with the message
(`recon_subproject_apple_ffmpeg`). The macOS leg's iOS check compiles
`recon_sensor_video` and the importer for iOS 26.0 and links them against
the iOS SDK (`tools/ios_build_check.sh`), leaving unresolved only their
dependencies' symbols (the core's, recon's other tiers' and Vulkan's).
Mutating the range, the reordering, the leading-picture skip or the crop
refusal fails a test.

**CI.** The macOS legs, the viewer's included, build without FFmpeg and
need no `ffmpeg` formula; the NVIDIA leg is unchanged.

**Validation.** Apple M4 (Mac mini), macOS 27.0.1, Release with warnings
as errors, the Orbbec SDK 2.10.6 and no FFmpeg: the 115 tests pass under
the validation layer with `VR_TEST_HEVC_BACKEND=videotoolbox`, the threaded
ones again under ThreadSanitizer (Debug), the iOS check links (Xcode 27.0),
and the viewer leg's targets build and their 6 tests pass.
RTX 5090, Ubuntu, Release with FFmpeg 8.0 and CUDA 13.4: the 115 tests
pass on NVDEC (`VR_TEST_HEVC_BACKEND=cuda`), and the host-only video tests
under ASan and UBSan. Not run on a camera; for iOS, compiled and linked
only.

### 2026-10-25 — The H.265 syntax reader is public, in a target of its own on every platform (amends the 2026-10-25 VideoToolbox decoder entry).

**The rule.** The H.265 syntax `HevcDecoder` reads itself on Apple is a
public header, `sensor/video/hevc_syntax.hpp` (`vr::sensor::hevc`), in a
target of its own, `recon_sensor_video_syntax`: host code on the family
core's base tier alone, with no codec, GPU or FFmpeg, so it builds on every
platform. `recon_sensor_video` links it privately, and VideoToolbox's path
reads streams through it as before. `ColorCodes`, the H.273 codes an SPS's
VUI declares, moves from the internal `frame_color.hpp` to the public
`color_codes.hpp`, since `Sps::color` is one; FFmpeg's path describes its
frames from the same type, and no copy is left.

**Why public.** `HevcDecoder` asks its caller for every access unit and
speaks of key frames: a lost unit costs the pictures up to the next one,
and `reset` starts the stream afresh at one. Public, the reader gives that
caller the key frames (`split_annex_b`, `is_irap`) and what a stream's SPS
declares, from the code the decoder reads with, at the cost of no
dependency past the core's base tier. Its callers are `HevcDecoder` on
Apple and `recon_sensor_video_hevc_syntax`.

**Its contract is recon's.** The parsers return `Result` and
`split_annex_b` a `Status`: bytes that do not parse are `IoError`, naming
why (the decoder passes them on under its name, so its codes are
unchanged); a NAL unit of another type than the call reads, or null data
with a size, is `InvalidArgument`; a host allocation failure is
`OutOfMemory` (`core::exception_status`, built with `-fexceptions`), as in
recon's other host calls, so no exception leaves the reader. A `Nal` of
fewer than three bytes has no payload, where its size less the header would
wrap. `BitReader` is public with the parsers, since every one of them reads
through it: the RBSP's bits past the emulation prevention bytes, u(n)
(`VKC_CHECK` past 32 bits), ue(v), se(v) and the trailing bits, with zeros
and `overrun()` past the end.

**Fields.** The structs keep the fields the decoder and the tests read;
`Pps::dependent_slice_segments_enabled` and `kBlaWLp`, which nothing read,
go.

**Tests.** `recon_sensor_video_hevc_syntax` links the public target alone
and runs on every leg: the bit reader's own cases (an 03 skipped after two
zeros and kept after one, Exp-Golomb codes up to 31 leading zeros and the
refusal of 32, reads past the end, the trailing bits, a unit with no
payload); the clips' syntax as before; every truncated SPS in storage of its
own length; a slice without its PPS or its SPS; each wrong NAL unit type;
and noise behind each header, refused only as `IoError`; and each call's
largest allocation failed, returned as `OutOfMemory`. The decoder tests
are unchanged. The macOS leg's iOS check compiles and links the new target
with the decoders.

**Validation.** RTX 5090, Ubuntu 26.04, Release with warnings as errors,
under the validation layer: the 107 tests pass without FFmpeg or CUDA, and
the 112 with FFmpeg 8.0 and CUDA 13.4 on NVDEC
(`VR_TEST_HEVC_BACKEND=cuda`); the host-only video tests pass under ASan
and UBSan (Debug, GCC 15). Apple M4 (Mac mini), macOS 27.0.1, Release with
no FFmpeg: the 112 tests pass on VideoToolbox
(`VR_TEST_HEVC_BACKEND=videotoolbox`) under the validation layer, and the
iOS check links (Xcode 27.0). Mutating the reader -- an 03 skipped after
one zero or never, a code of 32 zeros taken, se(v)'s sign, the trailing
bits unchecked, a short unit's payload, the null-data and type checks --
fails the test.

### 2026-10-25 — Frame prep samples NV12 plane images where they are, in a second entry of the colour pass, and the plane copies go (amends the 2026-09-28 decoded-frame and 2026-10-08 pixel-buffer importer entries).

**The rule.** `GpuFramePrep` reads a picture's `YuvImage::image` planes in
place. The colour pass has two entries over one body
(`undistort_color.glsl`: the lens, the coverage test, the chroma siting, the
bilinear weights and the conversion): `undistort_color.comp` reads planes
in a buffer or packed host words, and `undistort_color_images.comp` reads
the two plane images through combined image samplers at integer coordinates
(`texelFetch`), making each sample its 8-bit code again with
`round(v * 255)`. Both entries therefore give the same words for the same
samples. The two plane copies `record_inputs` recorded go, with the pass's
`sensor.raw_color` buffer (12.4 MB a 4K camera) and two of a frame's timed
spans. `KernelSetBuilder` binds storage buffers only, so the image entry's
layout is built by hand: two combined image samplers and the output, its
set from a pool of its own, its sampler nearest and clamped, which
`texelFetch` never uses to filter. No core change.

**What a plane image needs.** Frame prep takes `SAMPLED` usage and a view,
in `GENERAL` or `SHADER_READ_ONLY_OPTIMAL`, of a 2D image of one layer and
one sample: the entry reads it as a `sampler2D`, which a 3D, arrayed or
multisampled image's view does not match, where the copy took the first
layer and refused multisampling. `PictureReadback` still copies
the images and takes `TRANSFER_SRC`. One check serves both, told which
reader asks (`detail::ImageRead`), and each refusal names what its reader
needs. `PixelBufferImporter`'s planes carry both usages and a view, and
their one barrier out of `UNDEFINED` makes them visible to compute reads as
well as copies. Planes in a buffer (NVDEC, nvJPEG) and packed host words are
read as before.

**Why.** The copy cost 0.28-0.31 ms of GPU a 4K picture on the M5 Max (the
2026-09-28 decoded-frame entry), every set, and a scratch buffer per camera.
Sampling in place nets 0.80 ms of device time a 4-camera 4K set on the M4,
back to back, though the pass reads images more slowly than the copied
buffer (PERF.md's L3 row has both arms, and paced sets).

**Tests.** Plane images larger than the picture, sampled in place in either
layout, prepare to the words the same samples give from buffer rows, bit for
bit (`recon_sensor_gpu_frame_prep`, every leg), and on a Mac so does a
patterned picture on a real IOSurface, its rows longer than the picture,
against its own bytes in a buffer
(`recon_sensor_apple_pixel_buffer_importer`). Images without `SAMPLED`
usage (a view present), without a view, in a copy's layout, or 3D, arrayed
or multisampled (either plane) are refused by frame prep, and a
sampled-only image by readback. The claim-before-grow case no longer has
the copy's buffer to catch a growth recorded before its claim; it catches
it through the outputs.

**Validation.** Apple M4 (Mac mini), macOS 27.0.1, Release with warnings
as errors, under the validation layer: the 112 tests pass on VideoToolbox,
the sensor tests under sync validation too, and the iOS check links; four
4K surfaces' imported planes, views included, allocate in Metal exactly what
bare textures do. RTX 4090, Release with warnings as errors: the 107 tests
pass on the NVIDIA driver and on lavapipe under the validation layer, and
the frame-prep, readback, array and set-codec tests under sync validation.
Mutants, on every leg: growing the buffers before the claim (#282's
mutation) fails the claim-before-grow case, the validation layer reporting
the unresolved batch's outputs destroyed; Cb and Cr swapped, the luma read
a column over and an image without a view taken each fail, and so does the
check without its `SAMPLED` condition, without any one of its 2D, layer and
sample conditions, or with either plane's shape unchecked. The luma left
unrounded (`v * 255`) passes on the M4 and the RTX 4090 and fails on
lavapipe, which is why the entry rounds.

### 2026-10-25 — A run's sets, the fusion grid and the block layout are library calls: `RigSets` in the array's tier, `create_fusion_grid` in `tsdf` and `block_grid_params` in `volume`; opening a rig stays the caller's.

**The rule.** Three pieces the examples kept in `examples/common` move to
the tier each depends on, and no copy is left:
- `sensor::RigSets(array, frames)` (`sensor/array/rig_sets.hpp`, in
  `recon_sensor_array`): `next(metrics)` waits for a started array's next
  set on its sensors' notifications (`SensorArray::wait_set`), keeps each
  secondary's skew to the primary (`skew(i)`), and cuts the set that reaches
  the run's N frames to its first frames in the array's order; `IoError`
  after `kSilenceLimit` (10 s) with no set, naming a lone sync secondary's
  cause.
- `tsdf::create_fusion_grid` and `fusion_buckets` (`tsdf/fusion_grid.hpp`):
  the grid a fusing caller starts from, carrying the fuser's float `tsdf`
  and `weight` and packed `color`, its start sized to the voxel and capped
  to the device, as the examples' grid was.
- `volume::block_grid_params` (`volume/voxel_grid.hpp`): 8³-voxel blocks in
  the default hash layout, as many blocks as bucket entries. Both grid calls
  build with it, `codec::make_decode_grid` too, in place of its hand-built
  copy, and a static assertion holds the codec's 8³ to it.

**Why the library.** Six examples built the same grid from an example
header, codec_mesh took its layout from another, and `make_decode_grid`
wrote that layout out again by hand; the run's count and cut were tested
only where the Orbbec driver builds. In the library each has one home and
its tests run on every leg. Each goes to the tier it needs and no further: `RigSets` takes a `SensorArray`, and drivers
never link the array, so it is vendor-neutral and not the Orbbec driver's;
the fusion grid needs `volume` alone, which `tsdf` links; the layout is
`VoxelGridParams`', in `volume`, where `codec` reaches it.

**What stays the caller's.** Opening a rig is already three public calls
(`open_orbbec_sensors` or `OrbbecSensor::open`, `read_array_calibration`,
`SensorArray::open`), so no open helper is added: one would live in the
driver, which may not link the array. fuse_orbbec's one-camera choice
(`posed_camera`), its camera lines, the first frame's age and the counters'
report (`print_rig_stats`) stay in `fuse_orbbec_sets.hpp`: the library
prints nothing. The prep stays the caller's (`SensorArray::record_process`),
recorded into the set's front batch with its fusion, so a coded set stays 2
submits. Nothing changes in a run: fuse_orbbec prints the same lines, and
every grid, and so every coded frame, is the one it was.

**Tests.** `recon_sensor_array_rig_sets`, on every leg: a run made without
throwing, its skews empty before a set and read as copies, so none is left
behind by a set; the run's count, cut and skew over pushed fake sensors
(moved from `recon_example_fuse_orbbec_sets`), the cut past a missing
sensor letting the cut frame's picture go, a set past the run's frames
keeping none, the silence limit counted from the last set and then the
`IoError` naming a lone sync secondary (a 10-s wait), and the array's
refusal of a sensor that notifies nothing passed on; `recon_example_fuse_orbbec_sets` keeps the one-camera
choice and the report, where the driver is built. `recon_tsdf_fusion_grid`
is `recon_example_fusion_buckets`, plus the grid's layout and attributes
and a NaN voxel refused with no undefined conversion on the way.
`recon_volume_coords` checks `block_grid_params`, up to the most buckets
whose block pointers fit `int32` and saturating past it with no overflow;
the decode-grid tests and `recon_example_set_codec` pass unchanged on the
library's grids.

**Validation.** Release with warnings as errors, under the validation
layer. RTX 4090, Ubuntu 26.04, GCC 15, with Orbbec, FFmpeg, CUDA and
Assimp and synchronization validation: the 123 tests pass, the 4 that load
the Orbbec SDK not run and the JPEG test skipping on a GPU with no JPEG
engine. RTX 5090, with FFmpeg, CUDA and Assimp and synchronization
validation: the 118 pass. Apple M4 (Mac mini), macOS 27.0.1, with Orbbec,
Assimp and the viewers: the 133 pass on VideoToolbox, the three viewers
build, and the iOS check links. The changed tiers' tests pass under ASan
and UBSan (Debug, clang 21) on both GPUs. Mutating the moved code -- no
cut, a cut to none skipped, a cut set's frames uncounted, the primary's
skew kept, a skew read by reference, the silence limit not reset by a set,
a lone sync secondary's cause left out, fewer blocks than bucket entries,
another chain length, a fusion grid without `color` or with another band, a
NaN voxel's start passed to the integer conversion, the block count's int32
product -- fails a test or the build.

### 2026-10-26 — NVDEC's pictures land in two plane images, VideoToolbox's form: CUDA's one copy writes them, a reader takes each over from outside Vulkan, and nothing hands it back (amends the 2026-10-08 decoded-picture and 2026-10-25 plane-image entries).

**The rule.** `HevcDecoder`'s NVDEC path copies each picture, with the one
CUDA copy it always made, into an R8 luma image and an R8G8 chroma image of
the picture's size, made with `core::create_exported_image` (core 0.24.0)
and mapped into CUDA once (`CudaImageMapping`, public beside
`CudaBufferMapping`), and hands them out as `YuvImage::image` with
`kQueueFamilyExternal`: the form VideoToolbox's pictures have, so every
platform's HEVC picture is NV12's two plane images. Its NV12 buffer goes,
with no switch back. nvJPEG keeps its I420 buffer: it writes only linear
memory. `GpuFramePrep` samples the images through the entry Apple's
pictures already take, and `PictureReadback` copies them.

**The crossing.** A new set of images is handed over to CUDA once, in
`GENERAL`, by a batch of the decoder's own before CUDA's first write: the
core refuses to take back contents still `UNDEFINED`. Each picture's reader
then takes the images over from `VK_QUEUE_FAMILY_EXTERNAL` in the layout
they are in, as it took the buffer: `YuvImage::queue_family` covers images
now. A readback names them its family's after, or `kQueueFamilyIgnored`
where they are CONCURRENT, so a later stage does not take them again;
images that needed no transfer keep `kQueueFamilyIgnored`, as Apple's do.
Nothing hands them back to CUDA, as nothing handed the buffer back: CUDA
overwrites the whole picture, the layout stays `GENERAL`, and Vulkan only
reads them, so nothing need survive the crossing. A set is reused only once
no picture holds either image: the two are aliases of one owner, so one
count says. The data flow is unchanged: no submit or wait is added for a
picture, and a new set's hand-over is one submit, once. An EXCLUSIVE image
crosses only from outside Vulkan (core), so `GpuFramePrep` refuses one named
as another family's with the frame's checks; for readers on other families
the images take `HevcDecoder::Options::queue_families`, which NVDEC ignored,
by `PixelBufferImporter`'s rule, now one check for both
(`sensor/sharing_families.hpp`).

**Why.** One picture form on every platform: frame prep and readback read
NVDEC's planes through the image path they use for VideoToolbox's, and a
viewer of the cameras gets image planes on Linux too, where a buffer would
cost it a Vulkan copy into images every picture (calib's live mosaic copies
a picture's planes only from images and refuses planes in a buffer).
Measured on the lab RTX 4090 (driver 615.71.10), Release, a 3840x2160
picture from a device surface with padded rows, CUDA events, the median of
each of three interleaved rounds of 400 copies: the copy into the two images
takes 0.030 ms, into the buffer 0.016 ms. Frame prep of one 4K camera with a
640x576 depth frame, the same picture, device time, medians of 600 prepares
over six interleaved rounds, other processes keeping the GPU busy
throughout: 0.100 ms from the images, 0.107 ms from the buffer, with words
bit for bit the same. One multi-planar image with plane views was the
alternative; it needs multi-planar formats in the core, and two images
need none.

**Tests.** `recon_sensor_video_hevc`, on NVDEC and VideoToolbox both: the
patch clip and its full-range copy decode to libavcodec's MD5 through
readback, and on NVDEC the cropped clip too (the testsrc2 clips stay
VideoToolbox's, under NVDEC's smallest picture); each picture is two images,
on NVDEC of its size in `GENERAL` with `kQueueFamilyExternal`; a held
picture's images are its own and keep its samples, and on NVDEC a dropped
one's are reused; frame prep from each decoded picture gives, bit for bit,
the words the same samples give from NV12 planes in a buffer; a readback
names NVDEC's images its family's; and the decoder's queue families make
them CONCURRENT, a readback leaving those `kQueueFamilyIgnored`, and a list
the importer would refuse fails `create`. `recon_sensor_video_cuda_interop`:
R8 and R8G8 images of odd size, and an R8G8B8A8 one mapped as the colour
attachment its usage makes it, which CUDA writes, read back with its bytes
three rounds over after one hand-over, never handed back; a taken
descriptor, an empty image and a 16-bit format are refused.
`recon_sensor_gpu_frame_prep`: plane images taken over from outside Vulkan
or named as the pass's family give the same words, in either layout; another
family's EXCLUSIVE images and a family the device lacks are refused.
`recon_sensor_picture_readback`: the same for its copies, each family's
naming after. `recon_sensor_orbbec_hevc`: the driver's pictures are images
on both platforms, CONCURRENT with the families it is given.

**Validation.** Release with warnings as errors, under the validation layer
and synchronization validation. RTX 4090, Ubuntu 26.04, with Orbbec,
FFmpeg, CUDA and Assimp: the 129 tests pass, the 2 JPEG tests skipping with
no JPEG engine. RTX 5090, Ubuntu 26.04, GCC 15.2, the same options: the 129
tests pass, the JPEG tests on its JPEG engine; Debug with clang 21 under
ASan and UBSan, NVDEC and nvJPEG included, the 14 sensor tests the change
touches pass, the only leaks reported the driver's. Apple M4 (Mac mini), macOS
27.0.1, with Orbbec and Assimp, on VideoToolbox: the 129 tests pass, the
sensor tests under synchronization validation too, and the iOS check links.
Mutants on the RTX 4090: a set reused while a picture holds it, a readback
that leaves NVDEC's images named as outside Vulkan's, frame prep without its
EXCLUSIVE-image check, and a set never handed over to CUDA each fail a test.
The layer cannot see CUDA's writes, so it passes with a reader's acquire
left out, and with the hand-over left out where the images are still
recorded as in `GENERAL`: both rest on Vulkan's rule for external memory, as
the buffer's acquire did. A colour attachment mapped without CUDA's flag
also passes on the RTX 5090, which lays it out alike; the flag rests on
CUDA's documented rule.

### 2026-10-27 — A sensor's depth offset lives in the calibration file, version 3, and frame prep subtracts it, the calibration its one source (amends the 2026-10-12 calibration-file entry).

*Amended the same day (the colour-controls entry, below):* the colour
cameras' exposure, `ArrayCalibration::color_controls`, joins version 3 the
same way, rig-wide.

A depth sensor can read a few millimetres too far or too near along its
axis, a constant of the unit, so its fused surface sits off the other
sensors' and off its own colour. The offset is measured with the poses, so
it belongs in the rig's calibration beside them, and it is applied where
depth becomes metres, so every reader of prepared depth sees the same
corrected value.

- **`SensorCalibration::depth_offset_m`**, in metres, + where the depth
  reads too far. Optional, 0 when absent; finite and at most 0.05 m either
  way, since a sensor's error is millimetres and more is a unit or sign
  mistake. A double, as every other number of the file, so the file shows
  what its writer computed and reads it back exactly; the array narrows it
  to float once for the pass, as the pass narrows the models.
- **Version 3, only for a corrected calibration.** A version-2 reader
  ignores keys it does not know, so a corrected file written as version 2
  would run uncorrected there; written as version 3 it is refused
  (`Unsupported`), refusing rather than misreading as the 2026-10-18 entry
  does. A calibration without a correction is written as version 2, byte
  for byte as before, so nothing that needs no correction changes. The
  reader takes versions 2 and 3, and refuses a `depth_offset_m` in a file
  that says version 2, even a 0: a file edited by hand would otherwise run
  uncorrected on an older reader. A later correction joins version 3 the
  same way.
- **Frame prep subtracts it in the depth pass it already runs:**
  `z = raw * scale`, then for `z > 0`, `z = max(z - offset, 0)`. 0 stays no
  return whatever the sign, so a negative offset never makes a hole a
  surface, and a return the offset takes to 0 or below is no return. It
  runs before the within-colour mask, which tests the corrected point.
  `GpuFramePrepConfig::depth_offset_m` carries it; the depth push constants
  grow from 64 to 68 bytes, and `GpuFramePrep::create` refuses a non-finite
  offset, whose `max` the shader leaves undefined. No pass, buffer, copy or
  host step is added.
- **The calibration is the one source.** `SensorArray::open` copies
  `Options::prep` for each sensor and sets the copy's offset from the
  calibration it already reads for the poses. An offset in `Options::prep`
  is refused: it would be a second source, and one for every sensor alike.
  Callers without a calibration (datasets, an array opened to measure raw
  depth) keep raw depth, and a frame's `RgbdFrame::depth` stays raw.
- **Who sees it:** every reader of the prepared depth: fusion, the
  texturer's occlusion test and a renderer drawing the depth in place.

**Tests.** `recon_camera_array_calibration` reads a version-3 file (+4.17
and -0.96 mm, and a sensor with none), takes the bound at ±0.05 m and
refuses past it, a non-number, an offset in a version-2 file (even 0) and
version 4; validate and the writer refuse NaN, ±infinity and an offset past
the bound; a bare sensor is written byte for byte as before, -0.0 and 0
alike, and with an offset as version 3; and every field of every sensor
round-trips, the text being the uncorrected one's but for the version and
the offsets' lines. `recon_sensor_gpu_frame_prep` checks depth to be
exactly `raw * scale - offset` for +4.17 mm, -0.96 mm and an offset equal to
a return (a power-of-two scale, so whether the compiler fuses the multiply
and the subtract does not matter), 0 where the offset leaves no return and
a hole a hole for either sign; the within-colour mask keeping exactly the
pixels whose corrected point the colour camera records, where a mask on
the raw depth would get a column or two wrong at each side of every row;
and a non-finite offset refused at create. `recon_sensor_array_process`
gives three sensors offsets of 0, +4 and -2 mm: each prepared frame equals
a pass made alone with its sensor's offset, and differs from one with none
exactly where the offset is not 0; an offset in `Options::prep` is refused,
with a device and without, NaN included.

**Validation.** Release with warnings as errors, under the validation
layer. RTX 4090, Ubuntu 26.04.1, GCC 15.2, CUDA 13.4, with the Orbbec SDK
2.10.6, FFmpeg and Assimp, and synchronization validation: 127 of the 129
tests pass, and the two JPEG tests skip, with no JPEG engine; the camera
tier alone, with no Vulkan, builds and passes. Apple M4 (Mac mini), macOS
27.2, with Assimp and the viewers (no Orbbec SDK there): the 125 tests
pass, and the three changed ones under synchronization validation too.
Thirteen planted bugs each failed a test: the offset added, not
subtracted; a hole made a return by a negative offset; the offset
subtracted after the within-colour mask; the pass ignoring its config's
offset; a NaN offset taken at create; the writer always writing version 2;
a 0 offset written; an offset read from a version-2 file; version 3
refused; the 0.05 m bound unchecked; and the array dropping each sensor's
offset, giving it a neighbour's, or taking one from its options. No camera
was opened.

### 2026-10-27 — The Orbbec driver defaults to H.265 at 3840x2160 and 25 fps, and opens each colour camera at the calibration's exposure: written where the camera differs, auto white balance on, and read back; an array refuses a sensor opened at another (amends the 2026-09-28 H.265 entry and the depth-offset entry above).

Reconstruction textures from each camera's colour at the camera's own size,
so the rig streams the colour cameras' 4K mode, in the codec the GPU
decoders take at the least bandwidth. A rig's cameras under auto exposure
each meter their own view, so one surface comes out brighter in one camera
than in another, and anything fitted to the colour at one exposure does not
hold at the next; so every camera of a calibrated rig runs at one manual
exposure, recorded with the calibration.

- **H.265 at 3840x2160 and 25 fps is the driver's default**
  (`OrbbecStreamOptions`), depth 640x576 as before: the streams core
  0.25.0's Orbbec scenarios default to, so the driver and the scenarios
  agree. The Femto Mega offers 3840x2160 at 25 fps at most, and both
  streams run at one rate, so a set has 40 ms. H.265 takes about 21 Mbit/s
  per camera at 4K against MJPEG's 185, and decodes to the camera's MJPEG
  colour (the 2026-09-28 entry). `Mjpeg` stays selectable: for calibration
  captures, whose frames are each coded on their own at nine times the
  rate, and for diagnostics such as fuse_orbbec's set dump. The 1280x720
  mode stays selectable and uncorrected: its content sits 1.17 px lower
  than its profile's `cy` (the 4K value / 3) says, about 1.6 mrad, measured
  against 4K frames on MJPEG. Recorded here; nothing defaults to it.
- **`camera::ColorControls`**, an exposure time and a gain in the camera's
  units, both above 0, is a manual exposure: auto exposure off, auto white
  balance on. **`ArrayCalibration::color_controls`** holds the one for every
  camera of the array, a version-3 field beside the world, written only
  when present, as the depth offset is; empty is auto exposure. No value is
  a public default: the right exposure depends on the room's light and its
  mains frequency, so the rig's calibration names it.
- **Auto white balance stays on.** Switched off, the camera returns to its
  stored temperature (6500 K as shipped) within 0.8 s (measured), a warm
  cast under a room's warmer light; a fixed white balance would need a
  temperature chosen per room. Auto white balance off and the temperature
  are never written.
- **Written at every open** (`OrbbecStreamOptions::color_controls`, by
  `OrbbecSensor::open` after the sync and role checks), unlike the sync
  settings, which are written only when asked: a camera keeps the last
  exposure written to it, in another process too (measured), so a camera
  left as it was would keep an earlier run's. Auto exposure is written the
  same way. Only what differs is written: auto exposure off, then the
  exposure and the gain, then auto white balance on; or auto exposure on,
  the stored values untouched. A camera under auto exposure reads its stored
  values, not the ones it runs at (measured), so one switched from auto
  exposure has both values written.
- **Checked before the first write, read back after the last.** A manual
  exposure needs auto exposure, exposure, gain and auto white balance to
  read and write, and both values within the camera's ranges and on their
  steps, or the open is refused (`Unsupported`, the control named) with
  nothing written; auto
  exposure leaves a control the camera lacks alone, so another model still
  opens. The flags, and under a manual exposure both values, are read back,
  and a camera that differs is refused naming each field, as the sync check
  is. `SensorInfo::color_controls` holds the manual exposure as read back,
  empty under auto exposure. Reading five colour properties takes a median
  6.1 ms per camera, at most 13.8 ms (measured), so this adds tens of
  milliseconds once per camera to an open that takes seconds, and nothing
  per set.
- **An array checks it** (`SensorArray::open`): with a calibration that
  names colour controls, a sensor whose `SensorInfo::color_controls` differ,
  as one opened without them, is refused rather than coloured at another
  exposure, whether the calibration lists sensors or not. A calibration
  without them checks nothing.
- **One place sets them.** The examples' `open_orbbec_cameras` gives every
  camera its calibration's controls; `rig_viewer` opens through it too, so
  it has no open of its own. A capture with no calibration runs at auto
  exposure, written.

**Tests.** `recon_sensor_orbbec_color_controls` opens a stub camera whose
colour properties go through the driver's own sequence: a camera that
matches gets no write, in either mode; a manual exposure over auto exposure
writes auto exposure off, the exposure, the gain (though it reads the same
already) and auto white balance on, in that order, and over a manual one
only the value that differs; auto exposure over a stored manual exposure
writes auto exposure on alone, neither value written nor compared; auto
white balance off is switched on in either mode, and the temperature is
never written; an exposure or a gain past the camera's range or off its
steps, or a control the camera lacks, is refused with nothing written, while
auto exposure opens a camera that lacks any of them and touches none of
those; a write the SDK fails is `IoError` with its message; a gain or a flag
read back otherwise is refused naming it; and a camera its sync check
refuses gets no colour write. `recon_sensor_orbbec_open_sensors` opens every
camera of a rig at the options' controls. `recon_camera_array_calibration`
reads the controls, refuses them in a version-2 file, at 0, as a non-number
and as a non-object, writes them byte for byte, version 3 only with them,
and round-trips them beside depth offsets, the text otherwise unchanged.
`recon_sensor_orbbec_conversion` refuses a 0 before the camera is looked
for; `recon_sensor_array` refuses a sensor at auto exposure or at another
gain under a calibration's controls, naming both, listed sensors or not, and
checks nothing without them; `recon_example_config` maps a live scenario
that names no colour mode to exactly the driver's defaults;
`recon_example_fuse_orbbec_sets` checks the report's exposure.

**Validation.** Release with warnings as errors, under the validation layer.
RTX 4090, Ubuntu 26.04.1, GCC 15.2, CUDA 13.4, with the Orbbec SDK 2.10.6,
FFmpeg and Assimp, and synchronization validation: 128 of the 130 tests
pass, and the two JPEG tests skip, with no JPEG engine; the camera tier
alone builds and passes. Apple M4 (Mac mini), macOS 27.2, with the Orbbec
SDK 2.10.6, the viewers and Assimp: the 137 tests pass. Of twenty planted
bugs, nineteen each failed a test: the range check, the control check, the
values written only where they differ or always, auto white balance
following the mode, auto exposure touching a control the camera lacks, the
gain or a flag not read back, a 0 taken, the controls written before the
sync check or not reported, the writer's version and its controls line, the
reader taking the controls from a version-2 file, validate taking a 0, the
array not checking them, the driver defaulting to MJPEG or 30 fps, and the
report leaving out the exposure. The twentieth, `open_orbbec_cameras`
dropping the controls, no host test catches, since opening cameras needs a
device; on a rig, `SensorArray::open` then refuses every camera. Three more
each failed a test: the array checking the controls only where the
calibration lists sensors, the step check dropped, and a rig's cameras
opened without the options' controls. No camera was opened.

### 2026-10-28 — `io` is two targets: images on the family core's base tier alone, mesh files on `mesh` (amends the 2026-10-03 asset I/O entry).

**The rule.** `recon_image_io` holds image decoding and PNG encoding
(`io/image_io.hpp`, with stb's implementation private) and links
`volumetric_kit::core_base` alone: no Vulkan and no recon tier.
`recon_mesh_io` holds the mesh files: PLY export (`io/ply_writer.hpp`), the
host triangle containers (`io/triangle_mesh.hpp`) and the opt-in
normalization helpers (`io/mesh_normalization.hpp`), with tinyply's
implementation private. It links `recon_mesh` PUBLIC, because `write_ply`
takes a `mesh::Mesh`. `recon_mesh_io_assimp` replaces `recon_io_assimp` on
top of it (`io/mesh_io.hpp`, `io/textured_mesh.hpp`). The headers stay in
`io/` and the namespace stays `vr::io`. Each target's export header and
macro are named after it (`image_io_export.hpp` and `VR_IMAGE_IO_API`, and
likewise for the other two). No `recon_io` target remains, and no alias.

**Why.** `recon_io` linked `recon_mesh` PUBLIC for `write_ply` alone. So
every image user pulled in `volume`, `recon_core` and Vulkan: the Replica
sensor, and a calibration capture that only writes PNGs. The two halves also
have different users, images on the capture side and mesh files on the mesh
and codec side. With separate targets, each consumer names the half it uses
once; the files can move later without a second rename. The 2026-10-03
entry's "depends on `mesh`" now holds for `recon_mesh_io` alone.

**Tests.** `recon_io_image` links `recon_image_io` alone, and
`recon_subproject_defaults` fails when `recon_image_io` links anything but
the core's base tier. `recon_io_ply` and `recon_io_mesh_normalization` link
`recon_mesh_io`; `recon_io_mesh_file` and `recon_io_textured_mesh` link
`recon_mesh_io_assimp`, the latter with `recon_image_io` for the PNG it
writes. The examples link the half they call: `vr_example_common` links
images; `fuse_replica`, `codec_replica` and `fuse_orbbec` link mesh files
for their PLY.

**Validation.** Apple M5 Max, macOS 27.2, Release with warnings as errors
and Assimp 6.0.5: the build is warning-clean, and the 23 affected tests
(`io`, the examples that read images or write PLY, the subproject consumer)
pass under the validation layer with a device required.
`recon_io_image_test` links `librecon_image_io.a` and the core's base tier,
nothing else. Adding `recon_mesh` to `recon_image_io`'s links fails
`recon_subproject_defaults`.

### 2026-10-29 — The camera and sensor tiers are the package `volumetric_kit_sensor`, in recon's `sensor/` until it has a repository of its own, and recon's libraries link its frame tier alone (amends the 2026-08-02 sensor-tier, 2026-10-06 camera-tier, 2026-10-16 GLSL pinhole and 2026-10-28 io entries).

**The owner's decisions (2026-10-28).** On the recon split: "I approve all
decisions. However, for D2, we should split now." The sensor and grid
packages each become a repository of their own now, by way of an in-repo
package that builds alone and is then extracted with its history, the
paths before the move included: the extraction keeps the moved files' old
paths as well as `sensor/` and renames them, since `sensor/` alone starts
at the move. The names were approved as
proposed (the sensor design's D2): package `volumetric_kit_sensor`,
namespace `volumetric_kit::sensor` with `sensor::camera` for the
vocabulary, targets `sensor_*`, options and macros `VS_`. A future stereo
depth producer is NVIDIA-only at first (D3).

**The rule.**
- `sensor/` is a CMake project of its own, `volumetric_kit_sensor` 0.1.0,
  on volumetric_kit_core alone (pinned by commit, 0.29.0 or newer). Its
  tiers are `sensor_camera` (no Vulkan), `sensor_frame`,
  `sensor_interface`, `sensor_prep`, `sensor_array`, `sensor_video` and
  `sensor_video_syntax`, `sensor_apple`, `sensor_orbbec` and
  `sensor_image_io`; the umbrella `volumetric_kit::sensor` carries
  `VKC_PACKAGE_VERSION`.
- `sensor_frame` is the contract recon reads: `DeviceFrame` and
  `DeviceFrameSet` (`frame/device_frame.hpp`, a header of their own now),
  `DepthCameraParams` and `ColorCameraParams`, `ColorEncoding` and the sRGB
  curve, and `pinhole.glsl`, exported with `vkc_export_glsl`.
  `recon_tsdf`, `recon_texture` and `recon_sender` link it, and no recon
  library links another sensor tier. recon's examples link whichever tiers
  they assemble. recon exports no GLSL of its own now.
- Nothing under `sensor/` includes recon, or grid, its sibling on the core.
- recon declares the core, then the package, by `FetchContent_Declare` with
  `SOURCE_DIR ${PROJECT_SOURCE_DIR}/sensor`, then grid, and checks the
  package with `vkc_require_sibling`. An application declares it first, by
  URL and commit with `SOURCE_SUBDIR sensor`, and its pin wins. In recon's
  build the package's tests and `-Werror` follow recon's
  (`VS_BUILD_TESTS`, `VS_WARNINGS_AS_ERRORS` set from the `VR_` ones), as
  grid's did while it was in-tree, and it builds in `sensor/` under the
  build tree.
- Each package has its own device requirements, GLM aliases and version.
  `sensor::device_requirements()` names the decoders' optional extensions
  (`VK_KHR_external_memory_fd`, `VK_EXT_metal_objects`); recon's keeps
  `scalarBlockLayout`. An application opening one device for both merges
  the two lists with `core::merge`.
- Every name moved, with no alias and no forwarding header;
  [CHANGELOG.md](CHANGELOG.md)'s [0.1.0] maps each old name to its new one.
  recon's `VR_CAMERA_ONLY` is gone: `VS_CAMERA_ONLY` builds `sensor_camera`
  alone.

**Why.** A camera stack is anything that produces `DeviceFrame`s, so a new
one, such as a future stereo depth producer, is a new producer and no recon
change. The contract already existed; only its home and namespace change.
The move is one PR rather than tier by tier: halfway,
`volumetric_kit::sensor` and `volumetric_kit::recon::sensor` would have
stood side by side, breaking every unqualified `sensor::` lookup, and the
package could not build alone until the last step.

**Tests.** `sensor_package_boundary` fails on a file under `sensor/` that
includes `volumetric_kit/recon/` or `volumetric_kit/grid/`, and
`recon_library_sensor_links` on a recon library that links a sensor tier
other than `sensor_frame`, a generator expression around it included; each
was seen to fail by mutation. The package's subproject tests configure it as a
consumer would: its defaults, an old floor, a GLSL consumer, an old core, a
core without Vulkan, camera-only, and FFmpeg on Apple refused. The
frame-prep test's fusion half is recon's `recon_tsdf_prepared_frame`.

**Validation.** Release, validation layer on. Apple M5 Max: recon with the
package and grid fetched 83/83 (the package's 43 among them), the package
alone 43/43, camera-only with Vulkan disabled 6/6. RTX 4090 with Orbbec,
FFmpeg and CUDA: the package alone 41 passed and 2 skipped (the two JPEG
skips it had in recon), camera-only 6/6.
