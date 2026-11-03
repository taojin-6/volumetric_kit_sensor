// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/hevc_encoder.hpp
/// @brief An H.265/HEVC encoder on the GPU's hardware, the twin of
///        @ref HevcDecoder: the picture it codes next is handed out on the
///        device for the caller to write, and each picture sent comes back
///        as one Annex B access unit, in the order sent.
///
/// No FFmpeg, VideoToolbox or Core Video type appears here. Apple's other
/// way in, a `CVPixelBuffer` sent as it is, is video/hevc_encoder_apple.hpp.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/device_requirements.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/core/vulkan/sync.hpp"  // core::TimelinePoint
#include "volumetric_kit/sensor/video/color_codes.hpp"
#include "volumetric_kit/sensor/video/export.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor {

/// @return What a device needs to write @ref HevcEncoder::target's planes
///         from a compute shader: `device_requirements()`, plus
///         `shaderStorageImageExtendedFormats` for R8 and R8G8 storage
///         images (`r8` and `rg8` in GLSL).
VS_VIDEO_API core::DeviceRequirements hevc_encoder_device_requirements();

/// @brief Codes 8-bit 4:2:0 pictures as an H.265 Main stream (Annex B) on
///        the hardware of this build's backend (@ref video_backend):
///        VideoToolbox on Apple. Hardware only: @ref create says what is
///        missing rather than falling back. A build with no encoder (Linux,
///        until NVENC lands) refuses in @ref create.
///
/// One configuration: low latency (P-pictures only, no reordering, no
/// look-ahead, every NAL unit in temporal layer 0); @ref Options::qp as
/// every P-picture's slice QP and @ref Options::key_qp as every IDR's,
/// fixed for the session (the hardware may still vary QP inside a slice);
/// keys only when asked; the VUI from @ref Facts::color.
///
/// Pipelined, as @ref HevcDecoder is: a send hands over a picture and
/// returns at once, and @ref receive hands back each access unit in the
/// order sent. A coding thread of the encoder's own waits on the host for
/// each picture's writer, then codes it. The hardware holds a thread for the
/// whole encode in this configuration (VideoToolbox's low-latency mode codes
/// inside its call), and this keeps that thread off the caller's. On Apple
/// it runs at user-interactive quality of service, as a camera's delivery
/// does. Encoders on separate threads code side by side where the hardware
/// has the engines.
///
/// Two ways in:
/// - @ref target hands out a free picture on the device and records into
///   the caller's batch what its planes need; the caller writes them in that
///   batch (a compute shader of its own), submits it, and sends (@ref send)
///   the timeline value the batch sets as it completes. Neither host waits
///   for the device, and the picture never crosses to the host.
/// - On Apple, send_pixel_buffer (video/hevc_encoder_apple.hpp) sends a
///   `CVPixelBuffer` as it is, such as a camera's, with or without a device.
///
/// The first picture is a key, and so is the next after one that failed or
/// was dropped (@ref key_owed): a P-picture after a gap does not decode. A
/// failure other than a drop also loses the session, which the next picture
/// opens again.
///
/// @code
/// HevcEncoder::Options o;
/// o.width = 1920;
/// o.height = 1440;
/// o.qp = 24;
/// o.color = codes;  // the pictures' own, matrix stated
/// o.device = &device;  // made with hevc_encoder_device_requirements()
/// o.allocator = &allocator;
/// VKC_ASSIGN(HevcEncoder encoder, HevcEncoder::create(o));
/// core::CommandBatch batch(device, allocator);
/// VKC_ASSIGN(const YuvImage target, encoder.target(batch));
/// // ... record the kernel that writes target's planes into batch ...
/// VKC_ASSIGN(core::PendingBatch written,  // kept until the picture is coded
///            batch.submit_async({}, {{&timeline, n}}));
/// VKC_TRY(encoder.send(encoder.key_owed(), pts, {&timeline, n}));
/// HevcEncoder::CodedPicture coded;
/// VKC_ASSIGN(const bool got, encoder.receive(coded));  // waits for it
/// @endcode
///
/// @warning @ref target, the sends, @ref inject and @ref facts from one
///          thread; @ref receive from one thread, which may be another.
///          @ref in_flight, @ref key_owed and @ref lost from any.
class VS_VIDEO_API HevcEncoder {
 public:
  /// The longest side, in luma samples.
  static constexpr std::uint32_t kMaxSide = 8192;
  /// The most luma samples a picture holds: H.265 level 6.2's MaxLumaPs
  /// (Table A.8), past which VideoToolbox codes nothing.
  static constexpr std::uint64_t kMaxLumaSamples = 35'651'584;
  /// The most effort; 1 is the least.
  static constexpr int kMaxEffort = 7;
  /// Pictures sent and not yet received, at most: one coding while the
  /// device writes the next. A third would only queue.
  static constexpr std::uint32_t kInFlight = 2;

  /// @brief How @ref create opens an encoder.
  struct Options {
    /// The pictures' size: even, each side at most @ref kMaxSide, the two
    /// together at most @ref kMaxLumaSamples. The hardware may refuse a size
    /// inside that (`Status::Code::Unsupported`).
    std::uint32_t width = 0;
    std::uint32_t height = 0;  ///< See @ref width.
    /// Every P-picture's slice QP, 0 to 51. Fixed for the session: to change
    /// it, open another. VideoToolbox codes 1 to 51 and refuses 0.
    int qp = 22;
    /// Every IDR's slice QP, as @ref qp; empty for @ref qp.
    std::optional<int> key_qp;
    /// Speed against quality, 1 (fastest) to @ref kMaxEffort. VideoToolbox
    /// has one speed and ignores it.
    int effort = 4;
    /// The pictures' colour, which the VUI states: the caller's to state
    /// truly. The matrix must be stated (not 2). @ref create first gives
    /// codes that name one thing one code (matrix 5 is 6; transfer 6, 14 and
    /// 15 are 1), then accepts only these: matrix 1, 6, 7 or 9; transfer 1,
    /// 2, 7, 8, 13, 16, 17 or 18; primaries 1, 2, 5, 6, 9, 11, 12 or 22;
    /// either range; chroma siting 0, or 1 to 5 where this backend writes
    /// it (VideoToolbox does). Anything else is
    /// `Status::Code::Unsupported`, naming the code. A @ref target's
    /// matrix weights, range and siting are these.
    ColorCodes color;
    /// The device that writes @ref target's planes, made with
    /// @ref hevc_encoder_device_requirements; on Apple one that imports
    /// Metal textures (`Device::imports_metal_textures`). Null for a session
    /// that codes `CVPixelBuffer`s only, which has no @ref target. Borrowed:
    /// it must outlive the encoder and every target.
    const core::Device* device = nullptr;
    /// Allocates the target buffers of a backend that has its own (NVENC,
    /// when it lands). VideoToolbox's pictures come from its own pool and
    /// need none. Borrowed, as @ref device.
    core::Allocator* allocator = nullptr;
  };

  /// @brief What the session says of itself when it opens, for logs and for
  ///        tests to pin.
  struct Facts {
    std::string encoder;  ///< The hardware encoder, by the backend's name.
    ColorCodes color;     ///< What the VUI states: Options::color, normalized.
    int effort = 0;       ///< As applied; 0 where the backend has one speed.
    /// The horizontal strips each picture is split into across the GPU's
    /// encoders; 0 for the driver's own choice.
    int strips = 0;
    /// The whole configuration, one line, as the session reports it where
    /// it reports it.
    std::string settings;
  };

  /// @brief One coded picture, as @ref receive hands it back.
  struct CodedPicture {
    /// Annex B, each NAL unit after `00 00 00 01`, an IDR's VPS, SPS and PPS
    /// first.
    std::vector<std::uint8_t> access_unit;
    std::int64_t pts = 0;  ///< The one sent with it.
    bool key = false;      ///< An IDR, as the hardware marked it.
  };

  /// @brief A failure for tests to inject, through the backend's own path.
  enum class Fault : std::uint8_t {
    Drop,  ///< The hardware drops the next picture coded.
    /// The session is lost now, as when the system reclaims it: a picture
    /// being coded finishes, and those sent after it fail.
    Lose,
  };

  /// @return The encoder. `Status::Code::InvalidArgument` for a size, QP,
  ///         key QP or effort out of range, matrix 2, or a moved-from
  ///         @ref Options::device. `Status::Code::Unsupported` where there is
  ///         no hardware path, naming what is missing: a build with no
  ///         encoder; a colour code or siting outside @ref Options::color's
  ///         set; a QP the backend does not code; a device without
  ///         @ref hevc_encoder_device_requirements, or on Apple one that
  ///         imports no Metal textures; no hardware HEVC encoder, or one that
  ///         takes no QP a picture; or the hardware refusing the session
  ///         (busy, or at its session limit). Or `Status::Code::OutOfMemory`
  ///         or `Status::Code::Backend` if the backend fails.
  static core::Result<HevcEncoder> create(const Options& options);

  HevcEncoder(HevcEncoder&& other) noexcept;
  HevcEncoder& operator=(HevcEncoder&& other) noexcept;
  HevcEncoder(const HevcEncoder&) = delete;
  HevcEncoder& operator=(const HevcEncoder&) = delete;
  /// @brief Waits for the picture being coded, if any, and for the writer
  ///        of each target sent (its `written`), with no limit, as a
  ///        pending batch does: set every value a sent target waits for.
  ///        Pictures not yet coded are dropped.
  ~HevcEncoder();

  /// @brief A free target, NV12 for the device to write in @p batch: a
  ///        picture of the session's own, one of the few it cycles through
  ///        with @ref kInFlight in flight. A lost session is opened again
  ///        first.
  ///
  /// VideoToolbox: a picture of the session's pool, its two planes
  /// `image[0]` (R8) and `image[1]` (R8G8) over the picture's IOSurface, in
  /// `GENERAL`, with views and `STORAGE`, `SAMPLED`, `TRANSFER_SRC` and
  /// `TRANSFER_DST` usage, at least the picture's size, written from their
  /// corner (PixelBufferImporter's `Access::Write`). Its `kr`, `kb`,
  /// `full_range` and `chroma_location` are @ref Facts::color's, which the
  /// planes' codes must be coded by; `queue_family` is
  /// `kQueueFamilyIgnored`: this records into @p batch what the planes need,
  /// which on VideoToolbox is nothing, as they stay on the device's queue
  /// family.
  ///
  /// Unlike a decoder's picture, this is the encoder's input, not a picture
  /// the caller holds: write it only in @p batch, submit that batch, and
  /// @ref send the timeline value it sets as it completes; then drop
  /// the image, which is valid only until @ref send. A second call before
  /// @ref send gives up the first target. The encoder keeps a target's
  /// images for the batch that writes them, a lost session's too: they go
  /// with the encoder, or once 64 targets have come after the last on the
  /// same picture. Destroy the encoder only once every batch that wrote a
  /// target has completed.
  /// @return The target. `Status::Code::InvalidArgument` when
  ///         @ref kInFlight pictures are in flight (receive first), on a
  ///         moved-from encoder, or on one opened with no
  ///         @ref Options::device. A reopen's failure, as @ref create's. Or
  ///         @p batch's failure, or the pool's, which leave the session as
  ///         it was.
  core::Result<YuvImage> target(core::CommandBatch& batch);

  /// @brief Code the target last handed out (@ref target), an IDR when
  ///        @p key, once @p written is reached. Returns at once: the coding
  ///        thread waits on the host for @p written before it codes.
  /// @param key      An IDR, with its parameter sets.
  /// @param pts      Handed back with its access unit.
  /// @param written  The value the batch that writes the target sets as it
  ///                 completes (`CommandBatch::submit_async`'s signal); a
  ///                 null semaphore when the caller has waited for that
  ///                 batch already. The semaphore must outlive the
  ///                 picture's @ref receive, or the encoder.
  /// @return OK. `Status::Code::InvalidArgument` for no target taken, an
  ///         empty semaphore, no key while @ref key_owed,
  ///         @ref kInFlight pictures already in flight, or a moved-from
  ///         encoder, the target kept for a send that corrects it. A lost
  ///         session's reopening failure, as @ref create's. Coding failures
  ///         come from @ref receive.
  core::Status send(bool key, std::int64_t pts, core::TimelinePoint written);

  /// @brief The oldest picture sent, once coded, into @p picture. Its
  ///        access unit's buffer is swapped with the encoder's, so neither
  ///        side allocates once both have grown.
  /// @param picture     Filled when this returns true.
  /// @param timeout_ns  How long to wait for it; 0 polls. A finite wait
  ///                    waits even with nothing in flight, for a picture
  ///                    another thread sends meanwhile, so a receive thread
  ///                    of its own sleeps between pictures. The default,
  ///                    `UINT64_MAX`, waits for the oldest in flight, and
  ///                    with none returns false at once.
  /// @return true with @p picture filled. false when no picture is coded
  ///         within @p timeout_ns, or at once as @p timeout_ns says.
  ///         `Status::Code::IoError` for a picture the hardware dropped,
  ///         which keeps the session, and the failure of the host's wait for
  ///         a target's `written` (`Status::Code::Backend`, as for a lost
  ///         device), which keeps it too. Or the backend's failure, which
  ///         loses it: `Status::Code::IoError` for no access unit within 2 s
  ///         of the picture's writer, a P-picture where an IDR was asked, or
  ///         two access units for one picture; `Status::Code::Backend`;
  ///         `Status::Code::OutOfMemory`.
  ///         A failed picture fails each picture already sent after it too,
  ///         since each would follow a gap, and a key is owed from the first
  ///         failure. `Status::Code::InvalidArgument` on a moved-from
  ///         encoder.
  core::Result<bool> receive(CodedPicture& picture,
                             std::uint64_t timeout_ns = UINT64_MAX);

  /// @return Pictures sent and not yet received; 0 on a moved-from encoder.
  std::uint32_t in_flight() const noexcept;
  /// @return Whether the next picture must be a key: the first, and from
  ///         the coding thread's first failure until a key is sent. True on
  ///         a moved-from encoder.
  bool key_owed() const noexcept;
  /// @return Whether the session is lost; the next picture opens it again.
  ///         False on a moved-from encoder, which has none.
  bool lost() const noexcept;
  /// @return What the session said when it last opened; empty on a
  ///         moved-from encoder. Read it on the sending thread: the
  ///         @ref target or send that reopens a lost session rewrites it.
  const Facts& facts() const noexcept;
  /// @brief For tests: a failure through the backend's own path. Does
  ///        nothing on a moved-from encoder.
  void inject(Fault fault) noexcept;

 private:
  struct Impl;
  friend struct HevcEncoderAccess;  // the Apple entry's way in
  explicit HevcEncoder(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor
