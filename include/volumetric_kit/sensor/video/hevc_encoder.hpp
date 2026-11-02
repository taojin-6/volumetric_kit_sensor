// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/hevc_encoder.hpp
/// @brief An H.265/HEVC encoder on the GPU's hardware, the twin of
///        @ref HevcDecoder: each picture sent comes back as one Annex B
///        access unit, in the order sent.
///
/// No FFmpeg, VideoToolbox or Core Video type appears here. Apple's way in,
/// a `CVPixelBuffer` sent as it is, is video/hevc_encoder_apple.hpp.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/color_codes.hpp"
#include "volumetric_kit/sensor/video/export.hpp"

namespace volumetric_kit::sensor {

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
/// order sent. A coding thread of the encoder's own codes each picture.
/// The hardware holds a thread for the whole encode in this configuration
/// (VideoToolbox's low-latency mode codes inside its call), and this keeps
/// that thread off the caller's. Encoders on separate threads code side by
/// side where the hardware has the engines.
///
/// The way in, on Apple, is send_pixel_buffer
/// (video/hevc_encoder_apple.hpp): a `CVPixelBuffer` sent as it is, such as
/// a camera's.
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
/// VKC_ASSIGN(HevcEncoder encoder, HevcEncoder::create(o));
/// VKC_TRY(send_pixel_buffer(encoder, pixels, encoder.key_owed(), pts));
/// HevcEncoder::CodedPicture coded;
/// VKC_ASSIGN(const bool got, encoder.receive(coded));  // waits for it
/// @endcode
///
/// @warning Send and @ref inject from one thread; @ref receive from one
///          thread, which may be another. @ref in_flight, @ref key_owed and
///          @ref lost from any.
class VS_VIDEO_API HevcEncoder {
 public:
  /// The longest side, in luma samples.
  static constexpr std::uint32_t kMaxSide = 8192;
  /// The most luma samples a picture holds: H.265 level 6.2's MaxLumaPs
  /// (Table A.8), past which VideoToolbox codes nothing.
  static constexpr std::uint64_t kMaxLumaSamples = 35'651'584;
  /// The most effort; 1 is the least.
  static constexpr int kMaxEffort = 7;
  /// Pictures sent and not yet received, at most: one coding while the next
  /// is made. A third would only queue.
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
    /// `Status::Code::Unsupported`, naming the code.
    ColorCodes color;
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
    bool key = false;      ///< An IDR.
  };

  /// @brief A failure for tests to inject, through the backend's own path.
  enum class Fault : std::uint8_t {
    Drop,  ///< The hardware drops the next picture coded.
    /// The session is lost now, as when the system reclaims it: a picture
    /// being coded finishes, and those sent after it fail.
    Lose,
  };

  /// @return The encoder. `Status::Code::InvalidArgument` for a size, QP,
  ///         key QP or effort out of range, or matrix 2.
  ///         `Status::Code::Unsupported` where there is no hardware path,
  ///         naming what is missing: a build with no encoder; a colour code
  ///         or siting outside @ref Options::color's set; a QP the backend
  ///         does not code; no hardware HEVC encoder, or one that takes no
  ///         QP a picture; or the hardware refusing the session (busy, or at
  ///         its session limit). Or `Status::Code::OutOfMemory` or
  ///         `Status::Code::Backend` if the backend fails.
  static core::Result<HevcEncoder> create(const Options& options);

  HevcEncoder(HevcEncoder&& other) noexcept;
  HevcEncoder& operator=(HevcEncoder&& other) noexcept;
  HevcEncoder(const HevcEncoder&) = delete;
  HevcEncoder& operator=(const HevcEncoder&) = delete;
  /// @brief Waits for the picture being coded, if any; pictures not yet
  ///        coded are dropped.
  ~HevcEncoder();

  /// @brief The oldest picture sent, once coded, into @p picture. Its
  ///        access unit's buffer is swapped with the encoder's, so neither
  ///        side allocates once both have grown.
  /// @param picture     Filled when this returns true.
  /// @param timeout_ns  How long to wait for it; 0 polls.
  /// @return true with @p picture filled. false when nothing is in flight,
  ///         or the oldest is not coded within @p timeout_ns.
  ///         `Status::Code::IoError` for a picture the hardware dropped,
  ///         which keeps the session. Or the backend's failure, which loses
  ///         it: `Status::Code::IoError` for no access unit within 2 s, a
  ///         P-picture where an IDR was asked, or two access units for one
  ///         picture; `Status::Code::Backend`; `Status::Code::OutOfMemory`.
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
  ///         moved-from encoder.
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
