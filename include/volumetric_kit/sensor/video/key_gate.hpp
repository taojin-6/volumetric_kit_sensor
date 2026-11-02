// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file video/key_gate.hpp
/// @brief Which access units of a predicted H.265 stream can be decoded
///        after a loss.
///
/// Host code in `volumetric_kit::sensor_video_syntax`, on every platform. A
/// unit lost on the way leaves the ones after it predicted from a picture
/// the decoder never had, up to the next key; the units' numbers are what
/// show the loss.

#include <cstdint>
#include <optional>

#include "volumetric_kit/sensor/video/syntax_export.hpp"

namespace volumetric_kit::sensor {

/// @brief Waits for a key (an IRAP picture) at the start, after a gap in the
///        units' numbers, and after @ref lose(). The key that ends a wait,
///        the first one included, is @ref Admission::Restart:
///        `HevcDecoder::reset()` before sending it, which does nothing on a
///        decoder that has decoded nothing. An unnumbered unit is never a
///        gap, and the unit after it is not checked against the one before
///        it.
///
/// A gap is a number other than one up from the last: a unit lost, one
/// repeated, or a source that started its count again. A key within an
/// unbroken stream is @ref Admission::Take. One caller at a time: the gate is
/// not synchronized.
///
/// @code
/// const std::optional<int> type = hevc::first_slice_type(au, size);
/// switch (gate.admit(seq, type && hevc::is_irap(*type))) {
///   case KeyGate::Admission::Skip: return;  // ask the sender for a key
///   case KeyGate::Admission::Restart: VKC_TRY(decoder.reset()); break;
///   case KeyGate::Admission::Take: break;
/// }
/// if (!decoder.send(au, size, pts).ok()) gate.lose();
/// @endcode
class VS_VIDEO_SYNTAX_API KeyGate {
 public:
  /// @brief What to do with a unit.
  enum class Admission : std::uint8_t {
    Skip,     ///< Waiting for a key: do not send it.
    Take,     ///< The next unit of an unbroken stream: send it.
    Restart,  ///< The key that ends a wait: reset the decoder, then send it.
  };

  /// @brief Admit the next unit.
  /// @param seq The unit's number, one up per unit; empty for an unnumbered
  ///            unit.
  /// @param key Whether the unit is a key.
  /// @return What to do with it.
  Admission admit(std::optional<std::uint64_t> seq, bool key) noexcept;

  /// @brief Wait for the next key: a send or receive failed, or units were
  ///        lost unseen.
  void lose() noexcept;

  /// @return Whether it waits for a key.
  bool waiting() const noexcept { return waiting_; }

 private:
  bool waiting_ = true;  // the first unit sent is a key
  std::optional<std::uint64_t> last_;
};

}  // namespace volumetric_kit::sensor
