// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/hevc_slices.hpp
/// @brief Test support: what a test reads of a coded H.265 stream beyond
///        video/hevc_syntax.hpp, each sequence's profile and declared
///        sub-layers and each picture's first slice type and QP
///        (`volumetric_kit::sensor_video_syntax`, every build).
///
/// No decode path calls it. It walks the parameter sets with the same
/// readers as @ref hevc::parse_sps and @ref hevc::parse_pps, and the slice
/// header on from where @ref hevc::parse_slice_start stops. Clause numbers
/// are ITU-T H.265's.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// testing::SliceReader reader;
/// for (const std::vector<std::uint8_t>& unit : units) {
///   VKC_ASSIGN(const testing::SliceReader::Picture p,
///              reader.read(unit.data(), unit.size()));
///   // p.slice_type: 2 for an I slice; p.slice_qp: SliceQpY
/// }
/// const auto& sequence = reader.sequence();  // the last SPS read
/// @endcode

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/video/hevc_syntax.hpp"
#include "volumetric_kit/sensor/video/syntax_export.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief Reads access units in decode order: the parameter sets each one
///        carries serve it and those after it, and its first slice
///        segment's header is read through slice_qp_delta (7.3.6.1).
///
/// @warning Not thread-safe: use from one thread.
class VS_VIDEO_SYNTAX_API SliceReader {
 public:
  /// @brief What a sequence parameter set declares beyond @ref hevc::Sps.
  struct Sequence {
    hevc::Sps sps;           ///< As @ref hevc::parse_sps reads it.
    int profile_idc = 0;     ///< general_profile_idc: 1 for Main.
    int max_sub_layers = 1;  ///< sps_max_sub_layers_minus1 + 1.
  };

  /// @brief What one access unit's picture says of itself.
  struct Picture {
    int nal_type = 0;    ///< The first slice segment's nal_unit_type.
    int slice_type = 0;  ///< Its slice_type: 0 B, 1 P, 2 I.
    int slice_qp = 0;  ///< Its SliceQpY: 26 + init_qp_minus26 + slice_qp_delta.
    /// The highest TemporalId among the access unit's NAL units.
    int max_temporal_id = 0;
  };

  SliceReader();
  ~SliceReader();
  SliceReader(const SliceReader&) = delete;
  SliceReader& operator=(const SliceReader&) = delete;

  /// @brief Read one access unit of Annex B bytes.
  /// @return Its picture's fields; `Status::Code::IoError` if it does not
  ///         split, holds no slice segment, its first is not the first of a
  ///         picture, or a parameter set or that slice's header does not
  ///         parse or names a set not read; `Status::Code::InvalidArgument`
  ///         for null @p data with a size.
  core::Result<Picture> read(const std::uint8_t* data, std::size_t size);

  /// @return The last SPS read; empty before one.
  const std::optional<Sequence>& sequence() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace volumetric_kit::sensor::testing
