// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// H.265 through VideoToolbox's hardware decoder (Apple), with no FFmpeg:
// HevcDecoder's work there. It reads the parameter sets and slice headers
// itself (hevc_syntax.hpp), hands VideoToolbox each picture's slices as one
// length-prefixed sample, and puts the pictures, which come back in decode
// order, into display order by their picture order count, as the standard's
// output process does (C.5.2). Each is NV12 on an IOSurface that
// PixelBufferImporter hands to Vulkan as images, so no picture reaches the
// host.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "frame_color.hpp"
#include "videotoolbox.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/recon/sensor/apple/pixel_buffer_importer.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"
#include "volumetric_kit/recon/sensor/video/hevc_syntax.hpp"

namespace volumetric_kit::recon::sensor::video {

class VtHevc {
 public:
  // Unsupported where VideoToolbox has no hardware HEVC decoder or @p device
  // imports no Metal textures; InvalidArgument for @p queue_families the
  // importer refuses. @p who names the decoder in errors.
  static core::Result<std::unique_ptr<VtHevc>> open(
      const core::Device& device, const char* who,
      std::optional<VideoColorDescription> unlabelled_color,
      const std::uint32_t* queue_families, std::uint32_t queue_family_count);
  VtHevc(const VtHevc&) = delete;
  VtHevc& operator=(const VtHevc&) = delete;

  // HevcDecoder::send, receive and reset, with the codes they document.
  core::Status send(const std::uint8_t* data, std::size_t size,
                    std::int64_t pts);
  core::Result<std::optional<DecodedPicture>> receive();
  void reset() noexcept;

 private:
  // A decoded picture waiting to be shown, or to be taken.
  struct Held {
    std::int32_t poc = 0;
    std::int64_t pts = 0;
    PixelBufferPtr pixels;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    ColorCodes color;
  };

  VtHevc(PixelBufferImporter pictures, const char* who,
         std::optional<VideoColorDescription> unlabelled_color) noexcept
      : who_(who),
        unlabelled_color_(std::move(unlabelled_color)),
        pictures_(std::move(pictures)) {}

  core::Status store_parameter_set(const hevc::Nal& nal);
  // The picture of @p slices, the access unit's VCL NAL units, decoded.
  core::Status decode(const std::vector<hevc::Nal>& slices, std::int64_t pts);
  // A session, and the format its samples carry, for @p sps's sequence.
  core::Status ready_session(const hevc::Sps& sps);
  // The held picture first in display order shown: made ready for receive.
  void show_next();
  // Every held picture shown, in display order.
  void show_all();
  // @p refusal (Unsupported), standing from now on; the pictures held are
  // shown first.
  core::Status refuse(const core::Status& refusal);
  core::Status io_error(const std::string& why) const;
  core::Status unsupported(const std::string& why) const;

  const char* who_;
  std::optional<VideoColorDescription> unlabelled_color_;
  PixelBufferImporter pictures_;

  // The parameter sets by id: as sent, for the format description, and the
  // fields read from them. A change bumps params_version_.
  std::map<std::uint32_t, std::vector<std::uint8_t>> vps_bytes_;
  std::map<std::uint32_t, std::vector<std::uint8_t>> sps_bytes_;
  std::map<std::uint32_t, std::vector<std::uint8_t>> pps_bytes_;
  std::map<std::uint32_t, hevc::Sps> sps_;
  std::map<std::uint32_t, hevc::Pps> pps_;
  std::uint64_t params_version_ = 0;

  FormatPtr format_;
  SessionPtr session_;
  std::uint64_t format_version_ = 0;  // params_version_ format_ was made at
  std::uint32_t format_sps_ = 0;      // the SPS it describes
  bool format_full_range_ = false;    // the range session_ writes

  // Decoding order (8.1.3, 8.3.1).
  bool first_picture_ = true;  // the next IRAP starts a sequence afresh
  bool skip_rasl_ = false;     // the last IRAP's leading pictures are lost
  std::int32_t prev_tid0_poc_ = 0;

  std::vector<Held> held_;  // decoded, waiting for display order
  std::deque<Held> ready_;  // in display order, waiting for receive
  // An access unit's NAL units, its picture's slices among them, and those
  // slices length-prefixed, their storage kept from one unit to the next.
  std::vector<hevc::Nal> nals_;
  std::vector<hevc::Nal> slices_;
  std::vector<std::uint8_t> sample_;
  bool ended_ = false;
  std::string refusal_;
};

}  // namespace volumetric_kit::recon::sensor::video
