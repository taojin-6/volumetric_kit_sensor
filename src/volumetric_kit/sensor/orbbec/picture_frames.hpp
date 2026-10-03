// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed), and built only with the video decoders: the SDK
// colour frame a raw pair hands a decoded picture on in, from a decode thread
// through the mailbox. A picture the hardware left on the GPU travels with a
// frame whose bytes only name it, so it lives exactly as long as its frame --
// dropped with it, grouped into a rig's set with it, held with it -- and
// every step between the SDK and the raw frame goes on handling frame sets.
// A picture on the host travels as an I420 frame of its planes.

#include <memory>
#include <optional>

#include <libobsensor/ObSensor.hpp>

#include "volumetric_kit/recon/core/color_space.hpp"
#include "volumetric_kit/recon/sensor/raw_frame.hpp"
#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

// What an I420 frame's planes are coded in, carried in the frame's metadata
// since an SDK frame has no field for it: the matrix and range the decoder
// resolved, the stream's own when it names them, and the transfer and
// primaries it declares, none when ColorEncoding cannot name them.
struct PlanesColor {
  VideoColorMatrix matrix = VideoColorMatrix::Bt709;
  bool full_range = false;
  bool has_encoding = false;
  ChromaLocation chroma_location = ChromaLocation::Left;
  ColorEncoding encoding{};
};

// The description an I420 frame from raw_color_frame carries; empty for a
// frame that carries none.
std::optional<PlanesColor> planes_color(const ob::Frame& frame);

// The colour frame for @p picture. Where it is on the device, a frame that
// keeps it, and with it its device buffer or images, until the frame is
// freed: a 0x0 video frame of a compressed format whose bytes only name the
// picture, so a copy of the frame owns nothing, and once the frame is freed
// the copy carries no picture. Else an I420 frame of its planes (Y, then Cb
// and Cr at half size, rows packed) carrying its PlanesColor.
std::shared_ptr<ob::Frame> raw_color_frame(const DecodedPicture& picture);

// The picture a live frame from raw_color_frame carries on the device, which
// the copy returned holds; empty for any other frame.
std::optional<DecodedPicture> device_picture(const ob::Frame& frame);

// Point @p image at @p picture's planes where the hardware left them:
// NVDEC's or nvJPEG's buffer, taken over from kQueueFamilyExternal, or
// VideoToolbox's images. Its size and colour description are the caller's to
// set.
void place_device_color(const DecodedPicture& picture, YuvImage* image);

// A pair rebuilt around its decoded colour: `depth`, and `decoded` dated as
// `source`, the colour frame it was decoded from.
std::shared_ptr<ob::FrameSet> rebuilt_pair(std::shared_ptr<ob::Frame> depth,
                                           const ob::Frame& source,
                                           std::shared_ptr<ob::Frame> decoded);

}  // namespace volumetric_kit::recon::sensor::orbbec
