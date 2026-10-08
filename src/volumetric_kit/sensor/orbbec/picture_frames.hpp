// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): the SDK colour frame a pair hands a decoded
// picture on in, from a decode thread through the mailbox. The picture the
// hardware left on the GPU travels with a frame whose bytes only name it, so
// it lives exactly as long as its frame -- dropped with it, grouped into a
// rig's set with it, held with it -- and every step between the SDK and the
// sensor's frame goes on handling frame sets.

#include <memory>
#include <optional>

#include <libobsensor/ObSensor.hpp>

#include "volumetric_kit/recon/sensor/video/decoded_picture.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

// A frame that keeps @p picture, and with it its device buffer or images,
// until the frame is freed: a 0x0 video frame of a compressed format whose
// bytes only name the picture, so a copy of the frame owns nothing, and once
// the frame is freed the copy carries no picture.
std::shared_ptr<ob::Frame> picture_frame(const DecodedPicture& picture);

// The picture a live frame from picture_frame carries, which the copy
// returned holds; empty for any other frame.
std::optional<DecodedPicture> device_picture(const ob::Frame& frame);

// A pair rebuilt around its decoded colour: `depth`, and `decoded` dated as
// `source`, the colour frame it was decoded from.
std::shared_ptr<ob::FrameSet> rebuilt_pair(std::shared_ptr<ob::Frame> depth,
                                           const ob::Frame& source,
                                           std::shared_ptr<ob::Frame> decoded);

}  // namespace volumetric_kit::recon::sensor::orbbec
