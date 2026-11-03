// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

/// @file testing/hevc_conformance.hpp
/// @brief Test support: the rules every backend's @ref HevcEncoder keeps,
///        run on this build's backend and checked with sensor's own syntax
///        readers and decoder (`volumetric_kit::sensor_video_testing`, a
///        static target built wherever `sensor_video` is, with no test
///        framework).
///
/// One runner per platform calls it: sensor's tests on the CI legs, and an
/// application's on a device, so the rules are written once. Pictures go in
/// as pixel buffers (Apple) or as targets the device writes
/// (`HevcEncoder::target`, through testing/nv12_writer.hpp). No pixel
/// reaches the host here; a test that compares pictures reads them back
/// itself.
///
/// @code
/// namespace testing = volumetric_kit::sensor::testing;
///
/// testing::HevcEncoderCase c;
/// c.options.width = 1920;
/// c.options.height = 1440;
/// c.options.color = {6, 1, 6, true, 0};  // BT.601, full range
/// c.device = &device;  // writes the targets, decodes what it codes
/// c.allocator = &allocator;
/// const auto breach = testing::hevc_encoder_breach(c);
/// if (!breach) {
///   // create's refusal: no hardware path here; skip unless promised
///   // (testing::hevc_encoder_promised()), else fail
/// } else if (!breach->empty()) {
///   std::fprintf(stderr, "%s\n", breach->c_str());  // the rule broken
/// }
/// @endcode

#include <cstdint>
#include <string>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/video/hevc_encoder.hpp"
#include "volumetric_kit/sensor/video/testing_export.hpp"

namespace volumetric_kit::sensor::testing {

/// @brief One run of the encoder rules.
struct HevcEncoderCase {
  /// The session's options; the rules' refusals are made from them. For
  /// targets, its device and allocator are the case's.
  HevcEncoder::Options options;
  /// The device that writes the targets, made with
  /// `hevc_encoder_device_requirements()`, and decodes the coded stream
  /// through @ref HevcDecoder (rule C6); with pixel buffers, null leaves the
  /// decode out. Borrowed for the call.
  const core::Device* device = nullptr;
  /// The allocator the targets' writes and the decoder's pictures are made
  /// through; as @ref device.
  core::Allocator* allocator = nullptr;
  /// Apple: code IOSurface pixel buffers the run makes, through
  /// send_pixel_buffer, on a session with or without
  /// `HevcEncoder::Options::device`. Otherwise targets the run takes
  /// (`HevcEncoder::target`) and writes on the case's device with a compute
  /// kernel (@ref Nv12Writer), each sent with the value its batch sets.
  bool pixel_buffers = false;
  /// The main stream's pictures, at least 4: a key, P-pictures, and a key
  /// asked at the middle one.
  std::uint32_t pictures = 30;
};

/// @brief Run the encoder rules on @p c with this build's backend:
///        - C1 refusals: each argument out of range is
///          `Status::Code::InvalidArgument`, each code outside the set and
///          a siting the backend cannot write `Status::Code::Unsupported`;
///        - C2 the first picture must be a key, and a refused target is kept
///          for the send that corrects it;
///        - C3 one Annex B access unit a picture, 4-byte start codes, an IDR
///          with its VPS, SPS and PPS first, a P-picture with no IRAP, every
///          NAL unit in temporal layer 0;
///        - C4 the SPS: Main, 8-bit 4:2:0, the size, no reordering, and the
///          VUI as `HevcEncoder::Facts::color`, the options normalized;
///        - C5 each slice's QP the session's;
///        - C6 the stream decodes to the size and colour asked, and again
///          from the middle key after a reset (with @ref HevcEncoderCase::
///          device), and a target is described by that colour;
///        - C7 a dropped picture fails alone and owes a key;
///        - C8 a lost session fails what it had, never one picture coded
///          after one that failed, and the next picture, a key, opens it
///          again, once a picture coding at the loss is done;
///        - C9 a target is NV12 R8 and R8G8 storage images in `GENERAL` of
///          the session's size, on the device's queue family; one taken
///          twice before a send gives up the first, and one access unit
///          follows; a send with no target taken is refused;
///        - C10 the settings line is the backend's pinned one;
///        - C11 pixel buffers of another size, format or tag are refused,
///          and one with no colour tags codes as the session's colour; a
///          buffer waits for a target taken; a session with no device hands
///          out no target and codes buffers;
///        - C12 a picture sent before a move, or a target taken before one,
///          comes back after it;
///        - C13 pictures in flight come back in order, at most
///          `HevcEncoder::kInFlight` of them (a third target refused), a
///          dropped one failing the one sent after it, and sent on one
///          thread and received on another, whose finite waits sleep while
///          nothing is in flight;
///        - C13b (targets) the coding thread waits for each target's
///          writer: a batch that writes the target only after a value a
///          helper thread sets 1 s later; send returns before the writer's
///          value is reached, and the access unit comes back only after the
///          helper set its value.
///
///        What each rule checks holds whatever the timing: a loaded machine
///        changes which path a picture takes, not whether the rule holds.
///        C13's drop in flight needs one picture sent while the drop codes,
///        and tries up to 5 rounds for it.
/// @return Empty when the encoder keeps every rule; else the first rule it
///         breaks, named ("C5: ..."). `HevcEncoder::create`'s status where
///         @p c's session does not open (`Status::Code::Unsupported` where
///         there is no hardware path, which a runner skips unless one is
///         promised). `Status::Code::InvalidArgument` for a case this build
///         cannot run: off Apple, fewer than 4 pictures, or targets with no
///         device or allocator.
VS_VIDEO_TESTING_API core::Result<std::string> hevc_encoder_breach(
    const HevcEncoderCase& c);

}  // namespace volumetric_kit::sensor::testing
