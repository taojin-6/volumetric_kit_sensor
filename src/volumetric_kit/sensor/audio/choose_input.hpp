// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): which input Microphone::Options::device names,
// apart from PortAudio, so a test checks it with no device.

#include <cstddef>
#include <string>
#include <vector>

#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/sensor/audio/microphone.hpp"

namespace volumetric_kit::sensor::audio {

// The names of `inputs[i]` for each i in `which`, quoted, comma-separated;
// "none" for none.
inline std::string quoted_names(const std::vector<Microphone::Input>& inputs,
                                const std::vector<std::size_t>& which) {
  if (which.empty()) return "none";
  std::string names;
  for (const std::size_t i : which) {
    if (!names.empty()) names += ", ";
    names += "\"" + inputs[i].name + "\"";
  }
  return names;
}

// The index in `inputs` of the one `device` names: the default input for
// empty, else the one named exactly, else the one whose name contains it.
// NotFound, naming every input, when none matches; InvalidArgument, naming
// the matches, when several do.
inline core::Result<std::size_t> choose_input(
    const std::vector<Microphone::Input>& inputs, const std::string& device) {
  std::vector<std::size_t> matches;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (device.empty() ? inputs[i].is_default : inputs[i].name == device) {
      matches.push_back(i);
    }
  }
  if (matches.empty() && !device.empty()) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].name.find(device) != std::string::npos) {
        matches.push_back(i);
      }
    }
  }
  if (matches.empty()) {
    std::vector<std::size_t> every(inputs.size());
    for (std::size_t i = 0; i < every.size(); ++i) every[i] = i;
    const std::string what = device.empty() ? std::string("no default input")
                                            : "no input \"" + device + "\"";
    return core::Status::not_found(
        what + "; inputs found: " + quoted_names(inputs, every));
  }
  if (matches.size() > 1) {
    return core::Status::invalid_argument(
        "\"" + device +
        "\" names several inputs: " + quoted_names(inputs, matches));
  }
  return matches.front();
}

}  // namespace volumetric_kit::sensor::audio
