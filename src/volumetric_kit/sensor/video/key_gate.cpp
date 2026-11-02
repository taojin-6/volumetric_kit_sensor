// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/video/key_gate.hpp"

namespace volumetric_kit::sensor {

KeyGate::Admission KeyGate::admit(std::optional<std::uint64_t> seq,
                                  bool key) noexcept {
  if (seq && last_ && *seq != *last_ + 1) waiting_ = true;
  last_ = seq;
  if (!waiting_) return Admission::Take;
  if (!key) return Admission::Skip;
  waiting_ = false;
  return Admission::Restart;
}

void KeyGate::lose() noexcept { waiting_ = true; }

}  // namespace volumetric_kit::sensor
