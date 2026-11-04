// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "session_state.hpp"

namespace volumetric_kit::sensor::arkit {

SessionState::Act SessionState::run() {
  running_ = true;
  return Act::Run;
}

SessionState::Act SessionState::open() {
  if (phase_ != Phase::New) return Act::None;
  phase_ = Phase::Open;
  running_ = true;
  return Act::RunFresh;
}

SessionState::Act SessionState::start() {
  if (phase_ != Phase::Open && phase_ != Phase::Stopped) return Act::None;
  phase_ = Phase::Started;
  if (failed_ || running_) return Act::None;
  return run();
}

SessionState::Act SessionState::stop() {
  if (phase_ != Phase::Started) return Act::None;
  phase_ = Phase::Stopped;
  running_ = false;
  return Act::Pause;
}

SessionState::Act SessionState::close() {
  const bool ran = phase_ != Phase::New && phase_ != Phase::Closed;
  phase_ = Phase::Closed;
  running_ = false;
  return ran ? Act::Pause : Act::None;
}

void SessionState::interruption_began() {
  interrupted_ = true;
  running_ = false;
}

SessionState::Act SessionState::interruption_ended() {
  interrupted_ = false;
  return wanted() ? run() : Act::None;
}

void SessionState::failed() {
  failed_ = true;
  running_ = false;
}

bool SessionState::wanted() const {
  return !failed_ && (phase_ == Phase::Open || phase_ == Phase::Started);
}

}  // namespace volumetric_kit::sensor::arkit
