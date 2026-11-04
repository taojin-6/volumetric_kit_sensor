// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): ARKit's session as its feed drives it, with no
// ARKit type, so the Mac's tests run it. Each event the feed hears -- the
// sensor's open, start, stop and close, ARKit's interruption, its end and its
// failure -- moves the state and names what the feed does to the session
// (arkit_feed.mm). The session is wanted from the open until a stop, and from
// each start: an interruption that ends then runs it again, as ARKit waits to
// be run after one, and a start runs it where it is not running.
//
// Not thread-safe: the feed applies every event on the sensor's queue, where
// ARKit's calls arrive, so a stop either lands before an interruption's end
// reads it or pauses after that end's run.

#include <cstdint>

namespace volumetric_kit::sensor::arkit {

class SessionState {
 public:
  // What the feed does to its session.
  enum class Act : std::uint8_t {
    None,
    RunFresh,  // run it, resetting tracking: the first run, the world's
    Run,       // run it again, the world kept
    Pause,     // pause it, which lets the camera go
  };

  // The open's first run; wanted from here. RunFresh from a new state, else
  // None.
  Act open();
  // Frames from now; wanted from here. Run where the session is not running
  // -- paused by a stop, interrupted, or left waiting by an interruption that
  // ended while it was not wanted --, else None. None once closed or failed.
  Act start();
  // Pause, and not wanted until the next start; None where not started.
  Act stop();
  // The feed goes: Pause once it ran, and nothing runs after, a late end of
  // an interruption included.
  Act close();
  // ARKit has stopped the camera for a while: not running until run again.
  void interruption_began();
  // ARKit has the camera back and waits to be run: Run while wanted, else
  // None (the next start runs it).
  Act interruption_ended();
  // ARKit's session failed: nothing runs it again.
  void failed();

  // From the open until a stop, and from each start; never once closed or
  // failed.
  bool wanted() const;
  // Run since the last stop, close, interruption's start or failure.
  bool running() const { return running_; }
  // Between an interruption's start and its end.
  bool interrupted() const { return interrupted_; }

 private:
  enum class Phase : std::uint8_t { New, Open, Started, Stopped, Closed };
  Act run();  // Run, running from now

  Phase phase_ = Phase::New;
  bool running_ = false;
  bool interrupted_ = false;
  bool failed_ = false;
};

}  // namespace volumetric_kit::sensor::arkit
