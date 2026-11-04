// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// What ARKit's feed does to its session at each event (arkit/session_state.hpp,
// sensor_arkit_units), with no session and no device: the state drives a fake
// of ARKit's session as the project observed it, which delivers once run until
// paused or interrupted, and after an interruption waits to be run again. An
// interruption that ends during the open, between the open and the start, or
// after the start runs it again, and one still on at a start is run at the
// start; a stop pauses it and keeps it off through an interruption; a failed
// open's session, closed, stays off while a new one opens; a close during the
// open runs nothing after; ARKit's failure runs nothing again; one world (one
// run that resets tracking) for each open.

#include <cstdio>

#include "session_state.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"

namespace arkit = volumetric_kit::sensor::arkit;

namespace {

using Act = arkit::SessionState::Act;

// ARKit's session: running once run, until paused; an interruption stops its
// frames, and at its end it waits to be run again.
struct FakeSession {
  bool run = false;
  bool interrupted = false;
  bool waiting = false;  // an interruption ended, not run since
  int fresh = 0;         // runs that reset tracking: worlds
  int runs = 0;          // runs that kept the world
  int pauses = 0;

  void apply(Act act) {
    switch (act) {
      case Act::RunFresh:
        ++fresh;
        run = true;
        waiting = false;
        break;
      case Act::Run:
        ++runs;
        run = true;
        waiting = false;
        break;
      case Act::Pause:
        ++pauses;
        run = false;
        break;
      case Act::None:
        break;
    }
  }
  bool delivers() const { return run && !interrupted && !waiting; }
};

// The feed: each event to the state, and its act to the session.
struct Feed {
  arkit::SessionState state;
  FakeSession session;

  Act open() { return applied(state.open()); }
  Act start() { return applied(state.start()); }
  Act stop() { return applied(state.stop()); }
  Act close() { return applied(state.close()); }
  void interrupt() {
    session.interrupted = true;
    state.interruption_began();
  }
  Act end_interruption() {
    session.interrupted = false;
    session.waiting = true;
    return applied(state.interruption_ended());
  }
  void fail() {
    session.run = false;
    state.failed();
  }

 private:
  Act applied(Act act) {
    session.apply(act);
    return act;
  }
};

int test_events() {
  Feed f;
  // Nothing runs before the open, and a new state never ran.
  CHECK(f.start() == Act::None && f.stop() == Act::None);
  CHECK(!f.state.wanted() && !f.state.running());
  CHECK(f.open() == Act::RunFresh);
  CHECK(f.open() == Act::None);  // one open, one world
  CHECK(f.state.wanted() && f.state.running() && f.session.delivers());
  CHECK(f.start() == Act::None);  // running: not run again
  CHECK(f.start() == Act::None);  // started already
  CHECK(f.stop() == Act::Pause && !f.state.wanted());
  CHECK(f.stop() == Act::None);
  CHECK(f.start() == Act::Run && f.session.delivers());
  CHECK(f.close() == Act::Pause && !f.state.wanted());
  CHECK(f.close() == Act::None && f.start() == Act::None);
  CHECK(f.session.fresh == 1 && f.session.runs == 1);
  // A state closed before it opened ran nothing.
  arkit::SessionState unopened;
  CHECK(unopened.close() == Act::None && unopened.open() == Act::None);
  return 0;
}

int test_interrupted_before_start() {
  // During the open: its end runs the session again, so the open still gets
  // its first frame.
  {
    Feed f;
    f.open();
    f.interrupt();
    CHECK(f.state.interrupted() && !f.session.delivers());
    CHECK(f.end_interruption() == Act::Run);
    CHECK(!f.state.interrupted() && f.session.delivers());
  }
  // Between the open and the start (the session opened, not started): run
  // again at its end, and not again at the start.
  {
    Feed f;
    f.open();
    f.interrupt();
    CHECK(f.end_interruption() == Act::Run);
    CHECK(f.start() == Act::None);
    CHECK(f.session.delivers() && f.session.runs == 1);
  }
  // Still on at the start: run at the start, and again at its end.
  {
    Feed f;
    f.open();
    f.interrupt();
    CHECK(!f.state.running());
    CHECK(f.start() == Act::Run);
    CHECK(!f.session.delivers());  // the camera is not back yet
    CHECK(f.end_interruption() == Act::Run);
    CHECK(f.session.delivers() && f.session.fresh == 1);
  }
  return 0;
}

int test_interrupted_after_start() {
  Feed f;
  f.open();
  CHECK(f.start() == Act::None);
  for (int i = 0; i < 2; ++i) {  // twice: the device locked, then an app over
    f.interrupt();
    CHECK(!f.session.delivers() && f.state.wanted());
    CHECK(f.end_interruption() == Act::Run);
    CHECK(f.session.delivers());
  }
  CHECK(f.session.fresh == 1 && f.session.runs == 2);  // the world kept
  return 0;
}

int test_stopped() {
  Feed f;
  f.open();
  f.start();
  CHECK(f.stop() == Act::Pause && !f.session.delivers());
  // An interruption while stopped: its end leaves the camera off.
  f.interrupt();
  CHECK(f.end_interruption() == Act::None && !f.session.delivers());
  CHECK(!f.state.running());
  // The next start runs it, the world kept.
  CHECK(f.start() == Act::Run && f.session.delivers());
  CHECK(f.session.fresh == 1 && f.session.runs == 1);
  return 0;
}

int test_failed_open_then_reopen() {
  // The open gives up while ARKit has the session interrupted (the device
  // locked at launch): the feed closes it, and an end that comes later runs
  // nothing.
  Feed first;
  first.open();
  first.interrupt();
  CHECK(first.close() == Act::Pause);
  CHECK(first.end_interruption() == Act::None);
  CHECK(!first.session.delivers() && first.session.runs == 0);
  // The open again: a new feed, its own world.
  Feed second;
  CHECK(second.open() == Act::RunFresh && second.session.delivers());
  CHECK(second.start() == Act::None && second.session.delivers());
  CHECK(!first.session.delivers());
  return 0;
}

int test_close_during_open() {
  Feed f;
  f.open();  // no frame yet
  CHECK(f.close() == Act::Pause && !f.session.delivers());
  f.interrupt();
  CHECK(f.end_interruption() == Act::None);
  CHECK(f.start() == Act::None && f.stop() == Act::None);
  CHECK(!f.session.delivers() && !f.state.wanted());
  CHECK(f.session.runs == 0 && f.session.fresh == 1);
  return 0;
}

int test_failure() {
  Feed f;
  f.open();
  f.start();
  f.fail();
  CHECK(!f.state.wanted() && !f.state.running());
  f.interrupt();
  CHECK(f.end_interruption() == Act::None);
  CHECK(f.stop() == Act::Pause);  // the stop still pauses
  CHECK(f.start() == Act::None && !f.session.delivers());
  CHECK(f.session.runs == 0);
  return 0;
}

}  // namespace

int main() {
  if (test_events() != 0) return 1;
  if (test_interrupted_before_start() != 0) return 1;
  if (test_interrupted_after_start() != 0) return 1;
  if (test_stopped() != 0) return 1;
  if (test_failed_open_then_reopen() != 0) return 1;
  if (test_close_during_open() != 0) return 1;
  if (test_failure() != 0) return 1;
  std::puts("sensor_arkit_session: OK");
  return 0;
}
