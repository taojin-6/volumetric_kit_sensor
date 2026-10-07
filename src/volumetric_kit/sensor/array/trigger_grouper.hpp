// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): SensorArray's grouping of frames from several
// hardware-synchronised sensors into one set per sync trigger, by their
// timestamps on a shared clock. Pure bookkeeping over timestamps and the
// caller's opaque frame ids -- it holds no frame -- so a host test drives it
// with synthetic streams, compiling this source itself. The measurements
// behind its rules are in the 2026-09-27 rig decision.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace volumetric_kit::recon::sensor {

/// @brief Groups each sensor's frames into one set per trigger, around the
///        anchor (the sync primary)'s frames.
///
/// A frame of another sensor within @ref Config::tolerance_us of an anchor
/// frame belongs to its trigger. A trigger is ready when every sensor's frame
/// for it has arrived, or when each missing sensor has moved past it (sent a
/// later frame) or stayed silent for @ref Config::max_wait_us. An older ready
/// trigger is dropped for a newer one, so a slow consumer gets the newest set.
///
/// @code
/// TriggerGrouper grouper(config);
/// std::vector<std::uint64_t> released;  // ids whose frames the caller drops
/// grouper.add(camera, frame_us, frame_id, now_us, &released);
/// if (auto group = grouper.take(now_us, &released)) {
///   // group->ids[c]: camera c's frame for the trigger, if it came
/// }
/// @endcode
class TriggerGrouper {
 public:
  /// @brief The sensors and the grouping's tolerances.
  struct Config {
    std::size_t cameras = 0;  ///< How many sensors.
    /// The primary. Its frames name the triggers: a trigger without one is
    /// never handed out.
    std::size_t anchor = 0;
    /// A sensor's frame within this of an anchor frame (shared clock)
    /// belongs to its trigger.
    std::uint64_t tolerance_us = 5000;
    /// How long (host clock) a trigger waits for a sensor that has sent
    /// nothing newer before it is handed out without that sensor.
    std::uint64_t max_wait_us = 50000;
    /// Frames held per sensor; a fuller queue lets its oldest go.
    std::size_t queue_depth = 4;
  };

  /// @brief One trigger: an id per sensor, empty where that sensor's frame
  ///        for it never arrived.
  struct Group {
    std::uint64_t timestamp_us = 0;  ///< The anchor's frame's.
    /// One per sensor: its frame's id, empty where it never arrived.
    std::vector<std::optional<std::uint64_t>> ids;
  };

  /// @param config  The sensors and tolerances.
  /// @pre `Config::anchor` is below `Config::cameras`, and
  ///      `Config::queue_depth` is at least 1.
  explicit TriggerGrouper(const Config& config);

  /// @brief Hold a frame for grouping.
  /// @param camera    Its sensor, below `Config::cameras`.
  /// @param ts_us     Its timestamp on the shared clock (us).
  /// @param id        The caller's id for it, handed back in a group or in
  ///                  @p released.
  /// @param now_us    The host time it arrived (us), for the wait.
  /// @param released  Receives the ids a full queue let go.
  void add(std::size_t camera, std::uint64_t ts_us, std::uint64_t id,
           std::uint64_t now_us, std::vector<std::uint64_t>* released);

  /// @brief Take the newest trigger ready to hand out.
  ///
  /// A frame earlier than the tolerance allows of the earliest anchor frame
  /// held is near no trigger still to come -- its set already went out, the
  /// anchor missed it, or its sensor's clock is off -- so it is let go, and
  /// such a sensor costs its own frames, not the rig's sets.
  /// @param now_us    The host time now (us).
  /// @param released  Receives the ids let go: those, and every older ready
  ///                  trigger's.
  /// @return The trigger: an anchor frame, and each other sensor's frame
  ///         within the tolerance of it; empty while none is ready.
  std::optional<Group> take(std::uint64_t now_us,
                            std::vector<std::uint64_t>* released);

  /// @brief Forget every frame held.
  /// @param released  Receives their ids.
  void clear(std::vector<std::uint64_t>* released);

 private:
  struct Entry {
    std::uint64_t ts_us;
    std::uint64_t id;
    std::uint64_t arrived_us;
  };
  Config config_;
  std::vector<std::deque<Entry>> queues_;
};

}  // namespace volumetric_kit::recon::sensor
