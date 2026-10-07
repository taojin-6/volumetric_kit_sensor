// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "trigger_grouper.hpp"

#include <algorithm>
#include <iterator>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::recon::sensor {

TriggerGrouper::TriggerGrouper(const Config& config)
    : config_(config), queues_(config.cameras) {
  VKC_CHECK(config.anchor < config.cameras,
            "TriggerGrouper: the anchor must be one of the cameras");
  VKC_CHECK(config.queue_depth > 0,
            "TriggerGrouper: a queue holds at least one frame");
}

void TriggerGrouper::add(std::size_t camera, std::uint64_t ts_us,
                         std::uint64_t id, std::uint64_t now_us,
                         std::vector<std::uint64_t>* released) {
  VKC_CHECK(camera < queues_.size(), "TriggerGrouper::add: no such camera");
  std::deque<Entry>& q = queues_[camera];
  // Kept in timestamp order: a camera's clock can step back.
  auto at = q.end();
  while (at != q.begin() && std::prev(at)->ts_us > ts_us) --at;
  q.insert(at, Entry{ts_us, id, now_us});
  while (q.size() > config_.queue_depth) {
    released->push_back(q.front().id);
    q.pop_front();
  }
}

std::optional<TriggerGrouper::Group> TriggerGrouper::take(
    std::uint64_t now_us, std::vector<std::uint64_t>* released) {
  std::deque<Entry>& anchor = queues_[config_.anchor];
  std::optional<Group> newest;
  while (!anchor.empty()) {
    // The earliest anchor frame held names a trigger; each other camera's
    // front frame within the tolerance of it is that camera's frame for it.
    const Entry& trigger = anchor.front();
    Group group;
    group.timestamp_us = trigger.ts_us;
    group.ids.resize(queues_.size());
    group.ids[config_.anchor] = trigger.id;
    std::uint64_t first_arrival = trigger.arrived_us;
    bool waiting = false;
    for (std::size_t c = 0; c < queues_.size(); ++c) {
      if (c == config_.anchor) continue;
      std::deque<Entry>& q = queues_[c];
      // Earlier than the tolerance allows: near no anchor frame still to come
      // -- its set already went out, the anchor missed that trigger, or this
      // camera's clock is off.
      while (!q.empty() &&
             q.front().ts_us + config_.tolerance_us < trigger.ts_us) {
        released->push_back(q.front().id);
        q.pop_front();
      }
      if (q.empty()) {
        // Silent: its frame for this trigger may still be on the way.
        waiting = true;
      } else if (q.front().ts_us <= trigger.ts_us + config_.tolerance_us) {
        group.ids[c] = q.front().id;
        first_arrival = std::min(first_arrival, q.front().arrived_us);
      }
      // Otherwise it holds a later frame: past the trigger, it will never
      // send one for it.
    }
    const std::uint64_t waited =
        now_us > first_arrival ? now_us - first_arrival : 0;
    if (waiting && waited < config_.max_wait_us) break;
    for (std::size_t c = 0; c < queues_.size(); ++c) {
      if (group.ids[c]) queues_[c].pop_front();
    }
    if (newest) {
      for (const auto& id : newest->ids) {
        if (id) released->push_back(*id);
      }
    }
    newest = std::move(group);
  }
  return newest;
}

void TriggerGrouper::clear(std::vector<std::uint64_t>* released) {
  for (auto& q : queues_) {
    for (const Entry& e : q) released->push_back(e.id);
    q.clear();
  }
}

}  // namespace volumetric_kit::recon::sensor
