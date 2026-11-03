// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#pragma once

// Internal (not installed): where a DeckLink pair's pictures go, with no SDK
// type, so a test writes the pages as the card would on every Vulkan leg.
//
// - PagePool: an input's host pages, which the card's DMA writes through the
//   SDK's capture allocator and the device imports
//   (core::create_imported_host_buffer): made on demand, at most 8, each
//   found again by its address when its frame arrives.
// - DevicePictures: the device buffers the pictures are copied into, back
//   in the pool once no frame holds them, so none is made after warm-up.
// - Uploader: the upload thread. Each pair's two pictures are copied into
//   device buffers in one batch; the SDK's frames go back once its fence has
//   signalled, never before, as the card may write those pages again; the
//   pair is published, the newest winning, and its consumer woken.

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "pairing.hpp"
#include "volumetric_kit/core/base/result.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/core/vulkan/fwd.hpp"
#include "volumetric_kit/sensor/stereo/stereo_sensor.hpp"

namespace volumetric_kit::sensor::decklink {

class PagePool {
 public:
  static constexpr std::size_t kMaxBuffers = 8;

  // Unsupported, naming it, where `device` did not enable
  // VK_EXT_external_memory_host (sensor::device_requirements asks for it).
  static core::Result<std::shared_ptr<PagePool>> create(
      const core::Device& device, core::Allocator& allocator,
      std::size_t max_buffers = kMaxBuffers);

  PagePool(const PagePool&) = delete;
  PagePool& operator=(const PagePool&) = delete;
  PagePool(PagePool&&) = delete;
  PagePool& operator=(PagePool&&) = delete;
  ~PagePool();

  // Every buffer's size: the SDK's bufferSize, which may exceed a picture.
  // Set before the first take; again only to the same size.
  core::Status set_size(std::uint64_t bytes);
  std::uint64_t size() const;
  // A free buffer's pages for the card to write: one made when none is free
  // and fewer than the cap exist; null past the cap (the SDK's
  // E_OUTOFMEMORY), or the failure making one.
  core::Result<std::uint8_t*> take();
  // The buffer whose pages start at `data`, free again: the SDK let it go.
  void give_back(const void* data) noexcept;
  // The buffer whose pages start at `data`, to copy a picture from; null
  // for an address that is none of them.
  const core::Buffer* find(const void* data) const;
  std::size_t made() const;    // buffers made since create: the high mark
  std::size_t in_use() const;  // taken and not given back

 private:
  struct Entry {
    core::ImportedHostBuffer pages;
    bool in_use = false;
  };
  PagePool(const core::Device& device, core::Allocator& allocator,
           std::size_t max_buffers);

  const core::Device* device_;
  core::Allocator* allocator_;
  const std::size_t max_buffers_;
  mutable std::mutex mutex_;
  std::uint64_t size_ = 0;                       // guarded by mutex_
  std::vector<std::unique_ptr<Entry>> entries_;  // guarded by mutex_
};

class DevicePictures : public std::enable_shared_from_this<DevicePictures> {
 public:
  // The pairs in flight -- copying, ready, rectified, queued for the depth
  // source, in prep -- need about 11; past 24 a pair
  // fails rather than the device filling.
  static constexpr std::size_t kMaxBuffers = 24;

  static std::shared_ptr<DevicePictures> create(
      core::Allocator& allocator, std::uint64_t bytes,
      std::size_t max_buffers = kMaxBuffers);

  DevicePictures(const DevicePictures&) = delete;
  DevicePictures& operator=(const DevicePictures&) = delete;
  DevicePictures(DevicePictures&&) = delete;
  DevicePictures& operator=(DevicePictures&&) = delete;
  ~DevicePictures();

  // A free buffer (device-only; storage, copied into and out of), made when
  // none is free and fewer than the cap exist, back in the pool once its
  // last copy goes; null past the cap.
  core::Result<std::shared_ptr<const core::Buffer>> take();
  std::size_t made() const;
  std::size_t in_use() const;

 private:
  struct Entry {
    core::Buffer buffer;
    bool in_use = false;
  };
  DevicePictures(core::Allocator& allocator, std::uint64_t bytes,
                 std::size_t max_buffers);
  void give_back(Entry* entry) noexcept;

  core::Allocator* allocator_;
  const std::uint64_t bytes_;
  const std::size_t max_buffers_;
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Entry>> entries_;  // guarded by mutex_
};

// The pictures' shape: V210 at width x height, row_bytes a row.
struct Picture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_bytes = 0;
};

class Uploader {
 public:
  Uploader(const core::Device& device, core::Allocator& allocator,
           std::array<std::shared_ptr<PagePool>, 2> pages,
           std::shared_ptr<DevicePictures> pictures, Picture picture);
  Uploader(const Uploader&) = delete;
  Uploader& operator=(const Uploader&) = delete;
  Uploader(Uploader&&) = delete;
  Uploader& operator=(Uploader&&) = delete;
  ~Uploader();

  // The thread started, the counts and any failure cleared; `ready` is
  // called on it after each pair is published, and on a failure.
  void start(std::function<void()> ready);
  // The thread joined; the pair waiting and the pair ready let go.
  void stop() noexcept;

  // A pair in, from a callback thread: it waits for the thread, replacing
  // one still waiting (counted dropped).
  void submit(Pair pair);
  // The newest published pair, older ones counted dropped; empty when none;
  // the failure, once there is one.
  core::Result<std::optional<StereoFrame>> take();

  std::uint64_t dropped() const;
  std::uint64_t failed() const;  // pairs with no device buffer free
  std::uint64_t published() const;
  core::Status fault() const;  // OK until a copy or the pages failed

 private:
  void run();
  void upload(Pair pair);
  void fail(core::Status why);

  const core::Device* device_;
  core::Allocator* allocator_;
  const std::array<std::shared_ptr<PagePool>, 2> pages_;
  const std::shared_ptr<DevicePictures> pictures_;
  const Picture picture_;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;             // guarded by mutex_
  std::optional<Pair> waiting_;       // guarded by mutex_
  std::optional<StereoFrame> ready_;  // guarded by mutex_
  core::Status fault_;                // guarded by mutex_
  std::function<void()> notify_;      // guarded by mutex_
  std::uint64_t dropped_ = 0;         // guarded by mutex_
  std::uint64_t failed_ = 0;          // guarded by mutex_
  std::uint64_t published_ = 0;       // guarded by mutex_
  std::thread thread_;
};

}  // namespace volumetric_kit::sensor::decklink
