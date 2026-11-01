// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "upload.hpp"

#include <string>
#include <utility>

#include "volumetric_kit/core/vulkan/allocator.hpp"
#include "volumetric_kit/core/vulkan/command_batch.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/frame/color_space.hpp"
#include "volumetric_kit/sensor/yuv_image.hpp"

namespace volumetric_kit::sensor::decklink {

// --- PagePool ----------------------------------------------------------------

PagePool::PagePool(const core::Device& device, core::Allocator& allocator,
                   std::size_t max_buffers)
    : device_(&device), allocator_(&allocator), max_buffers_(max_buffers) {}

PagePool::~PagePool() = default;

core::Result<std::shared_ptr<PagePool>> PagePool::create(
    const core::Device& device, core::Allocator& allocator,
    std::size_t max_buffers) {
  if (!device.extension_enabled(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
    return core::Status::unsupported(
        "DecklinkStereo: the device did not enable "
        "VK_EXT_external_memory_host, which the card's pictures land in "
        "(sensor::device_requirements() asks for it where offered)");
  }
  return std::shared_ptr<PagePool>(
      new PagePool(device, allocator, max_buffers));
}

core::Status PagePool::set_size(std::uint64_t bytes) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (bytes == 0) {
    return core::Status::invalid_argument("DecklinkStereo: a 0-byte picture");
  }
  if (size_ != 0 && size_ != bytes && !entries_.empty()) {
    return core::Status::invalid_argument(
        "DecklinkStereo: the card asked for " + std::to_string(bytes) +
        "-byte pictures after " + std::to_string(size_) + "-byte ones");
  }
  size_ = bytes;
  return {};
}

std::uint64_t PagePool::size() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return size_;
}

core::Result<std::uint8_t*> PagePool::take() {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const std::unique_ptr<Entry>& e : entries_) {
    if (!e->in_use) {
      e->in_use = true;
      return e->pages.data;
    }
  }
  if (entries_.size() >= max_buffers_) return nullptr;
  if (size_ == 0) {
    return core::Status::invalid_argument(
        "DecklinkStereo: pages taken before their size was set");
  }
  VKC_ASSIGN(core::ImportedHostBuffer pages,
             core::create_imported_host_buffer(*device_, *allocator_, size_));
  auto entry = std::make_unique<Entry>();
  entry->pages = std::move(pages);
  entry->in_use = true;
  std::uint8_t* data = entry->pages.data;
  entries_.push_back(std::move(entry));
  return data;
}

void PagePool::give_back(const void* data) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const std::unique_ptr<Entry>& e : entries_) {
    if (e->pages.data == data) e->in_use = false;
  }
}

const core::Buffer* PagePool::find(const void* data) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const std::unique_ptr<Entry>& e : entries_) {
    if (e->pages.data == data) return &e->pages.buffer;
  }
  return nullptr;
}

std::size_t PagePool::made() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

std::size_t PagePool::in_use() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::size_t n = 0;
  for (const std::unique_ptr<Entry>& e : entries_) n += e->in_use ? 1 : 0;
  return n;
}

// --- DevicePictures ----------------------------------------------------------

DevicePictures::DevicePictures(core::Allocator& allocator, std::uint64_t bytes,
                               std::size_t max_buffers)
    : allocator_(&allocator), bytes_(bytes), max_buffers_(max_buffers) {}

DevicePictures::~DevicePictures() = default;

std::shared_ptr<DevicePictures> DevicePictures::create(
    core::Allocator& allocator, std::uint64_t bytes, std::size_t max_buffers) {
  return std::shared_ptr<DevicePictures>(
      new DevicePictures(allocator, bytes, max_buffers));
}

core::Result<std::shared_ptr<const core::Buffer>> DevicePictures::take() {
  const std::lock_guard<std::mutex> lock(mutex_);
  Entry* entry = nullptr;
  for (const std::unique_ptr<Entry>& e : entries_) {
    if (!e->in_use) {
      entry = e.get();
      break;
    }
  }
  if (entry == nullptr) {
    if (entries_.size() >= max_buffers_) {
      return std::shared_ptr<const core::Buffer>{};
    }
    core::BufferDesc desc;
    desc.size = bytes_;
    // Read by the rectifier's and prep's kernels; copied out by a readback.
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VKC_ASSIGN(core::Buffer buffer, allocator_->create_buffer(desc));
    auto made = std::make_unique<Entry>();
    made->buffer = std::move(buffer);
    entry = made.get();
    entries_.push_back(std::move(made));
  }
  entry->in_use = true;
  // The deleter keeps the pool alive while a frame holds its buffer.
  return std::shared_ptr<const core::Buffer>(
      &entry->buffer,
      [pool = shared_from_this(), entry](const core::Buffer*) noexcept {
        pool->give_back(entry);
      });
}

void DevicePictures::give_back(Entry* entry) noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  entry->in_use = false;
}

std::size_t DevicePictures::made() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

std::size_t DevicePictures::in_use() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  std::size_t n = 0;
  for (const std::unique_ptr<Entry>& e : entries_) n += e->in_use ? 1 : 0;
  return n;
}

// --- Uploader ----------------------------------------------------------------

namespace {

// A picture as the driver hands it out: V210 in its device buffer, BT.709
// limited range, chroma co-sited, written on the device's own family by a
// copy the host waited for.
YuvImage picture_of(std::shared_ptr<const core::Buffer> buffer,
                    const Picture& picture) {
  YuvImage image;
  image.layout = YuvLayout::V210;
  image.device = std::move(buffer);
  image.offset[0] = 0;
  image.stride[0] = picture.row_bytes;
  image.queue_family = kQueueFamilyIgnored;
  image.width = picture.width;
  image.height = picture.height;
  const YcbcrWeights bt709 = ycbcr_weights(VideoColorMatrix::Bt709);
  image.kr = bt709.kr;
  image.kb = bt709.kb;
  image.full_range = false;
  image.chroma_location = ChromaLocation::Left;
  return image;
}

}  // namespace

Uploader::Uploader(const core::Device& device, core::Allocator& allocator,
                   std::array<std::shared_ptr<PagePool>, 2> pages,
                   std::shared_ptr<DevicePictures> pictures, Picture picture)
    : device_(&device),
      allocator_(&allocator),
      pages_(std::move(pages)),
      pictures_(std::move(pictures)),
      picture_(picture) {}

Uploader::~Uploader() { stop(); }

void Uploader::start(std::function<void()> ready) {
  stop();
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = false;
    fault_ = {};
    notify_ = std::move(ready);
    dropped_ = failed_ = published_ = 0;
  }
  thread_ = std::thread([this] { run(); });
}

void Uploader::stop() noexcept {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  if (thread_.joinable()) thread_.join();
  std::optional<Pair> waiting;
  std::optional<StereoFrame> ready;
  const std::lock_guard<std::mutex> lock(mutex_);
  waiting = std::move(waiting_);
  waiting_.reset();
  ready = std::move(ready_);
  ready_.reset();
}

void Uploader::submit(Pair pair) {
  std::optional<Pair> replaced;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || !fault_.ok()) return;  // `pair` goes back to the SDK
    if (waiting_) {
      replaced = std::move(waiting_);
      ++dropped_;
    }
    waiting_ = std::move(pair);
  }
  wake_.notify_one();
}

core::Result<std::optional<StereoFrame>> Uploader::take() {
  std::optional<StereoFrame> frame;
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!fault_.ok()) return fault_;
  frame = std::move(ready_);
  ready_.reset();
  return frame;
}

std::uint64_t Uploader::dropped() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return dropped_;
}

std::uint64_t Uploader::failed() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return failed_;
}

std::uint64_t Uploader::published() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return published_;
}

core::Status Uploader::fault() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return fault_;
}

void Uploader::run() {
  for (;;) {
    std::optional<Pair> pair;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [this] { return stopping_ || waiting_.has_value(); });
      if (stopping_) return;
      pair = std::move(waiting_);
      waiting_.reset();
    }
    upload(std::move(*pair));
  }
}

void Uploader::fail(core::Status why) {
  std::function<void()> notify;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (fault_.ok()) fault_ = std::move(why);
    notify = notify_;
  }
  if (notify) notify();
}

void Uploader::upload(Pair pair) {
  const VkDeviceSize bytes =
      static_cast<VkDeviceSize>(picture_.row_bytes) * picture_.height;
  const Arrival* sides[2] = {&pair.left, &pair.right};
  const core::Buffer* source[2] = {};
  for (int i = 0; i < 2; ++i) {
    source[i] = pages_[i]->find(sides[i]->bytes);
    if (source[i] == nullptr || source[i]->size() < bytes) {
      fail(core::Status::io_error(
          "DecklinkStereo: the card wrote a picture outside the driver's "
          "pages, or into pages smaller than a picture"));
      return;
    }
  }
  std::shared_ptr<const core::Buffer> target[2];
  for (int i = 0; i < 2; ++i) {
    core::Result<std::shared_ptr<const core::Buffer>> taken = pictures_->take();
    if (!taken.ok()) {
      fail(taken.status().with_context("DecklinkStereo: a device picture"));
      return;
    }
    target[i] = std::move(*taken);
  }
  if (target[0] == nullptr || target[1] == nullptr) {
    // Every device picture is held: the pair fails, its frames go back.
    const std::lock_guard<std::mutex> lock(mutex_);
    ++failed_;
    return;
  }

  core::CommandBatch batch(*device_, *allocator_);
  core::Status copied = batch.copy(*source[0], 0, *target[0], 0, bytes);
  if (copied.ok()) copied = batch.copy(*source[1], 0, *target[1], 0, bytes);
  if (copied.ok()) copied = batch.submit();  // waits for the copy's fence
  if (!copied.ok()) {
    // The copy may still run: its pages stay out of the card's reach.
    lost_.push_back(std::move(pair));
    fail(copied.with_context("DecklinkStereo: copying a pair to the device"));
    return;
  }
  // The copy is done: the card may write these pages again.
  pair.left.hold.reset();
  pair.right.hold.reset();

  StereoFrame frame;
  frame.left = picture_of(std::move(target[0]), picture_);
  frame.right = picture_of(std::move(target[1]), picture_);
  frame.color_encoding = ColorEncoding{ColorEncoding::Transfer::Bt709,
                                       ColorEncoding::Primaries::Bt709};
  frame.timestamp_ns = pair.timestamp_ns;
  frame.skew_ns = pair.skew_ns;
  frame.sequence = pair.sequence;
  frame.arrival_ns = pair.arrival_ns;

  std::optional<StereoFrame> replaced;
  std::function<void()> notify;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) {
      replaced = std::move(ready_);
      ++dropped_;
    }
    ready_ = std::move(frame);
    ++published_;
    notify = notify_;
  }
  if (notify) notify();
}

}  // namespace volumetric_kit::sensor::decklink
