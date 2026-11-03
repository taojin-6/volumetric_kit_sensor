// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HevcEncoder's rules, every backend's: the argument checks, the colour set,
// the key owed after a failure, the lost session opened again, the faults,
// the target's description, and the coding thread, which waits for each
// target's writer, with its sends and receive. The backend's session is
// hevc_encode_session.hpp's.

#include "volumetric_kit/sensor/video/hevc_encoder.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

#include "frame_color.hpp"
#include "hevc_encode_session.hpp"
#include "hevc_encoder_access.hpp"
#include "volumetric_kit/core/base/check.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/vk_result.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"
#include "volumetric_kit/sensor/video/backend.hpp"
#include "volumetric_kit/sensor/video/frame_color.hpp"

namespace volumetric_kit::sensor {
namespace {

constexpr const char* kWho = "HevcEncoder";
constexpr int kMaxQp = 51;

core::Status invalid(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

core::Status moved_from() { return invalid("moved from"); }

// What create refuses before any backend opens: InvalidArgument.
std::string bad_options(const HevcEncoder::Options& o) {
  const auto size = std::to_string(o.width) + "x" + std::to_string(o.height);
  if (o.width == 0 || o.height == 0 || (o.width & 1) != 0 ||
      (o.height & 1) != 0) {
    return "a size of " + size + ": each side must be even and not 0";
  }
  if (o.width > HevcEncoder::kMaxSide || o.height > HevcEncoder::kMaxSide) {
    return "a size of " + size + ": a side past " +
           std::to_string(HevcEncoder::kMaxSide);
  }
  if (std::uint64_t{o.width} * o.height > HevcEncoder::kMaxLumaSamples) {
    return "a size of " + size + ": more than " +
           std::to_string(HevcEncoder::kMaxLumaSamples) +
           " luma samples (H.265 level 6.2)";
  }
  if (o.qp < 0 || o.qp > kMaxQp) {
    return "QP " + std::to_string(o.qp) + " is not 0 to 51";
  }
  if (o.key_qp && (*o.key_qp < 0 || *o.key_qp > kMaxQp)) {
    return "key QP " + std::to_string(*o.key_qp) + " is not 0 to 51";
  }
  if (o.effort < 1 || o.effort > HevcEncoder::kMaxEffort) {
    return "effort " + std::to_string(o.effort) + " is not 1 to " +
           std::to_string(HevcEncoder::kMaxEffort);
  }
  if (o.color.matrix == 2) {
    return "the matrix is unspecified (2): state the pictures' own";
  }
  return {};
}

// A picture's failure: IoError saying @p why, or OutOfMemory where the
// message cannot be made, so the coding thread never throws.
core::Status io_failure(const char* why) noexcept {
  try {
    return core::Status::io_error(std::string(kWho) + ": " + why);
  } catch (...) {
    return core::exception_status(kWho, core::Status::Code::OutOfMemory);
  }
}

// Waits on the host for @p written, with no limit, as a pending batch does:
// OK once it is reached, or for a null semaphore; else the wait's failure, as
// for a lost device. Vulkan's own call: the coding thread allocates nothing
// but that failure's message.
core::Status wait_for_writer(const core::TimelinePoint& written) noexcept {
  if (written.semaphore == nullptr) return {};
  const VkSemaphore semaphore = written.semaphore->handle();
  VkSemaphoreWaitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore;
  info.pValues = &written.value;
  const VkResult waited =
      vkWaitSemaphores(written.semaphore->device(), &info, UINT64_MAX);
  if (waited == VK_SUCCESS) return {};
  return core::vk_error(waited,
                        "HevcEncoder: waiting for a target's writer "
                        "(vkWaitSemaphores)");
}

// First in, first out, held in place: at most HevcEncoder::kInFlight
// entries, which the in-flight count bounds, and no allocation.
template <typename T>
class Ring {
  static_assert(std::is_nothrow_move_constructible_v<T> &&
                std::is_nothrow_move_assignable_v<T>);

 public:
  bool empty() const noexcept { return size_ == 0; }
  void push(T&& entry) noexcept {
    VKC_CHECK(size_ < kCapacity, "HevcEncoder: more than kInFlight queued");
    entries_[(head_ + size_) % kCapacity] = std::move(entry);
    ++size_;
  }
  T take() noexcept {
    T entry = std::move(entries_[head_]);
    head_ = (head_ + 1) % kCapacity;
    --size_;
    return entry;
  }

 private:
  static constexpr std::uint32_t kCapacity = HevcEncoder::kInFlight;
  std::array<T, kCapacity> entries_{};
  std::uint32_t head_ = 0;
  std::uint32_t size_ = 0;
};

}  // namespace

core::DeviceRequirements hevc_encoder_device_requirements() {
  core::DeviceRequirements reqs = device_requirements();
  // r8 and rg8 storage images: the target's planes, as a kernel writes them.
  reqs.features.shaderStorageImageExtendedFormats = VK_TRUE;
  return reqs;
}

VideoBackend video_backend() noexcept {
#if defined(__APPLE__)
  return VideoBackend::VideoToolbox;
#else
  return VideoBackend::Nvidia;
#endif
}

struct HevcEncoder::Impl {
  // A picture sent, waiting for the coding thread.
  struct Sent {
    std::uint64_t seq = 0;
    video::EncodePicture picture;
    bool key = false;
    std::int64_t pts = 0;
  };
  // A picture coded, or failed, waiting for receive.
  struct Coded {
    std::uint64_t seq = 0;
    core::Status status;
    std::vector<std::uint8_t> bytes;
    std::int64_t pts = 0;
    bool key = false;
  };

  Options options;  // as create was given them, colour normalized
  Facts facts;
  // The coding thread codes on it; the sender replaces it, and inject loses
  // it, only while the thread is not coding (coding false, under mutex). The
  // sender's target reaches it unlocked.
  std::unique_ptr<video::HevcEncodeSession> session;

  std::mutex mutex;
  std::condition_variable work;  // the coding thread waits for a picture
  std::condition_variable done;  // receive waits for a picture coded
  std::condition_variable idle;  // a reopen waits for coding to end
  // The target last handed out and not yet sent: the sender's alone.
  std::optional<video::EncodePicture> taken;
  Ring<Sent> sent;    // sent, not yet taken by the thread
  Ring<Coded> coded;  // coded or failed, not yet received
  // Buffers to code into: at most kInFlight, as each is in a picture
  // coding, coded or here, so create reserves them and nothing allocates.
  std::vector<std::vector<std::uint8_t>> spare;
  bool coding = false;  // the thread is inside the session
  bool stop = false;
  bool drop_next = false;     // Fault::Drop, for the next picture coded
  bool lose_pending = false;  // Fault::Lose, before the next picture coded
  std::uint64_t next_seq = 1;
  std::uint64_t fail_through = 0;  // every picture up to it fails uncoded
  std::atomic<std::uint32_t> in_flight{0};
  std::atomic<bool> key_owed{true};
  std::atomic<bool> lost{false};
  std::thread thread;

  // The coding thread. It never throws: its rings and buffers are made at
  // create, and each allocation the backend makes in it can fail, as that
  // picture's OutOfMemory.
  void run() noexcept;
  // A picture failed or was lost: a key is owed, and each picture sent so
  // far, which would follow the gap, fails too. Under mutex.
  void fail() {
    key_owed = true;
    fail_through = next_seq - 1;
  }
  // @p picture queued for the coding thread, moved from only once it is.
  core::Status send(video::EncodePicture&& picture, bool key, std::int64_t pts);
  // A lost session closed and opened afresh. Under mutex, through @p lock,
  // which it lets go while the backend opens.
  core::Status reopen(std::unique_lock<std::mutex>& lock);
};

void HevcEncoder::Impl::run() noexcept {
#if defined(__APPLE__)
  // The hardware's work runs on this thread, and it is on the live path: at
  // user-interactive quality of service, as a camera's delivery queue, not
  // the DEFAULT a std::thread gets, which a busy device runs behind.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  pthread_setname_np("HevcEncoder");
#endif
  std::unique_lock<std::mutex> lock(mutex);
  for (;;) {
    work.wait(lock, [this] { return stop || !sent.empty(); });
    if (stop) {
      // Each picture still sent is dropped uncoded, once its writer is done:
      // a batch may still be writing its target, whose images the session
      // holds.
      while (!sent.empty()) {
        Sent s = sent.take();
        lock.unlock();
        static_cast<void>(wait_for_writer(s.picture.written));
        s = Sent{};
        lock.lock();
      }
      return;
    }
    Sent s = sent.take();
    Coded c;
    c.seq = s.seq;
    c.pts = s.pts;
    if (!spare.empty()) {
      c.bytes = std::move(spare.back());
      spare.pop_back();
    }
    // A target's writer first, unlocked, whatever comes of the picture: a
    // loss or a reopen meanwhile fails it below, and a stop drops it.
    core::Status writer;
    if (s.picture.written.semaphore != nullptr) {
      lock.unlock();
      writer = wait_for_writer(s.picture.written);
      lock.lock();
      if (stop) continue;  // dropped, the rest once their writers are done
    }
    if (s.seq <= fail_through) {
      c.status = io_failure(
          "a picture sent before this one failed, or the session was lost, "
          "so this one would follow a gap");
    } else if (!writer.ok()) {
      c.status = std::move(writer);
      fail();
    } else {
      if (lose_pending) {
        lose_pending = false;
        session->lose();
      }
      const bool drop = std::exchange(drop_next, false);
      coding = true;
      lock.unlock();
      bool idr = false;
      core::Result<bool> got =
          session->encode(s.picture, s.key, drop, c.bytes, idr);
      s.picture = video::EncodePicture{};  // the buffer let go, unlocked
      lock.lock();
      coding = false;
      idle.notify_all();
      if (!got.ok()) {
        c.status = std::move(got).status();
        lost = true;
        fail();
      } else if (!*got) {
        c.status = io_failure("the hardware encoder dropped the picture");
        fail();
      } else {
        c.key = idr;
      }
    }
    if (!c.status.ok()) c.bytes.clear();
    coded.push(std::move(c));
    done.notify_all();
  }
}

core::Status HevcEncoder::Impl::reopen(std::unique_lock<std::mutex>& lock) {
  // Whatever is still queued was sent before the loss: it fails uncoded.
  fail_through = next_seq - 1;
  lose_pending = false;
  idle.wait(lock, [this] { return !coding; });
  // Unlocked, the coding thread does not touch the session: it codes only
  // what is sent after this. The new one closes the old one's hardware
  // session first, and takes over its targets' images once it opens: a
  // batch may still be writing one. Failing, it leaves the old one, closed,
  // with them, for the next try.
  lock.unlock();
  Facts opened;
  core::Result<std::unique_ptr<video::HevcEncodeSession>> made =
      video::HevcEncodeSession::open(options, opened, session.get());
  lock.lock();
  VKC_TRY(made.status());
  session = std::move(made).value();
  facts = std::move(opened);
  lost = false;
  return {};
}

core::Status HevcEncoder::Impl::send(video::EncodePicture&& picture, bool key,
                                     std::int64_t pts) {
  std::unique_lock<std::mutex> lock(mutex);
  if (in_flight >= kInFlight) {
    return invalid(std::to_string(kInFlight) +
                   " pictures are in flight already: receive first");
  }
  if (key_owed && !key) {
    return invalid(
        "a key is owed (the first picture, or the next after a failed, "
        "dropped or lost one)");
  }
  if (lost) VKC_TRY(reopen(lock));
  Sent s;
  s.seq = next_seq++;
  s.picture = std::move(picture);
  s.key = key;
  s.pts = pts;
  sent.push(std::move(s));
  ++in_flight;
  if (key) key_owed = false;
  work.notify_one();
  return {};
}

core::Result<HevcEncoder> HevcEncoder::create(const Options& options) try {
  if (const std::string bad = bad_options(options); !bad.empty()) {
    return invalid(bad);
  }
  auto impl = std::make_unique<Impl>();
  impl->spare.reserve(kInFlight);
  impl->options = options;
  impl->options.color = video::normalize_codes(options.color);
  if (const std::string outside = video::outside_code_set(
          impl->options.color, video::HevcEncodeSession::kWritesSiting);
      !outside.empty()) {
    return core::Status::unsupported(
        std::string(kWho) + ": " + outside +
        " is not one this encoder codes (HevcEncoder::Options::color)");
  }
  if (options.device != nullptr) {
    VKC_TRY(options.device->check_enabled(hevc_encoder_device_requirements())
                .with_context("HevcEncoder::create"));
  }
  VKC_ASSIGN(impl->session,
             video::HevcEncodeSession::open(impl->options, impl->facts));
  Impl* const self = impl.get();
  impl->thread = std::thread([self] { self->run(); });
  return HevcEncoder(std::move(impl));
} catch (...) {
  return core::exception_status("HevcEncoder::create",
                                core::Status::Code::OutOfMemory);
}

HevcEncoder::HevcEncoder(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
HevcEncoder::HevcEncoder(HevcEncoder&& other) noexcept = default;

HevcEncoder& HevcEncoder::operator=(HevcEncoder&& other) noexcept {
  if (this != &other) {
    HevcEncoder gone(std::move(*this));  // stopped as it goes
    impl_ = std::move(other.impl_);
  }
  return *this;
}

HevcEncoder::~HevcEncoder() {
  if (impl_ == nullptr) return;
  {
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stop = true;
  }
  impl_->work.notify_all();
  if (impl_->thread.joinable()) impl_->thread.join();
}

core::Result<YuvImage> HevcEncoder::target(core::CommandBatch& batch) try {
  if (impl_ == nullptr) return moved_from();
  Impl& m = *impl_;
  if (m.options.device == nullptr) {
    return invalid(
        "target: the session was opened with no device (Options::device)");
  }
  std::unique_lock<std::mutex> lock(m.mutex);
  if (m.in_flight >= kInFlight) {
    return invalid("target: " + std::to_string(kInFlight) +
                   " pictures are in flight already: receive first");
  }
  if (m.lost) VKC_TRY(m.reopen(lock));
  // Unlocked: a picture new to the session waits on the host for the
  // device's earlier work, which must hold up neither receive nor the coding
  // thread. Only this thread replaces the session or loses it here: a loss
  // still pending for the coding thread (lose_pending) came with lost, and
  // the reopen above ended it.
  video::HevcEncodeSession& session = *m.session;
  lock.unlock();
  video::EncodePicture picture;
  YuvImage out;
  bool gone = false;
  if (core::Status made = session.target(batch, picture, out, gone);
      !made.ok()) {
    if (gone) {
      // As when a picture's coding finds it lost: the next target reopens.
      lock.lock();
      m.lost = true;
      m.key_owed = true;
    }
    return made;  // a target taken before is kept
  }
  // Coded by the session's colour, which the VUI states.
  const ColorCodes& color = m.facts.color;
  const YcbcrWeights weights =
      ycbcr_weights(resolve_matrix(color.matrix, m.options.height));
  out.kr = weights.kr;
  out.kb = weights.kb;
  out.full_range = color.full_range;
  out.chroma_location = resolve_chroma_location(color.chroma_location);
  out.queue_family = kQueueFamilyIgnored;
  m.taken = std::move(picture);  // a second call gives up the first
  return out;
} catch (...) {
  return core::exception_status("HevcEncoder::target",
                                core::Status::Code::OutOfMemory);
}

core::Status HevcEncoder::send(bool key, std::int64_t pts,
                               core::TimelinePoint written) try {
  if (impl_ == nullptr) return moved_from();
  Impl& m = *impl_;
  if (!m.taken) return invalid("send: no target taken (target() first)");
  if (written.semaphore != nullptr && !written.semaphore->valid()) {
    return invalid("send: an empty semaphore for the writer's value");
  }
  m.taken->written = written;
  VKC_TRY(m.send(std::move(*m.taken), key, pts));
  m.taken.reset();
  return {};
} catch (...) {
  return core::exception_status("HevcEncoder: send",
                                core::Status::Code::OutOfMemory);
}

core::Result<bool> HevcEncoder::receive(CodedPicture& picture,
                                        std::uint64_t timeout_ns) try {
  if (impl_ == nullptr) return moved_from();
  Impl& m = *impl_;
  std::unique_lock<std::mutex> lock(m.mutex);
  if (m.coded.empty()) {
    // A poll returns at once, and so does a wait with no end with nothing in
    // flight, which on one thread would never end. A finite wait waits even
    // then, for a picture another thread sends meanwhile.
    if (timeout_ns == 0 || (timeout_ns == UINT64_MAX && m.in_flight == 0)) {
      return false;
    }
    const auto ready = [&m] { return !m.coded.empty(); };
    if (timeout_ns == UINT64_MAX) {
      m.done.wait(lock, ready);
    } else {
      const std::uint64_t most = std::chrono::nanoseconds::max().count();
      m.done.wait_for(
          lock, std::chrono::nanoseconds(std::min(timeout_ns, most)), ready);
    }
    if (m.coded.empty()) return false;
  }
  Impl::Coded c = m.coded.take();
  --m.in_flight;
  // Each buffer goes back within the capacity create reserved.
  if (!c.status.ok()) {
    m.spare.push_back(std::move(c.bytes));
    return std::move(c.status);
  }
  picture.access_unit.swap(c.bytes);
  picture.pts = c.pts;
  picture.key = c.key;
  c.bytes.clear();
  m.spare.push_back(std::move(c.bytes));  // the caller's old buffer
  return true;
} catch (...) {
  return core::exception_status("HevcEncoder::receive",
                                core::Status::Code::OutOfMemory);
}

std::uint32_t HevcEncoder::in_flight() const noexcept {
  return impl_ != nullptr ? impl_->in_flight.load() : 0;
}

bool HevcEncoder::key_owed() const noexcept {
  return impl_ == nullptr || impl_->key_owed.load();
}

bool HevcEncoder::lost() const noexcept {
  return impl_ != nullptr && impl_->lost.load();
}

const HevcEncoder::Facts& HevcEncoder::facts() const noexcept {
  static const Facts kNone;
  return impl_ != nullptr ? impl_->facts : kNone;
}

void HevcEncoder::inject(Fault fault) noexcept {
  if (impl_ == nullptr) return;
  Impl& m = *impl_;
  const std::lock_guard<std::mutex> lock(m.mutex);
  if (fault == Fault::Drop) {
    m.drop_next = true;
    return;
  }
  m.lost = true;
  m.key_owed = true;
  if (m.session == nullptr) return;
  // Idle, it is lost at once; coding, before the next picture it codes.
  if (!m.coding && m.sent.empty()) {
    m.session->lose();
  } else {
    m.lose_pending = true;
  }
}

const HevcEncoder::Options* HevcEncoderAccess::options(
    const HevcEncoder& encoder) noexcept {
  return encoder.impl_ != nullptr ? &encoder.impl_->options : nullptr;
}

core::Status HevcEncoderAccess::send(HevcEncoder& encoder,
                                     video::EncodePicture picture, bool key,
                                     std::int64_t pts) try {
  if (encoder.impl_ == nullptr) return moved_from();
  if (encoder.impl_->taken) {
    return invalid("a target is taken and not sent: send() it first");
  }
  return encoder.impl_->send(std::move(picture), key, pts);
} catch (...) {
  return core::exception_status("HevcEncoder: send",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor
