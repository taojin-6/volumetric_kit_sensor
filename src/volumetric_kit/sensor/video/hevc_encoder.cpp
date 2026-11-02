// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// HevcEncoder's rules, every backend's: the argument checks, the colour set,
// the key owed after a failure, the lost session opened again, the faults,
// and the coding thread with its send and receive. The backend's session is
// hevc_encode_session.hpp's.

#include "volumetric_kit/sensor/video/hevc_encoder.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "frame_color.hpp"
#include "hevc_encode_session.hpp"
#include "hevc_encoder_access.hpp"
#include "volumetric_kit/core/base/exception_status.hpp"
#include "volumetric_kit/sensor/video/backend.hpp"

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

}  // namespace

VideoBackend video_backend() noexcept {
#if defined(__APPLE__)
  return VideoBackend::VideoToolbox;
#else
  return VideoBackend::Nvidia;
#endif
}

#if !defined(__APPLE__)
// TODO(sensor): NVENC through FFmpeg (the encoder plan's HV4).
namespace video {

struct HevcEncodeSession::Impl {};
const bool HevcEncodeSession::kWritesSiting = false;

core::Result<std::unique_ptr<HevcEncodeSession>> HevcEncodeSession::open(
    const HevcEncoder::Options& /*options*/, HevcEncoder::Facts& /*facts*/) {
  return core::Status::unsupported(
      std::string(kWho) +
      ": this build has no HEVC encoder (NVENC is not "
      "built yet; VideoToolbox is Apple's)");
}

HevcEncodeSession::HevcEncodeSession(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
HevcEncodeSession::~HevcEncodeSession() = default;

core::Result<bool> HevcEncodeSession::encode(
    EncodePicture& /*picture*/, bool /*key*/, bool /*drop*/,
    std::vector<std::uint8_t>& /*access_unit*/) {
  return core::Status::unsupported(std::string(kWho) + ": no session");
}

void HevcEncodeSession::lose() noexcept {}

}  // namespace video
#endif

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
  // it, only while the thread is not coding (coding false, under mutex).
  std::unique_ptr<video::HevcEncodeSession> session;

  std::mutex mutex;
  std::condition_variable work;  // the coding thread waits for a picture
  std::condition_variable done;  // receive waits for a picture coded
  std::condition_variable idle;  // a reopen waits for coding to end
  std::deque<Sent> sent;         // sent, not yet taken by the thread
  std::deque<Coded> coded;       // coded or failed, not yet received
  std::vector<std::vector<std::uint8_t>> spare;  // buffers to code into
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

  void run();
  // A picture failed or was lost: a key is owed, and each picture sent so
  // far, which would follow the gap, fails too. Under mutex.
  void fail() {
    key_owed = true;
    fail_through = next_seq - 1;
  }
  core::Status send(video::EncodePicture picture, bool key, std::int64_t pts);
  // A lost session closed and opened afresh. Under mutex, through @p lock,
  // which it lets go while the backend opens.
  core::Status reopen(std::unique_lock<std::mutex>& lock);
};

void HevcEncoder::Impl::run() {
  std::unique_lock<std::mutex> lock(mutex);
  for (;;) {
    work.wait(lock, [this] { return stop || !sent.empty(); });
    if (stop) return;
    Sent s = std::move(sent.front());
    sent.pop_front();
    Coded c;
    c.seq = s.seq;
    c.pts = s.pts;
    c.key = s.key;
    if (!spare.empty()) {
      c.bytes = std::move(spare.back());
      spare.pop_back();
    }
    if (s.seq <= fail_through) {
      c.status = core::Status::io_error(
          std::string(kWho) +
          ": a picture sent before this one failed, or the session was "
          "lost, so this one would follow a gap");
    } else {
      if (lose_pending) {
        lose_pending = false;
        session->lose();
      }
      const bool drop = std::exchange(drop_next, false);
      coding = true;
      lock.unlock();
      core::Result<bool> got = session->encode(s.picture, s.key, drop, c.bytes);
      s.picture = video::EncodePicture{};  // the buffer let go, unlocked
      lock.lock();
      coding = false;
      idle.notify_all();
      if (!got.ok()) {
        c.status = got.status();
        lost = true;
        fail();
      } else if (!*got) {
        c.status = core::Status::io_error(
            std::string(kWho) + ": the hardware encoder dropped the picture");
        fail();
      }
    }
    if (!c.status.ok()) c.bytes.clear();
    coded.push_back(std::move(c));
    done.notify_all();
  }
}

core::Status HevcEncoder::Impl::reopen(std::unique_lock<std::mutex>& lock) {
  // Whatever is still queued was sent before the loss: it fails uncoded.
  fail_through = next_seq - 1;
  lose_pending = false;
  idle.wait(lock, [this] { return !coding; });
  std::unique_ptr<video::HevcEncodeSession> old = std::move(session);
  lock.unlock();
  old.reset();
  Facts opened;
  core::Result<std::unique_ptr<video::HevcEncodeSession>> made =
      video::HevcEncodeSession::open(options, opened);
  lock.lock();
  VKC_TRY(made.status());
  session = std::move(made).value();
  facts = std::move(opened);
  lost = false;
  return {};
}

core::Status HevcEncoder::Impl::send(video::EncodePicture picture, bool key,
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
  sent.push_back(std::move(s));
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
  impl->options = options;
  impl->options.color = video::normalize_codes(options.color);
  if (const std::string outside = video::outside_code_set(
          impl->options.color, video::HevcEncodeSession::kWritesSiting);
      !outside.empty()) {
    return core::Status::unsupported(
        std::string(kWho) + ": " + outside +
        " is not one this encoder codes (HevcEncoder::Options::color)");
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

core::Result<bool> HevcEncoder::receive(CodedPicture& picture,
                                        std::uint64_t timeout_ns) try {
  if (impl_ == nullptr) return moved_from();
  Impl& m = *impl_;
  std::unique_lock<std::mutex> lock(m.mutex);
  if (m.coded.empty()) {
    if (m.in_flight == 0 || timeout_ns == 0) return false;
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
  Impl::Coded c = std::move(m.coded.front());
  m.coded.pop_front();
  --m.in_flight;
  if (!c.status.ok()) {
    m.spare.push_back(std::move(c.bytes));
    return c.status;
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
  return encoder.impl_->send(std::move(picture), key, pts);
} catch (...) {
  return core::exception_status("HevcEncoder: send",
                                core::Status::Code::OutOfMemory);
}

}  // namespace volumetric_kit::sensor
