// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "driver.hpp"

#include <cmath>
#include <utility>

#include "volumetric_kit/core/base/check.hpp"

namespace volumetric_kit::sensor::arkit {

namespace {
constexpr const char* kWho = "ArkitSensor";
}  // namespace

core::Status check_options(const ArkitSensor::Options& options) {
  const auto refuse = [](const std::string& why) {
    return core::Status::invalid_argument(std::string(kWho) + ": " + why);
  };
  if (options.device == nullptr) return refuse("no device");
  if (options.color_width == 0 || options.color_height == 0 ||
      options.fps == 0) {
    return refuse("a picture size or rate of 0");
  }
  if (!std::isfinite(options.min_depth) || !std::isfinite(options.max_depth) ||
      !(options.min_depth > 0.0f) || !(options.max_depth > options.min_depth)) {
    return refuse("a depth range that is not finite, empty or starts at 0");
  }
  if (options.min_confidence > 2) {
    return refuse("min_confidence " + std::to_string(options.min_confidence) +
                  ", past ARKit's levels 0 to 2");
  }
  if (options.color_queue_family_count > kMaxColorQueueFamilies) {
    return refuse("color_queue_family_count " +
                  std::to_string(options.color_queue_family_count) +
                  ", past kMaxColorQueueFamilies");
  }
  return {};
}

// The depth copies frames hold, back here once the last copy of each frame
// goes; one outliving the driver is freed instead.
class Driver::CopyPool : public std::enable_shared_from_this<CopyPool> {
 public:
  std::shared_ptr<DepthCopy> take(std::size_t samples) {
    std::unique_ptr<DepthCopy> copy;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!free_.empty()) {
        copy = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (copy == nullptr) copy = std::make_unique<DepthCopy>();
    copy->depth.resize(samples);
    copy->confidence.resize(samples);
    std::weak_ptr<CopyPool> pool = weak_from_this();
    return std::shared_ptr<DepthCopy>(copy.release(), [pool](DepthCopy* c) {
      std::unique_ptr<DepthCopy> owned(c);
      if (const std::shared_ptr<CopyPool> back = pool.lock()) {
        std::lock_guard<std::mutex> lock(back->mutex_);
        back->free_.push_back(std::move(owned));
      }
    });
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<DepthCopy>> free_;  // guarded by mutex_
};

Driver::Retained& Driver::Retained::operator=(Retained&& o) noexcept {
  if (this != &o) {
    if (p_ != nullptr) CVPixelBufferRelease(p_);
    p_ = o.p_;
    o.p_ = nullptr;
  }
  return *this;
}

core::Result<std::unique_ptr<Driver>> Driver::create(
    const ArkitSensor::Options& options, std::string model) {
  VKC_TRY(check_options(options));
  VKC_ASSIGN(PixelBufferImporter importer,
             PixelBufferImporter::create(*options.device, kWho,
                                         options.color_queue_family_count > 0
                                             ? options.color_queue_families
                                             : nullptr,
                                         options.color_queue_family_count));
  return std::unique_ptr<Driver>(
      new Driver(options, std::move(model), std::move(importer)));
}

Driver::Driver(const ArkitSensor::Options& options, std::string model,
               PixelBufferImporter importer)
    : options_(options),
      importer_(std::move(importer)),
      copies_(std::make_shared<CopyPool>()) {
  info_.id = options.id;
  info_.model = std::move(model);
  info_.depth_to_color = camera::Mat4d(1.0);
  info_.role = SyncRole::FreeRun;
  info_.clock = ClockDomain::Host;
  info_.pose = PoseSource::Tracked;
  info_.fps = options.fps;
}

Driver::~Driver() = default;

void Driver::open_from(const Capture& capture) {
  const auto fail = [this](const std::string& why) {
    open_error_ = std::string(kWho) + ": ARKit's first frame: " + why;
    opened_cv_.notify_all();
  };
  if (capture.picture == nullptr) return fail("no picture");
  const auto width =
      static_cast<std::uint32_t>(CVPixelBufferGetWidth(capture.picture));
  const auto height =
      static_cast<std::uint32_t>(CVPixelBufferGetHeight(capture.picture));
  if (width != options_.color_width || height != options_.color_height) {
    return fail("a picture of " + std::to_string(width) + " x " +
                std::to_string(height) + ", not the format's " +
                std::to_string(options_.color_width) + " x " +
                std::to_string(options_.color_height));
  }
  core::Result<camera::CameraModel> color =
      color_camera_of(capture.intrinsics, width, height);
  if (!color.ok()) return fail(color.status().message());
  core::Result<camera::CameraModel> depth = depth_camera_of(
      color.value(),
      static_cast<std::uint32_t>(CVPixelBufferGetWidth(capture.depth)),
      static_cast<std::uint32_t>(CVPixelBufferGetHeight(capture.depth)),
      options_.min_depth, options_.max_depth);
  if (!depth.ok()) return fail(depth.status().message());
  info_.color = color.value();
  info_.depth = depth.value();
  opened_ = true;
  opened_cv_.notify_all();
}

void Driver::deliver(const Capture& capture, std::uint64_t arrival_ns) {
  std::function<void()> notify;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.tracking = capture.tracking;
    if (!opened_) {
      if (capture.depth != nullptr && open_error_.empty()) open_from(capture);
      return;
    }
    if (!started_) return;
    if (last_time_s_) {
      status_.missed +=
          missed_periods(*last_time_s_, capture.timestamp_s, options_.fps);
    }
    last_time_s_ = capture.timestamp_s;
    if (capture.depth == nullptr) return;  // a frame without the map
    const std::uint64_t sequence = next_sequence_++;
    ++received_;
    // Withheld: a pose ARKit does not trust would fuse a ghost surface.
    if (capture.tracking != ArkitTracking::Normal) {
      ++status_.withheld;
      return;
    }
    core::Result<Held> held = convert(capture);
    if (!held.ok()) {
      ++status_.refused;
      status_.last_refusal = held.status().message();
      return;
    }
    held->sequence = sequence;
    held->arrival_ns = arrival_ns;
    held->timestamp_ns = clock_.to_system_ns(capture.timestamp_s);
    queue_.push(std::move(held).value());
    notify = notify_;
  }
  if (notify) notify();
}

core::Result<Driver::Held> Driver::convert(const Capture& capture) {
  const camera::ImageSize& color_size = info_.color->size;
  const camera::ImageSize& depth_size = info_.depth->size;
  VKC_TRY(check_picture(capture.picture, color_size.width, color_size.height));
  // The pose before the depth is taken: a refused one costs no copy.
  VKC_ASSIGN(const camera::Mat4d color_to_world,
             color_to_world_of(capture.transform));
  const std::shared_ptr<DepthCopy> depth =
      copies_->take(std::size_t{depth_size.width} * depth_size.height);
  VKC_TRY(copy_depth(
      capture.depth, capture.confidence, depth_size.width, depth_size.height,
      depth->depth.data(),
      options_.min_confidence > 0 ? depth->confidence.data() : nullptr));
  Held held;
  VKC_ASSIGN(held.color, picture_color(capture.picture));
  VKC_ASSIGN(
      held.color_camera,
      color_camera_of(capture.intrinsics, color_size.width, color_size.height));
  VKC_ASSIGN(
      held.depth_camera,
      depth_camera_of(held.color_camera, depth_size.width, depth_size.height,
                      options_.min_depth, options_.max_depth));
  held.color_to_world = color_to_world;
  held.exposure = capture.exposure;
  held.picture = Retained(capture.picture);
  held.depth = depth;
  return held;
}

void Driver::interrupted(bool now) {
  std::lock_guard<std::mutex> lock(mutex_);
  status_.interrupted = now;
  if (!now && started_) clock_.remeasure(system_now_ns(), uptime_now_ns());
}

void Driver::fail(const std::string& why) {
  std::function<void()> notify;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_.session_error.empty()) status_.session_error = why;
    queue_.fail(std::string(kWho) + ": ARKit's session failed: " + why);
    opened_cv_.notify_all();
    notify = notify_;
  }
  if (notify) notify();
}

std::uint64_t Driver::next_sequence() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return next_sequence_;
}

core::Status Driver::wait_opened(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  opened_cv_.wait_for(lock, timeout, [this] {
    return opened_ || !open_error_.empty() || !status_.session_error.empty();
  });
  if (!status_.session_error.empty()) {
    return core::Status::io_error(
        std::string(kWho) +
        ": ARKit's session failed: " + status_.session_error);
  }
  if (!open_error_.empty()) return core::Status::io_error(open_error_);
  if (!opened_) {
    // The camera did not answer in time: a later open may find it.
    return core::Status::not_found(
        std::string(kWho) + ": no frame with depth from ARKit within " +
        std::to_string(timeout.count()) + " ms" +
        (status_.interrupted
             ? ": its session is interrupted (the device locked, the app in "
               "the background, or the camera in use elsewhere)"
             : ""));
  }
  return {};
}

core::Status Driver::set_queue_depth(std::size_t frames) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (frames == 0 || started_) {
    return core::Status::invalid_argument(
        std::string(kWho) +
        ": set_queue_depth needs at least 1 frame, "
        "before start");
  }
  queue_.set_depth(frames);
  return {};
}

core::Status Driver::set_frame_notify(std::function<void()> notify) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (started_) {
    return core::Status::invalid_argument(std::string(kWho) +
                                          ": set_frame_notify once started");
  }
  notify_ = std::move(notify);
  return {};
}

core::Status Driver::begin() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.failed()) {
    return core::Status::io_error(
        std::string(kWho) +
        ": ARKit's session failed: " + status_.session_error);
  }
  if (started_) return {};
  queue_.clear();
  queue_.reset_dropped();
  received_ = 0;
  delivered_ = 0;
  import_failed_ = 0;
  next_sequence_ = 0;
  last_time_s_.reset();
  status_.withheld = 0;
  status_.refused = 0;
  status_.missed = 0;
  status_.last_refusal.clear();
  clock_.start(system_now_ns(), uptime_now_ns());
  started_ = true;
  return {};
}

void Driver::end() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  started_ = false;
  queue_.clear();
}

bool Driver::started() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return started_;
}

core::Result<RgbdFrame> Driver::hand_out(Held held) {
  const camera::ImageSize& size = info_.color->size;
  RgbdFrame frame;
  const core::Status imported = importer_.import(held.picture.get(), size.width,
                                                 size.height, frame.color);
  if (!imported.ok()) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++import_failed_;
    return imported;
  }
  frame.color.kr = held.color.kr;
  frame.color.kb = held.color.kb;
  frame.color.full_range = held.color.full_range;
  frame.color.chroma_location = held.color.chroma_location;
  frame.depth = held.depth->depth.data();
  if (options_.min_confidence > 0) {
    frame.depth_confidence = held.depth->confidence.data();
    frame.min_confidence = options_.min_confidence;
  }
  frame.metres_per_unit = 0.001f;
  frame.depth_camera = held.depth_camera;
  frame.min_depth = options_.min_depth;
  frame.max_depth = options_.max_depth;
  frame.depth_noise = kArkitDepthNoise;
  frame.color_camera = held.color_camera;
  frame.color_encoding = held.color.encoding;
  frame.color_exposure = held.exposure;
  frame.color_to_world = held.color_to_world;
  frame.depth_to_color = camera::Mat4d(1.0);
  frame.timestamp_ns = held.timestamp_ns;
  frame.sequence = held.sequence;
  frame.arrival_ns = held.arrival_ns;
  frame.pixels = std::move(held.depth);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++delivered_;
  }
  return frame;
}

core::Result<std::optional<RgbdFrame>> Driver::poll() {
  VKC_ASSIGN(std::optional<Held> held, queue_.take_newest());
  if (!held) return IRgbdSensor::no_frame();
  VKC_ASSIGN(RgbdFrame frame, hand_out(std::move(*held)));
  return IRgbdSensor::some_frame(std::move(frame));
}

core::Status Driver::drain(std::vector<RgbdFrame>* out) {
  VKC_CHECK(out != nullptr, "ArkitSensor::drain needs somewhere to put them");
  std::vector<Held> taken;
  VKC_TRY(queue_.take_all(&taken));
  for (std::size_t i = 0; i < taken.size(); ++i) {
    core::Result<RgbdFrame> frame = hand_out(std::move(taken[i]));
    if (!frame.ok()) {
      // The frames before it stay handed out; the rest were taken and will
      // not be.
      for (std::size_t j = i + 1; j < taken.size(); ++j) queue_.discard();
      return frame.status();
    }
    out->push_back(std::move(frame).value());
  }
  return {};
}

SensorStats Driver::stats() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  SensorStats stats;
  stats.received = received_;
  stats.delivered = delivered_;
  stats.failed = status_.withheld + status_.refused + import_failed_;
  stats.dropped = queue_.dropped();
  return stats;
}

ArkitStatus Driver::status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_;
}

}  // namespace volumetric_kit::sensor::arkit
