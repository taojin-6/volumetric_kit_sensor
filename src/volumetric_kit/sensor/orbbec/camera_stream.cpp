// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "camera_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <utility>

#include "frame_conversion.hpp"
#include "volumetric_kit/recon/sensor/camera_conventions.hpp"

#if VR_ORBBEC_WITH_VIDEO
#include "hevc_color.hpp"
#include "jpeg_color.hpp"
#include "picture_frames.hpp"
#endif

namespace volumetric_kit::recon::sensor::orbbec {

namespace {

const char* or_empty(const char* s) { return s != nullptr ? s : ""; }

// How often discovery re-asks the network. The query itself blocks about a
// second probing for Ethernet devices, so this only spaces out the retries.
constexpr std::chrono::milliseconds kDiscoveryRetry{250};

// Pairs in a row process() may skip before it calls the camera broken: about
// a second at the 30 fps default.
constexpr std::uint32_t kMaxFailedPairsInARow = 30;

// How far the SDK's intrinsics may sit from the camera a frame is stamped with
// before the first-pair check refuses them, in pixels.
constexpr float kIntrinsicsTolerance = 1e-3f;

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (const std::string& item : items) {
    if (!out.empty()) out += ", ";
    out += item;
  }
  return out;
}

// "640x576@30 Y16, 320x288@30 Y16, ..." -- what a camera offers, for the error
// that says the requested mode is not among them. It runs while that error is
// being built, so it must not throw one of its own over it.
std::string list_modes(const ob::StreamProfileList& profiles) {
  std::string out;
  try {
    for (std::uint32_t i = 0; i < profiles.getCount(); ++i) {
      const auto profile = profiles.getProfile(i);
      if (!profile->is<ob::VideoStreamProfile>()) continue;
      const auto video = profile->as<ob::VideoStreamProfile>();
      if (!out.empty()) out += ", ";
      out += std::to_string(video->getWidth()) + "x" +
             std::to_string(video->getHeight()) + "@" +
             std::to_string(video->getFps()) + " " +
             ob::TypeHelper::convertOBFormatTypeToString(video->getFormat());
    }
  } catch (const std::exception&) {
    out += out.empty() ? "(none it could list)" : ", ... (the rest unreadable)";
  }
  return out;
}

// A camera that reports its image mirrored, flipped or rotated delivers pixels
// its intrinsics do not describe, and nothing downstream can tell. Refused
// rather than undone, since undoing it would mean writing the camera's
// settings.
core::Status check_orientation(ob::Device& device, const std::string& who) {
  struct BoolProp {
    OBPropertyID id;
    const char* name;
  };
  const BoolProp bool_props[] = {
      {OB_PROP_DEPTH_MIRROR_BOOL, "depth mirror"},
      {OB_PROP_DEPTH_FLIP_BOOL, "depth flip"},
      {OB_PROP_COLOR_MIRROR_BOOL, "colour mirror"},
      {OB_PROP_COLOR_FLIP_BOOL, "colour flip"},
  };
  for (const BoolProp& prop : bool_props) {
    if (device.isPropertySupported(prop.id, OB_PERMISSION_READ) &&
        device.getBoolProperty(prop.id)) {
      return core::Status::unsupported(
          who + " has " + prop.name +
          " switched on; its intrinsics describe the "
          "unmirrored image. Switch it off.");
    }
  }
  for (const OBPropertyID id :
       {OB_PROP_DEPTH_ROTATE_INT, OB_PROP_COLOR_ROTATE_INT}) {
    if (device.isPropertySupported(id, OB_PERMISSION_READ) &&
        device.getIntProperty(id) != 0) {
      return core::Status::unsupported(
          who +
          " rotates its image; its intrinsics describe the unrotated one. Set "
          "the rotation to 0.");
    }
  }
  return {};
}

// Whether two of the SDK's calibration structs hold the same bytes: the
// comparison the H.265 mode's calibration was measured to pass.
template <typename T>
bool same_bytes(const T& a, const T& b) noexcept {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}
// None has padding, whose bytes would be indeterminate.
static_assert(sizeof(OBCameraIntrinsic) ==
              4 * sizeof(float) + 2 * sizeof(std::int16_t));
static_assert(sizeof(OBCameraDistortion) ==
              8 * sizeof(float) + sizeof(OBCameraDistortionModel));
static_assert(sizeof(OBExtrinsic) == 12 * sizeof(float));

}  // namespace

core::Status sdk_error(const std::string& who, const std::string& what,
                       const std::exception& e) {
  return core::Status::io_error(who + ": " + what + ": " + e.what());
}

core::Status check_color_codec(const OrbbecStreamOptions& streams,
                               const std::string& who) {
#if VR_ORBBEC_WITH_VIDEO
  (void)streams;
  (void)who;
#else
  if (streams.color_codec == OrbbecColorCodec::Hevc || streams.raw) {
    return core::Status::unsupported(
        who + (streams.raw ? ": raw frames need" : ": H.265 colour needs") +
        " the video decoders, which this build left out (configure with "
        "-DVR_WITH_FFMPEG=ON)");
  }
#endif
  return {};
}

void configure_sdk_logging() {
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
}

core::Result<std::vector<std::shared_ptr<ob::Device>>> discover(
    ob::Context& context, const std::vector<std::string>& serials,
    std::uint32_t timeout_ms, const std::string& who) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::vector<std::string> answered;  // every serial any query listed
  std::vector<std::shared_ptr<ob::Device>> found(
      std::max<std::size_t>(serials.size(), 1));
  // Unnamed: the latest list that held the one camera, opened at the end.
  std::shared_ptr<ob::DeviceList> sole;
  for (;;) {
    const auto list = context.queryDeviceList();
    const std::uint32_t count = list->getCount();
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::string serial = or_empty(list->getSerialNumber(i));
      if (std::find(answered.begin(), answered.end(), serial) ==
          answered.end()) {
        answered.push_back(serial);
      }
      for (std::size_t k = 0; k < serials.size(); ++k) {
        if (found[k] == nullptr && serial == serials[k]) {
          found[k] = list->getDevice(i);
        }
      }
    }
    if (serials.empty()) {
      if (answered.size() > 1) {
        return core::Status::invalid_argument(
            who + ": " + std::to_string(answered.size()) +
            " cameras answered (" + join(answered) + "); name one");
      }
      if (count == 1) sole = list;
    } else if (std::all_of(found.begin(), found.end(),
                           [](const auto& d) { return d != nullptr; })) {
      return found;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      if (sole != nullptr) {
        found[0] = sole->getDevice(0);
        return found;
      }
      std::vector<std::string> missing;
      for (std::size_t k = 0; k < serials.size(); ++k) {
        if (found[k] == nullptr) missing.push_back(serials[k]);
      }
      const std::string wanted =
          serials.empty()
              ? std::string("no camera")
              : "camera" + std::string(missing.size() > 1 ? "s " : " ") +
                    join(missing) + " not";
      return core::Status::not_found(
          who + ": " + wanted + " found within " + std::to_string(timeout_ms) +
          " ms" +
          (answered.empty() ? std::string(" (none answered)")
                            : " (answered: " + join(answered) + ")"));
    }
    std::this_thread::sleep_for(kDiscoveryRetry);
  }
}

void Mailbox::on_frameset(std::shared_ptr<ob::FrameSet> frameset) {
  // Runs on the SDK's thread; it must never throw back into the SDK.
  if (frameset == nullptr) return;
  received.fetch_add(1, std::memory_order_relaxed);
  post(std::move(frameset));
}

void Mailbox::post(std::shared_ptr<ob::FrameSet> frameset) {
  if (frameset == nullptr) return;
  std::lock_guard<std::mutex> lock(mutex);
  pending.push_back(std::move(frameset));
  while (pending.size() > depth) {
    pending.pop_front();
    dropped.fetch_add(1, std::memory_order_relaxed);
  }
}

void Mailbox::on_devices_changed(const std::string& serial,
                                 const ob::DeviceList& removed) {
  try {
    for (std::uint32_t i = 0; i < removed.getCount(); ++i) {
      if (serial == or_empty(removed.getSerialNumber(i))) {
        std::lock_guard<std::mutex> lock(mutex);
        fault = "camera " + serial + " disconnected; reopen it to use it again";
        disconnected.store(true, std::memory_order_release);
        return;
      }
    }
  } catch (...) {
    // A list the SDK cannot read out is not evidence of a disconnect.
  }
}

core::Result<std::unique_ptr<CameraStream>> CameraStream::create(
    std::shared_ptr<ob::Context> context, std::shared_ptr<ob::Device> device,
    const OrbbecStreamOptions& streams, const camera::Mat4d& color_to_world,
    bool configure_logging, const std::string& who) {
  std::unique_ptr<CameraStream> s(new CameraStream());
  s->fps_ = streams.fps;
  s->color_codec_ = streams.color_codec;
  s->configure_ffmpeg_logging_ = configure_logging;
  s->vulkan_device_ = streams.raw ? streams.device : nullptr;
  s->vulkan_allocator_ = streams.raw ? streams.allocator : nullptr;
  s->context_ = std::move(context);
  s->device_ = std::move(device);
  try {
    const auto device_info = s->device_->getDeviceInfo();
    s->info_.name = or_empty(device_info->getName());
    s->info_.serial = or_empty(device_info->getSerialNumber());
    s->info_.firmware_version = or_empty(device_info->getFirmwareVersion());
    s->info_.connection_type = or_empty(device_info->getConnectionType());
    s->info_.ip_address = or_empty(device_info->getIpAddress());
    s->who_ = who + ": camera " + s->info_.serial;
    if (s->device_->getSupportedMultiDeviceSyncModeBitmap() != 0) {
      s->sync_settings_ =
          sync_settings_from(s->device_->getMultiDeviceSyncConfig());
    }
    s->info_.sync_mode = s->sync_settings_.mode;
    if (s->info_.sync_mode == OrbbecSyncMode::SoftwareTriggering) {
      return core::Status::unsupported(
          s->who_ +
          " is in software-triggering mode, which captures only when the host "
          "sends a trigger, and this driver sends none. Set another sync mode "
          "on the camera.");
    }

    VKC_TRY(check_orientation(*s->device_, s->who_));

    s->pipeline_ = std::make_shared<ob::Pipeline>(s->device_);
    const auto depth_modes =
        s->pipeline_->getStreamProfileList(OB_SENSOR_DEPTH);
    const auto color_modes =
        s->pipeline_->getStreamProfileList(OB_SENSOR_COLOR);
    // getVideoStreamProfile throws when nothing matches; the SDK's message
    // does not say what would have, so each lookup is caught on its own.
    try {
      s->depth_profile_ = depth_modes->getVideoStreamProfile(
          static_cast<int>(streams.depth_width),
          static_cast<int>(streams.depth_height), OB_FORMAT_Y16,
          static_cast<int>(streams.fps));
    } catch (const ob::Error&) {
      return core::Status::unsupported(
          s->who_ + " has no depth mode " +
          std::to_string(streams.depth_width) + "x" +
          std::to_string(streams.depth_height) + "@" +
          std::to_string(streams.fps) + " Y16; it offers " +
          list_modes(*depth_modes));
    }
#if VR_ORBBEC_WITH_VIDEO  // without it, open has refused Hevc and raw
    if (streams.raw && streams.color_codec == OrbbecColorCodec::Mjpeg) {
      // The camera's JPEGs themselves, for the decoder on the GPU: the SDK's
      // RGB mode would decode them on the host.
      try {
        s->wire_color_profile_ = color_modes->getVideoStreamProfile(
            static_cast<int>(streams.color_width),
            static_cast<int>(streams.color_height), OB_FORMAT_MJPG,
            static_cast<int>(streams.fps));
      } catch (const ob::Error&) {
        return core::Status::unsupported(
            s->who_ + " has no colour mode " +
            std::to_string(streams.color_width) + "x" +
            std::to_string(streams.color_height) + "@" +
            std::to_string(streams.fps) + " MJPG; it offers " +
            list_modes(*color_modes));
      }
    }
    if (streams.color_codec == OrbbecColorCodec::Hevc) {
      // TODO(sensor): the camera's H.265 encoder settings -- its key-frame
      // interval above all, since a lost frame costs the frames up to the
      // next key frame (30 at the default) -- are left as the camera has
      // them (the 2026-09-28 decision).
      try {
        s->wire_color_profile_ = color_modes->getVideoStreamProfile(
            static_cast<int>(streams.color_width),
            static_cast<int>(streams.color_height), OB_FORMAT_H265,
            static_cast<int>(streams.fps));
      } catch (const ob::Error&) {
        return core::Status::unsupported(
            s->who_ + " has no colour mode " +
            std::to_string(streams.color_width) + "x" +
            std::to_string(streams.color_height) + "@" +
            std::to_string(streams.fps) + " H265; it offers " +
            list_modes(*color_modes));
      }
    }
#endif
    try {
      // RGB: MJPG on the wire, decoded on the SDK's thread -- or, for H.265
      // and raw MJPEG, the mode whose calibration the decoded frames carry.
      s->color_profile_ = color_modes->getVideoStreamProfile(
          static_cast<int>(streams.color_width),
          static_cast<int>(streams.color_height), OB_FORMAT_RGB,
          static_cast<int>(streams.fps));
    } catch (const ob::Error&) {
      return core::Status::unsupported(
          s->who_ + " has no colour mode " +
          std::to_string(streams.color_width) + "x" +
          std::to_string(streams.color_height) + "@" +
          std::to_string(streams.fps) + " RGB; it offers " +
          list_modes(*color_modes));
    }

    // The pinhole camera of the undistorted colour image: undistortion keeps
    // the stream's intrinsics and drops only the distortion, and registration
    // re-projects depth with exactly these. process() checks both claims on
    // the first pair of every start.
    const OBCameraIntrinsic intrinsic =
        s->color_profile_->as<ob::VideoStreamProfile>()->getIntrinsic();
    VKC_ASSIGN(s->color_camera_,
               color_camera_from(intrinsic, Mat4f(color_to_world)));
    if (s->color_camera_.width != streams.color_width ||
        s->color_camera_.height != streams.color_height) {
      return core::Status::io_error(
          s->who_ + " reports colour intrinsics for a " +
          std::to_string(s->color_camera_.width) + "x" +
          std::to_string(s->color_camera_.height) + " image, not the " +
          std::to_string(streams.color_width) + "x" +
          std::to_string(streams.color_height) + " mode it opened");
    }
    VKC_ASSIGN(s->depth_camera_,
               depth_from_registered_color(
                   s->color_camera_, streams.color_width, streams.color_height,
                   streams.min_depth, streams.max_depth));
    // The decoded frames are the RGB mode's camera, so the mode on the wire
    // must have its calibration: undistortion and registration read it off
    // the frame, and the raw path off the RGB profile. H.265's matched byte
    // for byte on the Femto Mega at 720p, 1080p and 4K; a camera where they
    // do not is refused, not trusted. Host MJPEG streams the RGB mode itself,
    // so it is the way round a camera refused here.
    if (s->wire_color_profile_ != nullptr) {
      const auto wire = s->wire_color_profile_->as<ob::VideoStreamProfile>();
      const auto rgb = s->color_profile_->as<ob::VideoStreamProfile>();
      if (!same_bytes(wire->getIntrinsic(), rgb->getIntrinsic()) ||
          !same_bytes(wire->getDistortion(), rgb->getDistortion()) ||
          !same_bytes(wire->getExtrinsicTo(s->depth_profile_),
                      rgb->getExtrinsicTo(s->depth_profile_))) {
        const bool hevc = streams.color_codec == OrbbecColorCodec::Hevc;
        return core::Status::unsupported(
            s->who_ + "'s " + (hevc ? "H.265" : "MJPG") +
            " colour mode reports a calibration other than its RGB mode's, " +
            (streams.raw
                 ? "which a raw frame's colour camera is read from; stream "
                   "MJPEG without raw frames"
                 : "which the decoded frames are undistorted and registered "
                   "with; stream MJPEG"));
      }
    }

    if (streams.raw) {
      // Each camera as it captures, from the factory calibration: its lens,
      // and the depth camera posed through its extrinsic to the colour one,
      // which color_to_world poses. Nothing on the host undistorts or
      // registers.
      const auto depth_video = s->depth_profile_->as<ob::VideoStreamProfile>();
      const auto color_video = s->color_profile_->as<ob::VideoStreamProfile>();
      VKC_ASSIGN(s->raw_depth_camera_,
                 camera_model_from(depth_video->getIntrinsic(),
                                   depth_video->getDistortion(), "depth"));
      VKC_ASSIGN(s->raw_color_camera_,
                 camera_model_from(color_video->getIntrinsic(),
                                   color_video->getDistortion(), "colour"));
      if (s->raw_depth_camera_.size.width != streams.depth_width ||
          s->raw_depth_camera_.size.height != streams.depth_height) {
        return core::Status::io_error(
            s->who_ +
            " reports depth intrinsics for another size "
            "than the mode it opened");
      }
      s->raw_color_to_world_ = color_to_world;
      VKC_ASSIGN(
          s->raw_depth_to_color_,
          transform_from(s->depth_profile_->getExtrinsicTo(s->color_profile_)));
      s->min_depth_ = streams.min_depth;
      s->max_depth_ = streams.max_depth;
      s->raw_ = true;
    } else {
      s->undistort_color_ =
          std::make_shared<ob::UnDistortionFilter>(OB_STREAM_COLOR);
      s->align_to_color_ = std::make_shared<ob::Align>(OB_STREAM_COLOR);
      // Registered depth at the colour image's full size -- the frame's one
      // resolution. Set rather than left to the SDK's default.
      s->align_to_color_->setMatchTargetResolution(true);
    }

    s->device_callback_id_ = s->context_->registerDeviceChangedCallback(
        [mailbox = s->mailbox_, serial = s->info_.serial](
            std::shared_ptr<ob::DeviceList> removed,
            std::shared_ptr<ob::DeviceList> /*added*/) {
          if (removed != nullptr) mailbox->on_devices_changed(serial, *removed);
        });
    s->device_callback_registered_ = true;
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(s->who_.empty() ? who : s->who_, "opening the camera", e);
  }
  return s;
}

CameraStream::~CameraStream() {
  stop();
  if (device_callback_registered_) {
    try {
      context_->unregisterDeviceChangedCallback(device_callback_id_);
    } catch (...) {
      // Nothing to do at teardown; the context goes next regardless.
    }
  }
}

OrbbecCaptureStats CameraStream::stats() const noexcept {
  OrbbecCaptureStats s;
  s.received = mailbox_->received.load(std::memory_order_relaxed);
  s.delivered = delivered_;
  s.dropped = mailbox_->dropped.load(std::memory_order_relaxed) + discarded_;
  s.failed = failed_;
  s.host_pictures = host_pictures_;
#if VR_ORBBEC_WITH_VIDEO
  if (hevc_ != nullptr) s.lost = hevc_->lost();
  if (jpeg_ != nullptr) s.lost = jpeg_->lost();
#endif
  return s;
}

core::Status CameraStream::start() {
  Mailbox& box = *mailbox_;
  {
    std::lock_guard<std::mutex> lock(box.mutex);
    // Checked before `running`: a camera that went away while streaming is
    // not "already started".
    if (!box.fault.empty()) return core::Status::io_error(box.fault);
    if (running_) return {};
    box.pending.clear();
  }
  box.received = 0;
  box.dropped = 0;
  delivered_ = 0;
  failed_ = 0;
  discarded_ = 0;
  host_pictures_ = 0;
  host_picture_delivered_ = false;
  failed_in_a_row_ = 0;
  first_pair_checked_ = false;
  // The colour decoder: every H.265 pair goes through it, in order, and on
  // to the mailbox decoded; and every raw MJPEG pair, onto the GPU.
  hevc_.reset();
  jpeg_.reset();
#if VR_ORBBEC_WITH_VIDEO
  const auto post = [mailbox = mailbox_](std::shared_ptr<ob::FrameSet> fs) {
    mailbox->post(std::move(fs));
  };
  if (color_codec_ == OrbbecColorCodec::Hevc) {
    HevcColorDecoder::Options decoding;
    decoding.fps = fps_;
    decoding.rgb_profile = color_profile_;
    decoding.yuv = raw_;
    decoding.device = vulkan_device_;
    decoding.allocator = vulkan_allocator_;
    // Once: the first start sets FFmpeg's level, and a later one leaves it
    // to whoever changed it since.
    decoding.configure_ffmpeg_logging = configure_ffmpeg_logging_;
    configure_ffmpeg_logging_ = false;
    decoding.who = who_;
    VKC_ASSIGN(auto decoder, HevcColorDecoder::start(decoding, post));
    hevc_ = std::move(decoder);
  } else if (raw_) {
    JpegColorDecoder::Options decoding;
    {
      std::lock_guard<std::mutex> lock(box.mutex);
      decoding.depth = box.depth;
    }
    decoding.device = vulkan_device_;
    decoding.allocator = vulkan_allocator_;
    decoding.configure_ffmpeg_logging = configure_ffmpeg_logging_;
    configure_ffmpeg_logging_ = false;
    decoding.who = who_;
    VKC_ASSIGN(auto decoder, JpegColorDecoder::start(decoding, post));
    jpeg_ = std::move(decoder);
  }
#endif
  try {
    auto config = std::make_shared<ob::Config>();
    config->enableStream(depth_profile_);
    config->enableStream(wire_color_profile_ != nullptr ? wire_color_profile_
                                                        : color_profile_);
    // Only pairs: a frame set missing either half is never handed over --
    // except for H.265, where every colour frame must reach the decoder,
    // paired or not. Requiring pairs there, a secondary's colour frame went
    // whenever its depth did (under 2% of triggers), and each gap cost the
    // stream up to a second, to its next key frame: 8-30% of its pairs.
    config->setFrameAggregateOutputMode(
        color_codec_ == OrbbecColorCodec::Hevc
            ? OB_FRAME_AGGREGATE_OUTPUT_COLOR_FRAME_REQUIRE
            : OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    // Pair depth with the colour frame nearest it in time.
    pipeline_->enableFrameSync();
    pipeline_->start(config, [mailbox = mailbox_, hevc = hevc_,
                              jpeg = jpeg_](std::shared_ptr<ob::FrameSet> fs) {
      try {
#if VR_ORBBEC_WITH_VIDEO
        if (hevc != nullptr) {
          if (fs == nullptr) return;
          mailbox->received.fetch_add(1, std::memory_order_relaxed);
          hevc->push(std::move(fs));
          return;
        }
        if (jpeg != nullptr) {
          if (fs == nullptr) return;
          mailbox->received.fetch_add(1, std::memory_order_relaxed);
          jpeg->push(std::move(fs));
          return;
        }
#endif
        mailbox->on_frameset(std::move(fs));
      } catch (...) {
        // Never throw into the SDK's thread.
      }
    });
  } catch (const std::exception& e) {  // ob::Error is one
#if VR_ORBBEC_WITH_VIDEO
    if (hevc_ != nullptr) hevc_->stop();
    if (jpeg_ != nullptr) jpeg_->stop();
#endif
    return sdk_error(who_, "starting", e);
  }
  running_ = true;
  return {};
}

void CameraStream::stop() noexcept {
  if (running_) {
    try {
      pipeline_->stop();
    } catch (...) {
      // A camera that already went away cannot be stopped, and there is no
      // one to tell; a late frame lands in the mailbox, not in this object.
    }
    running_ = false;
  }
  // After the pipeline, so nothing more is pushed; before the mailbox is
  // cleared, so nothing more is posted.
#if VR_ORBBEC_WITH_VIDEO
  if (hevc_ != nullptr) hevc_->stop();
  if (jpeg_ != nullptr) jpeg_->stop();
#endif
  {
    std::lock_guard<std::mutex> lock(mailbox_->mutex);
    mailbox_->pending.clear();
  }
  depth_metres_.clear();
  color_packed_.clear();
}

void CameraStream::set_queue_depth(std::size_t depth) {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  mailbox_->depth = depth > 0 ? depth : 1;
}

core::Result<std::shared_ptr<ob::FrameSet>> CameraStream::take() {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  if (!mailbox_->fault.empty()) return core::Status::io_error(mailbox_->fault);
  if (!running_ || mailbox_->pending.empty()) {
    return std::shared_ptr<ob::FrameSet>{};
  }
  std::shared_ptr<ob::FrameSet> pair = std::move(mailbox_->pending.back());
  mailbox_->dropped.fetch_add(mailbox_->pending.size() - 1,
                              std::memory_order_relaxed);
  mailbox_->pending.clear();
  return pair;
}

core::Status CameraStream::take_all(
    std::vector<std::shared_ptr<ob::FrameSet>>* out) {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  if (!mailbox_->fault.empty()) return core::Status::io_error(mailbox_->fault);
  if (!running_) return {};
  for (auto& pair : mailbox_->pending) out->push_back(std::move(pair));
  mailbox_->pending.clear();
  return {};
}

void CameraStream::discard() noexcept { ++discarded_; }

core::Status CameraStream::apply_sync(const OrbbecSyncSettings& settings) {
  if (running_) {
    return core::Status::invalid_argument(
        who_ + ": sync settings are written before start");
  }
  try {
    device_->setMultiDeviceSyncConfig(sdk_sync_config(settings));
    sync_settings_ = sync_settings_from(device_->getMultiDeviceSyncConfig());
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(who_, "writing its sync settings", e);
  }
  info_.sync_mode = sync_settings_.mode;
  return {};
}

void CameraStream::withdraw() noexcept {
  --delivered_;
  ++discarded_;
  if (host_picture_delivered_) --host_pictures_;
  host_picture_delivered_ = false;
}

std::uint64_t CameraStream::timestamp_us(const ob::FrameSet& pair) noexcept {
  try {
    const auto depth = pair.getDepthFrame();
    return depth != nullptr ? depth->getTimeStampUs() : 0;
  } catch (...) {
    return 0;
  }
}

core::Result<std::optional<CapturedFrame>> CameraStream::process(
    const std::shared_ptr<ob::FrameSet>& frameset) {
  // The pair is delivered, or counted as failed by one of these two. `refuse`
  // is for a pair that contradicts the stream create() negotiated -- every
  // pair after it would too; `skip` for one the SDK failed on, which the next
  // may not be.
  const auto refuse = [this](core::Status why) {
    ++failed_;
    return why;
  };
  const auto skip = [this](const std::string& why)
      -> core::Result<std::optional<CapturedFrame>> {
    ++failed_;
    if (++failed_in_a_row_ < kMaxFailedPairsInARow) {
      return ICameraCapture::no_frame();
    }
    return core::Status::io_error(who_ + ": " +
                                  std::to_string(failed_in_a_row_) +
                                  " pairs in a row could not be processed; the "
                                  "last: " +
                                  why);
  };

  const std::uint32_t width = color_camera_.width;
  const std::uint32_t height = color_camera_.height;
  const std::size_t pixels = static_cast<std::size_t>(width) * height;
  std::uint64_t timestamp_us = 0;
  try {
    // Undistort colour, then register depth to it: the SDK's registration
    // ignores the colour lens, so the order is what puts both on one pinhole
    // camera (the 2026-09-26 decision).
    // TODO(sensor): the same steps as GPU kernels, keeping the frame on the
    // device through fusion -- which needs device-resident fusion entry
    // points, or the upload is merely moved. And measure the camera's own
    // registration (ALIGN_D2C_HW_MODE; the 2026-09-26 decision).
    const auto undistorted = undistort_color_->process(frameset);
    if (undistorted == nullptr) {
      return skip("colour undistortion produced no frame");
    }
    const auto registered = align_to_color_->process(undistorted);
    if (registered == nullptr) {
      return skip("depth registration produced no frame");
    }
    const auto pair = registered->as<ob::FrameSet>();
    const auto depth = pair->getDepthFrame();
    const auto color = pair->getColorFrame();
    if (depth == nullptr || color == nullptr) {
      return skip("a processed pair is missing its depth or colour frame");
    }
    if (depth->getWidth() != width || depth->getHeight() != height ||
        color->getWidth() != width || color->getHeight() != height) {
      return refuse(core::Status::io_error(
          who_ + ": processed pair is depth " +
          std::to_string(depth->getWidth()) + "x" +
          std::to_string(depth->getHeight()) + ", colour " +
          std::to_string(color->getWidth()) + "x" +
          std::to_string(color->getHeight()) + "; expected both " +
          std::to_string(width) + "x" + std::to_string(height)));
    }
    if (depth->getFormat() != OB_FORMAT_Y16 ||
        depth->getDataSize() < pixels * sizeof(std::uint16_t)) {
      return refuse(core::Status::io_error(
          who_ + ": registered depth is not a full Y16 image"));
    }
    if (color->getFormat() != OB_FORMAT_RGB ||
        color->getDataSize() < pixels * 3) {
      return refuse(core::Status::io_error(
          who_ + ": undistorted colour is not a full RGB image"));
    }
    if (!first_pair_checked_) {
      // The frames are stamped with the camera computed at create; hold the
      // SDK to it once per start. Registration must have re-projected into
      // these intrinsics, and undistortion must have kept them and dropped
      // only the distortion.
      const auto depth_video =
          depth->getStreamProfile()->as<ob::VideoStreamProfile>();
      const auto color_video =
          color->getStreamProfile()->as<ob::VideoStreamProfile>();
      if (!same_pinhole(depth_video->getIntrinsic(), color_camera_,
                        kIntrinsicsTolerance)) {
        return refuse(core::Status::io_error(
            who_ +
            ": registered depth reports intrinsics other than the colour "
            "camera's; the frame would be unprojected wrongly"));
      }
      if (!same_pinhole(color_video->getIntrinsic(), color_camera_,
                        kIntrinsicsTolerance)) {
        return refuse(core::Status::io_error(
            who_ +
            ": undistorted colour reports intrinsics other than the ones "
            "read at open; the frame would be projected wrongly"));
      }
      // `!= 0` refuses a NaN coefficient too.
      const OBCameraDistortion d = color_video->getDistortion();
      if (d.k1 != 0.0f || d.k2 != 0.0f || d.k3 != 0.0f || d.k4 != 0.0f ||
          d.k5 != 0.0f || d.k6 != 0.0f || d.p1 != 0.0f || d.p2 != 0.0f) {
        return refuse(core::Status::io_error(
            who_ + ": undistorted colour still reports lens distortion"));
      }
      first_pair_checked_ = true;
    }
    const float value_scale = depth->getValueScale();
    if (!std::isfinite(value_scale) || !(value_scale > 0.0f)) {
      return refuse(
          core::Status::io_error(who_ + ": depth frame reports value scale " +
                                 std::to_string(value_scale)));
    }
    depth_metres_.resize(pixels);
    color_packed_.resize(pixels);
    depth_to_metres(reinterpret_cast<const std::uint16_t*>(depth->getData()),
                    pixels, value_scale, depth_metres_.data());
    pack_rgb(color->getData(), pixels, color_packed_.data());
    // The camera's hardware clock -- on one clock with the rest of a rig's
    // cameras once OrbbecRig has the SDK sync them.
    timestamp_us = depth->getTimeStampUs();
  } catch (const std::exception& e) {  // ob::Error is one
    return skip(std::string("the SDK failed on it: ") + e.what());
  }

  CapturedFrame frame{};
  frame.depth = depth_metres_.data();
  frame.color = color_packed_.data();
  frame.depth_camera = depth_camera_;
  frame.color_camera = color_camera_;
  // The camera's colour is ordinary 8-bit sRGB: the canonical form, declared
  // by leaving the default.
  frame.timestamp_ns = timestamp_us * 1000;
  failed_in_a_row_ = 0;
  ++delivered_;
  return ICameraCapture::some_frame(frame);
}

core::Result<std::optional<RgbdFrame>> CameraStream::process_raw(
    const std::shared_ptr<ob::FrameSet>& pair) {
#if !VR_ORBBEC_WITH_VIDEO
  // open refuses raw frames without the decoder whose planes they carry.
  (void)pair;
  ++failed_;
  return core::Status::unsupported(who_ +
                                   ": raw frames need the video decoders");
#else
  // As process(): a pair that contradicts the stream is refused, one the SDK
  // failed on is skipped, and only a run of skips is an error.
  const auto refuse = [this](core::Status why) {
    ++failed_;
    return why;
  };
  const auto skip =
      [this](const std::string& why) -> core::Result<std::optional<RgbdFrame>> {
    ++failed_;
    if (++failed_in_a_row_ < kMaxFailedPairsInARow) {
      return std::optional<RgbdFrame>();
    }
    return core::Status::io_error(who_ + ": " +
                                  std::to_string(failed_in_a_row_) +
                                  " pairs in a row could not be processed; the "
                                  "last: " +
                                  why);
  };
  RgbdFrame frame;
  bool host_color = false;  // colour on the host, not left on the device
  try {
    const auto depth = pair->getDepthFrame();
    const auto color = pair->getColorFrame();
    if (depth == nullptr || color == nullptr) {
      return skip("a pair is missing its depth or colour frame");
    }
    const camera::ImageSize& d = raw_depth_camera_.size;
    const camera::ImageSize& c = raw_color_camera_.size;
    const auto dv = depth->as<ob::VideoFrame>();
    // A picture left on the device has a size of its own; other colour is a
    // video frame of it.
    const std::optional<DecodedPicture> picture = device_picture(*color);
    host_color = !picture;
    std::uint32_t color_width = 0;
    std::uint32_t color_height = 0;
    if (picture) {
      color_width = picture->width;
      color_height = picture->height;
    } else {
      const auto cv = color->as<ob::VideoFrame>();
      color_width = cv->getWidth();
      color_height = cv->getHeight();
    }
    if (dv->getWidth() != d.width || dv->getHeight() != d.height ||
        color_width != c.width || color_height != c.height) {
      return refuse(core::Status::io_error(
          who_ + ": a raw pair is depth " + std::to_string(dv->getWidth()) +
          "x" + std::to_string(dv->getHeight()) + ", colour " +
          std::to_string(color_width) + "x" + std::to_string(color_height) +
          "; expected " + std::to_string(d.width) + "x" +
          std::to_string(d.height) + " and " + std::to_string(c.width) + "x" +
          std::to_string(c.height)));
    }
    const std::size_t depth_pixels = std::size_t{d.width} * d.height;
    if (depth->getFormat() != OB_FORMAT_Y16 ||
        depth->getDataSize() < depth_pixels * sizeof(std::uint16_t)) {
      return refuse(
          core::Status::io_error(who_ + ": depth is not a full Y16 image"));
    }
    const float value_scale = depth->getValueScale();
    if (!std::isfinite(value_scale) || !(value_scale > 0.0f)) {
      return refuse(
          core::Status::io_error(who_ + ": depth frame reports value scale " +
                                 std::to_string(value_scale)));
    }
    frame.depth = reinterpret_cast<const std::uint16_t*>(depth->getData());
    frame.metres_per_unit = value_scale / 1000.0f;  // mm per unit
    frame.depth_camera = raw_depth_camera_;
    frame.min_depth = min_depth_;
    frame.max_depth = max_depth_;
    // The matrix, range and encoding the decoder resolved: the stream's own
    // when it names them, the Femto Mega's unlabelled BT.601 full range
    // otherwise -- as the host path converts -- and the transfer and
    // primaries it declares.
    const auto describe = [&](VideoColorMatrix matrix, bool full_range,
                              const std::optional<ColorEncoding>& encoding) {
      if (!encoding) {
        return refuse(core::Status::unsupported(
            who_ +
            ": the colour stream declares a transfer or primaries "
            "ColorEncoding cannot name"));
      }
      const YcbcrWeights weights = ycbcr_weights(matrix);
      frame.color.kr = weights.kr;
      frame.color.kb = weights.kb;
      frame.color.full_range = full_range;
      frame.color_encoding = *encoding;
      return core::Status{};
    };
    frame.color.width = c.width;
    frame.color.height = c.height;
    if (picture) {
      place_device_color(*picture, &frame.color);
      VKC_TRY(
          describe(picture->matrix, picture->full_range, picture->encoding));
    } else {
      const std::uint32_t cw = (c.width + 1) / 2;
      const std::uint32_t ch = (c.height + 1) / 2;
      const std::size_t luma = std::size_t{c.width} * c.height;
      const std::size_t chroma = std::size_t{cw} * ch;
      if (color->getFormat() != OB_FORMAT_I420 ||
          color->getDataSize() < luma + 2 * chroma) {
        return refuse(core::Status::io_error(
            who_ + ": decoded colour is not a full I420 image (format " +
            std::to_string(static_cast<int>(color->getFormat())) + ", " +
            std::to_string(color->getDataSize()) + " bytes)"));
      }
      const std::optional<PlanesColor> described = planes_color(*color);
      if (!described) {
        return refuse(core::Status::io_error(
            who_ + ": decoded colour carries no colour description"));
      }
      const std::uint8_t* planes = color->getData();
      frame.color.plane[0] = planes;
      frame.color.plane[1] = planes + luma;
      frame.color.plane[2] = planes + luma + chroma;
      frame.color.stride[0] = c.width;
      frame.color.stride[1] = cw;
      frame.color.stride[2] = cw;
      frame.color.chroma_location = described->chroma_location;
      VKC_TRY(describe(described->matrix, described->full_range,
                       described->has_encoding
                           ? std::optional<ColorEncoding>(described->encoding)
                           : std::nullopt));
    }
    frame.color_camera = raw_color_camera_;
    frame.color_to_world = raw_color_to_world_;
    frame.depth_to_color = raw_depth_to_color_;
    frame.timestamp_ns = depth->getTimeStampUs() * 1000;
    frame.sequence = depth->getIndex();
  } catch (const std::exception& e) {  // ob::Error is one
    return skip(std::string("the SDK failed on it: ") + e.what());
  }
  // Depth and host planes point into the pair. The context comes too, so a
  // frame kept past the stream releases its pair before the SDK goes.
  struct Held {
    std::shared_ptr<ob::Context> context;
    std::shared_ptr<ob::FrameSet> pair;  // destroyed first
  };
  frame.pixels = std::make_shared<Held>(Held{context_, pair});
  failed_in_a_row_ = 0;
  ++delivered_;
  host_picture_delivered_ = vulkan_device_ != nullptr && host_color;
  if (host_picture_delivered_) ++host_pictures_;
  return std::optional<RgbdFrame>(std::move(frame));
#endif
}

}  // namespace volumetric_kit::recon::sensor::orbbec
