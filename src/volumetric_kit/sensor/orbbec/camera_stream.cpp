// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "camera_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <thread>
#include <utility>

#include "frame_conversion.hpp"
#include "hevc_color.hpp"
#include "jpeg_color.hpp"
#include "picture_frames.hpp"

namespace volumetric_kit::recon::sensor::orbbec {

namespace {

const char* or_empty(const char* s) { return s != nullptr ? s : ""; }

// How often discovery re-asks the network. The query itself blocks about a
// second probing for Ethernet devices, so this only spaces out the retries.
constexpr std::chrono::milliseconds kDiscoveryRetry{250};

// Pairs in a row read() may skip before it calls the camera broken: about a
// second at the 30 fps default.
constexpr std::uint32_t kMaxFailedPairsInARow = 30;

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

// The SDK reports every failure as a thrown ob::Error; this repo returns
// Status across its API. `who` names the caller ("OrbbecSensor", ...).
core::Status sdk_error(const std::string& who, const std::string& what,
                       const std::exception& e) {
  return core::Status::io_error(who + ": " + what + ": " + e.what());
}

// The SDK's logger is process-wide: file sink off, console at WARN. One call
// per sink -- setLoggerSeverity sets every sink, the file one included.
void configure_sdk_logging() {
  ob::Context::setLoggerToFile(OB_LOG_SEVERITY_OFF, "");
  ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_WARN);
}

// Find the camera on the network, re-querying until it answers or the window
// closes; one query is not proof of absence for an Ethernet camera. A named
// camera is returned the moment it answers. An empty `serial` asks for the
// only camera: it is opened once the whole window has passed with no other
// answering, and refused as soon as a second does.
core::Result<std::shared_ptr<ob::Device>> discover(ob::Context& context,
                                                   const std::string& serial,
                                                   std::uint32_t timeout_ms,
                                                   const std::string& who) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::vector<std::string> answered;  // every serial any query listed
  // Unnamed: the latest list that held the one camera, opened at the end.
  std::shared_ptr<ob::DeviceList> sole;
  for (;;) {
    const auto list = context.queryDeviceList();
    const std::uint32_t count = list->getCount();
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::string answer = or_empty(list->getSerialNumber(i));
      if (!serial.empty() && answer == serial) return list->getDevice(i);
      if (std::find(answered.begin(), answered.end(), answer) ==
          answered.end()) {
        answered.push_back(answer);
      }
    }
    if (serial.empty()) {
      if (answered.size() > 1) {
        return core::Status::invalid_argument(
            who + ": " + std::to_string(answered.size()) +
            " cameras answered (" + join(answered) + "); name one");
      }
      if (count == 1) sole = list;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      if (sole != nullptr) return sole->getDevice(0);
      return core::Status::not_found(
          who + ": " +
          (serial.empty() ? std::string("no camera")
                          : "camera " + serial + " not") +
          " found within " + std::to_string(timeout_ms) + " ms" +
          (answered.empty() ? std::string(" (none answered)")
                            : " (answered: " + join(answered) + ")"));
    }
    std::this_thread::sleep_for(kDiscoveryRetry);
  }
}

}  // namespace

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

core::Result<std::unique_ptr<CameraStream>> open_camera(
    const std::string& serial, std::uint32_t discovery_timeout_ms,
    bool configure_logging, const OrbbecStreamOptions& streams,
    const camera::Mat4d& color_to_world, const std::string& who) {
  try {
    if (configure_logging) configure_sdk_logging();
    auto context = std::make_shared<ob::Context>();
    context->enableNetDeviceEnumeration(true);
    VKC_ASSIGN(auto device,
               discover(*context, serial, discovery_timeout_ms, who));
    return CameraStream::create(std::move(context), std::move(device), streams,
                                color_to_world, configure_logging, who);
  } catch (const std::exception& e) {  // ob::Error is one
    // Named when asked for by serial: the SDK can throw before the camera
    // reports it, and a rig's camera is one of several.
    return sdk_error(serial.empty() ? who : who + ": camera " + serial,
                     "opening the camera", e);
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
  s->vulkan_device_ = streams.device;
  s->vulkan_allocator_ = streams.allocator;
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
    // The camera's JPEGs or H.265 themselves, for the decoder on the GPU: the
    // SDK's RGB mode would decode them on the host.
    // TODO(sensor): the camera's H.265 encoder settings -- its key-frame
    // interval above all, since a lost frame costs the frames up to the next
    // key frame (30 at the default) -- are left as the camera has them (the
    // 2026-09-28 decision).
    const bool hevc = streams.color_codec == OrbbecColorCodec::Hevc;
    try {
      s->color_profile_ = color_modes->getVideoStreamProfile(
          static_cast<int>(streams.color_width),
          static_cast<int>(streams.color_height),
          hevc ? OB_FORMAT_H265 : OB_FORMAT_MJPG,
          static_cast<int>(streams.fps));
    } catch (const ob::Error&) {
      return core::Status::unsupported(
          s->who_ + " has no colour mode " +
          std::to_string(streams.color_width) + "x" +
          std::to_string(streams.color_height) + "@" +
          std::to_string(streams.fps) + (hevc ? " H265" : " MJPG") +
          "; it offers " + list_modes(*color_modes));
    }

    // Each camera as it captures, from the factory calibration of the modes
    // streamed: its lens, and the depth camera posed through its extrinsic to
    // the colour one, which color_to_world poses. Nothing on the host
    // undistorts or registers.
    const auto depth_video = s->depth_profile_->as<ob::VideoStreamProfile>();
    const auto color_video = s->color_profile_->as<ob::VideoStreamProfile>();
    VKC_ASSIGN(s->depth_camera_, camera_model_from(depth_video->getIntrinsic(),
                                                   depth_video->getDistortion(),
                                                   "depth", s->who_));
    VKC_ASSIGN(s->color_camera_, camera_model_from(color_video->getIntrinsic(),
                                                   color_video->getDistortion(),
                                                   "colour", s->who_));
    if (s->depth_camera_.size.width != streams.depth_width ||
        s->depth_camera_.size.height != streams.depth_height ||
        s->color_camera_.size.width != streams.color_width ||
        s->color_camera_.size.height != streams.color_height) {
      return core::Status::invalid_argument(
          s->who_ + " reports intrinsics for another size than its mode");
    }
    s->color_to_world_ = color_to_world;
    VKC_ASSIGN(
        s->depth_to_color_,
        transform_from(s->depth_profile_->getExtrinsicTo(s->color_profile_),
                       s->who_));
    s->min_depth_ = streams.min_depth;
    s->max_depth_ = streams.max_depth;

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

OrbbecStreamStats CameraStream::stats() const noexcept {
  OrbbecStreamStats s;
  s.received = mailbox_->received.load(std::memory_order_relaxed);
  s.delivered = delivered_;
  s.dropped = mailbox_->dropped.load(std::memory_order_relaxed) + discarded_;
  s.failed = failed_;
  if (hevc_ != nullptr) s.lost = hevc_->lost();
  if (jpeg_ != nullptr) {
    s.dropped += jpeg_->dropped();
    s.lost = jpeg_->lost();
  }
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
  failed_in_a_row_ = 0;
  // The colour decoder: every pair goes through it, H.265 in order, and on to
  // the mailbox decoded.
  hevc_.reset();
  jpeg_.reset();
  const auto post = [mailbox = mailbox_](std::shared_ptr<ob::FrameSet> fs) {
    mailbox->post(std::move(fs));
  };
  if (color_codec_ == OrbbecColorCodec::Hevc) {
    HevcColorDecoder::Options decoding;
    decoding.fps = fps_;
    decoding.device = vulkan_device_;
    decoding.allocator = vulkan_allocator_;
    // Once: the first start sets FFmpeg's level, and a later one leaves it
    // to whoever changed it since.
    decoding.configure_ffmpeg_logging = configure_ffmpeg_logging_;
    configure_ffmpeg_logging_ = false;
    decoding.who = who_;
    VKC_ASSIGN(auto decoder, HevcColorDecoder::start(decoding, post));
    hevc_ = std::move(decoder);
  } else {
    JpegColorDecoder::Options decoding;
    {
      std::lock_guard<std::mutex> lock(box.mutex);
      decoding.depth = box.depth;
    }
    decoding.device = vulkan_device_;
    decoding.allocator = vulkan_allocator_;
    decoding.who = who_;
    VKC_ASSIGN(auto decoder, JpegColorDecoder::start(decoding, post));
    jpeg_ = std::move(decoder);
  }
  try {
    auto config = std::make_shared<ob::Config>();
    config->enableStream(depth_profile_);
    config->enableStream(color_profile_);
    // Only pairs: a frame set missing either half is never handed over --
    // except for H.265, where every colour frame must reach the decoder,
    // paired or not. Requiring pairs there, a secondary's colour frame went
    // whenever its depth did (under 2% of triggers), and each gap cost the
    // stream up to a second, to its next key frame: 8-30% of its pairs.
    config->setFrameAggregateOutputMode(
        color_codec_ == OrbbecColorCodec::Hevc
            ? OB_FRAME_AGGREGATE_OUTPUT_COLOR_FRAME_REQUIRE
            : OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
    // Depth is paired with the colour frame nearest it in time: the SDK's
    // Pipeline turns frame sync on when it is made.
    pipeline_->start(config, [mailbox = mailbox_, hevc = hevc_,
                              jpeg = jpeg_](std::shared_ptr<ob::FrameSet> fs) {
      // Runs on the SDK's thread; it must never throw back into the SDK.
      try {
        if (fs == nullptr) return;
        mailbox->received.fetch_add(1, std::memory_order_relaxed);
        if (hevc != nullptr) {
          hevc->push(std::move(fs));
        } else {
          jpeg->push(std::move(fs));
        }
      } catch (...) {
      }
    });
  } catch (const std::exception& e) {  // ob::Error is one
    if (hevc_ != nullptr) hevc_->stop();
    if (jpeg_ != nullptr) jpeg_->stop();
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
  if (hevc_ != nullptr) hevc_->stop();
  if (jpeg_ != nullptr) jpeg_->stop();
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  mailbox_->pending.clear();
}

void CameraStream::set_queue_depth(std::size_t depth) {
  std::lock_guard<std::mutex> lock(mailbox_->mutex);
  mailbox_->depth = depth > 0 ? depth : 1;
}

core::Status CameraStream::decoder_failure() const {
  if (hevc_ != nullptr) return hevc_->failure();
  if (jpeg_ != nullptr) return jpeg_->failure();
  return {};
}

core::Result<std::shared_ptr<ob::FrameSet>> CameraStream::take() {
  VKC_TRY(decoder_failure());
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
  VKC_TRY(decoder_failure());
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

core::Status CameraStream::use_host_clock() {
  try {
    if (!device_->isGlobalTimestampSupported()) {
      return core::Status::unsupported(
          who_ +
          " has no global timestamps, which would put its frames on the "
          "host's clock");
    }
    device_->enableGlobalTimestamp(true);
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(who_, "enabling its global timestamps", e);
  }
  host_clock_ = true;
  return {};
}

core::Status CameraStream::sync_clock_to_host() {
  try {
    device_->timerSyncWithHost();
  } catch (const std::exception& e) {  // ob::Error is one
    return sdk_error(who_, "syncing its clock to the host's", e);
  }
  return {};
}

core::Result<std::optional<RgbdFrame>> CameraStream::read(
    const std::shared_ptr<ob::FrameSet>& pair) {
  // The pair is delivered, or counted as failed by one of these two. `refuse`
  // is for a pair that contradicts the stream create() negotiated -- every
  // pair after it would too; `skip` for one the SDK failed on, which the next
  // may not be.
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
  try {
    const auto depth = pair->getDepthFrame();
    const auto color = pair->getColorFrame();
    if (depth == nullptr || color == nullptr) {
      return skip("a pair is missing its depth or colour frame");
    }
    const camera::ImageSize& d = depth_camera_.size;
    const camera::ImageSize& c = color_camera_.size;
    const auto dv = depth->as<ob::VideoFrame>();
    // The colour decoders hand every picture on in a frame that carries it.
    const std::optional<DecodedPicture> picture = device_picture(*color);
    if (!picture) {
      return refuse(
          core::Status::io_error(who_ + ": colour is not a decoded picture"));
    }
    const std::uint32_t color_width = picture->yuv.width;
    const std::uint32_t color_height = picture->yuv.height;
    if (dv->getWidth() != d.width || dv->getHeight() != d.height ||
        color_width != c.width || color_height != c.height) {
      return refuse(core::Status::io_error(
          who_ + ": a pair is depth " + std::to_string(dv->getWidth()) + "x" +
          std::to_string(dv->getHeight()) + ", colour " +
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
    frame.depth_camera = depth_camera_;
    frame.min_depth = min_depth_;
    frame.max_depth = max_depth_;
    // The picture as the decoder coded it: the stream's matrix and range when
    // it names them, the Femto Mega's unlabelled BT.601 full range otherwise,
    // and the transfer and primaries it declares.
    if (!picture->encoding) {
      return refuse(core::Status::unsupported(
          who_ +
          ": the colour stream declares a transfer or primaries "
          "ColorEncoding cannot name"));
    }
    frame.color = picture->yuv;
    frame.color_encoding = *picture->encoding;
    frame.color_camera = color_camera_;
    frame.color_to_world = color_to_world_;
    frame.depth_to_color = depth_to_color_;
    // With the host's clock on, the SDK's mapping of the camera's clock onto
    // it: 0 until it has one, which a sensor array counts unmatched.
    frame.timestamp_ns = (host_clock_ ? depth->getGlobalTimeStampUs()
                                      : depth->getTimeStampUs()) *
                         1000;
    frame.sequence = depth->getIndex();
  } catch (const std::exception& e) {  // ob::Error is one
    return skip(std::string("the SDK failed on it: ") + e.what());
  }
  // Depth points into the pair, and the colour picture lives as long as its
  // frame. The context comes too, so a frame kept past the stream releases
  // its pair before the SDK goes.
  struct Held {
    std::shared_ptr<ob::Context> context;
    std::shared_ptr<ob::FrameSet> pair;  // destroyed first
  };
  frame.pixels = std::make_shared<Held>(Held{context_, pair});
  failed_in_a_row_ = 0;
  ++delivered_;
  return std::optional<RgbdFrame>(std::move(frame));
}

}  // namespace volumetric_kit::recon::sensor::orbbec
