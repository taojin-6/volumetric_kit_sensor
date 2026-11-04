// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// ARKit's world tracking as the driver's feed, iOS and iPadOS only: the
// configuration (one depth map, the format at the size and rate asked for,
// HDR off, gravity-aligned), the LiDAR check, the camera prompt, running and
// pausing the session as SessionState says at each event (session_state.hpp:
// run again after an interruption while wanted, and at a start where not
// running), relocalisation allowed, and each ARFrame's parts handed to the
// driver. Off iOS it refuses.

#include <TargetConditionals.h>

#include <memory>
#include <string>

#include "feeds.hpp"

#if TARGET_OS_IOS
#import <ARKit/ARKit.h>
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdint>

#include "session_state.hpp"
#endif

#if TARGET_OS_IOS
namespace volumetric_kit::sensor::arkit {
class CameraFeed;
}  // namespace volumetric_kit::sensor::arkit

// The session's delegate, on the sensor's queue; detached (feed null) on that
// queue before the feed goes, so a callback ARKit queued late finds none.
@interface VSArkitSessionDelegate : NSObject <ARSessionDelegate>
- (instancetype)initWithFeed:(volumetric_kit::sensor::arkit::CameraFeed*)feed;
- (void)detach;
@end
#endif

namespace volumetric_kit::sensor::arkit {

#if !TARGET_OS_IOS

core::Result<std::unique_ptr<Feed>> open_camera_feed(
    Driver& /*driver*/, const ArkitSensor::Options& /*options*/,
    dispatch_queue_t /*queue*/) {
  return core::Status::unsupported(
      "ArkitSensor: ARKit's camera runs on iOS and iPadOS only; "
      "ArkitSource::Pattern runs here");
}

#else

namespace {

ArkitTracking tracking_of(ARCamera* camera) {
  switch (camera.trackingState) {
    case ARTrackingStateNormal:
      return ArkitTracking::Normal;
    case ARTrackingStateNotAvailable:
      return ArkitTracking::NotAvailable;
    case ARTrackingStateLimited:
      break;
  }
  switch (camera.trackingStateReason) {
    case ARTrackingStateReasonRelocalizing:
      return ArkitTracking::Relocalizing;
    case ARTrackingStateReasonExcessiveMotion:
      return ArkitTracking::ExcessiveMotion;
    case ARTrackingStateReasonInsufficientFeatures:
      return ArkitTracking::InsufficientFeatures;
    default:
      return ArkitTracking::Initializing;
  }
}

// The ISO from a frame's EXIF, where it says; 0 where not.
float iso_of(NSDictionary<NSString*, id>* exif) {
  id ratings = exif[@"ISOSpeedRatings"];
  if (ratings == nil) {
    id nested = exif[@"{Exif}"];
    if ([nested isKindOfClass:[NSDictionary class]]) {
      ratings = static_cast<NSDictionary*>(nested)[@"ISOSpeedRatings"];
    }
  }
  if ([ratings isKindOfClass:[NSArray class]] &&
      static_cast<NSArray*>(ratings).count > 0) {
    ratings = static_cast<NSArray*>(ratings)[0];
  }
  return [ratings isKindOfClass:[NSNumber class]]
             ? static_cast<NSNumber*>(ratings).floatValue
             : 0.0f;
}

std::string format_text(ARVideoFormat* format) {
  return std::to_string(static_cast<long>(format.imageResolution.width)) + "x" +
         std::to_string(static_cast<long>(format.imageResolution.height)) +
         " at " + std::to_string(static_cast<long>(format.framesPerSecond)) +
         " Hz";
}

// The back wide camera's format at the size and rate asked for, the binned
// one (not recommended for high-resolution captures) where two have them;
// nil, with the formats offered in `offered`, where none does.
ARVideoFormat* video_format(const ArkitSensor::Options& options,
                            std::string* offered) {
  ARVideoFormat* found = nil;
  for (ARVideoFormat* format in ARWorldTrackingConfiguration
           .supportedVideoFormats) {
    if (format.captureDevicePosition != AVCaptureDevicePositionBack ||
        ![format.captureDeviceType
            isEqualToString:AVCaptureDeviceTypeBuiltInWideAngleCamera]) {
      continue;
    }
    if (!offered->empty()) *offered += ", ";
    *offered += format_text(format);
    if (static_cast<std::uint32_t>(format.imageResolution.width) !=
            options.color_width ||
        static_cast<std::uint32_t>(format.imageResolution.height) !=
            options.color_height ||
        static_cast<std::uint32_t>(format.framesPerSecond) != options.fps) {
      continue;
    }
    if (found == nil ||
        (found.isRecommendedForHighResolutionFrameCapturing &&
         !format.isRecommendedForHighResolutionFrameCapturing)) {
      found = format;
    }
  }
  return found;
}

// Whether the camera is allowed, asking the user the first time (a wait no
// timeout counts).
bool camera_allowed() {
  const AVAuthorizationStatus status =
      [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
  if (status == AVAuthorizationStatusAuthorized) return true;
  if (status != AVAuthorizationStatusNotDetermined) return false;
  __block BOOL granted = NO;
  dispatch_semaphore_t answered = dispatch_semaphore_create(0);
  [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                           completionHandler:^(BOOL allowed) {
                             granted = allowed;
                             dispatch_semaphore_signal(answered);
                           }];
  dispatch_semaphore_wait(answered, DISPATCH_TIME_FOREVER);
  return granted == YES;
}

}  // namespace

class CameraFeed final : public Feed {
 public:
  CameraFeed(Driver& driver, ArkitDepthMap map, dispatch_queue_t queue,
             ARWorldTrackingConfiguration* config)
      : driver_(driver), map_(map), queue_(queue), config_(config) {
    delegate_ = [[VSArkitSessionDelegate alloc] initWithFeed:this];
    session_ = [[ARSession alloc] init];
    session_.delegateQueue = queue_;
    session_.delegate = delegate_;
  }
  // Closed first, so an interruption's end ARKit queued late runs nothing;
  // then paused, and the delegate detached on the queue.
  ~CameraFeed() override {
    transition(&SessionState::close);
    session_.delegate = nil;
    VSArkitSessionDelegate* delegate = delegate_;
    dispatch_sync(queue_, ^{
      [delegate detach];
    });
  }
  CameraFeed(const CameraFeed&) = delete;
  CameraFeed& operator=(const CameraFeed&) = delete;

  // The first run: the one that sets the world.
  void run() { transition(&SessionState::open); }

  core::Status start() override {
    transition(&SessionState::start);
    return {};
  }

  // Paused, then the queue drained: a frame ARKit had queued before the pause
  // lands now, and the driver's end lets it go.
  void stop() noexcept override {
    transition(&SessionState::stop);
    dispatch_sync(queue_, ^{
                  });
  }

  // The delegate's calls, on the queue.
  void frame(ARFrame* frame) {
    const auto arrival = static_cast<std::uint64_t>(system_now_ns());
    ARDepthData* data = map_ == ArkitDepthMap::Smoothed
                            ? frame.smoothedSceneDepth
                            : frame.sceneDepth;
    Capture capture;
    if (data != nil) {
      capture.depth = data.depthMap;
      capture.confidence = data.confidenceMap;
    }
    capture.picture = frame.capturedImage;
    ARCamera* camera = frame.camera;
    const simd_float3x3 k = camera.intrinsics;
    const simd_float4x4 t = camera.transform;
    for (int c = 0; c < 3; ++c) {
      for (int r = 0; r < 3; ++r) {
        capture.intrinsics[std::size_t(c * 3 + r)] = k.columns[c][r];
      }
    }
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) {
        capture.transform[std::size_t(c * 4 + r)] = t.columns[c][r];
      }
    }
    capture.timestamp_s = frame.timestamp;
    capture.tracking = tracking_of(camera);
    capture.exposure.duration_s = static_cast<float>(camera.exposureDuration);
    capture.exposure.bias_ev = camera.exposureOffset;
    capture.exposure.iso = iso_of(frame.exifData);
    driver_.deliver(capture, arrival);
  }
  void failed(NSError* error) {
    state_.failed();
    NSString* text = error.localizedDescription;
    driver_.fail(text != nil ? std::string(text.UTF8String) : "no reason");
  }
  void interrupted() {
    state_.interruption_began();
    driver_.interrupted(true);
  }
  void interruption_ended() {
    driver_.interrupted(false);
    // ARKit waits to be run again: run while wanted (from the open until a
    // stop, and from each start), so an open still waiting gets its frame and
    // a stopped sensor's camera stays off.
    apply(state_.interruption_ended());
  }

 private:
  // `event` moves the state on the queue, where ARKit's calls move it, and a
  // run it names is made there too, as an interruption's end makes its own:
  // so a stop either lands before that end reads the state or pauses after
  // its run, and every run is made on the one queue. A pause is made after,
  // on the caller's thread, as before.
  void transition(SessionState::Act (SessionState::*event)()) {
    __block SessionState::Act act = SessionState::Act::None;
    CameraFeed* self = this;
    dispatch_sync(queue_, ^{
      act = (self->state_.*event)();
      if (act != SessionState::Act::Pause) self->apply(act);
    });
    if (act == SessionState::Act::Pause) apply(act);
  }

  void apply(SessionState::Act act) {
    switch (act) {
      case SessionState::Act::RunFresh:
        [session_ runWithConfiguration:config_
                               options:ARSessionRunOptionResetTracking |
                                       ARSessionRunOptionRemoveExistingAnchors];
        break;
      case SessionState::Act::Run:
        [session_ runWithConfiguration:config_];  // the world kept
        break;
      case SessionState::Act::Pause:
        [session_ pause];
        break;
      case SessionState::Act::None:
        break;
    }
  }

  Driver& driver_;
  const ArkitDepthMap map_;
  dispatch_queue_t queue_;
  ARWorldTrackingConfiguration* config_;
  VSArkitSessionDelegate* delegate_;
  ARSession* session_;
  SessionState state_;  // the queue's alone
};

core::Result<std::unique_ptr<Feed>> open_camera_feed(
    Driver& driver, const ArkitSensor::Options& options,
    dispatch_queue_t queue) {
  const ARFrameSemantics semantics =
      options.depth_map == ArkitDepthMap::Smoothed
          ? ARFrameSemanticSmoothedSceneDepth
          : ARFrameSemanticSceneDepth;
  if (![ARWorldTrackingConfiguration supportsFrameSemantics:semantics]) {
    return core::Status::unsupported(
        "ArkitSensor: this device has no LiDAR scanner, so ARKit gives no "
        "scene depth");
  }
  std::string offered;
  ARVideoFormat* format = video_format(options, &offered);
  if (format == nil) {
    return core::Status::unsupported(
        "ArkitSensor: the back wide camera has no format at " +
        std::to_string(options.color_width) + "x" +
        std::to_string(options.color_height) + " at " +
        std::to_string(options.fps) + " Hz; it offers " +
        (offered.empty() ? std::string("none") : offered));
  }
  ARWorldTrackingConfiguration* config =
      [[ARWorldTrackingConfiguration alloc] init];
  config.frameSemantics = semantics;
  config.worldAlignment = ARWorldAlignmentGravity;
  config.videoFormat = format;
  config.videoHDRAllowed = NO;  // 8-bit '420f'
  if (!camera_allowed()) {
    return core::Status::io_error(
        "ArkitSensor: the camera is not allowed (Settings, Privacy & "
        "Security, Camera)");
  }
  auto feed =
      std::make_unique<CameraFeed>(driver, options.depth_map, queue, config);
  feed->run();
  VKC_TRY(
      driver.wait_opened(std::chrono::milliseconds(options.open_timeout_ms)));
  return std::unique_ptr<Feed>(std::move(feed));
}

#endif

}  // namespace volumetric_kit::sensor::arkit

#if TARGET_OS_IOS
@implementation VSArkitSessionDelegate {
  volumetric_kit::sensor::arkit::CameraFeed* _feed;
}

- (instancetype)initWithFeed:(volumetric_kit::sensor::arkit::CameraFeed*)feed {
  self = [super init];
  if (self != nil) _feed = feed;
  return self;
}

- (void)detach {
  _feed = nullptr;
}

- (void)session:(ARSession*)session didUpdateFrame:(ARFrame*)frame {
  if (_feed != nullptr) _feed->frame(frame);
}

- (void)session:(ARSession*)session didFailWithError:(NSError*)error {
  if (_feed != nullptr) _feed->failed(error);
}

- (void)sessionWasInterrupted:(ARSession*)session {
  if (_feed != nullptr) _feed->interrupted();
}

- (void)sessionInterruptionEnded:(ARSession*)session {
  if (_feed != nullptr) _feed->interruption_ended();
}

- (BOOL)sessionShouldAttemptRelocalization:(ARSession*)session {
  return YES;  // one world for the sensor: found again, never reset
}
@end
#endif
