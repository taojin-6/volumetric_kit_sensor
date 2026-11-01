// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// DeckLink SDK smoke (VS_WITH_DECKLINK only), no card opened:
//   - the driver's own codes are the SDK's: v210, the profile, the three
//     modes;
//   - Desktop Video's library: missing, open_capture is NotFound naming the
//     install; present, its API is 16 or newer (it is loaded, nothing
//     enumerated);
//   - DecklinkStereo::open refuses bad options before any device or card,
//     each named, through its own mapping of the options.

#include <DeckLinkAPI.h>
#include <DeckLinkAPIVersion.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "capture.hpp"
#include "ports.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/decklink/decklink_stereo.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace dl = volumetric_kit::sensor::decklink;
namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;

namespace {

bool refused(const vkc::Result<sensor::DecklinkStereo>& r, const char* text) {
  if (r.ok() || r.status().domain() != vkc::Status::Code::InvalidArgument) {
    return false;
  }
  std::printf("  refused: %s\n", r.status().message().c_str());
  return r.status().message().find(text) != std::string::npos;
}

int gpu_main(vkc::test::GpuContext& gpu) {
  sensor::DecklinkStereo::Options ok;
  ok.left = {"bm-1", "SDI 2"};
  ok.right = {"bm-2", "SDI 1"};
  sensor::DecklinkStereo::Options o = ok;
  o.right.input = "SDI 2";
  CHECK(refused(sensor::DecklinkStereo::open(o, gpu.device, gpu.allocator),
                "both cameras on SDI 2"));
  o = ok;
  o.left.input = "HDMI";
  CHECK(refused(sensor::DecklinkStereo::open(o, gpu.device, gpu.allocator),
                "no connector \"HDMI\""));
  o = ok;
  o.width = 1920;
  o.height = 1080;
  CHECK(refused(sensor::DecklinkStereo::open(o, gpu.device, gpu.allocator),
                "no mode of 1920x1080 at 30 fps"));
  o = ok;
  o.fps = 60;
  o.max_skew_ns = 8'000'000;
  CHECK(refused(sensor::DecklinkStereo::open(o, gpu.device, gpu.allocator),
                "no mode of 3840x2160 at 60 fps"));
  o = ok;
  o.max_skew_ns = 16'666'667;
  CHECK(refused(sensor::DecklinkStereo::open(o, gpu.device, gpu.allocator),
                "is not in 1..16666666"));
  o = ok;
  o.calibration = sensor::StereoCalibration{};
  o.calibration->left.size = {1920, 1080};
  CHECK(refused(sensor::DecklinkStereo::open(o, gpu.device, gpu.allocator),
                "a lens calibrated at 1920x1080"));
  std::puts("DecklinkStereo::open: each refusal before any card, named");
  return 0;
}

}  // namespace

int main() {
  CHECK(dl::kV210 == static_cast<std::uint32_t>(bmdFormat10BitYUV));
  CHECK(dl::kFourSubDevicesHalfDuplex ==
        static_cast<std::uint32_t>(bmdProfileFourSubDevicesHalfDuplex));
  CHECK(dl::mode_of(3840, 2160, 24)->code ==
        static_cast<std::uint32_t>(bmdMode4K2160p24));
  CHECK(dl::mode_of(3840, 2160, 25)->code ==
        static_cast<std::uint32_t>(bmdMode4K2160p25));
  CHECK(dl::mode_of(3840, 2160, 30)->code ==
        static_cast<std::uint32_t>(bmdMode4K2160p30));
  std::puts("the driver's codes are the SDK's");

  IDeckLinkAPIInformation* api = CreateDeckLinkAPIInformationInstance();
  if (api == nullptr) {
    dl::CaptureConfig config;
    config.names = {"SDI 2 (bm-1)", "SDI 1 (bm-2)"};
    config.sub_device = {2, 0};
    config.mode = dl::mode_of(3840, 2160, 30)->code;
    const auto capture = dl::open_capture(config);
    CHECK(!capture.ok());
    CHECK(capture.status().domain() == vkc::Status::Code::NotFound);
    CHECK(capture.status().message().find("install Blackmagic Desktop Video") !=
          std::string::npos);
    std::printf("no Desktop Video: %s\n", capture.status().message().c_str());
  } else {
    std::int64_t version = 0;
    CHECK(api->GetInt(BMDDeckLinkAPIVersion, &version) == S_OK);
    api->Release();
    std::printf("Desktop Video's API %lld.%lld (SDK headers %s)\n",
                static_cast<long long>(version >> 24),
                static_cast<long long>((version >> 16) & 0xff),
                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
    CHECK((version >> 24) >= 16);
  }
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
