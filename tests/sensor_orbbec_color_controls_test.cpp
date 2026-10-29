// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// OrbbecSensor::open's colour controls against a camera stub
// (orbbec_camera_stub.hpp), whose properties go through the driver's own
// write and read-back: a camera that matches gets no write; a manual exposure
// is written auto exposure first, then exposure and gain, then auto white
// balance on; auto exposure leaves the stored values alone; auto white
// balance off and the white-balance temperature are never written; what the
// camera cannot take (a control it lacks, a value outside its range or off
// its steps) is refused before any write; a write that fails, and a
// value read back otherwise, are refused naming it; and a camera its sync
// check refuses gets no colour write. No camera or SDK context is opened.

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "orbbec_camera_stub.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/recon/camera/color_controls.hpp"
#include "volumetric_kit/recon/sensor/orbbec/orbbec_sensor.hpp"

namespace vkc = volumetric_kit::core;
namespace camera = volumetric_kit::recon::camera;
namespace sensor = volumetric_kit::recon::sensor;

namespace {

using Code = vkc::Status::Code;
using Writes = std::vector<std::string>;

constexpr const char* kSerial = "TEST-COLOUR-CAMERA";
const camera::ColorControls kLock{167, 60};

// The one camera that answers, as it ships: auto exposure and auto white
// balance, a stored exposure of 200 and gain of 60.
orbbec_stub::FakeCamera& attach() {
  orbbec_stub::reset();
  return orbbec_stub::cameras[kSerial];
}

// A camera already at `controls`: auto exposure off, auto white balance on.
void lock(orbbec_stub::FakeCamera& fake,
          const camera::ColorControls& controls) {
  fake.auto_exposure.value = 0;
  fake.exposure.value = static_cast<std::int32_t>(controls.exposure);
  fake.gain.value = static_cast<std::int32_t>(controls.gain);
}

sensor::OrbbecSensor::Options options(
    std::optional<camera::ColorControls> controls) {
  sensor::OrbbecSensor::Options o;
  o.serial = kSerial;
  o.color_controls = controls;
  return o;
}

// Whether no write touched the white-balance temperature or switched auto
// white balance off.
bool white_balance_untouched(const orbbec_stub::FakeCamera& fake) {
  for (const std::string& w : fake.color_writes) {
    if (w.rfind("white balance", 0) == 0) return false;
    if (w == "auto white balance 0") return false;
  }
  return fake.white_balance.value == 6500;
}

std::string message(const vkc::Status& s) {
  std::printf("  refused as expected: %s\n", s.message().c_str());
  return s.message();
}

bool has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

// A camera that already matches: no write, in either mode.
int test_matching() {
  orbbec_stub::FakeCamera& fake = attach();
  const auto automatic = sensor::OrbbecSensor::open(options(std::nullopt));
  CHECK(automatic.ok());
  CHECK(fake.color_writes.empty());
  CHECK(!automatic->info().color_controls);  // empty under auto exposure

  lock(fake, kLock);
  const auto manual = sensor::OrbbecSensor::open(options(kLock));
  CHECK(manual.ok());
  CHECK(fake.color_writes.empty());
  CHECK(manual->info().color_controls == kLock);  // as read back
  return 0;
}

// A manual exposure on a camera under auto exposure, its auto white balance
// off: auto exposure off, then both values, the gain too though it reads 60
// already (a camera under auto exposure reads its stored values), then auto
// white balance on.
int test_manual_order() {
  orbbec_stub::FakeCamera& fake = attach();
  fake.auto_white_balance.value = 0;
  const auto opened = sensor::OrbbecSensor::open(options(kLock));
  CHECK(opened.ok());
  CHECK((fake.color_writes == Writes{"auto exposure 0", "exposure 167",
                                     "gain 60", "auto white balance 1"}));
  CHECK(opened->info().color_controls == kLock);
  CHECK(white_balance_untouched(fake));

  // Already manual, at another exposure: only the value that differs.
  fake.color_writes.clear();
  const camera::ColorControls brighter{300, 60};
  const auto again = sensor::OrbbecSensor::open(options(brighter));
  CHECK(again.ok());
  CHECK((fake.color_writes == Writes{"exposure 300"}));
  CHECK(again->info().color_controls == brighter);
  return 0;
}

// Auto exposure over a camera left at a manual exposure: auto exposure on,
// and the stored exposure and gain neither written nor compared.
int test_automatic_over_lock() {
  orbbec_stub::FakeCamera& fake = attach();
  lock(fake, kLock);
  fake.ignored_write = "exposure";  // a compare would not matter either
  const auto opened = sensor::OrbbecSensor::open(options(std::nullopt));
  CHECK(opened.ok());
  CHECK((fake.color_writes == Writes{"auto exposure 1"}));
  CHECK(fake.exposure.value == 167 && fake.gain.value == 60);
  CHECK(!opened->info().color_controls);
  return 0;
}

// Auto white balance off in the camera is switched on in either mode; the
// temperature it would fall back to is never written.
int test_white_balance() {
  for (const bool manual : {false, true}) {
    orbbec_stub::FakeCamera& fake = attach();
    if (manual) lock(fake, kLock);
    fake.auto_white_balance.value = 0;
    fake.white_balance.value = 6500;
    const auto opened = sensor::OrbbecSensor::open(
        options(manual ? std::optional(kLock) : std::nullopt));
    CHECK(opened.ok());
    CHECK((fake.color_writes == Writes{"auto white balance 1"}));
    CHECK(white_balance_untouched(fake));
  }
  return 0;
}

// What the camera cannot take is refused before anything is written.
int test_refused_before_writing() {
  // Out of the camera's range, either value.
  for (const camera::ColorControls bad :
       {camera::ColorControls{2001, 60}, camera::ColorControls{167, 241}}) {
    orbbec_stub::FakeCamera& fake = attach();
    fake.auto_white_balance.value = 0;  // would be written, were it reached
    const vkc::Status s = sensor::OrbbecSensor::open(options(bad)).status();
    CHECK(s.domain() == Code::Unsupported);
    const std::string m = message(s);
    CHECK(has(m, kSerial) && has(m, "outside the camera's range"));
    CHECK(has(m, bad.exposure > 2000 ? "exposure 2001" : "gain 241"));
    CHECK(fake.color_writes.empty());
  }
  // Off the camera's steps: an exposure in steps of 3 from 1 takes 166, not
  // 167.
  {
    orbbec_stub::FakeCamera& fake = attach();
    fake.exposure.step = 3;
    fake.auto_white_balance.value = 0;
    const vkc::Status s = sensor::OrbbecSensor::open(options(kLock)).status();
    CHECK(s.domain() == Code::Unsupported);
    CHECK(
        has(message(s), "exposure 167 is off the camera's steps of 3 from 1"));
    CHECK(fake.color_writes.empty());
    const camera::ColorControls on_step{166, 60};
    const auto opened = sensor::OrbbecSensor::open(options(on_step));
    CHECK(opened.ok() && opened->info().color_controls == on_step);
  }
  // The edges of the range are taken.
  for (const camera::ColorControls edge :
       {camera::ColorControls{2000, 240}, camera::ColorControls{1, 1}}) {
    attach();
    const auto opened = sensor::OrbbecSensor::open(options(edge));
    CHECK(opened.ok() && opened->info().color_controls == edge);
  }
  // 0 is no manual exposure: refused before the camera is looked for.
  for (const camera::ColorControls zero :
       {camera::ColorControls{0, 60}, camera::ColorControls{167, 0}}) {
    attach();
    const vkc::Status s = sensor::OrbbecSensor::open(options(zero)).status();
    CHECK(s.domain() == Code::InvalidArgument);
    CHECK(has(message(s), "color_controls"));
    CHECK(orbbec_stub::lookups == 0);
  }
  return 0;
}

// A camera without one of the controls: a manual exposure is refused naming
// it, with nothing written; auto exposure opens, leaving what it lacks alone.
int test_missing_control() {
  const struct {
    orbbec_stub::FakeProperty orbbec_stub::FakeCamera::* property;
    const char* name;
  } controls[] = {
      {&orbbec_stub::FakeCamera::auto_exposure, "auto exposure"},
      {&orbbec_stub::FakeCamera::exposure, "exposure"},
      {&orbbec_stub::FakeCamera::gain, "gain"},
      {&orbbec_stub::FakeCamera::auto_white_balance, "auto white balance"},
  };
  for (const auto& c : controls) {
    orbbec_stub::FakeCamera& fake = attach();
    (fake.*c.property).writable = false;
    const vkc::Status s = sensor::OrbbecSensor::open(options(kLock)).status();
    CHECK(s.domain() == Code::Unsupported);
    const std::string m = message(s);
    CHECK(
        has(m, std::string("has no colour ") + c.name + " to read and write"));
    CHECK(fake.color_writes.empty());

    // The stub throws on any use of a control it lacks, so this opening
    // shows it is left alone.
    fake.auto_exposure.value = 0;
    fake.auto_white_balance.value = 0;
    const auto automatic = sensor::OrbbecSensor::open(options(std::nullopt));
    CHECK(automatic.ok());
    CHECK(!automatic->info().color_controls);
  }
  // None of them: auto exposure opens, writing nothing.
  orbbec_stub::FakeCamera& bare = attach();
  for (const auto& c : controls) (bare.*c.property).writable = false;
  CHECK(sensor::OrbbecSensor::open(options(std::nullopt)).ok());
  CHECK(bare.color_writes.empty());
  return 0;
}

// A write the SDK fails is IoError with its message, and stops the writes.
int test_write_failure() {
  orbbec_stub::FakeCamera& fake = attach();
  fake.failing_write = "exposure";
  const vkc::Status s = sensor::OrbbecSensor::open(options(kLock)).status();
  CHECK(s.domain() == Code::IoError);
  const std::string m = message(s);
  CHECK(has(m, kSerial) && has(m, "setting its colour controls") &&
        has(m, "test colour write failed"));
  CHECK((fake.color_writes == Writes{"auto exposure 0", "exposure 167"}));
  return 0;
}

// A value the camera reads back otherwise is refused, each field named.
int test_read_back() {
  orbbec_stub::FakeCamera& fake = attach();
  lock(fake, camera::ColorControls{167, 61});
  fake.ignored_write = "gain";
  vkc::Status s = sensor::OrbbecSensor::open(options(kLock)).status();
  CHECK(s.domain() == Code::Unsupported);
  std::string m = message(s);
  CHECK(has(m, kSerial) && has(m, "gain is 61, configured 60"));
  CHECK(!has(m, "exposure is"));
  CHECK((fake.color_writes == Writes{"gain 60"}));

  // A flag: auto exposure that will not go off, and auto white balance that
  // will not go on.
  orbbec_stub::FakeCamera& stuck = attach();
  stuck.ignored_write = "auto exposure";
  s = sensor::OrbbecSensor::open(options(kLock)).status();
  CHECK(s.domain() == Code::Unsupported);
  CHECK(has(message(s), "auto exposure is on, configured off"));
  stuck.ignored_write = "auto white balance";
  stuck.auto_white_balance.value = 0;
  s = sensor::OrbbecSensor::open(options(std::nullopt)).status();
  CHECK(s.domain() == Code::Unsupported);
  m = message(s);
  CHECK(has(m, "auto white balance is off, configured on"));
  return 0;
}

// The sync check comes first: a camera it refuses gets no colour write.
int test_after_sync() {
  orbbec_stub::FakeCamera& fake = attach();
  sensor::OrbbecSensor::Options o = options(kLock);
  o.sync.emplace();
  o.sync->mode = sensor::OrbbecSyncMode::Primary;  // the camera's: standalone
  const vkc::Status s = sensor::OrbbecSensor::open(o).status();
  CHECK(s.domain() == Code::Unsupported && has(message(s), "syncMode"));
  CHECK(fake.color_writes.empty() && fake.writes == 0);
  // Written, the colour follows.
  o.apply_sync = true;
  o.sync->trigger_out_enable = true;
  const auto opened = sensor::OrbbecSensor::open(o);
  CHECK(opened.ok() && fake.writes == 1);
  CHECK(fake.color_writes.size() == 3);
  CHECK(opened->info().color_controls == kLock);
  return 0;
}

}  // namespace

int main() {
  if (test_matching() != 0) return 1;
  if (test_manual_order() != 0) return 1;
  if (test_automatic_over_lock() != 0) return 1;
  if (test_white_balance() != 0) return 1;
  if (test_refused_before_writing() != 0) return 1;
  if (test_missing_control() != 0) return 1;
  if (test_write_failure() != 0) return 1;
  if (test_read_back() != 0) return 1;
  if (test_after_sync() != 0) return 1;
  std::puts("orbbec colour controls tests passed");
  return 0;
}
