// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// DepthNoise (frame/depth_noise.hpp), on the host: sigma(z) term by term; what
// known() takes as a model -- not the default, all zero, a negative, NaN or
// infinite term, and any one positive term with the rest 0; stereo() against
// hand values, and unknown unless all three inputs are finite and above 0; ==
// and != term by term; and a DeviceFrame's default is unknown. It links
// sensor_frame alone, as recon's libraries do.

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>

#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/sensor/frame/depth_noise.hpp"
#include "volumetric_kit/sensor/frame/device_frame.hpp"

namespace sensor = volumetric_kit::sensor;

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

bool near(double x, double y) {
  return std::fabs(x - y) <= 1e-6 * std::fabs(y);
}

int test_sigma() {
  // Each term alone, then all three: a + b z + c z^2.
  constexpr sensor::DepthNoise tof{0.0f, 0.95e-3f, 0.0f};
  static_assert(tof.sigma(0.0f) == 0.0f, "sigma is constexpr");
  CHECK(near(tof.sigma(2.0f), 1.9e-3));
  CHECK(near(sensor::DepthNoise{0.001f, 0.0f, 0.0f}.sigma(3.0f), 0.001));
  CHECK(near(sensor::DepthNoise{0.0f, 0.0f, 0.002f}.sigma(3.0f), 0.018));
  const sensor::DepthNoise all{0.001f, 0.002f, 0.003f};
  CHECK(near(all.sigma(2.0f), 0.001 + 0.004 + 0.012));
  CHECK(all.sigma(0.0f) == 0.001f);
  return 0;
}

int test_known() {
  CHECK(!sensor::DepthNoise{}.known());
  CHECK(!(sensor::DepthNoise{0.0f, 0.0f, 0.0f}.known()));
  CHECK(!(sensor::DepthNoise{-0.0f, 0.0f, 0.0f}.known()));
  // Any one positive term, the rest 0, is a model.
  CHECK((sensor::DepthNoise{0.001f, 0.0f, 0.0f}.known()));
  CHECK((sensor::DepthNoise{0.0f, 0.95e-3f, 0.0f}.known()));
  CHECK((sensor::DepthNoise{0.0f, 0.0f, 0.002f}.known()));
  CHECK((sensor::DepthNoise{0.001f, 0.002f, 0.003f}.known()));
  // A bad term spoils a model whatever the others are.
  for (const float bad : {-1e-6f, -1.0f, kNaN, kInf, -kInf}) {
    CHECK(!(sensor::DepthNoise{bad, 0.001f, 0.0f}.known()));
    CHECK(!(sensor::DepthNoise{0.001f, bad, 0.0f}.known()));
    CHECK(!(sensor::DepthNoise{0.0f, 0.001f, bad}.known()));
  }
  return 0;
}

int test_stereo() {
  // The D455's own depth at 1280x720: 0.13 px over fx 657.271 px, B 95.03 mm.
  const sensor::DepthNoise d455 =
      sensor::DepthNoise::stereo(0.13, 657.271, 0.0950313);
  CHECK(d455.a == 0.0f && d455.b == 0.0f);
  CHECK(near(d455.c, 0.13 / (657.271 * 0.0950313)));
  CHECK(near(d455.c, 2.0812881e-3));
  CHECK(d455.known());
  // At 2 m, sigma_d z^2 / (f B): 8.3 mm.
  CHECK(near(d455.sigma(2.0f), 4.0 * 0.13 / (657.271 * 0.0950313)));
  // Round numbers: 0.04 px over 40 px and 0.1 m is 0.01 per metre.
  CHECK(near(sensor::DepthNoise::stereo(0.04, 40.0, 0.1).c, 0.01));
  // Twice the sigma doubles c; twice the focal length or baseline halves it.
  const double c = sensor::DepthNoise::stereo(0.1, 500.0, 0.05).c;
  CHECK(near(sensor::DepthNoise::stereo(0.2, 500.0, 0.05).c, 2.0 * c));
  CHECK(near(sensor::DepthNoise::stereo(0.1, 1000.0, 0.05).c, c / 2.0));
  CHECK(near(sensor::DepthNoise::stereo(0.1, 500.0, 0.1).c, c / 2.0));
  // Unknown unless all three are finite and above 0.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {0.0, -0.1, nan, inf, -inf}) {
    CHECK(sensor::DepthNoise::stereo(bad, 500.0, 0.05) == sensor::DepthNoise{});
    CHECK(sensor::DepthNoise::stereo(0.1, bad, 0.05) == sensor::DepthNoise{});
    CHECK(sensor::DepthNoise::stereo(0.1, 500.0, bad) == sensor::DepthNoise{});
  }
  return 0;
}

int test_equality() {
  const sensor::DepthNoise x{0.001f, 0.002f, 0.003f};
  CHECK(x == x && !(x != x));
  CHECK((x == sensor::DepthNoise{0.001f, 0.002f, 0.003f}));
  CHECK((x != sensor::DepthNoise{0.0f, 0.002f, 0.003f}));
  CHECK((x != sensor::DepthNoise{0.001f, 0.0f, 0.003f}));
  CHECK((x != sensor::DepthNoise{0.001f, 0.002f, 0.0f}));
  // Terms are compared as floats: one model written two ways is one model.
  CHECK((sensor::DepthNoise{0.0f, 0.95e-3f, 0.0f} ==
         sensor::DepthNoise{0.0f, 0.00095f, 0.0f}));
  return 0;
}

int test_frame_default_unknown() {
  // A frame states no noise unless its producer does.
  CHECK(sensor::DeviceFrame{}.depth_noise == sensor::DepthNoise{});
  CHECK(!sensor::DeviceFrame{}.depth_noise.known());
  return 0;
}

}  // namespace

int main() {
  if (test_sigma() != 0) return 1;
  if (test_known() != 0) return 1;
  if (test_stereo() != 0) return 1;
  if (test_equality() != 0) return 1;
  if (test_frame_default_unknown() != 0) return 1;
  std::puts("sensor_frame_depth_noise: OK");
  return 0;
}
