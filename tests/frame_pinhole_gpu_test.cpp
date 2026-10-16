// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// core/shaders/pinhole.glsl on the device (shaders/pinhole_cases.comp)
// against the host's pinhole in double: the camera tier's conventions
// (camera/camera_model.hpp: pixel centres at integers, x = (u - cx) z / fx)
// under the file's rules. Two kinds of case, each with a camera of its own:
//
//   * edges, in exact arithmetic, against the expected answers written out:
//     the image's first pixel and its last half pixel, the right and bottom
//     edges, round half up, a 1 x 1 image, behind the camera, and NaN or
//     infinite points and intrinsics;
//   * a grid of pixels in and around five cameras of different intrinsics and
//     sizes, a 3840 x 2160 one among them, at four depths, against the host's
//     double within float tolerance (2e-3 px, the error of a few float
//     roundings at 4K); every pixel sits a quarter pixel from any boundary, so
//     no decision hinges on rounding.
//
// Skips where no device is present.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <vector>

#include "buffer_readback.hpp"
#include "test_check.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/recon/camera/camera_model.hpp"
#include "volumetric_kit/recon/core/math/vector_types.hpp"

#include "gpu_test.hpp"
#include "pinhole_cases_comp.spv.hpp"

namespace vr = volumetric_kit::recon;
namespace vkc = volumetric_kit::core;
namespace cam = volumetric_kit::recon::camera;

namespace {

constexpr std::uint32_t kRefused = ~0u;

// Mirrors Case in shaders/pinhole_cases.comp under std430.
struct PinholeCase {
  vr::Vec4f k;      // fx, fy, cx, cy
  vr::Vec4f point;  // xyz projected; w the depth `pixel` is unprojected at
  vr::Vec2f pixel;
  glm::uvec2 size;
};
static_assert(sizeof(PinholeCase) == 48 && offsetof(PinholeCase, point) == 16 &&
                  offsetof(PinholeCase, pixel) == 32 &&
                  offsetof(PinholeCase, size) == 40,
              "mirrors Case in shaders/pinhole_cases.comp");

// A camera of the camera tier, narrowed to the float values a kernel passes.
vr::Vec4f k_of(const cam::PinholeIntrinsics& k) {
  return vr::Vec4f(static_cast<float>(k.fx), static_cast<float>(k.fy),
                   static_cast<float>(k.cx), static_cast<float>(k.cy));
}

// What the host computes for a case, in double from its float values.
struct Reference {
  bool accepted = false;
  double u = 0.0;
  double v = 0.0;
  glm::uvec2 nearest{kRefused};
  vr::Vec3d unprojected{0.0};
};

Reference reference(const PinholeCase& c) {
  const double fx = c.k.x, fy = c.k.y, cx = c.k.z, cy = c.k.w;
  const double depth = c.point.w;
  Reference r;
  r.unprojected = vr::Vec3d((c.pixel.x - cx) * depth / fx,
                            (c.pixel.y - cy) * depth / fy, depth);
  const double z = c.point.z;
  if (!(z > 0.0)) return r;
  const double u = fx * (c.point.x / z) + cx;
  const double v = fy * (c.point.y / z) + cy;
  if (!(u >= 0.0 && u < c.size.x && v >= 0.0 && v < c.size.y)) return r;
  r.accepted = true;
  r.u = u;
  r.v = v;
  r.nearest = glm::uvec2(
      std::min(static_cast<std::uint32_t>(std::floor(u + 0.5)), c.size.x - 1),
      std::min(static_cast<std::uint32_t>(std::floor(v + 0.5)), c.size.y - 1));
  return r;
}

// An edge: a case whose answers are written out, in exact arithmetic (every
// divide is by a power of two).
struct Edge {
  const char* what;
  PinholeCase c;
  bool accepted;
  glm::uvec2 nearest;
};

std::vector<Edge> edges() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  // fx 32, fy 16, principal point (32, 8), 64 x 16: u = 32 x / z + 32.
  const vr::Vec4f k(32.0f, 16.0f, 32.0f, 8.0f);
  const glm::uvec2 size(64, 16);
  const auto at = [&](float x, float y, float z) {
    return PinholeCase{k, vr::Vec4f(x, y, z, 1.0f), vr::Vec2f(0.0f), size};
  };
  const glm::uvec2 no(kRefused);
  std::vector<Edge> e = {
      {"u = 0, the first pixel", at(-1.0f, 0.0f, 1.0f), true, {0, 8}},
      {"u = 0 at z = 2", at(-2.0f, 0.0f, 2.0f), true, {0, 8}},
      {"u = -0.25", at(-1.0078125f, 0.0f, 1.0f), false, no},
      {"u = 64, the right edge", at(1.0f, 0.0f, 1.0f), false, no},
      {"u = 63.5, rounds to 64, clamped",
       at(0.984375f, 0.0f, 1.0f),
       true,
       {63, 8}},
      {"u = 63.75, the last half pixel",
       at(0.9921875f, 0.0f, 1.0f),
       true,
       {63, 8}},
      {"u = 2.5 rounds half up", at(-0.921875f, 0.0f, 1.0f), true, {3, 8}},
      {"u = 2.4375 rounds down", at(-0.923828125f, 0.0f, 1.0f), true, {2, 8}},
      {"v = 0", at(0.0f, -0.5f, 1.0f), true, {32, 0}},
      {"v = 15.75, clamped", at(0.0f, 0.484375f, 1.0f), true, {32, 15}},
      {"v = 16, the bottom edge", at(0.0f, 0.5f, 1.0f), false, no},
      {"z = 0", at(0.0f, 0.0f, 0.0f), false, no},
      {"z = -0", at(0.0f, 0.0f, -0.0f), false, no},
      {"behind the camera", at(0.0f, 0.0f, -1.0f), false, no},
      {"NaN x", at(nan, 0.0f, 1.0f), false, no},
      {"NaN y", at(0.0f, nan, 1.0f), false, no},
      {"NaN z", at(0.0f, 0.0f, nan), false, no},
      {"x = +inf", at(inf, 0.0f, 1.0f), false, no},
      {"x = -inf", at(-inf, 0.0f, 1.0f), false, no},
      {"y = +inf", at(0.0f, inf, 1.0f), false, no},
  };
  // A NaN in each intrinsic.
  const char* const nan_k[] = {"NaN fx", "NaN fy", "NaN cx", "NaN cy"};
  for (int i = 0; i < 4; ++i) {
    Edge n{nan_k[i], at(0.25f, 0.25f, 1.0f), false, no};
    n.c.k[i] = nan;
    e.push_back(n);
  }
  // A 1 x 1 image: its one pixel takes [0, 1) x [0, 1).
  const vr::Vec4f one(1.0f, 1.0f, 0.0f, 0.0f);
  const auto at_one = [&](float x, float y) {
    return PinholeCase{one, vr::Vec4f(x, y, 1.0f, 1.0f), vr::Vec2f(0.0f),
                       glm::uvec2(1, 1)};
  };
  e.push_back({"1 x 1, (0.25, 0.25)", at_one(0.25f, 0.25f), true, {0, 0}});
  e.push_back(
      {"1 x 1, (0.75, 0.75), clamped", at_one(0.75f, 0.75f), true, {0, 0}});
  e.push_back({"1 x 1, u = 1", at_one(1.0f, 0.0f), false, no});
  e.push_back({"1 x 1, v = -0.25", at_one(0.0f, -0.25f), false, no});
  return e;
}

// Unprojections in exact arithmetic: {pixel, depth, the point}.
struct ExactUnprojection {
  vr::Vec2f pixel;
  float depth;
  vr::Vec3f point;
};
const ExactUnprojection kExactUnprojections[] = {
    {{32.0f, 8.0f}, 3.0f, {0.0f, 0.0f, 3.0f}},   // the principal point
    {{0.0f, 0.0f}, 2.0f, {-2.0f, -1.0f, 2.0f}},  // the corner
    {{64.0f, 16.0f}, 0.5f, {0.5f, 0.25f, 0.5f}},
};

// Five cameras, none alike: NFOV depth, a 4K colour camera, 720p, a
// half-pixel principal point with unequal focal lengths, and a 1 x 1 image.
struct Camera {
  cam::ImageSize size;
  cam::PinholeIntrinsics k;
};
const Camera kCameras[] = {
    {{640, 576}, {504.2, 504.6, 321.7, 287.3}},
    {{3840, 2160}, {2260.5, 2259.1, 1921.4, 1093.8}},
    {{1280, 720}, {610.3, 611.7, 638.2, 362.9}},
    {{320, 240}, {250.0, 180.0, 160.5, 119.5}},
    {{1, 1}, {0.75, 1.25, 0.1, 0.2}},
};

// Pixels along an axis of `n`: outside on both sides, the first and last
// pixels and their halves, and inside -- each a quarter pixel from an integer
// and from a half.
std::vector<float> axis_pixels(std::uint32_t n) {
  const float f = static_cast<float>(n);
  return {-5.75f,
          -0.25f,
          0.25f,
          0.75f,
          std::floor(0.37f * f) + 0.25f,
          std::floor(0.5f * f) + 0.75f,
          f - 1.25f,
          f - 0.75f,
          f - 0.25f,
          f + 0.25f,
          f + 6.75f};
}

int gpu_main(vr_test::GpuContext& gpu) {
  std::vector<PinholeCase> cases;
  const std::vector<Edge> edge_cases = edges();
  for (const Edge& e : edge_cases) cases.push_back(e.c);
  const std::size_t unproject_first = cases.size();
  for (const ExactUnprojection& x : kExactUnprojections) {
    cases.push_back({vr::Vec4f(32.0f, 16.0f, 32.0f, 8.0f),
                     vr::Vec4f(0.0f, 0.0f, -1.0f, x.depth), x.pixel,
                     glm::uvec2(64, 16)});
  }
  const std::size_t grid_first = cases.size();
  const float depths[] = {0.37f, 1.0f, 2.5f, 7.25f};
  for (const Camera& c : kCameras) {
    const vr::Vec4f k = k_of(c.k);
    for (const float v : axis_pixels(c.size.height)) {
      for (const float u : axis_pixels(c.size.width)) {
        for (const float z : depths) {
          // The point the host unprojects from (u, v), in double.
          const double x = (u - c.k.cx) * z / c.k.fx;
          const double y = (v - c.k.cy) * z / c.k.fy;
          cases.push_back(
              {k, vr::Vec4f(static_cast<float>(x), static_cast<float>(y), z, z),
               vr::Vec2f(u, v), glm::uvec2(c.size.width, c.size.height)});
        }
      }
    }
  }
  const auto n = static_cast<std::uint32_t>(cases.size());

  vkc::ComputeKernel kernel;
  vkc::KernelSetBuilder builder(gpu.device);
  CHECK(builder
            .add(kernel, "pinhole_cases", vr_pinhole_cases_comp_spv,
                 vr_pinhole_cases_comp_spv_size, 5)
            .ok());
  auto pool = builder.build();
  auto in = vr_test::upload_device_buffer(gpu.device, gpu.allocator,
                                          cases.data(), n * sizeof(cases[0]));
  auto accepted =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(std::uint32_t));
  auto projected =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(vr::Vec2f));
  auto nearest =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(glm::uvec2));
  auto unprojected =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(vr::Vec4f));
  CHECK(pool.ok() && in.ok() && accepted.ok() && projected.ok() &&
        nearest.ok() && unprojected.ok());
  kernel.set.write_storage_buffer(0, in->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(1, accepted->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(2, projected->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(3, nearest->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(4, unprojected->handle(), 0, VK_WHOLE_SIZE);
  CHECK(vkc::dispatch(gpu.device, kernel, nullptr, 0, vkc::group_count(n, 64),
                      gpu.device.caps().limits().maxComputeWorkGroupCount[0])
            .ok());
  auto got_accepted = vr_test::read_back<std::uint32_t>(
      gpu.device, gpu.allocator, *accepted, n);
  auto got_projected =
      vr_test::read_back<vr::Vec2f>(gpu.device, gpu.allocator, *projected, n);
  auto got_nearest =
      vr_test::read_back<glm::uvec2>(gpu.device, gpu.allocator, *nearest, n);
  auto got_unprojected =
      vr_test::read_back<vr::Vec4f>(gpu.device, gpu.allocator, *unprojected, n);
  CHECK(got_accepted.ok() && got_projected.ok() && got_nearest.ok() &&
        got_unprojected.ok());

  int failures = 0;
  const auto fail = [&](std::size_t i, const char* what) {
    if (++failures <= 20) {
      const PinholeCase& c = cases[i];
      std::fprintf(stderr,
                   "case %zu (%s): k (%g, %g, %g, %g) size %ux%u point "
                   "(%g, %g, %g) -> accepted %u px (%.6f, %.6f) nearest "
                   "(%u, %u)\n",
                   i, what, c.k.x, c.k.y, c.k.z, c.k.w, c.size.x, c.size.y,
                   c.point.x, c.point.y, c.point.z, got_accepted.value()[i],
                   got_projected.value()[i].x, got_projected.value()[i].y,
                   got_nearest.value()[i].x, got_nearest.value()[i].y);
    }
  };

  // Edges: the answers written out, and the pixel exactly the host's.
  for (std::size_t i = 0; i < edge_cases.size(); ++i) {
    const Edge& e = edge_cases[i];
    const Reference r = reference(e.c);
    const bool ok = got_accepted.value()[i] == 1u;
    const vr::Vec2f px = got_projected.value()[i];
    if (ok != e.accepted || got_nearest.value()[i] != e.nearest ||
        r.accepted != e.accepted ||
        (ok && (px.x != static_cast<float>(r.u) ||
                px.y != static_cast<float>(r.v)))) {
      fail(i, e.what);
    }
  }
  for (std::size_t j = 0; j < std::size(kExactUnprojections); ++j) {
    const std::size_t i = unproject_first + j;
    const vr::Vec4f p = got_unprojected.value()[i];
    if (vr::Vec3f(p) != kExactUnprojections[j].point ||
        got_accepted.value()[i] != 0u) {
      fail(i, "exact unprojection");
    }
  }

  // The grid: the host's decisions and nearest pixels exactly, its pixels
  // within 2e-3 px and its points within 2e-6 relative.
  std::size_t accepted_count = 0;
  for (std::size_t i = grid_first; i < cases.size(); ++i) {
    const Reference r = reference(cases[i]);
    const bool ok = got_accepted.value()[i] == 1u;
    accepted_count += ok ? 1 : 0;
    const vr::Vec2f px = got_projected.value()[i];
    if (ok != r.accepted || got_nearest.value()[i] != r.nearest ||
        (ok &&
         !(std::fabs(px.x - r.u) <= 2e-3 && std::fabs(px.y - r.v) <= 2e-3))) {
      fail(i, "grid projection");
    }
    const auto near = [](double got, double want) {
      return std::fabs(got - want) <= 2e-6 * (1.0 + std::fabs(want));
    };
    const vr::Vec4f p = got_unprojected.value()[i];
    if (!(near(p.x, r.unprojected.x) && near(p.y, r.unprojected.y) &&
          p.z == static_cast<float>(r.unprojected.z))) {
      fail(i, "grid unprojection");
    }
  }
  // The grid has pixels on both sides of the image's bounds.
  const std::size_t grid = cases.size() - grid_first;
  std::fprintf(stderr,
               "%zu edges, %zu exact unprojections, %zu grid cases "
               "(%zu accepted), %d failures\n",
               edge_cases.size(), std::size(kExactUnprojections), grid,
               accepted_count, failures);
  CHECK(accepted_count > 0 && accepted_count < grid);
  CHECK(failures == 0);
  return 0;
}

}  // namespace

int main() { return vr_test::run_on_gpu(gpu_main); }
