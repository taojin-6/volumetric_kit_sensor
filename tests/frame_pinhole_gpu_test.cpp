// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// sensor/frame/shaders/pinhole.glsl on the device (shaders/pinhole_cases.comp)
// against the host's pinhole in double: the camera tier's conventions
// (camera/camera_model.hpp: pixel centres at integers, x = (u - cx) z / fx, a
// rigid cam_to_world) under the file's rules. Every case has a camera of its
// own:
//
//   * edges, in exact arithmetic, against the expected answers written out:
//     the image's first pixel and its last half pixel, the right and bottom
//     edges, round half up, a 1 x 1 image, behind the camera, and NaN or
//     infinite points and intrinsics; and, for vs_pinhole_pixel, pixels past
//     the image it must extrapolate to and a NaN it must pass through;
//   * rigid poses, for vs_world_to_camera: two in exact arithmetic, written
//     out, and a general rotation against the host's R^T (world - t);
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

#include "volumetric_kit/core/testing/buffer_readback.hpp"
#include "volumetric_kit/core/testing/test_check.hpp"
#include "volumetric_kit/core/vulkan/compute_kernel.hpp"
#include "volumetric_kit/core/vulkan/compute_util.hpp"
#include "volumetric_kit/core/vulkan/vulkan.hpp"
#include "volumetric_kit/sensor/camera/camera_model.hpp"
#include "volumetric_kit/sensor/camera/geometry.hpp"
#include "volumetric_kit/sensor/frame/vector_types.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include "pinhole_cases_comp.spv.hpp"
#include "volumetric_kit/core/testing/gpu_test.hpp"
#include "volumetric_kit/sensor/frame/device_requirements.hpp"

namespace sensor = volumetric_kit::sensor;
namespace vkc = volumetric_kit::core;
namespace cam = volumetric_kit::sensor::camera;

namespace {

constexpr std::uint32_t kRefused = ~0u;

// Mirrors Case in shaders/pinhole_cases.comp under std430.
struct PinholeCase {
  sensor::Vec4f k;      // fx, fy, cx, cy
  sensor::Vec4f point;  // xyz projected; w the depth `pixel` is unprojected at
  sensor::Vec2f pixel;
  glm::uvec2 size;
  sensor::Mat4f pose;  // cam_to_world, which `world` is moved through
  sensor::Vec4f world;
};
static_assert(sizeof(PinholeCase) == 128 &&
                  offsetof(PinholeCase, point) == 16 &&
                  offsetof(PinholeCase, pixel) == 32 &&
                  offsetof(PinholeCase, size) == 40 &&
                  offsetof(PinholeCase, pose) == 48 &&
                  offsetof(PinholeCase, world) == 112,
              "mirrors Case in shaders/pinhole_cases.comp");

// A case with no pose to test: the identity, and the origin.
PinholeCase make_case(const sensor::Vec4f& k, const sensor::Vec4f& point,
                      const sensor::Vec2f& pixel, const glm::uvec2& size) {
  return {k, point, pixel, size, sensor::Mat4f(1.0f), sensor::Vec4f(0.0f)};
}

// A camera of the camera tier, narrowed to the float values a kernel passes.
sensor::Vec4f k_of(const cam::PinholeIntrinsics& k) {
  return sensor::Vec4f(static_cast<float>(k.fx), static_cast<float>(k.fy),
                       static_cast<float>(k.cx), static_cast<float>(k.cy));
}

// What the host computes for a case, in double from its float values.
struct Reference {
  bool in_front = false;  // vs_pinhole_pixel's answer, and its pixel
  double pu = 0.0;
  double pv = 0.0;
  bool accepted = false;  // vs_project_pinhole's, and its pixel
  double u = 0.0;
  double v = 0.0;
  glm::uvec2 nearest{kRefused};
  cam::Vec3d unprojected{0.0};
  cam::Vec3d to_camera{0.0};
};

Reference reference(const PinholeCase& c) {
  const double fx = c.k.x, fy = c.k.y, cx = c.k.z, cy = c.k.w;
  const double depth = c.point.w;
  Reference r;
  r.unprojected = cam::Vec3d((c.pixel.x - cx) * depth / fx,
                             (c.pixel.y - cy) * depth / fy, depth);
  // R^T (world - t): row i of R^T is column i of R.
  const cam::Vec3d d =
      cam::Vec3d(sensor::Vec3f(c.world)) - cam::Vec3d(c.pose[3]);
  for (int i = 0; i < 3; ++i) {
    r.to_camera[i] = glm::dot(cam::Vec3d(c.pose[i]), d);
  }
  const double z = c.point.z;
  if (!(z > 0.0)) return r;
  const double u = fx * (c.point.x / z) + cx;
  const double v = fy * (c.point.y / z) + cy;
  r.in_front = true;
  r.pu = u;
  r.pv = v;
  if (!(u >= 0.0 && u < c.size.x && v >= 0.0 && v < c.size.y)) return r;
  r.accepted = true;
  r.u = u;
  r.v = v;
  r.nearest = glm::uvec2(
      std::min(static_cast<std::uint32_t>(std::floor(u + 0.5)), c.size.x - 1),
      std::min(static_cast<std::uint32_t>(std::floor(v + 0.5)), c.size.y - 1));
  return r;
}

// The device's float against the host's double: a NaN for a NaN, the same
// infinity for an infinity, otherwise within `tolerance` (0 when exact).
bool matches(float got, double want, double tolerance) {
  if (std::isnan(want)) return std::isnan(got);
  if (std::isinf(want)) return static_cast<double>(got) == want;
  return std::fabs(got - want) <= tolerance;
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
  const sensor::Vec4f k(32.0f, 16.0f, 32.0f, 8.0f);
  const glm::uvec2 size(64, 16);
  const auto at = [&](float x, float y, float z) {
    return make_case(k, sensor::Vec4f(x, y, z, 1.0f), sensor::Vec2f(0.0f),
                     size);
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
  const sensor::Vec4f one(1.0f, 1.0f, 0.0f, 0.0f);
  const auto at_one = [&](float x, float y) {
    return make_case(one, sensor::Vec4f(x, y, 1.0f, 1.0f), sensor::Vec2f(0.0f),
                     glm::uvec2(1, 1));
  };
  e.push_back({"1 x 1, (0.25, 0.25)", at_one(0.25f, 0.25f), true, {0, 0}});
  e.push_back(
      {"1 x 1, (0.75, 0.75), clamped", at_one(0.75f, 0.75f), true, {0, 0}});
  e.push_back({"1 x 1, u = 1", at_one(1.0f, 0.0f), false, no});
  e.push_back({"1 x 1, v = -0.25", at_one(0.0f, -0.25f), false, no});
  return e;
}

// vs_pinhole_pixel in exact arithmetic, on edges()' camera: in front of it,
// wherever the pixel lands, its pixel; behind it or at a NaN depth, refused.
struct UnboundedEdge {
  const char* what;
  sensor::Vec3f point;
  bool in_front;
  sensor::Vec2f pixel;  // NaN where the pixel must be NaN
};
std::vector<UnboundedEdge> unbounded_edges() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  return {
      {"u = 64, past the right edge", {1.0f, 0.0f, 1.0f}, true, {64.0f, 8.0f}},
      {"u = -0.25, before the first pixel",
       {-1.0078125f, 0.0f, 1.0f},
       true,
       {-0.25f, 8.0f}},
      {"far outside, at z = 2", {-20.0f, 8.0f, 2.0f}, true, {-288.0f, 72.0f}},
      {"behind the camera", {0.5f, 0.0f, -1.0f}, false, {}},
      {"z = 0", {0.0f, 0.0f, 0.0f}, false, {}},
      {"NaN z", {0.0f, 0.0f, nan}, false, {}},
      {"NaN x passes through", {nan, 0.0f, 1.0f}, true, {nan, 8.0f}},
      {"x = +inf", {inf, 0.0f, 1.0f}, true, {inf, 8.0f}},
  };
}

// Rigid poses for vs_world_to_camera: {what, cam_to_world, the world point,
// the camera point}, in exact arithmetic.
struct PoseEdge {
  const char* what;
  sensor::Mat4f pose;
  sensor::Vec3f world;
  sensor::Vec3f camera;
};
std::vector<PoseEdge> pose_edges() {
  // 90 degrees about Z, so R^T is not R, and a translation: R (1, 2, 3) is
  // (-2, 1, 3), and t moves it to (-1.75, 0.5, 5).
  const sensor::Mat4f turned(0.0f, 1.0f, 0.0f, 0.0f,     // column 0
                             -1.0f, 0.0f, 0.0f, 0.0f,    // column 1
                             0.0f, 0.0f, 1.0f, 0.0f,     // column 2
                             0.25f, -0.5f, 2.0f, 1.0f);  // column 3
  return {
      {"identity", sensor::Mat4f(1.0f), {1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 3.0f}},
      {"turned and moved", turned, {-1.75f, 0.5f, 5.0f}, {1.0f, 2.0f, 3.0f}},
  };
}

// Unprojections in exact arithmetic: {pixel, depth, the point}.
struct ExactUnprojection {
  sensor::Vec2f pixel;
  float depth;
  sensor::Vec3f point;
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

int gpu_main(vkc::test::GpuContext& gpu) {
  std::vector<PinholeCase> cases;
  const std::vector<Edge> edge_cases = edges();
  for (const Edge& e : edge_cases) cases.push_back(e.c);
  const std::size_t unproject_first = cases.size();
  const sensor::Vec4f edge_k(32.0f, 16.0f, 32.0f, 8.0f);
  const glm::uvec2 edge_size(64, 16);
  for (const ExactUnprojection& x : kExactUnprojections) {
    cases.push_back(make_case(edge_k, sensor::Vec4f(0.0f, 0.0f, -1.0f, x.depth),
                              x.pixel, edge_size));
  }
  const std::size_t unbounded_first = cases.size();
  const std::vector<UnboundedEdge> unbounded_cases = unbounded_edges();
  for (const UnboundedEdge& e : unbounded_cases) {
    cases.push_back(make_case(edge_k, sensor::Vec4f(e.point, 1.0f),
                              sensor::Vec2f(0.0f), edge_size));
  }
  const std::size_t pose_first = cases.size();
  const std::vector<PoseEdge> pose_cases = pose_edges();
  for (const PoseEdge& e : pose_cases) {
    cases.push_back({edge_k, sensor::Vec4f(0.0f, 0.0f, 1.0f, 1.0f),
                     sensor::Vec2f(0.0f), edge_size, e.pose,
                     sensor::Vec4f(e.world, 1.0f)});
  }
  // A general rotation, so no entry of R is 0 or 1, against the host's
  // R^T (world - t) on the same float pose.
  const std::size_t general_pose = cases.size();
  {
    glm::dmat4 pose = glm::rotate(glm::dmat4(1.0), 0.7,
                                  glm::normalize(glm::dvec3(1.0, 2.0, 3.0)));
    pose[3] = glm::dvec4(0.3, -1.2, 2.5, 1.0);
    const glm::dvec4 world = pose * glm::dvec4(0.4, -0.3, 1.7, 1.0);
    cases.push_back({edge_k, sensor::Vec4f(0.0f, 0.0f, 1.0f, 1.0f),
                     sensor::Vec2f(0.0f), edge_size, sensor::Mat4f(pose),
                     sensor::Vec4f(world)});
  }
  const std::size_t grid_first = cases.size();
  const float depths[] = {0.37f, 1.0f, 2.5f, 7.25f};
  for (const Camera& c : kCameras) {
    const sensor::Vec4f k = k_of(c.k);
    for (const float v : axis_pixels(c.size.height)) {
      for (const float u : axis_pixels(c.size.width)) {
        for (const float z : depths) {
          // The point the host unprojects from (u, v), in double.
          const double x = (u - c.k.cx) * z / c.k.fx;
          const double y = (v - c.k.cy) * z / c.k.fy;
          cases.push_back(make_case(
              k,
              sensor::Vec4f(static_cast<float>(x), static_cast<float>(y), z, z),
              sensor::Vec2f(u, v), glm::uvec2(c.size.width, c.size.height)));
        }
      }
    }
  }
  const auto n = static_cast<std::uint32_t>(cases.size());

  vkc::ComputeKernel kernel;
  vkc::KernelSetBuilder builder(gpu.device);
  CHECK(builder
            .add(kernel, "pinhole_cases", vs_pinhole_cases_comp_spv,
                 vs_pinhole_cases_comp_spv_size, 7)
            .ok());
  auto pool = builder.build();
  auto in = vkc::test::upload_device_buffer(gpu.device, gpu.allocator,
                                            cases.data(), n * sizeof(cases[0]));
  auto accepted =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(std::uint32_t));
  auto projected =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(sensor::Vec2f));
  auto nearest =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(glm::uvec2));
  auto unprojected =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(sensor::Vec4f));
  auto unbounded =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(sensor::Vec4f));
  auto to_camera =
      vkc::device_storage_buffer(gpu.allocator, n * sizeof(sensor::Vec4f));
  CHECK(pool.ok() && in.ok() && accepted.ok() && projected.ok() &&
        nearest.ok() && unprojected.ok() && unbounded.ok() && to_camera.ok());
  kernel.set.write_storage_buffer(0, in->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(1, accepted->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(2, projected->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(3, nearest->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(4, unprojected->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(5, unbounded->handle(), 0, VK_WHOLE_SIZE);
  kernel.set.write_storage_buffer(6, to_camera->handle(), 0, VK_WHOLE_SIZE);
  CHECK(vkc::dispatch(gpu.device, kernel, nullptr, 0, vkc::group_count(n, 64),
                      gpu.device.caps().limits().maxComputeWorkGroupCount[0])
            .ok());
  auto got_accepted = vkc::test::read_back<std::uint32_t>(
      gpu.device, gpu.allocator, *accepted, n);
  auto got_projected = vkc::test::read_back<sensor::Vec2f>(
      gpu.device, gpu.allocator, *projected, n);
  auto got_nearest =
      vkc::test::read_back<glm::uvec2>(gpu.device, gpu.allocator, *nearest, n);
  auto got_unprojected = vkc::test::read_back<sensor::Vec4f>(
      gpu.device, gpu.allocator, *unprojected, n);
  auto got_unbounded = vkc::test::read_back<sensor::Vec4f>(
      gpu.device, gpu.allocator, *unbounded, n);
  auto got_to_camera = vkc::test::read_back<sensor::Vec4f>(
      gpu.device, gpu.allocator, *to_camera, n);
  CHECK(got_accepted.ok() && got_projected.ok() && got_nearest.ok() &&
        got_unprojected.ok() && got_unbounded.ok() && got_to_camera.ok());

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

  // vs_pinhole_pixel against the host: its answer exactly, and its pixel
  // within `tolerance`, NaN for NaN.
  const auto unbounded_ok = [&](std::size_t i, const Reference& r,
                                double tolerance) {
    const sensor::Vec4f got = got_unbounded.value()[i];
    if (!r.in_front) return got == sensor::Vec4f(-1.0f, -1.0f, 0.0f, 0.0f);
    return got.z == 1.0f && matches(got.x, r.pu, tolerance) &&
           matches(got.y, r.pv, tolerance);
  };

  // Edges: the answers written out, and the pixel exactly the host's.
  for (std::size_t i = 0; i < edge_cases.size(); ++i) {
    const Edge& e = edge_cases[i];
    const Reference r = reference(e.c);
    const bool ok = got_accepted.value()[i] == 1u;
    const sensor::Vec2f px = got_projected.value()[i];
    if (ok != e.accepted || got_nearest.value()[i] != e.nearest ||
        r.accepted != e.accepted ||
        (ok && (px.x != static_cast<float>(r.u) ||
                px.y != static_cast<float>(r.v))) ||
        !unbounded_ok(i, r, 0.0)) {
      fail(i, e.what);
    }
  }
  for (std::size_t j = 0; j < unbounded_cases.size(); ++j) {
    const std::size_t i = unbounded_first + j;
    const UnboundedEdge& e = unbounded_cases[j];
    const Reference r = reference(cases[i]);
    const sensor::Vec4f got = got_unbounded.value()[i];
    const bool written = e.in_front ? got.z == 1.0f &&
                                          matches(got.x, e.pixel.x, 0.0) &&
                                          matches(got.y, e.pixel.y, 0.0)
                                    : got.z == 0.0f;
    if (!written || r.in_front != e.in_front || !unbounded_ok(i, r, 0.0)) {
      fail(i, e.what);
    }
  }
  for (std::size_t j = 0; j < pose_cases.size(); ++j) {
    const std::size_t i = pose_first + j;
    const Reference r = reference(cases[i]);
    const sensor::Vec3f got(got_to_camera.value()[i]);
    if (got != pose_cases[j].camera || cam::Vec3d(got) != r.to_camera) {
      fail(i, pose_cases[j].what);
    }
  }
  {
    const Reference r = reference(cases[general_pose]);
    const cam::Vec3d got(sensor::Vec3f(got_to_camera.value()[general_pose]));
    // Near the camera point the pose was built from, and the host's formula
    // on the float pose within a few float roundings.
    if (!(glm::length(got - cam::Vec3d(0.4, -0.3, 1.7)) < 1e-5 &&
          glm::length(got - r.to_camera) < 2e-6)) {
      fail(general_pose, "a general rotation");
    }
  }
  for (std::size_t j = 0; j < std::size(kExactUnprojections); ++j) {
    const std::size_t i = unproject_first + j;
    const sensor::Vec4f p = got_unprojected.value()[i];
    if (sensor::Vec3f(p) != kExactUnprojections[j].point ||
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
    const sensor::Vec2f px = got_projected.value()[i];
    if (ok != r.accepted || got_nearest.value()[i] != r.nearest ||
        (ok &&
         !(std::fabs(px.x - r.u) <= 2e-3 && std::fabs(px.y - r.v) <= 2e-3)) ||
        !unbounded_ok(i, r, 2e-3)) {
      fail(i, "grid projection");
    }
    const auto near = [](double got, double want) {
      return std::fabs(got - want) <= 2e-6 * (1.0 + std::fabs(want));
    };
    const sensor::Vec4f p = got_unprojected.value()[i];
    if (!(near(p.x, r.unprojected.x) && near(p.y, r.unprojected.y) &&
          p.z == static_cast<float>(r.unprojected.z))) {
      fail(i, "grid unprojection");
    }
  }
  // The grid has pixels on both sides of the image's bounds.
  const std::size_t grid = cases.size() - grid_first;
  std::fprintf(stderr,
               "%zu edges, %zu exact unprojections, %zu unbounded edges, %zu "
               "poses, %zu grid cases (%zu accepted), %d failures\n",
               edge_cases.size(), std::size(kExactUnprojections),
               unbounded_cases.size(), pose_cases.size() + 1, grid,
               accepted_count, failures);
  CHECK(accepted_count > 0 && accepted_count < grid);
  CHECK(failures == 0);
  return 0;
}

}  // namespace

int main() {
  return vkc::test::run_on_gpu(sensor::device_requirements(), gpu_main);
}
