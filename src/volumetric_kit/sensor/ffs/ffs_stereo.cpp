// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

#include "volumetric_kit/sensor/ffs/ffs_stereo.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../video/cuda_driver.hpp"
#include "ffs_kernels.hpp"
#include "volumetric_kit/core/base/log.hpp"
#include "volumetric_kit/core/vulkan/buffer.hpp"
#include "volumetric_kit/core/vulkan/device.hpp"
#include "volumetric_kit/core/vulkan/external_memory.hpp"
#include "volumetric_kit/sensor/video/cuda_interop.hpp"

namespace volumetric_kit::sensor {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using video::CudaContextScope;

constexpr const char* kWho = "FfsStereo";
// The network's inputs and output, as tools/ffs/export_onnx.py names them.
constexpr const char* kLeft = "left_image";
constexpr const char* kRight = "right_image";
constexpr const char* kDisparity = "disparity";
// FFS's sides are multiples of this.
constexpr std::uint32_t kNetMultiple = 32;
// Views mapped into CUDA at most, the least recently read let go first: the
// rectifier writes two buffers while no pair is kept past the next, and a new
// pair of them for each frame one is kept. A buffer it has replaced is freed
// with the last pair holding it, which this matcher cannot see; its mapping
// stays until let go here, and keeps its memory (the import holds the
// descriptor's reference), so CUDA never reads freed memory.
constexpr std::size_t kMaxViews = 8;

core::Status refuse(const std::string& why) {
  return core::Status::invalid_argument(std::string(kWho) + ": " + why);
}

core::Status cuda_failure(const char* what, cudaError_t error) {
  return core::Status::backend_error(
      core::Status::Backend::Cuda, static_cast<std::int64_t>(error),
      std::string(kWho) + ": " + what + ": " + cudaGetErrorName(error));
}

std::uint32_t round_up(std::uint32_t v, std::uint32_t to) noexcept {
  return (v + to - 1) / to * to;
}

// TensorRT's messages, warnings and errors only, to the package's log. One for
// the process, as TensorRT keeps the first logger it is given; it keeps the
// last error for the Status of the call that failed.
class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity > Severity::kWARNING) return;
    try {
      core::log_message(severity == Severity::kWARNING ? core::LogLevel::Warning
                                                       : core::LogLevel::Error,
                        "sensor_ffs", message);
      if (severity != Severity::kWARNING) {
        const std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = message;
      }
    } catch (...) {
      // A message lost to an allocation failure.
    }
  }

  std::string last_error() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
  }

 private:
  mutable std::mutex mutex_;
  std::string last_error_;
};

Logger& logger() {
  static Logger instance;
  return instance;
}

core::Status tensorrt_failure(const std::string& what) {
  const std::string error = logger().last_error();
  return core::Status::backend_error(
      core::Status::Backend::Other, 1,
      std::string(kWho) + ": TensorRT " + what +
          (error.empty() ? std::string() : ": " + error));
}

template <typename T>
using TrtPtr = std::unique_ptr<T>;

// 64-bit FNV-1a: the model's bytes named in its plan's file.
std::uint64_t fnv1a(const std::vector<char>& bytes) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ull;
  for (const char c : bytes) {
    h ^= static_cast<unsigned char>(c);
    h *= 0x100000001b3ull;
  }
  return h;
}

std::string hex(const unsigned char* bytes, std::size_t count) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  for (std::size_t i = 0; i < count; ++i) {
    out += kDigits[bytes[i] >> 4];
    out += kDigits[bytes[i] & 0xf];
  }
  return out;
}

core::Result<std::vector<char>> read_file(const fs::path& path,
                                          const char* what) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) {
    return core::Status::not_found(std::string(kWho) + ": no " + what + " " +
                                   path.string());
  }
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  const std::streamoff size = in ? static_cast<std::streamoff>(in.tellg()) : -1;
  if (size < 0) {
    return core::Status::io_error(std::string(kWho) + ": cannot read " +
                                  path.string());
  }
  std::vector<char> bytes(static_cast<std::size_t>(size));
  in.seekg(0);
  if (!in.read(bytes.data(), size)) {
    return core::Status::io_error(std::string(kWho) + ": cannot read " +
                                  path.string());
  }
  return bytes;
}

// The plan written whole or not at all: to a file of this process's, then
// renamed over the name, so a reader never sees half of one.
core::Status write_plan(const fs::path& path, const void* data,
                        std::size_t bytes) {
  fs::path partial = path;
  partial += ".partial-" + std::to_string(::getpid());
  {
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    out.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(bytes));
    if (!out) {
      std::error_code ignored;
      fs::remove(partial, ignored);
      return core::Status::io_error(std::string(kWho) + ": cannot write " +
                                    partial.string());
    }
  }
  std::error_code ec;
  fs::rename(partial, path, ec);
  if (ec) {
    std::error_code ignored;
    fs::remove(partial, ignored);
    return core::Status::io_error(std::string(kWho) + ": cannot rename " +
                                  partial.string() + ": " + ec.message());
  }
  return {};
}

// One of the network's tensors, as the parsed network or the engine names it.
struct Tensor {
  std::string name;
  bool input = false;
  nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
  nvinfer1::Dims dims{};
};

std::string dims_text(const nvinfer1::Dims& dims) {
  std::string out;
  for (std::int32_t i = 0; i < dims.nbDims; ++i) {
    if (i != 0) out += " x ";
    out += std::to_string(dims.d[i]);
  }
  return out;
}

// The two inputs and the output tools/ffs/export_onnx.py writes, at the
// network's size.
core::Status check_signature(const std::vector<Tensor>& tensors,
                             std::uint32_t net_width,
                             std::uint32_t net_height) {
  const std::string expected =
      "inputs left_image and right_image, float 1 x 3 x " +
      std::to_string(net_height) + " x " + std::to_string(net_width) +
      ", and output disparity, float 1 x 1 x " + std::to_string(net_height) +
      " x " + std::to_string(net_width) +
      " (the rig's size rounded up to a multiple of 32)";
  if (tensors.size() != 3) {
    return refuse("the model has " + std::to_string(tensors.size()) +
                  " inputs and outputs; it needs " + expected);
  }
  for (const char* name : {kLeft, kRight, kDisparity}) {
    const bool input = name != kDisparity;
    const auto it =
        std::find_if(tensors.begin(), tensors.end(),
                     [&](const Tensor& t) { return t.name == name; });
    if (it == tensors.end() || it->input != input) {
      return refuse(std::string("the model has no ") +
                    (input ? "input " : "output ") + name + "; it needs " +
                    expected);
    }
    const std::int64_t want[4] = {1, input ? 3 : 1, net_height, net_width};
    bool same = it->dims.nbDims == 4;
    for (int i = 0; same && i < 4; ++i) same = it->dims.d[i] == want[i];
    if (!same || it->type != nvinfer1::DataType::kFLOAT) {
      return refuse(
          std::string("the model's ") + name + " is " + dims_text(it->dims) +
          (it->type == nvinfer1::DataType::kFLOAT ? " float" : " not float") +
          "; it needs " + expected);
    }
  }
  return {};
}

std::vector<Tensor> tensors_of(const nvinfer1::INetworkDefinition& network) {
  std::vector<Tensor> out;
  for (std::int32_t i = 0; i < network.getNbInputs(); ++i) {
    const nvinfer1::ITensor* t = network.getInput(i);
    out.push_back({t->getName(), true, t->getType(), t->getDimensions()});
  }
  for (std::int32_t i = 0; i < network.getNbOutputs(); ++i) {
    const nvinfer1::ITensor* t = network.getOutput(i);
    out.push_back({t->getName(), false, t->getType(), t->getDimensions()});
  }
  return out;
}

std::vector<Tensor> tensors_of(const nvinfer1::ICudaEngine& engine) {
  std::vector<Tensor> out;
  for (std::int32_t i = 0; i < engine.getNbIOTensors(); ++i) {
    const char* name = engine.getIOTensorName(i);
    out.push_back(
        {name, engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT,
         engine.getTensorDataType(name), engine.getTensorShape(name)});
  }
  return out;
}

// A depth buffer and CUDA's view of it, and its lease: held while a frame
// holds the buffer. The frame's deleter clears it with release order and
// take_depth reads it with acquire order, so whatever the frame's holders did
// with the buffer -- prep's read, and its wait -- happens before CUDA writes it
// again. CUDA writes every sample, so the buffer is not handed back from the
// family a reader took it to, as the decoders' pictures are not
// (cuda_pictures.cpp). The context goes last: a frame may outlive the matcher.
struct DepthBuffer {
  std::shared_ptr<CudaContext> context;
  core::ExportedBuffer exported;
  std::unique_ptr<CudaBufferMapping> mapping;
  std::atomic<bool> held{false};
};

// A rectifier's view, mapped into CUDA once: its descriptor is CUDA's from
// then on.
struct ViewMapping {
  const core::ExportedBuffer* view = nullptr;
  VkBuffer handle = VK_NULL_HANDLE;
  std::unique_ptr<CudaBufferMapping> mapping;
  std::uint64_t used = 0;  // the frame it was last read in
};

bool has_planes(const YuvImage& image) noexcept {
  return image.device != nullptr || image.host != nullptr ||
         image.image[0] != nullptr || image.image[1] != nullptr;
}

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

struct FfsStereo::Impl {
  const core::Device* device = nullptr;
  core::Allocator* allocator = nullptr;
  StereoRig rig;
  SensorInfo depth_info;
  Options options;
  DepthNoise depth_noise;  // every frame's, from the options and the rig
  float metres_per_unit = 0.0f;
  std::uint32_t net_width = 0, net_height = 0;

  // Declared first, so released last.
  std::shared_ptr<CudaContext> context;
  std::unique_ptr<CudaStream> stream;
  TrtPtr<nvinfer1::IRuntime> runtime;
  TrtPtr<nvinfer1::ICudaEngine> engine;
  TrtPtr<nvinfer1::IExecutionContext> execution;
  // The network's inputs and output, CUDA's own.
  float* left_input = nullptr;
  float* right_input = nullptr;
  float* disparity = nullptr;
  // Around the input kernel, the network and the depth kernel.
  cudaEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};
  // The frame's end, which the host waits on asleep (a blocking-sync event):
  // a stream synchronize in the primary context spins a core for the whole
  // network run.
  cudaEvent_t done = nullptr;
  std::vector<ViewMapping> views;
  std::vector<std::shared_ptr<DepthBuffer>> depths;
  std::uint64_t frames = 0;
  // A launch or wait that failed: CUDA may still run that frame's work.
  core::Status failed;

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  ~Impl();

  core::Status load_or_build(const fs::path& model);
  core::Status prepare_execution();
  core::Result<const void*> map_view(core::ExportedBuffer& view);
  core::Result<std::shared_ptr<DepthBuffer>> take_depth();
  core::Result<RgbdFrame> depth(const RectifiedPair& pair,
                                core::StageMetrics* metrics);
};

FfsStereo::Impl::~Impl() {
  if (context == nullptr) return;
  const CudaContextScope scope(context->handle());
  if (!scope.ok()) return;
  // TensorRT's objects and CUDA's memory in the context they were made in.
  execution.reset();
  engine.reset();
  runtime.reset();
  for (float* p : {left_input, right_input, disparity}) {
    if (p != nullptr) cudaFree(p);
  }
  for (cudaEvent_t e : events) {
    if (e != nullptr) cudaEventDestroy(e);
  }
  if (done != nullptr) cudaEventDestroy(done);
}

FfsStereo::FfsStereo(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

FfsStereo::~FfsStereo() = default;

core::Result<std::unique_ptr<FfsStereo>> FfsStereo::create(
    const core::Device& device, core::Allocator& allocator,
    const StereoRectifier& rectifier, const Options& options) try {
  if (!(std::isfinite(options.min_depth) && std::isfinite(options.max_depth) &&
        options.min_depth > 0.0f && options.min_depth < options.max_depth)) {
    return refuse("the depth range needs 0 < min_depth < max_depth");
  }
  if (!(std::isfinite(options.disparity_sigma) &&
        options.disparity_sigma >= 0.0f)) {
    return refuse(
        "disparity_sigma must be finite and at least 0 (0: unknown noise)");
  }
  if (options.plan_cache.empty()) {
    return refuse("no plan_cache: the plan takes minutes to build");
  }
  if (!device.exports_memory()) {
    return core::Status::unsupported(
        std::string(kWho) +
        ": the device does not export memory (VK_KHR_external_memory_fd), "
        "which CUDA reads the views and writes the depth through");
  }
  auto impl = std::make_unique<Impl>();
  impl->device = &device;
  impl->allocator = &allocator;
  impl->rig = rectifier.rig();
  impl->depth_info = rectifier.depth_info();
  impl->options = options;
  impl->depth_noise =
      DepthNoise::stereo(options.disparity_sigma,
                         impl->rig.camera.intrinsics.fx, impl->rig.baseline);
  impl->metres_per_unit = std::max(1e-4f, options.max_depth / 65535.0f);
  const camera::ImageSize size = impl->rig.camera.size;
  impl->net_width = round_up(size.width, kNetMultiple);
  impl->net_height = round_up(size.height, kNetMultiple);

  core::Result<std::unique_ptr<CudaContext>> opened = CudaContext::open(device);
  if (!opened) return opened.status().with_context(kWho);
  impl->context = std::move(opened).value();
  core::Result<std::unique_ptr<CudaStream>> stream =
      CudaStream::open(*impl->context);
  if (!stream) return stream.status().with_context(kWho);
  impl->stream = std::move(stream).value();

  const CudaContextScope scope(impl->context->handle());
  if (!scope.ok()) {
    return video::cuda_error(kWho, scope.result(),
                             "making the CUDA context current");
  }
  VKC_TRY(impl->load_or_build(options.model));
  VKC_TRY(impl->prepare_execution());
  return std::unique_ptr<FfsStereo>(new FfsStereo(std::move(impl)));
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("FfsStereo: out of host memory");
} catch (const fs::filesystem_error& e) {
  return core::Status::io_error(std::string(kWho) + ": " + e.what());
}

core::Status FfsStereo::Impl::load_or_build(const fs::path& model) {
  VKC_ASSIGN(const std::vector<char> onnx, read_file(model, "model"));

  // The plan's name: the GPU, TensorRT's version, the model's bytes and the
  // network's size. A plan is only ever good for all four.
  VKC_ASSIGN(const int ordinal, cuda_ordinal_of(*device));
  cudaDeviceProp props{};
  if (const cudaError_t e = cudaGetDeviceProperties(&props, ordinal);
      e != cudaSuccess) {
    return cuda_failure("reading the GPU's properties", e);
  }
  const std::uint64_t model_hash = fnv1a(onnx);
  unsigned char hash_bytes[8];
  for (int i = 0; i < 8; ++i) {
    hash_bytes[i] = static_cast<unsigned char>(model_hash >> (56 - 8 * i));
  }
  // Made before anything is built, so a cache that cannot be made costs no
  // build.
  std::error_code ec;
  fs::create_directories(options.plan_cache, ec);
  if (ec) {
    return core::Status::io_error(std::string(kWho) + ": cannot make " +
                                  options.plan_cache.string() + ": " +
                                  ec.message());
  }
  const fs::path plan =
      options.plan_cache /
      ("ffs-" + hex(hash_bytes, 8) + "-" + std::to_string(net_width) + "x" +
       std::to_string(net_height) + "-trt" +
       std::to_string(getInferLibVersion()) + "-gpu" +
       hex(reinterpret_cast<const unsigned char*>(props.uuid.bytes),
           sizeof(props.uuid.bytes)) +
       ".plan");

  runtime.reset(nvinfer1::createInferRuntime(logger()));
  if (runtime == nullptr) return tensorrt_failure("did not start");

  // A plan TensorRT will not load -- written by another version, or cut
  // short -- is built again.
  if (fs::is_regular_file(plan, ec)) {
    VKC_ASSIGN(const std::vector<char> bytes, read_file(plan, "plan"));
    engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (engine == nullptr) {
      core::log_message(core::LogLevel::Warning, "sensor_ffs",
                        "FfsStereo: building the plan again: TensorRT does "
                        "not load " +
                            plan.string());
    }
  }
  if (engine == nullptr) {
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
    if (builder == nullptr) return tensorrt_failure("builder did not start");
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    TrtPtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, logger()));
    if (network == nullptr || parser == nullptr) {
      return tensorrt_failure("did not make a network");
    }
    const std::string path = model.string();
    if (!parser->parse(onnx.data(), onnx.size(), path.c_str())) {
      std::string why;
      for (int i = 0; i < parser->getNbErrors(); ++i) {
        why += std::string(i == 0 ? "" : "; ") + parser->getError(i)->desc();
      }
      return refuse("TensorRT cannot parse the model " + path + ": " + why);
    }
    // Checked before the build, which takes minutes.
    VKC_TRY(check_signature(tensors_of(*network), net_width, net_height));
    core::log_message(
        core::LogLevel::Info, "sensor_ffs",
        "FfsStereo: building " + plan.string() + " (minutes, once)");
    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (config == nullptr) return tensorrt_failure("did not make a config");
    TrtPtr<nvinfer1::IHostMemory> built(
        builder->buildSerializedNetwork(*network, *config));
    if (built == nullptr) return tensorrt_failure("did not build the model");
    // A plan that cannot be kept still runs: the next create builds again.
    if (const core::Status written =
            write_plan(plan, built->data(), built->size());
        !written.ok()) {
      core::log_message(core::LogLevel::Warning, "sensor_ffs",
                        written.message() + "; running the plan unkept");
    }
    engine.reset(runtime->deserializeCudaEngine(built->data(), built->size()));
    if (engine == nullptr) return tensorrt_failure("did not load its plan");
  }
  return check_signature(tensors_of(*engine), net_width, net_height);
}

core::Status FfsStereo::Impl::prepare_execution() {
  execution.reset(engine->createExecutionContext());
  if (execution == nullptr) return tensorrt_failure("did not make a context");
  const std::size_t plane = std::size_t{net_width} * net_height;
  for (auto [p, floats] :
       {std::pair{&left_input, 3 * plane}, std::pair{&right_input, 3 * plane},
        std::pair{&disparity, plane}}) {
    if (const cudaError_t e =
            cudaMalloc(reinterpret_cast<void**>(p), floats * sizeof(float));
        e != cudaSuccess) {
      return cuda_failure("allocating the network's inputs and output", e);
    }
  }
  if (!execution->setTensorAddress(kLeft, left_input) ||
      !execution->setTensorAddress(kRight, right_input) ||
      !execution->setTensorAddress(kDisparity, disparity)) {
    return tensorrt_failure("did not take the network's buffers");
  }
  for (cudaEvent_t& e : events) {
    if (const cudaError_t r = cudaEventCreate(&e); r != cudaSuccess) {
      return cuda_failure("making the timing events", r);
    }
  }
  if (const cudaError_t r = cudaEventCreateWithFlags(
          &done, cudaEventBlockingSync | cudaEventDisableTiming);
      r != cudaSuccess) {
    return cuda_failure("making the frame's event", r);
  }
  return {};
}

core::Result<const void*> FfsStereo::Impl::map_view(
    core::ExportedBuffer& view) {
  // A descriptor still open is a buffer never imported: a new one, perhaps
  // where a mapped one was freed, whose mapping goes.
  if (view.fd.valid()) {
    views.erase(
        std::remove_if(views.begin(), views.end(),
                       [&](const ViewMapping& m) { return m.view == &view; }),
        views.end());
    core::Result<std::unique_ptr<CudaBufferMapping>> mapping =
        CudaBufferMapping::map(*context, view);
    if (!mapping) return mapping.status().with_context(kWho);
    if (views.size() == kMaxViews) {
      views.erase(
          std::min_element(views.begin(), views.end(),
                           [](const ViewMapping& a, const ViewMapping& b) {
                             return a.used < b.used;
                           }));
    }
    views.push_back(
        {&view, view.buffer.handle(), std::move(mapping).value(), frames});
    return reinterpret_cast<const void*>(views.back().mapping->pointer());
  }
  for (ViewMapping& m : views) {
    if (m.view == &view && m.handle == view.buffer.handle()) {
      m.used = frames;
      return reinterpret_cast<const void*>(m.mapping->pointer());
    }
  }
  return refuse(
      "a view was imported by something else: its descriptor is closed and "
      "this matcher has no mapping of it (never made, or let go: eight views "
      "stay mapped)");
}

core::Result<std::shared_ptr<DepthBuffer>> FfsStereo::Impl::take_depth() {
  for (const std::shared_ptr<DepthBuffer>& s : depths) {
    if (!s->held.load(std::memory_order_acquire)) return s;
  }
  auto made = std::make_shared<DepthBuffer>();
  made->context = context;
  const std::uint64_t bytes =
      (std::uint64_t{rig.camera.size.width} * rig.camera.size.height *
           sizeof(std::uint16_t) +
       3) /
      4 * 4;
  VKC_ASSIGN(made->exported,
             core::create_exported_buffer(*device, *allocator, bytes));
  device->set_object_name(
      VK_OBJECT_TYPE_BUFFER,
      core::debug_object_handle(made->exported.buffer.handle()),
      "sensor.ffs_depth");
  core::Result<std::unique_ptr<CudaBufferMapping>> mapping =
      CudaBufferMapping::map(*context, made->exported);
  if (!mapping) return mapping.status().with_context(kWho);
  made->mapping = std::move(mapping).value();
  depths.push_back(made);
  return made;
}

core::Result<RgbdFrame> FfsStereo::depth(const RectifiedPair& pair,
                                         core::StageMetrics* metrics) try {
  return impl_->depth(pair, metrics);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("FfsStereo: out of host memory");
}

core::Result<RgbdFrame> FfsStereo::Impl::depth(const RectifiedPair& pair,
                                               core::StageMetrics* metrics) {
  if (!failed.ok()) {
    return failed.with_context(
        "FfsStereo: an earlier frame failed and CUDA may still run it");
  }
  const Clock::time_point start = Clock::now();
  const camera::ImageSize size = rig.camera.size;
  if (pair.rig.camera.size.width != size.width ||
      pair.rig.camera.size.height != size.height ||
      pair.rig.camera.intrinsics.fx != rig.camera.intrinsics.fx ||
      pair.rig.baseline != rig.baseline) {
    return refuse("the pair is not from the rig this matcher was made for");
  }
  if (pair.left == nullptr || pair.right == nullptr ||
      pair.queue_family != kQueueFamilyExternal) {
    return refuse(
        "the pair's views are not exported (kQueueFamilyExternal): CUDA "
        "reads them in place");
  }
  // The views' layout, from their size: Luma8 a byte a pixel, four to a
  // word; Rgba8 a word a pixel.
  const std::uint64_t pixels = std::uint64_t{size.width} * size.height;
  const std::uint64_t luma_bytes = (pixels + 3) / 4 * 4;
  const std::uint64_t view_bytes = pair.left->buffer.size();
  if (pair.right->buffer.size() != view_bytes ||
      (view_bytes != luma_bytes && view_bytes != pixels * 4)) {
    return refuse("the views are not Luma8 or Rgba8 at the rig's size");
  }

  const CudaContextScope scope(context->handle());
  if (!scope.ok()) {
    return video::cuda_error(kWho, scope.result(),
                             "making the CUDA context current");
  }
  ++frames;
  VKC_ASSIGN(const void* left, map_view(*pair.left));
  VKC_ASSIGN(const void* right, map_view(*pair.right));
  VKC_ASSIGN(std::shared_ptr<DepthBuffer> target, take_depth());
  auto* depth_out =
      reinterpret_cast<std::uint16_t*>(target->mapping->pointer());

  ffs::InputParams in;
  in.width = size.width;
  in.height = size.height;
  in.net_width = net_width;
  in.net_height = net_height;
  in.luma = view_bytes == luma_bytes;
  ffs::DepthParams out;
  out.width = size.width;
  out.height = size.height;
  out.net_width = net_width;
  out.focal_baseline =
      static_cast<float>(rig.camera.intrinsics.fx * rig.baseline);
  out.min_depth = options.min_depth;
  out.max_depth = options.max_depth;
  out.units_per_metre = 1.0f / metres_per_unit;
  out.coverage = !in.luma;

  const cudaStream_t s = stream->handle();
  const bool timed = metrics != nullptr;
  cudaError_t e = timed ? cudaEventRecord(events[0], s) : cudaSuccess;
  if (e == cudaSuccess) {
    e = ffs::make_input(left, right, left_input, right_input, in, s);
  }
  if (e == cudaSuccess && timed) e = cudaEventRecord(events[1], s);
  bool enqueued = false;
  if (e == cudaSuccess) enqueued = execution->enqueueV3(s);
  if (enqueued && timed) e = cudaEventRecord(events[2], s);
  if (enqueued && e == cudaSuccess) {
    e = ffs::make_depth(disparity, left, right, depth_out, out, s);
  }
  if (enqueued && e == cudaSuccess && timed) e = cudaEventRecord(events[3], s);
  // Whatever was queued, waited for asleep.
  const Clock::time_point waiting = Clock::now();
  cudaError_t waited = cudaEventRecord(done, s);
  if (waited == cudaSuccess) waited = cudaEventSynchronize(done);
  const Clock::time_point finished = Clock::now();
  if (e == cudaSuccess) e = waited;
  if (!enqueued || e != cudaSuccess) {
    // What was queued may still run: the views and the depth buffer are let
    // go rather than reused, and no later frame runs.
    static_cast<void>(new std::shared_ptr<core::ExportedBuffer>(pair.left));
    static_cast<void>(new std::shared_ptr<core::ExportedBuffer>(pair.right));
    target->held.store(true, std::memory_order_relaxed);
    failed = e != cudaSuccess ? cuda_failure("running a frame", e)
                              : tensorrt_failure("did not run the network");
    return failed;
  }
  if (timed) {
    static constexpr const char* kRows[3] = {"ffs input", "ffs network",
                                             "ffs depth"};
    for (int i = 0; i < 3; ++i) {
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, events[i], events[i + 1]) == cudaSuccess) {
        metrics->add_gpu(kRows[i], ms);
      }
    }
    metrics->add_cpu("ffs", ms_between(start, finished));
    metrics->add_wait("ffs", ms_between(waiting, finished));
  }

  RgbdFrame frame;
  target->held.store(true, std::memory_order_relaxed);
  frame.depth_device.buffer = std::shared_ptr<const core::Buffer>(
      &target->exported.buffer, [keep = target](const core::Buffer*) {
        keep->held.store(false, std::memory_order_release);
      });
  frame.depth_device.offset = 0;
  frame.depth_device.queue_family = kQueueFamilyExternal;
  frame.metres_per_unit = metres_per_unit;
  frame.depth_camera = rig.camera;
  frame.min_depth = options.min_depth;
  frame.max_depth = options.max_depth;
  frame.depth_noise = depth_noise;
  const StereoFrame& source = pair.source;
  frame.color = has_planes(source.color) ? source.color : source.left;
  frame.color_camera = *depth_info.color;
  frame.color_encoding = source.color_encoding;
  frame.color_to_world = source.color_to_world;
  frame.depth_to_color = depth_info.depth_to_color;
  frame.timestamp_ns = source.timestamp_ns;
  frame.sequence = source.sequence;
  frame.arrival_ns = source.arrival_ns;
  frame.pixels = source.pixels;
  return frame;
}

}  // namespace volumetric_kit::sensor
