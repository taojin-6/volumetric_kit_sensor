// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Tao Jin

// The one DeckLink SDK user: the two inputs found, opened at the mode with
// format detection, and each frame stamped and handed on; and the capture
// allocator, through which the card writes every picture into a PagePool's
// pages (SDK manual 2.5.54-2.5.55).

#include <DeckLinkAPI.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "capture.hpp"
#include "ports.hpp"
#include "upload.hpp"
#include "volumetric_kit/core/base/log.hpp"

namespace volumetric_kit::sensor::decklink {

namespace {

static_assert(kV210 == bmdFormat10BitYUV);
static_assert(kFourSubDevicesHalfDuplex == bmdProfileFourSubDevicesHalfDuplex);

constexpr BMDTimeScale kNs = 1'000'000'000;
constexpr const char* kModel = "DeckLink 8K Pro";

// IUnknown's id (LinuxCOM.h's IID_IUnknown is a C compound literal).
constexpr REFIID kIUnknown = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                              0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46};

bool same_iid(const REFIID& a, const REFIID& b) {
  return std::memcmp(&a, &b, sizeof(REFIID)) == 0;
}

std::int64_t system_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// A string the SDK allocated, taken and freed.
std::string take_string(const char* text) {
  std::string out = text != nullptr ? text : "";
  std::free(const_cast<char*>(text));  // NOLINT: the SDK's malloc
  return out;
}

std::string hex(HRESULT hr) {
  char text[16];
  std::snprintf(text, sizeof text, "0x%08x", static_cast<unsigned>(hr));
  return text;
}

core::Status sdk_failure(const std::string& what, HRESULT hr) {
  return core::Status::io_error("DecklinkStereo: " + what + " failed (" +
                                hex(hr) + ")");
}

// One reference to an SDK object, released when it goes.
template <typename T>
class Ref {
 public:
  Ref() noexcept = default;
  explicit Ref(T* object) noexcept : object_(object) {}
  ~Ref() { reset(); }
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;
  Ref(Ref&& other) noexcept : object_(std::exchange(other.object_, nullptr)) {}
  Ref& operator=(Ref&& other) noexcept {
    if (this != &other) {
      reset();
      object_ = std::exchange(other.object_, nullptr);
    }
    return *this;
  }
  void reset() noexcept {
    if (object_ != nullptr) object_->Release();
    object_ = nullptr;
  }
  T* get() const noexcept { return object_; }
  T* operator->() const noexcept { return object_; }
  explicit operator bool() const noexcept { return object_ != nullptr; }
  T** out() noexcept {
    reset();
    return &object_;
  }

 private:
  T* object_ = nullptr;
};

template <typename T>
Ref<T> query(IUnknown* object, REFIID iid) {
  void* found = nullptr;
  if (object == nullptr || object->QueryInterface(iid, &found) != S_OK) {
    return Ref<T>();
  }
  return Ref<T>(static_cast<T*>(found));
}

// Reference counting for the objects the SDK holds of ours.
template <typename Interface>
class Counted : public Interface {
 public:
  ULONG AddRef() override { return ++count_; }
  ULONG Release() override {
    const ULONG left = --count_;
    if (left == 0) delete this;
    return left;
  }

 protected:
  Counted() = default;
  virtual ~Counted() = default;

 private:
  std::atomic<ULONG> count_{1};
};

// --- The capture allocator ---------------------------------------------------

// A PagePool buffer as the SDK sees it; back in the pool on its last
// release.
class PagesBuffer final : public Counted<IDeckLinkVideoBuffer> {
 public:
  PagesBuffer(std::shared_ptr<PagePool> pool, std::uint8_t* data,
              std::uint64_t size)
      : pool_(std::move(pool)), data_(data), size_(size) {}
  HRESULT QueryInterface(REFIID iid, LPVOID* out) override {
    if (out == nullptr) return E_POINTER;
    if (same_iid(iid, kIUnknown) || same_iid(iid, IID_IDeckLinkVideoBuffer)) {
      AddRef();
      *out = static_cast<IDeckLinkVideoBuffer*>(this);
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  HRESULT GetBytes(void** buffer) override {
    if (buffer == nullptr) return E_POINTER;
    *buffer = data_;
    return S_OK;
  }
  HRESULT GetSize(uint64_t* size) override {
    if (size == nullptr) return E_POINTER;
    *size = size_;
    return S_OK;
  }
  // Ordinary host memory: nothing to map or flush.
  HRESULT StartAccess(BMDBufferAccessFlags) override { return S_OK; }
  HRESULT EndAccess(BMDBufferAccessFlags) override { return S_OK; }

 private:
  ~PagesBuffer() override { pool_->give_back(data_); }
  std::shared_ptr<PagePool> pool_;
  std::uint8_t* data_;
  std::uint64_t size_;
};

class PagesAllocator final : public Counted<IDeckLinkVideoBufferAllocator> {
 public:
  explicit PagesAllocator(std::shared_ptr<PagePool> pool)
      : pool_(std::move(pool)) {}
  HRESULT QueryInterface(REFIID iid, LPVOID* out) override {
    if (out == nullptr) return E_POINTER;
    if (same_iid(iid, kIUnknown) ||
        same_iid(iid, IID_IDeckLinkVideoBufferAllocator)) {
      AddRef();
      *out = static_cast<IDeckLinkVideoBufferAllocator*>(this);
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  HRESULT AllocateVideoBuffer(IDeckLinkVideoBuffer** allocated) override try {
    if (allocated == nullptr) return E_POINTER;
    *allocated = nullptr;
    core::Result<std::uint8_t*> pages = pool_->take();
    if (!pages.ok()) {
      // The SDK hears only E_OUTOFMEMORY and drops the frame: say why once,
      // or start would blame the cabling.
      if (!warned_.exchange(true)) {
        core::log_message(core::LogLevel::Warning, "sensor",
                          "DecklinkStereo: making pages for the card: " +
                              pages.status().message());
      }
      return E_OUTOFMEMORY;
    }
    if (*pages == nullptr) return E_OUTOFMEMORY;  // the cap
    *allocated = new PagesBuffer(pool_, *pages, pool_->size());
    return S_OK;
  } catch (...) {
    return E_OUTOFMEMORY;
  }

 private:
  std::shared_ptr<PagePool> pool_;
  std::atomic<bool> warned_{false};
};

class PagesProvider final
    : public Counted<IDeckLinkVideoBufferAllocatorProvider> {
 public:
  explicit PagesProvider(std::shared_ptr<PagePool> pool)
      : pool_(std::move(pool)) {}
  HRESULT QueryInterface(REFIID iid, LPVOID* out) override {
    if (out == nullptr) return E_POINTER;
    if (same_iid(iid, kIUnknown) ||
        same_iid(iid, IID_IDeckLinkVideoBufferAllocatorProvider)) {
      AddRef();
      *out = static_cast<IDeckLinkVideoBufferAllocatorProvider*>(this);
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  // `size` may exceed rowBytes x height (manual 2.5.54.1): every page buffer
  // is made at it, aligned to the device's import alignment (4 KiB, past the
  // SDK's 16-byte floor).
  HRESULT GetVideoBufferAllocator(uint32_t size, uint32_t, uint32_t, uint32_t,
                                  BMDPixelFormat,
                                  IDeckLinkVideoBufferAllocator** out) override
      try {
    if (out == nullptr) return E_POINTER;
    *out = nullptr;
    if (!pool_->set_size(size).ok()) return E_OUTOFMEMORY;
    *out = new PagesAllocator(pool_);
    return S_OK;
  } catch (...) {
    return E_OUTOFMEMORY;
  }

 private:
  std::shared_ptr<PagePool> pool_;
};

// --- An input's frames -------------------------------------------------------

class InputCallback final : public Counted<IDeckLinkInputCallback> {
 public:
  InputCallback(int index, IDeckLinkInput* input, std::uint32_t mode,
                long width, long height, long row_bytes, bool read_bytes)
      : index_(index),
        input_(input),
        mode_(mode),
        width_(width),
        height_(height),
        row_bytes_(row_bytes),
        read_bytes_(read_bytes) {}

  void set_sink(CaptureSink* sink) noexcept { sink_.store(sink); }

  HRESULT QueryInterface(REFIID iid, LPVOID* out) override {
    if (out == nullptr) return E_POINTER;
    if (same_iid(iid, kIUnknown) || same_iid(iid, IID_IDeckLinkInputCallback)) {
      AddRef();
      *out = static_cast<IDeckLinkInputCallback*>(this);
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }

  HRESULT VideoInputFormatChanged(
      BMDVideoInputFormatChangedEvents, IDeckLinkDisplayMode* mode,
      BMDDetectedVideoInputFormatFlags flags) override try {
    CaptureSink* sink = sink_.load();
    if (sink == nullptr || mode == nullptr) return S_OK;
    const bool ten_bit_yuv = (flags & bmdDetectedVideoInputYCbCr422) != 0 &&
                             (flags & bmdDetectedVideoInput10BitDepth) != 0;
    if (mode->GetDisplayMode() == mode_ && ten_bit_yuv) return S_OK;
    const char* name = nullptr;
    std::string detected =
        mode->GetName(&name) == S_OK ? take_string(name) : "another mode";
    if ((flags & bmdDetectedVideoInputRGB444) != 0) detected += " RGB 4:4:4";
    if ((flags & bmdDetectedVideoInput8BitDepth) != 0) detected += " 8-bit";
    if ((flags & bmdDetectedVideoInput12BitDepth) != 0) detected += " 12-bit";
    sink->mode_changed(index_, std::move(detected));
    return S_OK;
  } catch (...) {
    return S_OK;
  }

  HRESULT VideoInputFrameArrived(IDeckLinkVideoInputFrame* frame,
                                 IDeckLinkAudioInputPacket*) override try {
    CaptureSink* sink = sink_.load();
    if (sink == nullptr || frame == nullptr) return S_OK;
    Arrival arrival;
    arrival.input = index_;
    // The two clocks, sampled together: their offset at this frame.
    BMDTimeValue card_now = 0;
    BMDTimeValue in_frame = 0;
    BMDTimeValue ticks = 0;
    const HRESULT clock =
        input_->GetHardwareReferenceClock(kNs, &card_now, &in_frame, &ticks);
    arrival.host_now_ns = system_now_ns();
    arrival.card_now_ns = card_now;
    BMDTimeValue card = 0;
    BMDTimeValue duration = 0;
    const HRESULT stamp =
        frame->GetHardwareReferenceTimestamp(kNs, &card, &duration);
    arrival.card_ns = card;
    // A frame the card gave no time for can be neither paired nor stamped.
    arrival.no_signal = clock != S_OK || stamp != S_OK ||
                        (frame->GetFlags() & bmdFrameHasNoInputSource) != 0;
    if (!arrival.no_signal && read_bytes_) {
      if (frame->GetWidth() != width_ || frame->GetHeight() != height_ ||
          frame->GetRowBytes() != row_bytes_) {
        sink->mode_changed(
            index_, std::to_string(frame->GetWidth()) + "x" +
                        std::to_string(frame->GetHeight()) + " pictures of " +
                        std::to_string(frame->GetRowBytes()) + " bytes a row");
        return S_OK;
      }
      // Only the address: which of the pool's buffers the card wrote.
      Ref<IDeckLinkVideoBuffer> buffer =
          query<IDeckLinkVideoBuffer>(frame, IID_IDeckLinkVideoBuffer);
      void* bytes = nullptr;
      if (buffer && buffer->StartAccess(bmdBufferAccessRead) == S_OK) {
        buffer->GetBytes(&bytes);
        buffer->EndAccess(bmdBufferAccessRead);
      }
      arrival.bytes = static_cast<const std::uint8_t*>(bytes);
    }
    frame->AddRef();
    arrival.hold = std::shared_ptr<void>(frame, [](void* held) {
      static_cast<IDeckLinkVideoInputFrame*>(held)->Release();
    });
    sink->arrived(std::move(arrival));
    return S_OK;
  } catch (...) {
    return S_OK;
  }

 private:
  ~InputCallback() override = default;
  const int index_;
  IDeckLinkInput* const input_;  // the capture's
  const std::uint32_t mode_;
  const long width_, height_, row_bytes_;
  const bool read_bytes_;
  std::atomic<CaptureSink*> sink_{nullptr};
};

// --- The two inputs ----------------------------------------------------------

struct Input {
  std::string name;
  Ref<IDeckLinkInput> input;
  Ref<IDeckLinkStatus> status;
  Ref<InputCallback> callback;
  Ref<PagesProvider> provider;  // null: the SDK's own buffers
  bool enabled = false;
  bool streaming = false;
};

class SdkCapture final : public Capture {
 public:
  SdkCapture(std::vector<Input> inputs, std::uint32_t mode)
      : inputs_(std::move(inputs)), mode_(mode) {}
  ~SdkCapture() override { stop(); }

  core::Status start(CaptureSink& sink,
                     std::chrono::milliseconds timeout) override {
    stop();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (Input& in : inputs_) {
      in.callback->set_sink(&sink);
      HRESULT hr = in.input->SetCallback(in.callback.get());
      if (hr != S_OK) {
        stop();
        return sdk_failure(in.name + ": SetCallback", hr);
      }
      const BMDVideoInputFlags flags = bmdVideoInputEnableFormatDetection;
      hr = in.provider
               ? in.input->EnableVideoInputWithAllocatorProvider(
                     mode_, bmdFormat10BitYUV, flags, in.provider.get())
               : in.input->EnableVideoInput(mode_, bmdFormat10BitYUV, flags);
      if (hr != S_OK) {
        stop();
        if (hr == E_ACCESSDENIED) {
          return core::Status::io_error(
              "DecklinkStereo: " + in.name +
              " is in use by another program: close it and start again");
        }
        return sdk_failure(in.name + ": EnableVideoInput", hr);
      }
      in.enabled = true;
    }
    // Each input locked to its camera before its frames start, so none
    // comes without a picture (inputs enabled before they locked delivered
    // frames without one).
    for (Input& in : inputs_) {
      for (;;) {
        bool locked = false;
        if (in.status->GetFlag(bmdDeckLinkStatusVideoInputSignalLocked,
                               &locked) == S_OK &&
            locked) {
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          stop();
          return core::Status::io_error(
              "DecklinkStereo: " + in.name + " locked to no signal in " +
              std::to_string(timeout.count()) +
              " ms: check that the camera is on and its SDI OUT is cabled "
              "to that input");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    for (Input& in : inputs_) {
      const HRESULT hr = in.input->StartStreams();
      if (hr != S_OK) {
        stop();
        return sdk_failure(in.name + ": StartStreams", hr);
      }
      in.streaming = true;
    }
    return {};
  }

  void stop() noexcept override {
    for (Input& in : inputs_) {
      if (in.streaming) in.input->StopStreams();
      if (in.enabled) in.input->DisableVideoInput();
      in.input->SetCallback(nullptr);
      in.callback->set_sink(nullptr);
      in.streaming = false;
      in.enabled = false;
    }
  }

 private:
  std::vector<Input> inputs_;
  const std::uint32_t mode_;
};

struct Device {
  Ref<IDeckLink> deck_link;
  std::string model;
  std::string name;  // the SDK's display name, e.g. "DeckLink 8K Pro (3)"
  std::int64_t sub_device = -1;
  std::int64_t group = -1;
  std::int64_t profile = 0;
  std::int64_t topological = 0;
};

core::Result<std::unique_ptr<Capture>> open(const CaptureConfig& config) {
  Ref<IDeckLinkIterator> iterator(CreateDeckLinkIteratorInstance());
  if (!iterator) {
    return core::Status::not_found(
        "DecklinkStereo: Desktop Video's library (libDeckLinkAPI.so) did not "
        "load: install Blackmagic Desktop Video 16.0 or newer");
  }
  std::vector<Device> found;
  std::string seen;
  for (;;) {
    Ref<IDeckLink> deck_link;
    if (iterator->Next(deck_link.out()) != S_OK) break;
    Device d;
    const char* model = nullptr;
    d.model = deck_link->GetModelName(&model) == S_OK ? take_string(model)
                                                      : "a DeckLink";
    const char* name = nullptr;
    d.name =
        deck_link->GetDisplayName(&name) == S_OK ? take_string(name) : d.model;
    seen += (seen.empty() ? "" : ", ") + d.model;
    Ref<IDeckLinkProfileAttributes> attributes =
        query<IDeckLinkProfileAttributes>(deck_link.get(),
                                          IID_IDeckLinkProfileAttributes);
    if (attributes) {
      attributes->GetInt(BMDDeckLinkSubDeviceIndex, &d.sub_device);
      attributes->GetInt(BMDDeckLinkDeviceGroupID, &d.group);
      attributes->GetInt(BMDDeckLinkProfileID, &d.profile);
      attributes->GetInt(BMDDeckLinkTopologicalID, &d.topological);
    }
    d.deck_link = std::move(deck_link);
    if (d.model.rfind(kModel, 0) == 0) found.push_back(std::move(d));
  }
  if (found.empty()) {
    return core::Status::not_found(std::string("DecklinkStereo: no ") + kModel +
                                   "; Desktop Video sees " +
                                   (seen.empty() ? "no card" : seen));
  }
  // The first card's sub-devices.
  const std::int64_t card = found.front().group;
  if (found.front().profile != kFourSubDevicesHalfDuplex) {
    return core::Status::unsupported(
        std::string("DecklinkStereo: the ") + kModel +
        "'s profile does not give each SDI connector its own input: set it "
        "to \"four sub-devices half-duplex\" (Desktop Video Setup, or the "
        "SDK's ActivateProfile sample, -p 3)");
  }
  const std::optional<Mode> mode = [&]() -> std::optional<Mode> {
    for (const std::uint32_t fps : {24u, 25u, 30u}) {
      const std::optional<Mode> m = mode_of(3840, 2160, fps);
      if (m && m->code == config.mode) return m;
    }
    return std::nullopt;
  }();
  if (!mode) {
    return core::Status::invalid_argument("DecklinkStereo: an unknown mode");
  }

  std::vector<Input> inputs;
  // Which SDK device each connector opened, to check against the cabling.
  std::string opened = "DecklinkStereo: opened";
  for (int i = 0; i < 2; ++i) {
    Device* device = nullptr;
    for (Device& d : found) {
      if (d.group == card && d.sub_device == config.sub_device[i]) device = &d;
    }
    if (device == nullptr) {
      return core::Status::not_found("DecklinkStereo: " + config.names[i] +
                                     ": the card has no such sub-device");
    }
    char topological[24];
    std::snprintf(topological, sizeof topological, "%llx",
                  static_cast<unsigned long long>(device->topological));
    opened += std::string(i == 0 ? " " : "; ") + config.names[i] +
              " as sub-device " + std::to_string(device->sub_device) + ", \"" +
              device->name + "\", topological ID " + topological;
    Input in;
    in.name = config.names[i];
    in.input =
        query<IDeckLinkInput>(device->deck_link.get(), IID_IDeckLinkInput);
    in.status =
        query<IDeckLinkStatus>(device->deck_link.get(), IID_IDeckLinkStatus);
    if (!in.input || !in.status) {
      return core::Status::unsupported("DecklinkStereo: " + in.name +
                                       " is not an input");
    }
    bool supported = false;
    BMDDisplayMode actual = bmdModeUnknown;
    const HRESULT hr = in.input->DoesSupportVideoMode(
        bmdVideoConnectionUnspecified, config.mode, bmdFormat10BitYUV,
        bmdNoVideoInputConversion, bmdSupportedVideoModeDefault, &actual,
        &supported);
    if (hr != S_OK || !supported) {
      return core::Status::unsupported("DecklinkStereo: " + in.name +
                                       " cannot capture " + mode->name +
                                       " as v210");
    }
    const bool pages = config.pages[i] != nullptr;
    in.callback = Ref<InputCallback>(new InputCallback(
        i, in.input.get(), config.mode, mode->width, mode->height,
        static_cast<long>(v210_row_bytes(mode->width)), pages));
    if (pages)
      in.provider = Ref<PagesProvider>(new PagesProvider(config.pages[i]));
    inputs.push_back(std::move(in));
  }
  core::log_message(core::LogLevel::Info, "sensor", opened);
  return std::unique_ptr<Capture>(
      new SdkCapture(std::move(inputs), config.mode));
}

}  // namespace

core::Result<std::unique_ptr<Capture>> open_capture(
    const CaptureConfig& config) try {
  return open(config);
} catch (const std::bad_alloc&) {
  return core::Status::out_of_memory("DecklinkStereo: out of host memory");
}

}  // namespace volumetric_kit::sensor::decklink
